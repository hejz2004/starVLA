# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [HKUST starVLA] in [2025].

from typing import Optional

import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.model.modules.vlm.qwen3_vl_dual import Qwen3VLForConditionalGeneration
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)

import torch.nn as nn


class _QWen3DualStream_VL_Interface(nn.Module):
    """
    Lightweight wrapper around the *vendored* dual-stream Qwen3-VL
    (`starVLA/model/modules/vlm/qwen3_vl_dual`), the DualStream-Qwenpi-VLA
    counterpart of `_QWen3_VL_Interface`.

    Differences vs. the plain interface:
      - Loads the vendored model class and injects `vision_config.dual_stream`
        (from `framework.qwenvl.dual_stream`) before `from_pretrained`.

    Everything else (processor usage, CausalLM-like interface, generate)
    matches `_QWen3_VL_Interface`, so frameworks can swap interfaces freely.
    Each sample in a forward is a separate scene. Only the images within
    that sample interact through its shared-token activation.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the dual-stream Qwen3-VL wrapper.

        The `dual_stream` block under ``framework.qwenvl`` controls the
        vision-tower extension, e.g.::

            dual_stream:
              enabled: true
              num_shared_tokens: 32
              num_heads: 16
              gate_hidden_size: 256
              gate_init_value: 0.01
              gate_output_weight_std: 0.001

        `interaction_indexes` default to
        sorted(set(deepstack_visual_indexes) | {depth-1}) and are validated
        inside the vision tower.
        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-2B-Instruct")
        # sdpa is the only backend the dual-stream stages rely on; the plain
        # interface also falls back to sdpa (PPU environments have no FA2).
        attn_implementation = "sdpa"

        # Deep-convert to plain Python types: the YAML block arrives as an
        # OmegaConf DictConfig whose nested lists (e.g. interaction_indexes)
        # would break `save_pretrained`'s JSON serialization if leaked.
        raw_dual_stream = qwenvl_config.get("dual_stream", {}) or {}
        if OmegaConf.is_config(raw_dual_stream):
            dual_stream_config = OmegaConf.to_container(raw_dual_stream, resolve=True)
        else:
            dual_stream_config = dict(raw_dual_stream)
        hf_config = Qwen3VLConfig.from_pretrained(model_id)
        if dual_stream_config.get("enabled", False):
            hf_config.vision_config.dual_stream = dual_stream_config
            logger.info(f"Dual-stream enabled with config: {dual_stream_config}")

        # load_weights=false builds the model from scratch (every parameter
        # randomly initialized) for from-scratch pretraining runs. Only the
        # config/processor files of `base_vlm` are read — no checkpoint load.
        load_weights = bool(qwenvl_config.get("load_weights", True))
        if load_weights:
            model, loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                config=hf_config,
                attn_implementation=attn_implementation,
                dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
            )
            if dual_stream_config.get("enabled", False) and getattr(
                model.model.visual, "interaction_stages", None
            ) is None:
                raise RuntimeError(
                    "dual_stream is enabled in the config but the vision tower has no interaction "
                    "stages — check that the vendored qwen3_vl_dual model was loaded."
                )
            # The base VLM checkpoint has no dual-stream keys, so every new
            # param arrives as "missing". Only params actually reported
            # missing are initialized here; a resumed DualStream checkpoint
            # (all keys present) passes through untouched.
            self._init_missing_dual_stream_params(
                model.model.visual, loading_info.get("missing_keys", [])
            )
        else:
            # `post_init` inside `__init__` applies the model's `_init_weights`
            # to every submodule (incl. the GateMLP override); `shared_init` is
            # seeded in `Qwen3VLVisionModel.__init__`.
            hf_config._attn_implementation = attn_implementation
            model = Qwen3VLForConditionalGeneration(hf_config).to(torch.bfloat16)
            logger.info(f"load_weights=false — built {model_id} with random init")

        if dual_stream_config.get("enabled", False) and getattr(
            model.model.visual, "interaction_stages", None
        ) is None:
            raise RuntimeError(
                "dual_stream is enabled in the config but the vision tower has no interaction "
                "stages — check that the vendored qwen3_vl_dual model was loaded."
            )

        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self.dual_stream_enabled = bool(dual_stream_config.get("enabled", False))

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    @staticmethod
    def _init_missing_dual_stream_params(vision_model, missing_keys) -> None:
        """Manually initialize dual-stream params absent from the loaded checkpoint.

        from_pretrained materializes missing *modules* (interaction_stages
        Linear/LayerNorm) through the model's `_init_weights` hook, but a bare
        nn.Parameter like `shared_init` survives meta-device loading as
        uninitialized memory, and the gate output layer has a non-standard
        init (near-zero weight, bias = log(g0/(1-g0)), spec §10) — both are
        seeded explicitly here. Keyed on the actually-missing checkpoint
        entries, so nothing is touched when they load from a DualStream
        checkpoint (training resume).
        """
        missing = set(missing_keys or [])

        def is_missing(suffix: str) -> bool:
            return any(k == suffix or k.endswith("." + suffix) for k in missing)

        dual_missing = sorted(k for k in missing if "interaction_stages." in k or k.endswith("shared_init"))
        if dual_missing:
            preview = ", ".join(dual_missing[:3]) + ("..." if len(dual_missing) > 3 else "")
            logger.info(f"checkpoint lacks {len(dual_missing)} dual-stream keys — initializing manually: {preview}")

        with torch.no_grad():
            if is_missing("shared_init"):
                vision_model.shared_init.normal_(0.0, vision_model.config.initializer_range)
            for idx, stage in vision_model.interaction_stages.items():
                fc2 = stage.gate_mlp.fc2
                if is_missing(f"interaction_stages.{idx}.gate_mlp.fc2.bias"):
                    fc2.bias.fill_(stage.gate_mlp.gate_init_bias)
                if is_missing(f"interaction_stages.{idx}.gate_mlp.fc2.weight"):
                    fc2.weight.normal_(0.0, stage.gate_mlp.gate_output_weight_std)

        bad = [
            name
            for name, p in vision_model.named_parameters()
            if ("interaction_stages" in name or name == "shared_init") and not torch.isfinite(p).all()
        ]
        if bad:
            raise RuntimeError(f"dual-stream parameters failed to initialize (non-finite): {bad[:5]}")

    def set_force_zero_gate(self, value: bool) -> None:
        """Diagnostic: force the cross-view gate gamma to 0 (spec §12.1)."""
        self.model.visual.set_force_zero_gate(value)

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to the underlying dual-stream Qwen3-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        The vision tower runs once at prefill over all views jointly; cached
        decode steps pass pixel_values=None so shared tokens are not
        recomputed (guaranteed by `prepare_inputs_for_generation`).
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Official Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    qwen_vl = _QWen3DualStream_VL_Interface(cfg)
    pass
