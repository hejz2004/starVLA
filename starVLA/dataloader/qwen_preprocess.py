"""CPU-only Qwen collation; preserve the interface's exact prompt/image policy."""
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf


class QwenProcessorCollator:
    def __init__(self, model_path, data_config):
        self.model_path = str(model_path)
        self.data_config = dict(data_config)
        self._interface = None

    def __call__(self, examples):
        if self._interface is None:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.model_path)
            processor.tokenizer.padding_side = "left"
            self._interface = SimpleNamespace(
                processor=processor,
                config=OmegaConf.create({"datasets": {"vla_data": self.data_config}}),
            )
        from starVLA.model.modules.vlm.QWen3DualStream import _QWen3DualStream_VL_Interface
        inputs = _QWen3DualStream_VL_Interface.build_qwenvl_inputs(
            self._interface, [x["image"] for x in examples], [x["lang"] for x in examples],
            return_cpu=True,
        )
        result = {"vlm_inputs": dict(inputs),
                  "actions": torch.from_numpy(np.stack([x["action"] for x in examples]))}
        if any(x.get("action_mask") is not None for x in examples):
            masks = []
            for sample in examples:
                action = sample["action"]
                mask = sample.get("action_mask")
                if mask is not None and np.shape(mask) != np.shape(action):
                    raise ValueError("action_mask shape does not match action shape")
                masks.append(np.ones_like(action, dtype=bool) if mask is None else np.asarray(mask, dtype=bool))
            result["action_mask"] = torch.from_numpy(np.stack(masks))
        if "state" in examples[0]:
            result["state"] = torch.from_numpy(np.stack([x["state"] for x in examples]))
        return result


def configure_worker_preprocessing(loader, cfg):
    if not cfg.datasets.vla_data.get("worker_preprocessing", False):
        return loader
    if cfg.framework.name not in ("DualStreamPI", "DualStreamQwenPI", "DualStreamQwenFM"):
        raise ValueError("worker_preprocessing currently supports DualStreamPI only")
    if cfg.datasets.vla_data.get("stages"):
        raise ValueError("worker_preprocessing with staged UMI requires a stage-aware integration")
    if loader.num_workers < 1:
        raise ValueError("worker_preprocessing requires num_workers > 0")
    # Do not fork after model/CUDA initialization. Only the processor is loaded
    # lazily in each worker; no model/weights are serialized into the workers.
    loader.multiprocessing_context = "spawn"
    data = cfg.datasets.vla_data
    loader.collate_fn = QwenProcessorCollator(cfg.framework.qwenvl.base_vlm,
        {"CoT_prompt": data.CoT_prompt} if "CoT_prompt" in data else {})
    loader.pin_memory = True
    return loader
