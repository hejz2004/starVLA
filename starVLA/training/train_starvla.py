# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import swanlab as wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

# Spawn workers re-import this file as __mp_main__; only the training
# process may initialize distributed state/DeepSpeed.
deepspeed_plugin = None
accelerator = None
if __name__ != "__mp_main__":
    deepspeed_plugin = None if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1" else DeepSpeedPlugin()
    # bf16 must be declared on the DS engine: the bare plugin leaves DeepSpeed in
    # fp32, and the engine force-disables any outer torch.autocast inside its
    # forward, so a manual autocast wrapper alone never reaches it.
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        mixed_precision="bf16" if deepspeed_plugin is not None else None,
        # We call scheduler.step() once per optimizer update below. Accelerate's
        # default wrapper otherwise advances it once per distributed process.
        step_scheduler_with_optimizer=False,
    )
    accelerator.print(accelerator.state)


def _unwrap_model(accelerator, model):
    """Avoid importing a broken optional DeepSpeed install in single-GPU mode."""
    if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1":
        return model.module if hasattr(model, "module") else model
    return accelerator.unwrap_model(model)


def _get_state_dict(accelerator, model):
    if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1":
        return _unwrap_model(accelerator, model).state_dict()
    return accelerator.get_state_dict(model)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    indexed = cfg.datasets.vla_data.dataset_py == "umi_indexed"
    previous = output_dir / "config.full.yaml"
    if indexed and previous.exists():
        if not cfg.trainer.get("is_resume", False):
            raise FileExistsError(f"Run already exists: {output_dir}; use a new run_id or resume explicitly")
        from starVLA.dataloader.umi_wyt import stage_identity
        saved = OmegaConf.load(previous)
        if (saved.datasets.vla_data.dataset_py != "umi_indexed"
                or stage_identity(saved, accelerator.num_processes) != stage_identity(cfg, accelerator.num_processes)):
            raise ValueError("Existing run has different UMI/state/batch semantics; use a new run for weights-only initialization")

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    if indexed and dist.is_initialized():
        dist.barrier()
    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.get('data_mix', '<none>')}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self._umi_initial_raw = vla_train_dataloader
        self.umi_stage_data = None

    def _sync_gradient_accumulation_steps(self):
        """Wire trainer.gradient_accumulation_steps into accelerate.

        The module-level Accelerator/DeepSpeedPlugin are built before the yaml
        is parsed, so the configured value never reaches accelerate without
        this (runtime gas would stay 1). The plugin must carry the same value:
        at engine creation accelerate lets a DeepSpeed-side value win over the
        GradientAccumulationPlugin's.
        """
        gas = int(getattr(self.config.trainer, "gradient_accumulation_steps", 1) or 1)
        if gas < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gas}")
        if deepspeed_plugin is not None:
            deepspeed_plugin.gradient_accumulation_steps = gas
            # Under DeepSpeed, accelerator.clip_grad_norm_ only reports the
            # engine global norm; actual clipping happens engine-side from
            # this config value (accelerate fills its auto default with 1.0).
            clip = getattr(self.config.trainer, "gradient_clipping", None)
            if clip is not None:
                deepspeed_plugin.deepspeed_config["gradient_clipping"] = float(clip)
        self.accelerator.gradient_accumulation_steps = gas
        logger.info(f"Gradient accumulation steps wired from config: {gas}")

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._sync_gradient_accumulation_steps()

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        # Restores batches_consumed/vla_epoch_count from strict checkpoints
        # (no-op on save) whenever accelerator.save_state/load_state run.
        self.accelerator.register_for_checkpointing(self)

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # The lr_scheduler must go through accelerator.prepare so that
        # accelerator.save_state/load_state persist its internal state
        # (accelerate only tracks schedulers it prepared).
        self.model, self.optimizer, self.lr_scheduler, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.vla_train_dataloader,
        )

        if self.config.datasets.vla_data.get("stages"):
            from starVLA.training.trainer_utils.umi_stage_data import UMIStageData
            self.umi_stage_data = UMIStageData(
                self.config, self.accelerator, self._umi_initial_raw, self.vla_train_dataloader
            )

        # DeepSpeed/accelerate state can only be loaded after the model and
        # optimizer are wrapped (ZeRO shards exist); legacy weights-only
        # checkpoints were already loaded pre-wrap in _init_checkpointing.
        self._load_strict_checkpoint()
        self._adjust_lr_scheduler_for_resume()

        self._init_swanlab()
        self._log_deepspeed_engine_config()

    def _log_deepspeed_engine_config(self):
        """Log engine-effective DeepSpeed values (yaml -> DS wiring audit)."""
        engine = self.model
        ds_cfg = getattr(engine, "_config", None)
        if ds_cfg is None or type(engine).__name__ != "DeepSpeedEngine":
            return
        self.accelerator.print(
            "[ds-audit] zero_stage=%s micro_batch=%s grad_accum=%s train_batch=%s "
            "grad_clipping=%s bf16=%s fp16=%s"
            % (
                ds_cfg.zero_optimization_stage,
                ds_cfg.train_micro_batch_size_per_gpu,
                ds_cfg.gradient_accumulation_steps,
                ds_cfg.train_batch_size,
                ds_cfg.gradient_clipping,
                getattr(ds_cfg, "bfloat16_config", None),
                getattr(ds_cfg, "fp16_enabled", getattr(ds_cfg, "fp16_config", None)),
            )
        )

    def state_dict(self):
        """Small trainer state persisted alongside accelerator checkpoints."""
        state = {
            "batches_consumed": int(getattr(self, "batches_consumed", 0)),
            "vla_epoch_count": int(getattr(self, "vla_epoch_count", 0)),
        }
        if self.umi_stage_data is not None:
            state["umi_data"] = self.umi_stage_data.state_dict()
        return state

    def load_state_dict(self, state):
        self.batches_consumed = int(state.get("batches_consumed", 0))
        self.vla_epoch_count = int(state.get("vla_epoch_count", 0))
        if self.umi_stage_data is not None:
            if "umi_data" not in state:
                raise ValueError("Old checkpoint has no UMI stage identity; use weights-only initialization in a new run")
            self.umi_stage_data.load_state_dict(state["umi_data"])

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_swanlab(self):
        """Initialize SwanLab on rank zero; keep distributed startup synchronized."""
        self._swanlab_enabled = False
        if os.environ.get("SWANLAB_MODE") == "disabled":
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    logdir=os.path.join(self.config.output_dir, "swanlog"),
                    project=self.config.get("swanlab_project", self.config.get("wandb_project", "starvla")),
                    workspace=self.config.get("swanlab_workspace", None),
                    mode=os.environ.get("SWANLAB_MODE", "online"),
                    public=False,
                    group="vla-train",
                    config=self._swanlab_config_dict(),
                )
                self._swanlab_enabled = True
            except Exception as exc:
                logger.warning(f"SwanLab init failed; continuing without SwanLab: {exc}")
                self._swanlab_enabled = False
        # Rendezvous after rank-0 SwanLab init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            if self.config.datasets.vla_data.dataset_py == "umi_indexed":
                # Preserve every data rule and resolved stage for reproducibility.
                OmegaConf.save(self.config.unwrap(), output_dir / "config.yaml", resolve=True)
            else:
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading.

        Two checkpoint kinds live under ``checkpoints/``:
        - ``steps_N/`` directories written by ``accelerator.save_state``
          (strict resume: model + ZeRO optimizer shards + LR scheduler +
          RNG + trainer data position). They can only be loaded after
          ``accelerator.prepare``, so here we only record the path; the
          actual load happens in ``_load_strict_checkpoint``.
        - ``steps_N_pytorch_model.pt`` / ``.safetensors`` flat weights files
          (legacy resume and the eval/deploy artifact). Loaded immediately.
        """
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._strict_ckpt_dir = None

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps, ckpt_kind = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if ckpt_kind == "state_dir":
                    self._strict_ckpt_dir = resume_from_checkpoint
                    logger.info(
                        f"Strict resume pending from checkpoint: {resume_from_checkpoint}, "
                        f"steps: {self.completed_steps} (full state loads after distributed wrap-up)"
                    )
                else:
                    logger.warning(
                        f"Resuming from LEGACY weights-only checkpoint: {resume_from_checkpoint}, "
                        f"steps: {self.completed_steps}. Optimizer/scheduler/RNG/data position are NOT restored; "
                        f"AdamW moments restart fresh and the LR scheduler is fast-forwarded by replay."
                    )
                    self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self._strict_ckpt_dir is not None:
            return  # scheduler state was restored from the strict checkpoint
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_strict_checkpoint(self):
        """Restore full training state from an accelerator state directory.

        Must run after accelerator.prepare (model/optimizer/scheduler wrapped).
        Restores model + optimizer (ZeRO shards on DeepSpeed), LR scheduler,
        per-process RNG and the trainer's data-position counters; then
        fast-forwards the data iterator to the saved position. The skip is
        applied in _create_data_iterators (which runs later and would
        otherwise overwrite the iterator).
        """
        if self._strict_ckpt_dir is None:
            return
        logger.info(f"Strict resume: loading full training state from {self._strict_ckpt_dir}")
        self.accelerator.load_state(self._strict_ckpt_dir)

        batches_consumed = int(getattr(self, "batches_consumed", 0))
        dataloader_len = max(len(self.vla_train_dataloader), 1)
        self._skip_batches = 0 if self.umi_stage_data is not None else batches_consumed % dataloader_len

        optimizer_state_params = 0
        try:
            optimizer_state_params = sum(len(states) for states in self.optimizer.state.values())
        except (AttributeError, TypeError):
            pass
        logger.info(
            f"Strict resume done: steps={self.completed_steps}, batches_consumed={batches_consumed}, "
            f"epoch_count={getattr(self, 'vla_epoch_count', 0)}, will skip {self._skip_batches} batches, "
            f"optimizer state entries={optimizer_state_params}"
        )

    def _save_checkpoint(self):
        """Save current training state.

        Writes two artifacts under checkpoints/:
        - ``steps_N/`` full accelerator state (model + optimizer, LR scheduler,
          RNG, data position) when ``trainer.save_full_state`` (default true);
          collective across ranks (DeepSpeed writes per-rank ZeRO shards).
        - ``steps_N_pytorch_model.pt`` flat weights on the main process — the
          eval/deploy artifact consumed by ``base_framework.from_pretrained``
          and the policy server; kept for backward compatibility.
        """
        save_full_state = getattr(self.config.trainer, "save_full_state", True)
        state_dir = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        if save_full_state:
            self.accelerator.save_state(state_dir)

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = _get_state_dict(self.accelerator, self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            if save_full_state:
                self.accelerator.print(f"✅ Checkpoint saved at {state_dir} and {checkpoint_path}")
            else:
                self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                if self.config.datasets.vla_data.dataset_py == "umi_indexed":
                    # Preserve every data rule and resolved stage for reproducibility.
                    OmegaConf.save(self.config.unwrap(), output_dir / "config.yaml", resolve=True)
                else:
                    self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        """Keep only the newest k checkpoints when trainer.keep_last_checkpoints > 0.

        A checkpoint at step N spans ``steps_N/`` (full accelerator state) and
        ``steps_N_pytorch_model.pt`` / ``steps_N_model.safetensors`` (flat
        weights); both artifacts are removed together. 0 keeps everything.
        """
        keep_k = int(getattr(self.config.trainer, "keep_last_checkpoints", 0) or 0)
        if keep_k <= 0:
            return
        steps = set()
        for name in os.listdir(self.checkpoint_dir):
            path = os.path.join(self.checkpoint_dir, name)
            match = re.fullmatch(r"steps_(\d+)", name)
            if match and os.path.isdir(path):
                steps.add(int(match.group(1)))
                continue
            match = re.fullmatch(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)", name)
            if match and os.path.isfile(path):
                steps.add(int(match.group(1)))
        if len(steps) <= keep_k:
            return
        pruned = sorted(steps)[:-keep_k]
        for step in pruned:
            base = os.path.join(self.checkpoint_dir, f"steps_{step}")
            if os.path.isdir(base):
                shutil.rmtree(base)
            for flat in (base + "_pytorch_model.pt", base + "_model.safetensors"):
                if os.path.exists(flat):
                    os.remove(flat)
        self.accelerator.print(
            f"Pruned {len(pruned)} old checkpoint(s) {pruned} (keep_last_checkpoints={keep_k})"
        )

    def _swanlab_config_dict(self):
        """Full resolved yaml config as a plain dict for the SwanLab Config panel."""
        try:
            cfg_obj = getattr(self.config, "_cfg", None)
            if cfg_obj is None:
                cfg_obj = self.config
            if OmegaConf.is_config(cfg_obj):
                cfg_obj = OmegaConf.to_container(cfg_obj, resolve=True)
            return cfg_obj
        except Exception as exc:
            logger.warning(f"SwanLab config dump failed; Config panel stays empty: {exc}")
            return {}

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        interface = getattr(_unwrap_model(self.accelerator, self.model), "qwen_vl_interface", None)
        visual = getattr(getattr(getattr(interface, "model", None), "model", None), "visual", None)
        moments = getattr(visual, "gate_moments", None)
        gate_metrics = moments.pop() if moments is not None else {}
        metrics.update(gate_metrics)
        if (self.completed_steps % self.config.trainer.logging_frequency == 0 or gate_metrics) and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            # len(dataloader) counts micro-batches; one optimizer step
            # consumes gradient_accumulation_steps of them, so scale by it
            # to keep the epoch metric exact when grad_accum > 1.
            metrics["epoch"] = round(
                self.accelerator.gradient_accumulation_steps
                * self.completed_steps
                / len(self.vla_train_dataloader),
                2,
            )
            if self.umi_stage_data is not None:
                metrics["epoch"] = round(self.completed_steps / self.config.datasets.vla_data.stage_total_updates, 6)
                metrics["data/stage_update"] = self.umi_stage_data.stage_updates
            if getattr(self, "_swanlab_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._swanlab_enabled = False
                    logger.warning(f"SwanLab log failed; disabling SwanLab: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators (fast-forwarding to the saved position on strict resume)."""
        if self.umi_stage_data is not None:
            if self.completed_steps < self.config.trainer.max_train_steps:
                self.umi_stage_data.start()
                self.vla_train_dataloader = self.umi_stage_data.prepared
            return
        self.vla_epoch_count = getattr(self, "vla_epoch_count", 0)
        self.batches_consumed = getattr(self, "batches_consumed", 0)
        # Epoch-seeded samplers (e.g. UMI block shuffle) must build this epoch's
        # permutation before iteration, or the skip below lands in the wrong stream.
        # The prepared DataLoaderShard forwards set_epoch to the real sampler;
        # its .sampler property is a plain SequentialSampler and would no-op.
        set_epoch = getattr(self.vla_train_dataloader, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.vla_epoch_count)
        self.vla_iter = iter(self.vla_train_dataloader)
        skip = getattr(self, "_skip_batches", 0)
        for _ in range(skip):
            next(self.vla_iter)
        self._skip_batches = 0
        if skip:
            logger.info(f"Data iterator fast-forwarded past {skip} batches (strict resume)")

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        if self.umi_stage_data is not None:
            batch = self.umi_stage_data.next()
            self.vla_train_dataloader = self.umi_stage_data.prepared
            self.batches_consumed = getattr(self, "batches_consumed", 0) + 1
            return batch
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        self.batches_consumed += 1
        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            if self.umi_stage_data is not None:
                self.umi_stage_data.commit(self.accelerator.sync_gradients)
                step_metrics["data/stage"] = self.umi_stage_data.stage_index + 1
                step_metrics["data/global_sample_cursor"] = self.umi_stage_data.cursor
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.accelerator.sync_gradients and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if self.accelerator.sync_gradients:
                self._log_metrics(step_metrics)

            if self.accelerator.sync_gradients and self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        if self.umi_stage_data is not None:
            return self.umi_stage_data.evaluate(_unwrap_model(self.accelerator, self.model), step_metrics or {})
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = _unwrap_model(self.accelerator, self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self._calculate_total_batch_size()}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        interface = getattr(_unwrap_model(self.accelerator, self.model), "qwen_vl_interface", None)
        visual = getattr(getattr(getattr(interface, "model", None), "model", None), "visual", None)
        moments = getattr(visual, "gate_moments", None)
        if moments is not None:
            frequency = int(self.config.trainer.get("gate_logging_frequency", 100))
            if frequency < 1:
                raise ValueError("gate_logging_frequency must be positive")
            moments.enabled = (self.completed_steps + 1) % frequency == 0
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # DS runs bf16 weights directly and force-disables outer autocast
            # in its forward; only the non-DS path needs the manual context.
            if deepspeed_plugin is not None:
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    total_loss = action_loss

            self.accelerator.backward(total_loss)

            grad_norm = None
            if self.config.trainer.gradient_clipping is not None:
                # Non-DS: this clips and returns the pre-clip total norm.
                # DS: the engine clips inside its own step, so the norm it
                # reports here is stale, re-read after optimizer.step().
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )

            self.optimizer.step()

            if deepspeed_plugin is not None:
                grad_norm = self.model.get_global_grad_norm()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients and not self.accelerator.optimizer_step_was_skipped:
                self.lr_scheduler.step()

        metrics = {
            "action_dit_loss": action_loss.item(),
        }
        if grad_norm is not None:
            try:
                metrics["grad_norm"] = float(grad_norm)
            except (TypeError, ValueError):
                pass
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = _get_state_dict(self.accelerator, self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_swanlab_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        if self.umi_stage_data is not None:
            self.umi_stage_data.close()
        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    from starVLA.dataloader.umi_wyt import resolve_umi_stages
    resolve_umi_stages(cfg, accelerator.num_processes)
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
