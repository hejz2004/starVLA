"""UMI LeRobot v3.0 dataloading for StarVLA.

Usage: set ``datasets.vla_data.dataset_py: umi_lerobot`` in the training YAML.
"""

from __future__ import annotations

import logging

from torch.utils.data import DataLoader

from starVLA.dataloader.umi_lerobot.normalize import UMINormalizedDataset, UMINormalizer
from starVLA.dataloader.umi_lerobot.umi_dataset import ARM_MODES, UMILerobotDataset
from starVLA.dataloader.umi_lerobot.umi_sampler import UMIBlockShuffleSampler

logger = logging.getLogger(__name__)


def umi_collate_fn(batch):
    """StarVLA consumes a list of per-example dicts (PIL images inside)."""
    if not batch:
        raise ValueError("cannot collate an empty UMI batch")
    return list(batch)


def _resolved_dim(data_cfg, model_cfg, name) -> int:
    data_value, model_value = data_cfg.get(name), (model_cfg.get(name) if model_cfg else None)
    if data_value is not None and model_value is not None and int(data_value) != int(model_value):
        raise ValueError(
            f"UMI {name} mismatch: datasets.vla_data={data_value}, framework.action_model={model_value}"
        )
    value = data_value if data_value is not None else model_value
    if value is None:
        raise ValueError(f"umi_lerobot requires {name} in datasets.vla_data or framework.action_model")
    return int(value)


def make_umi_lerobot_dataloader(cfg) -> DataLoader:
    data = cfg.datasets.vla_data
    action_model_cfg = getattr(getattr(cfg, "framework", None), "action_model", None)

    action_dim = _resolved_dim(data, action_model_cfg, "action_dim")
    state_dim = _resolved_dim(data, action_model_cfg, "state_dim")
    horizon = _resolved_dim(data, action_model_cfg, "action_horizon")
    arm_mode = data.get("arm_mode", "bimanual")
    expected = 16 if arm_mode == "bimanual" else 8
    if action_dim != expected or state_dim != expected:
        raise ValueError(
            f"arm_mode={arm_mode} implies action_dim=state_dim={expected}, got {action_dim}/{state_dim}"
        )

    dataset = UMILerobotDataset(
        data_root_dir=data.data_root_dir,
        action_horizon=horizon,
        image_size=data.get("image_size", [224, 224]),
        views=data.get("views"),
        arm_mode=arm_mode,
        load_video=bool(data.get("load_video", True)),
        cache_dir=data.get("cache_dir"),
        num_retry=int(data.get("num_retry", 20)),
        seed=int(data.get("seed", getattr(cfg, "seed", 42))),
        frame_stride=int(data.get("frame_stride", 1)),
        anchor_stride=int(data.get("anchor_stride", 1)),
    )

    statistics_path = data.get("normalization_statistics")
    if statistics_path:
        dataset = UMINormalizedDataset(dataset, UMINormalizer(statistics_path))
        logger.info(f"[umi_lerobot] normalization enabled from {statistics_path}")
    elif not data.get("allow_unnormalized", False):
        raise ValueError(
            "umi_lerobot requires datasets.vla_data.normalization_statistics "
            "(compute it with examples/umi/tools/compute_umi_lerobot_statistics.py) "
            "or an explicit allow_unnormalized: true"
        )

    workers = int(data.get("num_workers", 8))
    import torch

    seed = int(data.get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    # Block-shuffles the global window stream (Accelerate shards it per rank);
    # deterministic per epoch, so strict resume stays position-exact.
    sampler = UMIBlockShuffleSampler(
        len(dataset),
        block_size=int(data.get("shuffle_block_size", 4096)),
        seed=seed,
        shuffle=bool(data.get("shuffle", True)),
    )
    kwargs = dict(
        dataset=dataset,
        batch_size=int(data.per_device_batch_size),
        sampler=sampler,
        collate_fn=umi_collate_fn,
        num_workers=workers,
        pin_memory=bool(data.get("pin_memory", False)),
        drop_last=bool(data.get("drop_last", True)),
        generator=generator,
    )
    if workers > 0:
        kwargs.update(
            persistent_workers=bool(data.get("persistent_workers", True)),
            prefetch_factor=int(data.get("prefetch_factor", 4)),
            # fork (torch's Linux default): workers inherit the parent's memory
            # instead of re-importing __main__, which would re-run module-level
            # initialize_overwatch()/Accelerator and hang in init_process_group.
            multiprocessing_context=data.get("multiprocessing_context", "fork"),
        )
    loader = DataLoader(**kwargs)
    loader.umi_dataset = dataset
    loader.umi_sampler = sampler
    return loader


__all__ = [
    "UMILerobotDataset",
    "UMINormalizedDataset",
    "UMIBlockShuffleSampler",
    "ARM_MODES",
    "make_umi_lerobot_dataloader",
]
