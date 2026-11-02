"""Attention boundaries for packed, independent multi-view image samples."""
from numbers import Integral
import torch


def global_attention_indexes(config):
    cfg = getattr(config, "cross_view_attention", None) or {}
    unknown = set(cfg) - {"enabled", "start_layer"}
    if unknown:
        raise ValueError(f"Unknown cross_view_attention settings: {sorted(unknown)}")
    if not cfg.get("enabled", False):
        return ()
    depth = int(config.depth)
    start = cfg.get("start_layer", None)
    start = (2 * depth) // 3 if start is None else start
    if isinstance(start, bool) or not isinstance(start, Integral) or not 0 <= start < depth:
        raise ValueError(f"cross_view_attention.start_layer must be an integer in [0, {depth}), got {start}")
    return tuple(range(int(start), depth, 2))


def scene_cu_seqlens(grid_thw, counts, dtype=torch.int32):
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or not len(grid_thw):
        raise ValueError("Expected nonempty image_grid_thw with shape [images, 3]")
    if (grid_thw <= 0).any() or (grid_thw[:, 0] != 1).any():
        raise ValueError("Cross-view attention requires positive image grids with temporal size 1")
    if counts is None:
        raise ValueError("scene_image_counts is required to prevent cross-sample attention")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in counts):
        raise ValueError("scene_image_counts must contain nonnegative integers")
    if sum(counts) != len(grid_thw):
        raise ValueError("scene_image_counts must sum to the number of image grids")
    patches = grid_thw.prod(dim=-1)
    lengths = []
    offset = 0
    for n in counts:
        if n:
            lengths.append(patches[offset:offset+n].sum())
        offset += n
    lengths = torch.stack(lengths).to(dtype=dtype)
    return torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=dtype)])


def image_counts_from_input_ids(input_ids, vision_start_token_id, image_token_id, num_images):
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("Cross-view image batches require 2D input_ids to identify sample boundaries")
    counts = ((input_ids[:, :-1] == vision_start_token_id) &
              (input_ids[:, 1:] == image_token_id)).sum(dim=1).tolist()
    if sum(counts) != num_images:
        raise ValueError(f"Found {counts} images per sample, but image_grid_thw has {num_images} images")
    return counts
