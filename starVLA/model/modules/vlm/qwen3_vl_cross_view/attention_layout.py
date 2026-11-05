"""Attention boundaries for packed, independent multi-view image samples."""
from numbers import Integral
import torch


def global_attention_indexes(config):
    cfg = getattr(config, "cross_view_attention", None) or {}
    unknown = set(cfg) - {"enabled", "start_layer", "num_register_tokens"}
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


def register_token_count(config):
    cfg = getattr(config, "cross_view_attention", None) or {}
    count = cfg.get("num_register_tokens", 0)
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError(f"cross_view_attention.num_register_tokens must be a nonnegative integer, got {count}")
    return int(count)


def view_register_layout(cu_seqlens, num_register_tokens):
    """Prepend registers to each image while retaining the original patch order."""
    num_images = cu_seqlens.numel() - 1
    image_ids = torch.arange(num_images, device=cu_seqlens.device)
    local_cu_seqlens = cu_seqlens + num_register_tokens * torch.arange(
        num_images + 1, device=cu_seqlens.device, dtype=cu_seqlens.dtype
    )
    patch_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    num_patches = int(cu_seqlens[-1].item())
    patch_image_ids = torch.repeat_interleave(image_ids, patch_lengths, output_size=num_patches)
    patch_indices = torch.arange(num_patches, device=cu_seqlens.device) + num_register_tokens * (patch_image_ids + 1)
    register_indices = (
        local_cu_seqlens[:-1].long().unsqueeze(1) + torch.arange(num_register_tokens, device=cu_seqlens.device)
    ).flatten()
    return local_cu_seqlens, patch_indices, register_indices


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
