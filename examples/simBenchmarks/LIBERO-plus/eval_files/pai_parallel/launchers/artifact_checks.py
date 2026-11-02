"""Small action checks shared by the real worker and CPU-only tests."""
from __future__ import annotations

import numpy as np


def validate_action_components(world_vector, rotation_delta, open_gripper) -> None:
    parts = tuple(np.asarray(x, dtype=np.float32).reshape(-1)
                  for x in (world_vector, rotation_delta, open_gripper))
    if tuple(x.size for x in parts) != (3, 3, 1):
        raise ValueError(f"Invalid action component sizes: {[x.shape for x in parts]}")
    if not all(np.isfinite(x).all() for x in parts):
        raise ValueError("Policy returned a non-finite action component")


def validate_action_array(actions) -> np.ndarray:
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 7:
        raise ValueError(f"Expected nonempty N x 7 actions, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Action trajectory contains non-finite or non-numeric values")
    return array
