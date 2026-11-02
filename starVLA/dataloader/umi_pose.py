"""UMI model action geometry, separate from source poses and normalization.

Every target in a chunk uses the same current pose, independently per hand.
Only temporary quaternion copies are projected to unit length. State and
source arrays are never changed, and grippers always retain their target value.
"""
from copy import deepcopy
import operator

import numpy as np


WORLD = "world"
CURRENT_EEF_RELATIVE = "current_eef_relative"
TRANSFORM_VERSION = "umi-current-eef-relative-xyzw-v1"
MIN_QUATERNION_NORM = 1e-8


def validate_action_pose_repr(value):
    if value not in (WORLD, CURRENT_EEF_RELATIVE):
        raise ValueError("action_pose_repr must be world or current_eef_relative")
    return value


def model_access_meta(meta, action_pose_repr=WORLD):
    """Derive a model contract without changing the immutable source metadata."""
    result = deepcopy(meta)
    mode = validate_action_pose_repr(action_pose_repr)
    if mode == WORLD:
        result.pop("action_pose_repr", None)
    else:
        result["action_pose_repr"] = mode
    return result


def action_pose_details(action_pose_repr=CURRENT_EEF_RELATIVE):
    mode = validate_action_pose_repr(action_pose_repr)
    if mode == WORLD:
        return {"action_pose_repr": mode, "coordinate_frame": "world"}
    return {
        "version": TRANSFORM_VERSION,
        "action_pose_repr": mode,
        "coordinate_frame": "each_hand_current_eef",
        "reference_frame": "current_observation_t_for_all_targets",
        "future_offsets": list(range(1, 17)),
        "translation": "R_t_transpose @ (p_future - p_t)",
        "rotation": "conjugate(unit(q_t)) Hamilton_product unit(q_future)",
        "quaternion_order": "xyzw",
        "quaternion_unit_projection": "action_geometry_only; state_unchanged",
        "quaternion_sign": "preserve_source_lift; no_per_target_hemisphere_flip",
        "minimum_quaternion_norm_exclusive": MIN_QUATERNION_NORM,
        "gripper": "absolute_future_width_unchanged",
        "field_order": [f"robot{hand}.{field}" for hand in (1, 2)
                        for field in ("dx", "dy", "dz", "dqx", "dqy", "dqz", "dqw", "gripper")],
        "calculation_dtype": "float64",
        "output_dtype": "float32",
    }


def _array(value, label, *, audit_values=True):
    value = np.asarray(value)
    if audit_values and (value.ndim < 2 or value.shape[-1] != 16 or value.dtype.kind not in "fiu"):
        raise ValueError(f"{label} must be a real array [...,frames,16]")
    value = value.astype(np.float64, copy=True)
    if audit_values and not np.isfinite(value).all():
        raise ValueError(f"{label} contains NaN/Inf")
    return value


def _inputs(state, action, *, audit_values=True):
    state = _array(state, "state", audit_values=audit_values)
    action = _array(action, "action", audit_values=audit_values)
    if audit_values:
        if state.shape[-2] != 1 or action.shape[-2] < 1:
            raise ValueError("Expected one current state and a nonempty action horizon")
        try:
            shape = np.broadcast_shapes(state.shape[:-2], action.shape[:-2])
        except ValueError as error:
            raise ValueError("State/action batch dimensions do not broadcast") from error
        if shape != action.shape[:-2]:
            raise ValueError("State batch dimensions would expand the action shape")
    return state, action


def _unit(q, *, audit_values=True):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if audit_values and (not np.isfinite(norm).all() or np.any(norm <= MIN_QUATERNION_NORM)):
        raise ValueError("Invalid quaternion norm; cannot construct a pose rotation")
    return q / norm


def _conjugate(q):
    result = q.copy()
    result[..., :3] *= -1
    return result


def _multiply(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv),
                           aw * bw - np.sum(av * bv, axis=-1, keepdims=True)), axis=-1)


def _rotate(q, v):
    xyz = q[..., :3]
    return v + 2 * np.cross(xyz, np.cross(xyz, v) + q[..., 3:] * v)


def _fp32(value, *, audit_values=True):
    with np.errstate(over="ignore", invalid="ignore"):
        value = value.astype(np.float32)
    if audit_values and not np.isfinite(value).all():
        raise ValueError("Pose transform output overflows float32")
    return value


def actions_to_current_frame(state, action, *, audit_values=True):
    """World actions -> same-current-anchor local actions, shape [...,H,16]."""
    state, action = _inputs(state, action, audit_values=audit_values)
    result = action.copy()
    for start in (0, 8):
        anchor = _unit(state[..., start + 3:start + 7], audit_values=audit_values)
        future = _unit(action[..., start + 3:start + 7], audit_values=audit_values)
        inverse = _conjugate(anchor)
        result[..., start:start + 3] = _rotate(
            inverse, action[..., start:start + 3] - state[..., start:start + 3])
        result[..., start + 3:start + 7] = _multiply(inverse, future)
    return _fp32(result, audit_values=audit_values)


def actions_to_world_frame(state, relative_action):
    """Decode physical relative actions with the same anchor; reject invalid q.

    Call only after inverse normalization. Predicted quaternion copies are
    projected here, not in standardized coordinates or in the state input.
    """
    state, action = _inputs(state, relative_action)
    result = action.copy()
    for start in (0, 8):
        anchor = _unit(state[..., start + 3:start + 7])
        relative = _unit(action[..., start + 3:start + 7])
        result[..., start:start + 3] = (
            state[..., start:start + 3] + _rotate(anchor, action[..., start:start + 3]))
        result[..., start + 3:start + 7] = _multiply(anchor, relative)
    return _fp32(result)


def hold_current_actions(state, horizon, action_pose_repr=WORLD):
    """An auxiliary prediction strategy, never a replacement for ground truth."""
    mode = validate_action_pose_repr(action_pose_repr)
    if isinstance(horizon, (bool, np.bool_)):
        raise ValueError("horizon must be a positive integer")
    horizon = operator.index(horizon)
    state = _array(state, "state")
    if state.shape[-2] != 1 or horizon < 1:
        raise ValueError("Expected one current state and a positive horizon")
    result = np.repeat(state, horizon, axis=-2)
    if mode == CURRENT_EEF_RELATIVE:
        for start in (0, 8):
            result[..., start:start + 7] = 0
            result[..., start + 6] = 1
    return _fp32(result)
