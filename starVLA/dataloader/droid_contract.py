"""DROID controller goals in the UMI current-observation EEF convention.

Physical cache rows contain observed xyz/xyzw/open, then commanded xyz/xyzw/open.
The command on row t was issued after observation t: offsets are 0..H-1.
"""
import numpy as np

from .umi_pose import actions_to_current_frame

VERSION = "droid-command-current-eef-xyzw-open-v1"
CAMERAS = ("observation.images.exterior_1_left", "observation.images.exterior_2_left",
           "observation.images.wrist_left")


def contract(horizon=16, stride=3):
    return dict(version=VERSION, horizon=int(horizon), anchor_stride=int(stride), anchor_phase=0,
                command_offsets=list(range(horizon)), action_source="action.cartesian_position",
                anchor_source="observation.state.cartesian_position", euler="xyz_radians",
                quaternion="xyzw; episode-continuous lift; no target hemisphere flip",
                translation="R_obs_t.T @ (p_command - p_obs_t)",
                rotation="conjugate(q_obs_t) * q_command", gripper="1 - action.gripper_position",
                gripper_units="continuous open fraction", active_dimensions=list(range(8)),
                padded_dimensions=list(range(8, 16)), native_fps=15, cameras=list(CAMERAS))


def continuous_quaternions(euler):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler("xyz", np.asarray(euler, dtype=np.float64)).as_quat()
    if len(q) > 1:
        signs = np.where(np.sum(q[:-1] * q[1:], axis=1) < 0, -1, 1)
        q[1:] *= np.cumprod(signs)[:, None]
    return q


def cache_poses(observed, commanded, observed_grip, commanded_grip):
    observed, commanded = np.asarray(observed), np.asarray(commanded)
    oq, cq = continuous_quaternions(observed[:, 3:]), continuous_quaternions(commanded[:, 3:])
    # Share the initial lift; subsequent changes preserve each source trajectory.
    if np.dot(oq[0], cq[0]) < 0:
        cq *= -1
    return np.column_stack((observed[:, :3], oq, 1 - observed_grip,
                            commanded[:, :3], cq, 1 - commanded_grip)).astype(np.float32)


def relative_actions(rows, anchors, horizon=16):
    """Batched physical 8D targets; use the SAME observed anchor for all H commands."""
    rows = np.asarray(rows)
    anchors = np.asarray(anchors, dtype=np.int64)
    state = np.zeros((len(anchors), 1, 16), dtype=np.float32)
    action = np.zeros((len(anchors), horizon, 16), dtype=np.float32)
    state[:, 0, :8] = rows[anchors, :8]
    action[:, :, :8] = rows[anchors[:, None] + np.arange(horizon), 8:]
    # Identity quaternion for the inactive arm makes the shared UMI transform valid.
    state[:, :, 14] = 1
    action[:, :, 14] = 1
    return actions_to_current_frame(state, action)[:, :, :8]


def anchor_ranges(length, horizon, stride, languages):
    """Compact [first_anchor, count, language] runs; episode phase stays zero."""
    result = []
    for t in range(0, max(0, length - horizon + 1), stride):
        text = languages[t]
        if not text:
            continue
        if result and result[-1][2] == text and result[-1][0] + result[-1][1] * stride == t:
            result[-1][1] += 1
        else:
            result.append([t, 1, text])
    return result
