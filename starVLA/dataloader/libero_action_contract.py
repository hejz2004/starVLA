"""Versioned, observation-anchored native OSC goals with binary gripper intent.

Geometry is float64. Model tensors are float32. Only pose channels are normalized;
gripper is always 0=close / 1=open and the unused arm is zero and loss-masked.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

VERSION = 'libero-anchored-osc-goal-binary-v1'
REPRESENTATION = 'libero_anchored_osc_goal_binary'
LAYOUT = 'anchor_relative_site_position3_body_quaternion_xyzw4_binary_open1_masked_right8'
LOSS_MASK = np.array([True] * 8 + [False] * 8)
NORM_MASK = np.array([True] * 7 + [False] * 9)
OSC_SCALE = np.array([.05, .05, .05, .5, .5, .5], dtype=np.float64)


def unit_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.any(norm < 1e-12):
        raise ValueError('Expected finite nonzero xyzw quaternion')
    return q / norm


def canonical_quaternion(q):
    q = unit_quaternion(q)
    indices = np.argmax(np.abs(q), axis=-1)
    selected = np.take_along_axis(q, indices[..., None], axis=-1)
    return np.where(selected < 0, -q, q)


def conjugate(q):
    return np.asarray(q, dtype=np.float64) * np.array([-1., -1., -1., 1.])


def multiply(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz], axis=-1)


def rotate(q, v):
    q = unit_quaternion(q)
    v = np.asarray(v, dtype=np.float64)
    t = 2 * np.cross(q[..., :3], v)
    return v + q[..., 3:] * t + np.cross(q[..., :3], t)


def rotvec_to_quaternion(v):
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    factor = .5 * np.sinc(angle / (2*np.pi))
    return canonical_quaternion(np.concatenate([v * factor, np.cos(angle / 2)], axis=-1))


def quaternion_to_rotvec(q):
    q = unit_quaternion(q)
    q = np.where(q[..., 3:] < 0, -q, q)
    sine = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(sine, q[..., 3:])
    return q[..., :3] * np.divide(angle, sine, out=np.full_like(angle, 2.), where=sine > 1e-12)


def binary_open(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError('Stored gripper actions must use [0,1] open encoding, not native [-1,1]')
    return (value > .5).astype(np.float64)


def reconstruct_goals(states, actions, initial_goal_quaternion=None):
    """Native command at row i generates goal[i], not observed state[i+1].

    Zero native rotation retains the previous orientation goal. For datasets
    starting partway through controller history, pass its initial goal explicitly.
    Body/site constant orientation offset cancels in world-frame rotation updates.
    """
    states, actions = np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 8 or actions.shape != (len(states), 7) or not len(states):
        raise ValueError('Expected matching full-episode [T,8] states and [T,7] native-source actions')
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('Nonfinite state or action')
    observed_q = rotvec_to_quaternion(states[:, 3:6])
    scaled = np.clip(actions[:, :6], -1, 1) * OSC_SCALE
    goals = np.empty((len(states), 8), dtype=np.float64)
    goals[:, :3] = states[:, :3] + scaled[:, :3]
    goals[:, 7] = binary_open(actions[:, 6])
    previous = observed_q[0] if initial_goal_quaternion is None else canonical_quaternion(initial_goal_quaternion)
    for i in range(len(states)):
        if np.any(scaled[i, 3:] != 0):
            previous = canonical_quaternion(multiply(rotvec_to_quaternion(scaled[i, 3:]), observed_q[i]))
        goals[i, 3:7] = previous
    return goals


def encode_goals(goals, anchor_position, anchor_quaternion):
    """All H targets share ONE recorded observation anchor; no pairwise deltas."""
    goals = np.asarray(goals, dtype=np.float64)
    anchor_position = np.asarray(anchor_position, dtype=np.float64)
    q = unit_quaternion(anchor_quaternion)
    if goals.ndim < 2 or goals.shape[-1] != 8 or not np.isfinite(goals).all():
        raise ValueError('Expected [...,H,8] finite goals')
    inverse = conjugate(q)[..., None, :]
    result = np.zeros((*goals.shape[:-1], 16), dtype=np.float64)
    result[..., :3] = rotate(inverse, goals[..., :3] - anchor_position[..., None, :])
    result[..., 3:7] = canonical_quaternion(multiply(inverse, goals[..., 3:7]))
    result[..., 7] = binary_open(goals[..., 7])
    return result


def decode_goals(action, anchor_position, anchor_quaternion):
    action = np.asarray(action, dtype=np.float64)
    if action.ndim < 2 or action.shape[-1] != 16 or not np.isfinite(action).all():
        raise ValueError('Expected finite [...,H,16] actions')
    q = unit_quaternion(anchor_quaternion)[..., None, :]
    result = np.empty((*action.shape[:-1], 8), dtype=np.float64)
    result[..., :3] = np.asarray(anchor_position)[..., None, :] + rotate(q, action[..., :3])
    result[..., 3:7] = canonical_quaternion(multiply(q, action[..., 3:7]))
    result[..., 7] = action[..., 7]  # Model outputs may be outside [0,1]; evaluator uses >.5.
    return result


def goal_to_native_delta(goal, current_position, current_quaternion, previous_goal_quaternion=None):
    """Convert a decoded controller goal to official bounded OSC7 input.

    current_quaternion must describe the live EEF BODY, with the same fixed
    body-to-site offset as training. Runtime eval supplies controller-synced poses.
    """
    goal = np.asarray(goal, dtype=np.float64)
    if goal.shape != (8,) or not np.isfinite(goal).all():
        raise ValueError('Expected one finite controller goal')
    delta_p = goal[:3] - np.asarray(current_position, dtype=np.float64)
    delta_r = quaternion_to_rotvec(multiply(unit_quaternion(goal[3:7]), conjugate(unit_quaternion(current_quaternion))))
    if previous_goal_quaternion is not None:
        change = quaternion_to_rotvec(multiply(unit_quaternion(goal[3:7]), conjugate(unit_quaternion(previous_goal_quaternion))))
        if np.linalg.norm(change) <= 1e-7:
            # Native OSC zero rotation means keep its existing orientation goal.
            delta_r = np.zeros(3)
    pose = np.clip(np.r_[delta_p, delta_r] / OSC_SCALE, -1., 1.)
    grip = -1. if goal[7] > .5 else 1.
    return np.r_[pose, grip].astype(np.float32)


class GoalBinaryNormalizer:
    def __init__(self, payload):
        self.payload = json.loads(Path(payload).read_text()) if isinstance(payload, (str, Path)) else payload
        if self.payload.get('contract') != VERSION or self.payload.get('action_layout') != LAYOUT:
            raise ValueError('Wrong action contract; width checkpoints cannot use binary goal normalization')
        action = self.payload['action']
        if action.get('normalization') != 'pose_q01_q99_no_clip_gripper_identity':
            raise ValueError('Unexpected normalization convention')
        self.q01 = np.asarray(action['q01'], dtype=np.float64)
        self.q99 = np.asarray(action['q99'], dtype=np.float64)
        if self.q01.shape != (16,) or self.q99.shape != (16,) or not np.isfinite(self.q01).all() or not np.isfinite(self.q99).all():
            raise ValueError('Expected finite 16D statistics')
        self.center = (self.q01 + self.q99) / 2
        self.scale = (self.q99 - self.q01) / 2
        if np.any(self.scale[:7] <= 0):
            raise ValueError('Degenerate pose normalization')
        self.scale[7:] = 1.
        self.center[7:] = 0.

    def normalize_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] != 16 or not np.isfinite(action).all():
            raise ValueError('Expected finite 16D actions')
        result = ((action - self.center) / self.scale).astype(np.float32)
        result[..., 7] = binary_open(action[..., 7])
        result[..., 8:] = 0.
        return result

    def denormalize_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] != 16 or not np.isfinite(action).all():
            raise ValueError('Expected finite 16D model predictions')
        result = action * self.scale + self.center
        result[..., 8:] = 0.
        return result
