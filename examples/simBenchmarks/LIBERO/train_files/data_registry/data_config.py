"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.umi_lerobot.normalize import UMINormalizer
from starVLA.dataloader.umi_lerobot.umi_dataset import (
    quat_canonicalize, quat_conjugate, quat_multiply, quat_rotate,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


def _axis_angle_to_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """Convert absolute axis-angle EEF orientations to xyzw quaternions."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1)
    scale = np.empty_like(angle)
    small = angle < 1e-6
    angle2 = angle[small] ** 2
    scale[small] = 0.5 - angle2 / 48.0 + angle2 ** 2 / 3840.0
    scale[~small] = np.sin(angle[~small] / 2.0) / angle[~small]
    return np.concatenate(
        [rotvec * scale[..., None], np.cos(angle / 2.0)[..., None]], axis=-1
    )


def libero_state_windows_to_umi16(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert batches of [t, ..., t+H] EEF states to anchor-relative actions.

    Input is [N, H+1, 8]: position(3), absolute axis-angle(3), finger
    joints(2). Output is [N, H, 16], with the unused arm masked.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 3 or state.shape[-1] != 8 or state.shape[1] < 2:
        raise ValueError(f"LIBERO state windows must be [N, H+1, 8], got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("LIBERO state windows contain NaN/Inf")

    q = _axis_angle_to_xyzw(state[..., 3:6])
    anchor_inv = np.broadcast_to(quat_conjugate(q[:, :1]), q[:, 1:].shape)
    relative_position = quat_rotate(anchor_inv, state[:, 1:, :3] - state[:, :1, :3])
    relative_quaternion = quat_canonicalize(quat_multiply(anchor_inv, q[:, 1:]))
    gripper_width = state[:, 1:, 6:7] - state[:, 1:, 7:8]

    action = np.zeros((state.shape[0], state.shape[1] - 1, 16), dtype=np.float32)
    action[..., :3] = relative_position
    action[..., 3:7] = relative_quaternion
    action[..., 7:8] = gripper_width
    mask = np.zeros_like(action, dtype=bool)
    mask[..., :8] = True
    return action, mask


def libero_state_window_to_umi16(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map one [state_t, ..., state_t+H] window to a 16D UMI action chunk."""
    state = np.asarray(state)
    if state.ndim != 2 or state.shape[1] != 8 or state.shape[0] < 2:
        raise ValueError(f"LIBERO state window must be [H+1, 8], got {state.shape}")
    action, mask = libero_state_windows_to_umi16(state[None])
    return action[0], mask[0]


class LiberoUMI16Dataset(LeRobotSingleDataset):
    """Use future observed EEF poses, not per-step OSC commands, as UMI targets."""

    def __init__(self, *args, **kwargs):
        data_cfg = kwargs.get("data_cfg")
        stats_path = data_cfg.get("normalization_statistics") if data_cfg else None
        if not stats_path:
            raise ValueError("libero_umi16 requires datasets.vla_data.normalization_statistics")
        if data_cfg.get("include_state", False) not in (False, "False"):
            raise ValueError("libero_umi16 currently requires include_state: false")
        self.umi_normalizer = UMINormalizer(stats_path)
        if len(self.umi_normalizer.payload["action"]["q01"]) != 16:
            raise ValueError("libero_umi16 requires 16D action statistics")
        super().__init__(*args, **kwargs)

    def checkpoint_action_statistics(self) -> dict:
        # Training labels use these 16D external statistics, not the source
        # LeRobot dataset's 7D controller-command statistics.
        block = self.umi_normalizer.payload["action"]
        return {
            "q01": block["q01"],
            "q99": block["q99"],
            "mean": block["mean"],
            "std": block["std"],
            "mask": [True] * 16,
        }

    def _get_all_steps(self):
        # The base index contains every episode step and its cache is shared.
        # UMI windows require all H future observations to be real, not clamped.
        steps = super()._get_all_steps()
        lengths = dict(zip(self.trajectory_ids, self.trajectory_lengths))
        horizon = len(self.modality_configs["action"].delta_indices)
        return [(episode, index) for episode, index in steps
                if index + horizon < lengths[episode]]

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        state = np.concatenate(
            [np.asarray(data[key]) for key in self.modality_keys["state"]], axis=1
        )
        expected_rows = len(self.modality_configs["action"].delta_indices) + 1
        if state.shape != (expected_rows, 8):
            raise ValueError(f"LIBERO EEF state window must be [{expected_rows}, 8], got {state.shape}")
        action, mask = libero_state_window_to_umi16(state)
        action = self.umi_normalizer.normalize_action(action)
        action[:, 8:] = 0.0  # neutral inputs for the masked right-arm slots
        sample["action"] = action.astype(sample["action"].dtype, copy=False)
        sample["action_mask"] = mask
        return sample


class LiberoUMI16DataConfig(Libero4in1DataConfig):
    """LIBERO observations at t..t+16 for UMI-style relative action chunks."""

    # The mixture sampler uses action offsets to trim episode tails. The
    # targets are future observations t+1..t+16, so include that final frame.
    action_indices = list(range(1, 17))
    state_indices = list(range(17))

    def make_dataset(self, **kwargs):
        horizon = int(kwargs["data_cfg"].get("action_horizon", 16))
        if horizon != len(self.action_indices):
            raise ValueError(f"libero_umi16 expects action_horizon=16, got {horizon}")
        return LiberoUMI16Dataset(**kwargs)


class OpenPILibero4in1DataConfig(Libero4in1DataConfig):
    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys + self.action_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.roll": "q99",
                    "state.pitch": "q99",
                    "state.yaw": "q99",
                    "state.pad": "q99",
                    "state.gripper": "q99",
                },
            ),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "q99",
                },
            ),
        ])


from examples.simBenchmarks.LIBERO.train_files.data_registry.action_goal_binary_config import LiberoActionGoalBinaryDataConfig


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "libero_franka_umi16": LiberoUMI16DataConfig(),
    "libero_franka_goal_binary": LiberoActionGoalBinaryDataConfig(),
    "openpi_libero_franka": OpenPILibero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all_anchored_osc_goal_binary": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_goal_binary"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_goal_binary"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_goal_binary"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_goal_binary"),
    ],
    "libero_object_anchored_osc_goal_binary": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_goal_binary"),
    ],
    "libero_object_umi16": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_umi16"),
    ],
    "libero_all_umi16": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_umi16"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_umi16"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_umi16"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_umi16"),
    ],
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "openpi_libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "openpi_libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}
