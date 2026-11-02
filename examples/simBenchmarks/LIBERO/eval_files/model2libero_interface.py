# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, explicit action decoding, and chunk-cache scheduling.
"""

from collections import deque
from starVLA.dataloader.libero_action_contract import VERSION as GOAL_BINARY_VERSION, REPRESENTATION as GOAL_BINARY_REP, decode_goals, goal_to_native_delta
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size: Sequence[int] = (224, 224),
        action_interface: str = "anchored_goal_binary",
    ) -> None:
        interfaces = {"anchored_goal_binary": GOAL_BINARY_REP,
                      "native_osc7": "controller_delta"}
        if action_interface not in interfaces:
            raise ValueError(f"Unknown explicit action_interface: {action_interface!r}")
        self.action_interface = action_interface
        self.action_representation = interfaces[action_interface]
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be positive")
        self._server_metadata = meta
        received = meta.get("action_representation", "controller_delta")
        if received != self.action_representation:
            raise ValueError(f"Explicit interface {action_interface!r} requires {self.action_representation!r}, server declares {received!r}")
        if self.action_representation == GOAL_BINARY_REP:
            if meta.get("action_contract", {}).get("version") != GOAL_BINARY_VERSION:
                raise ValueError("Goal model lacks the required binary action contract metadata")
        self._anchor_pos = None
        self._anchor_quat = None

        self.image_size: tuple = tuple(image_size)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self._anchor_pos = None
        self._anchor_quat = None

    def step(self, example: dict, step: int = 0, *, eef_pos=None, eef_quat=None, diagnostic_seed=None, controller_eef_pos=None, controller_eef_quat=None, controller_goal_quat=None, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Resize images to self.image_size if needed.
        if self.image_size and example.get("image"):
            resized = []
            target_hw = self.image_size  # (H, W)
            for img in example["image"]:
                arr = np.asarray(img)
                if arr.shape[:2] != target_hw:
                    arr = np.asarray(
                        Image.fromarray(arr).resize(
                            (target_hw[1], target_hw[0]), Image.BILINEAR
                        )
                    )
                resized.append(arr)
            example = {**example, "image": resized}

        # Refresh chunk if needed. Anchored goals are relative to the EEF
        # observation at this exact chunk start, not the previous action step.
        refresh_chunk = step % self.action_chunk_size == 0 or self.raw_actions is None
        if self.action_representation == GOAL_BINARY_REP and (eef_pos is None or eef_quat is None):
            raise ValueError("Anchored-goal execution needs the current EEF position and quaternion")
        if self.action_representation == GOAL_BINARY_REP and any(x is None for x in (controller_eef_pos, controller_eef_quat, controller_goal_quat)):
            raise ValueError("Binary-goal execution requires synchronized native controller/body poses and previous goal")
        if refresh_chunk:
            if self.action_representation == GOAL_BINARY_REP:
                self._anchor_pos = np.asarray(eef_pos, dtype=np.float64).copy()
                self._anchor_quat = np.asarray(eef_quat, dtype=np.float64).copy()
            vla_input = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
            # Embodied policies degrade SILENTLY (no error) when the eval-time observation
            # differs from what the model saw during TRAINING. Verify these match the
            # training config used for this checkpoint:
            #   - state       : whether proprioceptive state is included (and its dim/order/normalization))
            #   - image size  : resize / crop resolution (e.g. 224x224)
            #   - image count : how many camera views are fed
            #   - image order : the ordering of those camera views
            #   - action normalization: unnorm_key must match the training dataset stats
            # ==============================================================================
            if diagnostic_seed is not None:
                vla_input["diagnostic_seed"] = int(diagnostic_seed)
            response = self.client.predict_action(vla_input)
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            batch = np.asarray(actions_batch)
            expected_dim = 16 if self.action_representation == GOAL_BINARY_REP else 7
            if (batch.ndim != 3 or batch.shape[0] != 1
                    or batch.shape[1] < self.action_chunk_size or batch.shape[2] != expected_dim
                    or not np.isfinite(batch).all()):
                raise ValueError(f"Invalid action chunk for {self.action_interface}: {batch.shape}")
            self.raw_actions = batch[0]  # (T, D), unnormalized server-side

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        if self.action_representation == GOAL_BINARY_REP:
            if raw_actions.shape != (1, 16):
                raise ValueError("Binary-goal checkpoint must return 16D actions")
            goal = decode_goals(raw_actions, self._anchor_pos, self._anchor_quat)[0]
            osc = goal_to_native_delta(goal, controller_eef_pos, controller_eef_quat, controller_goal_quat)
            return {"raw_action": {"world_vector": osc[:3], "rotation_delta": osc[3:6],
                "gripper_command": osc[6:7].copy(), "open_gripper": raw_actions[0, 7:8].copy()}}
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {"raw_action": raw_action}

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
