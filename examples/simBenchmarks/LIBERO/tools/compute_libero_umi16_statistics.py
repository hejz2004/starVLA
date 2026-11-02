"""Compute LIBERO UMI16 normalization statistics from LeRobot Parquet states.

This reads only observation.state, never videos or raw 7D controller actions.
Every valid anchor t with t+H inside its episode contributes H future EEF
targets. Action quantiles pool those H targets, matching the training labels.

Example:
    python examples/simBenchmarks/LIBERO/tools/compute_libero_umi16_statistics.py \
        --data-root-dir playground/Datasets/LEROBOT_LIBERO_DATA \
        --output playground/Datasets/LEROBOT_LIBERO_DATA/libero_umi16_statistics_h16.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from examples.simBenchmarks.LIBERO.train_files.data_registry.data_config import (  # noqa: E402
    DATASET_NAMED_MIXTURES,
    _axis_angle_to_xyzw,
    libero_state_windows_to_umi16,
)
from starVLA.dataloader.umi_lerobot.normalize import LIBERO_SCHEMA, MIN_SPAN  # noqa: E402
from starVLA.dataloader.umi_lerobot.umi_dataset import quat_canonicalize  # noqa: E402

ARM_DIM = 8
ACTION_DIM = 16


def _episode_states(dataset_dir: Path, info: dict, episode_index: int) -> np.ndarray:
    relative = info["data_path"].format(
        episode_chunk=episode_index // info["chunks_size"],
        episode_index=episode_index,
    )
    path = dataset_dir / relative
    table = pq.read_table(path, columns=["observation.state"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != ARM_DIM:
        raise ValueError(f"{path}: expected observation.state [T, 8], got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError(f"{path}: observation.state contains NaN/Inf")
    return state


def _statistics(rows: np.ndarray, name: str) -> dict:
    """Fit physical channels, then give the masked arm neutral spans."""
    if rows.ndim != 2 or rows.shape[1] != ARM_DIM or not len(rows):
        raise ValueError(f"{name}: expected nonempty [N, 8], got {rows.shape}")
    q01, q99 = np.percentile(rows, [1.0, 99.0], axis=0)
    bad = np.flatnonzero(q99 - q01 < MIN_SPAN)
    if bad.size:
        raise ValueError(
            f"{name}: degenerate q01/q99 span on active dims {bad.tolist()}; "
            "inspect the dataset or increase the sampled episode count"
        )
    return {
        "q01": np.concatenate([q01, np.full(ARM_DIM, -1.0)]).tolist(),
        "q99": np.concatenate([q99, np.full(ARM_DIM, 1.0)]).tolist(),
        "mean": np.concatenate([rows.mean(axis=0), np.zeros(ARM_DIM)]).tolist(),
        "std": np.concatenate([rows.std(axis=0), np.ones(ARM_DIM)]).tolist(),
    }


def compute(data_root_dir: Path, data_mix: str, horizon: int, max_episodes: int | None) -> dict:
    if horizon < 1:
        raise ValueError("action_horizon must be positive")
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise ValueError(f"unknown data_mix: {data_mix}")
    mixture = DATASET_NAMED_MIXTURES[data_mix]
    if not mixture or any(robot_type != "libero_franka_umi16" for _, _, robot_type in mixture):
        raise ValueError(f"{data_mix} must contain only LIBERO UMI16 datasets")

    state_rows, action_rows = [], []
    counts = {}
    for dataset_name, _, _ in mixture:
        dataset_dir = data_root_dir / dataset_name
        info = json.loads((dataset_dir / "meta/info.json").read_text())
        episodes = [
            json.loads(line)
            for line in (dataset_dir / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        windows_count = 0
        skipped_short = 0
        for episode in episodes:
            state = _episode_states(dataset_dir, info, int(episode["episode_index"]))
            if len(state) != int(episode["length"]):
                raise ValueError(f"{dataset_name} episode {episode['episode_index']}: length mismatch")
            if len(state) <= horizon:
                skipped_short += 1
                continue
            # sliding_window_view is a view; only the 8D active-arm targets
            # are kept in memory for exact pooled q01/q99 quantiles.
            windows = np.lib.stride_tricks.sliding_window_view(
                state, window_shape=horizon + 1, axis=0
            ).transpose(0, 2, 1)
            action, mask = libero_state_windows_to_umi16(windows)
            if not mask[..., :ARM_DIM].all() or mask[..., ARM_DIM:].any():
                raise ValueError("LIBERO action mask layout changed")
            action_rows.append(action[..., :ARM_DIM].reshape(-1, ARM_DIM))
            anchors = windows[:, 0, :]
            anchor_quat = quat_canonicalize(_axis_angle_to_xyzw(anchors[:, 3:6]))
            gripper_width = anchors[:, 6:7] - anchors[:, 7:8]
            state_rows.append(
                np.concatenate([anchors[:, :3], anchor_quat, gripper_width], axis=1).astype(np.float32)
            )
            windows_count += len(windows)
        counts[dataset_name] = {
            "episodes": len(episodes),
            "valid_windows": windows_count,
            "skipped_short_episodes": skipped_short,
        }
        print(f"{dataset_name}: {counts[dataset_name]}", flush=True)

    if not state_rows:
        raise ValueError("no complete LIBERO action windows found")
    state = np.concatenate(state_rows)
    action = np.concatenate(action_rows)
    payload = {
        "schema": LIBERO_SCHEMA,
        "data_root_dir": str(data_root_dir.resolve()),
        "data_mix": data_mix,
        "action_horizon": horizon,
        "num_samples": int(len(state)),
        "num_action_rows": int(len(action)),
        "max_episodes_per_dataset": max_episodes,
        "datasets": counts,
        "action_layout": "left [relative xyz, xyzw, future gripper width]; right masked",
        "action_mask": [True] * ARM_DIM + [False] * ARM_DIM,
        "state": _statistics(state, "state"),
        "action": _statistics(action, "action"),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-mix", default="libero_all_umi16")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-episodes-per-dataset", type=int, default=None)
    args = parser.parse_args()
    if args.max_episodes_per_dataset is not None and args.max_episodes_per_dataset < 1:
        parser.error("--max-episodes-per-dataset must be positive")

    payload = compute(
        args.data_root_dir,
        args.data_mix,
        args.action_horizon,
        args.max_episodes_per_dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output}: {payload['num_samples']} windows, "
          f"{payload['num_action_rows']} action rows", flush=True)


if __name__ == "__main__":
    main()
