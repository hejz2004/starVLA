"""Compute normalization statistics for the UMI LeRobot dataloader (offline).

Samples random windows through the low-dim fast path (no video decoding) and
writes a umi-lerobot-statistics-v2 JSON consumed by
starVLA/dataloader/umi_lerobot/normalize.py. The v2 scheme is uniform
per-dim q01/q99 min-max -> [-1,1] on all channels (translation,
quaternion, gripper alike); mean/std are kept as diagnostics only.

Action channels are pooled over the horizon axis before computing
statistics (chunk-anchored steps share a per-channel marginal).

Run on a host that can read the dataset:
    python examples/umi/tools/compute_umi_lerobot_statistics.py \
        --data_root_dir /path/to/umi_v30_... \
        --output statistics.json --num_samples 200000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.umi_lerobot.umi_dataset import UMILerobotDataset  # noqa: E402

RETRYABLE = (ValueError, TypeError, KeyError, OSError, IndexError, RuntimeError)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=200_000)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--anchor_stride", type=int, default=1)
    parser.add_argument("--arm_mode", default="bimanual", choices=["bimanual", "single_left", "single_right"])
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_log", type=int, default=20_000)
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    dataset = UMILerobotDataset(
        data_root_dir=args.data_root_dir,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        arm_mode=args.arm_mode,
        load_video=False,
        seed=args.seed,
        cache_dir=args.cache_dir,
        frame_stride=args.frame_stride,
        anchor_stride=args.anchor_stride,
    )
    print(f"episodes={dataset.table.num_episodes} windows={len(dataset)}")
    print(
        f"action_dim={dataset.action_dim} state_dim={dataset.state_dim} "
        f"horizon={dataset.horizon} frame_stride={dataset.stride}"
    )

    rng = np.random.default_rng(args.seed)
    state_rows, action_rows = [], []
    failures = 0

    while len(state_rows) < args.num_samples:
        candidate = int(rng.integers(len(dataset)))
        try:
            sample = dataset.read_lowdim(candidate)
        except RETRYABLE as error:
            failures += 1
            if failures <= 5 or failures % 1000 == 0:
                print(f"skip window {candidate}: {error}")
            continue
        state_rows.append(sample["state"][0])
        action_rows.append(sample["action"].reshape(-1))
        count = len(state_rows)
        if count % args.batch_log == 0:
            print(f"progress {count}/{args.num_samples} (skipped {failures})")

    state = np.stack(state_rows)                       # [N, D]
    action = np.stack(action_rows).reshape(-1, dataset.action_dim)  # [N*H, D]

    def block(x: np.ndarray) -> dict:
        q01, q99 = np.percentile(x, [1.0, 99.0], axis=0)
        mean, std = x.mean(axis=0), x.std(axis=0)
        return {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    payload = {
        "schema": "umi-lerobot-statistics-v2",
        "data_root_dir": str(args.data_root_dir),
        "num_samples": len(state_rows),
        "skipped": failures,
        "arm_mode": args.arm_mode,
        "action_horizon": args.action_horizon,
        "frame_stride": args.frame_stride,
        "anchor_stride": args.anchor_stride,
        "seed": args.seed,
        "state": block(state),
        "action": block(action),
    }

    for key in ("state", "action"):
        span = np.asarray(payload[key]["q99"]) - np.asarray(payload[key]["q01"])
        order = np.argsort(span)
        summary = ", ".join(f"dim{int(i)}={span[int(i)]:.3e}" for i in order[:5])
        print(f"{key} smallest q01/q99 spans: {summary}")
        risky = np.flatnonzero(span < 1e-4)
        if risky.size:
            print(
                f"WARNING: {key} dims with span < 1e-4 will hard-fail UMINormalizer "
                f"(degenerate dims {risky.tolist()}); inspect these channels before training."
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {output} (count={len(state_rows)}, skipped={failures})")


if __name__ == "__main__":
    main()
