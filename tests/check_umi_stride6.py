"""Host-side validation of umi_lerobot frame_stride=6 (5 Hz) support.

Checks, on the real curated UMI dataset:
  1. stride-6 table anchors: per-episode max(length - 6*16, 0), total windows
  2. stride-6 windows read and validate; action (16,16), state (1,16), finite
  3. cross-check: stride-6 action[k] == stride-1 action[6*(k+1)-1] at the same
     anchor (both are the relative pose t -> t + 6*(k+1))
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.umi_lerobot.umi_dataset import UMILerobotDataset

ROOT = "/mnt/nas/public/roban_umi/restricted_data/derived/umi_v30_curated_luna_annotated_rational_v1_20261005"
HORIZON = 16
STRIDE = 6


def main():
    t0 = time.time()
    ds6 = UMILerobotDataset(
        data_root_dir=ROOT,
        action_horizon=HORIZON,
        load_video=False,
        frame_stride=STRIDE,
        cache_dir="/tmp/umi_cache_s6",
    )
    print(f"[stride6] episodes={ds6.table.num_episodes} windows={len(ds6)} ({time.time()-t0:.1f}s)")

    lengths = ds6.table.length
    expected = int(np.maximum(lengths - HORIZON * STRIDE, 0).sum())
    assert len(ds6) == expected, f"window total {len(ds6)} != expected {expected}"
    per_ep = ds6.table.anchors()
    sample_pos = [0, ds6.table.num_episodes // 2, ds6.table.num_episodes - 1]
    for pos in sample_pos:
        assert per_ep[pos] == max(int(lengths[pos]) - HORIZON * STRIDE, 0), f"anchors wrong at ep {pos}"
    print(f"[stride6] anchors = length - {HORIZON * STRIDE} OK (ep0 len={int(lengths[0])} anchors={int(per_ep[0])})")

    indices = [0, len(ds6) // 3, len(ds6) // 2, len(ds6) - 1]
    windows6 = {}
    for idx in indices:
        w = ds6.read_window(idx)
        action, state = ds6.make_lowdim(w)
        assert action.shape == (HORIZON, 8 * len(ds6.arms)), f"action shape {action.shape}"
        assert state.shape == (1, 8 * len(ds6.arms)), f"state shape {state.shape}"
        assert np.isfinite(action).all() and np.isfinite(state).all(), f"non-finite at window {idx}"
        windows6[idx] = (w, action, state)
        print(f"[stride6] window {idx}: pos={w['position']} anchor={w['anchor']} lang={w['lang'][:50]!r} OK")

    ds1 = UMILerobotDataset(
        data_root_dir=ROOT,
        action_horizon=HORIZON * STRIDE,
        load_video=False,
        frame_stride=1,
        cache_dir="/tmp/umi_cache_s1",
    )
    print(f"[stride1] windows={len(ds1)} horizon={HORIZON * STRIDE}")

    for idx, (w6, action6, state6) in windows6.items():
        pos, anchor = w6["position"], w6["anchor"]
        w1 = ds1.read_window(_locate(ds1, pos, anchor))
        _, action1, state1 = _lowdim(ds1, w1)
        for k in range(HORIZON):
            a6 = action6[k]          # t -> t + 6*(k+1)
            a1 = action1[STRIDE * (k + 1) - 1]  # t -> t + 6*(k+1) via stride-1
            if not np.allclose(a6, a1, rtol=0, atol=1e-6):
                raise AssertionError(
                    f"window {idx} k={k}: stride6 action differs from stride1 (max diff "
                    f"{np.abs(a6 - a1).max():.3e})"
                )
        assert np.allclose(state6, state1, rtol=0, atol=1e-6), f"window {idx}: state mismatch"
        print(f"[cross] window {idx}: stride6 actions == stride1 actions at 6x lags OK")

    print("ALL_CHECKS_PASSED")


def _locate(ds, pos, anchor):
    base = int(ds.table._cumulative[pos - 1]) if pos else 0
    return base + anchor


def _lowdim(ds, w):
    action, state = ds.make_lowdim(w)
    return w, action, state


if __name__ == "__main__":
    main()
