"""Host-side validation of umi_lerobot anchor_stride=6 (with frame_stride=6).

Checks, on the real curated UMI dataset:
  1. window total == sum(ceil((length - 6*16) / 6)) over episodes
  2. spot windows read and validate; anchor is a multiple of 6;
     consecutive window anchors inside an episode differ by 6
  3. semantic equivalence: stride-6/anchor-6 action[k] == stride-1
     (horizon=96) action[6*(k+1)-1] at the same anchor, all lags
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
ANCHOR = 6


def main():
    t0 = time.time()
    ds = UMILerobotDataset(
        data_root_dir=ROOT,
        action_horizon=HORIZON,
        load_video=False,
        frame_stride=STRIDE,
        anchor_stride=ANCHOR,
        cache_dir="/tmp/umi_cache_s6a6",
    )
    print(f"[a6] episodes={ds.table.num_episodes} windows={len(ds)} ({time.time()-t0:.1f}s)")

    span = np.maximum(ds.table.length - HORIZON * STRIDE, 0)
    expected = int(((span + ANCHOR - 1) // ANCHOR).sum())
    assert len(ds) == expected, f"window total {len(ds)} != expected {expected}"
    print(f"[a6] total == sum(ceil(span/{ANCHOR})) OK; span-frame total {int(span.sum())} -> {expected} windows")

    indices = [0, len(ds) // 3, len(ds) // 2, len(ds) - 1]
    windows = {}
    for idx in indices:
        w = ds.read_window(idx)
        action, state = ds.make_lowdim(w)
        assert action.shape == (HORIZON, 8 * len(ds.arms)), f"action shape {action.shape}"
        assert state.shape == (1, 8 * len(ds.arms)), f"state shape {state.shape}"
        assert np.isfinite(action).all() and np.isfinite(state).all(), f"non-finite at window {idx}"
        assert w["anchor"] % ANCHOR == 0, f"anchor {w['anchor']} not a multiple of {ANCHOR}"
        last_frame = w["anchor"] + HORIZON * STRIDE
        assert last_frame < int(ds.table.length[w["position"]]), f"window {idx} exceeds episode end"
        windows[idx] = (w, action, state)
        print(f"[a6] window {idx}: pos={w['position']} anchor={w['anchor']} lang={w['lang'][:40]!r} OK")

    mid = len(ds) // 2
    w_a = ds.read_window(mid)
    w_b = ds.read_window(mid + 1)
    if w_a["position"] == w_b["position"]:
        assert w_b["anchor"] - w_a["anchor"] == ANCHOR, "consecutive anchors not anchor_stride apart"
        print(f"[a6] consecutive anchors inside episode differ by {ANCHOR} OK")

    w0 = ds.read_window(0)
    assert w0["anchor"] == 0 and w0["position"] == 0, "first window must anchor frame 0 of episode 0"
    print("[a6] first window anchors frame 0 OK")

    ds1 = UMILerobotDataset(
        data_root_dir=ROOT,
        action_horizon=HORIZON * STRIDE,
        load_video=False,
        frame_stride=1,
        cache_dir="/tmp/umi_cache_s1",
    )
    print(f"[stride1] windows={len(ds1)} horizon={HORIZON * STRIDE}")

    for idx, (w6, action6, state6) in windows.items():
        pos, anchor = w6["position"], w6["anchor"]
        w1 = ds1.read_window(_locate(ds1, pos, anchor))
        _, action1, state1 = _lowdim(ds1, w1)
        for k in range(HORIZON):
            a6 = action6[k]              # t -> t + 6*(k+1)
            a1 = action1[STRIDE * (k + 1) - 1]  # t -> t + 6*(k+1) via stride-1
            if not np.allclose(a6, a1, rtol=0, atol=1e-6):
                raise AssertionError(
                    f"window {idx} k={k}: stride6 action differs from stride1 "
                    f"(max diff {np.abs(a6 - a1).max():.3e})"
                )
        assert np.allclose(state6, state1, rtol=0, atol=1e-6), f"window {idx}: state mismatch"
        print(f"[cross] window {idx}: stride6/anchor6 actions == stride1 actions at 6x lags OK")

    print("ALL_CHECKS_PASSED")


def _locate(ds, pos, anchor):
    base = int(ds.table._cumulative[pos - 1]) if pos else 0
    return base + anchor


def _lowdim(ds, w):
    action, state = ds.make_lowdim(w)
    return w, action, state


if __name__ == "__main__":
    main()
