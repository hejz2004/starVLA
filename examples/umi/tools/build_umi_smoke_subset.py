"""Build a small UMI LeRobot v3.0 subset for container smoke tests.

The full curated dataset lives on /mnt/nas which the starvla-pi container
cannot see. This script copies ONE data parquet file plus its matching
camera video files and the meta rows of exactly the episodes inside that
data file, so the subset is a self-consistent mini dataset (file_starts
stays exact because no episode of the copied file is dropped).

Run on the host (pyarrow + NAS access):
    python examples/umi/tools/build_umi_smoke_subset.py \
        --src_root /mnt/nas/.../umi_v30_curated_... \
        --dst_root /mnt/workspace/Native_Policy/user/hjz/umi_smoke_subset
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.umi_lerobot.episode_index import CAMERA_ORDER, _META_SCAN_COLUMNS, _meta_files

HORIZON, STRIDE, ANCHOR = 16, 6, 6


def windows_of(length: int) -> int:
    span = max(length - HORIZON * STRIDE, 0)
    return (span + ANCHOR - 1) // ANCHOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True)
    parser.add_argument("--dst_root", required=True)
    parser.add_argument("--min_windows", type=int, default=96)
    parser.add_argument("--max_video_gib", type=float, default=15.0)
    args = parser.parse_args()

    src, dst = Path(args.src_root), Path(args.dst_root)
    rows = []
    for meta_file in _meta_files(src):
        rows.extend(pq.read_table(meta_file, columns=_META_SCAN_COLUMNS).to_pylist())
    print(f"meta rows: {len(rows)}")

    groups = defaultdict(list)
    for r in rows:
        groups[(int(r["data/chunk_index"]), int(r["data/file_index"]))].append(r)

    candidates = []
    for key, group in sorted(groups.items()):
        windows = sum(windows_of(int(r["length"])) for r in group)
        if windows < args.min_windows:
            continue
        video_keys = {
            (cam, int(r[f"videos/observation.images.{cam}/chunk_index"]), int(r[f"videos/observation.images.{cam}/file_index"]))
            for r in group
            for cam in CAMERA_ORDER
        }
        video_paths = [src / "videos" / f"observation.images.{cam}" / f"chunk-{c:03d}" / f"file-{f:03d}.mp4" for cam, c, f in video_keys]
        if not all(p.exists() for p in video_paths):
            continue
        video_bytes = sum(p.stat().st_size for p in video_paths)
        if video_bytes > args.max_video_gib * 1024**3:
            continue
        data_path = src / "data" / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.parquet"
        candidates.append((video_bytes, windows, key, group, data_path, video_paths))
        print(f"candidate data chunk-{key[0]:03d}/file-{key[1]:03d}: {len(group)} eps, {windows} windows, video {video_bytes/1024**3:.2f} GiB")

    if not candidates:
        raise SystemExit("no group satisfies min_windows / max_video_gib")
    candidates.sort(key=lambda c: (c[0], -c[1]))
    video_bytes, windows, key, group, data_path, video_paths = candidates[0]
    print(f"selected chunk-{key[0]:03d}/file-{key[1]:03d}: {len(group)} eps, {windows} windows, video {video_bytes/1024**3:.2f} GiB")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "meta" / "info.json", dst / "meta" / "info.json")

    (dst / "data" / f"chunk-{key[0]:03d}").mkdir(parents=True, exist_ok=True)
    print(f"copy {data_path} ({data_path.stat().st_size/1024**2:.1f} MiB)")
    shutil.copy2(data_path, dst / "data" / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.parquet")
    for p in video_paths:
        p_dst = dst / p.relative_to(src)
        p_dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"copy {p.relative_to(src)} ({p.stat().st_size/1024**3:.2f} GiB)")
        shutil.copy2(p, p_dst)

    chosen = {int(r["episode_index"]) for r in group}
    tables = [pq.read_table(f) for f in _meta_files(src)]
    full = tables[0]
    from pyarrow import concat_tables

    full = concat_tables(tables) if len(tables) > 1 else full
    mask = np.isin(full["episode_index"].to_numpy(), list(chosen))
    subset = full.filter(mask)
    meta_dir = dst / "meta" / "episodes" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, meta_dir / "file-000.parquet")
    print(f"meta subset: {subset.num_rows} episodes written")

    from starVLA.dataloader.umi_lerobot.umi_dataset import UMILerobotDataset

    ds = UMILerobotDataset(dst, action_horizon=HORIZON, load_video=False, frame_stride=STRIDE, anchor_stride=ANCHOR)
    print(f"verify: episodes={ds.table.num_episodes} windows={len(ds)}")
    for idx in (0, len(ds) // 2, len(ds) - 1):
        sample = ds.read_lowdim(idx)
        assert sample["action"].shape == (HORIZON, 16) and np.isfinite(sample["action"]).all()
    print("SUBSET_OK")


if __name__ == "__main__":
    main()
