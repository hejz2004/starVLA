"""Offline preparation of a referenced LeRobot v3 DROID catalog and train statistics.

Run: python -m starVLA.dataloader.droid_prepare --source-root ... --output ...
Output is immutable and published with READY only after all statistics are written.
No data Parquet scanning, Euler conversion or statistics fitting occurs in training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pyarrow.parquet as pq

from .droid_contract import CAMERAS, anchor_ranges, cache_poses, contract, relative_actions
from .umi_normalization import WeightedMoments, content_fingerprint


def file_record(path, root, checksum=False):
    stat = path.stat()
    record = dict(path=str(path.relative_to(root)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if checksum:
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def fixed_array(table, name):
    array = table[name].combine_chunks()
    # The conversion stores 1D gripper signals as scalars, while 6D poses are lists.
    return np.asarray(array.values if hasattr(array, "values") else array).reshape(len(array), -1)


def split_for_episode(episode, seed, val_fraction):
    digest = hashlib.sha256(f"droid:{seed}:{episode}".encode()).digest()
    return "val" if int.from_bytes(digest[:8], "big") / 2**64 < val_fraction else "train"


def prepare(source_root, output, *, horizon=16, stride=3, seed=42, val_fraction=.01, max_episodes=None):
    source, output = Path(source_root).resolve(), Path(output).resolve()
    if horizon < 1 or stride < 1 or not 0 <= val_fraction < 1:
        raise ValueError("Invalid horizon/stride/validation fraction")
    # An interrupted generation stays inspectable; never mix or overwrite generations.
    output.mkdir(parents=True, exist_ok=False)
    (output / "lowdim").mkdir()
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 15:
        raise ValueError("Expected DROID LeRobot v3 at 15 Hz")
    columns = ["episode_index", "length", "dataset_from_index", "dataset_to_index",
               "data/chunk_index", "data/file_index"]
    for camera in CAMERAS:
        columns += [f"videos/{camera}/{field}" for field in
                    ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    meta_files = sorted((source / "meta/episodes").glob("*/*.parquet"))
    episodes = sorted((row for path in meta_files for row in pq.read_table(path, columns=columns).to_pylist()),
                      key=lambda x: x["episode_index"])
    if len({e["episode_index"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate metadata episode IDs")
    if len(episodes) != info["total_episodes"] or sum(e["length"] for e in episodes) != info["total_frames"]:
        raise ValueError("Episode manifest differs from info.json")
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    grouped = defaultdict(list)
    for e in episodes:
        path = info["data_path"].format(chunk_index=e["data/chunk_index"], file_index=e["data/file_index"])
        grouped[path].append(e)
    sources = [file_record(source / "meta/info.json", source, True)]
    sources += [file_record(p, source, True) for p in meta_files]
    sources += [file_record(source / p, source) for p in sorted(grouped)]
    identity = dict(schema="droid-index-v1", source_root=str(source), contract=contract(horizon, stride),
                    seed=seed, val_fraction=val_fraction, max_episodes=max_episodes, sources=sources,
                    split_policy="sha256(droid:seed:episode_id) first uint64 < val_fraction",
                    language_policy="first nonempty of three annotations on anchor row",
                    quality_policy="successful episodes; finite poses; grippers in [0,1]; contiguous frames/times; complete windows",
                    idle_filter=False)
    db = sqlite3.connect(output / "catalog.sqlite")
    db.execute("CREATE TABLE episodes (episode INTEGER PRIMARY KEY, split TEXT, length INTEGER, shard TEXT, offset INTEGER, cameras TEXT)")
    db.execute("CREATE TABLE ranges (split TEXT, episode INTEGER, first_anchor INTEGER, count INTEGER, language TEXT)")
    moments = WeightedMoments(8)
    counts = Counter()
    video_records = {}
    for file_number, (relative, owned) in enumerate(sorted(grouped.items())):
        started = time.monotonic()
        table = pq.read_table(source / relative, columns=[
            "index", "episode_index", "frame_index", "timestamp", "is_episode_successful",
            "observation.state.cartesian_position", "observation.state.gripper_position",
            "action.cartesian_position", "action.gripper_position",
            "language_instruction", "language_instruction_2", "language_instruction_3"])
        global_indices = table["index"].to_numpy()
        if np.any(np.diff(global_indices) != 1):
            raise ValueError(f"Non-contiguous or unsorted source file {relative}")
        ids, frames, times = (table[k].to_numpy() for k in ("episode_index", "frame_index", "timestamp"))
        successful = table["is_episode_successful"].to_numpy(zero_copy_only=False)
        obs, cmd = (fixed_array(table, k) for k in
                    ("observation.state.cartesian_position", "action.cartesian_position"))
        og, cg = (fixed_array(table, k).ravel() for k in
                  ("observation.state.gripper_position", "action.gripper_position"))
        languages = [table[k].to_pylist() for k in
                     ("language_instruction", "language_instruction_2", "language_instruction_3")]
        shard = f"lowdim/{file_number:05d}"
        low = np.lib.format.open_memmap(output / (shard + ".npy"), mode="w+", dtype="float32", shape=(len(table), 16))
        stamp = np.lib.format.open_memmap(output / (shard + "-time.npy"), mode="w+", dtype="float64", shape=(len(table),))
        stamp[:] = times
        for e in owned:
            ep, length = int(e["episode_index"]), int(e["length"])
            counts["examined_episodes"] += 1
            begin = int(np.searchsorted(global_indices, e["dataset_from_index"]))
            end = begin + length
            if (length != e["dataset_to_index"] - e["dataset_from_index"] or end > len(table)
                    or not np.all(ids[begin:end] == ep)
                    or global_indices[begin] != e["dataset_from_index"]):
                raise ValueError(f"Episode {ep} spans an unsupported file boundary or metadata mismatch")
            if not successful[begin:end].all():
                counts["rejected_unsuccessful"] += 1
                continue
            text = [next((str(lang[i]).strip() for lang in languages if lang[i] and str(lang[i]).strip()), "")
                    for i in range(begin, end)]
            ranges = anchor_ranges(length, horizon, stride, text)
            if not ranges:
                counts["rejected_no_language_or_short"] += 1
                continue
            local_times = times[begin:end]
            if (not np.isfinite(local_times).all() or not np.array_equal(frames[begin:end], np.arange(length))
                    or not np.allclose(local_times, np.arange(length) / 15, atol=2e-4, rtol=0)):
                counts["rejected_noncontiguous"] += 1
                continue
            arrays = (obs[begin:end], cmd[begin:end], og[begin:end], cg[begin:end])
            if (not all(np.isfinite(a).all() for a in arrays)
                    or any(np.any((g < 0) | (g > 1)) for g in arrays[2:])):
                counts["rejected_invalid_pose_or_gripper"] += 1
                continue
            cameras = []
            for key in CAMERAS:
                prefix = f"videos/{key}/"
                video = info["video_path"].format(video_key=key, chunk_index=e[prefix + "chunk_index"],
                                                 file_index=e[prefix + "file_index"])
                start, stop = float(e[prefix + "from_timestamp"]), float(e[prefix + "to_timestamp"])
                path = source / video
                if (not path.is_file() or not np.isfinite([start, stop]).all()
                        or stop - start <= local_times[-1] - 1e-4 or start < 0):
                    break
                if video not in video_records:
                    video_records[video] = file_record(path, source)
                cameras.append(dict(camera=key, video_path=video, from_timestamp=start, to_timestamp=stop))
            if len(cameras) != 3:
                counts["rejected_video_metadata"] += 1
                continue
            physical = cache_poses(*arrays)
            low[begin:end] = physical
            split = split_for_episode(ep, seed, val_fraction)
            db.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?)",
                       (ep, split, length, shard, begin, json.dumps(cameras)))
            for first, count, lang in ranges:
                db.execute("INSERT INTO ranges VALUES (?,?,?,?,?)", (split, ep, first, count, lang))
                counts[split + "_anchors"] += count
                if split == "train":
                    for offset in range(0, count, 2048):
                        anchors = first + np.arange(offset, min(count, offset + 2048)) * stride
                        moments.update(relative_actions(physical, anchors, horizon).reshape(-1, 8))
            counts[split + "_episodes"] += 1
            counts["retained_frames"] += length
        low.flush(); stamp.flush()
        del low, stamp, table
        db.commit()
        print(json.dumps(dict(file=file_number + 1, files=len(grouped), seconds=round(time.monotonic()-started, 2),
                              counts=dict(counts))), flush=True)
    db.execute("CREATE INDEX range_split ON ranges(split, episode, first_anchor)")
    db.commit(); db.close()
    if not moments.count:
        raise ValueError("No training targets survived preparation")
    identity["videos"] = [video_records[k] for k in sorted(video_records)]
    fingerprint = content_fingerprint(identity)
    std = np.sqrt(moments.M2 / moments.count)
    mean16 = np.r_[moments.mean, np.zeros(8)].astype(np.float32)
    scale16 = np.r_[np.where(std < 1e-6, 1., std), np.ones(8)].astype(np.float32)
    statistics = dict(schema="droid-mean-std-v1", fingerprint=fingerprint, mode="mean_std", clipping=False,
                      action=dict(mean=mean16.tolist(), std=scale16.tolist(), raw_std=std.tolist()),
                      active_dimensions=8, target_appearances=moments.count, fit_split="train",
                      weighting="each target position of each valid stride-3 window equally",
                      accumulation="FP64 Chan population moments", transform_dtype="float32", min_std=1e-6,
                      moments=moments.state_dict())
    manifest = dict(identity, fingerprint=fingerprint, counts=dict(counts),
                    catalog_sha256=hashlib.sha256((output / "catalog.sqlite").read_bytes()).hexdigest())
    for name, value in (("manifest.json", manifest), ("statistics.json", statistics)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    ready = dict(fingerprint=fingerprint, statistics_sha256=hashlib.sha256((output / "statistics.json").read_bytes()).hexdigest())
    (output / "READY.tmp").write_text(json.dumps(ready) + "\n")
    (output / "READY.tmp").replace(output / "READY")
    print(json.dumps(dict(status="ready", output=str(output), **ready, counts=dict(counts))), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.01)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    prepare(args.source_root, args.output, horizon=args.horizon, stride=args.stride, seed=args.seed,
            val_fraction=args.val_fraction, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
