"""Cache original LIBERO commands, initial simulator states and model XMLs.

    python extract_official_libero_lowdim.py --report-root /path/to/report --workers 8

The report root must contain libero-original-info.json and
libero-original-tree.json downloaded from the official dataset publisher.
Only HDF5 metadata and low-dimensional arrays are read; RGB payloads are skipped.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from libero_hdf5_source import CachedHTTPRangeReader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(record, revision, root):
    root = Path(root)
    relative = record["path"]
    output = root / "official_lowdim" / Path(relative).with_suffix(".npz")
    metadata = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and metadata.exists():
        result = json.loads(metadata.read_text())
        if result["revision"] != revision or result["sha256"] != sha(output):
            raise ValueError(f"Different or corrupt low-dimensional cache: {relative}")
        return result
    url = f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{revision}/{relative}"
    started = time.monotonic()
    arrays = {}
    with CachedHTTPRangeReader(url, record["size"], root / "official_source_ranges") as reader:
        with h5py.File(reader, "r") as source:
            group = source["data"]
            demos = sorted(group, key=lambda name: int(name.split("_")[-1]))
            for name in demos:
                demo = group[name]
                arrays[f"{name}__actions"] = demo["actions"][:]
                arrays[f"{name}__ee_states"] = demo["obs/ee_states"][:]
                arrays[f"{name}__gripper_states"] = demo["obs/gripper_states"][:]
                arrays[f"{name}__initial_state"] = demo["states"][0]
                model = demo.attrs["model_file"]
                arrays[f"{name}__model_xml"] = np.asarray(model.decode() if isinstance(model, bytes) else model)
                if "init_state" in demo.attrs:
                    arrays[f"{name}__attribute_init_state"] = demo.attrs["init_state"]
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
        result = dict(
            path=relative, revision=revision, source_size=record["size"],
            declared_lfs_sha256=record.get("lfs", {}).get("oid"),
            lowdim_path=str(output), sha256=sha(output), demos=len(demos),
            frames=sum(len(arrays[f"{name}__actions"]) for name in demos),
            network_bytes=reader.network_bytes, network_requests=reader.network_requests,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        metadata.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.report_root
    revision = json.loads((root / "libero-original-info.json").read_text())["sha"]
    tree = json.loads((root / "libero-original-tree.json").read_text())
    suites = {"libero_object", "libero_goal", "libero_spatial", "libero_10"}
    records = [r for r in tree if r["type"] == "file" and r["path"].endswith(".hdf5")
               and r["path"].split("/")[0] in suites]
    if len(records) != 40:
        raise ValueError(f"Expected 40 official task files, got {len(records)}")
    print(json.dumps(dict(status="started", tasks=len(records), revision=revision, workers=args.workers)), flush=True)
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(extract, record, revision, str(root)): record["path"] for record in records}
        for future in as_completed(jobs):
            try:
                result = future.result()
            except Exception as exc:
                message = str(exc)
                if "https://" in message:
                    message = type(exc).__name__
                print(json.dumps(dict(status="error", path=jobs[future], error=message)), flush=True)
                raise
            completed.append(result)
            print(json.dumps(dict(status="extracted", completed=len(completed), **result)), flush=True)
    manifest = dict(status="complete", source_repository="yifengzhu-hf/LIBERO-datasets", revision=revision,
                    records=sorted(completed, key=lambda r: r["path"]))
    (root / "official_lowdim_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="complete", tasks=len(completed), network_bytes=sum(r["network_bytes"] for r in completed))), flush=True)


if __name__ == "__main__":
    main()
