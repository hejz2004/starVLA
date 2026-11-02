#!/usr/bin/env python3
"""Independent artifact and coverage acceptance; no simulator or model imports."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from artifact_checks import validate_action_array

SUITE_SIZES = {
    "libero_10": 2519, "libero_goal": 2591,
    "libero_object": 2518, "libero_spatial": 2402,
}
CATEGORY_SIZES = {
    "Camera Viewpoints": 1599, "Robot Initial States": 1550,
    "Language Instructions": 1537, "Light Conditions": 1142,
    "Background Textures": 1076, "Sensor Noise": 1601, "Objects Layout": 1525,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_classification(path):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = {}
    category_counts = Counter()
    for suite, size in SUITE_SIZES.items():
        items = source[suite]
        require(len(items) == size, f"Classification {suite} has incorrect size")
        indexed = {item["id"] - 1: item for item in items}
        require(len(indexed) == len(items) and set(indexed) == set(range(size)),
                f"Classification {suite} IDs are not unique and contiguous")
        require(len({item["name"] for item in items}) == size,
                f"Classification {suite} variant names are not unique")
        mapping[suite] = indexed
        category_counts.update(item["category"] for item in items)
    require(dict(category_counts) == CATEGORY_SIZES, "Official seven-category inventory differs")
    return mapping


def artifact_path(worker, relative, expected_subdir, suffix):
    require(isinstance(relative, str), "Artifact path is not a string")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            f"Unsafe relative artifact path: {relative}")
    require(path.parts[0] == expected_subdir and path.suffix == suffix,
            f"Unexpected artifact location: {relative}")
    resolved = (worker / path).resolve()
    require(resolved.is_relative_to(worker.resolve()), f"Artifact leaves worker directory: {relative}")
    require(resolved.is_file() and resolved.stat().st_size > 0, f"Missing/empty artifact: {resolved}")
    return resolved


def audit_run(run_root, manifest, classification, allow_partial=False, workers_per_card=8):
    run_root = Path(run_root).resolve()
    require(workers_per_card >= 1, "workers_per_card must be positive")
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows, "Manifest is empty")
    mapping = load_classification(classification)
    require(len({int(r["gpu_id"]) for r in rows}) == len(rows), "Duplicate manifest GPU")
    require(len({int(r["port"]) for r in rows}) == len(rows), "Duplicate manifest port")
    covered = defaultdict(list)
    for row in rows:
        suite, start, end = row["suite"], int(row["start_idx"]), int(row["end_idx"])
        require(suite in SUITE_SIZES, f"Unknown suite {suite}")
        require(0 <= start < end <= SUITE_SIZES[suite], f"Invalid manifest range for {suite}")
        covered[suite].extend(range(start, end))
    for suite, ids in covered.items():
        require(len(set(ids)) == len(ids), f"Overlapping task coverage in {suite}")
        ordered = sorted(ids)
        require(ordered == list(range(ordered[0], ordered[-1] + 1)), f"Coverage gap in {suite}")
    if not allow_partial:
        require(workers_per_card == 8 and len(rows) == 16, "Full protocol requires 16 cards x 8 workers")
        require(set(covered) == set(SUITE_SIZES), "Full protocol is missing suites")
        for suite, size in SUITE_SIZES.items():
            require(sorted(covered[suite]) == list(range(size)), f"Full task coverage differs in {suite}")
        expected_gpu_suite = {**{i: "libero_10" for i in range(7)},
                              **{i: "libero_goal" for i in range(7, 10)},
                              **{i: "libero_object" for i in range(10, 13)},
                              **{i: "libero_spatial" for i in range(13, 16)}}
        require({int(r["gpu_id"]): r["suite"] for r in rows} == expected_gpu_suite,
                "Full GPU-to-suite partition differs from the guide")
    seen_trials, expected_worker_dirs, expected_card_dirs = set(), set(), set()
    suite_counts = defaultdict(lambda: Counter(episodes=0, successes=0))
    category_counts = defaultdict(lambda: Counter(episodes=0, successes=0))
    suite_category_counts = defaultdict(lambda: defaultdict(lambda: Counter(episodes=0, successes=0)))
    video_count = action_count = result_count = 0
    for row in rows:
        suite = row["suite"]
        start, end, gpu, port = (int(row[k]) for k in ("start_idx", "end_idx", "gpu_id", "port"))
        card = run_root / suite / f"card_{gpu}_{start}_{end}"
        expected_card_dirs.add(card)
        policy_evidence = card / "policy_interface.json"
        require(policy_evidence.is_file(), f"Missing matching policy interface evidence: {card}")
        require(isinstance(json.loads(policy_evidence.read_text(encoding="utf-8")), dict),
                f"Invalid policy interface evidence: {card}")
        count = min(workers_per_card, end - start)
        done = card / "card.done"
        require(done.is_file(), f"Missing card.done: {card}")
        expected_done = f"suite={suite} range=[{start},{end}) gpu={gpu} port={port} workers={count}"
        require(done.read_text().strip() == expected_done, f"Card completion differs: {card}")
        current = start
        base, remainder = divmod(end - start, count)
        for worker_id in range(count):
            worker_end = current + base + (worker_id < remainder)
            worker = card / f"worker_{worker_id:02d}"
            expected_worker_dirs.add(worker)
            require((worker / "status").is_file() and (worker / "status").read_text().strip() == "success",
                    f"Worker process did not exit successfully: {worker}")
            require((worker / "worker.done").is_file()
                    and (worker / "worker.done").read_text().strip() == f"[{current}, {worker_end})",
                    f"Worker completion range differs: {worker}")
            trials_path = worker / "logs" / "trials.jsonl"
            require(trials_path.is_file(), f"Missing trial records: {worker}")
            records = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            require([r.get("task_id") for r in records] == list(range(current, worker_end)),
                    f"Missing/duplicate/out-of-order trial in {worker}")
            worker_categories = defaultdict(lambda: Counter(episodes=0, successes=0))
            worker_videos, worker_actions = set(), set()
            for record in records:
                task_id = record["task_id"]
                expected_variant = mapping[suite][task_id]
                require(record.get("suite") == suite and record.get("variant_id") == task_id + 1,
                        f"Wrong suite/variant ID in {worker}")
                require(record.get("variant") == expected_variant["name"]
                        and record.get("category") == expected_variant["category"],
                        f"Wrong variant/category in {worker}, task {task_id}")
                require(record.get("episode_idx") == record.get("initial_state_index") == 0,
                        "Each Plus variant must use initial state 0 exactly once")
                require(record.get("seed") == 7, "Worker/simulation seed differs from 7")
                require(type(record.get("success")) is bool, "Trial success must be a Boolean")
                identity = (suite, task_id, 0)
                require(identity not in seen_trials, f"Duplicate trial identity: {identity}")
                seen_trials.add(identity)
                video = artifact_path(worker, record.get("video_path"), "videos", ".mp4")
                action = artifact_path(worker, record.get("action_path"), "actions", ".npy")
                require(video not in worker_videos and action not in worker_actions, "Artifact filename collision")
                worker_videos.add(video)
                worker_actions.add(action)
                array = validate_action_array(np.load(action, allow_pickle=False))
                require(type(record.get("frames")) is int and type(record.get("step_count")) is int
                        and record["frames"] == record["step_count"] == len(array),
                        f"Video/action step counts differ: {identity}")
                require(1 <= len(array) <= {"libero_spatial": 220, "libero_object": 280,
                                           "libero_goal": 300, "libero_10": 520}[suite],
                        f"Action trajectory exceeds suite step limit: {identity}")
                elapsed = record.get("elapsed_seconds")
                require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0,
                        f"Invalid episode duration: {identity}")
                category, success = record["category"], int(record["success"])
                for target in (suite_counts[suite], category_counts[category],
                               suite_category_counts[suite][category], worker_categories[category]):
                    target.update(episodes=1, successes=success)
            require(set(p.resolve() for p in (worker / "videos").glob("*.mp4")) == worker_videos,
                    f"Extra/missing video artifacts: {worker}")
            require(set(p.resolve() for p in (worker / "actions").glob("*.npy")) == worker_actions,
                    f"Extra/missing action artifacts: {worker}")
            result = worker / "logs" / f"{suite}.json"
            require(result.is_file(), f"Missing worker category results: {worker}")
            counts = json.loads(result.read_text(encoding="utf-8"))
            require(set(counts) <= set(CATEGORY_SIZES), f"Unknown result category: {worker}")
            for category in set(counts) | set(worker_categories):
                require(category in counts, f"Missing result category: {category}")
                require(counts[category] == {"total_count": worker_categories[category]["episodes"],
                                             "success_count": worker_categories[category]["successes"]},
                        f"JSON aggregate disagrees with trial records: {worker}, {category}")
            result_count += 1
            video_count += len(worker_videos)
            action_count += len(worker_actions)
            current = worker_end
    require(set(run_root.glob("libero_*/card_*")) == expected_card_dirs, "Extra/missing card directories")
    require(set(run_root.glob("libero_*/card_*/worker_*")) == expected_worker_dirs,
            "Extra/missing worker directories")
    actual_json = set(run_root.glob("libero_*/card_*/worker_*/logs/*.json"))
    require(actual_json == {p / "logs" / f"{p.parent.parent.name}.json" for p in expected_worker_dirs},
            "Unexpected/missing worker result JSON files")
    if not allow_partial:
        require(result_count == 128 and len(seen_trials) == 10030, "Full worker/trial total differs")
        require({k: v["episodes"] for k, v in category_counts.items()} == CATEGORY_SIZES,
                "Full seven-category coverage differs")
    return {
        "status": "passed", "full_protocol": not allow_partial,
        "card_count": len(rows), "worker_count": result_count,
        "episodes": len(seen_trials), "nonempty_videos": video_count,
        "finite_action_trajectories": action_count,
        "suites": {k: dict(v) for k, v in suite_counts.items()},
        "categories": {k: dict(v) for k, v in category_counts.items()},
        "suite_categories": {k: {c: dict(v) for c, v in counts.items()}
                             for k, counts in suite_category_counts.items()},
        "policy_seed": None, "worker_numpy_seed": 7, "simulation_seed": 7,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--workers-per-card", type=int, default=8)
    args = parser.parse_args()
    result = audit_run(args.run_root, args.manifest, args.classification,
                       args.allow_partial, args.workers_per_card)
    (args.run_root / "COMPLETION_AUDIT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("suites", "categories", "suite_categories")}))


if __name__ == "__main__":
    main()
