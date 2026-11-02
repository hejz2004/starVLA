#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from audit_results import audit_run


SUITE_SIZES = {
    "libero_10": 2519,
    "libero_goal": 2591,
    "libero_object": 2518,
    "libero_spatial": 2402,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--workers-per-card", type=int, default=8)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ranges[row["suite"]].append((int(row["start_idx"]), int(row["end_idx"])))
    return ranges


def validate_ranges(ranges: dict[str, list[tuple[int, int]]], allow_partial: bool) -> dict[str, int]:
    expected: dict[str, int] = {}
    for suite, suite_ranges in ranges.items():
        ordered = sorted(suite_ranges)
        cursor = ordered[0][0]
        if not allow_partial and cursor != 0:
            raise RuntimeError(f"{suite}: coverage starts at {cursor}, expected 0")
        covered = 0
        for start, end in ordered:
            if start != cursor:
                raise RuntimeError(f"{suite}: gap or overlap around {cursor}; next range is [{start}, {end})")
            if end <= start:
                raise RuntimeError(f"{suite}: invalid range [{start}, {end})")
            covered += end - start
            cursor = end
        if not allow_partial and cursor != SUITE_SIZES[suite]:
            raise RuntimeError(f"{suite}: coverage ends at {cursor}, expected {SUITE_SIZES[suite]}")
        expected[suite] = covered
    return expected


def main() -> None:
    args = parse_args()
    audit = audit_run(args.run_root, args.manifest, args.classification,
                      args.allow_partial, args.workers_per_card)
    ranges = load_manifest(args.manifest)
    expected = validate_ranges(ranges, args.allow_partial)

    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "successes": 0})
    category_totals: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"episodes": 0, "successes": 0})
    )
    result_files = sorted(args.run_root.glob("**/worker_*/logs/*.json"))
    if not result_files:
        raise RuntimeError(f"No worker JSON results found under {args.run_root}")

    for path in result_files:
        suite = path.stem
        if suite not in SUITE_SIZES:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for category, counts in data.items():
            episodes = int(counts.get("total_count", 0))
            successes = int(counts.get("success_count", 0))
            if successes > episodes:
                raise RuntimeError(f"{path}: successes exceed episodes for {category}")
            totals[suite]["episodes"] += episodes
            totals[suite]["successes"] += successes
            category_totals[suite][category]["episodes"] += episodes
            category_totals[suite][category]["successes"] += successes

    print("suite\tsuccesses\tepisodes\tsuccess_rate")
    overall_successes = 0
    overall_episodes = 0
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        episodes = totals[suite]["episodes"]
        successes = totals[suite]["successes"]
        expected_episodes = expected.get(suite, 0)
        if episodes != expected_episodes:
            raise RuntimeError(f"{suite}: collected {episodes} episodes, manifest expects {expected_episodes}")
        rate = successes / episodes if episodes else 0.0
        print(f"{suite}\t{successes}\t{episodes}\t{rate:.6f}")
        overall_successes += successes
        overall_episodes += episodes

    overall_rate = overall_successes / overall_episodes if overall_episodes else 0.0
    print(f"overall\t{overall_successes}\t{overall_episodes}\t{overall_rate:.6f}")

    summary = {
        "result_file_count": len(result_files),
        "suites": dict(totals),
        "categories": {suite: dict(values) for suite, values in category_totals.items()},
        "overall": {
            "episodes": overall_episodes,
            "successes": overall_successes,
            "success_rate": overall_rate,
        },
    }
    summary["artifact_audit"] = audit
    summary["category_totals"] = audit["categories"]
    output_path = args.run_root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    (args.run_root / "COMPLETION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "name", "successes", "episodes", "success_rate"])
        for suite, counts in summary["suites"].items():
            writer.writerow(["suite", suite, counts["successes"], counts["episodes"],
                             counts["successes"] / counts["episodes"] if counts["episodes"] else 0])
        for category, counts in audit["categories"].items():
            writer.writerow(["category", category, counts["successes"], counts["episodes"],
                             counts["successes"] / counts["episodes"]])
        writer.writerow(["overall", "overall", overall_successes, overall_episodes, overall_rate])


if __name__ == "__main__":
    main()

