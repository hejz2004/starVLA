"""Extract reproducible synchronized RGB pairs from local LeRobot LIBERO videos.

Requires numpy, Pillow, pyarrow, and av. Official IDs are resolved through the
LIBERO task map, then matched by full language against LeRobot metadata.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import platform
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import PIL
import pyarrow
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


TASKS = [
    ("libero_spatial", 0, "物体空间关系、背景干扰"),
    ("libero_spatial", 4, "遮挡、狭窄区域"),
    ("libero_object", 0, "近距离物体局部匹配"),
    ("libero_object", 9, "不同物体形状和纹理"),
    ("libero_goal", 0, "局部把手、接触区域"),
    ("libero_goal", 7, "小目标与精细操作"),
    ("libero_10", 4, "相似物体、多步操作"),
    ("libero_10", 9, "强遮挡、视野变化、多阶段操作"),
]
CAMERAS = {
    "agentview": "observation.images.image",
    "wrist": "observation.images.wrist_image",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_task_map(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"No literal libero_task_map in {path}")


def decode_frames(path, indices, length, timestamps, fps):
    selected = {}
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        require(float(stream.average_rate) == fps, f"Unexpected FPS: {path}")
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            require(frame.time is not None, f"Missing PTS: {path}, frame {index}")
            require(abs(frame.time - timestamps[index]) < 1e-5 if index < length else False,
                    f"Video/parquet timestamp mismatch: {path}, frame {index}")
            if index in indices:
                require((frame.width, frame.height) == (256, 256), f"Unexpected size: {path}")
                selected[index] = (frame.to_image().convert("RGB"), float(frame.time))
    require(count == length, f"Video/parquet length mismatch: {path}: {count} != {length}")
    require(set(selected) == set(indices), f"Missing selected frames: {path}")
    return selected


def make_preview(output, pairs, task_records):
    (output / "previews").mkdir()
    sections = []
    for task in task_records:
        rows = [r for r in pairs if r["suite"] == task["suite"] and r["task_id"] == task["task_id"]]
        # Five demonstrations, five synchronized pairs per row. Previews alone
        # are resized; every dataset PNG retains its source resolution.
        sheet = Image.new("RGB", (1280, 5 * 160 + 30), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{task['suite']} / {task['task_id']} | each cell: agentview LEFT, wrist RIGHT", fill="black")
        demo_html = []
        for demo_number, demo in enumerate(task["selected_episode_indices"]):
            demo_rows = [r for r in rows if r["episode_index"] == demo]
            cards = []
            for slot, pair in enumerate(demo_rows):
                for camera_index, camera in enumerate(CAMERAS):
                    with Image.open(output / pair["images"][camera]) as im:
                        sheet.paste(im.resize((128, 128)), (slot * 256 + camera_index * 128, 30 + demo_number * 160))
                draw.text((slot * 256 + 4, 160 + demo_number * 160),
                          f"ep={demo} f={pair['frame_index']} t={pair['timestamp']:.2f}s", fill="black")
                cards.append('<figure>' + ''.join(
                    f'<img loading="lazy" src="{html.escape(pair["images"][c])}" alt="{c}">'
                    for c in CAMERAS) + f'<figcaption>frame {pair["frame_index"]} / {pair["episode_length"] - 1}'
                    f' · {pair["timestamp"]:.2f}s</figcaption></figure>')
            demo_html.append(f'<h3>episode {demo}</h3><div class="frames">' + ''.join(cards) + '</div>')
        preview = f"previews/{task['suite']}_task_{task['task_id']:02d}.jpg"
        sheet.save(output / preview, quality=92)
        task["preview"] = preview
        sections.append(f'<details open><summary>{task["suite"]} / {task["task_id"]}: '
                        f'{html.escape(task["language"])}</summary><p>{task["phenomena"]}</p>'
                        + ''.join(demo_html) + '</details>')
    (output / "index.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>LIBERO multiview pilot</title><style>
body{font:15px system-ui;margin:24px;background:#f5f6f8;color:#18212c}
details{background:white;margin:18px 0;padding:18px;border-radius:10px}
summary{font-weight:600;cursor:pointer}.frames{display:flex;overflow:auto;gap:12px}
figure{margin:0;flex:0 0 512px}img{width:256px;height:256px}figcaption{padding:6px}
</style><h1>LIBERO multiview pilot · 200 pairs</h1>
<p>8 tasks × 5 demonstrations × 5 synchronized frames. Left: agentview; right: wrist.
Seed 42 by default; see manifest.json for this build. Source: LeRobot no_noops, 256×256 RGB.
Uniform sampling includes both endpoints. Target phenomena are task-level intentions, not frame-level annotations.</p>
<a href="pairs.jsonl">pairs.jsonl</a> · <a href="manifest.json">manifest.json</a>
''' + ''.join(sections) + '</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    task_map = read_task_map(args.task_map)
    rng = random.Random(args.seed)
    selections = []
    # Validate task identities and all selected source paths before writing.
    for suite, task_id, phenomena in TASKS:
        task_name = task_map[suite][task_id]
        language = re.sub(r"^[A-Z_]+SCENE\d+_", "", task_name).replace("_", " ")
        suite_root = source / f"{suite}_no_noops_1.0.0_lerobot"
        info = json.loads((suite_root / "meta/info.json").read_text())
        matches = [t for t in read_jsonl(suite_root / "meta/tasks.jsonl") if t["task"] == language]
        require(len(matches) == 1, f"Ambiguous or missing task: {suite}/{task_id}: {language}")
        candidates = sorted([e for e in read_jsonl(suite_root / "meta/episodes.jsonl")
                             if e["tasks"] == [language]], key=lambda e: e["episode_index"])
        require(len(candidates) >= 5, f"Fewer than five demonstrations: {suite}/{task_id}")
        chosen = sorted(rng.sample(candidates, 5), key=lambda e: e["episode_index"])
        task = dict(suite=suite, task_id=task_id, task_name=task_name, language=language,
                    phenomena=phenomena, source_task_index=matches[0]["task_index"],
                    candidate_episode_indices=[e["episode_index"] for e in candidates],
                    selected_episode_indices=[e["episode_index"] for e in chosen],
                    candidate_count=len(candidates), pairs=25)
        for episode in chosen:
            require(episode["length"] >= 5, f"Short demonstration: {episode}")
            fmt = dict(episode_index=episode["episode_index"],
                       episode_chunk=episode["episode_index"] // info["chunks_size"])
            for relative in [info["data_path"].format(**fmt)] + [
                info["video_path"].format(**fmt, video_key=v) for v in CAMERAS.values()
            ]:
                require((suite_root / relative).is_file(), f"Missing source: {suite_root / relative}")
        selections.append((suite_root, info, task, chosen))
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / "build_multiview_pilot.py")
    shutil.copy2(args.task_map, output / "libero_suite_task_map.py")
    write_json(output / "selection.json", dict(seed=args.seed, tasks=[s[2] for s in selections]))
    pairs, demos, source_files = [], [], {}
    for suite_root, info, task, chosen in selections:
        for metadata in ("info.json", "tasks.jsonl", "episodes.jsonl"):
            path = suite_root / "meta" / metadata
            source_files[str(path)] = sha256(path)
        for episode in chosen:
            episode_id, length = episode["episode_index"], episode["length"]
            fmt = dict(episode_index=episode_id, episode_chunk=episode_id // info["chunks_size"])
            parquet = suite_root / info["data_path"].format(**fmt)
            table = pq.read_table(parquet, columns=["frame_index", "timestamp", "episode_index", "task_index", "index"]).to_pydict()
            require(table["frame_index"] == list(range(length)), f"Invalid frame sequence: {parquet}")
            require(set(table["episode_index"]) == {episode_id}, f"Episode mismatch: {parquet}")
            require(set(table["task_index"]) == {task["source_task_index"]}, f"Task mismatch: {parquet}")
            source_files[str(parquet)] = sha256(parquet)
            indices = [i * (length - 1) // 4 for i in range(5)]
            decoded, videos = {}, {}
            for camera, key in CAMERAS.items():
                path = suite_root / info["video_path"].format(**fmt, video_key=key)
                source_files[str(path)] = sha256(path)
                decoded[camera] = decode_frames(path, indices, length, table["timestamp"], info["fps"])
                videos[camera] = str(path)
            demo_id = f"{task['suite']}_task_{task['task_id']:02d}_episode_{episode_id:06d}"
            demos.append(dict(demo_id=demo_id, suite=task["suite"], task_id=task["task_id"],
                              episode_index=episode_id, episode_length=length, frame_indices=indices,
                              source_parquet=str(parquet), source_videos=videos))
            for slot, frame_index in enumerate(indices):
                pair_id = f"{demo_id}_frame_{frame_index:06d}"
                folder = Path("images") / task["suite"] / f"task_{task['task_id']:02d}" / f"episode_{episode_id:06d}"
                (output / folder).mkdir(parents=True, exist_ok=True)
                paths, checksums, pts = {}, {}, {}
                for camera in CAMERAS:
                    im, timestamp = decoded[camera][frame_index]
                    relative = folder / f"frame_{frame_index:06d}_{camera}.png"
                    im.save(output / relative)
                    with Image.open(output / relative) as saved:
                        require(np.array_equal(np.asarray(saved), np.asarray(im)), f"PNG mismatch: {relative}")
                    paths[camera] = relative.as_posix()
                    checksums[camera] = sha256(output / relative)
                    pts[camera] = timestamp
                require(pts["agentview"] == pts["wrist"], f"Camera desynchronization: {pair_id}")
                pairs.append(dict(pair_id=pair_id, demo_id=demo_id, suite=task["suite"], task_id=task["task_id"],
                                  task_name=task["task_name"], language=task["language"], phenomena=task["phenomena"],
                                  source_task_index=task["source_task_index"], episode_index=episode_id,
                                  episode_length=length, frame_slot=slot, frame_index=frame_index,
                                  source_global_index=table["index"][frame_index], timestamp=table["timestamp"][frame_index],
                                  video_timestamps=pts, normalized_progress=frame_index / (length - 1),
                                  images=paths, image_sha256=checksums, image_size=[256, 256],
                                  source_parquet=str(parquet), source_videos=videos))
        print(f"{task['suite']}/{task['task_id']}: episodes={task['selected_episode_indices']}, 25 pairs", flush=True)
    require(len(pairs) == 200 and len(demos) == 40, "Incorrect totals")
    require(len({r["pair_id"] for r in pairs}) == 200, "Duplicate pair IDs")
    require(set(Counter(r["demo_id"] for r in pairs).values()) == {5}, "Incorrect pairs/demo")
    (output / "pairs.jsonl").write_text(''.join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs))
    write_json(output / "demonstrations.json", demos)
    with (output / "pairs.csv").open("w", newline="") as f:
        fields = ["pair_id", "suite", "task_id", "language", "episode_index", "frame_index", "timestamp", "agentview", "wrist"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in pairs:
            writer.writerow({k: r["images"][k] if k in CAMERAS else r[k] for k in fields})
    task_records = [s[2] for s in selections]
    make_preview(output, pairs, task_records)
    write_json(output / "manifest.json", dict(
        status="complete", created_at=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        source_root=str(source), source_variant="LeRobot no_noops 1.0.0; existing AV1 videos",
        task_id_convention="zero-based official LIBERO task_order_index=0; NOT LeRobot task_index",
        sampling="random.Random(seed), sample without replacement from sorted episode IDs in listed task order",
        frame_sampling="floor(i * (N - 1) / 4), i=0..4, on the no_noops trajectory, inclusive endpoints",
        cameras=CAMERAS, image_format="RGB PNG, 256x256; decoded source pixels, no resize/flip/augmentation",
        tasks=task_records, num_tasks=8, num_demonstrations=40, num_pairs=200, num_images=400,
        source_sha256=source_files, task_map_sha256=sha256(args.task_map),
        extractor_sha256=sha256(Path(__file__)), pairs_sha256=sha256(output / "pairs.jsonl"),
        versions=dict(python=platform.python_version(), av=av.__version__, pillow=PIL.__version__,
                      numpy=np.__version__, pyarrow=pyarrow.__version__),
        limitations=["Frame indices and episode IDs refer to the local no_noops LeRobot version, not original HDF5 demo IDs.",
                     "PNG adds no compression loss, but source AV1 video is already compressed.",
                     "Target phenomena describe task selection; uniform frames are not semantic event annotations."]
    ))
    write_json(output / "validation.json", dict(status="passed", tasks=8, demonstrations=40, pairs=200,
        images=400, source_videos_checked=80, checks=["official task language matches LeRobot task metadata",
        "all parquet episode/task/frame indices checked", "full decoded video counts match parquet lengths",
        "all decoded frame timestamps match parquet within 1e-5 seconds", "selected camera PTS exactly equal",
        "PNG round-trip pixel equality and 256x256 RGB", "five unique demonstrations/task and five frames/demo",
        "source files and output images SHA256 recorded"]))
    (output / "README.txt").write_text(
        "LIBERO multiview pilot\n\n8 tasks x 5 demonstrations x 5 frames = 200 pairs / 400 RGB PNGs.\n"
        "Open index.html to inspect all pairs. Left: agentview. Right: wrist.\n"
        "pairs.jsonl: one synchronized pair per row; image paths relative to this directory.\n"
        "pairs.csv: compact index. demonstrations.json: selected frame indices and source paths.\n"
        "manifest.json: sampling recipe, source hashes, task mapping and limitations.\n"
        "selection.json: frozen candidate pools and sampled episode IDs. validation.json: checks performed.\n"
        "previews/: one contact sheet per task (preview images only are resized).\n\n"
        "Source is the existing LeRobot no_noops 1.0.0 conversion with AV1 video.\n"
        "Frame sampling: floor(i*(N-1)/4), i=0..4, including first/last frames.\n"
        "Task IDs use official LIBERO order 0; source_task_index records the different LeRobot ID.\n"
        "No crops, resizing, rotations or augmentations are applied to dataset PNGs.\n"
        "Phenomena are task-level sampling intentions, not verified labels for every frame.\n\n"
        "Rebuild into a new directory using the copied script and task map:\n"
        f"python build_multiview_pilot.py --source-root {source} "
        f"--task-map libero_suite_task_map.py --output /path/to/new-output --seed {args.seed}\n")
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
