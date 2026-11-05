"""Read-only UMI / Luna explorer. Run in starvla-pai-check; no GPU needed."""
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager, asynccontextmanager
import argparse
import json
import math
import os
import sqlite3
import threading
import time
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ASSETS = Path(os.environ.get("UMI_ASSETS", "/mnt/workspace/Native_Policy/user/hjz/umi_wyt_assets/clean20_sameprompt_relative_20261015_v1"))
ROOT = Path(os.environ.get("UMI_ROOT", "/mnt/nas/public/roban_umi"))
SOURCE = ROOT / "restricted_data/derived/umi_v30_curated_luna_annotated_v1"
VIDEOS = ROOT / "restricted_data/source/umi/videos"
DB = ASSETS / "views/stage_01/metadata.sqlite3"
CACHE = REPO / "playground/Reports/umi_viewer_20261104"
CATALOG = CACHE / "catalog.sqlite3"
PLAN = json.loads((ASSETS / "wyt_input_plan.json").read_text())
META = json.loads((ASSETS / "views/stage_01/meta.json").read_text())
CAMERAS = ["head_left", "wrist_left", "wrist_right", "head_right"]
SIGNALS = [f"observation.umi.robot{arm}_{field}" for arm in (1, 2) for field in ("finger_eef_pose", "sensor_magnetic_encoder")]
STATUS = {"ready": False, "indexed": 0, "error": None}
INDEX_LOCK = threading.Lock()
READ_LOCK = threading.Lock()  # bound concurrent NAS row-group reads and RAM use


@contextmanager
def source_db():
    c = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


@lru_cache(maxsize=64)
def metadata(ep):
    with source_db() as c:
        row = c.execute("select payload from episodes where episode_index=?", (ep,)).fetchone()
    if not row:
        raise HTTPException(404, "Episode 不存在")
    return json.loads(row[0])


@lru_cache(maxsize=8)
def ranges(stage):
    return np.load(ASSETS / f"views/{stage}/ranges.npy", mmap_mode="r")


def membership(ep):
    for stage in [f"stage_{i:02}" for i in range(1, 6)] + ["validation"]:
        r = ranges(stage)
        a, b = np.searchsorted(r[:, 0], [ep, ep + 1])
        if a < b:
            return stage, r[a:b, 1:].tolist()
    return "excluded", []


def fingerprint():
    s = DB.stat()
    return f"v2:{DB}:{s.st_size}:{s.st_mtime_ns}"


def build_catalog():
    with INDEX_LOCK:
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            if CATALOG.exists():
                with sqlite3.connect(CATALOG) as c:
                    if c.execute("select value from info where key='fingerprint'").fetchone()[0] == fingerprint():
                        STATUS.update(ready=True, indexed=c.execute("select count(*) from episodes").fetchone()[0])
                        return
            tmp = CATALOG.with_suffix(".building")
            if tmp.exists():
                tmp.unlink()
            stages = {}
            for name in [f"stage_{i:02}" for i in range(1, 6)] + ["validation"]:
                stages.update((int(ep), name) for ep in np.unique(ranges(name)[:, 0]))
            out = sqlite3.connect(tmp)
            out.execute("create table episodes(id integer primary key, task text, duration real, frames integer, stage text, category text)")
            out.execute("create table info(key text primary key, value text)")
            batch = []
            with source_db() as src:
                for row in src.execute("select episode_index,payload from episodes order by episode_index"):
                    m = json.loads(row["payload"])
                    category = m.get("source_path_parent", "").rstrip("/").split("/")[-1]
                    batch.append((row[0], " · ".join(m.get("task_texts", [])), m["duration_seconds"], m["length"], stages.get(row[0], "excluded"), category))
                    if len(batch) >= 2000:
                        out.executemany("insert into episodes values(?,?,?,?,?,?)", batch)
                        STATUS["indexed"] += len(batch)
                        batch.clear()
                out.executemany("insert into episodes values(?,?,?,?,?,?)", batch)
                STATUS["indexed"] += len(batch)
            out.execute("create index stage_idx on episodes(stage,id)")
            out.execute("insert into info values('fingerprint',?)", (fingerprint(),))
            out.commit()
            out.close()
            tmp.replace(CATALOG)
            STATUS["ready"] = True
        except Exception as e:
            logging.exception("Catalog build failed")
            STATUS["error"] = str(e)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=build_catalog, daemon=True).start()
    yield


app = FastAPI(title="UMI Atlas", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.get("/api/summary")
def summary():
    return {
        "total_episodes": META["database_counts"]["episodes"], "fps": META["fps"],
        "train_episodes": PLAN["formal"]["training_pool"]["episodes"],
        "training_windows": PLAN["formal"]["training_pool"]["effective_5hz_windows"],
        "dataset": SOURCE.name, "host": "ppu-dev", "index": dict(STATUS),
        "stages": PLAN["stages"],
    }


@app.get("/api/episodes")
def episodes(q: str = Query("", max_length=160), stage: str = "", offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100)):
    if stage and stage not in [f"stage_{i:02}" for i in range(1, 6)] + ["validation", "excluded"]:
        raise HTTPException(400, "Unknown stage")
    if not STATUS["ready"]:
        raise HTTPException(503, f"正在建立搜索索引：{STATUS['indexed']:,} 条；仍可通过 episode ID 打开")
    with sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        predicates, args = [], []
        if q.strip():
            if q.strip().isdigit():
                predicates.append("id=?")
                args.append(int(q.strip()))
            else:
                predicates.append("(task like ? escape '\\' or category like ? escape '\\')")
                term = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                args.extend([f"%{term}%"] * 2)
        if stage:
            predicates.append("stage=?")
            args.append(stage)
        where = " where " + " and ".join(predicates) if predicates else ""
        total = c.execute("select count(*) from episodes" + where, args).fetchone()[0]
        rows = c.execute("select * from episodes" + where + " order by id limit ? offset ?", args + [limit, offset]).fetchall()
    return {"items": [dict(r) for r in rows], "total": total, "offset": offset, "limit": limit}


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise HTTPException(400, "Path outside data root")
    return path


def read_episode(ep):
    with source_db() as c:
        segments = c.execute("select * from segments where episode_index=? order by episode_row_offset_start", (ep,)).fetchall()
    pieces = []
    for seg in segments:
        path = safe_path(SOURCE, seg["data_file"])
        with pq.ParquetFile(path) as f:
            cols = [n for n in f.schema_arrow.names if n.startswith("annotation.") or n in ["episode_index", "frame_index", "timestamp"] + SIGNALS]
            start = seg["file_row_start"]
            stop = seg["file_row_end_exclusive"]
            base = 0
            for group in range(f.num_row_groups):
                end = base + f.metadata.row_group(group).num_rows
                if start < end and stop > base:
                    t = f.read_row_group(group, columns=cols, use_threads=False)
                    # Copy only selected episode buffers, releasing the full row group.
                    selected = t.slice(max(start, base) - base, min(stop, end) - max(start, base))
                    pieces.append(pa.Table.from_pydict(selected.to_pydict(), schema=selected.schema))
                base = end
    if not pieces:
        raise HTTPException(404, "没有该 episode 的数值数据")
    t = pa.concat_tables(pieces).to_pydict()
    if any(x != ep for x in t["episode_index"]):
        raise RuntimeError("Episode row mapping mismatch")
    return t


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


@lru_cache(maxsize=6)
def detail(ep):
    m = metadata(ep)
    with READ_LOCK:
        t = read_episode(ep)
    n = len(t["timestamp"])
    if n != m["length"]:
        raise RuntimeError(f"Episode length mismatch: {n} != {m['length']}")
    fps, duration = m["fps"], m["duration_seconds"]
    spans = []
    for i in range(n):
        key = (t["annotation.segment_index"][i], t["annotation.instruction"][i], bool(t["annotation.valid"][i]))
        if not spans or key != spans[-1]["key"]:
            spans.append({"key": key, "segment_index": key[0], "instruction": key[1] or "无标注",
                          "valid": key[2], "start": float(t["timestamp"][i]), "row_start": i,
                          "frame_start": t["frame_index"][i],
                          "source_start": t.get("annotation.start_s", [None] * n)[i],
                          "source_end": t.get("annotation.end_s", [None] * n)[i],
                          "curated_start": t.get("annotation.curated_start_s", [None] * n)[i],
                          "curated_end": t.get("annotation.curated_end_s", [None] * n)[i]})
        spans[-1]["end"] = float(t["timestamp"][i+1]) if i+1 < n else min(duration, float(t["timestamp"][i]) + 1/fps)
        spans[-1]["row_end"] = i+1
        spans[-1]["frame_end"] = t["frame_index"][i]+1
    for s in spans:
        del s["key"]
    stage, valid_ranges = membership(ep)
    cameras = []
    for name in CAMERAS:
        cm = next((c for c in m["cameras"] if c["camera"] == "observation.images." + name), None)
        if cm:
            cameras.append({"name": name, "start": cm["from_timestamp"], "end": cm["to_timestamp"],
                            "url": f"/api/video/{ep}/{name}"})
    return clean({
        "id": ep, "task": " · ".join(m["task_texts"]), "fps": fps, "duration": duration,
        "frames": n, "stage": stage, "cameras": cameras, "segments": spans,
        "valid_frames": sum(bool(v) for v in t["annotation.valid"]),
        "timestamps": t["timestamp"], "frame_indices": t["frame_index"],
        "signals": {k.split("observation.umi.")[1]: t[k] for k in SIGNALS},
        "training_ranges": valid_ranges, "anchor_stride": 6, "action_horizon": 16,
        "source": {"dataset": str(SOURCE), "parquet": m["data_path"], "recording": m.get("source_recording_key"), "set": m.get("source_set_id")},
    })


@app.get("/api/episode/{ep}")
def episode(ep: int):
    if ep < 0:
        raise HTTPException(404, "Episode 不存在")
    try:
        return JSONResponse(detail(ep))
    except HTTPException:
        raise
    except Exception:
        logging.exception("Episode load failed: %s", ep)
        raise HTTPException(500, "读取 episode 失败，请查看远程服务日志")


@app.get("/api/video/{ep}/{camera}")
def video(ep: int, camera: str):
    if camera not in CAMERAS:
        raise HTTPException(404, "Unknown camera")
    m = metadata(ep)
    cam = next((c for c in m["cameras"] if c["camera"] == "observation.images." + camera), None)
    if not cam or not cam["video_path"].startswith("videos/"):
        raise HTTPException(404, "Camera unavailable")
    path = safe_path(VIDEOS, cam["video_path"][len("videos/"):])
    if not path.is_file():
        raise HTTPException(404, "视频文件不存在")
    # Starlette FileResponse handles byte ranges, suffix requests and 416.
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/")
def home():
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
