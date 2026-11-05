"""Focused integration checks. Run inside starvla-pai-check with server on 8766."""
import ast
import json
import time
import urllib.error
import urllib.request
import pathlib
import sys
import numpy as np
import av
import server

BASE = "http://127.0.0.1:8766"
report = {"checks": [], "episodes": [], "video_checks": [], "browser_validation": "Not performed: user requested no Computer Use."}


def get(path, headers=None):
    return urllib.request.urlopen(urllib.request.Request(BASE + path, headers=headers or {}), timeout=60)


def body(path):
    with get(path) as r:
        return json.load(r)


def error_status(path, expected, headers=None):
    try:
        get(path, headers)
        raise AssertionError(f"Expected {expected} for {path}")
    except urllib.error.HTTPError as e:
        assert e.code == expected, (path, e.code, expected)


ast.parse((server.HERE / "server.py").read_text())
summary = body("/api/summary")
assert summary["index"]["ready"]
assert summary["index"]["indexed"] == 320712
assert body("/api/episodes?q=24348")["items"][0]["id"] == 24348
assert body("/api/episodes?q=shoe&limit=2")["total"] > 0
assert not body("/api/episodes?q=%25")["items"], "SQL LIKE wildcard must be literal"
first = body("/api/episodes?limit=2")["items"]
second = body("/api/episodes?limit=2&offset=2")["items"]
assert not ({r["id"] for r in first} & {r["id"] for r in second})
assert body("/api/episodes?q=zzzzzz_nonexistent")["total"] == 0
for stage in ["stage_01", "stage_02", "stage_03", "stage_04", "stage_05", "validation", "excluded"]:
    result = body("/api/episodes?stage=" + stage + "&limit=2")
    assert all(x["stage"] == stage for x in result["items"])
    if stage.startswith("stage"):
        assert result["total"] == len(np.unique(server.ranges(stage)[:, 0]))
report["checks"].append("catalog/search/escaped wildcard/pagination/stage filters")
error_status("/api/episode/-1", 404)
error_status("/api/episode/99999999", 404)
error_status("/api/episodes?stage=bad", 400)
error_status("/api/episodes?limit=999999", 422)
error_status("/api/video/0/not_a_camera", 404)
report["checks"].append("invalid IDs/cameras/filters rejected")

for ep in [0, 24348, 127240, 7075]:
    start = time.monotonic()
    d = body(f"/api/episode/{ep}")
    raw = server.read_episode(ep)
    assert len(raw["timestamp"]) == d["frames"]
    reconstructed = []
    for seg in d["segments"]:
        a, b = seg["row_start"], seg["row_end"]
        assert b > a and seg["end"] > seg["start"]
        assert all(v == seg["segment_index"] for v in raw["annotation.segment_index"][a:b])
        assert all((v or "无标注") == seg["instruction"] for v in raw["annotation.instruction"][a:b])
        assert all(bool(v) == seg["valid"] for v in raw["annotation.valid"][a:b])
        assert seg["start"] == raw["timestamp"][a]
        reconstructed.extend(range(a, b))
    assert reconstructed == list(range(d["frames"]))
    assert abs(d["segments"][-1]["end"] - d["duration"]) < 1e-3
    assert d["valid_frames"] == sum(raw["annotation.valid"])
    for a, b in d["training_ranges"]:
        for anchor in range(a, b, 6):
            assert anchor + 16 < d["frames"]
            assert all(raw["annotation.valid"][anchor:anchor+17])
            assert len(set(raw["annotation.segment_index"][anchor:anchor+17])) == 1
            assert len(set(raw["annotation.instruction"][anchor:anchor+17])) == 1
    report["episodes"].append({"id": ep, "frames": d["frames"], "segments": len(d["segments"]), "stage": d["stage"], "seconds": round(time.monotonic()-start, 3)})
    print(report["episodes"][-1], flush=True)
report["checks"].append("full-row annotation reconstruction, curated timestamps, real same-segment training anchors")

ep = 24348
m = server.metadata(ep)
for camera in server.CAMERAS:
    cm = next(c for c in m["cameras"] if c["camera"].endswith("." + camera))
    path = server.safe_path(server.VIDEOS, cm["video_path"][7:])
    route = f"/api/video/{ep}/{camera}"
    with get(route, {"Range": "bytes=0-511"}) as r:
        assert r.status == 206
        data = r.read()
        assert r.headers["Content-Range"].startswith("bytes 0-511/")
    with path.open("rb") as f:
        assert data == f.read(512)
    with get(route, {"Range": "bytes=-256"}) as r:
        assert r.status == 206
        tail = r.read()
    with path.open("rb") as f:
        f.seek(-256, 2)
        assert tail == f.read()
    error_status(route, 416, {"Range": f"bytes={path.stat().st_size}-"})
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        target = cm["from_timestamp"] + m["duration_seconds"]/2
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time >= target:
                assert abs(frame.time - target) < .07
                report["video_checks"].append({"camera": camera, "codec": stream.codec_context.name, "decoded_size": [frame.width, frame.height], "time_error_seconds": round(frame.time-target, 6), "range": "head/suffix/416 pass"})
                break
        else:
            raise AssertionError("No video frame near target")
    print(report["video_checks"][-1], flush=True)
report["checks"].append("4-camera real H264 decoding and byte-range parity")
with get("/") as r:
    assert r.status == 200 and b"UMI Atlas" in r.read()
for asset in ["/static/app.js", "/static/style.css"]:
    with get(asset) as r:
        assert r.status == 200
report["checks"].append("HTML / JS / CSS serving")
deps = server.CACHE / "check_deps"
if deps.exists():
    sys.path.insert(0, str(deps))
    import quickjs
    ctx = quickjs.Context()
    code = (server.HERE / "static/app.js").read_text()
    ctx.eval("new Function(" + json.dumps(code) + ")")
    report["checks"].append("QuickJS parses complete frontend JavaScript")
report["result"] = "passed"
(server.CACHE / "checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, ensure_ascii=False, indent=2))
