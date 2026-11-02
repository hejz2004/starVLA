"""Read immutable UMI window indexes without enumerating individual windows.

By default values retain source signs. An explicit offline quaternion_signs
map can select equivalent antipodes before normalization. No padding, replacement
sampling, unit projection, coordinate changes, or quality filtering occur.
The access index is built offline by build_umi_access_index.py.
"""
from collections import OrderedDict
import hashlib
import json
import math
import operator
import os
from pathlib import Path, PurePosixPath
import sqlite3
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset
from starVLA.dataloader.umi_video_time import VideoTimeWindow, video_time_policy as resolve_video_policy
from starVLA.dataloader.umi_language import (
    FINE_COLUMNS, FINE_LANGUAGE_POLICY, FINE_LANGUAGE_SOURCE, extract_fine_language,
    SAME_SEGMENT_PROTOCOL, SAME_SEGMENT_POLICY,
)


CAMERAS = tuple("observation.images." + name for name in
                ("head_left", "head_right", "wrist_left", "wrist_right"))
TRAINING_CAMERAS = tuple(key for key in CAMERAS if key != "observation.images.head_right")


def selected_cameras(camera_keys=None):
    """Keep source metadata intact; only selected views are decoded for the model."""
    if camera_keys is None:
        return CAMERAS  # Existing four-view experiments keep their original behavior.
    if isinstance(camera_keys, (str, bytes)):
        raise TypeError("camera_keys must be an ordered sequence of camera names")
    keys = tuple(camera_keys)
    if not keys or any(not isinstance(key, str) or key not in CAMERAS for key in keys):
        raise ValueError("camera_keys must be a nonempty subset of the indexed UMI cameras")
    if len(set(keys)) != len(keys):
        raise ValueError("camera_keys must not contain duplicate views")
    return keys

SIGNALS = (
    "observation.umi.robot1_finger_eef_pose",
    "observation.umi.robot1_sensor_magnetic_encoder",
    "observation.umi.robot2_finger_eef_pose",
    "observation.umi.robot2_sensor_magnetic_encoder",
)
WIDTHS = (7, 1, 7, 1)
COLUMNS = ("episode_index", "frame_index", "timestamp", "task_index") + SIGNALS


def compute_view_fingerprint(meta):
    """Identify the selected window view, independently of its current location.

    Keep this stdlib-only formula identical to build_umi_access_index.py. Hashes
    identify the compiled artifacts; this function does not rehash their bytes.
    Source/catalog paths, build times and selection-list filenames are excluded.
    """
    identity = {
        "identity_version": "umi-access-view-identity-v1",
        "version": meta["version"],
        "rule_fingerprint": meta["rule_fingerprint"],
        "catalog_artifacts_sha256": meta["catalog_artifacts_sha256"],
        "artifacts_sha256": {
            name: meta["artifacts"][name]["sha256"]
            for name in ("ranges.npy", "cumulative.npy", "metadata.sqlite3")
        },
        "horizon": meta["horizon"], "fps": meta["fps"],
        "signals": meta["signals"], "signal_widths": meta["signal_widths"],
        "video_keys": meta["video_keys"],
        "representation": meta["representation"],
        "index_order": meta["index_order"],
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _BoundedCache:
    def __init__(self, max_items, max_bytes=None, close=None):
        self.max_items, self.max_bytes, self.closer = max_items, max_bytes, close
        self.items, self.bytes = OrderedDict(), 0
        self.track = False
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self.items:
            if self.track:
                self.misses += 1
            return None
        if self.track:
            self.hits += 1
        value, size = self.items.pop(key)
        self.items[key] = (value, size)
        return value

    def put(self, key, value, size=0):
        if self.max_items == 0 or (self.max_bytes is not None and size > self.max_bytes):
            return value
        if key in self.items:
            self._remove(key)
        while self.items and (len(self.items) >= self.max_items or
                              (self.max_bytes is not None and self.bytes + size > self.max_bytes)):
            self._remove(next(iter(self.items)))
        self.items[key] = (value, size)
        self.bytes += size
        return value

    def _remove(self, key):
        value, size = self.items.pop(key)
        self.bytes -= size
        if self.track:
            self.evictions += 1
        if self.closer is not None:
            self.closer(value)

    def clear(self):
        while self.items:
            self._remove(next(iter(self.items)))


class UMIIndexedDataset(Dataset):
    """Map each integer to one complete current+16-future window.

    Metadata stays in read-only SQLite; ranges are memory mapped. Per-worker
    caches have explicit bounds. SQLite/video handles are opened lazily in the
    worker and discarded on fork/pickle. This class never writes source files.

    Optional anchor striding restarts at each valid range's beginning. It only
    selects current frames; every selected action still contains the next 16
    consecutive source frames, without temporal downsampling of the labels.
    """
    def __init__(self, index_dir, *, source_root=None, video_root=None, image_size=(224, 224),
                 row_cache_mib=128, row_cache_items=32, file_cache_items=4,
                 video_cache_items=8, episode_cache_items=128,
                 source_identity="size_mtime", decode_tolerance_seconds=None,
                 return_metadata=False, allowed_video_roots=(), performance_diagnostics=False,
                 video_time_policy="legacy_microsecond", boundary_slack_seconds=None,
                 camera_keys=None, quaternion_signs=None, language_source="original_task",
                 batched_video_decode=False, batched_video_max_span_seconds=2,
                 anchor_stride_frames=1, first_anchor_tolerance_seconds=None, language_window_policy="anchor_only", excluded_episode_indices=(), verify_quaternion_continuity=False, audit_window_contract=False):
        if isinstance(anchor_stride_frames, (bool, np.bool_)):
            raise ValueError("anchor_stride_frames must be a positive integer")
        try:
            self.anchor_stride_frames = operator.index(anchor_stride_frames)
        except TypeError as error:
            raise ValueError("anchor_stride_frames must be a positive integer") from error
        if not 1 <= self.anchor_stride_frames <= np.iinfo(np.int64).max:
            raise ValueError("anchor_stride_frames must be a positive int64 integer")
        if language_source not in ("original_task", FINE_LANGUAGE_SOURCE):
            raise ValueError("language_source must be original_task or luna_fine")
        self.language_source = language_source
        if language_window_policy not in ("anchor_only", "same_segment"):
            raise ValueError("language_window_policy must be anchor_only or same_segment")
        if language_window_policy == "same_segment" and language_source != FINE_LANGUAGE_SOURCE:
            raise ValueError("same_segment requires luna_fine language")
        self.language_window_policy = language_window_policy
        if type(audit_window_contract) is not bool:
            raise ValueError("audit_window_contract must be a boolean")
        self.audit_window_contract = audit_window_contract
        if any(isinstance(ep, bool) or not isinstance(ep, (int, np.integer)) or ep < 0
               for ep in excluded_episode_indices):
            raise ValueError("excluded_episode_indices must contain nonnegative integers")
        self.excluded_episode_indices = frozenset(int(ep) for ep in excluded_episode_indices)
        if type(verify_quaternion_continuity) is not bool:
            raise ValueError("verify_quaternion_continuity must be a boolean")
        if verify_quaternion_continuity and quaternion_signs is None:
            raise ValueError("Continuity verification requires an explicit quaternion sign map")
        self.verify_quaternion_continuity = verify_quaternion_continuity
        self.columns = COLUMNS + (("annotation.instruction",) if language_source == FINE_LANGUAGE_SOURCE else ())
        self.camera_keys = selected_cameras(camera_keys)
        if type(batched_video_decode) is not bool:
            raise ValueError("batched_video_decode must be a boolean")
        from .umi_video_batch import validate_batch_span
        self.batched_video_decode = batched_video_decode
        self.batched_video_max_span_seconds = validate_batch_span(batched_video_max_span_seconds)
        self.index_dir = Path(index_dir).resolve()
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        if (language_window_policy == "same_segment"
                and self.meta.get("selection_language_protocol",
                    self.meta.get("window_contract", {}).get("language_protocol")) != SAME_SEGMENT_PROTOCOL):
            raise ValueError("same_segment requires an index built for the same-segment protocol")
        if self.excluded_episode_indices and not audit_window_contract:
            declared = self.meta.get("excluded_episode_indices", [])
            if not self.excluded_episode_indices.issubset(set(declared)):
                raise ValueError("Prepared index does not declare the required episode exclusions")
        if (self.meta.get("version") != "roban-umi-access-v1" or self.meta.get("horizon") != 16
                or self.meta.get("status") != "completed"):
            raise ValueError("Require a completed roban-umi-access-v1 index with horizon=16")
        # Legacy full indexes derive the same identity without rewriting meta.
        # A stored identity is checked against the declared artifact hashes;
        # payload hash verification remains an offline index-validation step.
        self.view_fingerprint = compute_view_fingerprint(self.meta)
        if ("view_fingerprint" in self.meta
                and self.meta["view_fingerprint"] != self.view_fingerprint):
            raise ValueError("Access view fingerprint disagrees with index metadata")
        if tuple(self.meta["signals"]) != SIGNALS or tuple(self.meta["signal_widths"]) != WIDTHS:
            raise ValueError("Access index differs from robot1/robot2 world-pose-16D contract")
        for name in ("ranges.npy", "cumulative.npy", "metadata.sqlite3"):
            if (self.index_dir / name).stat().st_size != self.meta["artifacts"][name]["size_bytes"]:
                raise ValueError(f"Incomplete runtime index file: {name}")
        if tuple(self.meta["video_keys"]) != CAMERAS:
            raise ValueError("Access index does not use the established four-view order")
        self.source_root = Path(source_root or self.meta["source_path"]).resolve()
        # A separate, explicit videos directory supports sources such as Luna
        # that preserve camera offsets without containing a videos/ link.
        if video_root is not None and not isinstance(video_root, (str, Path)):
            raise TypeError("video_root must be one directory path")
        if video_root is not None and not str(video_root).strip():
            raise ValueError("video_root must be a nonempty directory path")
        self.video_root = Path(video_root).resolve() if video_root is not None else None
        if self.video_root is not None and not self.video_root.is_dir():
            raise ValueError(f"video_root is not an existing directory: {self.video_root}")
        # Curated datasets may link videos to a sibling source-data directory.
        # Extra read roots are explicit configuration, never inferred from an
        # arbitrary link target. Numeric Parquet remains confined to source_root.
        if isinstance(allowed_video_roots, (str, Path)):
            raise TypeError("allowed_video_roots must be a sequence of directory paths")
        self.allowed_video_roots = tuple(Path(path).resolve() for path in allowed_video_roots)
        self.fps = float(self.meta["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid FPS in access index")
        if source_identity not in ("size_mtime", "size"):
            raise ValueError("source_identity must be size_mtime, or explicitly size after relocation")
        self.source_identity = source_identity
        self.image_size = tuple(operator.index(x) for x in image_size)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers")
        self.row_cache_mib = float(row_cache_mib)
        if not math.isfinite(self.row_cache_mib) or self.row_cache_mib < 0:
            raise ValueError("row_cache_mib must be finite and nonnegative")
        for name, value, minimum in (
            ("row_cache_items", row_cache_items, 0), ("file_cache_items", file_cache_items, 1),
            ("video_cache_items", video_cache_items, 1), ("episode_cache_items", episode_cache_items, 1),
        ):
            if operator.index(value) < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            setattr(self, name, operator.index(value))
        self.decode_tolerance_seconds = (1 / self.fps + 1e-6 if decode_tolerance_seconds is None
                                         else float(decode_tolerance_seconds))
        if not math.isfinite(self.decode_tolerance_seconds) or self.decode_tolerance_seconds <= 0:
            raise ValueError("decode_tolerance_seconds must be finite and positive")
        self.first_anchor_tolerance_seconds = (None if first_anchor_tolerance_seconds is None
                                               else float(first_anchor_tolerance_seconds))
        if (self.first_anchor_tolerance_seconds is not None
                and (not math.isfinite(self.first_anchor_tolerance_seconds)
                     or self.first_anchor_tolerance_seconds < self.decode_tolerance_seconds)):
            raise ValueError("first_anchor_tolerance_seconds must be finite and at least the normal tolerance")
        self.video_policy_name = video_time_policy
        # Validate once at construction, also included in run provenance.
        self.video_policy, slack = resolve_video_policy(video_time_policy, boundary_slack_seconds)
        self.boundary_slack_seconds = float(slack)
        self.return_metadata = bool(return_metadata)
        self.performance_diagnostics = bool(performance_diagnostics)
        if self.performance_diagnostics and not self.return_metadata:
            raise ValueError("performance_diagnostics requires return_metadata")
        self._pid = None
        self._db = self._rows = self._files = self._videos = self._episodes = self._tasks = None
        self._ranges = self._cumulative = self._sampling_cumulative = None
        self._open_arrays()
        self.quaternion_signs = None
        if quaternion_signs is not None:
            from .umi_quaternion import QuaternionSigns, transformed_meta
            self.quaternion_signs = QuaternionSigns(quaternion_signs, self.meta, audit_assets=self.audit_window_contract)
            self.meta = transformed_meta(self.meta, self.quaternion_signs.manifest)

    def _open_arrays(self):
        self._ranges = np.load(self.index_dir / "ranges.npy", mmap_mode="r", allow_pickle=False)
        self._cumulative = np.load(self.index_dir / "cumulative.npy", mmap_mode="r", allow_pickle=False)
        n = int(self.meta["total_ranges"])
        if self.audit_window_contract and (self._ranges.shape != (n, 3) or self._cumulative.shape != (n,) or n == 0
                or self._ranges.dtype != np.dtype("int64")
                or self._cumulative.dtype != np.dtype("int64")
                or int(self._cumulative[-1]) != int(self.meta["total_windows"])):
            raise ValueError("Access index shape/count mismatch")
        lengths = self._ranges[:, 2] - self._ranges[:, 1]
        if self.audit_window_contract and (np.any(lengths <= 0) or not np.array_equal(np.cumsum(lengths, dtype=np.int64), self._cumulative)):
            raise ValueError("Invalid range cumulative counts")
        # Keep the original dense arrays. Optional diagnostics follow audit_window_contract;
        # the opt-in logical index uses ceil(range_length / stride) counts.
        self._sampling_cumulative = None
        if self.anchor_stride_frames > 1:
            counts = lengths // self.anchor_stride_frames + (lengths % self.anchor_stride_frames != 0)
            self._sampling_cumulative = np.cumsum(counts, dtype=np.int64)
            self._num_windows = int(self._sampling_cumulative[-1])
        else:
            self._num_windows = int(self.meta["total_windows"])

    def _worker(self):
        if self._pid == os.getpid():
            return
        self.close()
        if self._ranges is None:
            self._open_arrays()
        self._db = sqlite3.connect((self.index_dir / "metadata.sqlite3").as_uri() + "?mode=ro", uri=True)
        self._db.execute("PRAGMA query_only=ON")
        self._db.execute("PRAGMA cache_size=-8192")
        self._db.execute("PRAGMA mmap_size=0")
        self._rows = _BoundedCache(self.row_cache_items, int(self.row_cache_mib * 1024**2))
        self._files = _BoundedCache(self.file_cache_items, close=lambda item: item[0].close())
        self._videos = _BoundedCache(self.video_cache_items, close=lambda item: item[0].close())
        self._episodes = _BoundedCache(self.episode_cache_items)
        self._tasks = _BoundedCache(1024)
        for cache in (self._rows, self._files, self._videos, self._episodes, self._tasks):
            cache.track = self.performance_diagnostics
        self._pid = os.getpid()

    def close(self):
        if getattr(self, 'quaternion_signs', None) is not None:
            self.quaternion_signs.close()
        for name in ("_videos", "_files", "_rows", "_episodes", "_tasks"):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()
            setattr(self, name, None)
        if getattr(self, "_db", None) is not None:
            self._db.close()
        self._db = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_pid", "_db", "_rows", "_files", "_videos", "_episodes", "_tasks",
                    "_ranges", "_cumulative", "_sampling_cumulative"):
            state[key] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self._num_windows

    def locate(self, index):
        """Return episode ID and episode-local row offset, not a global file row."""
        episode, anchor, _ = self._locate_with_source_index(index)
        return episode, anchor

    def source_dense_index(self, index):
        """Map a logical sample to its unchanged dense access-view index."""
        return self._locate_with_source_index(index)[2]

    def _locate_with_source_index(self, index):
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(f"UMI index {index} outside [0,{len(self)})")
        if self._ranges is None:
            self._open_arrays()
        cumulative = self._cumulative if self.anchor_stride_frames == 1 else self._sampling_cumulative
        position = int(np.searchsorted(cumulative, index, side="right"))
        base = int(cumulative[position - 1]) if position else 0
        offset = (index - base) * self.anchor_stride_frames
        dense_base = int(self._cumulative[position - 1]) if position else 0
        episode, begin, _ = self._ranges[position]
        return int(episode), int(begin) + offset, dense_base + offset

    def _path(self, relative, *, video=False):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid relative asset path: {relative}")
        if video and self.video_root is not None:
            if (len(path.parts) < 2 or path.parts[0] != "videos"
                    or "\\" in relative or ":" in relative):
                raise ValueError(f"Explicit video_root requires videos/<suffix>: {relative}")
            resolved = self.video_root.joinpath(*path.parts[1:]).resolve()
            if not resolved.is_relative_to(self.video_root):
                raise ValueError(f"Video asset escapes explicit video_root: {relative}")
            return resolved
        resolved = (self.source_root / relative).resolve()
        roots = (self.source_root,) + (self.allowed_video_roots if video else ())
        if not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError(f"Asset escapes source root: {relative}")
        return resolved

    def _episode(self, episode):
        cached = self._episodes.get(episode)
        if cached is not None:
            return cached
        row = self._db.execute("SELECT payload FROM episodes WHERE episode_index=?", (episode,)).fetchone()
        if row is None:
            raise ValueError(f"No metadata for episode {episode}")
        metadata = json.loads(row[0])
        segments = self._db.execute(
            "SELECT data_file,file_row_start,file_row_end_exclusive,episode_row_offset_start,"
            "episode_row_offset_end_exclusive FROM segments WHERE episode_index=? "
            "ORDER BY episode_row_offset_start", (episode,)).fetchall()
        return self._episodes.put(episode, (metadata, segments))

    def _parquet(self, relative):
        cached = self._files.get(relative)
        if cached is not None:
            return cached
        path = self._path(relative)
        if self.audit_window_contract:
            expected = self._db.execute("SELECT size_bytes,mtime_ns FROM data_files WHERE data_file=?",
                                        (relative,)).fetchone()
            if expected is None:
                raise ValueError(f"Unindexed data file {relative}")
            stat = path.stat()
            if stat.st_size != expected[0] or (self.source_identity == "size_mtime" and stat.st_mtime_ns != expected[1]):
                raise ValueError(f"Source file changed since indexing: {relative}")
        reader = pq.ParquetFile(path)
        if self.audit_window_contract:
            missing = set(self.columns) - set(reader.schema_arrow.names)
            if missing:
                reader.close()
                raise ValueError(f"Missing columns in {relative}: {sorted(missing)}")
        ends = np.cumsum([reader.metadata.row_group(i).num_rows for i in range(reader.num_row_groups)],
                         dtype=np.int64)
        return self._files.put(relative, (reader, ends))

    def _file_slice(self, relative, begin, end):
        reader, ends = self._parquet(relative)
        if self.audit_window_contract and not 0 <= begin < end <= int(ends[-1]):
            raise ValueError(f"File row range outside {relative}: [{begin},{end})")
        pieces = []
        for group in range(int(np.searchsorted(ends, begin, side="right")),
                           int(np.searchsorted(ends, end - 1, side="right")) + 1):
            key = (relative, group)
            table = self._rows.get(key)
            if table is None:
                table = reader.read_row_group(group, columns=list(self.columns), use_threads=False)
                self._rows.put(key, table, table.nbytes)
            group_start = int(ends[group - 1]) if group else 0
            a, b = max(begin, group_start) - group_start, min(end, int(ends[group])) - group_start
            pieces.append(table.slice(a, b - a))
        return pa.concat_tables(pieces)

    def read_lowdim(self, index):
        """Return unscaled state/action in the declared sign policy, without images."""
        self._worker()
        episode, anchor, source_index = self._locate_with_source_index(index)
        if self.audit_window_contract and episode in self.excluded_episode_indices:
            raise ValueError(f"Excluded episode in access view: {episode}")
        metadata, segments = self._episode(episode)
        stop = anchor + 17
        if self.audit_window_contract and stop > metadata["length"]:
            raise ValueError(f"Incomplete indexed window: episode={episode}, anchor={anchor}")
        pieces, locations, next_row = [], [], anchor
        for relative, file_a, file_b, ep_a, ep_b in segments:
            a, b = max(anchor, ep_a), min(stop, ep_b)
            if a >= b:
                continue
            if self.audit_window_contract and (a != next_row or file_b - file_a != ep_b - ep_a):
                raise ValueError(f"Broken segment map: episode={episode}, anchor={anchor}")
            local_a, local_b = file_a + a - ep_a, file_a + b - ep_a
            pieces.append(self._file_slice(relative, local_a, local_b))
            locations.append({"data_file": relative, "file_row_start": local_a, "file_row_end_exclusive": local_b})
            next_row = b
        if self.audit_window_contract and next_row != stop:
            raise ValueError(f"Missing indexed rows: episode={episode}, anchor={anchor}")
        table = pa.concat_tables(pieces)
        identities = table["episode_index"].to_numpy() if self.audit_window_contract else None
        frames = table["frame_index"].to_numpy()
        task_ids = table["task_index"].to_numpy()
        timestamps = table["timestamp"].to_numpy()
        if self.audit_window_contract and (len(table) != 17 or not np.all(identities == episode) or not np.all(np.diff(frames) == 1)
                or not np.all(task_ids == task_ids[0]) or not np.isfinite(timestamps).all()):
            raise ValueError(f"Indexed row identities no longer match: episode={episode}, anchor={anchor}")
        chunks = []
        for key, width in zip(SIGNALS, WIDTHS):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if width == 1 and values.shape == (17,):
                values = values[:, None]
            if self.audit_window_contract and (values.shape != (17, width) or not np.isfinite(values).all()):
                raise ValueError(f"Invalid indexed signal {key}: episode={episode}, anchor={anchor}")
            chunks.append(values)
        values = np.concatenate(chunks, axis=1)
        quaternion_trace = {}
        if self.verify_quaternion_continuity:
            from .umi_quaternion import continuity_report
            quaternion_trace["raw"] = continuity_report(values)
        if self.quaternion_signs is not None:
            values = self.quaternion_signs.apply(values, locations)
        if self.verify_quaternion_continuity:
            quaternion_trace["signed"] = continuity_report(values)
            if quaternion_trace["signed"]["negative_pairs"]:
                raise ValueError(f"Quaternion sign discontinuity after correction: episode={episode}, anchor={anchor}")
        task_id = int(task_ids[0])
        language_trace = {}
        if self.language_source == FINE_LANGUAGE_SOURCE:
            text = table["annotation.instruction"][0].as_py()
            language_trace = dict(language_source=FINE_LANGUAGE_SOURCE,
                language_policy=SAME_SEGMENT_PROTOCOL if self.language_window_policy == "same_segment" else "luna-anchor-instruction-v1",
                window_contract_check="disabled; prepared index used")
        else:
            text = self._tasks.get(task_id)
            if text is None:
                record = self._db.execute("SELECT task FROM tasks WHERE task_index=?", (task_id,)).fetchone()
                if record is None or not isinstance(record[0], str) or not record[0].strip():
                    raise ValueError(f"No language for task_index={task_id}")
                text = self._tasks.put(task_id, record[0])
        sampling_trace = {"source_dense_index": source_index} if self.anchor_stride_frames > 1 else {}
        if quaternion_trace:
            sampling_trace["quaternion_continuity"] = quaternion_trace
        return {"state": values[:1].copy(), "action": values[1:].copy(), "lang": text,
                "trace": {"dataset_index": int(index), "episode_index": episode,
                          "episode_row_offset": anchor, "frame_index": int(frames[0]),
                          "future_frame_indices": frames[1:].tolist(), "timestamp": float(timestamps[0]),
                          "task_index": task_id, "locations": locations,
                          "rule_fingerprint": self.meta["rule_fingerprint"],
                          "view_fingerprint": self.view_fingerprint, **language_trace, **sampling_trace},
                "cameras": metadata["cameras"]}

    def _open_video(self, camera):
        import av
        relative = camera["video_path"]
        cached = self._videos.get(relative)
        if cached is None:
            container = av.open(str(self._path(relative, video=True)))
            stream = container.streams.video[0]
            stream.thread_count = 1
            cached = self._videos.put(relative, (container, stream))
        return cached

    def _video_window(self, camera, timestamp, stream):
        tolerance = self.decode_tolerance_seconds
        # Allow a later same-episode frame only at the initial 5 Hz anchor.
        if timestamp == 0 and self.first_anchor_tolerance_seconds is not None:
            tolerance = self.first_anchor_tolerance_seconds
        return VideoTimeWindow(camera["from_timestamp"], camera["to_timestamp"], timestamp,
                                 stream.time_base, tolerance,
                                 policy=self.video_policy_name, boundary_slack_seconds=self.boundary_slack_seconds)

    def _render_video_frame(self, camera, window, stream, closest, closest_time):
        relative = camera["video_path"]
        image = Image.fromarray(closest.to_ndarray(format="rgb24")).resize(self.image_size)
        return image, {"camera": camera["camera"], "video_path": relative,
                       "requested_seconds": float(window.target), "decoded_seconds": float(closest_time),
                       "offset_seconds": float(closest_time - window.target),
                       "decoded_pts": closest.pts, "time_base": str(stream.time_base),
                       "boundary_policy": self.video_policy["version"],
                       "boundary_slack_seconds": self.boundary_slack_seconds,
                       "used_boundary_slack": (not window.legacy and
                                                not window.start <= closest_time < window.end)}

    def _video_frame(self, camera, timestamp):
        container, stream = self._open_video(camera)
        window = self._video_window(camera, timestamp, stream)
        container.seek(window.seek_pts, stream=stream, backward=True, any_frame=False)
        closest, closest_time = window.select(container.decode(video=0))
        return self._render_video_frame(camera, window, stream, closest, closest_time)

    def __getitem__(self, index):
        try:
            start = time.monotonic() if self.performance_diagnostics else None
            data = self.read_lowdim(index)
            lowdim_end = time.monotonic() if self.performance_diagnostics else None
            cameras = {camera["camera"]: camera for camera in data["cameras"]}
            images, decoded = [], []
            for key in self.camera_keys:
                image, timing = self._video_frame(cameras[key], data["trace"]["timestamp"])
                images.append(image)
                decoded.append(timing)
            sample = {"image": images, "lang": data["lang"], "state": data["state"],
                      "action": data["action"], "robot_tag": "new_embodiment"}
            if self.return_metadata:
                sample["umi_metadata"] = dict(data["trace"], video_decode=decoded, reader_pid=os.getpid())
            if self.performance_diagnostics:
                sample["umi_metadata"]["performance"] = dict(
                    lowdim_seconds=lowdim_end-start, video_seconds=time.monotonic()-lowdim_end,
                    caches={name: dict(hits=cache.hits, misses=cache.misses, evictions=cache.evictions,
                                       bytes=cache.bytes, items=len(cache.items)) for name, cache in
                            (("rows", self._rows), ("files", self._files), ("videos", self._videos),
                             ("episodes", self._episodes), ("tasks", self._tasks))})
            return sample
        except Exception as error:
            raise RuntimeError(f"UMI read failed at dataset_index={index}: {error}") from error

    def __getitems__(self, indices):
        """Opt-in DataLoader batch fetch; ordinary item reads retain their path."""
        if not self.batched_video_decode:
            return [self[index] for index in indices]
        from .umi_video_batch import decode_video_requests
        indices = list(indices)
        if not indices:
            return []
        try:
            lowdim, lowdim_seconds, requests = [], [], []
            for position, index in enumerate(indices):
                started = time.monotonic()
                data = self.read_lowdim(index)
                lowdim.append(data)
                lowdim_seconds.append(time.monotonic() - started)
                cameras = {camera["camera"]: camera for camera in data["cameras"]}
                requests.extend(((position, view), cameras[key], data["trace"]["timestamp"])
                                for view, key in enumerate(self.camera_keys))
            started = time.monotonic()
            decoded = decode_video_requests(
                requests, open_video=self._open_video, make_window=self._video_window,
                render_frame=self._render_video_frame,
                max_span_seconds=self.batched_video_max_span_seconds)
            video_seconds = (time.monotonic() - started) / len(indices)
            samples = []
            for position, data in enumerate(lowdim):
                pairs = [decoded[position, view] for view in range(len(self.camera_keys))]
                sample = {"image": [image for image, _ in pairs], "lang": data["lang"],
                          "state": data["state"], "action": data["action"],
                          "robot_tag": "new_embodiment"}
                if self.return_metadata:
                    sample["umi_metadata"] = dict(data["trace"],
                        video_decode=[timing for _, timing in pairs], reader_pid=os.getpid())
                if self.performance_diagnostics:
                    sample["umi_metadata"]["performance"] = dict(
                        lowdim_seconds=lowdim_seconds[position], video_seconds=video_seconds,
                        batched_video_decode=True, batched_video_samples=len(indices),
                        video_seconds_policy="batch total amortized across returned samples",
                        caches={name: dict(hits=cache.hits, misses=cache.misses, evictions=cache.evictions,
                                           bytes=cache.bytes, items=len(cache.items)) for name, cache in
                                (("rows", self._rows), ("files", self._files), ("videos", self._videos),
                                 ("episodes", self._episodes), ("tasks", self._tasks))})
                samples.append(sample)
            return samples
        except Exception as error:
            raise RuntimeError(f"UMI batch read failed at dataset_indices={indices}: {error}") from error

    def sampling_identity(self):
        """Identify anchor selection separately from the view used to bind stats."""
        identity = {"version": "umi-anchor-stride-v1", "source_view_fingerprint": self.view_fingerprint,
                    "anchor_stride_frames": self.anchor_stride_frames, "phase_frames": 0,
                    "phase_origin": "valid_range_begin", "source_dense_windows": int(self.meta["total_windows"]),
                    "num_windows": len(self), "source_fps": self.fps,
                    "action_horizon_frames": 16, "action_stride_frames": 1}
        if self.first_anchor_tolerance_seconds is not None:
            identity.update(version="umi-anchor-stride-first-anchor-tolerance-v1",
                            video_time_policy=self.video_policy_name,
                            boundary_slack_seconds=self.boundary_slack_seconds,
                            decode_tolerance_seconds=self.decode_tolerance_seconds,
                            first_anchor_tolerance_seconds=self.first_anchor_tolerance_seconds)
        encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return dict(identity, sampling_fingerprint=hashlib.sha256(encoded).hexdigest())

    def provenance(self):
        result = {"dataset_type": "umi_indexed", "index_dir": str(self.index_dir),
                "source_root": str(self.source_root), "rule_fingerprint": self.meta["rule_fingerprint"],
                "view_fingerprint": self.view_fingerprint,
                "allowed_video_roots": [str(path) for path in self.allowed_video_roots],
                "num_windows": len(self), "num_episodes": self.meta["total_episodes"],
                "trainable_episodes": self.meta["trainable_episodes"],
                "state_shape": [1, 16], "action_shape": [16, 16], "dtype": "float32",
                "camera_order": list(self.camera_keys), "image_size": list(self.image_size),
                "normalization": "none", "source_identity": self.source_identity,
                "decode_tolerance_seconds": self.decode_tolerance_seconds,
                "first_anchor_tolerance_seconds": self.first_anchor_tolerance_seconds,
                "video_time_policy": dict(self.video_policy),
                "note": "Source checks and decode tolerance do not prove physical label quality or sensor synchronization."}
        if self.video_root is not None:
            result.update(video_root=str(self.video_root),
                          video_path_mapping="videos/<suffix> -> video_root/<suffix>")
        if self.language_source == FINE_LANGUAGE_SOURCE:
            result.update(language_source=self.language_source,
                          language_policy=dict(SAME_SEGMENT_POLICY if self.language_window_policy == "same_segment"
                                               else FINE_LANGUAGE_POLICY))
        if self.excluded_episode_indices:
            result["excluded_episode_indices"] = sorted(self.excluded_episode_indices)
        if self.batched_video_decode:
            result.update(batched_video_decode=True,
                          batched_video_max_span_seconds=self.batched_video_max_span_seconds)
        if self.anchor_stride_frames > 1:
            result["sampling"] = self.sampling_identity()
        if self.quaternion_signs is not None:
            result['quaternion_transform'] = self.quaternion_signs.provenance()
        return result


def collate_umi_samples(batch):
    # QwenPI accepts a list of per-example dictionaries and handles PIL images.
    return batch


class UMINormalizedDataset(Dataset):
    """Apply action geometry then fixed scaling outside the immutable raw reader.

    Access raw low-dimensional values through raw_dataset.read_lowdim(). The
    wrapper deliberately does not alias that method to a transformed result.
    Normalization parameters are picklable; raw worker handles remain managed
    by UMIIndexedDataset's existing process-local lifecycle.
    """
    def __init__(self, raw_dataset, normalizer, action_pose_repr="world"):
        from starVLA.dataloader.umi_pose import model_access_meta, validate_action_pose_repr
        from starVLA.dataloader.umi_normalization import build_representation
        self.raw_dataset = raw_dataset
        self.normalizer = normalizer
        self.action_pose_repr = validate_action_pose_repr(action_pose_repr)
        source_meta = getattr(raw_dataset, "meta", None)
        self.model_meta = None if source_meta is None else model_access_meta(source_meta, self.action_pose_repr)
        if self.action_pose_repr != "world" and self.model_meta is None:
            raise ValueError("Relative actions require the source pose metadata")
        representation = getattr(normalizer, "representation", None)
        if (representation is not None and self.model_meta is not None
                and representation != build_representation(self.model_meta)):
            raise ValueError("Normalizer representation differs from the model action representation")

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        raw = self.raw_dataset[index]
        return self._normalize_sample(raw)

    def _normalize_sample(self, raw):
        from starVLA.dataloader.umi_pose import CURRENT_EEF_RELATIVE, actions_to_current_frame
        sample = dict(raw)
        action = raw["action"]
        if self.action_pose_repr == CURRENT_EEF_RELATIVE:
            action = actions_to_current_frame(raw["state"], action,
                audit_values=getattr(self.raw_dataset, "audit_window_contract", True))
        if getattr(self.raw_dataset, "verify_quaternion_continuity", False):
            from .umi_quaternion import continuity_report
            current = np.array(raw["state"], copy=True)
            if self.action_pose_repr == CURRENT_EEF_RELATIVE:
                current[:, [3, 4, 5, 11, 12, 13]] = 0
                current[:, [6, 14]] = 1
            check = continuity_report(np.concatenate((current, action)))
            if check["negative_pairs"]:
                raise ValueError("Quaternion sign discontinuity in physical model action targets")
            if "umi_metadata" in raw:
                raw = dict(raw, umi_metadata=dict(raw["umi_metadata"], model_action_quaternion_continuity=check))
        sample["state"] = self.normalizer.normalize_state(raw["state"])
        sample["action"] = self.normalizer.normalize_action(action)
        if "umi_metadata" in raw:
            sample["umi_metadata"] = dict(raw["umi_metadata"], normalization=self.normalizer.provenance())
            if self.action_pose_repr != "world":
                sample["umi_metadata"]["action_pose_repr"] = self.action_pose_repr
        return sample

    def __getitems__(self, indices):
        batch_getter = getattr(self.raw_dataset, "__getitems__", None)
        raw = batch_getter(indices) if batch_getter is not None else [self.raw_dataset[index] for index in indices]
        return [self._normalize_sample(sample) for sample in raw]

    def provenance(self):
        raw = self.raw_dataset.provenance()
        mode = getattr(self.normalizer, "mode", "mean_std")
        return dict(raw, raw_reader_normalization="none", normalization=mode,
                    model_lowdim_space="fixed_mean_std" if mode == "mean_std" else "physical",
                    action_pose_repr=self.action_pose_repr, representation=getattr(self.normalizer, "representation", None),
                    normalization_details=self.normalizer.provenance())

    def close(self):
        self.raw_dataset.close()


def apply_umi_normalization(raw_dataset, data):
    """Resolve an explicit transform; never fit statistics at training startup."""
    from starVLA.dataloader.umi_pose import model_access_meta, validate_action_pose_repr
    mode = data.get("normalization", "none")
    action_pose_repr = validate_action_pose_repr(data.get("action_pose_repr", "world"))
    if mode == "none":
        if data.get("normalization_statistics") is not None:
            raise ValueError("normalization_statistics was supplied with normalization=none")
        if data.get("normalization_experiment_contract") is not None:
            raise ValueError("normalization_experiment_contract was supplied with normalization=none")
        if action_pose_repr == "world":
            return raw_dataset
    elif mode != "mean_std":
        raise ValueError("UMI normalization must be none or mean_std")
    path = data.get("normalization_statistics")
    if mode == "mean_std" and (not path or not isinstance(path, (str, Path))):
        raise ValueError("mean_std requires an explicit normalization_statistics file")
    purpose = data.get("normalization_purpose", "engineering")
    if purpose not in ("engineering", "formal"):
        raise ValueError("normalization_purpose must be engineering or formal")
    contract_path = data.get("normalization_experiment_contract")
    contract = None
    if contract_path is not None:
        if not isinstance(contract_path, (str, Path)):
            raise TypeError("normalization_experiment_contract must be a JSON file path")
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    from starVLA.dataloader.umi_normalization import UMINormalizer
    normalizer = UMINormalizer(
        mode, statistics_path=path, access_meta=model_access_meta(raw_dataset.meta, action_pose_repr),
        current_view_fingerprint=raw_dataset.view_fingerprint,
        purpose=purpose, experiment_contract=contract, audit_values=data.get("audit_window_contract", True),
    )
    return UMINormalizedDataset(raw_dataset, normalizer, action_pose_repr=action_pose_repr)


def make_umi_dataloader(cfg):
    """Factory used by the existing trainer; Accelerate shards the loader once."""
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from starVLA.dataloader.umi_sampler import UMIBlockShuffleSampler

    data = cfg.datasets.vla_data
    for key, expected in (("action_dim", 16), ("state_dim", 16), ("action_horizon", 16)):
        if int(cfg.framework.action_model[key]) != expected:
            raise ValueError(f"Indexed UMI requires framework.action_model.{key}={expected}")
    options = {key: data[key] for key in (
        "source_root", "video_root", "allowed_video_roots", "image_size", "row_cache_mib", "row_cache_items", "file_cache_items",
        "video_cache_items", "episode_cache_items", "source_identity", "decode_tolerance_seconds",
        "return_metadata", "performance_diagnostics", "video_time_policy", "boundary_slack_seconds",
        "camera_keys", "quaternion_signs", "language_source", "batched_video_decode",
        "batched_video_max_span_seconds", "anchor_stride_frames",
        "first_anchor_tolerance_seconds", "language_window_policy", "excluded_episode_indices", "verify_quaternion_continuity", "audit_window_contract") if key in data and data[key] is not None}
    dataset = UMIIndexedDataset(data.index_dir, **options)
    dataset = apply_umi_normalization(dataset, data)
    sampler = UMIBlockShuffleSampler(len(dataset), block_size=int(data.get("shuffle_block_size", 4096)),
                                     interleave_blocks=int(data.get("shuffle_interleave_blocks", 1)),
                                     seed=int(data.get("seed", 42)), shuffle=data.get("shuffle", True))
    workers = int(data.get("num_workers", 2))
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    kwargs = dict(dataset=dataset, batch_size=int(data.get("per_device_batch_size", 1)),
                  sampler=sampler, collate_fn=collate_umi_samples, num_workers=workers,
                  pin_memory=data.get("pin_memory", False), drop_last=data.get("drop_last", True),
                  generator=torch.Generator().manual_seed(int(data.get("seed", 42))))
    if workers:
        kwargs.update(persistent_workers=data.get("persistent_workers", True),
                      prefetch_factor=int(data.get("prefetch_factor", 2)),
                      multiprocessing_context=data.get("multiprocessing_context", "spawn"))
    loader = DataLoader(**kwargs)
    if data.get("write_access_record", True) and (not dist.is_initialized() or dist.get_rank() == 0):
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        record = dict(dataset.provenance(), sampler=sampler.state_dict(),
                      num_workers=workers, per_device_batch_size=kwargs["batch_size"],
                      drop_last=kwargs["drop_last"],
                      resume_status="sampler API available; trainer checkpoint integration is separate")
        temporary = output / "dataset_access.json.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "dataset_access.json")
    return loader
