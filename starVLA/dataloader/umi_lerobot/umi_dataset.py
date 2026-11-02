"""UMI LeRobot v3.0 sample dataset for StarVLA.

Each sample is one (episode, anchor) window: anchor frame is the observation,
the following ``action_horizon`` frames define the action chunk. All window
rows come from a single episode.

Action space (per arm, 8 dims): chunk-anchored relative EEF pose.
    [ R_t^T (p_{t+k} - p_t) (3), quat(R_t^T R_{t+k}) (4), gripper_{t+k} (1) ]
Rotation stays quaternion, canonicalized to one hemisphere (largest-magnitude
component positive) because q and -q are the same rotation.

State (per arm, 8 dims): absolute anchor pose [pos(3), quat(4), gripper(1)].
Bimanual mode concatenates robot1 (left) and robot2 (right) blocks -> 16 dims.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from starVLA.dataloader.umi_lerobot.episode_index import CAMERA_ORDER, EpisodeTable

logger = logging.getLogger(__name__)

ARM_POSE_KEYS = {"left": "observation.umi.robot1_finger_eef_pose", "right": "observation.umi.robot2_finger_eef_pose"}
ARM_GRIP_KEYS = {"left": "observation.umi.robot1_sensor_magnetic_encoder", "right": "observation.umi.robot2_sensor_magnetic_encoder"}

_WINDOW_COLUMNS = [
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "annotation.valid",
    ARM_POSE_KEYS["left"],
    ARM_POSE_KEYS["right"],
    ARM_GRIP_KEYS["left"],
    ARM_GRIP_KEYS["right"],
]

ARM_MODES = ("bimanual", "single_left", "single_right")
STATE_MODES = ("absolute",)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw convention, a/b are [..., 4]."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., :3] *= -1.0
    return out


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by unit quaternions q (xyzw): v + 2w(u×t)+... closed form."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_canonicalize(q: np.ndarray) -> np.ndarray:
    """Pick one representative per ±q pair: the largest-magnitude component is positive."""
    q = np.asarray(q, dtype=np.float64)
    lead = np.argmax(np.abs(q), axis=-1)
    sign = np.take_along_axis(q, lead[..., None], axis=-1)
    return q * np.where(sign < 0, -1.0, 1.0)


def _to_float_array(values, shape) -> np.ndarray:
    """list-of-lists (possibly with None) -> float array with NaN for missing."""
    out = np.full(shape, np.nan, dtype=np.float64)
    for i, row in enumerate(values):
        if row is None:
            continue
        if np.isscalar(row):
            out[i] = row
        else:
            for j, v in enumerate(row):
                if v is not None:
                    out[i, j] = v
    return out


class _BoundedLRU:
    def __init__(self, max_items: int, close=None):
        self.max_items, self.close = max_items, close
        self.items: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self.items:
            return None
        value = self.items.pop(key)
        self.items[key] = value
        return value

    def put(self, key, value):
        if key in self.items:
            del self.items[key]
        while len(self.items) >= self.max_items:
            _, evicted = self.items.popitem(last=False)
            if self.close is not None:
                self.close(evicted)
        self.items[key] = value
        return value

    def clear(self):
        while self.items:
            _, evicted = self.items.popitem()
            if self.close is not None:
                self.close(evicted)


class UMILerobotDataset:
    """Map each integer to one relative-EEF action-chunk sample from a UMI v3.0 dataset."""

    def __init__(
        self,
        data_root_dir,
        action_horizon: int = 16,
        image_size=(224, 224),
        views: list[str] | None = None,
        arm_mode: str = "bimanual",
        load_video: bool = True,
        video_workers_cache: int = 8,
        parquet_cache_items: int = 16,
        row_cache_mib: int = 128,
        num_retry: int = 20,
        seed: int = 42,
        # Wrist GoPros record at ~29.915 fps (VFR), so their 33.43 ms frame
        # interval exceeds the 2x16.67 ms acceptance window of a 0.5-frame
        # tolerance; targets drifting to mid-interval (phase grows with video
        # length) then fail spuriously. 1.0 frame (33.3 ms) still returns the
        # nearest frame to the target timestamp.
        decode_tolerance_frames: float = 1.0,
        cache_dir=None,
        strict: bool = False,
        frame_stride: int = 1,
        anchor_stride: int = 1,
    ):
        if arm_mode not in ARM_MODES:
            raise ValueError(f"arm_mode must be one of {ARM_MODES}")
        self.root = Path(data_root_dir)
        self.horizon = int(action_horizon)
        self.stride = int(frame_stride)
        if self.stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.anchor_stride = int(anchor_stride)
        if self.anchor_stride < 1:
            raise ValueError("anchor_stride must be >= 1")
        self.image_size = tuple(int(x) for x in image_size)
        self.arm_mode = arm_mode
        self.load_video = bool(load_video)
        self.num_retry = int(num_retry)
        self.seed = int(seed)
        self.strict = bool(strict)
        self.replacement_count = 0
        self.decode_tolerance = float(decode_tolerance_frames) / 30.0  # seconds, dataset is 30 fps
        self.table = EpisodeTable(
            self.root,
            self.horizon,
            cache_dir=cache_dir,
            frame_stride=self.stride,
            anchor_stride=self.anchor_stride,
        )

        arms = {"left": ARM_POSE_KEYS["left"], "right": ARM_POSE_KEYS["right"]}
        if self.arm_mode == "bimanual":
            self.arms = ("left", "right")
        elif self.arm_mode == "single_left":
            self.arms = ("left",)
        else:
            self.arms = ("right",)
        self._pose_keys = tuple(arms[arm] for arm in self.arms)
        self._grip_keys = tuple(ARM_GRIP_KEYS[arm] for arm in self.arms)
        self.action_dim = 8 * len(self.arms)
        self.state_dim = 8 * len(self.arms)
        if views is None:
            self.view_names = list(CAMERA_ORDER)
        else:
            requested = [str(name) for name in views]
            if not requested:
                raise ValueError("views must not be empty (omit the key for all cameras)")
            unknown = sorted(set(requested) - set(CAMERA_ORDER))
            if unknown:
                raise ValueError(f"unknown UMI views {unknown}; valid: {list(CAMERA_ORDER)}")
            if len(set(requested)) != len(requested):
                raise ValueError(f"duplicate UMI views: {requested}")
            self.view_names = requested
        self.epoch = 0

        self._pid = None
        self._readers = self._row_groups = self._videos = None
        self._row_cache_bytes = int(row_cache_mib) * 1024**2
        self._row_cache_used = 0
        self._parquet_cache_items = int(parquet_cache_items)
        self._video_cache_items = int(video_workers_cache)

    # -- worker lifecycle -------------------------------------------------

    def _ensure_worker(self):
        if self._pid == os.getpid():
            return
        self._close_handles()
        self._readers = _BoundedLRU(self._parquet_cache_items, close=lambda entry: entry[0].close())
        self._row_groups = OrderedDict()
        self._videos = _BoundedLRU(self._video_cache_items, close=lambda c: c[0].close())
        self._row_cache_used = 0
        self._pid = os.getpid()

    def _close_handles(self):
        for cache in (self._readers, self._videos):
            if cache is not None:
                cache.clear()
        self._readers = self._row_groups = self._videos = None
        self._row_cache_used = 0
        self._pid = None

    def close(self):
        self._close_handles()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_pid", "_readers", "_row_groups", "_videos"):
            state[key] = None
        state["_row_cache_used"] = 0
        return state

    # -- window indexing --------------------------------------------------

    def __len__(self) -> int:
        return self.table.total_windows()

    def _candidate(self, index: int, attempt: int) -> int:
        if attempt == 0:
            return index % max(len(self), 1)
        rng = np.random.default_rng([self.seed, self.epoch, index, attempt])
        return int(rng.integers(len(self)))

    # -- low-dim reading --------------------------------------------------

    def _reader(self, path: Path):
        import pyarrow.parquet as pq

        cached = self._readers.get(path)
        if cached is not None:
            return cached
        reader = pq.ParquetFile(path)
        missing = set(_WINDOW_COLUMNS) - set(reader.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        ends = np.cumsum([reader.metadata.row_group(i).num_rows for i in range(reader.num_row_groups)], dtype=np.int64)
        entry = (reader, ends)
        return self._readers.put(path, entry)

    def _read_rows(self, path: Path, begin: int, end: int):
        import pyarrow as pa

        reader, ends = self._reader(path)
        total = int(ends[-1])
        if not 0 <= begin < end <= total:
            raise ValueError(f"row range [{begin},{end}) outside {path} ({total} rows)")
        pieces = []
        for group in range(
            int(np.searchsorted(ends, begin, side="right")),
            int(np.searchsorted(ends, end - 1, side="right")) + 1,
        ):
            table = self._row_groups.get((str(path), group))
            if table is None:
                table = reader.read_row_group(group, columns=_WINDOW_COLUMNS, use_threads=False)
                size = table.nbytes
                while self._row_groups and self._row_cache_used + size > self._row_cache_bytes:
                    _, evicted = self._row_groups.popitem(last=False)
                    self._row_cache_used -= evicted.nbytes
                self._row_groups[(str(path), group)] = table
                self._row_cache_used += size
            group_start = int(ends[group - 1]) if group else 0
            a, b = max(begin, group_start) - group_start, min(end, int(ends[group])) - group_start
            pieces.append(table.slice(a, b - a))
        return pa.concat_tables(pieces)

    def read_window(self, index: int):
        """Validated raw window: poses [H+1, n_arms, 7], grips [H+1, n_arms], plus trace.

        The window is anchor frame t plus the frames t+stride, ..., t+stride*H
        (frame_stride subsamples the 30 fps stream; stride=1 is native 30 fps).
        All rows in the span are read, but only the strided frames are used
        and validated.
        """
        self._ensure_worker()
        position, anchor = self.table.locate(index)
        span = self.stride * self.horizon
        row_begin = self.table.file_row_begin(position) + anchor
        table = self._read_rows(self.table.data_path(position), row_begin, row_begin + span + 1)

        episode_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        task_ids = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        valid = np.asarray([bool(v) if v is not None else False for v in table["annotation.valid"].to_pylist()])

        n = self.horizon + 1
        poses = np.stack(
            [_to_float_array(table[key].to_pylist(), (span + 1, 7)) for key in self._pose_keys], axis=1
        )[:: self.stride]
        grips = np.stack(
            [_to_float_array(table[key].to_pylist(), (span + 1, 1)) for key in self._grip_keys], axis=1
        )[:: self.stride]

        expected_episode = int(self.table.episode_index[position])
        if len(table) != span + 1 or not np.all(episode_ids == expected_episode):
            raise ValueError(f"window rows do not belong to episode {expected_episode}")
        if not np.all(np.diff(frames) == 1):
            raise ValueError("frame_index is not contiguous inside the window")
        if not np.all(valid[:: self.stride]):
            raise ValueError("annotation.valid is False inside the window")
        if not (np.isfinite(poses).all() and np.isfinite(grips).all()):
            raise ValueError("NaN/Inf in pose or gripper signals")
        if not np.isfinite(timestamps).all():
            raise ValueError("NaN/Inf in timestamps")
        quat_norm = np.linalg.norm(poses[..., 3:7], axis=-1)
        if not np.all(np.abs(quat_norm - 1.0) < 1e-3):
            raise ValueError("quaternion is not unit norm")
        if not len(set(task_ids[:: self.stride].tolist())) == 1:
            raise ValueError("task_index changes inside the window")

        tasks = self.table.tasks[position]
        task_id = int(task_ids[0])
        if tasks and 0 <= task_id < len(tasks) and tasks[task_id].strip():
            lang = tasks[task_id]
        elif tasks and tasks[0].strip():
            lang = tasks[0]
        else:
            raise ValueError(f"no language for task_index {task_id} in episode {expected_episode}")

        return {
            "poses": poses,
            "grips": grips,
            "lang": lang,
            "timestamp": float(timestamps[0]),
            "position": position,
            "anchor": anchor,
        }

    # -- action/state derivation -----------------------------------------

    def _arm_action_state(self, poses: np.ndarray, grips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """poses [H+1, 7], grips [H+1, 1] -> action [H, 8], state [8]."""
        q = poses[:, 3:7]
        p = poses[:, :3]
        q_anchor_inv = np.broadcast_to(quat_conjugate(q[0]), (self.horizon, 4))
        delta_p = p[1:] - p[0]
        rel_p = quat_rotate(q_anchor_inv, delta_p)
        rel_q = quat_multiply(q_anchor_inv, q[1:])
        rel_q = quat_canonicalize(rel_q)
        action = np.concatenate([rel_p, rel_q, grips[1:]], axis=1)
        state = np.concatenate([p[0], quat_canonicalize(q[0][None])[0], grips[0]], axis=0)
        return action.astype(np.float32), state.astype(np.float32)

    def make_lowdim(self, window) -> tuple[np.ndarray, np.ndarray]:
        actions, states = [], []
        for arm in range(window["poses"].shape[1]):
            action, state = self._arm_action_state(window["poses"][:, arm, :], window["grips"][:, arm, :])
            actions.append(action)
            states.append(state)
        return np.concatenate(actions, axis=1), np.concatenate(states, axis=0)[None, :]

    # -- video decoding ---------------------------------------------------

    def _video_frame(self, position: int, camera: str, row_timestamp: float) -> Image.Image:
        import av

        container_stream = self._videos.get((position, camera))
        if container_stream is None:
            path = self.table.video_path(position, camera)
            container = av.open(str(path))
            stream = container.streams.video[0]
            stream.thread_count = 1
            container_stream = (container, stream)
            self._videos.put((position, camera), container_stream)
        container, stream = container_stream

        window_start = float(getattr(self.table, f"{camera}_from_timestamp")[position])
        window_end = float(getattr(self.table, f"{camera}_to_timestamp")[position])
        target = window_start + row_timestamp
        if not window_start - 1e-6 <= target <= window_end + 1e-6:
            raise ValueError(f"timestamp {target:.4f}s outside {camera} window [{window_start:.4f},{window_end:.4f}]")

        time_base = float(stream.time_base)
        container.seek(int(target / time_base), stream=stream, backward=True, any_frame=False)
        best, best_gap = None, None
        for frame in container.decode(video=0):
            frame_time = float(frame.pts) * time_base
            gap = abs(frame_time - target)
            if frame_time > target + self.decode_tolerance:
                break
            if best_gap is None or gap < best_gap:
                best, best_gap = frame, gap
        if best is None or best_gap is None or best_gap > self.decode_tolerance:
            raise ValueError(f"no decoded frame within tolerance for {camera} at {target:.4f}s")
        image = best.to_ndarray(format="rgb24")
        return Image.fromarray(image).resize(self.image_size)

    # -- dataset interface ------------------------------------------------

    def read_lowdim(self, index: int):
        """Fast path without video decoding (used by the statistics script)."""
        window = self.read_window(index)
        action, state = self.make_lowdim(window)
        return {"state": state, "action": action, "lang": window["lang"]}

    def __getitem__(self, index: int):
        last_error = None
        for attempt in range(self.num_retry + 1):
            candidate = self._candidate(index, attempt)
            try:
                window = self.read_window(candidate)
                action, state = self.make_lowdim(window)
                sample = {
                    "lang": window["lang"],
                    "state": state,
                    "action": action,
                    "action_mask": np.ones_like(action, dtype=bool),
                    "robot_tag": "new_embodiment",
                    "view_names": list(self.view_names),
                }
                if self.load_video:
                    sample["image"] = [
                        self._video_frame(window["position"], camera, window["timestamp"])
                        for camera in self.view_names
                    ]
                return sample
            except (ValueError, TypeError, KeyError, OSError, IndexError, RuntimeError) as error:
                last_error = error
                if self.strict:
                    raise RuntimeError(
                        f"UMI window {index} invalid (attempt {attempt}, candidate {candidate}): {error}"
                    ) from error
                self.replacement_count += 1
                if self.replacement_count <= 10 or self.replacement_count % 1000 == 0:
                    logger.warning(
                        "[umi_lerobot] window %d replaced by random candidate %d (attempt %d): %s "
                        "[worker replacements: %d]",
                        index,
                        candidate,
                        attempt,
                        error,
                        self.replacement_count,
                    )
        raise RuntimeError(
            f"UMI window {index} invalid after {self.num_retry + 1} attempts: {last_error}"
        ) from last_error

    # -- provenance ---------------------------------------------------------

    def save_dataset_statistics(self, output_path):
        from pathlib import Path as _P

        output_path = _P(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dataset_type": "umi_lerobot",
            "data_root_dir": str(self.root),
            "num_episodes": self.table.num_episodes,
            "num_windows": len(self),
            "kept_filter": bool(getattr(self.table, "kept_filter_applied", False)),
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "action_horizon": self.horizon,
            "frame_stride": self.stride,
            "anchor_stride": self.anchor_stride,
            "arm_mode": self.arm_mode,
            "image_size": list(self.image_size),
            "view_names": self.view_names,
            "normalization": "external",
        }
        import json

        output_path.write_text(json.dumps(record, indent=2))
        logger.info(f"[umi_lerobot] dataset statistics written to {output_path}")
        return record
