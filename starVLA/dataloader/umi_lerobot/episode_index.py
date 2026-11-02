"""Episode-level index over a UMI LeRobot v3.0 dataset.

Scans meta/episodes/*.parquet once into a compact table (data file, row
range, per-camera video files, task texts) and exposes a virtual window
space: one integer index per valid (episode, anchor) pair. Windows never
cross an episode. When ``selection/kept_episodes.parquet`` exists (curated
releases), episodes are filtered to that membership: shared data/video
files may still contain excluded episodes' rows, so membership must be
enforced before any row access. The table is cached under ~/.cache and
rebuilt only when the source fingerprint changes (meta files, the kept
manifest, and the excluded manifest whose row ranges anchor shared data
files that begin with excluded episodes' rows).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

CAMERA_ORDER = ("head_left", "head_right", "wrist_left", "wrist_right")

_META_SCAN_COLUMNS = [
    "episode_index",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "tasks",
]
for _cam in CAMERA_ORDER:
    for _suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
        _META_SCAN_COLUMNS.append(f"videos/observation.images.{_cam}/{_suffix}")


def _meta_files(root: Path) -> list[str]:
    return sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))


def _scan_one(path: str) -> list[dict]:
    return pq.read_table(path, columns=_META_SCAN_COLUMNS).to_pylist()


KEPT_MANIFEST = Path("selection") / "kept_episodes.parquet"


def _fingerprint(root: Path, num_files: int, frame_stride: int = 1, anchor_stride: int = 1) -> str:
    stat = os.stat(root / "meta" / "info.json")
    payload = {
        "root": str(root),
        "meta_files": num_files,
        "cache_format": 3,
        "frame_stride": int(frame_stride),
        "anchor_stride": int(anchor_stride),
        "info_mtime_ns": stat.st_mtime_ns,
        "info_size": stat.st_size,
        "columns": _META_SCAN_COLUMNS,
    }
    kept = root / KEPT_MANIFEST
    if kept.exists():
        kept_stat = os.stat(kept)
        payload["kept_mtime_ns"] = kept_stat.st_mtime_ns
        payload["kept_size"] = kept_stat.st_size
    excluded = root / "selection" / "excluded_episodes.parquet"
    if excluded.exists():
        excl_stat = os.stat(excluded)
        payload["excluded_mtime_ns"] = excl_stat.st_mtime_ns
        payload["excluded_size"] = excl_stat.st_size
    return json.dumps(payload, sort_keys=True)


class EpisodeTable:
    """Immutable episode directory plus window cumulative counts.

    The virtual window space is parameterized by ``frame_stride`` (temporal
    gap between frames inside one action chunk) and ``anchor_stride`` (gap
    between consecutive window anchors; 1 anchors every frame, 6 samples
    one anchor per 5 Hz step).
    """

    def __init__(
        self,
        data_root_dir,
        action_horizon: int,
        cache_dir=None,
        frame_stride: int = 1,
        anchor_stride: int = 1,
    ):
        self.root = Path(data_root_dir)
        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Not a LeRobot v3.0 dataset root: {self.root}")
        self.horizon = int(action_horizon)
        if self.horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        self.stride = int(frame_stride)
        if self.stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.anchor_stride = int(anchor_stride)
        if self.anchor_stride < 1:
            raise ValueError("anchor_stride must be >= 1")

        cache_path = self._cache_path(cache_dir)
        loaded = False
        if cache_path is not None and cache_path.exists():
            try:
                self._load_cache(cache_path)
                loaded = True
                logger.info(f"[umi_lerobot] episode table loaded from cache: {cache_path}")
            except Exception as error:
                logger.warning(f"[umi_lerobot] ignoring stale cache {cache_path}: {error}")

        if not loaded:
            self._build()
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self._write_cache(cache_path)
                except OSError as error:
                    logger.warning(f"[umi_lerobot] could not write cache {cache_path}: {error}")
        self._cumulative = np.cumsum(self.anchors())

    @staticmethod
    def _compute_file_starts(rows: list[dict]) -> dict:
        """dataset_from_index is global; store each data file's global start row.

        A file's parquet row i then corresponds to global row file_start + i.
        meta/episodes only lists kept episodes, so the min over meta alone
        overestimates the base whenever a file begins with excluded episodes'
        rows; _merge_excluded_file_starts closes those holes.
        """
        starts = {}
        for r in rows:
            key = (int(r["data/chunk_index"]), int(r["data/file_index"]))
            start = int(r["dataset_from_index"])
            if key not in starts or start < starts[key]:
                starts[key] = start
        return starts

    def _merge_excluded_file_starts(self):
        """Fold excluded episodes' row ranges into the per-file base rows.

        Curated releases drop excluded episodes from meta but their rows stay
        at the start of shared data files. Their dataset_from_index (from
        selection/excluded_episodes.parquet) is the missing lower bound; the
        data file's base is the min over kept and excluded episodes alike.
        """
        path = self.root / "selection" / "excluded_episodes.parquet"
        if not path.exists():
            return
        table = pq.read_table(str(path), columns=["dataset_from_index", "data_path"])
        pattern = re.compile(r"chunk-(\d+)/file-(\d+)\.parquet")
        merged = 0
        for start, dpath in zip(table["dataset_from_index"].to_pylist(), table["data_path"].to_pylist()):
            match = pattern.search(str(dpath))
            if match is None:
                continue
            key = (int(match.group(1)), int(match.group(2)))
            start = int(start)
            if key not in self.file_starts or start < self.file_starts[key]:
                self.file_starts[key] = start
                merged += 1
        if merged:
            logger.info(f"[umi_lerobot] merged {merged} excluded-episode bases into file_starts")

    def file_row_begin(self, position: int) -> int:
        """File-local row offset of the episode's first frame."""
        key = (int(self.data_chunk[position]), int(self.data_file[position]))
        return int(self.row_start[position]) - self.file_starts[key]

    def _cache_path(self, cache_dir) -> Path | None:
        base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "starVLA" / "umi_lerobot"
        files = _meta_files(self.root)
        if not files:
            raise FileNotFoundError(f"No meta/episodes parquet files under {self.root}")
        digest = __import__("hashlib").sha256(
            _fingerprint(self.root, len(files), self.stride, self.anchor_stride).encode()
        ).hexdigest()[:16]
        return base / f"episodes_{digest}.parquet"

    def _build(self):
        files = _meta_files(self.root)
        logger.info(f"[umi_lerobot] scanning {len(files)} episode-meta files under {self.root}")
        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            for part in pool.map(_scan_one, files):
                rows.extend(part)
        if not rows:
            raise ValueError(f"Episode meta is empty under {self.root}")

        self.file_starts = self._compute_file_starts(rows)
        self._merge_excluded_file_starts()

        kept_path = self.root / KEPT_MANIFEST
        self.kept_filter_applied = kept_path.exists()
        if self.kept_filter_applied:
            kept = set(pq.read_table(str(kept_path), columns=["episode_index"])["episode_index"].to_pylist())
            before = len(rows)
            rows = [r for r in rows if int(r["episode_index"]) in kept]
            if not rows:
                raise ValueError("No episodes remain after kept-membership filter")
            if len(rows) < before:
                logger.warning(
                    f"[umi_lerobot] dropped {before - len(rows)} episodes outside selection/kept_episodes.parquet"
                )
            logger.info(f"[umi_lerobot] kept-membership filter: {len(rows)} episodes (manifest has {len(kept)})")
        else:
            logger.info("[umi_lerobot] no selection/kept_episodes.parquet; using all episodes in meta")

        rows.sort(key=lambda r: int(r["episode_index"]))
        index = np.array([int(r["episode_index"]) for r in rows], dtype=np.int64)
        if len(np.unique(index)) != len(index):
            raise ValueError("Duplicate episode_index in meta/episodes")

        self.episode_index = index
        self.length = np.array([int(r["length"]) for r in rows], dtype=np.int64)
        self.row_start = np.array([int(r["dataset_from_index"]) for r in rows], dtype=np.int64)
        self.row_end = np.array([int(r["dataset_to_index"]) for r in rows], dtype=np.int64)
        self.data_chunk = np.array([int(r["data/chunk_index"]) for r in rows], dtype=np.int32)
        self.data_file = np.array([int(r["data/file_index"]) for r in rows], dtype=np.int32)
        self.tasks = [list(r["tasks"] or []) for r in rows]
        for cam in CAMERA_ORDER:
            prefix = f"videos/observation.images.{cam}"
            for suffix in ("chunk_index", "file_index"):
                arr = np.array(
                    [int(r[f"{prefix}/{suffix}"]) for r in rows], dtype=np.int32
                )
                setattr(self, f"{cam}_{suffix}", arr)
            for suffix in ("from_timestamp", "to_timestamp"):
                arr = np.array([float(r[f"{prefix}/{suffix}"]) for r in rows], dtype=np.float64)
                setattr(self, f"{cam}_{suffix}", arr)

        if not np.array_equal(self.row_start + self.length, self.row_end):
            raise ValueError("dataset_from_index + length != dataset_to_index for some episode")
        logger.info(
            f"[umi_lerobot] episode table: {len(index)} episodes, {int(self.length.sum())} frames"
        )

    @property
    def num_episodes(self) -> int:
        return len(self.episode_index)

    def anchors(self) -> np.ndarray:
        """Per-episode number of valid action-chunk anchors.

        A window at frame anchor t uses frames t, t+stride, ..., t+stride*horizon,
        so the last index t + stride*horizon must stay inside the episode. Valid
        frame anchors are 0, anchor_stride, 2*anchor_stride, ...; the count is
        ceil(span / anchor_stride) where span = length - stride*horizon.
        """
        span = np.maximum(self.length - self.horizon * self.stride, 0)
        return (span + self.anchor_stride - 1) // self.anchor_stride

    def total_windows(self) -> int:
        return int(self.anchors().sum())

    def locate(self, index: int) -> tuple[int, int]:
        """Map a global window index to (episode table row, anchor frame)."""
        if not 0 <= index < int(self._cumulative[-1]):
            raise IndexError(f"window index {index} outside [0, {int(self._cumulative[-1])})")
        position = int(np.searchsorted(self._cumulative, index, side="right"))
        base = int(self._cumulative[position - 1]) if position else 0
        return position, (index - base) * self.anchor_stride

    def data_path(self, position: int) -> Path:
        return self.root / "data" / f"chunk-{int(self.data_chunk[position]):03d}" / f"file-{int(self.data_file[position]):03d}.parquet"

    def video_path(self, position: int, camera: str) -> Path:
        chunk = int(getattr(self, f"{camera}_chunk_index")[position])
        file = int(getattr(self, f"{camera}_file_index")[position])
        return self.root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"

    def _save_columns(self) -> dict:
        cols = {
            "episode_index": self.episode_index,
            "length": self.length,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "data_chunk": self.data_chunk,
            "data_file": self.data_file,
            "file_base_row": np.array(
                [self.file_starts[(int(c), int(f))] for c, f in zip(self.data_chunk, self.data_file)],
                dtype=np.int64,
            ),
        }
        for cam in CAMERA_ORDER:
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                cols[f"{cam}_{suffix}"] = getattr(self, f"{cam}_{suffix}")
        return cols

    def _write_cache(self, path: Path):
        import pyarrow as pa

        table = pa.table(self._save_columns() | {"tasks": self.tasks})
        tmp = path.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)

    def _load_cache(self, path: Path):
        table = pq.read_table(path)
        data = {name: table[name].to_numpy() for name in table.schema.names}
        self.episode_index = np.asarray(data["episode_index"], dtype=np.int64)
        self.length = np.asarray(data["length"], dtype=np.int64)
        self.row_start = np.asarray(data["row_start"], dtype=np.int64)
        self.row_end = np.asarray(data["row_end"], dtype=np.int64)
        self.data_chunk = np.asarray(data["data_chunk"], dtype=np.int32)
        self.data_file = np.asarray(data["data_file"], dtype=np.int32)
        self.file_starts = {}
        for c, f, s in zip(self.data_chunk, self.data_file, np.asarray(data["file_base_row"], dtype=np.int64)):
            self.file_starts[(int(c), int(f))] = int(s)
        self._merge_excluded_file_starts()
        self.tasks = table["tasks"].to_pylist()
        for cam in CAMERA_ORDER:
            for suffix in ("chunk_index", "file_index"):
                setattr(self, f"{cam}_{suffix}", np.asarray(data[f"{cam}_{suffix}"], dtype=np.int32))
            for suffix in ("from_timestamp", "to_timestamp"):
                setattr(self, f"{cam}_{suffix}", np.asarray(data[f"{cam}_{suffix}"], dtype=np.float64))
