"""Indexed DROID controller goals and three views with bounded video decoding."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .droid_contract import contract, relative_actions
from .umi_normalization import content_fingerprint
from .umi_video_batch import decode_video_requests
from .umi_video_time import VideoTimeWindow


class EpisodeBlockSampler(Sampler):
    """Shuffle <=64-anchor blocks inside ranges; interleave 8 shuffled blocks.

    No block crosses an episode/range. Short tails remain. Every anchor appears
    once per epoch; Accelerate alone shards this global stream. A DataLoader may
    separately drop its final incomplete batch, as recorded in dataset_access.
    """
    def __init__(self, counts, *, block_size=64, interleave_blocks=8, seed=42):
        if block_size < 1 or interleave_blocks < 1 or any(int(c) < 1 for c in counts):
            raise ValueError("Positive block size/interleave/range counts required")
        self.block_size, self.interleave_blocks, self.seed = int(block_size), int(interleave_blocks), int(seed)
        starts, lengths, offset = [], [], 0
        for count in counts:
            for local in range(0, int(count), self.block_size):
                starts.append(offset + local)
                lengths.append(min(self.block_size, int(count) - local))
            offset += int(count)
        self.starts, self.lengths = np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64)
        self.size, self.epoch = offset, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.size

    def __iter__(self):
        from .umi_sampler import _mix64
        seed = _mix64(self.seed ^ _mix64(self.epoch))
        generator = torch.Generator().manual_seed(_mix64(seed ^ 0xB10C5EED))
        order = torch.randperm(len(self.starts), generator=generator).tolist()
        for begin in range(0, len(order), self.interleave_blocks):
            group = []
            for block in order[begin:begin + self.interleave_blocks]:
                local = torch.Generator().manual_seed(_mix64(seed ^ _mix64(block)))
                group.append((int(self.starts[block]), torch.randperm(int(self.lengths[block]), generator=local).tolist()))
            for t in range(max(len(indices) for _, indices in group)):
                for start, indices in group:
                    if t < len(indices):
                        yield start + indices[t]

    def state_dict(self):
        return dict(version="droid-episode-block-v1", size=self.size, blocks=len(self.starts),
                    block_size=self.block_size, interleave_blocks=self.interleave_blocks,
                    seed=self.seed, epoch=self.epoch,
                    ranges_sha256=hashlib.sha256(self.starts.tobytes() + self.lengths.tobytes()).hexdigest())


class DROIDIndexedDataset(Dataset):
    def __init__(self, index_dir, *, split="train", horizon=16, stride=3, image_size=(224, 224),
                 video_cache_items=6, lowdim_cache_items=2, return_metadata=False):
        self.root = Path(index_dir).resolve()
        ready = json.loads((self.root / "READY").read_text())
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.stats = json.loads((self.root / "statistics.json").read_text())
        identity = {k: v for k, v in self.manifest.items() if k not in ("fingerprint", "counts", "catalog_sha256")}
        fingerprint = content_fingerprint(identity)
        if (fingerprint != ready["fingerprint"] or fingerprint != self.manifest["fingerprint"]
                or fingerprint != self.stats["fingerprint"]
                or hashlib.sha256((self.root / "statistics.json").read_bytes()).hexdigest() != ready["statistics_sha256"]
                or hashlib.sha256((self.root / "catalog.sqlite").read_bytes()).hexdigest() != self.manifest["catalog_sha256"]):
            raise ValueError("DROID index/statistics fingerprint or integrity mismatch")
        if self.manifest["contract"] != contract(horizon, stride):
            raise ValueError("DROID horizon/stride/action contract differs from offline index")
        if split not in ("train", "val") or video_cache_items < 1 or lowdim_cache_items < 1:
            raise ValueError("Invalid split or cache limits")
        self.source = Path(self.manifest["source_root"])
        self.split, self.horizon, self.stride = split, int(horizon), int(stride)
        self.image_size = tuple(map(int, image_size))
        self.video_cache_items, self.lowdim_cache_items = int(video_cache_items), int(lowdim_cache_items)
        self.return_metadata = bool(return_metadata)
        self.mean = np.asarray(self.stats["action"]["mean"], dtype=np.float32)
        self.scale = np.asarray(self.stats["action"]["std"], dtype=np.float32)
        if (self.mean.shape != (16,) or self.scale.shape != (16,) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)
                or self.stats["fit_split"] != "train" or self.stats["clipping"]):
            raise ValueError("Invalid DROID fixed mean/std")
        db = self._connect()
        rows = db.execute("SELECT episode, first_anchor, count, language FROM ranges WHERE split=? ORDER BY episode, first_anchor", (split,)).fetchall()
        db.close()
        self.episodes = np.asarray([r[0] for r in rows], dtype=np.int64)
        self.firsts = np.asarray([r[1] for r in rows], dtype=np.int64)
        self.counts = np.asarray([r[2] for r in rows], dtype=np.int64)
        self.languages = [r[3] for r in rows]
        self.ends = np.cumsum(self.counts)
        if not len(rows):
            raise ValueError(f"DROID {split} index is empty")
        self._pid, self._db = None, None
        self._videos, self._lowdim, self._episodes = OrderedDict(), OrderedDict(), OrderedDict()

    def _connect(self):
        return sqlite3.connect(f"file:{self.root / 'catalog.sqlite'}?mode=ro&immutable=1", uri=True)

    def _worker(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
            self._db = self._connect()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None, _db=None, _videos=OrderedDict(), _lowdim=OrderedDict(), _episodes=OrderedDict())
        return state

    def close(self):
        for container, _ in getattr(self, "_videos", {}).values():
            container.close()
        if getattr(self, "_db", None) is not None:
            self._db.close()
        self._db = None
        self._pid = None
        self._videos, self._lowdim, self._episodes = OrderedDict(), OrderedDict(), OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def _episode(self, episode):
        if episode not in self._episodes:
            row = self._db.execute("SELECT length, shard, offset, cameras FROM episodes WHERE episode=? AND split=?", (episode, self.split)).fetchone()
            if row is None:
                raise ValueError(f"Unknown episode {episode}")
            self._episodes[episode] = (row[0], row[1], row[2], json.loads(row[3]))
            if len(self._episodes) > 128:
                self._episodes.popitem(last=False)
        self._episodes.move_to_end(episode)
        return self._episodes[episode]

    def read_lowdim(self, index):
        self._worker()
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        r = int(np.searchsorted(self.ends, index, side="right"))
        local = index - (int(self.ends[r-1]) if r else 0)
        anchor = int(self.firsts[r]) + local * self.stride
        episode = int(self.episodes[r])
        length, shard, offset, cameras = self._episode(episode)
        if shard not in self._lowdim:
            rows = np.load(self.root / (shard + ".npy"), mmap_mode="r", allow_pickle=False)
            times = np.load(self.root / (shard + "-time.npy"), mmap_mode="r", allow_pickle=False)
            self._lowdim[shard] = (rows, times)
            if len(self._lowdim) > self.lowdim_cache_items:
                self._lowdim.popitem(last=False)
        self._lowdim.move_to_end(shard)
        rows, times = self._lowdim[shard]
        physical = relative_actions(rows[offset:offset + length], [anchor], self.horizon)[0]
        action = np.zeros((self.horizon, 16), dtype=np.float32)
        action[:, :8] = (physical - self.mean[:8]) / self.scale[:8]
        if not np.isfinite(action).all():
            raise ValueError(f"Non-finite normalized target: episode={episode}, anchor={anchor}")
        mask = np.zeros_like(action, dtype=bool)
        mask[:, :8] = True
        sample = dict(action=action, action_mask=mask, lang=self.languages[r])
        trace = dict(index=index, episode=episode, anchor=anchor, timestamp=float(times[offset + anchor]),
                     command_frame_range=[anchor, anchor + self.horizon - 1])
        return sample, cameras, trace

    def _open_video(self, camera):
        import av
        path = camera["video_path"]
        if path not in self._videos:
            container = av.open(str(self.source / path))
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            stream.codec_context.thread_count = 1
            self._videos[path] = (container, stream)
            if len(self._videos) > self.video_cache_items:
                old, _ = self._videos.popitem(last=False)[1]
                old.close()
        self._videos.move_to_end(path)
        return self._videos[path]

    def _window(self, camera, timestamp, stream):
        return VideoTimeWindow(camera["from_timestamp"], camera["to_timestamp"], timestamp,
                               stream.time_base, 1 / 15 / 2 + .001,
                               policy="engineering_boundary_slack", boundary_slack_seconds=.001)

    def _render(self, camera, window, stream, frame, pts):
        from PIL import Image
        image = frame.to_image().convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
        timing = dict(camera=camera["camera"], video_path=camera["video_path"],
                      target_seconds=float(window.target), frame_seconds=float(pts), error_seconds=float(pts-window.target))
        return image, timing

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        raw = [self.read_lowdim(index) for index in indices]
        requests = [((i, view), camera, trace["timestamp"])
                    for i, (_, cameras, trace) in enumerate(raw) for view, camera in enumerate(cameras)]
        try:
            decoded = decode_video_requests(requests, open_video=self._open_video, make_window=self._window,
                                            render_frame=self._render, max_span_seconds=2)
        except Exception as error:
            raise RuntimeError(f"DROID video decode failed for indices={list(indices)}; no replacement sampling") from error
        result = []
        for i, (sample, cameras, trace) in enumerate(raw):
            pairs = [decoded[(i, view)] for view in range(len(cameras))]
            sample["image"] = [pair[0] for pair in pairs]
            if self.return_metadata:
                sample["droid_metadata"] = dict(trace, video_decode=[pair[1] for pair in pairs])
            result.append(sample)
        return result

    def provenance(self):
        return dict(backend="droid_indexed", index_dir=str(self.root), fingerprint=self.manifest["fingerprint"],
                    split=self.split, samples=len(self), counts=self.manifest["counts"], contract=self.manifest["contract"],
                    statistics=str(self.root / "statistics.json"), normalization="train mean/std, FP32, no clipping",
                    image_size=list(self.image_size), batched_video_max_span_seconds=2,
                    video_cache_items=self.video_cache_items, lowdim_cache_items=self.lowdim_cache_items,
                    video_time_policy="engineering_boundary_slack=1ms; nearest PTS tolerance=half native frame + 1ms")


def collate_droid(batch):
    return batch


def make_droid_dataloader(cfg):
    import torch.distributed as dist
    data, model = cfg.datasets.vla_data, cfg.framework.action_model
    if int(model.action_dim) != 16 or bool(model.get("use_state", False)):
        raise ValueError("DROID v1 requires action_dim=16, use_state=false")
    dataset = DROIDIndexedDataset(data.index_dir, split=data.get("split", "train"), horizon=int(model.action_horizon),
                                  stride=int(data.get("anchor_stride_frames", 3)), image_size=data.get("image_size", [224, 224]),
                                  video_cache_items=int(data.get("video_cache_items", 6)),
                                  lowdim_cache_items=int(data.get("lowdim_cache_items", 2)),
                                  return_metadata=data.get("return_metadata", False))
    sampler = EpisodeBlockSampler(dataset.counts, block_size=int(data.get("shuffle_block_size", 64)),
                                  interleave_blocks=int(data.get("shuffle_interleave_blocks", 8)), seed=int(data.get("seed", 42)))
    workers = int(data.get("num_workers", 4))
    kwargs = dict(dataset=dataset, batch_size=int(data.per_device_batch_size), sampler=sampler, collate_fn=collate_droid,
                  num_workers=workers, pin_memory=False, drop_last=bool(data.get("drop_last", True)),
                  generator=torch.Generator().manual_seed(int(data.get("seed", 42))))
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=int(data.get("prefetch_factor", 2)), multiprocessing_context="spawn")
    loader = DataLoader(**kwargs)
    if not dist.is_initialized() or dist.get_rank() == 0:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        record = dict(dataset.provenance(), sampler=sampler.state_dict(), num_workers=workers,
                      per_device_batch_size=kwargs["batch_size"], drop_last=kwargs["drop_last"],
                      sharding="Accelerate only", resume="trainer batches_consumed + epoch replay; fixed world size and config")
        (output / "dataset_access.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        (output / "dataset_statistics.json").write_text(json.dumps(dataset.stats, indent=2) + "\n")
    return loader
