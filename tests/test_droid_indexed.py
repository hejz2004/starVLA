"""DROID geometry, command timing, split/statistics and sampler regression tests."""
import json
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from starVLA.dataloader.droid_contract import CAMERAS, anchor_ranges, cache_poses, relative_actions
from starVLA.dataloader.droid_indexed import DROIDIndexedDataset, EpisodeBlockSampler
from starVLA.dataloader.droid_prepare import prepare, split_for_episode


def test_eef_geometry_uses_same_observation_and_same_row_command():
    n, horizon = 20, 16
    obs = np.zeros((n, 6)); obs[:, 0] = np.arange(n) * .1; obs[:, 5] = np.pi / 2
    cmd = obs.copy(); cmd[:, 1] += .05; cmd[:, 5] += .2
    rows = cache_poses(obs, cmd, np.zeros(n), np.linspace(0, 1, n))
    target = relative_actions(rows, [0, 3], horizon)
    for i, t in enumerate([0, 3]):
        expected = Rotation.from_euler("xyz", obs[t, 3:]).inv().apply(cmd[t:t+horizon, :3] - obs[t, :3])
        np.testing.assert_allclose(target[i, :, :3], expected, atol=1e-6)
        np.testing.assert_allclose(target[i, :, 7], 1 - np.linspace(0, 1, n)[t:t+horizon], atol=1e-6)
        rotation = Rotation.from_euler("xyz", obs[t, 3:]).inv() * Rotation.from_euler("xyz", cmd[t:t+horizon, 3:])
        np.testing.assert_allclose(Rotation.from_quat(target[i, :, 3:7]).as_matrix(), rotation.as_matrix(), atol=1e-6)
    # First target is the recorded same-row command, not the next observation.
    np.testing.assert_allclose(target[0, 0, :3], [.05, 0, 0], atol=1e-6)


def test_quaternion_lift_and_anchor_stride():
    obs = np.zeros((25, 6)); obs[:, 5] = np.linspace(2.8, 3.5, 25)
    # Equivalent wrapped Euler values must produce a continuous quaternion lift.
    obs[:, 5] = (obs[:, 5] + np.pi) % (2 * np.pi) - np.pi
    rows = cache_poses(obs, obs, np.zeros(25), np.zeros(25))
    assert np.all(np.sum(rows[:-1, 3:7] * rows[1:, 3:7], axis=1) > .99)
    result = relative_actions(rows, [0], 16)[0]
    np.testing.assert_allclose(result[:, 6], np.cos(np.arange(16) * (.7 / 24) / 2), atol=1e-6)
    assert anchor_ranges(25, 16, 3, ["x"] * 25) == [[0, 4, "x"]]
    assert anchor_ranges(25, 16, 3, [""] * 3 + ["x"] * 22) == [[3, 3, "x"]]


def test_episode_blocks_cover_tails_and_epoch_replay():
    counts = [5, 129, 7, 64]
    sampler = EpisodeBlockSampler(counts)
    order = list(sampler)
    assert sorted(order) == list(range(sum(counts)))
    assert list(sampler) == order
    boundaries = np.cumsum([0] + counts)
    for start, length in zip(sampler.starts, sampler.lengths):
        assert length <= 64
        assert np.searchsorted(boundaries, start, side="right") == np.searchsorted(boundaries, start + length - 1, side="right")
    sampler.set_epoch(1)
    assert list(sampler) != order
    assert sorted(sampler) == sorted(order)


def make_source(root):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    train_id = next(i for i in range(100) if split_for_episode(i, 42, .5) == "train")
    val_id = next(i for i in range(100) if split_for_episode(i, 42, .5) == "val")
    metadata, data = [], []
    for j, ep in enumerate(sorted([train_id, val_id])):
        start = len(data)
        for t in range(40):
            data.append({"index": start+t, "episode_index": ep, "frame_index": t, "timestamp": t/15,
                         "is_episode_successful": True,
                         "observation.state.cartesian_position": [0., 0., 0., 0., 0., 0.],
                         "action.cartesian_position": [float(t) / 100 + (10 if ep == val_id else 0), 0., 0., 0., 0., 0.],
                         "observation.state.gripper_position": 0., "action.gripper_position": t/40,
                         "language_instruction": "", "language_instruction_2": f"task {ep}", "language_instruction_3": "fallback"})
        meta = dict(episode_index=ep, length=40, dataset_from_index=start, dataset_to_index=start+40)
        meta.update({"data/chunk_index": 0, "data/file_index": 0})
        for key in CAMERAS:
            path = root / f"videos/{key}/chunk-000/file-000.mp4"
            path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            meta.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                         f"videos/{key}/from_timestamp": j*40/15, f"videos/{key}/to_timestamp": (j+1)*40/15})
        metadata.append(meta)
    pq.write_table(pa.Table.from_pylist(data), root / "data/chunk-000/file-000.parquet")
    # This leftover file must never be enumerated by the preparer.
    (root / "data/chunk-000/file-999.parquet").write_bytes(b"not parquet")
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "meta/info.json").write_text(json.dumps(dict(codebase_version="v3.0", fps=15, total_episodes=2, total_frames=80,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")))
    return train_id, val_id


def test_preparation_train_only_stats_mask_fingerprint(tmp_path):
    train_id, val_id = make_source(tmp_path / "source")
    output = tmp_path / "index"
    manifest = prepare(tmp_path / "source", output, val_fraction=.5)
    assert manifest["counts"]["train_anchors"] == manifest["counts"]["val_anchors"] == 9
    stats = json.loads((output / "statistics.json").read_text())
    expected = np.array([np.arange(t, t+16)/100 for t in range(0, 25, 3)])
    np.testing.assert_allclose(stats["action"]["mean"][0], expected.mean(), atol=1e-7)
    np.testing.assert_allclose(stats["action"]["std"][0], expected.std(), atol=1e-7)
    assert stats["target_appearances"] == 9*16
    dataset = DROIDIndexedDataset(output)
    sample, cameras, trace = dataset.read_lowdim(1)
    assert trace["episode"] == train_id and trace["anchor"] == 3
    assert sample["lang"] == f"task {train_id}" and len(cameras) == 3
    assert sample["action_mask"][:, :8].all() and not sample["action_mask"][:, 8:].any()
    assert np.all(sample["action"][:, 8:] == 0)
    assert "state" not in sample
    dataset.close()
    with pytest.raises(ValueError, match="contract"):
        DROIDIndexedDataset(output, stride=1)
    stats["action"]["mean"][0] += 1
    (output / "statistics.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="integrity"):
        DROIDIndexedDataset(output)
