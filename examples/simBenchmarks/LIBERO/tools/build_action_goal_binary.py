"""Build native-command goal sidecars, binary gripper labels and exact H16 stats.

Recorded state supplies each command's reference pose, never a future reached
pose target. History receipts seed the preexisting orientation goal and validate
removed-frame history; all motion targets are independently reconstructed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.libero_action_contract import (
    VERSION, LAYOUT, LOSS_MASK, NORM_MASK, OSC_SCALE, GoalBinaryNormalizer,
    reconstruct_goals, encode_goals, rotvec_to_quaternion, multiply, conjugate,
    quaternion_to_rotvec,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--history-sidecars', required=True, help='Validated source controller-history receipt')
    p.add_argument('--history-manifest', required=True)
    args = p.parse_args()
    data, output, history = Path(args.data_root), Path(args.output), Path(args.history_sidecars)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    reference = json.loads(Path(args.history_manifest).read_text())
    if reference.get('contract') != 'libero-umi-aligned-actuator-aperture-v1':
        raise ValueError('Unknown controller-history provenance')
    rows, pose_samples = [], []
    max_rotation = max_position = max_roundtrip = 0.
    total_frames = total_windows = 0
    for record in sorted(reference['records'], key=lambda a: (a['dataset'], a['episode'])):
        dataset, episode = record['dataset'], record['episode']
        source = data / dataset / 'data' / f'chunk-{episode // 1000:03d}' / f'episode_{episode:06d}.parquet'
        receipt = history / record['label_path']
        if sha(source) != record['parquet_sha256'] or sha(receipt) != record['label_sha256']:
            raise ValueError(f'Source/history fingerprint mismatch: {source}')
        table = pq.read_table(source, columns=['observation.state', 'action']).to_pydict()
        state, action = np.asarray(table['observation.state'], dtype=np.float64), np.asarray(table['action'], dtype=np.float64)
        with np.load(receipt, allow_pickle=False) as archive:
            original = archive['goals'].copy()
            recorded_native = archive['native_actions'].copy()
            indices = archive['retained_indices'].copy()
        expected_native = action.copy()
        expected_native[:, 6] = 1 - 2 * action[:, 6]
        np.testing.assert_allclose(expected_native, recorded_native, atol=5e-6, rtol=0)
        # Only seed the orientation already held before the first retained row;
        # no future reference goal is used to generate a target.
        goals = reconstruct_goals(state, action, initial_goal_quaternion=original[0, 3:7])
        position_error = float(np.max(np.abs(goals[:, :3] - original[:, :3])))
        rotation_error = float(np.max(np.linalg.norm(quaternion_to_rotvec(
            multiply(goals[:, 3:7], conjugate(original[:, 3:7]))), axis=1)))
        if position_error > 1e-10 or rotation_error > 2e-6:
            raise ValueError(f'Independent native-goal reconstruction mismatch: {source}: {position_error}, {rotation_error}')
        max_position, max_rotation = max(max_position, position_error), max(max_rotation, rotation_error)
        anchors = rotvec_to_quaternion(state[:, 3:6])
        target = output / 'labels' / dataset / f'episode_{episode:06d}.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, goals=goals, anchor_position=state[:, :3], anchor_quaternion=anchors,
            source_actions=action.astype(np.float32), source_frame_indices=indices,
            initial_orientation_goal=original[0, 3:7])
        windows = len(state) - 16 + 1
        if windows <= 0:
            raise ValueError('Episode has no complete H16 window')
        starts = np.arange(windows)
        encoded = encode_goals(goals[starts[:, None] + np.arange(16)], state[starts, :3], anchors[starts])
        assert np.isin(encoded[..., 7], [0, 1]).all() and np.all(encoded[..., 8:] == 0)
        pose_samples.append(encoded[..., :7].reshape(-1, 7))
        rows.append(dict(dataset=dataset, episode=episode, frames=len(state), windows=windows,
            source_parquet=str(source.resolve()), source_sha256=sha(source), label=str(target.relative_to(output)),
            label_sha256=sha(target), initial_history_sha256=sha(receipt),
            position_reconstruction_error_m=position_error, rotation_reconstruction_error_rad=rotation_error))
        total_frames += len(state)
        total_windows += windows
    pose = np.concatenate(pose_samples)
    low, high = np.quantile(pose, [.01, .99], axis=0)
    # Constant or nearly constant pose components still have an invertible map.
    high = np.maximum(high, low + 1e-6)
    action_stats = dict(q01=np.r_[low, np.zeros(9)].tolist(), q99=np.r_[high, np.ones(9)].tolist(),
        mean=np.r_[pose.mean(0), np.zeros(9)].tolist(), std=np.r_[pose.std(0), np.ones(9)].tolist(),
        mask=LOSS_MASK.tolist(), normalization_mask=NORM_MASK.tolist(),
        normalization='pose_q01_q99_no_clip_gripper_identity', clipping=False,
        action_type=VERSION, contract=VERSION, action_layout=LAYOUT,
        gripper_encoding='0=closed,1=open', osc_scale=OSC_SCALE.tolist())
    payload = dict(contract=VERSION, action_layout=LAYOUT, action=action_stats,
        horizon=16, timing='image[t] -> native controller goals[t:t+16]', anchor='one recorded image observation per chunk',
        source='native action commands; recorded state is command reference, not future reached target')
    normalizer = GoalBinaryNormalizer(payload)
    for values in pose_samples:
        # Verify the actual float32 model boundary without storing another corpus copy.
        test = np.zeros((len(values), 16));test[:, :7] = values
        restored = normalizer.denormalize_action(normalizer.normalize_action(test))
        max_roundtrip = max(max_roundtrip, float(np.max(np.abs(restored[:, :7] - values))))
    publish(output / 'statistics.json', payload)
    manifest = dict(contract=VERSION, horizon=16, episodes=len(rows), frames=total_frames, windows=total_windows,
        source_data_root=str(data.resolve()), records=rows, max_position_reconstruction_error_m=max_position,
        max_rotation_reconstruction_error_rad=max_rotation, max_normalization_roundtrip_abs=max_roundtrip,
        history_manifest_sha256=sha(args.history_manifest), builder_sha256=sha(__file__))
    publish(output / 'manifest.json', manifest)
    publish(output / 'READY.json', dict(contract=VERSION, status='complete', episodes=len(rows), frames=total_frames,
        windows=total_windows, manifest_sha256=sha(output / 'manifest.json'), statistics_sha256=sha(output / 'statistics.json')))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
