"""Recover missing controller receipts from uniquely matched original LIBERO demos.

Replay only the official ten settling commands. For subsequent goals, inject the
recorded command-reference poses into the real OSC controller's set_goal method;
no future observed pose or full trajectory resimulation supplies a target.
Requires robosuite 1.4.0, mujoco 3.2.3 and an installed official LIBERO checkout.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from match_libero_source_commands import retained_indices

CONTRACT = 'libero-original-command-history-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover_task(records, data_root, report_root, builder_sha):
    import mujoco
    import robosuite
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    if robosuite.__version__ != '1.4.0' or mujoco.__version__ != '3.2.3':
        raise ValueError('Recovery runtime differs from the verified simulator versions')
    root, data = Path(report_root), Path(data_root)
    source = Path(records[0]['official_lowdim_path'])
    suite = source.parent.name
    task_name = source.stem.removesuffix('_demo')
    task_receipt = root / 'controller_task_receipts' / suite / (task_name + '.json')
    if task_receipt.exists():
        previous = json.loads(task_receipt.read_text())
        if previous['builder_sha256'] != builder_sha:
            raise ValueError(f'Recovery code changed; use a new receipt directory: {task_receipt}')
        for row in previous['records']:
            if sha(root / 'source_sidecars' / row['label_path']) != row['label_sha256']:
                raise ValueError('Corrupt cached controller receipt')
        return previous
    bddl = Path(get_libero_path('bddl_files')) / suite / (task_name + '.bddl')
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                     has_offscreen_renderer=False, hard_reset=True)
    env.seed(0)
    results = []
    source_hash = sha(source)
    with np.load(source, allow_pickle=False) as archive:
        for record in records:
            if source_hash != record['official_lowdim_sha256']:
                raise ValueError('Original low-dimensional source changed')
            dataset, episode, demo = record['dataset'], record['episode'], record['official_demo']
            parquet = data / dataset / 'data' / f'chunk-{episode//1000:03d}' / f'episode_{episode:06d}.parquet'
            if sha(parquet) != record['parquet_sha256']:
                raise ValueError('LeRobot data changed after source matching')
            table = pq.read_table(parquet, columns=['observation.state', 'action']).to_pydict()
            state = np.asarray(table['observation.state'], dtype=np.float64)
            actions = np.asarray(table['action'], dtype=np.float64)
            native = actions.copy()
            native[:, 6] = 1 - 2 * native[:, 6]
            original_actions = archive[demo + '__actions']
            indices = retained_indices(original_actions)
            np.testing.assert_allclose(original_actions[indices], native, atol=5e-6, rtol=0)
            env.reset()
            env.set_init_state(archive[demo + '__initial_state'])
            for _ in range(10):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
            controller = env.robots[0].controller
            controller.update(force=True)
            position_error = float(np.max(np.abs(obs['robot0_eef_pos'] - state[0, :3])))
            rotation_error = float((Rotation.from_quat(obs['robot0_eef_quat']) *
                                    Rotation.from_rotvec(state[0, 3:6]).inv()).magnitude())
            if position_error > 5e-6 or rotation_error > 5e-6:
                raise ValueError(f'Initial replay differs from recorded observation: {dataset}/{episode}: '
                                 f'{position_error} m, {rotation_error} rad')
            body = env.sim.data.get_body_xmat(env.robots[0].robot_model.eef_name)
            site_to_body = controller.ee_ori_mat.T @ body
            initial_goal = Rotation.from_matrix(controller.goal_ori @ site_to_body).as_quat()
            # Use the shipped controller code with recorded current observations.
            # Save/restore update because it normally pulls the live simulator pose.
            original_update = controller.update
            controller.update = lambda: None
            goals = np.empty((len(state), 8), dtype=np.float64)
            try:
                for i in range(len(state)):
                    controller.ee_pos = state[i, :3].copy()
                    controller.ee_ori_mat = Rotation.from_rotvec(state[i, 3:6]).as_matrix() @ site_to_body.T
                    controller.set_goal(native[i, :6])
                    goals[i, :3] = controller.goal_pos
                    goals[i, 3:7] = Rotation.from_matrix(controller.goal_ori @ site_to_body).as_quat()
                goals[:, 7] = (actions[:, 6] > .5).astype(np.float64)
            finally:
                controller.update = original_update
            relative = Path(dataset) / f'episode_{episode:06d}.npz'
            output = root / 'source_sidecars' / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, goals=goals, native_actions=native,
                                retained_indices=indices, initial_goal_quaternion=initial_goal,
                                site_to_body=site_to_body)
            temporary.replace(output)
            results.append(dict(record, label_path=str(relative), label_sha256=sha(output),
                                initial_position_replay_error_m=position_error,
                                initial_rotation_replay_error_rad=rotation_error))
    env.close()
    payload = dict(status='complete', contract=CONTRACT, builder_sha256=builder_sha,
                   bddl_sha256=sha(bddl), source_lowdim_sha256=source_hash, records=results)
    task_receipt.parent.mkdir(parents=True, exist_ok=True)
    task_receipt.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--report-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    mapping_path = args.report_root / 'official_episode_mapping.json'
    mapping = json.loads(mapping_path.read_text())
    if mapping['status'] != 'complete':
        raise ValueError('Source matching is incomplete')
    grouped = {}
    for record in mapping['records']:
        grouped.setdefault(record['official_lowdim_path'], []).append(record)
    rows, tasks = [], []
    builder_sha = sha(__file__)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(recover_task, records, str(args.data_root), str(args.report_root), builder_sha)
                   for records in grouped.values()]
        for future in as_completed(futures):
            task = future.result()
            tasks.append({k:v for k,v in task.items() if k != 'records'})
            rows.extend(task['records'])
            print(json.dumps(dict(completed_tasks=len(tasks), episodes=len(rows),
                                  dataset=task['records'][0]['dataset'],
                                  task=task['records'][0]['task'])), flush=True)
    rows.sort(key=lambda r: (r['dataset'], r['episode']))
    manifest = dict(status='complete', contract=CONTRACT, episodes=len(rows),
                    frames=sum(r['frames'] for r in rows), records=rows, tasks=tasks,
                    builder_sha256=builder_sha, mapping_sha256=sha(mapping_path),
                    official_lowdim_manifest_sha256=sha(args.report_root/'official_lowdim_manifest.json'),
                    runtime=dict(robosuite='1.4.0', mujoco='3.2.3', seed=0, settling_steps=10,
                                 dummy_action=[0,0,0,0,0,0,-1], camera_observations=False,
                                 hard_reset=True, libero_config_path=os.environ['LIBERO_CONFIG_PATH']),
                    target_source='real OSC set_goal on current recorded poses and matched commands',
                    max_initial_position_replay_error_m=max(r['initial_position_replay_error_m'] for r in rows),
                    max_initial_rotation_replay_error_rad=max(r['initial_rotation_replay_error_rad'] for r in rows))
    output = args.report_root/'source_sidecar_manifest.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('records','tasks')}), flush=True)


if __name__ == '__main__':
    main()
