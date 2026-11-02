"""Launch independent policy replicas and bounded LIBERO simulator workers."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import (
    SUITES, TASK_COUNTS, OUTPUT_LAYOUT, VIDEO_LAYOUT, aggregate, atomic_json, digest, select_jobs,
    suite_directory, write_summary,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--devices', default='0,1')
    p.add_argument('--workers-per-device', type=int, default=2)
    p.add_argument('--suites', nargs='+', choices=SUITES, default=['libero_object'])
    p.add_argument('--num-trials-per-task', type=int, default=50)
    p.add_argument('--task-ids', type=int, nargs='*', default=[])
    p.add_argument('--trial-ids', type=int, nargs='*', default=[])
    p.add_argument('--max-tasks', type=int, default=-1)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--num-steps-wait', type=int, default=10)
    p.add_argument('--action-interface', choices=['anchored_goal_binary', 'native_osc7'], default='anchored_goal_binary')
    p.add_argument('--unnorm-key')
    p.add_argument('--libero-home', default=str(ROOT.parent / 'LIBERO'))
    p.add_argument('--simulation-python', default=sys.executable)
    p.add_argument('--simulation-site', default='')
    p.add_argument('--policy-python', default=sys.executable)
    p.add_argument('--base-port', type=int, default=11093)
    p.add_argument('--server-start-timeout', type=float, default=900)
    p.add_argument('--config-override', action='append', default=[])
    p.add_argument('--fp32', action='store_true', help='Disable default BF16 inference')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Print schedule without creating output or loading models')
    p.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--port', type=int, default=11093, help=argparse.SUPPRESS)
    return p


def serve(args):
    import logging
    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.seed_utils import set_seed_everywhere
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    logging.basicConfig(level=logging.INFO)

    class SeededPolicy(PolicyServerWrapper):
        def predict_action(self, examples, diagnostic_seed=None, **kwargs):
            if diagnostic_seed is None:
                raise ValueError('Parallel evaluator requires an explicit per-request seed')
            set_seed_everywhere(int(diagnostic_seed))
            return super().predict_action(examples=examples, **kwargs)

    set_seed_everywhere(args.seed)
    policy = SeededPolicy(str(Path(args.ckpt).resolve()), device='cuda', use_bf16=not args.fp32,
                          config_overrides=args.config_override)
    metadata = dict(policy.metadata, request_seeded=True, seed_protocol='sha256-v1 episode identity + chunk',
                    visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__)
    print('POLICY_READY', args.port, flush=True)
    WebsocketPolicyServer(policy=policy, host='127.0.0.1', port=args.port,
                          idle_timeout=-1, metadata=metadata).serve_forever()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def make_protocol(args):
    ckpt = Path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if args.action_interface == 'anchored_goal_binary':
        from omegaconf import OmegaConf
        from starVLA.dataloader.libero_action_contract import VERSION
        model_cfg = OmegaConf.load(ckpt.parent.parent / 'config.yaml')
        if model_cfg.datasets.vla_data.get('action_type') != VERSION:
            raise ValueError('Explicit anchored_goal_binary evaluation requires a checkpoint trained with the new action contract')
    if len(set(args.suites)) != len(args.suites) or args.num_steps_wait < 0:
        raise ValueError('Duplicate suites or negative wait steps')
    source_hashes = {}
    for directory in ('examples/simBenchmarks/LIBERO/eval_files', 'deployment/model_server',
                      'starVLA/model', 'starVLA/dataloader', 'starVLA/training/trainer_utils'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    configs = {}
    for path in sorted(ckpt.parent.parent.glob('*')):
        if path.name in ('config.yaml', 'config.full.yaml', 'dataset_statistics.json'):
            configs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not configs:
        raise ValueError('No colocated model configuration found')
    jobs = [j for suite in args.suites for j in select_jobs(suite, TASK_COUNTS[suite],
            args.num_trials_per_task, args.task_ids, args.trial_ids, args.max_tasks)]
    if not jobs:
        raise ValueError('No selected episodes')
    simulation_site = str(Path(args.simulation_site).resolve()) if args.simulation_site else ''
    dependencies = {}
    if simulation_site:
        if not Path(simulation_site).is_dir():
            raise FileNotFoundError(simulation_site)
        for metadata in sorted(Path(simulation_site).glob('*.dist-info/METADATA')):
            dependencies[metadata.parent.name] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    return dict(schema=1, checkpoint=file_identity(ckpt), model_config_sha256=configs,
        source_sha256=source_hashes, jobs=jobs, seed=args.seed,
        seed_protocol='sha256-v1 episode identity + chunk', deterministic_inference=True,
        action_interface=args.action_interface,
        gripper_encoding='binary_open_code', gripper_threshold=0.5,
        num_steps_wait=args.num_steps_wait, unnorm_key=args.unnorm_key,
        bf16=not args.fp32, config_overrides=args.config_override, save_video=not args.no_video,
        simulation_python=str(Path(args.simulation_python).resolve()),
        policy_python=str(Path(args.policy_python).resolve()),
        simulation_site=simulation_site, simulation_dependencies=dependencies,
        libero_home=str(Path(args.libero_home).resolve()), render='osmesa; original per-step cameras',
        numba_disable_jit=True, resolution=256, video_fps=10,
        video_cameras=['agentview_image', 'robot0_eye_in_hand_image'] if not args.no_video else [],
        video_layout=VIDEO_LAYOUT if not args.no_video else None, output_layout=OUTPUT_LAYOUT)


def simulation_env(args):
    env = dict(os.environ)
    paths = [str(ROOT), str(Path(args.libero_home).resolve())]
    if args.simulation_site:
        paths.insert(0, str(Path(args.simulation_site).resolve()))
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env.update(PYTHONPATH=os.pathsep.join(paths), LIBERO_CONFIG_PATH=str(Path(args.libero_home).resolve() / 'libero'),
        MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        NUMBA_DISABLE_JIT='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', LP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', DEBUG='')
    return env


def worker_command(args, suite, shard, shards, port, protocol_digest):
    cmd = [args.simulation_python, '-u', str(Path(__file__).with_name('eval_libero.py')),
        '--args.host', '127.0.0.1', '--args.port', str(port), '--args.pretrained-path', str(Path(args.ckpt).resolve()),
        '--args.task-suite-name', suite, '--args.num-trials-per-task', str(args.num_trials_per_task),
        '--args.max-tasks', str(args.max_tasks), '--args.seed', str(args.seed),
        '--args.num-steps-wait', str(args.num_steps_wait), '--args.action-interface', args.action_interface,
        '--args.shard-index', str(shard), '--args.num-shards', str(shards), '--args.deterministic-inference',
        '--args.results-dir', str(Path(args.output_dir).resolve()), '--args.protocol-digest', protocol_digest,
        '--args.video-out-path', str(Path(args.output_dir).resolve())]
    if args.unnorm_key:
        cmd += ['--args.unnorm-key', args.unnorm_key]
    if args.task_ids:
        cmd += ['--args.task-ids'] + [str(x) for x in args.task_ids]
    if args.trial_ids:
        cmd += ['--args.trial-ids'] + [str(x) for x in args.trial_ids]
    if args.resume:
        cmd += ['--args.resume']
    if args.no_video:
        cmd += ['--args.no-save-video']
    return cmd


def stop_processes(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for child in children:
        try:
            child.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def run(args):
    for field in ('simulation_python', 'policy_python'):
        executable = shutil.which(getattr(args, field))
        if executable is None:
            raise FileNotFoundError(getattr(args, field))
        setattr(args, field, executable)
    devices = args.devices.split(',')
    if not all(d.isdigit() for d in devices) or len(set(devices)) != len(devices) or args.workers_per_device < 1:
        raise ValueError('Use distinct numeric devices and positive workers-per-device')
    if not 1 <= args.base_port <= 65536 - len(devices) or args.server_start_timeout <= 0:
        raise ValueError('Invalid port range or startup timeout')
    protocol = make_protocol(args)
    protocol_digest = digest(protocol)
    shards = len(devices) * args.workers_per_device
    schedule = []
    for suite in args.suites:
        for shard in range(shards):
            jobs = select_jobs(suite, TASK_COUNTS[suite], args.num_trials_per_task,
                args.task_ids, args.trial_ids, args.max_tasks, shard, shards)
            if jobs:
                device_index = shard // args.workers_per_device
                schedule.append(dict(suite=suite, shard=shard, device=devices[device_index],
                    port=args.base_port + device_index, episodes=len(jobs)))
    if args.dry_run:
        print(json.dumps(dict(protocol_digest=protocol_digest, expected=len(protocol['jobs']),
            policy_replicas=len(devices), max_simulators=shards, schedule=schedule), indent=2))
        return
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = output / 'protocol.json'
    if manifest.exists():
        if not args.resume:
            raise FileExistsError('Existing run; pass --resume or use a new output-dir')
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError('Resume protocol mismatch; use a new output-dir for changed settings/code')
    else:
        if any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError('Output directory is not an empty evaluation run')
        atomic_json(manifest, protocol)
    state = aggregate(output, protocol['jobs'], protocol_digest, not args.no_video, required_video_layout=VIDEO_LAYOUT)
    write_summary(output, state)
    if state['status'] == 'complete':
        print('Already complete:', state['completed'], flush=True)
        return
    sim_env = simulation_env(args)
    preflight = subprocess.run([args.simulation_python, '-c',
        'import json,sys,numpy,mujoco,robosuite; from libero.libero import benchmark; '
        'print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,mujoco=mujoco.__version__,robosuite=robosuite.__version__,task_counts={s:benchmark.get_benchmark_dict()[s]().n_tasks for s in ' + repr(args.suites) + '})))'],
        cwd=ROOT, env=sim_env, capture_output=True, text=True)
    (output / 'simulation_preflight.log').write_text(preflight.stdout + preflight.stderr)
    if preflight.returncode:
        raise RuntimeError('Simulator preflight failed; see simulation_preflight.log')
    current_runtime = json.loads(preflight.stdout.strip().splitlines()[-1])
    if current_runtime['mujoco'] != '3.2.3' or current_runtime['robosuite'] != '1.4.0':
        raise ValueError('LIBERO evaluation requires mujoco 3.2.3 and robosuite 1.4.0, matching action-data generation')
    if any(current_runtime['task_counts'][s] != TASK_COUNTS[s] for s in args.suites):
        raise ValueError('Benchmark task count differs from the launch plan')
    runtime_file = output / 'simulation_runtime.json'
    if runtime_file.exists() and json.loads(runtime_file.read_text()) != current_runtime:
        raise ValueError('Simulator runtime changed; resume refused')
    atomic_json(runtime_file, current_runtime)
    children, logs, servers, active = [], [], [], []
    started = time.monotonic()
    try:
        # Reserve-check every port before loading any model. Existing servers are never reused or killed.
        reservations = []
        try:
            for index in range(len(devices)):
                sock = socket.socket()
                sock.bind(('127.0.0.1', args.base_port + index))
                reservations.append(sock)
        finally:
            for sock in reservations:
                sock.close()
        for index, device in enumerate(devices):
            port = args.base_port + index
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''), DEBUG='')
            cmd = [args.policy_python, '-u', str(Path(__file__).resolve()), '--serve',
                   '--ckpt', args.ckpt, '--output-dir', str(output), '--port', str(port), '--seed', str(args.seed)]
            if args.fp32:
                cmd += ['--fp32']
            for override in args.config_override:
                cmd += ['--config-override', override]
            log = (output / f'policy_{device}.log').open('a', buffering=1)
            logs.append(log)
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            servers.append((port, child))
        ready = set()
        deadline = time.monotonic() + args.server_start_timeout
        from deployment.model_server.tools import msgpack_numpy
        from websockets.sync.client import connect
        while len(ready) < len(servers):
            for port, child in servers:
                if child.poll() is not None:
                    raise RuntimeError(f'Policy server on port {port} exited; inspect policy log')
                if port in ready:
                    continue
                try:
                    with connect(f'ws://127.0.0.1:{port}', open_timeout=1, proxy=None,
                                 max_size=None, compression=None, ping_interval=None) as ws:
                        metadata = msgpack_numpy.unpackb(ws.recv(timeout=2))
                    if not metadata.get('request_seeded') or Path(metadata['ckpt_path']).resolve() != Path(args.ckpt).resolve():
                        raise ValueError('Unexpected policy metadata')
                    atomic_json(output / f'policy_metadata_{port}.json', metadata)
                    ready.add(port)
                    print('Policy ready:', port, flush=True)
                except (OSError, TimeoutError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError('Policy startup timed out')
            time.sleep(.5)
        pending = list(schedule)
        missing = set(state['missing'])
        pending = [item for item in pending if any(j['key'] in missing for j in select_jobs(item['suite'],
            TASK_COUNTS[item['suite']], args.num_trials_per_task, args.task_ids, args.trial_ids,
            args.max_tasks, item['shard'], shards))]
        last_progress = 0
        while pending or active:
            if any(child.poll() is not None for _, child in servers):
                raise RuntimeError('Policy server exited during evaluation')
            for item in list(pending):
                if sum(a['device'] == item['device'] for a in active) >= args.workers_per_device:
                    continue
                log = (suite_directory(output, item['suite']) / f"worker_{item['shard']}.log").open('a', buffering=1)
                logs.append(log)
                cmd = worker_command(args, item['suite'], item['shard'], shards, item['port'], protocol_digest)
                child = subprocess.Popen(cmd, cwd=ROOT, env=sim_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                active.append(dict(item, child=child))
                pending.remove(item)
            for item in list(active):
                code = item['child'].poll()
                if code is not None:
                    active.remove(item)
                    if code:
                        raise RuntimeError(f"Simulator shard failed: {item['suite']}/{item['shard']} exit={code}")
            if time.monotonic() - last_progress >= 10 or not (pending or active):
                state = aggregate(output, protocol['jobs'], protocol_digest, not args.no_video, required_video_layout=VIDEO_LAYOUT)
                state['elapsed_seconds_this_launch'] = time.monotonic() - started
                write_summary(output, state)
                print(f"Progress: {state['completed']}/{state['expected']}, successes={state['successes']}", flush=True)
                last_progress = time.monotonic()
            time.sleep(.5)
        if state['status'] != 'complete':
            raise RuntimeError('Workers exited but expected episodes/artifacts are missing')
        atomic_json(output / 'completion.json', dict(status='complete', protocol_digest=protocol_digest,
            elapsed_seconds=time.monotonic() - started, devices=devices, workers_per_device=args.workers_per_device))
    except BaseException as exc:
        state = aggregate(output, protocol['jobs'], protocol_digest, not args.no_video, required_video_layout=VIDEO_LAYOUT)
        state['status'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        state['error'] = str(exc)
        write_summary(output, state)
        raise
    finally:
        stop_processes(children)
        for log in logs:
            log.close()
        lock.close()


if __name__ == '__main__':
    args = parser().parse_args()
    if args.serve:
        serve(args)
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        run(args)
