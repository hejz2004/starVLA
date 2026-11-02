"""Two independent 8-GPU evaluation shards; master publishes combined results."""
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
SUITES = {0: ['libero_spatial', 'libero_goal'], 1: ['libero_object', 'libero_10']}
FOLDERS = {'libero_spatial':'Spatial', 'libero_goal':'Goal', 'libero_object':'Object', 'libero_10':'Long'}


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def command(benchmark, rank, checkpoint, output):
    common = ['--ckpt', str(checkpoint), '--output-dir', str(output), '--devices', '0,1,2,3,4,5,6,7',
              '--workers-per-device', '20', '--policy-python', '/opt/qwenpi/train/bin/python',
              '--suites', *SUITES[rank]]
    if benchmark == 'libero':
        return [sys.executable, '-u', str(ROOT/'examples/simBenchmarks/LIBERO/eval_files/parallel_eval.py'),
                *common, '--simulation-python', '/opt/qwenpi/libero/bin/python',
                '--simulation-site', '/opt/qwenpi/libero/lib/python3.10/site-packages',
                '--libero-home', str(ROOT.parent/'LIBERO'), '--num-trials-per-task', '50',
                '--action-interface', 'anchored_goal_binary', '--base-port', '11093']
    if benchmark == 'libero_plus':
        gate = os.environ.get('FORCE_ZERO_GATE', '0')
        if gate not in ('0', '1'):
            raise ValueError('FORCE_ZERO_GATE must be 0 or 1')
        if gate == '1':
            common.append('--force-zero-gate')
        return [sys.executable, '-u', str(ROOT/'examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/launchers/evaluation_controller.py'),
                *common, '--simulation-python', '/opt/qwenpi/plus/bin/python',
                '--simulation-site', '/opt/qwenpi/plus/lib/python3.10/site-packages',
                '--libero-home', str(ROOT/'playground/Artifacts/libero_plus_shared/source'),
                '--libero-assets', '/mnt/workspace/Data/LIBERO_PLUS_EVAL_4976dc3/assets',
                '--cpu-budget', '80', '--base-port', '12883']
    raise ValueError(benchmark)


def _merge(output, benchmark):
    expected = 2000 if benchmark == 'libero' else 10030
    parts = []
    statuses = []
    for rank in (0,1):
        p = output/f'node{rank}'/'summary.json'
        if p.exists():
            parts.append(json.loads(p.read_text()))
        p = output/f'node{rank}.status.json'
        statuses.append(json.loads(p.read_text()) if p.exists() else {})
    complete = len(parts)==2 and all(s.get('status')=='complete' for s in parts) and all(s.get('exit_code')==0 for s in statuses)
    failed = any(s.get('exit_code',0)!=0 for s in statuses)
    suites = [s for p in parts for s in p['suites']]
    completed = sum(p['completed'] for p in parts)
    successes = sum(p['successes'] for p in parts)
    if complete and (completed!=expected or {s['suite'] for s in suites}!=set(FOLDERS) or len(suites)!=4):
        raise RuntimeError('Incomplete or duplicated suite coverage')
    result = dict(status='complete' if complete else 'failed' if failed else 'running',
                  benchmark=benchmark, expected=expected, completed=completed, successes=successes,
                  success_rate=successes/completed if completed else None, suites=suites,
                  node_statuses=statuses, updated_at=time.time())
    atomic(output/'summary.json', result)
    return result


def merge(output, benchmark):
    try:
        return _merge(output, benchmark)
    except OSError as exc:
        if exc.errno != errno.ESTALE:
            raise
        # Peer summaries are atomically replaced on NAS. A stale handle is
        # transient; discard this snapshot and reopen everything next cycle.
        print(f"NAS summary temporarily stale; retrying next cycle: {exc.filename}", flush=True)
        return {'status': 'running'}


def main():
    rank = int(os.environ['EVAL_NODE_RANK'])
    benchmark = os.environ['BENCHMARK']
    output = Path(os.environ['RUN_ROOT'])
    ckpt = Path(os.environ['CKPT_PATH'])
    assert rank in SUITES
    for suite in SUITES[rank]:
        target = output/FOLDERS[suite]
        target.symlink_to(Path(f'node{rank}')/FOLDERS[suite], target_is_directory=True)
    atomic(output/f'node{rank}.status.json', dict(status='running', suites=SUITES[rank], started_at=time.time()))
    child = subprocess.Popen(command(benchmark,rank,ckpt,output/f'node{rank}'), cwd=ROOT)
    try:
        while child.poll() is None:
            if rank==0:
                state = merge(output,benchmark)
                if state['status']=='failed':
                    raise RuntimeError('Peer evaluation node failed')
            time.sleep(15)
        code = child.returncode
        atomic(output/f'node{rank}.status.json', dict(status='complete' if code==0 else 'failed',exit_code=code,finished_at=time.time()))
        if code:
            raise RuntimeError(f'Node {rank} evaluator exited with {code}')
        if rank==0:
            while True:
                state = merge(output,benchmark)
                if state['status']=='failed':
                    raise RuntimeError('Peer evaluation node failed')
                if state['status']=='complete':
                    atomic(output/'completion.json', dict(status='complete',benchmark=benchmark,checkpoint=str(ckpt),
                           expected=state['expected'],completed=state['completed'],gpus=16,nodes=2,finished_at=time.time()))
                    print(json.dumps({k:state[k] for k in ('status','expected','completed','successes','success_rate')}),flush=True)
                    break
                time.sleep(15)
    except BaseException:
        if child.poll() is None:
            child.terminate()
        atomic(output/f'node{rank}.status.json', dict(status='failed',exit_code=1,finished_at=time.time()))
        raise

if __name__=='__main__':
    main()
