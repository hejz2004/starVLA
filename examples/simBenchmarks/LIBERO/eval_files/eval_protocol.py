"""Pure helpers shared by sequential and parallel LIBERO evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90')
MAX_STEPS = dict(zip(SUITES, (220, 280, 300, 520, 400)))
TASK_COUNTS = dict(zip(SUITES, (10, 10, 10, 10, 90)))
SUITE_FOLDERS = dict(zip(SUITES, ('Spatial', 'Object', 'Goal', 'Long', 'LIBERO90')))
VIDEO_LAYOUT = 'main_left_wrist_right'
OUTPUT_LAYOUT = 'suite-directories-v1'


def suite_directory(output, suite):
    return Path(output) / SUITE_FOLDERS[suite]



def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def episode_key(suite, task_id, trial):
    return f'{suite}_task{task_id:02d}_trial{trial:03d}'


def inference_seed(seed, suite, task_id, trial, chunk):
    # Explicit identities, independent of process scheduling, worker count and resume.
    return int(digest([int(seed), suite, int(task_id), int(trial), int(chunk)])[:8], 16)


def select_jobs(suite, n_tasks, n_trials, task_ids=(), trial_ids=(), max_tasks=-1,
                shard_index=0, num_shards=1):
    if suite not in SUITES or n_trials < 1 or num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError('Invalid suite, trial count or shard assignment')
    tasks = list(task_ids) if task_ids else list(range(n_tasks if max_tasks <= 0 else min(n_tasks, max_tasks)))
    trials = list(trial_ids) if trial_ids else list(range(n_trials))
    if len(set(tasks)) != len(tasks) or len(set(trials)) != len(trials):
        raise ValueError('Duplicate task/trial IDs')
    if any(t < 0 or t >= n_tasks for t in tasks) or any(t < 0 or t >= n_trials for t in trials):
        raise ValueError('Task/trial IDs outside requested bounds')
    jobs = [dict(suite=suite, task_id=t, trial=e, key=episode_key(suite, t, e))
            for e in trials for t in tasks]
    return [j for i, j in enumerate(jobs) if i % num_shards == shard_index]


def valid_result(path, expected, protocol_digest, require_video=True, *, required_video_layout=None):
    path = Path(path)
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    for field in ('suite', 'task_id', 'trial', 'key'):
        if row.get(field) != expected[field]:
            raise ValueError(f'Result identity mismatch: {path}')
    if row.get('protocol_digest') != protocol_digest:
        raise ValueError(f'Result protocol mismatch: {path}')
    if type(row.get('success')) is not bool or type(row.get('steps')) is not int or row['steps'] < 1:
        raise ValueError(f'Invalid completed result: {path}')
    if require_video:
        if required_video_layout is not None and row.get('video_layout') != required_video_layout:
            return False
        value = row.get('video')
        if not isinstance(value, str) or not value:
            return False
        artifact = path.parent.parent / value
        if not artifact.is_file() or artifact.stat().st_size == 0:
            return False
    trace = row.get('trace')
    if not trace or not (path.parent.parent / trace).is_file():
        return False
    return True


def aggregate(output, jobs, protocol_digest, require_video=True, *, required_video_layout=None):
    output = Path(output)
    expected = {j['key']: j for j in jobs}
    # Read historical flat records as well; new writers always use suite directories.
    paths = list((output / 'results').glob('*.json'))
    for folder in SUITE_FOLDERS.values():
        paths.extend((output / folder / 'results').glob('*.json'))
    unexpected = [p.stem for p in paths if p.stem not in expected]
    if unexpected:
        raise ValueError(f'Unexpected result identities: {unexpected[:5]}')
    records = {}
    for path in paths:
        if path.stem in records:
            raise ValueError(f'Duplicate result identity: {path.stem}')
        records[path.stem] = path
    rows, missing = [], []
    for key, job in expected.items():
        default = suite_directory(output, job['suite']) / 'results' / f'{key}.json'
        path = records.get(key, default)
        if valid_result(path, job, protocol_digest, require_video, required_video_layout=required_video_layout):
            rows.append(json.loads(path.read_text()))
        else:
            missing.append(key)
    groups = []
    for suite, task_id in sorted({(j['suite'], j['task_id']) for j in jobs}):
        selected = [r for r in rows if r['suite'] == suite and r['task_id'] == task_id]
        total = sum(j['suite'] == suite and j['task_id'] == task_id for j in jobs)
        successes = sum(r['success'] for r in selected)
        groups.append(dict(suite=suite, task_id=task_id, expected=total, completed=len(selected),
                           successes=successes, success_rate=successes / len(selected) if selected else None))
    successes = sum(r['success'] for r in rows)
    suites = []
    for suite in sorted({j['suite'] for j in jobs}):
        selected = [r for r in rows if r['suite'] == suite]
        count = sum(r['success'] for r in selected)
        suites.append(dict(suite=suite, completed=len(selected), successes=count,
                           success_rate=count / len(selected) if selected else None))
    return dict(status='complete' if not missing else 'incomplete', expected=len(jobs),
                completed=len(rows), successes=successes,
                success_rate=successes / len(rows) if rows else None,
                tasks=groups, suites=suites, missing=missing)


def write_summary(output, state):
    """Write the global summary and independent summaries inside each suite."""
    output = Path(output)
    atomic_json(output / 'summary.json', state)
    for row in state['suites']:
        suite = row['suite']
        tasks = [t for t in state['tasks'] if t['suite'] == suite]
        expected = sum(t['expected'] for t in tasks)
        missing = [key for key in state['missing'] if key.startswith(suite + '_task')]
        status = 'complete' if row['completed'] == expected else state['status']
        summary = dict(row, status=status, expected=expected, tasks=tasks, missing=missing)
        for field in ('elapsed_seconds_this_launch', 'error'):
            if field in state:
                summary[field] = state[field]
        atomic_json(suite_directory(output, suite) / 'summary.json', summary)
