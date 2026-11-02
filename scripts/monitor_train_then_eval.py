#!/usr/bin/env python3
"""Monitor one recorded training job and submit its two evaluations once."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dlc-submit-01'))
import dlc_cli as dlc
from alibabacloud_pai_dlc20201203.models import GetJobRequest


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def training_progress(training):
    output = Path(training['output'])
    result = {'nodes': [], 'alerts': []}
    for path in sorted((output / 'launch').glob('node*.log')):
        stat = path.stat()
        with path.open('rb') as f:
            f.seek(max(0, stat.st_size - 180000))
            text = f.read().decode(errors='replace')
        steps = re.findall(r'(\d+)/' + str(training['max_train_steps']), text)
        losses = re.findall(r'Step\s+(\d+), Loss:.*?action_dit_loss.*?:\s*([\d.eE+\-]+|nan|inf)', text, re.S | re.I)
        points = [{'step': int(s), 'loss': float(v)} for s, v in losses]
        timing = re.findall(r'data_times=([\d.]+), model_times=([\d.]+)', text)
        errors = [line[:300] for line in text.replace('\r', '\n').splitlines()
                  if any(s in line for s in ('Traceback (most recent call last)', 'CUDA out of memory',
                                             'ChildFailedError', 'DistBackendError'))]
        node = {'log': path.name, 'step': int(steps[-1]) if steps else None,
                'recent_losses': points[-10:], 'log_age_seconds': round(time.time() - stat.st_mtime),
                'latest_data_model_seconds': timing[-1] if timing else None, 'error_lines': errors[-5:]}
        result['nodes'].append(node)
        if errors:
            result['alerts'].append(path.name + ': training error in recent log')
        if any(not math.isfinite(p['loss']) for p in points):
            result['alerts'].append(path.name + ': non-finite loss')
        if node['log_age_seconds'] > 1800:
            result['alerts'].append(path.name + ': log unchanged for over 30 minutes')
    return result


def ensure_submission(entry, client):
    """Reuse receipts; reconcile uncertain Create responses before any new call."""
    run = Path(entry['run_dir'])
    receipt = run / 'submission.json'
    if receipt.exists():
        return json.loads(receipt.read_text())['JobId']
    if run.exists():
        identity = json.loads((run / 'identity.json').read_text())
        response = run / 'submission-response.json'
        job_id = json.loads(response.read_text()).get('JobId') if response.exists() else None
        if not job_id:
            matches = dlc.matching_jobs(client, identity)
            if len(matches) != 1:
                raise RuntimeError('Submission outcome unresolved; do not resend CreateJob')
            job_id = matches[0]['JobId']
        dlc.save_json(receipt, dict(identity, JobId=job_id, recovered=True))
        return job_id
    dlc.main(['submit', '--job', entry['job'], '--run-dir', str(run),
              '--work-root', str(ROOT), '--region', 'cn-wulanchabu'])
    return json.loads(receipt.read_text())['JobId']


def tick(manifest, client):
    training = manifest['training']
    job = dlc.call(client, 'get_job', training['job_id'], GetJobRequest())
    state = {'checked_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
             'training_job_id': training['job_id'], 'training_status': job['Status'],
             'phase': 'monitoring_training', **training_progress(training)}
    if job['Status'] in ('Failed', 'Stopped', 'Deleted'):
        state['phase'] = 'training_failed_no_evaluation'
        state['reason'] = job.get('ReasonMessage')
        return state
    if job['Status'] != 'Succeeded':
        return state
    needed = [Path(manifest['checkpoint']), Path(manifest['final_model'])]
    needed += [Path(training['output']) / name for name in
               ('config.yaml', 'config.full.yaml', 'dataset_statistics.json')]
    if not all(p.is_file() and p.stat().st_size > 0 for p in needed):
        state['phase'] = 'waiting_for_final_checkpoint'
        return state
    state['phase'] = 'submitting_evaluations'
    state['evaluations'] = []
    for entry in manifest['evaluations']:
        job_id = ensure_submission(entry, client)
        state['evaluations'].append({'benchmark': entry['benchmark'], 'job_id': job_id,
                                     'output': entry['output'], 'gpus': entry['gpus']})
    state['phase'] = 'evaluations_submitted'
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    manifest = json.loads(args.manifest.read_text())
    folder = args.manifest.parent
    # Local lock survives NAS rename semantics; held for this process lifetime.
    lock = open('/tmp/starvla-watch-' + manifest['training']['job_id'] + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / 'monitor.pid').write_text(str(os.getpid()) + '\n')
    client = dlc.make_client('cn-wulanchabu', argparse.Namespace(
        credentials_config='~/.aliyun/config.json', credentials_profile=None))
    while True:
        try:
            state = tick(manifest, client)
        except Exception as exc:
            # SDK exceptions may contain credentials. Persist only the class.
            state = {'checked_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                     'phase': 'retrying_monitor_or_reconciling_submission', 'error_type': type(exc).__name__}
        save(folder / 'monitor_status.json', state)
        with (folder / 'monitor_history.jsonl').open('a') as f:
            f.write(json.dumps(state, ensure_ascii=False) + '\n')
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if args.once or state['phase'] in ('evaluations_submitted', 'training_failed_no_evaluation'):
            return
        time.sleep(manifest['interval_seconds'])


if __name__ == '__main__':
    main()
