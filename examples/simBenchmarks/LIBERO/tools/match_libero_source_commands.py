"""Match every LeRobot episode to an official demo without relying on episode order."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retained_indices(actions):
    indices = []
    previous = None
    for i, action in enumerate(actions):
        noop = np.linalg.norm(action[:6]) < 1e-4
        if previous is not None:
            noop = noop and action[6] == previous[6]
        if not noop:
            indices.append(i)
            previous = action
    return np.asarray(indices, dtype=np.int64)


def fingerprint(actions):
    actions = np.array(actions, dtype='<f4', copy=True)
    actions[actions == 0] = 0  # Canonicalize signed zero.
    return hashlib.sha256(actions.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--report-root', type=Path, required=True)
    args = parser.parse_args()
    records = []
    cache = {}
    suites = ['libero_object', 'libero_goal', 'libero_spatial', 'libero_10']
    for suite in suites:
        dataset = suite + '_no_noops_1.0.0_lerobot'
        meta = args.data_root / dataset / 'meta/episodes.jsonl'
        for line in meta.read_text().splitlines():
            episode = json.loads(line)
            index = episode['episode_index']
            task = episode['tasks'][0].replace(' ', '_')
            candidates = list((args.report_root / 'official_lowdim' / suite).glob('*' + task + '_demo.npz'))
            if len(candidates) != 1:
                raise ValueError(f'Expected one source task for {dataset}/{index}: {candidates}')
            source = candidates[0]
            if source not in cache:
                by_hash = {}
                all_actions = []
                with np.load(source, allow_pickle=False) as archive:
                    for key in archive.files:
                        if not key.endswith('__actions'):
                            continue
                        demo = key.removesuffix('__actions')
                        actions = archive[key]
                        indices = retained_indices(actions)
                        native = actions[indices]
                        item = (demo, indices, native)
                        by_hash.setdefault(fingerprint(native), []).append(item)
                        all_actions.append(item)
                cache[source] = by_hash, all_actions
            parquet = args.data_root / dataset / 'data' / f'chunk-{index // 1000:03d}' / f'episode_{index:06d}.parquet'
            table = pq.read_table(parquet, columns=['action', 'observation.state']).to_pydict()
            native = np.asarray(table['action'], dtype=np.float64)
            native[:, 6] = 1 - 2 * native[:, 6]
            by_hash, all_actions = cache[source]
            matches = by_hash.get(fingerprint(native), [])
            if not matches:
                matches = [item for item in all_actions if item[2].shape == native.shape
                           and np.allclose(item[2], native, atol=5e-6, rtol=0)]
            if len(matches) != 1:
                raise ValueError(f'Non-unique source match: {dataset}/{index}: {[x[0] for x in matches]}')
            demo, indices, original = matches[0]
            state = np.asarray(table['observation.state'])
            if len(native) != episode['length'] or not np.isfinite(state).all():
                raise ValueError(f'Invalid episode: {parquet}')
            record = dict(dataset=dataset, episode=index, task=episode['tasks'][0],
                          official_lowdim_path=str(source), official_lowdim_sha256=sha(source),
                          official_demo=demo, frames=len(native),
                          first_retained_index=int(indices[0]), removed_frames=int(indices[-1]+1-len(indices)),
                          max_action_error=float(np.abs(original-native).max()),
                          starts_with_zero_rotation=bool(np.all(native[0, 3:6] == 0)),
                          parquet_sha256=sha(parquet))
            records.append(record)
        print(json.dumps(dict(dataset=dataset, matched=sum(r['dataset'] == dataset for r in records))), flush=True)
    output = args.report_root / 'official_episode_mapping.json'
    payload = dict(status='complete', matching='unique full retained native command sequence',
                   noop_threshold=1e-4, episodes=len(records),
                   frames=sum(r['frames'] for r in records),
                   zero_rotation_prefix_episodes=sum(r['starts_with_zero_rotation'] for r in records),
                   max_action_error=max(r['max_action_error'] for r in records), records=records)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k:v for k,v in payload.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
