"""Verify completed bundle fingerprints and real dual-camera training samples."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.dataloader.libero_action_contract import VERSION, LOSS_MASK


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--goal-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.goal_root
    ready = json.loads((root/'READY.json').read_text())
    if ready.get('status') != 'complete' or ready.get('contract') != VERSION:
        raise ValueError('Bundle has no complete matching receipt')
    if sha(root/'manifest.json') != ready['manifest_sha256'] or sha(root/'statistics.json') != ready['statistics_sha256']:
        raise ValueError('Bundle manifest/statistics fingerprint mismatch')
    manifest = json.loads((root/'manifest.json').read_text())
    grouped = {}
    for record in manifest['records']:
        dataset, episode = record['dataset'], record['episode']
        source = args.data_root/dataset/'data'/f'chunk-{episode//1000:03d}'/f'episode_{episode:06d}.parquet'
        if sha(source) != record['source_sha256'] or sha(root/record['label']) != record['label_sha256']:
            raise ValueError(f'Source or label fingerprint mismatch: {source}')
        grouped.setdefault(dataset, []).append(record)
    counts = dict(episodes=len(manifest['records']), frames=sum(r['frames'] for r in manifest['records']),
                  windows=sum(r['windows'] for r in manifest['records']))
    for key, value in counts.items():
        if manifest[key] != value or ready[key] != value:
            raise ValueError(f'Inconsistent {key} count')
    cfg = OmegaConf.create(dict(action_type=VERSION, controller_goal_root=str(root),
                               data_root_dir=str(args.data_root), data_mix='libero_all_anchored_osc_goal_binary',
                               drop_incomplete_action_chunks=True, video_backend='torchvision_av',
                               normalization_statistics=str(root/'statistics.json'), include_state=False,
                               action_mode='abs', state_action_dtype='float32', action_horizon=16,
                               image_size=[224,224], load_all_data_for_training=True))
    mixture = get_vla_dataset(data_cfg=cfg)
    datasets = {dataset.dataset_name: dataset for dataset in mixture.datasets}
    if set(datasets) != set(grouped):
        raise ValueError('Registered training mixture differs from the bundle datasets')
    suites = []
    for name, records in sorted(grouped.items()):
        dataset = datasets[name]
        expected_windows = sum(r['windows'] for r in records)
        if len(dataset) != expected_windows:
            raise ValueError(f'Loader window count differs: {name}')
        samples = []
        for index in (0, len(dataset)//2, len(dataset)-1):
            sample = dataset[index]
            action, mask = sample['action'], sample['action_mask']
            if action.shape != (16,16) or action.dtype != np.float32 or not np.isfinite(action).all():
                raise ValueError('Invalid float32 H16 action tensor')
            np.testing.assert_array_equal(mask, np.broadcast_to(LOSS_MASK, (16,16)))
            if not np.isin(action[:,7], [0,1]).all() or np.any(action[:,8:] != 0) or 'state' in sample:
                raise ValueError('Gripper, padding or state-free sample contract differs')
            if len(sample['image']) != 2 or any(image.size != (224,224) for image in sample['image']):
                raise ValueError('Dual-camera training images differ')
            if not isinstance(sample['lang'], str) or not sample['lang']:
                raise ValueError('Missing task instruction')
            samples.append(dict(index=index, trajectory=[int(x) for x in dataset._all_steps[index]],
                                action_shape=list(action.shape), image_sizes=[list(im.size) for im in sample['image']],
                                language=sample['lang']))
        exported = dataset.checkpoint_action_statistics()
        if exported != dataset.normalizer.payload['action']:
            raise ValueError('Checkpoint action statistics differ')
        suites.append(dict(dataset=name, episodes=len(records), frames=sum(r['frames'] for r in records),
                           windows=len(dataset), samples=samples))
        print(json.dumps(dict(dataset=name, windows=len(dataset), samples_verified=len(samples))), flush=True)
    payload = dict(status='complete', contract=VERSION, **counts, suites=suites,
                   training_mixture=cfg.data_mix,
                   all_source_and_label_hashes_verified=True, float32_actions=True,
                   binary_gripper=True, state_free=True, dual_camera=True,
                   loss_mask=LOSS_MASK.tolist(), bundle_manifest_sha256=sha(root/'manifest.json'))
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in payload.items() if k != 'suites'}), flush=True)


if __name__ == '__main__':
    main()
