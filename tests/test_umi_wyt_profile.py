"""Check compact config expansion against the pre-refactor real-data baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.umi_wyt import (PROFILE_METADATA, RELATIVE_EEF,
    make_wyt_umi_dataloader, plain_config, resolve_umi_stages,
    resolve_wyt_data_profile, stage_identity)


def expect_error(call, text):
    try:
        call()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('Invalid configuration was accepted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    legacy = OmegaConf.create(Path(args.baseline).read_text())
    compact = OmegaConf.load('examples/umi/train_files/starvla_dualstream_umi_v2.yaml')
    initial = plain_config(compact)
    assert len(initial.datasets.vla_data) == 7
    # The requested batch change is independent of profile expansion.
    legacy.datasets.vla_data.per_device_batch_size = compact.datasets.vla_data.per_device_batch_size
    for cfg in (legacy, compact):
        resolve_umi_stages(cfg, 64)
    left = OmegaConf.to_container(legacy.datasets.vla_data, resolve=True)
    right = {k: v for k, v in OmegaConf.to_container(compact.datasets.vla_data, resolve=True).items()
             if k not in PROFILE_METADATA}
    assert left == right, {k: (left.get(k),right.get(k)) for k in set(left)|set(right) if left.get(k)!=right.get(k)}
    assert stage_identity(legacy,64) == stage_identity(compact,64)
    assert compact.trainer.max_train_steps == 77732
    # Omission still resolves to relative EEF; explicit conflicts fail at both entry points.
    omitted = plain_config(initial)
    del omitted.datasets.vla_data.action_pose_repr
    resolve_wyt_data_profile(omitted)
    assert omitted.datasets.vla_data.action_pose_repr == RELATIVE_EEF
    bad = plain_config(initial)
    bad.datasets.vla_data.action_pose_repr = 'world'
    expect_error(lambda: resolve_umi_stages(bad,64), 'world actions are forbidden')
    expect_error(lambda: make_wyt_umi_dataloader(bad), 'world actions are forbidden')
    bad = plain_config(initial)
    bad.datasets.vla_data.profile = 'typo'
    expect_error(lambda: resolve_wyt_data_profile(bad), 'Unknown UMI data profile')
    with tempfile.TemporaryDirectory() as temporary:
        bad = plain_config(initial)
        bad.datasets.vla_data.assets_root = temporary
        source = Path(initial.datasets.vla_data.assets_root)/'wyt_input_plan.json'
        (Path(temporary)/source.name).write_bytes(source.read_bytes()+b'\n')
        expect_error(lambda: resolve_wyt_data_profile(bad), 'SHA256')
    # Compact and expanded configs generate byte-identical physical/normalized labels.
    loaders = []
    for label,cfg in (('legacy',legacy),('compact',compact)):
        cfg.output_dir = str(output/label)
        cfg.datasets.vla_data.num_workers = 0
        loaders.append(make_wyt_umi_dataloader(cfg))
    a,b = (loader.dataset for loader in loaders)
    indices = [0,1,len(a)-1] + np.random.default_rng(42).integers(0,len(a),5).tolist()
    for index in indices:
        raw_a,raw_b = a.raw_dataset.read_lowdim(index),b.raw_dataset.read_lowdim(index)
        assert raw_a['lang'] == raw_b['lang']
        for key in ('state','action'):
            np.testing.assert_array_equal(raw_a[key],raw_b[key])
        sample_a,sample_b = a._normalize_sample(raw_a),b._normalize_sample(raw_b)
        for key in ('state','action'):
            np.testing.assert_array_equal(sample_a[key],sample_b[key])
    full_a,full_b = a[0],b[0]
    for image_a,image_b in zip(full_a['image'],full_b['image'],strict=True):
        np.testing.assert_array_equal(np.asarray(image_a),np.asarray(image_b))
    # Disabling the model's state input cannot alter labels in the loader.
    off = plain_config(initial)
    off.output_dir = str(output/'state_off')
    off.framework.action_model.use_state = False
    off.datasets.vla_data.num_workers = 0
    loader_off = make_wyt_umi_dataloader(off)
    sample_off = loader_off.dataset[0]
    for key in ('state','action'):
        np.testing.assert_array_equal(sample_off[key],full_b[key])
    for loader in loaders+[loader_off]:
        loader.dataset.close()
    artifact = json.loads((Path(initial.datasets.vla_data.assets_root)/'provenance.json').read_text())
    for name,digest in artifact['module_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    # Inference and resume snapshots must contain the complete resolved rules.
    snapshot = output/'resolved_config.txt'
    snapshot.write_text(OmegaConf.to_yaml(compact,resolve=True))
    saved = OmegaConf.create(snapshot.read_text())
    assert saved.datasets.vla_data.action_pose_repr == RELATIVE_EEF
    assert len(saved.datasets.vla_data.stages) == 5
    assert saved.datasets.vla_data.language_window_policy == 'same_segment'
    # Existing smoke YAML uses the same pinned recipe and its original short budget.
    smoke = OmegaConf.load('examples/umi/train_files/starvla_dualstream_umi_smoke.yaml')
    resolve_umi_stages(smoke,1)
    assert smoke.trainer.max_train_steps == 10
    assert smoke.datasets.vla_data.stage_global_batch_size == 16
    result = dict(yaml_data_keys=7,expanded_settings_equal=True,legacy_resume_identity_equal=True,
        relative_eef_when_omitted=True,world_rejected=True,unknown_profile_rejected=True,
        altered_plan_rejected=True,real_windows_bitwise_equal=len(indices),
        current_three_images_bitwise_equal=True,state_ablation_labels_equal=True,
        wyt_modules_unchanged=8,full_config_snapshot=True,stages=5,
        formal_updates_global2048=77732,smoke_effective_batch=16)
    (output/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
