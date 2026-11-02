"""StarVLA adapter around WYT's frozen, unmodified UMI reader modules."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch.distributed as dist
from omegaconf import OmegaConf


def plain_config(cfg):
    cfg = getattr(cfg, '_cfg', cfg)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


# This profile selects the verified WYT Oct20 run's immutable input plan.
# WYT's raw reader remains byte-identical; only local paths/defaults resolve here.
WYT_RELATIVE_PROFILE = 'wyt_umi_relative_v1'
WYT_PLAN_SHA256 = '31a47613300b9c2426f3b93386c41796f6d0aca3e49beeedbc0c164441c29901'
PROFILE_METADATA = {'profile', 'assets_root', 'umi_root', 'profile_plan_sha256'}
RELATIVE_EEF = 'current_eef_relative'


def resolve_wyt_data_profile(cfg):
    """Expand the concise YAML before stage planning or reader construction.

    Reader tuning and immutable dataset rules come from WYT's pinned JSON plan.
    Explicit YAML/CLI values override defaults, except the relative-EEF contract.
    Expanded legacy configs are accepted without requiring a profile selector.
    """
    data = cfg.datasets.vla_data
    if data.get('dataset_py') != 'umi_indexed':
        return
    supplied = OmegaConf.to_container(getattr(data, '_cfg', data), resolve=True)
    if supplied.get('action_pose_repr', RELATIVE_EEF) != RELATIVE_EEF:
        raise ValueError('This UMI pretraining path requires action_pose_repr=current_eef_relative; world actions are forbidden')
    profile = supplied.get('profile')
    if profile is None:
        if supplied.get('action_pose_repr') != RELATIVE_EEF:
            raise ValueError('Expanded UMI configs must explicitly declare current_eef_relative, or select the WYT relative profile')
        return
    if profile != WYT_RELATIVE_PROFILE:
        raise ValueError(f'Unknown UMI data profile {profile!r}; expected {WYT_RELATIVE_PROFILE!r}')
    for key in ('assets_root', 'umi_root'):
        if not supplied.get(key):
            raise ValueError(f'{profile} requires {key}')
    assets = Path(supplied['assets_root']).expanduser().resolve()
    umi_root = Path(supplied['umi_root']).expanduser().resolve()
    encoded = (assets / 'wyt_input_plan.json').read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != WYT_PLAN_SHA256:
        raise ValueError('WYT input plan SHA256 differs from the versioned relative-EEF profile')
    plan = json.loads(encoded)
    defaults = dict(plan['data'])
    if defaults.get('action_pose_repr') != RELATIVE_EEF:
        raise ValueError('WYT profile does not declare current_eef_relative')
    # The pinned plan uses /public/roban_umi. Remap only these three physical
    # source locations; source files and immutable index identities are untouched.
    for key in ('source_root', 'video_root', 'quaternion_signs'):
        suffix = Path(defaults[key]).relative_to('/public/roban_umi')
        defaults[key] = str(umi_root / suffix)
    stages = [dict(name=s['name'], index_dir=str(assets / 'views' / s['name']),
                   num_windows=s['mapped_5hz_windows']) for s in plan['stages']]
    defaults.update(
        dataset_py='umi_indexed', allowed_video_roots=[],
        normalization_statistics=str(assets / 'relative' / Path(defaults['normalization_statistics']).name),
        normalization_experiment_contract=str(assets / 'relative' / Path(defaults['normalization_experiment_contract']).name),
        normalization_purpose=plan['purpose'],
        index_dir=stages[0]['index_dir'], active_stage=stages[0]['name'], stages=stages,
        validation_index_dir=str(assets / 'views' / plan['evaluation']['name']),
        evaluation_batch_size=16, per_device_batch_size=plan['training']['batch_size'],
        num_workers=plan['training']['num_workers'], prefetch_factor=plan['training']['prefetch_factor'],
        return_metadata=True, performance_diagnostics=False, seed=int(cfg.get('seed', 42)),
    )
    # Lists such as stages/cameras are replaced as a whole by explicit overrides.
    resolved = OmegaConf.merge(OmegaConf.create(defaults), OmegaConf.create(supplied))
    resolved.profile_plan_sha256 = digest
    cfg.datasets.vla_data = resolved



def export_statistics(dataset, output_dir):
    """Export the actual FP32 scales, including std-floor handling."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in ('state', 'action'):
        mean, scale = dataset.normalizer._params[name]
        stats[name] = dict(normalization='mean_std', mean=mean.tolist(), std=scale.tolist())
    stats['meta'] = dataset.provenance()
    path = output / 'dataset_statistics.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'umi_indexed': stats}, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def inverse_exported_actions(value, stats):
    """Invert exported FP32 scaling, preserving continuous quaternion/gripper values."""
    value = np.asarray(value, dtype=np.float32)
    mean, scale = (np.asarray(stats[key], dtype=np.float32) for key in ('mean', 'std'))
    if mean.shape != (16,) or scale.shape != (16,) or value.shape[-1] != 16:
        raise ValueError('Indexed UMI requires 16D mean/std actions')
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid indexed UMI normalization parameters')
    return value * scale + mean


def make_wyt_umi_dataloader(cfg):
    from .umi_indexed_dataset import make_umi_dataloader
    local = plain_config(cfg)
    resolve_wyt_data_profile(local)
    output = Path(local.output_dir)
    stage_name = local.datasets.vla_data.get('active_stage', 'train')
    local.output_dir = str(output / 'data_access' / stage_name)
    loader = make_umi_dataloader(local)
    if getattr(loader.dataset, "action_pose_repr", None) != RELATIVE_EEF:
        loader.dataset.close()
        raise RuntimeError("UMI reader failed to produce current_eef_relative actions")
    loader.umi_dataset = loader.dataset
    loader.umi_sampler = loader.sampler
    if stage_name != 'validation_monitor' and (not dist.is_initialized() or dist.get_rank() == 0):
        export_statistics(loader.dataset, output)
    return loader


def resolve_umi_stages(cfg, world_size):
    """Resolve one-pass budgets before scheduler construction/config snapshots.

    Each stage drops only its incomplete final optimizer update. A user-supplied
    max_train_steps can cap this pass for smoke tests; null means all five stages.
    """
    from .umi_indexed_dataset import compute_view_fingerprint
    resolve_wyt_data_profile(cfg)
    data = cfg.datasets.vla_data
    if data.dataset_py != 'umi_indexed' or not data.get('stages'):
        return
    stride = int(data.anchor_stride_frames)
    if stride < 1:
        raise ValueError('anchor_stride_frames must be positive')
    global_batch = (int(data.per_device_batch_size) * int(world_size)
                    * int(cfg.trainer.gradient_accumulation_steps))
    if global_batch < 1:
        raise ValueError('Global batch size must be positive')
    stages = []
    for stage in data.stages:
        directory = Path(stage.index_dir)
        meta = json.loads((directory / 'meta.json').read_text())
        ranges = np.load(directory / 'ranges.npy', mmap_mode='r', allow_pickle=False)
        lengths = ranges[:, 2] - ranges[:, 1]
        count = int(np.sum((lengths + stride - 1) // stride, dtype=np.int64))
        if stage.get('num_windows') is not None and int(stage.num_windows) != count:
            raise ValueError(f'Window count differs for {stage.name}')
        updates = count // global_batch
        if updates < 1:
            raise ValueError(f'Stage {stage.name} is smaller than one global optimizer batch')
        stages.append(dict(name=str(stage.name), index_dir=str(directory), num_windows=count,
                           updates=updates, view_fingerprint=compute_view_fingerprint(meta)))
    if len({s['name'] for s in stages}) != len(stages):
        raise ValueError('Stage names must be unique')
    data.stages = stages
    data.index_dir = stages[0]['index_dir']
    data.active_stage = stages[0]['name']
    budget = sum(s['updates'] for s in stages)
    requested = cfg.trainer.get('max_train_steps')
    if requested is None:
        cfg.trainer.max_train_steps = budget
    elif not 0 < int(requested) <= budget:
        raise ValueError(f'max_train_steps must be in [1,{budget}], or null for one pass')
    data.stage_global_batch_size = global_batch
    data.stage_total_updates = budget


def stage_identity(cfg, world_size):
    """Resume identity excludes transport/worker tuning but includes semantics."""
    local = plain_config(cfg)
    resolve_wyt_data_profile(local)
    data = OmegaConf.to_container(local.datasets.vla_data, resolve=True)
    excluded = {'index_dir', 'active_stage', 'num_workers', 'prefetch_factor', 'pin_memory',
                'persistent_workers', 'multiprocessing_context', 'row_cache_mib', 'row_cache_items',
                'file_cache_items', 'video_cache_items', 'episode_cache_items',
                'performance_diagnostics', 'return_metadata', 'write_access_record',
                'validation_index_dir', 'evaluation_batch_size'} | PROFILE_METADATA
    identity = {k: v for k, v in data.items() if k not in excluded}
    identity['stages'] = [{k: v for k, v in s.items() if k != 'index_dir'} for s in identity['stages']]
    for key in ('normalization_statistics', 'normalization_experiment_contract'):
        identity[key] = hashlib.sha256(Path(data[key]).read_bytes()).hexdigest()
    signs = Path(data['quaternion_signs']) / 'manifest.json'
    identity['quaternion_signs'] = hashlib.sha256(signs.read_bytes()).hexdigest()
    identity.update(world_size=int(world_size), gradient_accumulation_steps=int(local.trainer.gradient_accumulation_steps),
                    use_state=bool(local.framework.action_model.get('use_state', True)))
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
