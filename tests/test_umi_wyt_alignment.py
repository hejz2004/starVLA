"""Real-data audit plus stage/committed-cursor integration checks (no full VLM)."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from starVLA.dataloader.umi_wyt import make_wyt_umi_dataloader, plain_config, resolve_umi_stages
from starVLA.training.trainer_utils.umi_stage_data import UMIStageData


CONFIG = 'examples/umi/train_files/starvla_dualstream_umi_v2.yaml'


def data_audit(output):
    cfg = OmegaConf.load(CONFIG)
    cfg.output_dir = str(output)
    resolve_umi_stages(cfg, 64)
    assert cfg.trainer.max_train_steps == sum(s.updates for s in cfg.datasets.vla_data.stages)
    assert cfg.datasets.vla_data.stage_global_batch_size == 2048
    reference = plain_config(cfg)
    reference.datasets.vla_data.per_device_batch_size = 24
    reference.trainer.max_train_steps = None
    resolve_umi_stages(reference, 64)
    assert reference.trainer.max_train_steps == 103643
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.shuffle = False
    cfg.datasets.vla_data.audit_window_contract = True
    cfg.datasets.vla_data.verify_quaternion_continuity = True
    loader = make_wyt_umi_dataloader(cfg)
    dataset, raw = loader.dataset, loader.dataset.raw_dataset
    assert len(dataset) == 31842655
    rng = np.random.default_rng(42)
    indices = [0, 1, len(raw)-1] + rng.integers(0, len(raw), 29).tolist()
    records = []
    for index in indices:
        lowdim = raw.read_lowdim(index)
        trace = lowdim['trace']
        frames = np.array([trace['frame_index']] + trace['future_frame_indices'])
        assert np.array_equal(np.diff(frames), np.ones(16))
        assert trace['episode_index'] not in cfg.datasets.vla_data.excluded_episode_indices
        sample = dataset._normalize_sample({k: lowdim[k] for k in ('state', 'action', 'lang')})
        physical = dataset.normalizer.inverse_action(sample['action'])
        for start in (0, 8):
            current = lowdim['state'][0, start:start+8]
            future = lowdim['action'][:, start:start+8]
            rotation = Rotation.from_quat(current[3:7])
            expected_position = rotation.inv().apply(future[:, :3].astype(np.float64)-current[:3].astype(np.float64))
            expected_quaternion = (rotation.inv() * Rotation.from_quat(future[:, 3:7])).as_quat(canonical=False)
            np.testing.assert_allclose(physical[:, start:start+3], expected_position, atol=2e-6)
            np.testing.assert_allclose(physical[:, start+3:start+7], expected_quaternion, atol=2e-6)
            np.testing.assert_allclose(physical[:, start+7], future[:, 7], atol=2e-6)
        np.testing.assert_allclose(dataset.normalizer.inverse_state(sample['state']), lowdim['state'], atol=2e-6)
        records.append(dict(index=index, episode=trace['episode_index'], anchor=trace['frame_index'], lang=sample['lang']))
    sample = dataset[0]
    assert len(sample['image']) == 3 and all(image.size == (224, 224) for image in sample['image'])
    assert sample['state'].shape == (1, 16) and sample['action'].shape == (16, 16)
    artifact = json.loads((Path(cfg.datasets.vla_data.index_dir).parents[1] / 'provenance.json').read_text())
    for relative, digest in artifact['module_sha256'].items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == digest
    # Verify all copied immutable payloads, deduplicating the shared catalog.
    checked = set()
    for stage in cfg.datasets.vla_data.stages:
        directory = Path(stage.index_dir)
        meta = json.loads((directory / 'meta.json').read_text())
        for name, contract in meta['artifacts'].items():
            if name not in ('ranges.npy', 'cumulative.npy', 'metadata.sqlite3'):
                continue
            path = directory / name
            inode = (path.stat().st_dev, path.stat().st_ino)
            if inode in checked:
                continue
            checked.add(inode)
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert digest == contract['sha256'], str(path)
    dataset.close()
    return dict(real_lowdim_windows=len(records), full_image_windows=1, records=records,
                copied_payloads_sha256_verified=len(checked), wyt_modules_verified=8,
                updates_global1536=reference.trainer.max_train_steps, updates_global2048=cfg.trainer.max_train_steps)


def stage_checks(output, world=False):
    accelerator = Accelerator(cpu=not world, gradient_accumulation_steps=2)
    cfg = OmegaConf.load(CONFIG)
    cfg.output_dir = str(output)
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.shuffle = False
    cfg.trainer.gradient_accumulation_steps = 2
    resolve_umi_stages(cfg, accelerator.num_processes)
    # Engineering budget caps exercise real view transitions without reading 159M windows.
    cfg.datasets.vla_data.stages = list(cfg.datasets.vla_data.stages[:2])
    cfg.datasets.vla_data.stages[0].updates = 2
    cfg.datasets.vla_data.stages[1].updates = 1
    raw = make_wyt_umi_dataloader(cfg)
    prepared = accelerator.prepare_data_loader(raw, device_placement=False)
    data = UMIStageData(cfg, accelerator, raw, prepared)
    initial_rng = torch.get_rng_state().clone()
    data.start()
    assert torch.equal(torch.get_rng_state(), initial_rng)
    for micro in range(2):
        batch = data.next()
        assert batch[0]['umi_metadata']['dataset_index'] == micro * accelerator.num_processes + accelerator.process_index
        data.commit(sync_gradients=(micro == 1))
    saved = data.state_dict()
    assert saved['global_sample_cursor'] == 2 * accelerator.num_processes
    data.close()
    raw = make_wyt_umi_dataloader(cfg)
    data = UMIStageData(cfg, accelerator, raw, accelerator.prepare_data_loader(raw, device_placement=False))
    data.load_state_dict(saved)
    data.start()
    for micro in range(2):
        batch = data.next()
        assert batch[0]['umi_metadata']['dataset_index'] == (2+micro)*accelerator.num_processes + accelerator.process_index
        data.commit(sync_gradients=(micro == 1))
    boundary = data.state_dict()
    assert boundary['stage_updates'] == 2
    data.close()
    # Resume exactly at a stage boundary: next batch belongs to stage 2, index 0.
    raw = make_wyt_umi_dataloader(cfg)
    data = UMIStageData(cfg, accelerator, raw, accelerator.prepare_data_loader(raw, device_placement=False))
    data.load_state_dict(boundary)
    data.start()
    assert data.stage_index == 1
    for micro in range(2):
        batch = data.next()
        assert batch[0]['umi_metadata']['dataset_index'] == micro*accelerator.num_processes + accelerator.process_index
        data.commit(sync_gradients=(micro == 1))
    state2 = data.state_dict()
    data.close()
    # Changed sampler/data must fail strict restore.
    cfg.datasets.vla_data.seed = 43
    raw = make_wyt_umi_dataloader(cfg)
    invalid = UMIStageData(cfg, accelerator, raw, accelerator.prepare_data_loader(raw, device_placement=False))
    try:
        invalid.load_state_dict(state2)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed data identity was accepted')
    invalid.close()
    accelerator.wait_for_everyone()
    return dict(stage_transition=True, mid_stage_resume=True, boundary_resume=True,
                incompatible_resume_rejected=True, ranks=accelerator.num_processes,
                accumulation=2)


def model_checks(output):
    from unittest.mock import patch
    from starVLA.model.tools import FrameworkTools
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    from starVLA.training.trainer_utils.config_tracker import wrap_config
    cfg = OmegaConf.load(CONFIG)
    cfg.output_dir = str(output)
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.shuffle = False
    loader = make_wyt_umi_dataloader(wrap_config(cfg))
    batch = loader.dataset[0]
    exported = json.loads((output / 'dataset_statistics.json').read_text())
    # Values outside [-1,1], especially quaternion channel 6 and grippers, must
    # retain continuous scaling through both inference entry points.
    extreme = np.full((16, 16), 3.25, dtype=np.float32)
    expected = loader.dataset.normalizer.inverse_action(extreme)
    np.testing.assert_array_equal(FrameworkTools.unnormalize_actions(extreme, exported['umi_indexed']['action']), expected)
    with patch('deployment.model_server.policy_norm_processor.read_mode_config', return_value=({}, exported)):
        processor = PolicyNormProcessor('unused_checkpoint.pt')
    np.testing.assert_array_equal(processor.unapply_actions(extreme), expected)
    # A small genuine DiT action head exercises both state branches and gradients.
    results = {}
    actions, state = torch.from_numpy(batch['action'])[None], torch.from_numpy(batch['state'])[None]
    shared = None
    for enabled in (True, False):
        action_config = dict(action_dim=16, state_dim=16, use_state=enabled, action_horizon=16,
            num_inference_timesteps=2, num_target_vision_tokens=2, add_pos_embed=True, max_seq_len=64,
            noise_beta_alpha=1.5, noise_beta_beta=1.0, noise_s=0.999, num_timestep_buckets=1000,
            decoder_input_norm='layer_norm', diffusion_model_cfg=dict(num_layers=2,
                input_embedding_dim=32, cross_attention_dim=32, attention_head_dim=16,
                num_attention_heads=2, dropout=0.0, interleave_self_attention=True,
                cross_condition_norm='layer_norm'))
        head = LayerwiseFlowmatchingActionHead(OmegaConf.create({'framework': {'action_model': action_config}}))
        if shared is None:
            shared = {k: v.clone() for k, v in head.state_dict().items() if not k.startswith('state_encoder.')}
        else:
            head.load_state_dict(shared, strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            loss = head([torch.randn(1, 8, 32) for _ in range(2)], actions, state)
            assert torch.isfinite(loss)
            loss.backward()
            if enabled:
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.state_encoder.parameters())
            else:
                assert head.state_encoder is None
                assert not any(k.startswith('state_encoder.') for k in head.state_dict())
            optimizer.step()
            losses.append(float(loss.detach()))
        results[str(enabled)] = losses
    loader.dataset.close()
    return dict(mean_std_inverse_without_clipping=True, state_encoder_gradient=True,
                no_state_encoder_when_disabled=True, real_action_head_optimizer_steps=results)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(16, 16)

    def forward(self, examples):
        state = torch.from_numpy(np.stack([x['state'] for x in examples]))
        truth = torch.from_numpy(np.stack([x['action'] for x in examples]))
        value = self.projection(state) + torch.randn_like(state) * 0.01
        return {'action_loss': (value-truth).square().mean()}


def checkpoint_checks(output):
    from starVLA.training.train_starvla import VLATrainer, accelerator
    from starVLA.training.trainer_utils.config_tracker import wrap_config
    cfg = OmegaConf.load(CONFIG)
    cfg.output_dir = str(output)
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.shuffle = False
    cfg.trainer.gradient_accumulation_steps = 2
    resolve_umi_stages(cfg, 1)
    cfg.datasets.vla_data.stages = list(cfg.datasets.vla_data.stages[:2])
    cfg.datasets.vla_data.stages[0].updates = 2
    cfg.datasets.vla_data.stages[1].updates = 1
    cfg.trainer.max_train_steps = 3
    model = TinyTrainingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1/(step+1))
    raw = make_wyt_umi_dataloader(cfg)
    trainer = VLATrainer(wrap_config(cfg), model, raw, optimizer, scheduler, accelerator)
    trainer.prepare_training()
    trainer._create_data_iterators()

    def update():
        identifiers = []
        for _ in range(2):
            batch = trainer._get_next_batch()
            identifiers.append(batch[0]['umi_metadata']['dataset_index'])
            trainer._train_step(batch)
            trainer.umi_stage_data.commit(accelerator.sync_gradients)
            if accelerator.sync_gradients:
                trainer.completed_steps += 1
        return identifiers

    assert update() == [0, 1]
    saved_weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer._save_checkpoint()
    saved_cursor = trainer.state_dict()['umi_data']['global_sample_cursor']
    assert saved_cursor == 2
    assert update() == [2, 3]
    uninterrupted = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    # Restore through the actual trainer/Accelerate path, including AdamW,
    # scheduler, RNG and registered custom data state, then replay update 2.
    trainer._strict_ckpt_dir = str(output / 'checkpoints/steps_1')
    trainer.completed_steps = 1
    trainer._load_strict_checkpoint()
    assert trainer.lr_scheduler.scheduler.last_epoch == 1
    assert trainer.state_dict()['umi_data']['global_sample_cursor'] == 2
    for key, value in saved_weights.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value, rtol=0, atol=0)
    trainer._create_data_iterators()
    assert update() == [2, 3]
    for key, value in uninterrupted.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value, rtol=0, atol=0)
    batch = trainer._get_next_batch()
    assert trainer.umi_stage_data.stage_index == 1
    assert batch[0]['umi_metadata']['dataset_index'] == 0
    trainer.umi_stage_data.commit(False)
    batch = trainer._get_next_batch()
    trainer.umi_stage_data.commit(True)
    trainer.umi_stage_data.close()
    return dict(actual_accelerate_checkpoint=True, optimizer_scheduler_rng_restored=True,
                resumed_update_weights_bitwise_equal=True, cursor=saved_cursor,
                next_stage_after_restore=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['data', 'stage', 'distributed', 'model', 'checkpoint'], default='data')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'data':
        result = data_audit(output)
    elif args.mode == 'model':
        result = model_checks(output)
    elif args.mode == 'checkpoint':
        result = checkpoint_checks(output)
    else:
        result = stage_checks(output, args.mode == 'distributed')
    result_path = output / (args.mode + '.json')
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
