"""LIBERO native-command goals, anchored at each training image, binary gripper."""
import hashlib
import json
from pathlib import Path
import numpy as np
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.libero_action_contract import (
    VERSION, LOSS_MASK, GoalBinaryNormalizer, encode_goals,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LiberoActionGoalBinaryDataset(LeRobotSingleDataset):
    def __init__(self, *args, **kwargs):
        cfg = kwargs['data_cfg']
        if cfg.get('action_type') != VERSION or cfg.get('include_state', False) not in (False, 'False'):
            raise ValueError('Binary-goal data needs its versioned action contract and include_state=false')
        if cfg.get('action_mode', 'abs') != 'abs':
            raise ValueError('Goals are already anchored; generic action_mode must be abs')
        self.goal_root = Path(cfg.controller_goal_root)
        self.normalizer = GoalBinaryNormalizer(cfg.normalization_statistics)
        ready = json.loads((self.goal_root / 'READY.json').read_text())
        manifest_path = self.goal_root / 'manifest.json'
        if ready.get('contract') != VERSION or ready.get('status') != 'complete':
            raise ValueError('No complete matching goal-label generation receipt')
        if sha(cfg.normalization_statistics) != ready['statistics_sha256'] or sha(manifest_path) != ready['manifest_sha256']:
            raise ValueError('Label manifest/statistics differ from generation receipt')
        manifest = json.loads(manifest_path.read_text())
        self.goal_records = {(x['dataset'], x['episode']): x for x in manifest['records']}
        self._goal_episode, self._goal_payload = None, None
        self._verified_sources = set()
        super().__init__(*args, **kwargs)

    def checkpoint_action_statistics(self):
        return dict(self.normalizer.payload['action'])

    def _get_all_steps(self):
        horizon = len(self.modality_configs['action'].delta_indices)
        return [(int(episode), index) for episode, length in zip(self.trajectory_ids, self.trajectory_lengths)
                for index in range(max(0, int(length) - horizon + 1))]

    def get_step_data(self, trajectory_id, base_index):
        data = super().get_step_data(trajectory_id, base_index)
        if self._goal_episode != trajectory_id:
            record = self.goal_records[(self.dataset_name, int(trajectory_id))]
            path = self.goal_root / record['label']
            if trajectory_id not in self._verified_sources:
                source = self.dataset_path / 'data' / f'chunk-{int(trajectory_id)//1000:03d}' / f'episode_{int(trajectory_id):06d}.parquet'
                if sha(path) != record['label_sha256'] or sha(source) != record['source_sha256']:
                    raise ValueError('Training data changed after goal-label generation')
                self._verified_sources.add(trajectory_id)
            with np.load(path, allow_pickle=False) as archive:
                self._goal_payload = {k: archive[k].copy() for k in ('goals', 'anchor_position', 'anchor_quaternion')}
            self._goal_episode = trajectory_id
        payload = self._goal_payload
        horizon = len(self.modality_configs['action'].delta_indices)
        goals = payload['goals'][base_index:base_index+horizon]
        if len(goals) != horizon:
            raise ValueError('Incomplete native-command goal window')
        physical = encode_goals(goals, payload['anchor_position'][base_index], payload['anchor_quaternion'][base_index])
        data['_action_goal_binary'] = self.normalizer.normalize_action(physical)
        return data

    def _pack_sample(self, data):
        sample = super()._pack_sample(data)
        sample['action'] = np.asarray(data['_action_goal_binary'], dtype=np.float32)
        sample['action_mask'] = np.broadcast_to(LOSS_MASK, sample['action'].shape).copy()
        return sample


class LiberoActionGoalBinaryDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = ['video.primary_image', 'video.wrist_image']
    state_keys = ['state.x', 'state.y', 'state.z', 'state.roll', 'state.pitch', 'state.yaw', 'state.pad', 'state.gripper']
    action_keys = ['action.x', 'action.y', 'action.z', 'action.roll', 'action.pitch', 'action.yaw', 'action.gripper']
    language_keys = ['annotation.human.action.task_description']
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {name: ModalityConfig(delta_indices=indices, modality_keys=keys) for name, indices, keys in (
            ('video', self.observation_indices, self.video_keys), ('state', self.state_indices, self.state_keys),
            ('action', self.action_indices, self.action_keys), ('language', [0], self.language_keys))}

    def transform(self):
        return ComposedModalityTransform(transforms=[])

    def make_dataset(self, **kwargs):
        if int(kwargs['data_cfg'].get('action_horizon', 16)) != 16:
            raise ValueError('This generated label bundle declares action_horizon=16')
        return LiberoActionGoalBinaryDataset(**kwargs)
