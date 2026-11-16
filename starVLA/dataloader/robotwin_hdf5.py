"""Native RoboTwin clean trajectories with UMI's anchored, future-16 EEF contract.

Only an offline JSON index/statistics artifact is generated. Images and physical
poses remain in the original HDF5 files. No LeRobot conversion is involved.
"""
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .umi_pose import actions_to_current_frame, actions_to_world_frame, action_pose_details
from .umi_quaternion import continuous_signs
from .umi_normalization import WeightedMoments, content_fingerprint

VERSION = 'robotwin-native-umi-relative-eef-v1'
CAMERAS = ('head_camera', 'left_camera', 'right_camera')
HORIZON = 16


def contract():
    return dict(action_pose_details(), version=VERSION, dataset_type='robotwin_hdf5',
                state_representation='absolute_xyz_xyzw_gripper',
                source_quaternion_order='wxyz', hand_order=['left', 'right'],
                gripper='absolute_future_command_0_closed_1_open; not metres',
                pose_reference='RoboTwin get_arm_pose EE frame; no additional TCP offset',
                language_split='seen', camera_order=list(CAMERAS),
                temporal_sampling='native consecutive saved observations; no fixed-Hz claim',
                tail_policy='drop incomplete 16-future windows; no padding')


def poses_from_observation(endpose, *, continuous=False):
    """RoboTwin xyz+wxyz -> UMI left xyz+xyzw+g, right xyz+xyzw+g."""
    hands = []
    for hand in ('left', 'right'):
        pose = np.asarray(endpose[f'{hand}_endpose'], dtype=np.float64)
        grip = np.asarray(endpose[f'{hand}_gripper'], dtype=np.float64)
        if pose.shape[-1] != 7 or grip.shape != pose.shape[:-1]:
            raise ValueError('Invalid RoboTwin EEF/gripper shape')
        quat = pose[..., [4, 5, 6, 3]].copy()
        if not np.isfinite(pose).all() or not np.isfinite(grip).all():
            raise ValueError('Nonfinite source pose/gripper')
        if np.any(np.linalg.norm(quat, axis=-1) <= 1e-8):
            raise ValueError('Zero source quaternion')
        if np.any((grip < -1e-6) | (grip > 1 + 1e-6)):
            raise ValueError('Expected RoboTwin gripper command in [0,1]')
        if continuous:
            quat *= continuous_signs(quat)[:, None]
        else:
            # Single observations have the same deterministic initial gauge as UMI.
            ordered = quat[..., [3, 0, 1, 2]]
            first = np.take_along_axis(ordered, np.argmax(ordered != 0, axis=-1)[..., None], -1)
            quat *= np.where(first < 0, -1, 1)
        hands.append(np.concatenate((pose[..., :3], quat, grip[..., None]), axis=-1))
    return np.concatenate(hands, axis=-1).astype(np.float32)


def read_poses(h5):
    data = {f'{h}_{k}': h5[f'endpose/{h}_{k}'][:] for h in ('left', 'right')
            for k in ('endpose', 'gripper')}
    return poses_from_observation(data, continuous=True)


def to_environment_actions(anchor, relative):
    world = actions_to_world_frame(np.asarray(anchor).reshape(1, 16), relative)
    out = world.copy()
    for start in (0, 8):
        out[..., start+3:start+7] = world[..., [start+6, start+3, start+4, start+5]]
        out[..., start+7] = np.clip(out[..., start+7], 0, 1)
    return out


def _identity(path):
    s = path.stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare_assets(source_root, destination, *, expected_tasks=50, expected_episodes=50,
                   image_encoding='legacy_cv2_rgb', task_names=None):
    """Fit FP64 mean/std over exactly the complete windows used for training."""
    import h5py
    root = Path(source_root).resolve()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Preserve existing assets: {destination}')
    if image_encoding not in ('legacy_cv2_rgb', 'standard_rgb'):
        raise ValueError('Explicit image encoding required')
    tasks = sorted(root.glob('dataset/*/aloha-agilex_clean_50'))
    if task_names is not None:
        selected = list(task_names)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError('Task selection must be nonempty and unique')
        available = {p.parent.name for p in tasks}
        missing = set(selected) - available
        if missing:
            raise ValueError(f'Missing selected tasks: {sorted(missing)}')
        tasks = [p for p in tasks if p.parent.name in set(selected)]
    if len(tasks) != expected_tasks:
        raise ValueError(f'Expected {expected_tasks} clean tasks, found {len(tasks)}')
    moments = {key: WeightedMoments(16) for key in ('state', 'action')}
    records, lengths, duplicate_steps, motion_steps = [], [], 0, 0
    for taskdir in tasks:
        task = taskdir.parent.name
        paths = list((taskdir/'data').glob('episode*.hdf5'))
        paths.sort(key=lambda p: int(p.stem.removeprefix('episode')))
        if {p.name for p in paths} != {f'episode{i}.hdf5' for i in range(expected_episodes)}:
            raise ValueError(f'Incomplete episode IDs in {task}')
        for path in paths:
            language_path = taskdir/'instructions'/(path.stem+'.json')
            instructions = json.loads(language_path.read_text()).get('seen', [])
            if not instructions or any(not isinstance(s, str) or not s.strip() for s in instructions):
                raise ValueError(f'Missing seen instructions: {language_path}')
            with h5py.File(path, 'r') as h5:
                poses = read_poses(h5)
                n = len(poses)
                if n <= HORIZON:
                    raise ValueError(f'Too short for full future-16 window: {path}')
                for camera in CAMERAS:
                    column = h5[f'observation/{camera}/rgb']
                    if column.shape != (n,):
                        raise ValueError(f'Image/pose length mismatch: {path}, {camera}')
                    decode_image(column[0], image_encoding)
                anchors = poses[:n-HORIZON, None, :]
                moments['state'].update(anchors[:, 0])
                for offset in range(1, HORIZON+1):
                    target = poses[offset:n-HORIZON+offset, None, :]
                    relative = actions_to_current_frame(anchors, target)
                    moments['action'].update(relative[:, 0])
                step = actions_to_current_frame(poses[:-1, None], poses[1:, None])[:, 0]
                moving = np.maximum(np.linalg.norm(step[:, :3], axis=-1),
                                    np.linalg.norm(step[:, 8:11], axis=-1)) > 1e-6
                motion_steps += int(moving.sum())
                duplicate_steps += int((np.max(np.abs(poses[1:]-poses[:-1]), axis=-1)<1e-8).sum())
            records.append(dict(task=task, episode=int(path.stem.removeprefix('episode')),
                path=str(path.relative_to(root)), source_identity=_identity(path),
                instructions_path=str(language_path.relative_to(root)),
                instructions_sha256=hashlib.sha256(language_path.read_bytes()).hexdigest(),
                instructions=instructions, frames=n, windows=n-HORIZON))
            lengths.append(n)
        print(f'Indexed {task}: {len(paths)} episodes; total {len(records)}', flush=True)
    scaling = {}
    for key, accumulator in moments.items():
        stats = accumulator.statistics()
        scaling[key] = dict(normalization='mean_std', mean=stats['mean'], std=stats['scale'],
                            count=stats['count'], constant_dimensions=stats['constant_dimensions'])
    payload = dict(contract=contract(), source_root=str(root), image_encoding=image_encoding,
                   episodes=records, statistics=scaling,
                   summary=dict(tasks=len(tasks), episodes=len(records), frames=sum(lengths),
                                windows=sum(n-HORIZON for n in lengths),
                                future_offset=1, horizon=HORIZON,
                                moving_adjacent_pose_pairs=motion_steps,
                                exactly_duplicate_adjacent_pose_pairs=duplicate_steps,
                                t_target_relative_pose='identically zero translation + identity rotation',
                                t_plus_1_target='next recorded achieved EEF pose; not an original control command'))
    payload['fingerprint'] = content_fingerprint(payload)
    _atomic_json(destination, payload)
    return payload


def decode_image(value, encoding, image_size=None):
    encoded = Image.open(io.BytesIO(bytes(value)))
    if encoding == 'legacy_cv2_rgb' and encoded.info.get('comment') == b'XPL-RGB1':
        raise ValueError('Marked standard RGB JPEG conflicts with legacy dataset encoding')
    image = encoded.convert('RGB')
    if encoding == 'legacy_cv2_rgb':
        # Legacy imencode received simulator RGB as BGR: PIL sees swapped R/B.
        image = Image.fromarray(np.asarray(image)[..., ::-1].copy())
    elif encoding != 'standard_rgb':
        raise ValueError(f'Unknown JPEG color contract: {encoding}')
    return image.resize(tuple(image_size)) if image_size else image


class RobotwinNormalizer:
    def __init__(self, statistics):
        self.statistics = statistics
        self.params = {}
        for name in ('state', 'action'):
            record = statistics[name]
            mean, scale = (np.asarray(record[k], dtype=np.float32) for k in ('mean', 'std'))
            if record.get('normalization') != 'mean_std' or mean.shape != (16,) or scale.shape != (16,):
                raise ValueError('Expected fixed 16D mean/std')
            if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError('Invalid normalization statistics')
            self.params[name] = mean, scale

    def apply(self, values, name):
        mean, scale = self.params[name]
        return (np.asarray(values, dtype=np.float32)-mean)/scale

    def inverse(self, values, name='action'):
        mean, scale = self.params[name]
        return np.asarray(values, dtype=np.float32)*scale+mean


class RobotwinHDF5Dataset(Dataset):
    def __init__(self, assets_path, *, source_root=None, include_state=True,
                 image_size=(224, 224), max_open_files=8, return_metadata=False):
        self.assets_path = Path(assets_path)
        self.assets = json.loads(self.assets_path.read_text())
        body = {k:v for k,v in self.assets.items() if k != 'fingerprint'}
        if self.assets['fingerprint'] != content_fingerprint(body) or self.assets['contract'] != contract():
            raise ValueError('Changed or incompatible RoboTwin assets')
        self.root = Path(source_root or self.assets['source_root']).resolve()
        self.records = self.assets['episodes']
        self.ends = np.cumsum([x['windows'] for x in self.records], dtype=np.int64)
        self.normalizer = RobotwinNormalizer(self.assets['statistics'])
        self.include_state = include_state
        self.image_size = image_size
        self.max_open_files = int(max_open_files)
        if self.max_open_files < 1:
            raise ValueError('max_open_files must be positive')
        self.return_metadata = return_metadata
        self._files, self._pid = OrderedDict(), None

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        return dict(self.__dict__, _files=OrderedDict(), _pid=None)

    def close(self):
        for h5, _ in self._files.values():
            h5.close()
        self._files.clear()

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = int(np.searchsorted(self.ends, index, side='right'))
        return ep, int(index-(self.ends[ep-1] if ep else 0))

    def _episode(self, ep):
        import h5py
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if ep in self._files:
            self._files.move_to_end(ep)
            return self._files[ep]
        record = self.records[ep]
        path = (self.root/record['path']).resolve()
        if not path.is_relative_to(self.root) or _identity(path) != record['source_identity']:
            raise ValueError(f'Source changed after preparation: {path}')
        h5 = h5py.File(path, 'r')
        try:
            poses = read_poses(h5)
        except BaseException:
            h5.close()
            raise
        while len(self._files) >= self.max_open_files:
            self._files.popitem(last=False)[1][0].close()
        self._files[ep] = (h5, poses)
        return h5, poses

    def read_lowdim(self, index):
        ep, t = self.locate(index)
        _, poses = self._episode(ep)
        state = poses[t:t+1].copy()
        future = poses[t+1:t+1+HORIZON].copy()
        if len(future) != HORIZON:
            raise RuntimeError('Index crossed episode boundary')
        return state, future

    def __getitem__(self, index):
        ep, t = self.locate(index)
        h5, poses = self._episode(ep)
        state, future = poses[t:t+1], poses[t+1:t+1+HORIZON]
        record = self.records[ep]
        language = record['instructions'][np.random.randint(len(record['instructions']))]
        sample = dict(image=[decode_image(h5[f'observation/{cam}/rgb'][t], self.assets['image_encoding'],
                                          self.image_size) for cam in CAMERAS],
                      lang=language, robot_tag='robotwin_hdf5',
                      action=self.normalizer.apply(actions_to_current_frame(state, future), 'action'))
        if self.include_state:
            sample['state'] = self.normalizer.apply(state, 'state')
        if self.return_metadata:
            sample['robotwin_metadata'] = dict(task=record['task'], episode=record['episode'], anchor=t,
                                              target_indices=list(range(t+1,t+17)), source=record['path'])
        return sample

    def save_dataset_statistics(self, path):
        meta = dict(contract(), assets_fingerprint=self.assets['fingerprint'],
                    image_size=list(self.image_size), image_encoding=self.assets['image_encoding'],
                    include_state=self.include_state)
        _atomic_json(path, {'robotwin_hdf5': dict(self.assets['statistics'], meta=meta)})


def collate_fn(batch):
    return batch


def make_robotwin_dataloader(cfg):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    data = cfg.datasets.vla_data
    action = cfg.framework.action_model
    if (int(action.action_dim), int(action.state_dim), int(action.action_horizon)) != (16,16,16):
        raise ValueError('RoboTwin UMI interface requires 16D state/action and horizon=16')
    if data.get('action_pose_repr') != 'current_eef_relative':
        raise ValueError('RoboTwin native reader requires current_eef_relative')
    use_state = bool(action.get('use_state', True))
    if bool(data.get('include_state', use_state)) != use_state:
        raise ValueError('Model/data state input settings disagree')
    dataset = RobotwinHDF5Dataset(data.assets_path, source_root=data.get('data_root_dir'),
        include_state=use_state, image_size=tuple(data.get('image_size', [224,224])),
        max_open_files=data.get('max_open_files', 8), return_metadata=data.get('return_metadata', False))
    workers = int(data.get('num_workers', 4))
    options = dict(batch_size=int(data.per_device_batch_size), shuffle=True,
                   generator=torch.Generator().manual_seed(int(cfg.get('seed',42))),
                   collate_fn=collate_fn, num_workers=workers, pin_memory=False, drop_last=True)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=int(data.get('prefetch_factor',2)),
                       multiprocessing_context='spawn')
    # Accelerate performs distributed sharding once; do not add another DistributedSampler.
    loader = DataLoader(dataset, **options)
    if not dist.is_initialized() or dist.get_rank()==0:
        dataset.save_dataset_statistics(Path(cfg.output_dir)/'dataset_statistics.json')
    return loader
