"""Committed UMI sample cursors and stage transitions for the existing trainer."""
from contextlib import contextmanager
import random

import numpy as np
import torch

from starVLA.dataloader.umi_wyt import make_wyt_umi_dataloader, plain_config, stage_identity


@contextmanager
def preserve_rng(generator=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generator_state = generator.get_state() if generator is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if generator is not None:
            generator.set_state(generator_state)


class UMIStageData:
    """Advance progress only after a successful training microbatch.

    Model/optimizer/scheduler are prepared once. Only data loaders change. The
    cursor is a count in the sampler's global shuffled stream, not prefetch.
    """
    def __init__(self, cfg, accelerator, raw_loader, prepared_loader):
        self.cfg, self.accelerator = plain_config(cfg), accelerator
        self.stages = self.cfg.datasets.vla_data.stages
        self.identity = stage_identity(self.cfg, accelerator.num_processes)
        self.raw, self.prepared = raw_loader, prepared_loader
        self.stage_index = self.epoch = self.cursor = self.stage_updates = 0
        self.iterator = self.first = None
        self.pending = False
        self.micro_global = int(self.cfg.datasets.vla_data.per_device_batch_size) * accelerator.num_processes
        self.gas = int(self.cfg.trainer.gradient_accumulation_steps)

    def state_dict(self):
        if self.pending or self.cursor % (self.micro_global * self.gas):
            raise RuntimeError('UMI checkpoints require a committed optimizer boundary')
        return dict(version=1, identity=self.identity, stage_index=self.stage_index,
                    epoch=self.epoch, global_sample_cursor=self.cursor,
                    stage_updates=self.stage_updates)

    def load_state_dict(self, state):
        if state.get('version') != 1 or state.get('identity') != self.identity:
            raise ValueError('UMI data/checkpoint identity changed; use a new run with weights-only initialization')
        index, epoch, cursor, updates = (int(state[k]) for k in
            ('stage_index', 'epoch', 'global_sample_cursor', 'stage_updates'))
        if not 0 <= index < len(self.stages) or epoch != 0:
            raise ValueError('Invalid one-pass UMI stage/epoch')
        if not 0 <= updates <= int(self.stages[index].updates):
            raise ValueError('Invalid UMI stage update count')
        if cursor != updates * self.micro_global * self.gas:
            raise ValueError('UMI cursor must match committed optimizer updates')
        if index != self.stage_index:
            self._activate(index)
        self.epoch, self.cursor, self.stage_updates = epoch, cursor, updates

    def _stop_iterator(self):
        self.first = None
        if self.iterator is not None and hasattr(self.iterator, 'close'):
            self.iterator.close()
        self.iterator = None
        self.prepared.end()
        base = self.prepared.base_dataloader
        iterator = getattr(base, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
            base._iterator = None

    def _close_loader(self):
        self._stop_iterator()
        self.prepared._is_accelerate_prepared = False
        self.accelerator._dataloaders = [d for d in self.accelerator._dataloaders if d is not self.prepared]
        self.raw.dataset.close()

    def _activate(self, index):
        self._close_loader()
        self.stage_index = index
        local = plain_config(self.cfg)
        stage = self.stages[index]
        local.datasets.vla_data.index_dir = stage.index_dir
        local.datasets.vla_data.active_stage = stage.name
        self.raw = make_wyt_umi_dataloader(local)
        if len(self.raw.dataset) != int(stage.num_windows):
            raise ValueError('Active stage length differs from the resolved plan')
        self.prepared = self.accelerator.prepare_data_loader(self.raw, device_placement=False)
        self.epoch = self.cursor = self.stage_updates = 0
        self.accelerator.print(f'[umi-stage] entering {stage.name}, updates={stage.updates}')

    def start(self):
        self._stop_iterator()
        if self.stage_updates == int(self.stages[self.stage_index].updates):
            if self.stage_index + 1 >= len(self.stages):
                raise StopIteration('UMI one-pass stage plan completed')
            self._activate(self.stage_index + 1)
        self.raw.sampler.set_epoch(self.epoch, start_index=self.cursor)
        self.prepared.set_epoch(self.epoch)
        # Preserve Accelerate's epoch too: DataLoaderShard.__iter__ calls
        # set_epoch(self.iteration), which must not reset a restored cursor.
        self.prepared.iteration = self.epoch
        with preserve_rng(self.raw.generator):
            self.raw.generator.manual_seed(int(self.cfg.seed) + 1000003 * self.stage_index + self.epoch)
            self.iterator = iter(self.prepared)
            self.first = next(self.iterator)

    def next(self):
        if self.pending:
            raise RuntimeError('Previous UMI batch was not committed')
        if self.stage_updates == int(self.stages[self.stage_index].updates):
            self.start()
        elif self.iterator is None:
            self.start()
        if self.first is not None:
            batch, self.first = self.first, None
        else:
            batch = next(self.iterator)
        if len(batch) != int(self.cfg.datasets.vla_data.per_device_batch_size):
            raise RuntimeError('UMI stage requires complete per-rank microbatches')
        self.pending = True
        return batch

    def commit(self, sync_gradients):
        if not self.pending:
            raise RuntimeError('No pending UMI batch')
        self.cursor += self.micro_global
        self.pending = False
        if sync_gradients:
            if self.cursor % (self.micro_global * self.gas):
                raise RuntimeError('Unexpected gradient accumulation boundary in UMI stage')
            self.stage_updates += 1

    def evaluate(self, model, metrics):
        """Use a separate fixed validation batch without consuming train data."""
        local = plain_config(self.cfg)
        data = local.datasets.vla_data
        data.index_dir, data.active_stage = data.validation_index_dir, 'validation_monitor'
        data.anchor_stride_frames, data.shuffle = 1, False
        data.num_workers = 0
        data.per_device_batch_size = int(data.get('evaluation_batch_size', 16))
        loader = None
        training = model.training
        with preserve_rng():
            try:
                torch.manual_seed(9876)
                model.eval()
                loader = make_wyt_umi_dataloader(local)
                batch = next(iter(loader))
                with torch.no_grad():
                    inputs = [{k: v for k, v in sample.items() if k != 'action'} for sample in batch]
                    pred = np.asarray(model.predict_action(examples=inputs)['normalized_actions'])
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite UMI validation prediction')
                truth = np.stack([s['action'] for s in batch])
                metrics['validation/normalized_action_mse'] = float(np.mean((pred - truth) ** 2))
                error = loader.dataset.normalizer.inverse_action(pred) - loader.dataset.normalizer.inverse_action(truth)
                metrics['validation/position_rmse'] = float(np.sqrt(np.mean(error[..., [0, 1, 2, 8, 9, 10]] ** 2)))
                return metrics
            finally:
                model.train(training)
                if loader is not None:
                    loader.dataset.close()

    def close(self):
        self._close_loader()
