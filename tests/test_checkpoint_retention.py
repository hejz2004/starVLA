"""Checkpoint retention keeps flat weights and resumable state at the same step."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training.train_starvla import VLATrainer


@pytest.mark.parametrize('save_format', ['pt', 'safetensors'])
def test_latest_checkpoint_state_and_final_non_interval_step(tmp_path, save_format):
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = OmegaConf.create({'output_dir': str(tmp_path), 'trainer': {
        'save_full_state': True, 'save_final_full_state': True,
        'keep_last_checkpoints': 1, 'save_format': save_format}})
    trainer.checkpoint_dir = str(tmp_path / 'checkpoints')
    Path(trainer.checkpoint_dir).mkdir()
    trainer.model = torch.nn.Linear(2, 2)
    trainer.umi_stage_data = None
    trainer._swanlab_enabled = False
    saved = []

    def save_state(path):
        Path(path).mkdir()
        torch.save({'step': trainer.completed_steps}, Path(path) / 'training_state.pt')
        saved.append(trainer.completed_steps)

    trainer.accelerator = SimpleNamespace(is_main_process=True, save_state=save_state,
        wait_for_everyone=lambda: None, print=lambda *args: None)
    suffix = '_pytorch_model.pt' if save_format == 'pt' else '_model.safetensors'
    with patch('starVLA.training.train_starvla._get_state_dict',
               side_effect=lambda accelerator, model: model.state_dict()):
        for step in (2000, 4000, 14000):
            trainer.completed_steps = step
            trainer._save_checkpoint()
            assert sorted(p.name for p in Path(trainer.checkpoint_dir).iterdir()) == [
                f'steps_{step}', f'steps_{step}{suffix}']
        trainer.completed_steps = 15000
        trainer._finalize_training()
        # Re-finalization must not attempt to save the same full state again.
        trainer._finalize_training()
    assert saved == [2000, 4000, 14000, 15000]
    assert sorted(p.name for p in Path(trainer.checkpoint_dir).iterdir()) == [
        'steps_15000', f'steps_15000{suffix}']
    state = torch.load(Path(trainer.checkpoint_dir)/'steps_15000/training_state.pt', weights_only=True)
    assert state['step'] == 15000
    flat = Path(trainer.checkpoint_dir) / f'steps_15000{suffix}'
    export = tmp_path/'final_model'/('pytorch_model.pt' if save_format == 'pt' else 'model.safetensors')
    assert flat.stat().st_ino == export.stat().st_ino
    assert [json.loads(line)['steps'] for line in (tmp_path/'summary.jsonl').read_text().splitlines()] == saved
