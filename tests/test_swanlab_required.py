"""Required experiment tracking must fail visibly instead of silently disabling."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import torch

from starVLA.training import train_starvla as training


class RequiredSwanLabTests(unittest.TestCase):
    def trainer(self, required):
        trainer = object.__new__(training.VLATrainer)
        trainer.config = OmegaConf.create(dict(swanlab_required=required, run_id='tracker-test',
                                               output_dir='/tmp/starvla-tracker-unit-test'))
        trainer.accelerator = SimpleNamespace(is_main_process=True, device=torch.device('cpu'),
                                              wait_for_everyone=Mock())
        return trainer

    def test_required_initialization_failure_stops_training(self):
        trainer = self.trainer(True)
        with patch.dict('os.environ', {'SWANLAB_MODE': 'online'}), \
             patch.object(training.wandb, 'init', side_effect=ConnectionError('unavailable')), \
             patch.object(training.dist, 'is_initialized', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Required SwanLab initialization failed'):
                trainer._init_swanlab()
        self.assertFalse(trainer._swanlab_enabled)

    def test_optional_tracking_preserves_existing_behavior(self):
        trainer = self.trainer(False)
        with patch.dict('os.environ', {'SWANLAB_MODE': 'online'}), \
             patch.object(training.wandb, 'init', side_effect=ConnectionError('unavailable')):
            trainer._init_swanlab()
        self.assertFalse(trainer._swanlab_enabled)
        trainer.accelerator.wait_for_everyone.assert_called_once()

    def test_required_tracking_cannot_be_disabled(self):
        trainer = self.trainer(True)
        with patch.dict('os.environ', {'SWANLAB_MODE': 'disabled'}):
            with self.assertRaisesRegex(ValueError, 'requires SwanLab'):
                trainer._init_swanlab()


if __name__ == '__main__':
    unittest.main()
