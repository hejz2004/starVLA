"""Current training construction and strict loading for the restored architecture."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.DualStreamQwenPI import DualStreamQwen_PI
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI


class DualStreamFrameworkTests(unittest.TestCase):
    def interface(self, config):
        self.assertEqual(config.framework.qwenvl.dual_stream.num_shared_tokens, 32)
        interface = torch.nn.Module()
        interface.model = torch.nn.Module()
        interface.model.config = SimpleNamespace(hidden_size=64, text_config=SimpleNamespace(num_hidden_layers=2))
        interface.model.model = torch.nn.Module()
        interface.model.model.visual = torch.nn.Module()
        interface.model.model.visual.shared_init = torch.nn.Parameter(torch.randn(32, 64))
        return interface

    def build(self):
        cfg = OmegaConf.create({'framework': {'name': 'DualStreamPI', 'action_model': {
            'action_dim': 16, 'state_dim': 16, 'use_state': False, 'action_horizon': 16}},
            'experiment': {'action_head_seed': 43}})
        with patch('starVLA.model.framework.VLM4A.DualStreamQwenPI._QWen3DualStream_VL_Interface',
                   side_effect=self.interface):
            return DualStreamQwen_PI(cfg)

    def test_construction_preserves_action_seed_and_current_training_methods(self):
        torch.manual_seed(1)
        first = self.build()
        torch.manual_seed(2)
        second = self.build()
        for key, value in first.action_model.state_dict().items():
            torch.testing.assert_close(value, second.action_model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(first.action_horizon, 16)
        self.assertTrue(all(p.requires_grad for p in first.parameters()))
        self.assertIs(DualStreamQwen_PI.forward, Qwen_PI.forward)
        self.assertIs(DualStreamQwen_PI.predict_action, Qwen_PI.predict_action)

    def test_shared_state_strict_roundtrip_and_reject_register_checkpoint(self):
        first, second = self.build(), self.build()
        second.load_state_dict(first.state_dict(), strict=True)
        torch.testing.assert_close(first.qwen_vl_interface.model.model.visual.shared_init,
                                   second.qwen_vl_interface.model.model.visual.shared_init)
        with self.assertRaisesRegex(ValueError, 'MV2 register'):
            second.load_state_dict({'qwen_vl_interface.model.model.visual.register_tokens.weight':
                                    torch.zeros(4, 64)}, strict=False)

    def test_cannot_mix_dualstream_with_late_global_attention(self):
        cfg = OmegaConf.create({'framework': {'name': 'DualStreamPI', 'qwenvl': {
            'cross_view_attention': {'enabled': True}}}})
        with self.assertRaisesRegex(ValueError, 'cannot both be enabled'):
            DualStreamQwen_PI(cfg)
