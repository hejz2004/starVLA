"""BF16 inference results must cross the NumPy policy boundary as float32."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI


class QwenPIInferenceDtypeTests(unittest.TestCase):
    def test_standard_and_realtime_export_bfloat16_actions(self):
        examples = [dict(image=[np.zeros((8, 8, 3), dtype=np.uint8)] * 2, lang='test')]
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                values = torch.linspace(-1, 1, 256, dtype=dtype).reshape(1, 16, 16)
                head = SimpleNamespace(predict_action=Mock(return_value=values),
                                       predict_action_realtime=Mock(return_value=values))
                model = SimpleNamespace(
                    config=SimpleNamespace(datasets=SimpleNamespace(vla_data=SimpleNamespace())),
                    action_model=head,
                    _encode_vl_hidden_states=lambda *args: ([torch.zeros(1, 2, 4, dtype=dtype)], None),
                )
                with patch('torch.autocast', side_effect=lambda *args, **kwargs: nullcontext()):
                    standard = Qwen_PI.predict_action(model, examples)
                    realtime = Qwen_PI.predict_action_realtime(model, examples,
                        prev_action_chunk_normalized=np.zeros((1, 16, 16), dtype=np.float32))
                for result in (standard, realtime):
                    self.assertEqual(result['normalized_actions'].dtype, np.float32)
                    np.testing.assert_array_equal(result['normalized_actions'], values.float().numpy())


if __name__ == '__main__':
    unittest.main()
