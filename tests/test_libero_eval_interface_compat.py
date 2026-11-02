"""Check the two explicit LIBERO interfaces and reject retired width semantics."""
import unittest
import numpy as np
from examples.simBenchmarks.LIBERO.eval_files import model2libero_interface as interface
from examples.simBenchmarks.LIBERO.eval_files.eval_libero import Args, _resolve_gripper_command
from examples.simBenchmarks.LIBERO.eval_files.parallel_eval import parser, worker_command


class FakePolicy:
    metadata = {'action_chunk_size': 1}
    action = np.array([.1, .2, .3, .4, .5, .6, 1.], dtype=np.float32)
    def __init__(self, *args, **kwargs):
        pass
    def get_server_metadata(self):
        return self.metadata
    def predict_action(self, payload):
        return {'data': {'actions': self.action[None, None, :]}}


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.original = interface.WebsocketClientPolicy
        interface.WebsocketClientPolicy = FakePolicy
        FakePolicy.metadata = {'action_chunk_size': 1}
        FakePolicy.action = np.array([.1, .2, .3, .4, .5, .6, 1.], dtype=np.float32)

    def tearDown(self):
        interface.WebsocketClientPolicy = self.original

    def test_native_commands_and_binary_gripper(self):
        client = interface.ModelClient(action_interface='native_osc7', action_ensemble=False)
        for prediction, expected in [(0., 1.), (.5, 1.), (1., -1.)]:
            FakePolicy.action[6] = prediction
            result = client.step({'lang': 'native', 'image': []})['raw_action']
            np.testing.assert_array_equal(result['world_vector'], FakePolicy.action[:3])
            np.testing.assert_array_equal(result['rotation_delta'], FakePolicy.action[3:6])
            self.assertEqual(_resolve_gripper_command(result)[0], expected)
        for invalid in [np.zeros(16), np.full(7, np.nan)]:
            FakePolicy.action = invalid
            with self.assertRaises(ValueError):
                client.step({'lang': 'native', 'image': []})

    def test_explicit_interface_and_metadata(self):
        self.assertEqual(Args().action_interface, 'anchored_goal_binary')
        with self.assertRaises(ValueError):
            interface.ModelClient(action_interface='legacy_umi16')
        FakePolicy.metadata['action_representation'] = 'libero_umi16'
        for selected in ['native_osc7', 'anchored_goal_binary']:
            with self.assertRaises(ValueError):
                interface.ModelClient(action_interface=selected)

    def test_only_binary_native_gripper_commands(self):
        for value in [-1., 1.]:
            self.assertEqual(_resolve_gripper_command({'gripper_command': [value]})[0], value)
        for value in [0., .4, np.nan]:
            with self.assertRaises(ValueError):
                _resolve_gripper_command({'gripper_command': [value]})
        for value in [[], [0., 1.], [np.nan]]:
            with self.assertRaises(ValueError):
                _resolve_gripper_command({'open_gripper': value})

    def test_parallel_worker_uses_only_remaining_interface(self):
        args = parser().parse_args(['--ckpt', '/tmp/model.pt', '--output-dir', '/tmp/eval',
                                  '--action-interface', 'native_osc7'])
        command = worker_command(args, 'libero_object', 0, 2, 11093, 'test')
        self.assertIn('native_osc7', command)
        self.assertFalse(any('gripper-' in item for item in command))
        self.assertFalse(hasattr(args, 'gripper_mode'))


if __name__ == '__main__':
    unittest.main()
