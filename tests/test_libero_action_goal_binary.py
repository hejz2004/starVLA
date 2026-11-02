"""Semantic tests for anchored native goals and native binary gripper execution."""
from pathlib import Path
import json
import unittest
import numpy as np
from starVLA.dataloader.libero_action_contract import (
    VERSION, REPRESENTATION, LOSS_MASK, GoalBinaryNormalizer, reconstruct_goals,
    encode_goals, decode_goals, goal_to_native_delta, rotvec_to_quaternion,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'playground/Datasets/LIBERO_OSC_GOAL_BINARY_V1'


class ActionGoalTests(unittest.TestCase):
    def test_targets_are_native_goals_not_reached_states_or_cumsum(self):
        states = np.zeros((3, 8));states[:, 0] = [0, .005, .011]
        actions = np.zeros((3, 7));actions[:, 0] = .8;actions[:, 6] = [0, 1, 0]
        goals = reconstruct_goals(states, actions)
        np.testing.assert_allclose(goals[:, 0], [.04, .045, .051])
        encoded = encode_goals(goals, states[0, :3], [0, 0, 0, 1])
        np.testing.assert_allclose(encoded[:, 0], [.04, .045, .051])
        self.assertFalse(np.allclose(encoded[:, 0], [.04, .08, .12]))
        changed = states.copy();changed[:, 6:] = np.array([.035, -.035])
        np.testing.assert_array_equal(goals, reconstruct_goals(changed, actions))
        np.testing.assert_array_equal(encoded[:, 7], [0, 1, 0])

    def test_one_anchor_for_whole_chunk_and_rotation_history(self):
        states = np.zeros((3, 8));states[:, :3] = [[1, 2, 3], [1.1, 2.1, 3], [1.2, 2.2, 3]]
        states[:, 5] = [np.pi/2, np.pi/2+.1, np.pi/2+.2]
        commands = np.zeros((3, 7));commands[0, 5] = .2
        goals = reconstruct_goals(states, commands)
        np.testing.assert_array_equal(goals[1:, 3:7], np.repeat(goals[:1, 3:7], 2, axis=0))
        anchor = rotvec_to_quaternion(states[0, 3:6])
        encoded = encode_goals(goals, states[0, :3], anchor)
        np.testing.assert_allclose(encoded[:, :3], [[0, 0, 0], [.1, -.1, 0], [.2, -.2, 0]], atol=1e-12)
        np.testing.assert_allclose(decode_goals(encoded, states[0, :3], anchor), goals, atol=1e-12)

    def test_pose_normalization_leaves_binary_channel_and_padding(self):
        normalizer = GoalBinaryNormalizer(BUNDLE / 'statistics.json')
        values = np.zeros((2, 16));values[:, 6] = 1;values[:, 7] = [0, 1]
        encoded = normalizer.normalize_action(values)
        np.testing.assert_array_equal(encoded[:, 7], [0, 1])
        np.testing.assert_array_equal(encoded[:, 8:], 0)
        restored = normalizer.denormalize_action(encoded)
        np.testing.assert_allclose(restored[:, :7], values[:, :7], atol=1e-7)
        encoded[:, 7] = [-.2, 1.2]
        np.testing.assert_array_equal(normalizer.denormalize_action(encoded)[:, 7], encoded[:, 7])
        with self.assertRaises(ValueError):
            GoalBinaryNormalizer({'contract': 'old-width'})
        for value, command in [(.49, 1), (.5, 1), (.51, -1)]:
            goal = np.array([0, 0, 0, 0, 0, 0, 1, value])
            self.assertEqual(goal_to_native_delta(goal, [0, 0, 0], [0, 0, 0, 1])[-1], command)

    def test_client_uses_fixed_anchor_and_binary_not_width_feedback(self):
        from examples.simBenchmarks.LIBERO.eval_files import model2libero_interface as module
        calls = []
        actions = np.zeros((16, 16));actions[:, 6] = 1;actions[:, 0] = np.arange(1, 17)*.005
        actions[:, 7] = np.arange(16) % 2
        class Policy:
            def __init__(self, *args, **kwargs):pass
            def get_server_metadata(self):return dict(action_chunk_size=16, action_representation=REPRESENTATION, action_contract={'version': VERSION})
            def predict_action(self, payload):calls.append(payload);return {'data': {'actions': actions[None]}}
        original = module.WebsocketClientPolicy;module.WebsocketClientPolicy = Policy
        try:
            client = module.ModelClient(action_ensemble=False)
            common = dict(eef_quat=[0,0,0,1], controller_eef_quat=[0,0,0,1], controller_goal_quat=[0,0,0,1])
            first = client.step({'lang':'goal','image':[]}, 0, eef_pos=[0,0,0], controller_eef_pos=[0,0,0], **common)['raw_action']
            second = client.step({'lang':'goal','image':[]}, 1, eef_pos=[.002,0,0], controller_eef_pos=[.002,0,0], **common)['raw_action']
            self.assertEqual(len(calls), 1)
            self.assertAlmostEqual(float(first['world_vector'][0]), .1, places=6)
            self.assertAlmostEqual(float(second['world_vector'][0]), .16, places=6)
            self.assertEqual(first['gripper_command'][0], 1)
            self.assertEqual(second['gripper_command'][0], -1)
            with self.assertRaises(ValueError):module.ModelClient(action_interface='legacy_umi16')
        finally:module.WebsocketClientPolicy = original

    def test_against_actual_native_osc_goal_generation(self):
        from scipy.spatial.transform import Rotation
        from libero.libero import benchmark
        from examples.simBenchmarks.LIBERO.eval_files.eval_libero import _get_libero_env
        from examples.simBenchmarks.LIBERO.eval_files.action_goal_binary import controller_reference
        suite = benchmark.get_benchmark_dict()['libero_object']()
        env, _ = _get_libero_env(suite.get_task(4), 64, 7)
        try:
            env.reset();env.set_init_state(suite.get_task_init_states(4)[0])
            for _ in range(10):env.step([0.]*6+[-1.])
            for action in [[.2,-.3,.1,.1,.2,-.1,1], [.1,.2,.3,0,0,0,-1], [-.2,0,.1,-.3,.1,.2,1]]:
                reference = controller_reference(env)
                c = env.robots[0].controller
                body = Rotation.from_quat(reference['controller_eef_quat']).as_matrix()
                offset = body.T @ c.ee_ori_mat
                state = np.r_[reference['controller_eef_pos'], Rotation.from_matrix(body).as_rotvec(), 0, 0][None]
                stored = np.array(action, dtype=float)[None];stored[:,6] = (1-stored[:,6])/2
                goal = reconstruct_goals(state, stored, reference['controller_goal_quat'])[0]
                recovered = goal_to_native_delta(goal, reference['controller_eef_pos'], reference['controller_eef_quat'], reference['controller_goal_quat'])
                c.set_goal(np.asarray(action[:6]))
                expected_pos, expected_ori = c.goal_pos.copy(), c.goal_ori.copy()
                c.set_goal(recovered[:6])
                np.testing.assert_allclose(c.goal_pos, expected_pos, atol=1e-7)
                np.testing.assert_allclose(c.goal_ori, expected_ori, atol=1e-6)
                self.assertEqual(recovered[6], action[6])
                np.testing.assert_allclose(c.goal_pos, goal[:3], atol=1e-7)
                np.testing.assert_allclose(c.goal_ori, Rotation.from_quat(goal[3:7]).as_matrix() @ offset, atol=1e-6)
        finally:env.close()


if __name__ == '__main__':
    unittest.main()
