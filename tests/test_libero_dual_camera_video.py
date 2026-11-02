"""Exercise stitched camera video, suite directories, and resume artifacts."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import json
import tempfile
import unittest

import av
import numpy as np
from examples.simBenchmarks.LIBERO.eval_files import eval_libero as evaluator
from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import (
    VIDEO_LAYOUT, aggregate, atomic_json, select_jobs, suite_directory, valid_result, write_summary,
)


class FakeEnv:
    def __init__(self):
        self.sim = SimpleNamespace(get_state=lambda: np.zeros(4))
        self.frames = 0

    def observation(self):
        return dict(agentview_image=np.full((32, 32, 3), [200, 20 + self.frames * 20, 10], dtype=np.uint8),
                    robot0_eye_in_hand_image=np.full((32, 32, 3), [10, 30 + self.frames * 20, 200], dtype=np.uint8),
                    robot0_gripper_qpos=np.array([.04, -.04]),
                    robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.array([0, 0, 0, 1]))

    def reset(self):
        self.frames = 0

    def set_init_state(self, state):
        return self.observation()

    def step(self, action):
        self.frames += 1
        return self.observation(), 0., self.frames >= 3, {}

    def close(self):
        pass


class FakePolicy:
    def __init__(self, **kwargs):
        self._server_metadata = {}
        self.action_chunk_size = 1
        self.action_representation = 'controller_delta'
        self.raw_actions = np.array([[0., 0., 0., 0., 0., 0., 1.]])
        self.client = SimpleNamespace(close=lambda: None)

    def reset(self, **kwargs):
        pass

    def step(self, **kwargs):
        return {'raw_action': dict(world_vector=np.zeros(3), rotation_delta=np.zeros(3), open_gripper=[1.])}


def run_fake(args):
    suite = SimpleNamespace(n_tasks=1, get_task=lambda i: object(),
                            get_task_init_states=lambda i: [np.zeros(4)])
    with mock.patch.object(evaluator.benchmark, 'get_benchmark_dict', return_value={args.task_suite_name: lambda: suite}), \
         mock.patch.object(evaluator, '_get_libero_env', return_value=(FakeEnv(), 'test paired cameras')), \
         mock.patch.object(evaluator, 'ModelClient', FakePolicy):
        evaluator.eval_libero(args)


class DualCameraVideoTests(unittest.TestCase):
    def make_args(self, output, suite='libero_goal', save_video=True):
        return evaluator.Args(task_suite_name=suite, num_trials_per_task=1, num_steps_wait=0,
                              action_interface='native_osc7', results_dir=str(output), save_video=save_video,
                              video_out_path=str(output), protocol_digest='stitched')

    def result(self, output, suite='libero_goal'):
        job = select_jobs(suite, 1, 1)[0]
        path = suite_directory(output, suite) / 'results' / (job['key'] + '.json')
        return job, path, json.loads(path.read_text())

    def test_encoded_video_has_main_left_wrist_right_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); args = self.make_args(output)
            run_fake(args)
            job, path, row = self.result(output)
            self.assertEqual((row['steps'], row['video_frames'], row['video_fps']), (3, 3, 10))
            self.assertEqual(row['video_layout'], VIDEO_LAYOUT)
            self.assertNotIn('wrist_video', row)
            video = suite_directory(output, 'libero_goal') / row['video']
            self.assertEqual(video.parent, output / 'Goal/videos')
            self.assertEqual(len(list(output.rglob('*.mp4'))), 1)
            with av.open(str(video)) as container:
                self.assertEqual(float(container.streams.video[0].average_rate), 10)
                frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
            self.assertEqual(len(frames), 3)
            for i, frame in enumerate(frames):
                self.assertEqual(frame.shape, (32, 64, 3))
                for half, expected in [(frame[:, :32], [200, 20 + i * 20, 10]),
                                       (frame[:, 32:], [10, 30 + i * 20, 200])]:
                    np.testing.assert_allclose(half.mean(axis=(0, 1)), expected, atol=6)
            self.assertTrue(valid_result(path, job, 'stitched', required_video_layout=VIDEO_LAYOUT))
            args.resume = True
            with mock.patch.object(evaluator, 'ModelClient', side_effect=AssertionError('must skip')):
                suite = SimpleNamespace(n_tasks=1)
                with mock.patch.object(evaluator.benchmark, 'get_benchmark_dict', return_value={'libero_goal': lambda: suite}):
                    evaluator.eval_libero(args)
            video.unlink()
            self.assertFalse(valid_result(path, job, 'stitched', required_video_layout=VIDEO_LAYOUT))
            self.assertEqual(aggregate(output, [job], 'stitched', required_video_layout=VIDEO_LAYOUT)['completed'], 0)
            run_fake(args)
            self.assertTrue(valid_result(path, job, 'stitched', required_video_layout=VIDEO_LAYOUT))

    def test_goal_and_long_are_separate_and_have_independent_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); jobs = []
            for suite in ['libero_goal', 'libero_10']:
                run_fake(self.make_args(output, suite))
                job, path, row = self.result(output, suite); jobs.append(job)
                self.assertTrue((path.parent.parent / row['video']).is_file())
                self.assertTrue((path.parent.parent / row['trace']).is_file())
            state = aggregate(output, jobs, 'stitched', required_video_layout=VIDEO_LAYOUT)
            self.assertEqual((state['expected'], state['completed']), (2, 2))
            write_summary(output, state)
            for suite in ['libero_goal', 'libero_10']:
                summary = json.loads((suite_directory(output, suite) / 'summary.json').read_text())
                self.assertEqual((summary['suite'], summary['expected'], summary['completed']), (suite, 1, 1))
                self.assertEqual(summary['status'], 'complete')
            self.assertFalse((output / 'results').exists())
            self.assertFalse((output / 'videos').exists())
            # Detect duplicates even when someone accidentally copies a result across suite folders.
            job, path, row = self.result(output)
            duplicate = output / 'Long/results' / path.name
            atomic_json(duplicate, row)
            with self.assertRaisesRegex(ValueError, 'Duplicate result'):
                aggregate(output, jobs, 'stitched')

    def test_no_video_disables_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_fake(self.make_args(output, save_video=False))
            job, path, row = self.result(output)
            self.assertIsNone(row['video']); self.assertIsNone(row['video_layout'])
            self.assertEqual(row['video_frames'], 0)
            self.assertEqual(list(output.rglob('*.mp4')), [])
            self.assertTrue(valid_result(path, job, 'stitched', require_video=False, required_video_layout=VIDEO_LAYOUT))

    def test_stitched_validation_rejects_missing_empty_or_wrong_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / 'results').mkdir()
            job = select_jobs('libero_goal', 1, 1)[0]
            path = output / 'results' / (job['key'] + '.json')
            row = dict(job, protocol_digest='stitched', success=True, steps=1, video='main.mp4', trace='trace.npz')
            (output / 'main.mp4').write_bytes(b'video'); (output / 'trace.npz').write_bytes(b'trace')
            atomic_json(path, row)
            self.assertTrue(valid_result(path, job, 'stitched'))  # historical flat record
            self.assertFalse(valid_result(path, job, 'stitched', required_video_layout=VIDEO_LAYOUT))
            row['video_layout'] = VIDEO_LAYOUT
            for value in [None, '', 'missing.mp4', 'empty.mp4']:
                (output / 'empty.mp4').touch()
                row['video'] = value; atomic_json(path, row)
                self.assertFalse(valid_result(path, job, 'stitched', required_video_layout=VIDEO_LAYOUT))


if __name__ == '__main__':
    unittest.main()
