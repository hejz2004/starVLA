"""Scheduling invariance and strict artifact/protocol acceptance for eval resumes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import (
    aggregate, atomic_json, inference_seed, select_jobs, valid_result,
)


class ParallelEvaluationTests(unittest.TestCase):
    def test_partition_preserves_episode_identity_and_seed(self):
        complete = select_jobs('libero_object', 10, 50, [0, 4, 9], [5, 6, 8, 9])
        reference = {j['key']: inference_seed(7, j['suite'], j['task_id'], j['trial'], 2) for j in complete}
        for shards in (1, 2, 4, 7):
            partition = [j for shard in range(shards) for j in select_jobs(
                'libero_object', 10, 50, [0, 4, 9], [5, 6, 8, 9], shard_index=shard, num_shards=shards)]
            self.assertEqual(len(partition), len(complete))
            self.assertEqual({j['key'] for j in partition}, set(reference))
            for job in partition:
                self.assertEqual(inference_seed(7, job['suite'], job['task_id'], job['trial'], 2), reference[job['key']])
        self.assertNotEqual(inference_seed(7, 'libero_object', 4, 5, 0), inference_seed(7, 'libero_object', 4, 5, 1))

    def test_filter_validation(self):
        for options in ({'task_ids': [10]}, {'trial_ids': [50]}, {'task_ids': [4, 4]},
                        {'num_shards': 0}, {'shard_index': 2, 'num_shards': 2}):
            with self.assertRaises(ValueError):
                select_jobs('libero_object', 10, 50, **options)

    def test_resume_requires_committed_matching_artifacts(self):
        jobs = select_jobs('libero_object', 10, 50, [4], [5, 6])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'results').mkdir()
            (output / 'videos').mkdir()
            (output / 'traces').mkdir()
            row = dict(jobs[0], protocol_digest='frozen', success=True, steps=42,
                       video='videos/test.mp4', trace='traces/test.npz')
            path = output / 'results' / (row['key'] + '.json')
            atomic_json(path, row)
            self.assertFalse(valid_result(path, jobs[0], 'frozen'))
            (output / row['video']).write_bytes(b'video')
            (output / row['trace']).write_bytes(b'trace')
            self.assertTrue(valid_result(path, jobs[0], 'frozen'))
            with self.assertRaises(ValueError):
                valid_result(path, jobs[0], 'changed')
            state = aggregate(output, jobs, 'frozen')
            self.assertEqual((state['status'], state['completed'], state['successes']), ('incomplete', 1, 1))
            self.assertEqual(state['missing'], [jobs[1]['key']])
            atomic_json(output / 'results' / 'unexpected.json', row)
            with self.assertRaises(ValueError):
                aggregate(output, jobs, 'frozen')


if __name__ == '__main__':
    unittest.main()
