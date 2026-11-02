"""Regression checks for NAS summary recovery and deferred evaluation submission."""
import errno
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

monitor = load('train_eval_monitor', 'scripts/monitor_train_then_eval.py')
eval_node = load('a100_eval_node', 'examples/simBenchmarks/LIBERO/eval_files/pai_a100/run_node.py')


class MonitorTests(unittest.TestCase):
    def test_transient_stale_summary_recovers_next_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank, suites in eval_node.SUITES.items():
                (root / f'node{rank}').mkdir()
                eval_node.atomic(root / f'node{rank}' / 'summary.json', {
                    'status': 'complete', 'completed': 1000, 'successes': 400,
                    'suites': [{'suite': s} for s in suites]})
                eval_node.atomic(root / f'node{rank}.status.json', {'exit_code': 0})
            with patch.object(Path, 'read_text', side_effect=OSError(errno.ESTALE, 'stale')):
                self.assertEqual(eval_node.merge(root, 'libero')['status'], 'running')
            state = eval_node.merge(root, 'libero')
            self.assertEqual((state['status'], state['completed'], state['successes']), ('complete', 2000, 800))
            with patch.object(Path, 'read_text', side_effect=PermissionError(errno.EACCES, 'denied')):
                with self.assertRaises(PermissionError):
                    eval_node.merge(root, 'libero')

    def test_trigger_only_after_success_and_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'training': {'output': tmp, 'job_id': 'train', 'max_train_steps': 7500},
                        'checkpoint': str(root / 'checkpoint.pt'), 'final_model': str(root / 'final.pt'),
                        'evaluations': [{'benchmark': b, 'gpus': 16, 'output': tmp} for b in ('libero', 'libero_plus')]}
            with patch.object(monitor.dlc, 'call') as call, patch.object(monitor, 'ensure_submission', return_value='eval') as submit:
                for status, phase in [('Running', 'monitoring_training'), ('Failed', 'training_failed_no_evaluation'),
                                      ('Succeeded', 'waiting_for_final_checkpoint')]:
                    call.return_value = {'Status': status}
                    self.assertEqual(monitor.tick(manifest, None)['phase'], phase)
                    submit.assert_not_called()
                for name in ('checkpoint.pt', 'final.pt', 'config.yaml', 'config.full.yaml', 'dataset_statistics.json'):
                    (root / name).write_text('saved')
                state = monitor.tick(manifest, None)
                self.assertEqual(state['phase'], 'evaluations_submitted')
                self.assertEqual([c.args[0]['benchmark'] for c in submit.call_args_list], ['libero', 'libero_plus'])

    def test_receipt_prevents_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            entry = {'run_dir': str(run), 'job': 'job.json'}
            def submit(args):
                run.mkdir()
                (run / 'submission.json').write_text(json.dumps({'JobId': 'eval-1'}))
            with patch.object(monitor.dlc, 'main', side_effect=submit) as create:
                self.assertEqual(monitor.ensure_submission(entry, None), 'eval-1')
                self.assertEqual(monitor.ensure_submission(entry, None), 'eval-1')
                create.assert_called_once()

    def test_uncertain_submit_reconciles_without_new_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'identity.json').write_text(json.dumps({'DisplayName': 'unique-eval'}))
            entry = {'run_dir': tmp, 'job': 'job.json'}
            with patch.object(monitor.dlc, 'main') as create, patch.object(monitor.dlc, 'matching_jobs') as matches:
                matches.return_value = []
                with self.assertRaises(RuntimeError):
                    monitor.ensure_submission(entry, None)
                matches.return_value = [{'JobId': 'already-created'}]
                self.assertEqual(monitor.ensure_submission(entry, None), 'already-created')
                create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
