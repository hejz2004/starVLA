"""Queue, topology, resumption and real artifact identity regressions."""
import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from plus_protocol import (WorkQueue,make_jobs,cpu_plan,read_result,load_results,
    atomic_json,result_path,SUITE_SIZES,REPRESENTATION,VIDEO_LAYOUT)
from evaluation_controller import parser
HERE=Path(__file__).resolve().parent


def jobs(n=20):
    return [dict(suite='libero_goal',task_id=i,trial=0,key=f'job{i}',variant_id=i+1,
                 variant=f'variant{i}',category='Sensor Noise') for i in range(n)]


class QueueTests(unittest.TestCase):
    def test_defaults_and_full_manifest(self):
        a=parser().parse_args(['--ckpt','x','--output-dir','y','--libero-home','z','--libero-assets','a'])
        self.assertEqual(a.workers_per_device,20);self.assertEqual(len(a.devices.split(',')),16)
        c=json.loads((HERE/'task_classification.json').read_text())
        selected=make_jobs(c,list(SUITE_SIZES))
        self.assertEqual(len(selected),10030)
        self.assertEqual(len({j['key'] for j in selected}),10030)
        self.assertEqual({j['category'] for j in selected},{x['category'] for v in c.values() for x in v})
        self.assertTrue(all(j['trial']==0 and j['variant_id']==j['task_id']+1 for j in selected))
        smoke=make_jobs(c,list(SUITE_SIZES),max_per_category=1)
        self.assertEqual(len(smoke),28)
    def test_rebuilt_metadata_order_does_not_change_dependency_versions(self):
        from plus_protocol import dependency_versions
        with tempfile.TemporaryDirectory() as tmp:
            sites=[Path(tmp)/name for name in ('first','second')]
            for site,requirements in zip(sites,[['numpy','cloudpickle'],['cloudpickle','numpy']]):
                package=site/'gym-0.25.2.dist-info';package.mkdir(parents=True)
                (package/'METADATA').write_text('Name: gym\nVersion: 0.25.2\n'+''.join('Requires-Dist: '+r+'\n' for r in requirements))
            self.assertEqual(dependency_versions(sites[0]),dependency_versions(sites[1]))
            (sites[1]/'gym-0.25.2.dist-info/METADATA').write_text('Name: gym\nVersion: 0.26.0\n')
            self.assertNotEqual(dependency_versions(sites[0]),dependency_versions(sites[1]))

    def test_changed_protocol_refuses_resume_before_loading_models(self):
        from unittest.mock import patch
        import evaluation_controller as controller
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ckpt=root/'ckpt.pt';ckpt.write_bytes(b'fixture')
            output=root/'output';output.mkdir();atomic_json(output/'protocol.json',{'version':'old'})
            a=parser().parse_args(['--ckpt',str(ckpt),'--output-dir',str(output),
                '--libero-home',tmp,'--libero-assets',tmp,'--devices','0','--resume'])
            with patch.object(controller,'make_protocol',return_value={'version':'new'}),patch.object(controller,'prepare_resources') as prepare:
                with self.assertRaisesRegex(ValueError,'protocol mismatch'):controller.run(a)
                prepare.assert_not_called()
    def test_160_cpu_320_workers_share_cores_without_escaping_allocation(self):
        cpus=list(range(20,180));plan=cpu_plan(cpus,list(map(str,range(16))),20)
        self.assertEqual(len(plan),16)
        from collections import Counter
        self.assertEqual(Counter(c for p in plan for w in p['worker_cpus'] for c in w),Counter({c:2 for c in cpus}))
        self.assertTrue(all(len(p['cpus'])==10 and len(p['policy_cpus'])==2 for p in plan))
        self.assertTrue(all(set(p['policy_cpus'])<=set(p['cpus']) for p in plan))
        with self.assertRaises(ValueError):cpu_plan([1],['0','1'],10)
    def test_concurrent_claims_are_unique_even_for_unsuccessful_task_results(self):
        q=WorkQueue(jobs(1000));seen=[]
        def consume(worker):
            while True:
                row=q.claim(str(worker))
                if row.get('done') or row.get('wait'):return
                with q.lock:seen.append(row['key'])
                q.finish(str(worker),row['key'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(consume,range(160)))
        self.assertEqual(len(seen),1000);self.assertEqual(len(set(seen)),1000)
        self.assertEqual(q.snapshot()['pending'],0)
    def test_one_claim_per_worker_and_bounded_retry(self):
        q=WorkQueue(jobs(1),max_attempts=2)
        a=q.claim('w');self.assertEqual(q.claim('w'),a)
        with self.assertRaises(ValueError):q.finish('wrong',a['key'])
        q.fail('w','environment error');b=q.claim('other')
        self.assertEqual(b['key'],a['key']);self.assertEqual(b['attempt'],2)
        q.fail('other','persistent environment error')
        self.assertIn(a['key'],q.failed);self.assertTrue(q.claim('last')['done'])
    def test_completed_items_never_replayed_and_cross_suite_stealing(self):
        selected=jobs(2);selected[1]=dict(selected[1],suite='libero_10')
        q=WorkQueue(selected,completed=['job0'])
        row=q.claim('w',last_suite='libero_goal')
        self.assertEqual(row['key'],'job1');self.assertEqual(row['suite'],'libero_10')
        q.finish('w',row['key']);q.finish('w',row['key'])
        self.assertTrue(q.claim('w')['done'])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.job=jobs(1)[0];self.path=result_path(self.root,self.job)
        self.base=self.path.parent.parent
        (self.base/'traces').mkdir(parents=True)
        (self.base/'videos').mkdir()
        actions=np.zeros((2,7));actions[:,6]=[-1,1]
        np.savez_compressed(self.base/'traces/x.npz',actions=actions)
        (self.base/'videos/x.mp4').write_bytes(b'nonempty-video-fixture')
        self.row=dict(self.job,protocol_digest='test',success=False,steps=2,
            action_representation=REPRESENTATION,trace='traces/x.npz',video='videos/x.mp4',
            video_frames=2,video_layout=VIDEO_LAYOUT)
        atomic_json(self.path,self.row)
    def tearDown(self):self.tmp.cleanup()
    def test_resume_keeps_completed_task_failures(self):
        rows=load_results(self.root,[self.job],'test',True)
        self.assertFalse(rows[self.job['key']]['success'])
        self.assertTrue(WorkQueue([self.job],completed=rows).claim('w')['done'])
    def test_missing_video_is_pending_not_completed(self):
        (self.base/'videos/x.mp4').unlink()
        self.assertIsNone(read_result(self.root,self.job,'test'))
    def test_wrong_protocol_and_variant_rejected(self):
        with self.assertRaises(ValueError):read_result(self.root,self.job,'changed')
        self.row['variant_id']=999;atomic_json(self.path,self.row)
        with self.assertRaises(ValueError):read_result(self.root,self.job,'test')
    def test_unexpected_episode_rejected(self):
        atomic_json(self.path.parent/'unknown.json',self.row)
        with self.assertRaises(ValueError):load_results(self.root,[self.job],'test',True)

if __name__=='__main__':unittest.main()
