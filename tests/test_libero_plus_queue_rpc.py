"""Regression coverage for lost queue acknowledgments and slow shared storage."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

LAUNCHERS=Path(__file__).resolve().parents[1]/'examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/launchers'
sys.path.insert(0,str(LAUNCHERS))
import evaluation_controller as controller
import plus_protocol as protocol
from queue_rpc import QueueRPC,ResultCoordinator,make_queue_server


def jobs(count=2):
    return [dict(key=f'libero_goal_task{i:02d}_trial000',suite='libero_goal',
                 task_id=i,trial=0,category='test') for i in range(count)]


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.queue=protocol.WorkQueue(jobs(),max_attempts=3)
        self.rows={}
        self.coordinator=ResultCoordinator(self.queue,self.rows,
            lambda job:dict(job,success=True,steps=1))

    def test_slow_artifact_validation_does_not_block_claims(self):
        first=self.queue.claim('w0');entered=threading.Event();release=threading.Event()
        def slow(job):
            entered.set();self.assertTrue(release.wait(3));return dict(job,success=True,steps=1)
        self.coordinator.read_result=slow
        with ThreadPoolExecutor(max_workers=2) as pool:
            completing=pool.submit(self.coordinator.complete,'w0',first['key'])
            self.assertTrue(entered.wait(2))
            claiming=pool.submit(self.queue.claim,'w1')
            try:self.assertNotEqual(claiming.result(timeout=1)['key'],first['key'])
            finally:release.set()
            self.assertEqual(completing.result(timeout=2),{'ok':True})

    def test_slow_summary_writes_do_not_block_claims(self):
        entered=threading.Event();release=threading.Event()
        def slow(*args,**kwargs):
            entered.set();self.assertTrue(release.wait(3))
        with tempfile.TemporaryDirectory() as tmp,ThreadPoolExecutor(max_workers=2) as pool:
            with patch.object(protocol,'atomic_json',slow),patch.object(controller,'atomic_json',slow):
                writing=pool.submit(controller.write_progress,Path(tmp),jobs(),self.coordinator,time.monotonic())
                self.assertTrue(entered.wait(2))
                claiming=pool.submit(self.queue.claim,'w0')
                try:self.assertIn('key',claiming.result(timeout=1))
                finally:release.set()
                self.assertEqual(writing.result(timeout=2)['completed'],0)

    def test_concurrent_duplicate_completions_count_once(self):
        first=self.queue.claim('w0')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _:self.coordinator.complete('w0',first['key']),range(32)))
        self.assertEqual(len(self.rows),1)
        self.assertEqual(self.queue.done,{first['key']})

    def test_published_result_is_recovered_after_worker_exit(self):
        first=self.queue.claim('w0')
        result=self.coordinator.recover_worker('w0','worker exited')
        self.assertTrue(result['recovered']);self.assertFalse(result['requeued'])
        self.assertIn(first['key'],self.queue.done);self.assertEqual(len(self.rows),1)

    def test_missing_result_infrastructure_retries_are_bounded(self):
        queue=protocol.WorkQueue(jobs(1),max_attempts=3)
        coordinator=ResultCoordinator(queue,{},lambda job:None)
        for attempt in range(1,4):
            job=queue.claim('w0');self.assertEqual(job['attempt'],attempt)
            recovered=coordinator.recover_worker('w0','simulator exited')
            self.assertEqual(recovered['requeued'],attempt<3)
        self.assertEqual(len(queue.failed),1);self.assertFalse(queue.done)

    def test_episode_python_error_is_not_retried(self):
        first=self.queue.claim('w0')
        self.coordinator.dispatch('/error',dict(worker='w0',key=first['key'],error='invalid action'))
        self.assertIn(first['key'],self.queue.failed)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.queue=protocol.WorkQueue(jobs(),max_attempts=3);self.rows={}
        self.coordinator=ResultCoordinator(self.queue,self.rows,lambda job:dict(job,success=True,steps=1))
        self.server=make_queue_server(('127.0.0.1',0),self.coordinator,'test-token',{'w0','w1'})
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        self.thread.start();self.url='http://127.0.0.1:'+str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)

    def lose_ack_once(self,path):
        base=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        class Opener:
            calls=0
            def open(instance,request,timeout):
                instance.calls+=1
                response=base.open(request,timeout=timeout)
                if request.full_url.endswith(path) and instance.calls==1:
                    response.read();response.close()
                    raise TimeoutError('simulated lost response after server commit')
                return response
        return Opener()

    def test_lost_claim_ack_does_not_skip_an_episode(self):
        opener=self.lose_ack_once('/claim')
        rpc=QueueRPC(self.url,'w0','test-token',opener=opener,backoff=0)
        first=rpc('/claim',{})
        self.assertEqual(opener.calls,2)
        self.assertEqual(self.queue.attempts[first['key']],1)
        self.assertEqual(self.queue.active,{'w0':first['key']})
        self.assertEqual(self.queue.snapshot()['pending'],1)

    def test_lost_complete_ack_is_retried_without_double_counting(self):
        first=self.queue.claim('w0');opener=self.lose_ack_once('/complete')
        rpc=QueueRPC(self.url,'w0','test-token',opener=opener,backoff=0)
        self.assertEqual(rpc('/complete',dict(key=first['key'])),{'ok':True})
        self.assertEqual(opener.calls,2);self.assertEqual(len(self.rows),1)
        self.assertEqual(self.queue.done,{first['key']})

    def test_protocol_error_is_not_retried(self):
        opener=self.lose_ack_once('/complete')
        rpc=QueueRPC(self.url,'unknown-worker','test-token',opener=opener,backoff=0)
        with self.assertRaises(urllib.error.HTTPError):rpc('/claim',{})
        self.assertEqual(opener.calls,1)

    def test_network_retries_are_bounded(self):
        class Opener:
            calls=0
            def open(instance,*args,**kwargs):
                instance.calls+=1;raise TimeoutError('simulated controller unavailable')
        opener=Opener();rpc=QueueRPC(self.url,'w0','test-token',opener=opener,max_attempts=4,backoff=0)
        with self.assertRaises(TimeoutError):rpc('/claim',{})
        self.assertEqual(opener.calls,4)


if __name__=='__main__':unittest.main()
