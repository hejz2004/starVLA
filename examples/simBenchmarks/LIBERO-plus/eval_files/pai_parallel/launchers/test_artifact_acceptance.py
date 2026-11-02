"""Atomic variant artifacts avoid collisions even when instructions are identical."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from eval_worker import execute_job,single_thread_video
from plus_protocol import atomic_json,suite_directory,read_result,REPRESENTATION,VIDEO_LAYOUT
from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import episode_key

class CommitTests(unittest.TestCase):
    def test_identical_video_names_from_distinct_variants_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_dir=tmp,port=1,seed=7,ckpt='fixture',protocol_digest='test',worker_id='d0_w0',no_video=False)
            def fake_eval(args):
                folder=suite_directory(args.results_dir,args.task_suite_name)
                (folder/'videos').mkdir(parents=True);(folder/'traces').mkdir()
                (folder/'videos/same_instruction.mp4').write_bytes(b'video')
                actions=np.zeros((2,7));actions[:,6]=[-1,1]
                np.savez_compressed(folder/'traces/x.npz',actions=actions)
                key=episode_key(args.task_suite_name,args.task_ids[0],0)
                row=dict(key=key,suite=args.task_suite_name,task_id=args.task_ids[0],trial=0,
                    protocol_digest='test',success=False,steps=2,trace='traces/x.npz',video='videos/same_instruction.mp4',
                    video_layout=VIDEO_LAYOUT,video_frames=2,action_representation=REPRESENTATION)
                atomic_json(folder/'results'/(key+'.json'),row)
            module=SimpleNamespace(Args=lambda **kw:SimpleNamespace(**kw),eval_libero=fake_eval)
            for i in range(2):
                job=dict(suite='libero_goal',task_id=i,trial=0,key=episode_key('libero_goal',i,0),variant_id=i+1,
                    variant='same instruction',category='Camera Viewpoints',attempt=1)
                execute_job(job,a,module)
                self.assertIsNotNone(read_result(tmp,job,'test'))
            self.assertEqual(len(list((Path(tmp)/'Goal/videos').rglob('*.mp4'))),2)
            self.assertEqual(len(list((Path(tmp)/'.staging').iterdir())),0)
    def test_episode_exception_cannot_commit_a_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output_dir=tmp,port=1,seed=7,ckpt='fixture',protocol_digest='test',worker_id='w',no_video=False)
            def error(args):raise RuntimeError('simulator failed')
            module=SimpleNamespace(Args=lambda **kw:SimpleNamespace(**kw),eval_libero=error)
            with self.assertRaises(RuntimeError):execute_job(dict(key='x',suite='libero_goal',task_id=0),a,module)
            self.assertFalse(list(Path(tmp).glob('*/results/*.json')))
    def test_real_ffmpeg_dual_camera_video_decodes(self):
        import imageio.v2 as imageio
        with tempfile.TemporaryDirectory() as tmp:
            frames=[np.concatenate([np.full((256,256,3),20+i,dtype=np.uint8),np.full((256,256,3),180+i,dtype=np.uint8)],axis=1) for i in range(3)]
            job=dict(suite='libero_10',task_id=1268,key='libero_10_task1268_trial000',category='Language Instructions')
            instruction='Place the white mug on the left plate and the yellow and white mug on the right plate. ' * 5
            path=single_thread_video(Path(tmp)/('rollout_'+instruction.replace(' ','_')+'_episode0_success.mp4'),frames,job=job,instruction=instruction)
            self.assertEqual(path.name,'language_instructions_suc.mp4')
            self.assertEqual(path.parent.name,job['key'])
            self.assertFalse(list(Path(tmp).rglob('*.tmp.mp4')))
            reader=imageio.get_reader(path)
            try:
                self.assertEqual(reader.count_frames(),3)
                first=reader.get_data(0)
                self.assertEqual(first.shape[1:],(512,3))
                self.assertGreater(first.shape[0],256)
                self.assertEqual(first.shape[0]%16,0)
                self.assertAlmostEqual(float(first[-256:,:256].mean()),20,delta=5)
                self.assertAlmostEqual(float(first[-256:,256:].mean()),180,delta=5)
                self.assertEqual(reader.get_meta_data()['fps'],10)
            finally:reader.close()
if __name__=='__main__':unittest.main()
