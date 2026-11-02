"""Persistent Plus worker using the exact ordinary LIBERO action execution loop."""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import urllib.request
import numpy as np
from PIL import Image
from libero.libero import get_libero_path
from libero.libero.envs.bddl_utils import get_problem_info
from plus_protocol import ROOT, atomic_json, suite_directory
from video_artifacts import save_rollout_video as single_thread_video, video_filename
sys.path.insert(0,str(ROOT))
# Public render helpers used by the real Plus noise preflight.
from examples.simBenchmarks.LIBERO.eval_files.eval_libero import (
    ModelClient as GoalModelClient,benchmark,_get_libero_env,LIBERO_ENV_RESOLUTION,LIBERO_DUMMY_ACTION,
)


def resolve_task_instruction(task):
    """Read task text from BDDL, retaining the selected language variant."""
    filename = task.bddl_file
    if "_view_" in filename:
        filename = filename.split("_view_", 1)[0] + ".bddl"
    source = Path(get_libero_path("bddl_files")) / task.problem_folder / filename
    instruction = " ".join(get_problem_info(str(source))["language_instruction"].split())
    return instruction, str(source)


class ModelClient(GoalModelClient):
    # WYT image inputs with the checkpoint's latest action decoder.
    def step(self,example,step=0,**kwargs):
        images=[np.asarray(Image.fromarray(np.asarray(img)).resize(
            (self.image_size[1],self.image_size[0]),Image.Resampling.BICUBIC))
            if np.asarray(img).shape[:2]!=self.image_size else np.asarray(img)
            for img in example['image']]
        return super().step({**example,'image':images},step=step,**kwargs)



def execute_job(job,a,standard):
    """Isolate staging names; commit the global result only after video and trace."""
    output=Path(a.output_dir);job_started=time.monotonic()
    staging=output/'.staging';staging.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=job['key']+'-',dir=staging) as tmp:
        standard.eval_libero(standard.Args(host='127.0.0.1',port=a.port,
            task_suite_name=job['suite'],num_trials_per_task=1,task_ids=[job['task_id']],
            seed=a.seed,num_steps_wait=10,pretrained_path=a.ckpt,unnorm_key='franka',
            action_interface='anchored_goal_binary',deterministic_inference=False,
            results_dir=tmp,protocol_digest=a.protocol_digest,video_out_path=tmp,
            save_video=not a.no_video))
        src=suite_directory(tmp,job['suite'])
        row=json.loads((src/'results'/(job['key']+'.json')).read_text())
        target=suite_directory(output,job['suite'])
        (target/'traces').mkdir(parents=True,exist_ok=True)
        trace=target/'traces'/(job['key']+'.npz')
        (src/row['trace']).replace(trace);row['trace']=str(trace.relative_to(target))
        if not a.no_video:
            video=target/'videos'/job['key']/video_filename(job['category'],row['success'])
            video.parent.mkdir(parents=True,exist_ok=True)
            (src/row['video']).replace(video);row['video']=str(video.relative_to(target))
            row['video_annotation']='suite-perturbation-instruction-header-v1'
        row.update(job,worker_id=a.worker_id,port=a.port,
                   wall_seconds_including_environment_setup=time.monotonic()-job_started)
        atomic_json(target/'results'/(job['key']+'.json'),row)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-url',required=True);p.add_argument('--worker-id',required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--ckpt',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--protocol-digest',required=True)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--no-video',action='store_true')
    a=p.parse_args()
    os.nice(1)
    logging.basicConfig(level=logging.INFO)
    from examples.simBenchmarks.LIBERO.eval_files import eval_libero as standard
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError: pass
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def rpc(path,payload):
        payload=dict(payload,worker=a.worker_id)
        req=urllib.request.Request(a.queue_url+path,data=json.dumps(payload).encode(),
            headers={'Content-Type':'application/json','Authorization':os.environ['PLUS_QUEUE_TOKEN']})
        with opener.open(req,timeout=30) as r:return json.load(r)
    progress=Path(a.output_dir)/'workers'/a.worker_id/'progress.json'
    current={}
    standard._save_rollout_video=lambda path,frames: single_thread_video(
        path,frames,job=current,instruction=current['instruction'])
    OriginalClient=ModelClient
    class ProgressClient(OriginalClient):
        def reset(self,task_description):
            current['instruction']=str(task_description)
            return super().reset(task_description=task_description)
        def step(self,*args,**kwargs):
            result=super().step(*args,**kwargs)
            if kwargs.get('step',0)%16==0:
                atomic_json(progress,dict(key=current.get('key'),step=kwargs.get('step'),time=time.time()))
            return result
    standard.ModelClient=ProgressClient
    def plus_env(task,resolution,seed):
        instruction,source=resolve_task_instruction(task)
        current.update(task_name=task.name,instruction=instruction,
                       instruction_source_bddl=source)
        return _get_libero_env(task._replace(language=instruction),resolution,seed)
    standard._get_libero_env=plus_env
    last_suite=None
    while True:
        job=rpc('/claim',dict(last_suite=last_suite))
        if job.get('done'):return
        if job.get('wait'):time.sleep(1);continue
        current.clear();current.update(job)
        atomic_json(progress,dict(key=job['key'],step=-1,time=time.time()))
        try:
            execute_job(current,a,standard)
        except Exception:
            error=traceback.format_exc();logging.error(error)
            atomic_json(Path(a.output_dir)/'errors'/(job['key']+f"_attempt{job['attempt']}.json"),
                dict(job=job,worker=a.worker_id,error=error))
            rpc('/error',dict(key=job['key'],error=error))
        else:
            rpc('/complete',dict(key=job['key']))
        last_suite=job['suite']
if __name__=='__main__':main()
