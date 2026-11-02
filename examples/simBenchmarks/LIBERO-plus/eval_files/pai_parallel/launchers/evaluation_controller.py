#!/usr/bin/env python3
"""Single-node Plus: one model per PPU, twenty persistent workers per device."""
from __future__ import annotations
import argparse
import fcntl
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from plus_protocol import (ROOT,SUITE_SIZES,SCHEMA,VERSION,REPRESENTATION,VIDEO_LAYOUT,
    WorkQueue,atomic_json,cpu_plan,digest,file_hash,load_results,make_jobs,make_protocol,
    read_result,summarize)
BUNDLE=Path(__file__).resolve().parents[1]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--libero-home',required=True);p.add_argument('--libero-assets',required=True)
    p.add_argument('--libero-base-assets',default='');p.add_argument('--simulation-site',default='')
    p.add_argument('--policy-python',default=sys.executable);p.add_argument('--simulation-python',default=sys.executable)
    p.add_argument('--devices',default=','.join(map(str,range(16))))
    p.add_argument('--workers-per-device',type=int,default=20);p.add_argument('--cpu-budget',type=int)
    p.add_argument('--suites',nargs='+',choices=SUITE_SIZES,default=list(SUITE_SIZES))
    p.add_argument('--task-ids',nargs='*',type=int,default=[])
    p.add_argument('--max-per-category',type=int,default=0,help='Bounded real smoke coverage; 0 is all variants')
    p.add_argument('--classification',default=str(BUNDLE/'launchers/task_classification.json'))
    p.add_argument('--base-port',type=int,default=12883);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--server-start-timeout',type=float,default=900)
    p.add_argument('--worker-stall-timeout',type=float,default=900)
    p.add_argument('--resume',action='store_true');p.add_argument('--no-video',action='store_true')
    p.add_argument('--force-zero-gate',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def affinity_exec(python,argv,cpus,env,log):
    # Set affinity before importing Torch / GL and spawning their child threads.
    code='import os;os.sched_setaffinity(0,'+repr(set(cpus))+');os.execv('+repr(python)+','+repr([python]+argv)+')'
    return subprocess.Popen([python,'-c',code],cwd=ROOT,env=env,stdout=log,
                            stderr=subprocess.STDOUT,start_new_session=True)


def stop(children):
    for p in children:
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+8
    for p in children:
        try:p.wait(timeout=max(.01,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()


def prepare_resources(a,output):
    env=dict(os.environ,RUN_ROOT=str(output),STARVLA_DIR=str(ROOT),CKPT_PATH=a.ckpt,
        LIBERO_HOME=a.libero_home,LIBERO_ASSETS=a.libero_assets,LIBERO_BASE_ASSETS=a.libero_base_assets,
        PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    (output/'preflight').mkdir(exist_ok=True)
    with (output/'preflight/inputs.log').open('w') as log:
        subprocess.run([a.policy_python,str(BUNDLE/'preflight.py'),'inputs'],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    paths=[str(ROOT),str(output/'libero_source')]
    if a.simulation_site:paths.insert(0,a.simulation_site)
    sim=dict(os.environ,PYTHONPATH=os.pathsep.join(paths),LIBERO_HOME=str(output/'libero_source'),
        LIBERO_CONFIG_PATH=str(output/'libero_config'),MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',
        NUMBA_DISABLE_JIT='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',CUDA_VISIBLE_DEVICES='',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',LP_NUM_THREADS='1',MAGICK_THREAD_LIMIT='1',
        OPENCV_FOR_THREADS_NUM='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
        PYTHONUNBUFFERED='1',DEBUG='',NO_ALBUMENTATIONS_UPDATE='1')
    sim['RUN_ROOT']=str(output)
    sim['STARVLA_DIR']=str(ROOT)
    with (output/'preflight/simulation.log').open('w') as log:
        subprocess.run([a.simulation_python,str(BUNDLE/'preflight.py'),'simulation'],cwd=ROOT,env=sim,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    current=json.loads((output/'preflight/simulation_runtime.json').read_text())
    runtime=output/'simulation_runtime.json'
    if runtime.exists() and json.loads(runtime.read_text())!=current:
        raise ValueError('Simulator runtime changed; refusing resume')
    atomic_json(runtime,current)
    return sim


def run(a):
    devices=a.devices.split(',')
    if not all(x.isdigit() for x in devices) or len(set(devices))!=len(devices):raise ValueError('Distinct numeric devices required')
    if not 1<=a.base_port<=65536-len(devices):raise ValueError('Invalid policy port range')
    if a.max_per_category<0:raise ValueError('Invalid variant count')
    for name in ['policy_python','simulation_python']:
        value=shutil.which(getattr(a,name))
        if not value:raise ValueError('Interpreter unavailable: '+name)
        setattr(a,name,os.path.abspath(value))
    a.ckpt=str(Path(a.ckpt).resolve(strict=True))
    jobs=make_jobs(json.loads(Path(a.classification).read_text()),a.suites,a.task_ids,a.max_per_category)
    protocol=make_protocol(a,jobs);pdigest=digest(protocol)
    available=sorted(os.sched_getaffinity(0));budget=a.cpu_budget or len(available)
    if budget<1:raise ValueError('CPU budget must be positive')
    if budget>len(available) and not a.dry_run:raise ValueError('CPU budget exceeds actual allowed CPU affinity')
    cpus=available[:budget] if budget<=len(available) else list(range(budget))
    topology=cpu_plan(cpus,devices,a.workers_per_device)
    plan=dict(schema=SCHEMA,policy_replicas=len(devices),workers_per_device=a.workers_per_device,
        total_workers=len(devices)*a.workers_per_device,cpu_count=len(cpus),topology=topology,
        expected=len(jobs),suites={s:sum(j['suite']==s for j in jobs) for s in a.suites},
        scheduling='global queue; suite locality; cross-suite work stealing',
        protocol_digest=pdigest,hardware_verified=False)
    if a.dry_run:print(json.dumps(plan,indent=2));return
    output=Path(a.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        manifest=output/'protocol.json'
        if manifest.exists():
            if not a.resume:raise ValueError('Existing run requires --resume')
            previous=json.loads(manifest.read_text())
            if previous!=protocol:
                changed=sorted(k for k in previous.keys()|protocol.keys() if previous.get(k)!=protocol.get(k))
                raise ValueError('Resume protocol mismatch: '+', '.join(changed))
        else:
            if any(p.name!='.lock' for p in output.iterdir()):raise ValueError('New output must be empty')
            atomic_json(manifest,protocol)
        rows=load_results(output,jobs,pdigest,not a.no_video)
    except BaseException:
        lock.close();raise
    queue=WorkQueue(jobs,rows,max_attempts=1)
    started=time.monotonic();children=[];logs=[];servers=[];workers={};http=None
    handler=logging.FileHandler(output/'bootstrap.log');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.basicConfig(level=logging.INFO,handlers=[handler,logging.StreamHandler(sys.stdout)],force=True)
    def summary(status='running',error=None):
        with queue.lock:
            s=summarize(output,jobs,rows,queue,started,status,error)
            atomic_json(output/'queue_state.json',queue.snapshot())
        logging.info('Progress %s/%s successes=%s active=%s pending=%s',s['completed'],s['expected'],s['successes'],len(s['queue']['in_flight']),s['queue']['pending'])
        return s
    try:
        atomic_json(output/'launch.json',dict(plan,pid=os.getpid(),started_at=time.time()))
        if len(rows)==len(jobs):
            summary('complete')
            if not (output/'completion.json').exists():atomic_json(output/'completion.json',dict(status='complete',protocol_digest=pdigest,expected=len(jobs),completed=len(rows),recovered_from_completed_artifacts=True))
            return
        sim=prepare_resources(a,output)
        count=subprocess.check_output([a.policy_python,'-c','import torch;print(torch.cuda.device_count())'],text=True).strip()
        if int(count)<=max(map(int,devices)):raise ValueError('Requested PPU device is unavailable')
        token=secrets.token_hex(24);valid_workers={f'd{d}_w{w:02d}' for d in devices for w in range(a.workers_per_device)}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    if self.headers.get('Authorization')!=token:raise ValueError('Wrong queue token')
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=1000000:raise ValueError('Invalid request size')
                    body=json.loads(self.rfile.read(n));worker=body['worker']
                    if worker not in valid_workers:raise ValueError('Unknown worker')
                    with queue.lock:
                        if self.path=='/claim':answer=queue.claim(worker,body.get('last_suite'))
                        elif self.path=='/complete':
                            key=body['key']
                            if key not in queue.jobs:raise ValueError('Unknown result')
                            if key not in queue.done:
                                if queue.active.get(worker)!=key:raise ValueError('Wrong result owner')
                                row=read_result(output,queue.jobs[key],pdigest,not a.no_video)
                                if row is None:raise ValueError('Incomplete episode artifacts')
                                rows[key]=row;queue.finish(worker,key)
                            answer={'ok':True}
                        elif self.path=='/error':
                            if queue.active.get(worker)!=body['key']:raise ValueError('Wrong failed job owner')
                            queue.fail(worker,body['error']);answer={'ok':True}
                        else:raise ValueError('Unknown operation')
                    data=json.dumps(answer).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
                except Exception as exc:
                    self.send_response(400);self.end_headers();self.wfile.write(str(exc).encode())
        class QueueServer(ThreadingHTTPServer):
            request_queue_size=256
            daemon_threads=True
        http=QueueServer(('127.0.0.1',0),Handler)
        threading.Thread(target=http.serve_forever,daemon=True).start()
        url='http://127.0.0.1:'+str(http.server_address[1])
        reservations=[]
        try:
            for i in range(len(devices)):
                sock=socket.socket();sock.bind(('127.0.0.1',a.base_port+i));reservations.append(sock)
        finally:
            for sock in reservations:sock.close()
        def start_policy(i):
            group=topology[i];device=group['device'];port=a.base_port+i
            directory=output/'policies'/('device_'+device);directory.mkdir(parents=True,exist_ok=True)
            log=(directory/'policy.log').open('a',buffering=1);logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',DEBUG='')
            argv=['-u',str(BUNDLE/'policy_entry.py'),'--ckpt_path',a.ckpt,'--port',str(port),
                  '--metadata_path',str(directory/'metadata.json'),'--use_bf16',
                  '--torch-threads',str(min(2,len(group['policy_cpus'])))]
            if a.force_zero_gate:argv.append('--force-zero-gate')
            process=affinity_exec(a.policy_python,argv,group['policy_cpus'],env,log)
            children.append(process);servers.append(process)
            return port,process
        def await_ready(items):
            from websockets.sync.client import connect
            from deployment.model_server.tools import msgpack_numpy
            deadline=time.monotonic()+a.server_start_timeout;pending=dict(items)
            while pending:
                for port,process in list(pending.items()):
                    if process.poll() is not None:raise RuntimeError(f'Policy {port} exited')
                    try:
                        with connect(f'ws://127.0.0.1:{port}',open_timeout=1,proxy=None,max_size=None,compression=None,ping_interval=None) as ws:
                            m=msgpack_numpy.unpackb(ws.recv(timeout=2))
                        if m.get('action_contract',{}).get('version')!=VERSION or Path(m.get('ckpt_path','')).resolve()!=Path(a.ckpt):
                            raise ValueError('Policy metadata mismatch')
                        if m.get('force_zero_gate') is not a.force_zero_gate:
                            raise ValueError('Policy force_zero_gate mismatch')
                        logging.info('Policy ready %s force_zero_gate=%s',port,a.force_zero_gate);del pending[port]
                    except (OSError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('Policy startup timed out')
                time.sleep(.5)
        # Limit simultaneous NAS/checkpoint loads; models stay resident across all suites.
        for start in range(0,len(devices),4):await_ready([start_policy(i) for i in range(start,min(start+4,len(devices)))])
        def spawn(i,w,restarts=0):
            group=topology[i];worker=f"d{group['device']}_w{w:02d}"
            directory=output/'workers'/worker;directory.mkdir(parents=True,exist_ok=True)
            log=(directory/'run.log').open('a',buffering=1);logs.append(log)
            argv=['-u',str(BUNDLE/'launchers/eval_worker.py'),'--queue-url',url,'--worker-id',worker,
                '--port',str(a.base_port+i),'--ckpt',a.ckpt,'--output-dir',str(output),'--protocol-digest',pdigest,'--seed',str(a.seed)]
            if a.no_video:argv.append('--no-video')
            process=affinity_exec(a.simulation_python,argv,group['worker_cpus'][w],dict(sim,PLUS_QUEUE_TOKEN=token),log)
            children.append(process);workers[worker]=dict(process=process,group=i,index=w,restarts=restarts,spawned=time.time())
            atomic_json(directory/'launch.json',dict(pid=process.pid,device=group['device'],cpus=group['worker_cpus'][w],port=a.base_port+i,restarts=restarts))
        for i in range(len(devices)):
            for w in range(a.workers_per_device):spawn(i,w);time.sleep(.1)
        last=0
        while workers:
            if any(p.poll() is not None for p in servers):raise RuntimeError('Policy exited during rollout')
            for worker,item in list(workers.items()):
                process=item['process'];progress=output/'workers'/worker/'progress.json'
                touched=max(item['spawned'],progress.stat().st_mtime if progress.exists() else 0)
                stalled=process.poll() is None and worker in queue.active and time.time()-touched>a.worker_stall_timeout
                if stalled:stop([process])
                code=process.poll()
                if code is None:continue
                del workers[worker]
                with queue.lock:
                    key=queue.active.get(worker)
                    if key:
                        # A process can die after committing files but before its acknowledgment.
                        row=read_result(output,queue.jobs[key],pdigest,not a.no_video)
                        if row is not None:rows[key]=row;queue.finish(worker,key)
                        else:queue.fail(worker,'worker exited or stalled')
                if code or stalled:
                    raise RuntimeError('Worker failed: '+worker+'; see its log and use --resume after fixing the error')
            if queue.snapshot()['failed']:raise RuntimeError('Episode error; inspect errors/ and resume after fixing it')
            if time.monotonic()-last>=10:
                summary();last=time.monotonic()
            time.sleep(.5)
        failures=queue.snapshot()['failed']
        if failures or len(rows)!=len(jobs):raise RuntimeError(f'Unfinished variants or terminal errors: {len(failures)}')
        summary('complete')
        atomic_json(output/'completion.json',dict(status='complete',schema=SCHEMA,protocol_digest=pdigest,
            expected=len(jobs),completed=len(rows),devices=devices,workers_per_device=a.workers_per_device,
            elapsed_seconds=time.monotonic()-started))
    except BaseException as exc:
        summary('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',str(exc));raise
    finally:
        if http:http.shutdown();http.server_close()
        stop(children)
        for log in logs:log.close()
        lock.close()


def main():
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    run(parser().parse_args())
if __name__=='__main__':main()
