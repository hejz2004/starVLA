#!/usr/bin/env python3
"""Real LIBERO worker capacity probe. Short rollouts are NOT accuracy results."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from examples.simBenchmarks.LIBERO.eval_files import parallel_eval as parallel
from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import atomic_json,digest


def worker(a):
    import numpy as np
    from examples.simBenchmarks.LIBERO.eval_files import eval_libero as evaluation
    output=Path(a.output_dir)
    timing=dict(first_step_at=None,last_step_at=None,steps=0,
        measured_first_at=None,measured_last_at=None,measured_steps=0,
        client_seconds=0.0,infer_request_seconds=[],video_seconds=0.0)
    OriginalClient=evaluation.ModelClient
    class TimedClient(OriginalClient):
        def step(self,*args,**kwargs):
            step=kwargs.get('step',0)
            refresh=step%self.action_chunk_size==0 or self.raw_actions is None
            start=time.monotonic()
            result=super().step(*args,**kwargs)
            end=time.monotonic()
            if timing['first_step_at'] is None: timing['first_step_at']=start
            timing['last_step_at']=end
            timing['steps']+=1
            timing['client_seconds']+=end-start
            if step>=32:
                if timing['measured_first_at'] is None: timing['measured_first_at']=start
                timing['measured_last_at']=end
                timing['measured_steps']+=1
                if refresh: timing['infer_request_seconds'].append(end-start)
            if timing['steps']%16==0: atomic_json(output/'progress.json',timing)
            return result
    original_video=evaluation._save_rollout_video
    def timed_video(*args,**kwargs):
        start=time.monotonic()
        result=original_video(*args,**kwargs)
        timing['video_seconds']+=time.monotonic()-start
        return result
    evaluation.ModelClient=TimedClient
    evaluation._save_rollout_video=timed_video
    evaluation.MAX_STEPS=dict(evaluation.MAX_STEPS,**{a.suite:a.steps})
    started=time.monotonic();error=None
    try:
        evaluation.eval_libero(evaluation.Args(host='127.0.0.1',port=a.port,
            task_suite_name=a.suite,task_ids=[0],num_trials_per_task=a.trials,
            seed=7,pretrained_path=a.ckpt,action_interface='anchored_goal_binary',
            deterministic_inference=True,results_dir=str(output),
            protocol_digest=digest(dict(benchmark_only=True,steps=a.steps,trials=a.trials)),
            video_out_path=str(output),save_video=True))
    except BaseException as exc:
        error=str(exc)
        raise
    finally:
        timing.update(benchmark_only=True,rollout_cap=a.steps,trials=a.trials,
            elapsed_seconds=time.monotonic()-started,error=error)
        lat=timing['infer_request_seconds']
        timing['request_latency_p50']=float(np.median(lat)) if lat else None
        timing['request_latency_p95']=float(np.percentile(lat,95)) if lat else None
        atomic_json(output/'metrics.json',timing)


def gpu_sample():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True)
    values=[]
    for line in p.stdout.splitlines():
        try:
            gpu,memory,util=[float(x.strip()) for x in line.split(',')]
            values.append(dict(device=int(gpu),memory_mib=memory,util_percent=util))
        except ValueError: pass
    return values


def main(a):
    output=Path(a.output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'config.json',dict(vars(a),benchmark_only=True,
        architecture='one seeded model service per device; CPU simulator workers',
        video=True,policy_bf16=True,cpu_affinity=len(os.sched_getaffinity(0)),
        selection='fewest workers within 5 percent of best measured steady rollout throughput'))
    defaults=parallel.parser().parse_args(['--ckpt',a.ckpt,'--output-dir',str(output),
        '--simulation-python',sys.executable,'--policy-python',sys.executable,
        '--simulation-site',str(ROOT/'playground/Artifacts/libero_sim_runtime')])
    sim_env=parallel.simulation_env(defaults)
    servers=[];children=[];logs=[];reports=[]
    def interrupted(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    def start_server(device):
        import socket
        from websockets.sync.client import connect
        from deployment.model_server.tools import msgpack_numpy
        port=a.base_port+device
        with socket.socket() as s: s.bind(('127.0.0.1',port))
        log=(output/f'policy_{device}.log').open('w',buffering=1);logs.append(log)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',DEBUG='')
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'examples/simBenchmarks/LIBERO/eval_files/parallel_eval.py'),
            '--serve','--ckpt',a.ckpt,'--output-dir',str(output),'--port',str(port)],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);servers.append((device,p))
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if p.poll() is not None: raise RuntimeError(f'Policy {device} exited; see log')
            try:
                with connect(f'ws://127.0.0.1:{port}',open_timeout=1,proxy=None,
                    max_size=None,compression=None,ping_interval=None) as ws:
                    metadata=msgpack_numpy.unpackb(ws.recv(timeout=2))
                if not metadata.get('request_seeded'): raise RuntimeError('Unseeded service')
                atomic_json(output/f'policy_metadata_{device}.json',metadata)
                print(f'POLICY_READY device={device}',flush=True)
                return
            except (OSError,TimeoutError): time.sleep(.5)
        raise TimeoutError('Policy startup timed out')
    def run_case(devices,workers):
        name=f'd{len(devices)}_w{workers}';case=output/name;case.mkdir()
        print(f'CASE_START {name}',flush=True)
        active=[];telemetry=[];started=time.monotonic()
        try:
            for device in devices:
                for index in range(workers):
                    wd=case/f'device_{device}_worker_{index}';wd.mkdir()
                    log=(wd/'worker.log').open('w',buffering=1);logs.append(log)
                    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                        '--worker','--ckpt',a.ckpt,'--output-dir',str(wd),
                        '--port',str(a.base_port+device),'--steps',str(a.steps),
                        '--trials',str(a.trials),'--suite',a.suite],cwd=ROOT,env=sim_env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(p);active.append((wd,p))
            while any(p.poll() is None for _,p in active):
                if any(p.poll() is not None for _,p in servers): raise RuntimeError('Policy exited')
                failed=[(str(wd),p.returncode) for wd,p in active if p.poll() not in [None,0]]
                if failed: raise RuntimeError(str(failed))
                if time.monotonic()-started>a.case_timeout: raise TimeoutError(name)
                telemetry.append(dict(elapsed=time.monotonic()-started,gpus=gpu_sample()))
                time.sleep(2)
            metrics=[json.loads((wd/'metrics.json').read_text()) for wd,_ in active]
            first=min(m['first_step_at'] for m in metrics);last=max(m['last_step_at'] for m in metrics)
            mf=min(m['measured_first_at'] for m in metrics if m['measured_first_at'] is not None)
            ml=max(m['measured_last_at'] for m in metrics if m['measured_last_at'] is not None)
            steps=sum(m['steps'] for m in metrics);measured=sum(m['measured_steps'] for m in metrics)
            hashes=[]
            for wd,_ in active:
                hashes.append({r['key']:(r['initial_state_sha256'],r['first_images_sha256'],r['first_chunk_sha256'])
                    for f in wd.glob('*/results/*.json') for r in [json.loads(f.read_text())]})
            elapsed=time.monotonic()-started
            result=dict(case=name,status='complete',devices=devices,workers_per_device=workers,
                total_workers=len(active),steps=steps,measured_steps=measured,
                rollout_steps_per_second=steps/(last-first),steady_steps_per_second=measured/(ml-mf),
                elapsed_seconds=elapsed,end_to_end_steps_per_second=steps/elapsed,
                deterministic_hashes_match=all(h==hashes[0] for h in hashes),
                gpu_peak_memory_mib={str(d):max((g['memory_mib'] for t in telemetry for g in t['gpus'] if g['device']==d),default=0) for d in devices},
                gpu_mean_util_percent={str(d):sum(g['util_percent'] for t in telemetry for g in t['gpus'] if g['device']==d)/max(1,sum(g['device']==d for t in telemetry for g in t['gpus'])) for d in devices},
                worker_metrics=metrics)
        except Exception as exc:
            result=dict(case=name,status='failed',devices=devices,workers_per_device=workers,error=str(exc))
        finally: parallel.stop_processes([p for _,p in active])
        atomic_json(case/'telemetry.json',telemetry);atomic_json(case/'result.json',result)
        reports.append(result);atomic_json(output/'summary.json',dict(status='running',cases=reports))
        print('CASE_RESULT '+json.dumps({k:v for k,v in result.items() if k!='worker_metrics'}),flush=True)
        return result
    try:
        if a.start_devices_together:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(start_server,[0,1]))
        else:
            start_server(0)
        for w in a.worker_counts:
            r=run_case([0],w)
            if r['status']!='complete': break
        good=[r for r in reports if r['status']=='complete' and r.get('deterministic_hashes_match')]
        if not good: raise RuntimeError('No successful single-card probe')
        best=max(r['steady_steps_per_second'] for r in good)
        selected=min(r['workers_per_device'] for r in good if r['steady_steps_per_second']>=.95*best)
        if not a.start_devices_together:
            start_server(1)
        for w in (a.dual_worker_counts or sorted(set([max(1,selected//2),selected]))):
            run_case([0,1],w)
        dual=[r for r in reports if len(r['devices'])==2 and r['status']=='complete' and r.get('deterministic_hashes_match')]
        if not dual: raise RuntimeError('No successful two-card probe')
        bestdual=max(r['steady_steps_per_second'] for r in dual)
        chosen=min((r for r in dual if r['steady_steps_per_second']>=.95*bestdual),key=lambda r:r['workers_per_device'])
        atomic_json(output/'summary.json',dict(status='complete',cases=reports,
            single_card_recommended_workers=selected,
            max_tested_stable_workers=max(r['workers_per_device'] for r in good),
            recommended_devices=chosen['devices'],recommended_workers_per_device=chosen['workers_per_device'],
            benchmark_only=True,hard_worker_limit_tested=False))
        print('BENCHMARK_COMPLETE',flush=True)
    except BaseException as exc:
        atomic_json(output/'summary.json',dict(status='failed',error=str(exc),cases=reports));raise
    finally:
        parallel.stop_processes(children)
        for log in logs: log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--worker-counts',nargs='+',type=int,default=[1,2,4,8,12,16])
    p.add_argument('--dual-worker-counts',nargs='+',type=int,default=[])
    p.add_argument('--start-devices-together',action='store_true')
    p.add_argument('--steps',type=int,default=128);p.add_argument('--trials',type=int,default=2)
    p.add_argument('--suite',default='libero_object');p.add_argument('--base-port',type=int,default=12093)
    p.add_argument('--port',type=int,default=12093);p.add_argument('--case-timeout',type=float,default=600)
    a=p.parse_args()
    if a.worker: worker(a)
    else: main(a)
