"""Versioned Plus jobs, CPU placement, resumable artifacts and work stealing."""
from __future__ import annotations
from collections import Counter, deque
import hashlib
from importlib.metadata import PathDistribution
import json
import os
from pathlib import Path
import threading
import time
import sys
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import (
    MAX_STEPS, VIDEO_LAYOUT, SUITE_FOLDERS, atomic_json, digest, episode_key,
    suite_directory, valid_result,
)
from starVLA.dataloader.libero_action_contract import VERSION, REPRESENTATION
SUITE_SIZES={'libero_10':2519,'libero_goal':2591,'libero_object':2518,'libero_spatial':2402}
SCHEMA='libero-plus-anchored-goal-binary-v2'


def make_jobs(classification,suites,task_ids=(),max_per_category=0):
    if len(set(suites))!=len(suites) or not suites: raise ValueError('Select distinct Plus suites')
    jobs=[]
    for suite in suites:
        items=classification[suite]
        if len(items)!=SUITE_SIZES[suite] or {x['id'] for x in items}!=set(range(1,len(items)+1)):
            raise ValueError('Plus classification count/IDs differ: '+suite)
        counts=Counter()
        for item in sorted(items,key=lambda x:x['id']):
            task_id=item['id']-1
            if task_ids and task_id not in task_ids: continue
            if max_per_category and counts[item['category']]>=max_per_category: continue
            counts[item['category']]+=1
            jobs.append(dict(suite=suite,task_id=task_id,trial=0,key=episode_key(suite,task_id,0),
                variant_id=item['id'],variant=item['name'],category=item['category']))
        if task_ids and any(x<0 or x>=len(items) for x in task_ids): raise ValueError('Variant outside suite')
    if not jobs: raise ValueError('No selected Plus variants')
    return jobs


def cpu_plan(cpus,devices,workers):
    cpus=sorted(set(cpus))
    if workers<1 or not devices or len(set(devices))!=len(devices): raise ValueError('Invalid topology')
    if len(cpus)<len(devices): raise ValueError('Not enough CPUs for model services')
    groups=[]
    for i,device in enumerate(devices):
        group=cpus[i*len(cpus)//len(devices):(i+1)*len(cpus)//len(devices)]
        # All CPUs are usable by simulators; model preprocessing shares the last two.
        groups.append(dict(device=str(device),cpus=group,policy_cpus=group[-min(2,len(group)):],
            worker_cpus=[[group[w%len(group)]] for w in range(workers)]))
    return groups


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_versions(site):
    # Gym wheels reorder metadata requirements on rebuild; versions are stable.
    return {p.name: PathDistribution(p).version for p in sorted(Path(site).glob('*.dist-info'))
            if (p/'METADATA').is_file()} if site else {}


def make_protocol(args,jobs):
    from omegaconf import OmegaConf
    ckpt=Path(args.ckpt).resolve(strict=True)
    cfg=OmegaConf.load(ckpt.parent.parent/'config.yaml')
    am=cfg.framework.action_model
    if (cfg.datasets.vla_data.get('action_type')!=VERSION or am.action_dim!=16
        or am.action_horizon!=16 or am.get('use_state',False)):
        raise ValueError('Checkpoint must use the latest stateless anchored-goal binary contract')
    # Hash execution and action-adapter files, not tests or unrelated training tools.
    critical=[Path(__file__),Path(__file__).with_name('evaluation_controller.py'),Path(__file__).with_name('eval_worker.py'),
        Path(__file__).parents[1]/'policy_entry.py',Path(__file__).parents[1]/'preflight.py',
        Path(__file__).with_name('video_artifacts.py')]
    critical += [ROOT/'examples/simBenchmarks/LIBERO/eval_files'/name for name in
                 ['eval_libero.py','model2libero_interface.py','action_goal_binary.py']]
    critical += [ROOT/'starVLA/dataloader/libero_action_contract.py',ROOT/'deployment/model_server/policy_norm_processor.py']
    sources={str(p.relative_to(ROOT)):file_hash(p) for p in critical}
    stat=ckpt.stat()
    dependencies=dependency_versions(args.simulation_site)
    return dict(schema=SCHEMA,checkpoint=dict(path=str(ckpt),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),
        model_files={p.name:file_hash(p) for p in ckpt.parent.parent.iterdir()
                     if p.name in ['config.yaml','config.full.yaml','dataset_statistics.json']},
        source_sha256=sources,jobs=jobs,seed=args.seed,action_contract=VERSION,
        force_zero_gate=bool(getattr(args,'force_zero_gate',False)),
        action_representation=REPRESENTATION,action_interface='anchored_goal_binary',
        seed_protocol='wyt-reference-unseeded-policy',policy_seed=None,request_seeded=False,action_chunk_size=16,
        instruction_source='bddl.language_instruction',instruction_protocol='libero-plus-bddl-v1',image_order=['agentview_image','robot0_eye_in_hand_image'],
        image_rotation_degrees=180,image_resize='PIL BICUBIC 224x224',state_sent_to_policy=False,
        num_steps_wait=10,max_steps={s:MAX_STEPS[s] for s in args.suites},
        initial_state_index=0,render='osmesa',numba_disable_jit=True,bf16=True,
        save_video=not args.no_video,video_layout=VIDEO_LAYOUT,video_fps=10,
        classification_sha256=file_hash(args.classification),
        libero_home=str(Path(args.libero_home).resolve()),libero_assets=str(Path(args.libero_assets).resolve()),
        libero_base_assets=str(Path(args.libero_base_assets).resolve()) if args.libero_base_assets else '',
        simulation_python=args.simulation_python,policy_python=args.policy_python,
        simulation_site=str(Path(args.simulation_site).resolve()) if args.simulation_site else '',
        simulation_dependencies=dependencies)


def result_path(output,job): return suite_directory(output,job['suite'])/'results'/(job['key']+'.json')


def read_result(output,job,protocol_digest,video=True):
    p=result_path(output,job)
    if not valid_result(p,job,protocol_digest,video,required_video_layout=VIDEO_LAYOUT): return None
    row=json.loads(p.read_text())
    for name in ['variant_id','variant','category']:
        if row.get(name)!=job[name]: raise ValueError('Plus result identity mismatch: '+str(p))
    if row.get('action_representation')!=REPRESENTATION or row['steps']>MAX_STEPS[job['suite']]:
        raise ValueError('Invalid action protocol or step count: '+str(p))
    if video and row.get('video_frames')!=row['steps']:raise ValueError('Video/action frame count mismatch')
    base=p.parent.parent.resolve()
    for name in ['trace']+(['video'] if video else []):
        target=(base/row[name]).resolve()
        if not target.is_relative_to(base): raise ValueError('Artifact escapes suite output')
    return row


def load_results(output,jobs,pdigest,video):
    known={j['key']:j for j in jobs}; rows={}
    for folder in SUITE_FOLDERS.values():
        for p in (Path(output)/folder/'results').glob('*.json'):
            if p.stem not in known: raise ValueError('Unexpected result: '+str(p))
            if p!=result_path(output,known[p.stem]):raise ValueError('Result is in the wrong suite directory')
            if p.stem in rows: raise ValueError('Duplicate episode result')
            row=read_result(output,known[p.stem],pdigest,video)
            if row is not None: rows[p.stem]=row
    return rows


class WorkQueue:
    """Parent-owned queue: one claim per worker, locality plus cross-suite stealing."""
    def __init__(self,jobs,completed=(),max_attempts=2):
        self.jobs={j['key']:j for j in jobs}; self.done=set(completed)
        self.pending={s:deque() for s in sorted({j['suite'] for j in jobs})}
        for j in jobs:
            if j['key'] not in self.done: self.pending[j['suite']].append(j['key'])
        self.active={};self.attempts=Counter();self.failed={};self.max_attempts=max_attempts
        self.lock=threading.RLock()
    def claim(self,worker,last_suite=None):
        with self.lock:
            if worker in self.active: return dict(self.jobs[self.active[worker]],attempt=self.attempts[self.active[worker]])
            available={s:len(q)*MAX_STEPS[s] for s,q in self.pending.items() if q}
            if not available: return {'wait':True} if self.active else {'done':True}
            suite=max(available,key=available.get)
            if last_suite in available and available[last_suite]>=.5*available[suite]: suite=last_suite
            key=self.pending[suite].popleft();self.active[worker]=key;self.attempts[key]+=1
            return dict(self.jobs[key],attempt=self.attempts[key])
    def finish(self,worker,key):
        with self.lock:
            if key in self.done: return
            if self.active.get(worker)!=key: raise ValueError('Result does not belong to this worker')
            del self.active[worker];self.done.add(key)
    def fail(self,worker,error):
        with self.lock:
            key=self.active.pop(worker,None)
            if key is None:return
            if self.attempts[key]>=self.max_attempts:self.failed[key]=str(error)
            else:self.pending[self.jobs[key]['suite']].append(key)
    def snapshot(self):
        with self.lock:
            return dict(pending=sum(map(len,self.pending.values())),in_flight=dict(self.active),
                        completed=len(self.done),failed=dict(self.failed),attempts=dict(self.attempts))


def summarize(output,jobs,rows,queue,started,status='running',error=None):
    values=list(rows.values()); suites=[]
    for suite in sorted({j['suite'] for j in jobs}):
        selected=[r for r in values if r['suite']==suite];expected=[j for j in jobs if j['suite']==suite]
        categories=[]
        for category in sorted({j['category'] for j in expected}):
            r=[x for x in selected if x['category']==category];count=sum(x['success'] for x in r)
            categories.append(dict(category=category,expected=sum(j['category']==category for j in expected),
                completed=len(r),successes=count,success_rate=count/len(r) if r else None))
        successes=sum(r['success'] for r in selected)
        item=dict(suite=suite,expected=len(expected),completed=len(selected),successes=successes,
            success_rate=successes/len(selected) if selected else None,categories=categories)
        atomic_json(suite_directory(output,suite)/'summary.json',item);suites.append(item)
    count=sum(r['success'] for r in values)
    summary=dict(schema=SCHEMA,status=status,expected=len(jobs),completed=len(values),successes=count,
        success_rate=count/len(values) if values else None,suites=suites,error=error,
        elapsed_seconds_this_launch=time.monotonic()-started,queue=queue.snapshot(),
        completed_steps=sum(r['steps'] for r in values))
    atomic_json(Path(output)/'summary.json',summary)
    return summary
