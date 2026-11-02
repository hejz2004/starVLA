"""One BF16 anchored-goal policy replica per PPU, with WYT input and RNG semantics."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from starVLA.dataloader.libero_action_contract import VERSION, REPRESENTATION


def validate_metadata(meta):
    if meta.get('action_representation')!=REPRESENTATION or meta.get('action_chunk_size')!=16:
        raise ValueError('Expected latest anchored-goal binary/chunk16 interface')
    if meta.get('action_contract',{}).get('version')!=VERSION:
        raise ValueError('Missing versioned binary goal contract')
    if meta.get('available_unnorm_keys')!=['franka']:
        raise ValueError('Expected matched franka normalization')


def configure_gate_ablation(policy, force_zero_gate):
    interface = getattr(policy._framework, 'qwen_vl_interface', None)
    visual = getattr(getattr(interface, 'model', None), 'visual', None)
    stages = getattr(visual, 'interaction_stages', None)
    if force_zero_gate:
        if not stages or not callable(getattr(interface, 'set_force_zero_gate', None)):
            raise ValueError('force_zero_gate requires an active dual-stream vision tower')
        interface.set_force_zero_gate(True)
    flags = {name: bool(stage.force_zero_gate) for name, stage in stages.items()} if stages else {}
    if any(value != force_zero_gate for value in flags.values()):
        raise ValueError('Interaction stage gate state does not match requested ablation')
    return dict(force_zero_gate=force_zero_gate, interaction_stage_force_zero_gate=flags)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt_path',required=True);p.add_argument('--port',required=True,type=int)
    p.add_argument('--metadata_path',required=True);p.add_argument('--use_bf16',action='store_true',required=True)
    p.add_argument('--idle_timeout',type=int,default=-1)
    p.add_argument('--torch-threads',type=int,default=2)
    p.add_argument('--force-zero-gate',action='store_true')
    a=p.parse_args()
    if a.idle_timeout!=-1: p.error('Use idle_timeout=-1 for shared evaluation')
    import numpy as np
    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    from examples.simBenchmarks.LIBERO.eval_files.eval_protocol import atomic_json
    torch.set_num_threads(a.torch_threads);torch.set_num_interop_threads(1)
    class PlusPolicy(PolicyServerWrapper):
        def predict_action(self,examples,**kwargs):
            if len(examples)!=1 or 'state' in examples[0]:
                raise ValueError('One stateless observation is required')
            if len(examples[0]['image'])!=2: raise ValueError('Expected agentview and wrist')
            result=super().predict_action(examples=examples,**kwargs)
            values=np.asarray(result['actions'])
            if values.shape!=(1,16,16) or not np.isfinite(values).all():
                raise ValueError('Expected finite binary-goal [1,16,16] actions')
            return result
    logging.basicConfig(level=logging.INFO)
    policy=PlusPolicy(a.ckpt_path,device='cuda',use_bf16=True,unnorm_key='franka')
    gate_meta=configure_gate_ablation(policy,a.force_zero_gate)
    validate_metadata(policy.metadata)
    meta=dict(policy.metadata,request_seeded=False,policy_seed=None,seed_protocol='wyt-reference-unseeded-policy',
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,
        precision='bf16',policy_threads=a.torch_threads,image_order=['agentview','wrist'],**gate_meta)
    atomic_json(a.metadata_path,meta)
    print('PLUS_POLICY_READY '+json.dumps(meta),flush=True)
    # Match WYT: inference uses the policy RNG without explicit reseeding.
    WebsocketPolicyServer(policy=policy,host='127.0.0.1',port=a.port,idle_timeout=-1,metadata=meta).serve_forever()
if __name__=='__main__':main()
