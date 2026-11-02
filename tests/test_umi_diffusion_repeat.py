"""Exercise real framework forwards without loading VLM weights; audit epoch budget."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.umi_wyt import resolve_umi_stages
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI


class CaptureHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.seen = None

    def forward(self, features, actions, state, **kwargs):
        self.seen = (features, actions, state, kwargs)
        return actions.square().mean() * self.scale


def forward_probe(cls, repeat):
    model = cls.__new__(cls)
    baseframework.__init__(model)
    values = {} if repeat is None else {'repeated_diffusion_steps': repeat}
    model.config = OmegaConf.create({'framework': {'action_model': values}})
    model.action_horizon = 16
    features = [torch.arange(2*5*8,dtype=torch.float32).reshape(2,5,8)]
    attention_mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,0]])
    model._encode_vl_hidden_states = lambda images,instructions: (features,attention_mask)
    model.action_model = CaptureHead()
    mask = np.ones((16,16),dtype=bool)
    mask[-1,-1] = False
    examples = [dict(image=[],lang=f'example {i}',
        action=np.full((16,16),i+1,dtype=np.float32),
        state=np.full((1,16),i+3,dtype=np.float32)) for i in range(2)]
    examples[0]['action_mask'] = mask
    result = model.forward(examples)
    expected_repeat = 2 if repeat is None else repeat
    seen_features,seen_actions,seen_state,kwargs = model.action_model.seen
    expected_actions = torch.from_numpy(np.stack([x['action'] for x in examples]))
    expected_state = torch.from_numpy(np.stack([x['state'] for x in examples]))
    torch.testing.assert_close(seen_actions,expected_actions.repeat(expected_repeat,1,1))
    torch.testing.assert_close(seen_state,expected_state.repeat(expected_repeat,1,1))
    torch.testing.assert_close(seen_features[0],features[0].repeat(expected_repeat,1,1))
    torch.testing.assert_close(kwargs['encoder_attention_mask'],attention_mask.bool().repeat(expected_repeat,1))
    expected_mask = torch.from_numpy(np.stack([mask,np.ones_like(mask)]))
    torch.testing.assert_close(kwargs['action_mask'],expected_mask.repeat(expected_repeat,1,1))
    assert torch.isfinite(result['action_loss'])
    result['action_loss'].backward()
    assert model.action_model.scale.grad is not None
    return seen_actions.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    repeats = {}
    for cls in (Qwen_PI,):
        repeats[cls.__name__] = {str(value):forward_probe(cls,value) for value in (1,2,3,None)}
        for invalid in (0,-1,True,1.5,'2'):
            try:
                forward_probe(cls,invalid)
            except ValueError as error:
                assert 'positive integer' in str(error)
            else:
                raise AssertionError(f'{cls.__name__} accepted repeat={invalid!r}')
    cfg = OmegaConf.load('examples/umi/train_files/starvla_qwenpi_da3_umi_v2.yaml')
    assert cfg.framework.action_model.repeated_diffusion_steps == 1
    assert cfg.datasets.vla_data.per_device_batch_size == 32
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert cfg.trainer.max_train_steps == 77732
    resolve_umi_stages(cfg,64)
    assert cfg.datasets.vla_data.stage_global_batch_size == 2048
    stages = []
    for stage in cfg.datasets.vla_data.stages:
        ranges = np.load(Path(stage.index_dir)/'ranges.npy',mmap_mode='r')
        count = int(np.sum((ranges[:,2]-ranges[:,1]+5)//6))
        assert count == stage.num_windows
        stages.append(dict(name=stage.name,windows=count,steps=count//2048,dropped=count%2048))
    assert sum(s['windows'] for s in stages) == 159198571
    assert [s['steps'] for s in stages] == [15548,15546,15547,15545,15546]
    assert sum(s['steps'] for s in stages) == cfg.trainer.max_train_steps
    assert cfg.datasets.vla_data.stage_total_updates == cfg.trainer.max_train_steps
    smoke = OmegaConf.load('examples/umi/train_files/starvla_qwenpi_da3_umi_smoke.yaml')
    assert smoke.framework.action_model.repeated_diffusion_steps == 1
    result = dict(repeat_forward_batch_sizes=repeats,invalid_repeat_rejected=True,
        global_batch_size=2048,total_windows=159198571,stages=stages,
        steps_per_epoch=sum(s['steps'] for s in stages),dropped_windows=sum(s['dropped'] for s in stages),
        flat_pool_floor_steps=159198571//2048,raw_vlm_weights_loaded=False)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'repeat_budget.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
