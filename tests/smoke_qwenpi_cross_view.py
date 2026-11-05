"""Real Qwen3-VL-2B + original QwenPI action-head smoke (one accelerator).
No dataset, optimizer step, training submission or checkpoint write.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(23)
    cfg=OmegaConf.load('examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_da3_libero_action_goals.yaml')
    cfg.trainer.gradient_checkpointing=True
    model=Qwen_PI(cfg).cuda().train()
    vision=model.qwen_vl_interface.model.model.visual
    assert vision.global_attention_indexes==(16,18,20,22)
    assert vision.num_register_tokens == 4
    assert not any('interaction_stages' in k or 'shared_init' in k or 'gate_mlp' in k for k in model.state_dict())
    rng=np.random.default_rng(23)
    def example(views):
        mask=np.ones((16,16),dtype=bool);mask[:,8:]=False
        return dict(image=[Image.fromarray(rng.integers(0,256,(224,224,3),dtype=np.uint8)) for _ in range(views)],
            lang='pick up the object',action=rng.normal(0,.1,(16,16)).astype(np.float32),action_mask=mask)
    batch=[example(3),example(2)]
    loss=model(batch)['action_loss'];assert torch.isfinite(loss);loss.backward()
    grads={}
    for i in vision.global_attention_indexes:
        g=vision.blocks[i].attn.qkv.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
        grads[str(i)]=float(g.float().norm())
    register_grad = vision.register_tokens.weight.grad
    assert register_grad is not None and torch.isfinite(register_grad).all() and register_grad.abs().sum() > 0
    register_grad_norm = float(register_grad.float().norm())
    model.zero_grad(set_to_none=True);model.eval()
    actions=model.predict_action([dict(image=batch[0]['image'],lang=batch[0]['lang'])])['normalized_actions']
    assert actions.shape==(1,16,16) and np.isfinite(actions).all()
    report = dict(
        status="passed",
        framework=cfg.framework.name,
        device=torch.cuda.get_device_name(),
        global_layers=list(vision.global_attention_indexes),
        batch_views=[3, 2],
        loss=float(loss.detach()),
        global_qkv_grad_norms=grads,
        action_shape=list(actions.shape),
        registers_per_view=vision.num_register_tokens,
        register_grad_norm=register_grad_norm,
        gradient_checkpointing=True,
        loaded_base_vlm_weights=cfg.framework.qwenvl.load_weights,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
