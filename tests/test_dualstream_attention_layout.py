"""Forward/backward and checkpoint replay parity for per-forward bucket reuse."""
import copy
import pytest
import torch
from unittest.mock import patch
from omegaconf import OmegaConf
from starVLA.model.framework.VLM4A.DualStreamQwenPI import DualStreamQwen_PI
from starVLA.model.modules.vlm.qwen3_vl_dual.modeling_qwen3_vl import Qwen3VLVisionModel
from tests.test_dualstream_optimizations import tiny_model


@pytest.mark.parametrize('global_gc,vision_gc', [(True,None),(True,False),(True,True),(False,None),(False,True)])
def test_vision_checkpoint_override_preserves_language_and_action(global_gc, vision_gc):
    interface = torch.nn.Module()
    interface.model = tiny_model()
    # Match the real interface's Qwen3-to-QwenPI config adapter.
    interface.model.config.hidden_size = interface.model.config.text_config.hidden_size
    cfg = OmegaConf.create({'framework': {'name': 'DualStreamPI', 'qwenvl': {
        'vision_gradient_checkpointing': vision_gc}}, 'trainer': {'gradient_checkpointing': global_gc}})
    with patch('starVLA.model.framework.VLM4A.DualStreamQwenPI._QWen3DualStream_VL_Interface',
               return_value=interface):
        model = DualStreamQwen_PI(cfg)
    visual = model.qwen_vl_interface.model.model.visual
    language = model.qwen_vl_interface.model.model.language_model
    expected = global_gc if vision_gc is None else vision_gc
    assert visual.gradient_checkpointing == expected
    assert all(block.gradient_checkpointing == expected for block in visual.blocks)
    assert language.gradient_checkpointing == global_gc
    assert all(layer.gradient_checkpointing == global_gc for layer in language.layers)
    assert all(p.dtype == torch.float32 for p in model.action_model.parameters())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_layout_reuse_outputs_gradients_and_changing_inputs(device, checkpointed, mixed):
    torch.manual_seed(147)
    cfg=copy.deepcopy(tiny_model(production_geometry=True).config.vision_config)
    cfg.depth=24;cfg.deepstack_visual_indexes=[5,11,17]
    cfg.dual_stream['interaction_indexes']=[5,11,17,23]
    cfg.batch_vision_attention=True;cfg.reuse_vision_attention_layout=False
    dtype=torch.bfloat16 if device=='cuda' else torch.float32
    old=Qwen3VLVisionModel(cfg).to(device=device,dtype=dtype).train()
    new=copy.deepcopy(old);new.config.reuse_vision_attention_layout=True
    if checkpointed:
        for model in [old,new]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    grids=([[1,4,4],[1,4,6],[1,4,4],[1,6,4]] if mixed else [[1,4,4]]*4)
    grid=torch.tensor(grids,device=device)
    a=torch.randn(int(grid.prod(-1).sum()),3*2*16*16,device=device,dtype=dtype,requires_grad=True)
    b=a.detach().clone().requires_grad_()
    x=old(a,grid,scene_image_counts=[2,2]);y=new(b,grid,scene_image_counts=[2,2])
    tol=.02 if device=='cuda' else 1e-5
    for left,right in zip([x[0],*x[1]],[y[0],*y[1]]):
        torch.testing.assert_close(left,right,rtol=tol,atol=tol)
    sum(v.float().square().mean() for v in [x[0],*x[1]]).backward()
    sum(v.float().square().mean() for v in [y[0],*y[1]]).backward()
    pairs=[(a.grad,b.grad)]
    for (name,p),(_,q) in zip(old.named_parameters(),new.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:pairs.append((p.grad,q.grad))
    lhs=torch.cat([p.float().flatten() for p,q in pairs]);rhs=torch.cat([q.float().flatten() for p,q in pairs])
    relative=(lhs-rhs).norm()/lhs.norm().clamp_min(1e-12)
    assert torch.isfinite(rhs).all() and relative<tol
    # A subsequent input with different lengths/view counts cannot reuse stale indices.
    grid2=torch.tensor([[1,4,6]]*3,device=device)
    pixels=torch.randn(72,3*2*16*16,device=device,dtype=dtype)
    old.eval();new.eval()
    with torch.inference_mode():
        expected=old(pixels,grid2,scene_image_counts=[1,2])
        actual=new(pixels,grid2,scene_image_counts=[1,2])
        for p,q in zip([expected[0],*expected[1]],[actual[0],*actual[1]]):
            torch.testing.assert_close(p,q,rtol=tol,atol=tol)
        changed=pixels.clone();changed[:24]+=3
        isolated=new(changed,grid2,scene_image_counts=[1,2])
        torch.testing.assert_close(actual[0][6:],isolated[0][6:],rtol=0,atol=0)
