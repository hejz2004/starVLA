"""Numerical and cursor checks for the opt-in performance path.

The CPU bypass tests need no weights or external runtime.
"""
import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from starVLA.model.modules.vlm.qwen3_vl_dual.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def tiny_model(backend="sdpa", device="cpu", dtype=torch.float32, production_geometry=False):
    text_width = 256 if production_geometry else 128
    cfg = Qwen3VLConfig(
        vision_config=dict(depth=2, hidden_size=128, num_heads=2, intermediate_size=256,
                           out_hidden_size=text_width, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[0], dual_stream=dict(enabled=True,
                           interaction_indexes=[0, 1], num_shared_tokens=32 if production_geometry else 4, num_heads=2,
                           intermediate_size=256, gate_hidden_size=16, gate_init_value=.1)),
        text_config=dict(vocab_size=128, hidden_size=text_width, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=text_width//2, rope_scaling=dict(rope_type="default", mrope_section=[16, 24, 24] if production_geometry else [8, 12, 12]),
                         pad_token_id=0), image_token_id=41, video_token_id=43,
        vision_start_token_id=40, vision_end_token_id=42,
    )
    cfg._attn_implementation = backend
    return Qwen3VLForConditionalGeneration(cfg).to(device=device, dtype=dtype)


def multimodal_inputs(device="cpu", dtype=torch.float32):
    # Unequal text lengths, left padding, two views in one sample and one in the other.
    a = [1, 40, 41, 41, 41, 41, 42, 40, 41, 41, 41, 41, 42, 3]
    b = [0] * 5 + [1, 40, 41, 41, 41, 41, 42, 3, 4]
    ids = torch.tensor([a, b], device=device)
    return dict(input_ids=ids, attention_mask=ids.ne(0).long(),
                pixel_values=torch.randn(48, 3*2*16*16, device=device, dtype=dtype),
                image_grid_thw=torch.tensor([[1, 4, 4]]*3, device=device), use_cache=False,
                output_hidden_states=True, return_dict=True)


@pytest.mark.parametrize("training", [False, True])
def test_lm_head_bypass_exact_hidden_and_gradients(training):
    torch.manual_seed(21)
    base = tiny_model().train(training)
    fast = copy.deepcopy(base)
    inputs = multimodal_inputs()
    expected = base(**inputs)
    with patch.object(fast.lm_head, "forward", side_effect=AssertionError("unused lm_head executed")):
        actual = fast(**inputs, skip_lm_head=True)
    assert actual.logits is None and expected.logits is not None
    for a, b in zip(expected.hidden_states, actual.hidden_states):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    expected.hidden_states[-1].square().mean().backward()
    actual.hidden_states[-1].square().mean().backward()
    for (name, a), (_, b) in zip(base.named_parameters(), fast.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    with pytest.raises(ValueError, match="labels"):
        fast(**inputs, skip_lm_head=True, labels=inputs["input_ids"])




def check_relative(actual, expected, label, tolerance=.03):
    assert torch.isfinite(actual).all(), label
    a, b = actual.float(), expected.float()
    relative = (a-b).norm() / b.norm().clamp_min(1e-8)
    print(f"PARITY {label}: relative_l2={relative.item():.6g} max_abs={(a-b).abs().max().item():.6g}")
    assert relative < tolerance, (label, relative.item())








@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_prefetch_preserves_values_order_and_consumption_on_resume():
    from starVLA.training.trainer_utils.cuda_prefetch import CudaPrefetchIterator
    from torch.utils.data import DataLoader, TensorDataset
    from accelerate import Accelerator
    loader=DataLoader(TensorDataset(torch.arange(35).reshape(7,5)),batch_size=2,pin_memory=True)
    prepared=Accelerator().prepare(loader)
    prepared.device=None
    original=list(loader)
    iterator=CudaPrefetchIterator(iter(prepared),"cuda")
    seen=[]
    for expected in original:
        actual=next(iterator)
        torch.testing.assert_close(actual[0].cpu(),expected[0])
        seen.append(actual)
    with pytest.raises(StopIteration):next(iterator)
    # Strict resume skips only consumed batches, not the lookahead batch.
    resumed=CudaPrefetchIterator(iter(prepared),"cuda")
    for _ in range(2):next(resumed)
    torch.testing.assert_close(next(resumed)[0].cpu(),original[2][0])


class ProcessorExamples(torch.utils.data.Dataset):
    def __len__(self):return 5
    def __getitem__(self,i):
        from PIL import Image
        sample=dict(image=[Image.fromarray(np.full((224,224,3),i*20+j,dtype=np.uint8)) for j in range(2)],
                    lang=["pick the red block","put the long object in the drawer"][i%2],
                    action=np.full((16,16),i/5,dtype=np.float32),
                    state=np.full((1,16),i/7,dtype=np.float32))
        if i%2==0:
            sample['action_mask']=np.tile(np.arange(16)<8,(16,1))
        return sample


@pytest.mark.skipif(not os.path.isdir('/hdfs/share-data-1/users/hjz/Workspace/pretrained/Qwen3-VL-2B-Instruct'),
                    reason='local processor assets required')
@pytest.mark.parametrize('inherited_rank_environment', [False, True])
@pytest.mark.parametrize('framework_name', ['DualStreamPI', 'QwenPI', 'QwenFM'])
def test_real_processor_spawn_workers_and_h2d_equal_original(monkeypatch, inherited_rank_environment, framework_name):
    from transformers import AutoProcessor
    from starVLA.dataloader.qwen_preprocess import configure_worker_preprocessing
    from starVLA.model.modules.vlm.QWen3DualStream import _QWen3DualStream_VL_Interface
    from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
    from starVLA.training.trainer_utils.cuda_prefetch import CudaPrefetchIterator
    path='/hdfs/share-data-1/users/hjz/Workspace/pretrained/Qwen3-VL-2B-Instruct'
    processor=AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side='left'
    interface=SimpleNamespace(processor=processor,config=OmegaConf.create({'datasets':{'vla_data':{'CoT_prompt':'Task: {instruction}'}}}),
                              model=SimpleNamespace(device=torch.device('cpu')))
    if inherited_rank_environment:
        # torchrun data workers inherit these, but must never join NCCL merely
        # because their processor/model module imports the logging helper.
        for key,value in {'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0',
                          'MASTER_ADDR':'127.0.0.1','MASTER_PORT':'9'}.items():
            monkeypatch.setenv(key,value)
    ds=ProcessorExamples()
    cfg=OmegaConf.create({'framework':{'name':framework_name,'qwenvl':{'base_vlm':path}},
                         'datasets':{'vla_data':{'worker_preprocessing':True,'CoT_prompt':'Task: {instruction}'}}})
    loader=configure_worker_preprocessing(torch.utils.data.DataLoader(ds,batch_size=2,num_workers=2),cfg)
    assert loader.pin_memory and loader.multiprocessing_context.get_start_method()=='spawn'
    iterator=CudaPrefetchIterator(loader,'cuda') if torch.cuda.is_available() else iter(loader)
    for start,batch in zip(range(0,len(ds),2),iterator):
        raw=[ds[i] for i in range(start,min(start+2,len(ds)))]
        interface_class = _QWen3DualStream_VL_Interface if framework_name == 'DualStreamPI' else _QWen3_VL_Interface
        expected=interface_class.build_qwenvl_inputs(interface,
            [x['image'] for x in raw],[x['lang'] for x in raw])
        assert batch['vlm_inputs'].keys()==expected.keys()
        for key,value in expected.items():torch.testing.assert_close(batch['vlm_inputs'][key].cpu(),value,rtol=0,atol=0)
        torch.testing.assert_close(batch['actions'].cpu(),torch.from_numpy(np.stack([x['action'] for x in raw])),rtol=0,atol=0)
        torch.testing.assert_close(batch['state'].cpu(),torch.from_numpy(np.stack([x['state'] for x in raw])),rtol=0,atol=0)
        masks=np.stack([x.get('action_mask',np.ones_like(x['action'],dtype=bool)) for x in raw])
        torch.testing.assert_close(batch['action_mask'].cpu(),torch.from_numpy(masks),rtol=0,atol=0)


def test_preprocessed_framework_loss_masks_repeat_and_gradients():
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
    from starVLA.model.framework.base_framework import baseframework
    from tests.test_umi_diffusion_repeat import CaptureHead
    from contextlib import nullcontext
    model=Qwen_PI.__new__(Qwen_PI);baseframework.__init__(model)
    model.config=OmegaConf.create({'framework':{'action_model':{'repeated_diffusion_steps':2}}})
    model.action_horizon=16;model.action_model=CaptureHead()
    features=[torch.randn(2,5,8)];mask=torch.tensor([[1,1,1,1,1],[0,0,1,1,1]])
    model._encode_vl_hidden_states=lambda *args,**kwargs:(features,mask)
    raw=[ProcessorExamples()[i] for i in range(2)]
    packed=dict(vlm_inputs={},actions=torch.from_numpy(np.stack([x['action'] for x in raw])),
                state=torch.from_numpy(np.stack([x['state'] for x in raw])),
                action_mask=torch.from_numpy(np.stack([x.get('action_mask',np.ones_like(x['action'],dtype=bool)) for x in raw])))
    with patch('torch.autocast',side_effect=lambda *a,**kw:nullcontext()):
        a=model(raw);old=model.action_model.seen
        b=model(packed);new=model.action_model.seen
    torch.testing.assert_close(a['action_loss'],b['action_loss'],rtol=0,atol=0)
    for x,y in zip(old[0],new[0]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    for i in [1,2]:torch.testing.assert_close(old[i],new[i],rtol=0,atol=0)
    for k in old[3]:torch.testing.assert_close(old[3][k],new[3][k],rtol=0,atol=0)
    ga=torch.autograd.grad(a['action_loss'],model.action_model.scale)
    gb=torch.autograd.grad(b['action_loss'],model.action_model.scale)
    torch.testing.assert_close(ga[0],gb[0],rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("production_geometry", [False, True])
def test_full_policy_actions_loss_and_gradients_with_worker_batch(production_geometry):
    from starVLA.model.framework.VLM4A.DualStreamQwenPI import DualStreamQwen_PI
    from starVLA.model.modules.vlm.QWen3DualStream import _QWen3DualStream_VL_Interface
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
    torch.manual_seed(22)
    inputs=multimodal_inputs('cuda',torch.bfloat16)
    # QwenPI supplies these flags itself.
    inputs={k:v for k,v in inputs.items() if isinstance(v,torch.Tensor)}
    def interface(cfg,backend):
        result=_QWen3DualStream_VL_Interface.__new__(_QWen3DualStream_VL_Interface)
        torch.nn.Module.__init__(result)
        result.model=tiny_model(backend,'cuda',torch.bfloat16,production_geometry)
        result.model.config.hidden_size=result.model.config.text_config.hidden_size
        result.config=cfg
        result.build_qwenvl_inputs=lambda **kwargs:inputs
        return result
    def policy(backend,bypass):
        cfg=OmegaConf.create({'framework':{'name':'DualStreamPI','qwenvl':{'skip_unused_lm_head':bypass},
            'action_model':{'action_dim':16,'state_dim':16,'use_state':False,'action_horizon':16,
                            'repeated_diffusion_steps':1,'decoder_input_norm':'layer_norm',
                            'diffusion_model_cfg':{'dropout':0.,'final_dropout':False,'cross_condition_norm':'layer_norm'}}},
            'datasets':{'vla_data':{}},'experiment':{'action_head_seed':43}})
        with patch('starVLA.model.framework.VLM4A.DualStreamQwenPI._QWen3DualStream_VL_Interface',
                   side_effect=lambda config:interface(config,backend)):
            return DualStreamQwen_PI(cfg).cuda().bfloat16()
    baseline=policy('sdpa',False)
    candidate=policy('sdpa',True)
    candidate.qwen_vl_interface.model.config.vision_config.batch_vision_attention=True
    candidate.load_state_dict(baseline.state_dict())
    raw=[{'image':[],'lang':'test','action':np.random.default_rng(i).normal(size=(16,16)).astype(np.float32),
          'action_mask':np.tile(np.arange(16)<8,(16,1))} for i in range(2)]
    batch={'vlm_inputs':inputs,'actions':torch.from_numpy(np.stack([x['action'] for x in raw])).cuda(),
           'action_mask':torch.from_numpy(np.stack([x['action_mask'] for x in raw])).cuda()}
    baseline.train();candidate.train()
    torch.manual_seed(101);a=baseline(raw)['action_loss']
    torch.manual_seed(101);b=candidate(batch)['action_loss']
    torch.testing.assert_close(a,b,rtol=.01,atol=.001)
    a.backward();b.backward()
    ga=[];gb=[]
    for (name,x),(_,y) in zip(baseline.named_parameters(),candidate.named_parameters()):
        assert (x.grad is None)==(y.grad is None),name
        if x.grad is not None:
            assert torch.isfinite(y.grad).all(),name
            ga.append(x.grad.float().flatten());gb.append(y.grad.float().flatten())
    check_relative(torch.cat(gb),torch.cat(ga),'whole policy gradients')
    baseline.eval();candidate.eval()
    torch.manual_seed(202);expected=baseline.predict_action(raw)['normalized_actions']
    torch.manual_seed(202);actual=candidate.predict_action(batch)['normalized_actions']
    check_relative(torch.from_numpy(actual),torch.from_numpy(expected),'4-step predicted actions',tolerance=.01)
    print('POLICY_LOSS',float(a.detach()),float(b.detach()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_trainer_prefetch_cursor_epoch_and_strict_resume():
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, TensorDataset
    from starVLA.training.train_starvla import VLATrainer
    accelerator = Accelerator()
    loader = accelerator.prepare(DataLoader(TensorDataset(torch.arange(6)), batch_size=2, pin_memory=True))
    loader.device = None

    def trainer(skip=0, consumed=0):
        result = VLATrainer.__new__(VLATrainer)
        result.accelerator = accelerator
        result.vla_train_dataloader = loader
        result.umi_stage_data = None
        result._cuda_prefetch = True
        result._skip_batches = skip
        result.batches_consumed = consumed
        result._create_data_iterators()
        return result

    original = trainer()
    assert original.batches_consumed == 0
    torch.testing.assert_close(original._get_next_batch()[0].cpu(), torch.tensor([0, 1]))
    assert original.batches_consumed == 1  # lookahead is not a consumed batch
    resumed = trainer(skip=1, consumed=1)
    assert resumed.batches_consumed == 1
    for expected in ([2, 3], [4, 5], [0, 1]):
        torch.testing.assert_close(resumed._get_next_batch()[0].cpu(), torch.tensor(expected))
    assert resumed.batches_consumed == 4 and resumed.vla_epoch_count == 1




@pytest.mark.parametrize('mixed_sizes', [False, True])
def test_batched_sdpa_vision_preserves_outputs_and_gradients(mixed_sizes):
    torch.manual_seed(67)
    base=tiny_model(production_geometry=True).model.visual
    fast=copy.deepcopy(base)
    fast.config.batch_vision_attention=True
    grids=torch.tensor([[1,4,4],[1,4,6] if mixed_sizes else [1,4,4],[1,4,4]])
    pixels=torch.randn(int(grids.prod(-1).sum()),3*2*16*16)
    expected=base(pixels,grids,scene_image_counts=[2,1])
    actual=fast(pixels,grids,scene_image_counts=[2,1])
    for a,b in [(actual[0],expected[0]),*zip(actual[1],expected[1])]:
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
    expected[0].square().mean().backward();actual[0].square().mean().backward()
    for (name,a),(_,b) in zip(base.named_parameters(),fast.named_parameters()):
        assert (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=2e-6)
