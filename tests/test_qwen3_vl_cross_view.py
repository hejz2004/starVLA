"""Late alternating QwenPI attention: topology, isolation, gradients and HF integration.
Run: PYTHONPATH=. python tests/test_qwen3_vl_cross_view.py -v
"""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel as OriginalVision
from starVLA.model.modules.vlm.qwen3_vl_cross_view import Qwen3VLVisionModel, Qwen3VLForConditionalGeneration
from starVLA.model.modules.vlm.qwen3_vl_cross_view.attention_layout import (
    global_attention_indexes, scene_cu_seqlens, image_counts_from_input_ids,
)

torch.set_num_threads(2)


def vision_config(enabled=True):
    cfg = Qwen3VLVisionConfig(depth=6, hidden_size=64, intermediate_size=128,
        num_heads=4, out_hidden_size=64, deepstack_visual_indexes=[1, 3, 5],
        patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)
    cfg.cross_view_attention = dict(enabled=enabled, start_layer=4)
    cfg._attn_implementation = 'sdpa'
    return cfg


def pixels(grids):
    return torch.randn(sum(t*h*w for t,h,w in grids), 3*2*16*16)


class VisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = Qwen3VLVisionModel(vision_config()).eval()

    def test_schedule_and_no_extra_parameters(self):
        self.assertEqual(self.model.global_attention_indexes, (4,))
        cfg=vision_config();cfg.depth=24;cfg.cross_view_attention['start_layer']=16
        self.assertEqual(global_attention_indexes(cfg), (16,18,20,22))
        cfg.cross_view_attention['start_layer']=None
        self.assertEqual(global_attention_indexes(cfg), (16,18,20,22))
        base=OriginalVision(vision_config(False))
        self.assertEqual(set(base.state_dict()),set(self.model.state_dict()))
        self.model.load_state_dict(base.state_dict(),strict=True)
        self.assertFalse(any('shared_init' in k or 'interaction_stages' in k or 'gate_mlp' in k for k in self.model.state_dict()))

    def test_disabled_matches_upstream_outputs_and_deepstack(self):
        cfg=vision_config(False);base=OriginalVision(cfg).eval();model=Qwen3VLVisionModel(cfg).eval()
        model.load_state_dict(base.state_dict(),strict=True)
        grids=[(1,4,4),(1,2,4)];x=pixels(grids);grid=torch.tensor(grids)
        a,da=base(x,grid);b,db=model(x,grid)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x,y in zip(da,db):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_cross_view_effect_reaches_next_local_block_and_final_output(self):
        g=torch.tensor([[1,4,4]]*2);x=pixels(g.tolist());y=x.clone();y[16:]+=2
        a,da=self.model(x,g,scene_image_counts=[2]);b,db=self.model(y,g,scene_image_counts=[2])
        # First view is unchanged before the first global block.
        for first,second in zip(da[:2],db[:2]):torch.testing.assert_close(first[:4],second[:4],rtol=0,atol=0)
        # Last tap is after a local block that consumes the globally fused stream.
        self.assertGreater((da[2][:4]-db[2][:4]).abs().max().item(),1e-6)
        self.assertGreater((a[:4]-b[:4]).abs().max().item(),1e-6)
        x.requires_grad_();out,_=self.model(x,g,scene_image_counts=[2]);out[:4].square().sum().backward()
        self.assertGreater(x.grad[16:].abs().sum().item(),0)

    def test_sample_isolation_with_mixed_views_and_resolutions(self):
        grids=[(1,4,4),(1,2,4),(1,4,2),(1,4,4),(1,2,2)]
        g=torch.tensor(grids);x=pixels(grids);y=x.clone();y[24:]+=3
        a,da=self.model(x,g,scene_image_counts=[2,0,3]);b,db=self.model(y,g,scene_image_counts=[2,0,3])
        torch.testing.assert_close(a[:6],b[:6],rtol=0,atol=0)
        solo,ds=self.model(x[:24],g[:2],scene_image_counts=[2])
        torch.testing.assert_close(a[:6],solo,atol=1e-6,rtol=1e-5)
        for batched,single in zip(da,ds):torch.testing.assert_close(batched[:6],single,atol=1e-6,rtol=1e-5)
        x.requires_grad_();out,_=self.model(x,g,scene_image_counts=[2,0,3]);out[:6].sum().backward()
        self.assertEqual(x.grad[24:].abs().sum().item(),0)

    def test_attention_boundaries_and_positions(self):
        seen=[];handles=[]
        def hook(module,args,kwargs):
            seen.append((kwargs['cu_seqlens'].tolist(),kwargs['position_embeddings'][1].detach()))
        for blk in self.model.blocks:handles.append(blk.register_forward_pre_hook(hook,with_kwargs=True))
        g=torch.tensor([[1,4,4]]*4);self.model(pixels(g.tolist()),g,scene_image_counts=[2,2])
        for h in handles:h.remove()
        for i,(bounds,sin) in enumerate(seen):
            self.assertEqual(bounds,[0,32,64] if i==4 else [0,16,32,48,64])
            self.assertEqual(sin.abs().sum().item()==0,i==4)

    def test_view_permutation(self):
        g=torch.tensor([[1,4,4]]*2);x=pixels(g.tolist());a,_=self.model(x,g,scene_image_counts=[2])
        b,_=self.model(torch.cat([x[16:],x[:16]]),g,scene_image_counts=[2])
        torch.testing.assert_close(a,torch.cat([b[4:],b[:4]]),rtol=1e-5,atol=1e-6)

    def test_single_image_and_validation(self):
        g=torch.tensor([[1,4,4]]);out,_=self.model(pixels(g.tolist()),g,scene_image_counts=[1])
        self.assertEqual(out.shape,(4,64));self.assertTrue(torch.isfinite(out).all())
        for counts in (None,[2],[-1,2],[1.0],[True]):
            with self.assertRaises(ValueError):scene_cu_seqlens(g,counts)
        with self.assertRaises(ValueError):scene_cu_seqlens(torch.tensor([[2,4,4]]),[1])
        for start in (-1,6,1.5,True):
            cfg=vision_config();cfg.cross_view_attention['start_layer']=start
            with self.assertRaises(ValueError):Qwen3VLVisionModel(cfg)
        cfg=vision_config();cfg.cross_view_attention['typo']=1
        with self.assertRaises(ValueError):Qwen3VLVisionModel(cfg)

    def test_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            other=Qwen3VLVisionModel.from_pretrained(tmp,attn_implementation='sdpa').eval()
            self.assertEqual(other.global_attention_indexes,(4,))
            g=torch.tensor([[1,4,4]]*2);x=pixels(g.tolist())
            a,_=self.model(x,g,scene_image_counts=[2]);b,_=other(x,g,scene_image_counts=[2])
            torch.testing.assert_close(a,b)


def tiny_model():
    vc=vision_config();vc.deepstack_visual_indexes=[1,3]
    cfg=Qwen3VLConfig(vision_config=vc.to_dict(),text_config=dict(vocab_size=64,
        hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=2,head_dim=16,max_position_embeddings=128,
        rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}),
        image_token_id=3,video_token_id=6,vision_start_token_id=2,vision_end_token_id=4,
        pad_token_id=0,eos_token_id=5)
    cfg._attn_implementation='sdpa'
    return Qwen3VLForConditionalGeneration(cfg)


def tiny_inputs(device='cpu'):
    tokens=[10,2,3,3,3,3,4,2,3,3,3,3,4,11]
    return dict(input_ids=torch.tensor([tokens,tokens],device=device),
        attention_mask=torch.ones(2,len(tokens),dtype=torch.long,device=device),
        pixel_values=pixels([(1,4,4)]*4).to(device),image_grid_thw=torch.tensor([[1,4,4]]*4,device=device))


class IntegrationTests(unittest.TestCase):
    def test_full_vlm_batch_isolation_and_cached_generation(self):
        torch.manual_seed(20);m=tiny_model().eval();a=tiny_inputs();b={k:v.clone() for k,v in a.items()};b['pixel_values'][32:]+=2
        with torch.no_grad():
            x=m(**a).logits;y=m(**b).logits
            torch.testing.assert_close(x[0],y[0],rtol=0,atol=0)
            generated=m.generate(**a,max_new_tokens=2,do_sample=False)
        self.assertEqual(generated.shape[0],2)
        self.assertGreater(generated.shape[1],a['input_ids'].shape[1])
        self.assertEqual(image_counts_from_input_ids(a['input_ids'],2,3,4),[2,2])
        with self.assertRaises(ValueError):image_counts_from_input_ids(a['input_ids'],2,3,3)
        with self.assertRaises(ValueError):m.model.get_video_features(a['pixel_values'],a['image_grid_thw'])

    def test_checkpointed_backward_matches_plain(self):
        torch.manual_seed(21);m=tiny_model().train();other=copy.deepcopy(m)
        other.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        a=tiny_inputs()
        losses=[]
        for model in [m,other]:
            loss=model(**a,labels=a['input_ids'],use_cache=False).loss;loss.backward();losses.append(loss)
            self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(*losses)
        for (name,p),(_,q) in zip(m.named_parameters(),other.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4,msg=name)

    @unittest.skipUnless(torch.cuda.is_available(),'requires accelerator')
    def test_accelerator_bf16_backward(self):
        m=tiny_model().to(device='cuda',dtype=torch.bfloat16).train()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        a=tiny_inputs('cuda');loss=m(**a,labels=a['input_ids'],use_cache=False).loss;loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for i in m.model.visual.global_attention_indexes:
            grad=m.model.visual.blocks[i].attn.qkv.weight.grad
            self.assertIsNotNone(grad);self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)


class ConfigAndFrameworkTests(unittest.TestCase):
    def test_yaml_architecture_and_checkpoint_defaults(self):
        from omegaconf import OmegaConf
        files=list(Path('examples/umi/train_files').glob('starvla_qwenpi_da3*.yaml'))
        files+=list(Path('examples/simBenchmarks/LIBERO/train_files').glob('starvla_qwenpi_da3*.yaml'))
        self.assertEqual(len(files),5)
        for p in files:
            c=OmegaConf.load(p)
            self.assertEqual(c.framework.name,'QwenPI')
            self.assertNotIn('dual_stream',c.framework.qwenvl)
            self.assertTrue(c.framework.qwenvl.cross_view_attention.enabled)
            self.assertEqual(c.framework.qwenvl.cross_view_attention.start_layer,16)
            self.assertIsNone(c.trainer.get('pretrained_checkpoint'))
        self.assertFalse(Path('starVLA/model/framework/VLM4A/DualStreamQwenPI.py').exists())

    def test_framework_rejects_legacy_weights_even_with_nonstrict_loading(self):
        from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
        from starVLA.model.framework.base_framework import baseframework
        model=Qwen_PI.__new__(Qwen_PI);baseframework.__init__(model)
        for key in ('qwen_vl_interface.model.model.visual.shared_init',
                    'qwen_vl_interface.model.model.visual.interaction_stages.5.q1.weight'):
            with self.assertRaisesRegex(ValueError,'Legacy dualstream'):
                model.load_state_dict({key:torch.zeros(1)},strict=False)


if __name__=='__main__':unittest.main()
