"""Tests for language gates: native equivalence, sample isolation and checkpoint gradients."""
import copy
import unittest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from starVLA.model.modules.vlm.qwen3_vl_language_gate import install_language_gates, language_gate_context, packed_sample_ids
torch.set_num_threads(2)

class GateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        cfg=Qwen3VLVisionConfig(depth=4,hidden_size=64,intermediate_size=128,num_heads=4,
            out_hidden_size=64,deepstack_visual_indexes=[1,3],patch_size=16,
            spatial_merge_size=2,temporal_patch_size=2,num_position_embeddings=64)
        cfg._attn_implementation="sdpa"
        self.base=Qwen3VLVisionModel(cfg).eval()
        self.gated=install_language_gates(copy.deepcopy(self.base),[1,3])
        self.grid=torch.tensor([[1,4,4],[1,2,4],[1,4,4]])
        self.owner=torch.tensor([0,0,1])
        self.x=torch.randn(40,3*2*16*16)
        self.condition=torch.randn(2,64)

    def forward_gate(self, model=None, condition=None):
        with language_gate_context(self.condition if condition is None else condition,self.owner):
            return (self.gated if model is None else model)(self.x,self.grid)

    def test_gate_one_exact_native_equivalence(self):
        for i in self.gated.gated_layers:self.gated.blocks[i].attn.force_gate_one=True
        a,da=self.base(self.x,self.grid);b,db=self.forward_gate()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x,y in zip(da,db):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_initial_language_neutral_then_sample_specific(self):
        c=self.condition.clone();c[1]+=3
        a,_=self.forward_gate();b,_=self.forward_gate(condition=c)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for i in self.gated.gated_layers:
            torch.nn.init.normal_(self.gated.blocks[i].attn.film.weight,std=.1)
        a,_=self.forward_gate();b,_=self.forward_gate(condition=c)
        torch.testing.assert_close(a[:6],b[:6],rtol=0,atol=0)
        self.assertGreater((a[6:]-b[6:]).abs().max().item(),1e-7)

    def test_checkpoint_gradients_and_single_stats_count(self):
        eager=self.gated.train()
        checkpointed=copy.deepcopy(eager)
        checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        values=[]
        for model in (eager,checkpointed):
            model.gate_moments.enabled=True
            out,taps=self.forward_gate(model=model)
            loss=out.square().mean()+sum(t.square().mean() for t in taps)
            count=model.gate_moments.value[2].item()
            loss.backward()
            self.assertEqual(count,model.gate_moments.value[2].item())
            values.append((out,model.gate_moments.pop()))
        torch.testing.assert_close(values[0][0],values[1][0],atol=0,rtol=0)
        self.assertEqual(values[0][1],values[1][1])
        self.assertTrue(.94<values[0][1]["gate/mean"]<.97)
        for i in eager.gated_layers:
            for name in ("film","gate_proj"):
                a=getattr(eager.blocks[i].attn,name).weight.grad
                b=getattr(checkpointed.blocks[i].attn,name).weight.grad
                self.assertIsNotNone(a);self.assertGreater(a.abs().sum().item(),0)
                torch.testing.assert_close(a,b)

    def test_invalid_ownership_and_context(self):
        with self.assertRaises(ValueError):self.gated(self.x,self.grid)
        for ids in (torch.tensor([0,1]),torch.tensor([0,0,2])):
            with self.assertRaises(ValueError):packed_sample_ids(self.grid,ids,2,40)
        with self.assertRaises(ValueError):packed_sample_ids(self.grid,self.owner,2,39)

if __name__=="__main__":unittest.main()

