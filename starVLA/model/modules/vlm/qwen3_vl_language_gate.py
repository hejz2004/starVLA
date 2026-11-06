"""Native Qwen3-VL language-conditioned element-wise SDPA output gates.

Targets Transformers 4.57.6. Qwen3-VL packs images contiguously, with no
Qwen2.5-style window permutation. Conditioning is passed explicitly through
checkpointed block kwargs; no per-batch tensors are stored on modules.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.distributed as dist
from torch import nn
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
from transformers.models.qwen3_vl import modeling_qwen3_vl as upstream

_CONTEXT = ContextVar("qwen3_language_gate", default=None)


@contextmanager
def language_gate_context(condition, image_sample_ids):
    token = _CONTEXT.set((condition, image_sample_ids))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class FrozenLanguageConditioner(nn.Module):
    def __init__(self, model_id, vision_dim):
        super().__init__()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        self.clip = CLIPTextModelWithProjection.from_pretrained(model_id)
        self.clip.requires_grad_(False)
        self.clip.eval()
        self.projector = nn.Linear(self.clip.config.projection_dim, vision_dim)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def forward(self, instructions):
        tokens = self.tokenizer(
            list(instructions), padding=True, truncation=True,
            max_length=self.clip.config.max_position_embeddings, return_tensors="pt",
        ).to(self.projector.weight.device)
        with torch.no_grad():
            embeds = self.clip(**tokens).text_embeds
        return self.projector(embeds.to(self.projector.weight.dtype))


def packed_sample_ids(grid_thw, image_sample_ids, batch_size, num_tokens):
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("Expected image_grid_thw [images,3]")
    ids = image_sample_ids.to(device=grid_thw.device, dtype=torch.long)
    if ids.ndim != 1 or len(ids) != len(grid_thw):
        raise ValueError("Image ownership must match processor image_grid_thw")
    if len(ids) == 0 or bool(((ids < 0) | (ids >= batch_size)).any()):
        raise ValueError("Image ownership outside language batch")
    lengths = grid_thw.prod(-1)
    if bool((lengths <= 0).any()) or int(lengths.sum()) != num_tokens:
        raise ValueError("Packed patch count does not match image_grid_thw")
    return torch.repeat_interleave(ids, lengths, output_size=num_tokens)


class GateMoments:
    """Three detached scalars only; weighted across tokens, layers and ranks."""
    def __init__(self):
        self.enabled = False
        self.value = None

    @torch.no_grad()
    def add(self, gate):
        centered = gate.detach().float() - 0.95
        value = torch.stack((
            centered.sum(dtype=torch.float64),
            centered.square().sum(dtype=torch.float64),
            torch.tensor(centered.numel(), device=gate.device, dtype=torch.float64),
        ))
        self.value = value if self.value is None else self.value + value

    @torch.no_grad()
    def pop(self):
        if self.value is None:
            return {}
        value, self.value = self.value, None
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        mean = value[0] / value[2]
        std = (value[1] / value[2] - mean.square()).clamp_min(0).sqrt()
        return {"gate/mean": (mean + 0.95).item(), "gate/std": std.item()}


class ForwardGateStats:
    """Checkpoint recomputation must not count a block a second time."""
    def __init__(self, moments):
        self.moments = moments
        self.seen = set()

    def add(self, layer, gate):
        if layer not in self.seen:
            self.moments.add(gate)
            self.seen.add(layer)


class LanguageGatedVisionAttention(upstream.Qwen3VLVisionAttention):
    def init_gate(self, layer):
        self.layer_index = layer
        self.film = nn.Linear(self.dim, 2 * self.dim, device=self.qkv.weight.device)
        self.gate_proj = nn.Linear(self.dim, self.dim, device=self.qkv.weight.device)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.normal_(self.gate_proj.weight, std=1e-3)
        nn.init.constant_(self.gate_proj.bias, 3.0)
        self.force_gate_one = False  # equivalence tests only

    def elementwise_gate(self, hidden_states, condition, token_sample_ids):
        gamma, beta = self.film(condition.to(self.film.weight.dtype)).chunk(2, -1)
        gamma = gamma.index_select(0, token_sample_ids)
        beta = beta.index_select(0, token_sample_ids)
        u = (1 + gamma) * hidden_states + beta
        logits = self.gate_proj(u.to(self.gate_proj.weight.dtype))
        return logits.float().sigmoid().reshape(-1, self.num_heads, self.head_dim)

    def forward(
        self, hidden_states, cu_seqlens, rotary_pos_emb=None,
        position_embeddings=None, gate_condition=None, gate_sample_ids=None,
        gate_stats=None, **kwargs,
    ):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = upstream.apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_interface = upstream.eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = upstream.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self, query_states, key_states, value_states, attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen, max_length_k=max_seqlen,
                is_causal=False, **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self, q, k, v, attention_mask=None, scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False, **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
        if gate_condition is None or gate_sample_ids is None:
            raise ValueError("Missing per-forward gate conditioning")
        gate = self.elementwise_gate(hidden_states, gate_condition, gate_sample_ids)
        if gate_stats is not None:
            gate_stats.add(self.layer_index, gate)
        if not self.force_gate_one:
            attn_output = attn_output.reshape(seq_length, self.num_heads, self.head_dim)
            attn_output = attn_output * gate.to(attn_output.dtype)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class LanguageGatedVisionModel(upstream.Qwen3VLVisionModel):
    def forward(self, hidden_states, grid_thw, **kwargs):
        context = _CONTEXT.get()
        if context is None:
            raise ValueError("Vision language conditioning context was not provided")
        condition, image_sample_ids = context
        ids = packed_sample_ids(grid_thw, image_sample_ids, len(condition), len(hidden_states))
        stats = ForwardGateStats(self.gate_moments) if self.gate_moments.enabled else None
        return super().forward(
            hidden_states, grid_thw, gate_condition=condition,
            gate_sample_ids=ids, gate_stats=stats, **kwargs,
        )


def install_language_gates(vision, layers):
    if type(vision) is not upstream.Qwen3VLVisionModel:
        raise TypeError("Language gating requires the original Qwen3-VL vision encoder")
    layers = list(layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Gate layer list must be nonempty and unique")
    if any(not isinstance(i, int) or i < 0 or i >= len(vision.blocks) for i in layers):
        raise ValueError("Gate layer outside actual vision depth")
    vision.__class__ = LanguageGatedVisionModel
    vision.gated_layers = tuple(layers)
    vision.gate_moments = GateMoments()
    for layer in layers:
        attention = vision.blocks[layer].attn
        attention.__class__ = LanguageGatedVisionAttention
        attention.init_gate(layer)
    return vision
