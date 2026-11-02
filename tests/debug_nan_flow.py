"""Bisect where NaN first appears in the dual-stream flow (normal data only)."""

import numpy as np
import torch
from PIL import Image

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_image(size=224, seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (size, size, 3), dtype=np.uint8))


from omegaconf import OmegaConf

YAML = "examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero.yaml"
cfg = OmegaConf.load(YAML)
model_id = str(cfg.framework.qwenvl.base_vlm)

sample = {
    "action": np.random.uniform(-1, 1, size=(8, 7)).astype(np.float32),
    "image": [make_image(224, 0), make_image(224, 1), make_image(224, 2)],
    "lang": "pick up the red block and place it in the bowl",
    "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32),
}
sample2 = dict(sample)
sample2["image"] = [make_image(224, 3), make_image(224, 4)]
sample2["lang"] = "open the drawer"
batch_images = [sample["image"], sample2["image"]]
instructions = [sample["lang"], sample2["lang"]]

print("=" * 70)
print("[A] official transformers plain forward (no dual stream)")
from transformers import Qwen3VLForConditionalGeneration as OfficialQwen3VL
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"
messages = []
for imgs, inst in zip(batch_images, instructions):
    content = [{"type": "image", "image": img} for img in imgs]
    content.append({"type": "text", "text": inst})
    messages.append([{"role": "user", "content": content}])
plain_inputs = processor.apply_chat_template(
    messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(device)
official = OfficialQwen3VL.from_pretrained(model_id, attn_implementation="sdpa", dtype=torch.bfloat16).to(device)
official.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    plain_out = official(**plain_inputs, output_hidden_states=True, return_dict=True)
hs = plain_out.hidden_states
bad = [i for i, h in enumerate(hs) if not torch.isfinite(h).all()]
print(f"  hidden_states layers: {len(hs)}, non-finite layers: {bad}")
print(f"  logits finite: {torch.isfinite(plain_out.logits).all().item()}, max|logits| = {plain_out.logits.abs().max().item():.3f}")
del official
torch.cuda.empty_cache()

print("=" * 70)
print("[B] dual-stream vision tower alone")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from starVLA.model.modules.vlm.QWen3DualStream import _QWen3DualStream_VL_Interface

iface = _QWen3DualStream_VL_Interface(cfg)
iface = iface.to(device)
iface.eval()

qin = iface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
print(f"  pixel shape = {tuple(qin['pixel_values'].shape)}, grid = {qin['image_grid_thw'].tolist()}")

vis = iface.model.model.visual
gamma_stats = {}


def make_hook(name):
    def hook(module, args, output):
        shared_new, fused = output
        gamma_stats[name] = (fused.abs().max().item(), fused.abs().mean().item())

    return hook


handles = [st.register_forward_hook(make_hook(n)) for n, st in vis.interaction_stages.items()]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    merged, ds = vis(qin["pixel_values"], grid_thw=qin["image_grid_thw"])
for h in handles:
    h.remove()
print(f"  merged finite: {torch.isfinite(merged).all().item()}, max|merged| = {merged.abs().max().item():.3f}")
for i, f in enumerate(ds):
    print(f"  deepstack[{i}] finite: {torch.isfinite(f).all().item()}, max|f| = {f.abs().max().item():.3f}")
for n, (mx, mean) in gamma_stats.items():
    print(f"  stage {n}: fused max={mx:.3f} mean={mean:.6f}")

print("=" * 70)
print("[C] dual-stream full LLM hidden states")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    dual_out = iface(**qin, output_attentions=False, output_hidden_states=True, return_dict=True)
hs = dual_out.hidden_states
bad = [i for i, h in enumerate(hs) if not torch.isfinite(h).all()]
print(f"  hidden_states layers: {len(hs)}, non-finite layers: {bad}")
if bad:
    i0 = bad[0]
    h = hs[i0]
    nb = (1 - torch.isfinite(h)).sum().item()
    print(f"  first bad layer {i0}: {nb}/{h.numel()} non-finite, max|finite| = {h[torch.isfinite(h)].abs().max().item():.3f}")
    print(f"  logits finite: {torch.isfinite(dual_out.logits).all().item()}")
else:
    print(f"  logits finite: {torch.isfinite(dual_out.logits).all().item()}, max|logits| = {dual_out.logits.abs().max().item():.3f}")

print("=" * 70)
print("DONE")
