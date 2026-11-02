"""Smoke test: normal data flow of the dual-stream stack (no perturbation checks).

Run in container:  PYTHONPATH=. python tests/smoke_dual_stream_flow.py

Covers:
  1. dual-stream vision tower normal forward (small random config, CPU)
  2. registry: original QwenPI intact + dual-stream frameworks registered
  3. DualStreamQwenPI end-to-end (2B weights): forward / backward / predict_action
  4. optimizer LR groups contain interaction_stages
  5. original QwenPI still builds and forwards through the plain interface
"""

import numpy as np
import torch
from PIL import Image

torch.manual_seed(0)


def make_image(size=224, seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (size, size, 3), dtype=np.uint8))


print("=" * 70)
print("[1/5] dual-stream vision tower normal forward (small config, CPU)")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from starVLA.model.modules.vlm.qwen3_vl_dual.modeling_qwen3_vl import Qwen3VLVisionModel

cfg = Qwen3VLVisionConfig(
    depth=6,
    hidden_size=64,
    num_heads=4,
    intermediate_size=128,
    out_hidden_size=128,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    in_channels=3,
    hidden_act="gelu_pytorch_tanh",
    initializer_range=0.02,
    num_position_embeddings=1024,
    deepstack_visual_indexes=[1, 3],
)
cfg.dual_stream = {
    "enabled": True,
    "interaction_indexes": [1, 3, 5],
    "num_shared_tokens": 8,
    "num_heads": 4,
    "intermediate_size": 128,
    "gate_hidden_size": 16,
    "gate_init_value": 0.01,
    "gate_output_weight_std": 0.001,
    "attention_dropout": 0.0,
}
tower = Qwen3VLVisionModel(cfg)
tower.eval()

PATCH = 16
MERGE = 2
grids = [(1, 4, 4), (1, 4, 4), (1, 2, 4)]  # one scene, 3 views, mixed resolutions
pixel_values = torch.cat([torch.randn(t * h * w, 3 * 2 * PATCH * PATCH) for (t, h, w) in grids])
grid_thw = torch.tensor(grids, dtype=torch.long)
with torch.no_grad():
    merged, deepstack = tower(pixel_values, grid_thw=grid_thw)
n_merged = sum((h // MERGE) * (w // MERGE) for _, h, w in grids)
assert merged.shape == (n_merged, 128), merged.shape
assert len(deepstack) == 2, len(deepstack)
for f in deepstack:
    assert f.shape == (n_merged, 128), f.shape
assert torch.isfinite(merged).all()
assert all(torch.isfinite(f).all() for f in deepstack)
n_new = sum(p.numel() for p in tower.interaction_stages.parameters()) + tower.shared_init.numel()
print(f"  merged {tuple(merged.shape)}, deepstack x{len(deepstack)}, new params {n_new} — OK")

print("[2/5] registry: original QwenPI intact + dual-stream registered")
from starVLA.model.framework.base_framework import _auto_import_framework_modules
from starVLA.model.tools import FRAMEWORK_REGISTRY

_auto_import_framework_modules()
reg = FRAMEWORK_REGISTRY.list()
for name in ["QwenPI", "QwenFM", "DualStreamQwenPI", "DualStreamQwenFM"]:
    assert name in reg, f"{name} missing from registry: {sorted(reg)}"
print(f"  registry OK: {sorted(reg)}")

print("[3/5] DualStreamQwenPI end-to-end (2B weights, fake batch)")
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.DualStreamQwenPI import DualStreamQwen_PI

yaml_path = "examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero.yaml"
cfg = OmegaConf.load(yaml_path)
model = DualStreamQwen_PI(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"  model built on {device}; visual.interaction_stages = {list(model.qwen_vl_interface.model.visual.interaction_stages.keys())}")

img = make_image(224, 0)
sample = {
    "action": np.random.uniform(-1, 1, size=(8, 7)).astype(np.float32),
    "image": [img, img, img],
    "lang": "pick up the red block and place it in the bowl",
    "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32),
}
sample2 = dict(sample)
sample2["image"] = [make_image(224, 1), make_image(224, 2)]
sample2["lang"] = "open the drawer"

out = model([sample, sample2])
action_loss = out["action_loss"]
assert torch.isfinite(action_loss), action_loss
print(f"  forward action_loss = {action_loss.item():.6f}")

action_loss.backward()
vis = model.qwen_vl_interface.model.visual
no_grad_names = [n for n, p in vis.interaction_stages.named_parameters() if p.grad is None]
if no_grad_names:
    print(f"  params WITHOUT grad ({len(no_grad_names)}): {no_grad_names[:10]}{' ...' if len(no_grad_names) > 10 else ''}")
assert not no_grad_names, f"{len(no_grad_names)} interaction params got no grad"
bad_grad = [n for n, p in vis.interaction_stages.named_parameters() if not torch.isfinite(p.grad).all()]
assert not bad_grad, f"non-finite grads: {bad_grad}"
assert vis.shared_init.grad is not None
print(f"  backward OK: {sum(1 for _ in vis.interaction_stages.parameters())} interaction params + shared_init all have finite grads")

pred = model.predict_action([sample])
na = pred["normalized_actions"]
assert na.shape[0] == 1 and na.shape[2] == 7, na.shape
assert na.shape[1] == model.action_horizon, (na.shape, model.action_horizon)
print(f"  predict_action shape = {na.shape} — OK")

print("[4/5] optimizer LR groups contain interaction_stages")
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

groups = build_param_lr_groups(model, cfg)
group_names = [g.get("name") for g in groups]
assert "qwen_vl_interface.model.visual.interaction_stages" in group_names, group_names
counts = {g.get("name"): len(g["params"]) for g in groups}
print(f"  param groups: {counts}")

del model, out, action_loss
if device.type == "cuda":
    torch.cuda.empty_cache()

print("[5/5] original QwenPI regression (plain interface, 2B weights)")
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

qp_cfg = OmegaConf.create(
    {
        "framework": {
            "name": "QwenPI",
            "qwenvl": {
                "base_vlm": cfg.framework.qwenvl.base_vlm,
                "attn_implementation": "sdpa",
                "vl_hidden_dim": 2048,
                "num_vl_layers": 36,
            },
            "action_model": {"action_dim": 7, "state_dim": 7, "action_horizon": 8},
        },
        "datasets": {"vla_data": {}},
    }
)
qwen_pi = Qwen_PI(qp_cfg).to(device)
with torch.no_grad():
    qp_out = qwen_pi([sample, sample2])
assert torch.isfinite(qp_out["action_loss"]), qp_out
print(f"  QwenPI forward action_loss = {qp_out['action_loss'].item():.6f}")
with torch.no_grad():
    qp_pred = qwen_pi.predict_action([sample])
print(f"  QwenPI predict_action shape = {qp_pred['normalized_actions'].shape} — OK")

print("=" * 70)
print("ALL SMOKE CHECKS PASSED")
