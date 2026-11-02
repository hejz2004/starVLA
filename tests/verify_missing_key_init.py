"""Verify: manual init only touches dual-stream params actually missing.

Case 1: fresh load from the plain Qwen3-VL-2B checkpoint — all dual-stream
        keys are missing -> manually seeded (gate bias / weight, shared_init).
Case 2: save a DualStream checkpoint, then load it — keys all present ->
        values pass through bitwise untouched (training-resume scenario).
"""

import math
import os
import shutil
import tempfile

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.vlm.QWen3DualStream import _QWen3DualStream_VL_Interface

YAML = "examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero.yaml"

print("[case 1] fresh load from the plain 2B checkpoint (dual-stream keys missing)")
cfg = OmegaConf.load(YAML)
iface = _QWen3DualStream_VL_Interface(cfg)
vis = iface.model.model.visual

expected_bias = math.log(0.01 / 0.99)
b5 = vis.interaction_stages["5"].gate_mlp.fc2.bias
w5 = vis.interaction_stages["5"].gate_mlp.fc2.weight
si = vis.shared_init
print(f"  gate fc2 bias  = {b5.float().mean().item():.4f} (expect ~ {expected_bias:.4f})")
print(f"  gate fc2 wstd  = {w5.float().std().item():.5f} (expect ~ 0.001)")
print(f"  shared_init    = mean {si.mean().item():+.5f}, std {si.std().item():.4f} (expect ~0 / ~0.02)")
assert abs(b5.float().mean().item() - expected_bias) < 0.01  # bf16 storage rounds -4.5951 to -4.59375
assert 0 < w5.float().std().item() < 10 * 0.001
assert abs(si.mean().item()) < 0.01 and 0.005 < si.std().item() < 0.05
assert all(torch.isfinite(p).all() for n, p in vis.named_parameters() if "interaction_stages" in n or n == "shared_init")
print("  case 1 OK: missing params seeded with spec §10 init")

print("[case 2] save a DualStream checkpoint, reload -> values untouched")
tmp = tempfile.mkdtemp(prefix="verify_dual_ckpt_", dir="/workspace/hjz")
try:
    iface.model.save_pretrained(tmp)
    iface.processor.save_pretrained(tmp)
    saved_files = sorted(os.listdir(tmp))
    print(f"  saved: {saved_files}")

    cfg2 = OmegaConf.load(YAML)
    cfg2.framework.qwenvl.base_vlm = tmp
    iface2 = _QWen3DualStream_VL_Interface(cfg2)
    vis2 = iface2.model.model.visual

    for name, a, b in [
        ("shared_init", vis.shared_init, vis2.shared_init),
        ("gate fc2 bias  (stage 5)", vis.interaction_stages["5"].gate_mlp.fc2.bias, vis2.interaction_stages["5"].gate_mlp.fc2.bias),
        ("gate fc2 weight(stage 5)", vis.interaction_stages["5"].gate_mlp.fc2.weight, vis2.interaction_stages["5"].gate_mlp.fc2.weight),
        ("q1.weight      (stage 11)", vis.interaction_stages["11"].q1.weight, vis2.interaction_stages["11"].q1.weight),
        ("norm_ffn.bias  (stage 23)", vis.interaction_stages["23"].norm_ffn.bias, vis2.interaction_stages["23"].norm_ffn.bias),
    ]:
        same = torch.equal(a, b)
        print(f"  {name}: bitwise equal after reload = {same}")
        assert same, f"{name} was re-initialized on resume — must stay untouched"
    print("  case 2 OK: nothing re-initialized when the checkpoint has the keys")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("VERIFIED")
