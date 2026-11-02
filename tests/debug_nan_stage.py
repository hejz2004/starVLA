"""Pinpoint the first op that produces NaN inside interaction stage 5."""

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

from starVLA.model.modules.vlm.QWen3DualStream import _QWen3DualStream_VL_Interface

iface = _QWen3DualStream_VL_Interface(cfg)
iface = iface.to(device)
iface.eval()

vis = iface.model.model.visual

print("=== parameter finiteness of new modules ===")
bad_params = []
for name, p in vis.named_parameters():
    if ("interaction_stages" in name or "shared_init" in name) and not torch.isfinite(p).all():
        bad_params.append(name)
print(f"  non-finite params: {bad_params if bad_params else 'none'}")
g0 = vis.interaction_stages["5"].gate_mlp.fc2.bias.detach().float().mean().item()
print(f"  stage5 gate fc2 bias mean = {g0:.4f} (expect ~ -4.595)")

qin = iface.build_qwenvl_inputs(
    images=[[make_image(224, 0), make_image(224, 1), make_image(224, 2)], [make_image(224, 3), make_image(224, 4)]],
    instructions=["pick up the red block", "open the drawer"],
)

stage = vis.interaction_stages["5"]


def pre_hook(module, args):
    private, shared = args[0], args[1]
    pf = torch.isfinite(private).all().item()
    sf = torch.isfinite(shared).all().item()
    print(f"  STAGE IN  private finite={pf} max={private.float().abs().max().item():.4f}")
    print(f"  STAGE IN  shared   finite={sf} max={shared.float().abs().max().item():.4f}")


def make_sub_hook(name):
    def h(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if isinstance(o, torch.Tensor):
            f = torch.isfinite(o).all().item()
            print(f"    {name}: finite={f} max={o.float().abs().max().item():.4f}")

    return h


handle_pre = stage.register_forward_pre_hook(pre_hook)
sub_handles = []
for sub_name, m in stage.named_modules():
    if sub_name:
        sub_handles.append(m.register_forward_hook(make_sub_hook(f"stage5.{sub_name}")))

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    merged, ds = vis(qin["pixel_values"], grid_thw=qin["image_grid_thw"])

handle_pre.remove()
for h in sub_handles:
    h.remove()

print(f"  merged finite: {torch.isfinite(merged).all().item()}")
print("DONE")
