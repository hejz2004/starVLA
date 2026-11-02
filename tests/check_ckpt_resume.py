"""Integrity check for the ZeRO-2 save/resume smoke checkpoints.

Loads steps_30 and steps_45 (CPU) and verifies: identical key sets,
dual-stream parameters present, all tensors finite, and that training
actually progressed between the two saves.

Run inside the container from the repo root:
    python3 tests/check_ckpt_resume.py
"""

import torch

BASE = "playground/Checkpoints/smoke_dualstream_libero_randinit/checkpoints"


def main():
    c30 = torch.load(f"{BASE}/steps_30_pytorch_model.pt", map_location="cpu")
    c45 = torch.load(f"{BASE}/steps_45_pytorch_model.pt", map_location="cpu")
    print("keys:", len(c30), len(c45), "| identical key sets:", set(c30) == set(c45))

    ds_keys = sorted(k for k in c30 if "shared_init" in k or "interaction_stages" in k)
    print("dual-stream keys:", len(ds_keys), "| e.g.", ds_keys[0] if ds_keys else None)

    bad30 = sum(1 for v in c30.values() if not torch.isfinite(v.float()).all())
    bad45 = sum(1 for v in c45.values() if not torch.isfinite(v.float()).all())
    print("non-finite tensors:", bad30, bad45)

    changed = [k for k in c30 if not torch.equal(c30[k], c45[k])]
    print("params changed 30->45:", len(changed))
    for k in sorted(changed)[:3]:
        delta = (c30[k].float() - c45[k].float()).abs().max().item()
        print(f"  e.g. {k}: max|delta|={delta:.3e}")


if __name__ == "__main__":
    main()
