"""Minimal DeepSpeed ZeRO dual-PPU training smoke.

Verifies the vendor deepspeed build (0.18.8+ali) can train a tiny model on
2 PPUs with ZeRO stage 2 (and stage 3 via DS_STAGE=3): engine creation,
forward/backward/optimizer steps with rank-specific data (so gradients
genuinely differ before allreduce), loss finiteness, and bit-identical
parameters across ranks after synced optimizer steps.

Run inside the container from the repo root:
    accelerate launch --num_processes 2 tests/deepspeed_zero_dual_ppu_smoke.py
    DS_STAGE=3 accelerate launch --num_processes 2 tests/deepspeed_zero_dual_ppu_smoke.py
"""

import os
import sys

import deepspeed
import torch
import torch.distributed as dist


def main() -> int:
    stage = int(os.environ.get("DS_STAGE", "2"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 32),
    )

    ds_config = {
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-3, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0},
        },
        "zero_optimization": {"stage": stage},
        "steps_per_print": 100,
    }
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"[stage {stage}] engine ready, world={world}, zero stage={engine.zero_optimization_stage()}", flush=True)

    for step in range(6):
        gen = torch.Generator(device="cuda").manual_seed(1000 + step * 10 + rank)
        x = torch.randn(8, 64, generator=gen, device="cuda")
        loss = engine(x).pow(2).mean()
        engine.backward(loss)
        engine.step()
        if not torch.isfinite(loss):
            print(f"[rank {rank}] stage {stage} step {step} NON-FINITE LOSS {loss.item()}", flush=True)
            return 1
        if rank == 0:
            print(f"[stage {stage}] step {step} loss {loss.item():.6f}", flush=True)

    mismatches = 0
    for name, p in engine.module.named_parameters():
        local = p.float().detach().clone()
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        for other in range(world):
            if not torch.equal(local, gathered[other]):
                diff = (local - gathered[other]).abs().max().item()
                print(f"[rank {rank}] stage {stage} PARAM MISMATCH {name} vs rank {other}: max|d|={diff}", flush=True)
                mismatches += 1

    partitioned = sum(1 for p in engine.module.parameters() if hasattr(p, "ds_tensor"))
    total = sum(1 for p in engine.module.parameters())
    if rank == 0:
        tag = "PASS" if mismatches == 0 else "FAIL"
        print(
            f"[stage {stage}] params bit-identical across ranks: {mismatches == 0}; "
            f"partitioned {partitioned}/{total} parameters — {tag}",
            flush=True,
        )
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
