# DualStreamPI bottleneck research branch

Branch: `experiment/dualstream-bottleneck`.

This branch builds on the current starVLA training/data implementation. The
vision tower and processor interface are restored from `803dbc3`; the framework
inherits current QwenPI action training and inference, including independent
seeded action-head initialization, action masks and BF16-to-NumPy conversion.
No distillation code is introduced.

## Architecture

- Each view retains its private features through the original 24 vision blocks.
- Each scene has 32 shared tokens; activations are isolated across batch samples.
- After zero-based layers 5, 11, 17 and 23, shared tokens attend to all scene
  patches, run shared self-attention/FFN, and broadcast context to the patches.
- Fusion is `F = P + sigmoid(gate(P, C)) * C`, initially gated near 0.01.
- Fused features feed DeepStack and the final merger. The next private vision
  block continues from the private stream, not from the fused output.
- The existing late-alternating MV2 architecture remains separately selectable.
  Its register-token checkpoints are rejected by DualStreamPI.

The framework names `DualStreamPI`, `DualStreamQwenPI` and `DualStreamQwenFM`
resolve to the same class. Historical DualStream checkpoint tensor names are
preserved. Reproducing an old experiment still requires its matched config,
action normalization and source data.

## Configuration entry points

- `examples/droid/train_files/dualstream_droid_smoke.yaml`: DROID EEF index,
  anchor stride 3, consecutive H16 actions, full-parameter training; 32-step
  startup configuration, not the full pretraining recipe.
- `examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero_action_goals.yaml`:
  controller-goal action interface, microbatch 64, accumulation 1, 15000 updates,
  750 warmup. Launch with four processes for global batch 256.

Both configs initialize the VLM and leave `trainer.pretrained_checkpoint` unset.
For DROID-to-LIBERO transfer, supply a compatible **DualStream** DROID checkpoint
and matching source provenance. No DualStream DROID training/checkpoint is
created by making this branch.

Before distributed training, run `python scripts/ensure_swanlab.py` with the
training interpreter (SwanLab 0.10.5). The DROID preparation/supervision scripts
named for native/MV2 still enforce those architectures; they are not DualStream
submission tools. Adapt their architecture validation before using them for a
future DualStream transfer run.

## Validation

Targeted tests cover private-stream independence, cross-view information flow,
scene isolation, view permutation, gate initialization, checkpoint roundtrip,
framework registration, action-head seed matching, and existing MV2/BF16
behavior. Full-scale training or rollout evaluation has not been launched.
