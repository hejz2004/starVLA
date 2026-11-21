# DualStreamPI bottleneck research branch

Branch: `experiment/dualstream-bottleneck`.

This branch builds on the current starVLA training/data implementation. The
vision tower and processor interface originate from `803dbc3`; the framework
inherits current QwenPI action training and inference, including independent
seeded action-head initialization, action masks and BF16-to-NumPy conversion.
No distillation code is introduced.

## Architecture

- The original 24 vision blocks retain per-image self-attention. Cross-view
  information enters their residual stream at the interaction layers below.
- By default each scene has 32 shared tokens; activations are isolated across batch samples.
- By default after zero-based layers 5, 11, 17 and 23, shared tokens attend to all scene
  patches, run shared self-attention/FFN, and broadcast context to the patches.
- Fusion is `F = P + sigmoid(gate(P, C)) * C`, initially gated near 0.01.
- Fused features replace `hidden_states`, so subsequent vision blocks process
  cross-view context. DeepStack and the final merger read `hidden_states`
  through the standard Qwen3-VL output path, without a separate fused output.
- Shared-token states still propagate between interaction layers. This is
  separate from writing fused patch features back into the vision backbone.
- Attention output uses the Transformers `[batch, tokens, heads, head_dim]`
  layout directly. The historical extra transpose mixed token/head axes and
  is corrected here, with an explicit per-head attention reference test.
- The existing late-alternating MV2 architecture remains separately selectable.
  Its register-token checkpoints are rejected by DualStreamPI.

The framework names `DualStreamPI`, `DualStreamQwenPI` and `DualStreamQwenFM`
resolve to the same class. Historical DualStream checkpoint tensor names are
preserved, but the forward semantics changed on 2026-11-19: historical versions
sent fusion only to output mergers, while this branch now feeds it into later
vision blocks. Matching tensor shapes do not establish behavioral compatibility.
Reproducing an old experiment requires its frozen source version, matched
config, action normalization and source data.

## Configuration entry points

The shared-token count and fusion positions are independent hyperparameters:

```yaml
framework:
  qwenvl:
    dual_stream:
      enabled: true
      num_shared_tokens: 32
      interaction_indexes: [5, 11, 17, 23]
```

`num_shared_tokens` accepts any positive integer (for example 8, 16, 32 or 64).
`interaction_indexes` selects the blocks **after** which fusion is applied,
using zero-based indexes: `[5, 11, 17, 23]` means after blocks 6, 12, 18 and 24.
Any nonempty list of distinct integers in `[0, depth - 1]` is supported and
sorted into execution order. For example `[7, 15]` inserts only two stages;
neither a DeepStack tap nor the final block is required. DeepStack output sites
and merger modules retain their original configuration. Use `enabled: false`
to disable fusion; empty lists, duplicates, nonintegers and out-of-range indexes
are rejected. At the vision-model level, omitted/null positions default to the
DeepStack taps plus the final block; framework defaults explicitly select the
four positions above. Effective positions/count are saved in the HF config.

Each insertion has its own attention, FFN and gate parameters; `shared_init`
is shared across stages. With the current 1024-dimensional vision features,
4096-dimensional fusion FFN and 256-dimensional gate MLP, the added parameter
count is `21,530,113 * number_of_stages + 1024 * num_shared_tokens`.
Changing positions or token count changes the checkpoint architecture: use
the matching configuration when resuming or transferring trained weights.

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

Targeted tests cover view independence before the first interaction, cross-view
information and gradients reaching subsequent vision blocks,
scene isolation, view permutation, gate initialization, checkpoint roundtrip,
framework registration, action-head seed matching, and existing MV2/BF16
behavior. Full-model training/resume and vision-inference checks are described
below; these checks do not establish trained-policy rollout success.

## Performance migration (2026-11-19)

`examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero_optimized.yaml`
keeps the existing LIBERO controller-goal targets, batch 64, 15000 updates and
learning rates. It enables gradient checkpointing and the following optimizations.
The four-GPU formal recipe was subsequently submitted using frozen source
snapshots, so later edits to this worktree do not alter those running jobs.

- `framework.qwenvl.skip_unused_lm_head: true`: action training/inference returns
  hidden states without projecting vocabulary logits. The ordinary language
  forward retains its LM head. Labels with an explicitly skipped head are rejected.
- `framework.qwenvl.batch_vision_attention: true`: SDPA groups equal-length images
  into batches. Different lengths form separate buckets; image and scene boundaries
  remain intact. DualStream interactions still operate separately within each scene.
- `datasets.vla_data.worker_preprocessing: true`: spawn workers run the same Qwen
  processor/chat template, preserving pixels, token IDs, masks, actions and states.
  This supports DualStream and non-staged data loaders with workers > 0.
- `datasets.vla_data.cuda_prefetch: true`: pinned CPU batches move on a dedicated
  CUDA stream with one-batch lookahead. The trainer cursor counts consumed batches,
  including resume skipping and epoch resets, rather than prefetched batches.

Vision, language, bottleneck interaction and Action DiT attention use SDPA. The
experimental external FA4 adapter, its configuration and its kernel-specific tests
were removed at the user's request after it provided no incremental throughput
benefit in this workload. Historical measurements and source snapshots remain in
the artifact directory; they are not dependencies of the retained SDPA code.

### Correctness checks

Two actual-training bugs were fixed:

- Batched vision attention allocates its output using the query dtype. An FP32
  residual stream under BF16 autocast previously caused an `index_copy` dtype
  mismatch on torch 2.8.
- Spawned preprocessing workers no longer initialize a distributed process group
  through logging just because they inherit rank environment variables.

The real two-rank training entry completed three updates at batch 4 per GPU,
without gradient checkpointing, saved a complete step-3 checkpoint, and resumed
for step 4. SwanLab 0.10.5 logging succeeded. Peak allocated memory was 41.31 GiB
on the logged rank. Resume functionality was checked, not bitwise equivalence
with an uninterrupted trajectory.

On the full 3.889B-parameter model at batch 4, the optimized SDPA path has 0.395%
parameter-gradient relative L2 difference from the original BF16 path, with
identical predicted actions. This is a numerical check, not a long-run convergence
guarantee. Dedicated 24-layer vision tests cover 32 shared tokens at sites
5/11/17/23, CPU FP32 and CUDA BF16, uniform/mixed image sizes and view counts.
Merger/DeepStack outputs and all parameter gradients match exactly in these
small-width cases. Perturbing one view affects the subsequent ViT stream within
its scene, without changing another scene. Cross-view gradients and the mixed
FP32-residual/BF16-autocast case are covered explicitly.

Regression coverage also includes unused-head bypass, language labels, raw versus
worker-processed policy loss/actions/gradients, masks, state, repeated diffusion,
spawn worker rank inheritance, pinned transfer ordering, and trainer epoch/resume
cursors. No external FA4 runtime is needed for these tests.

Full-model artifacts and source provenance:
`/hdfs/share-data-1/users/hjz/Workspace/artifacts/dualstream-2gpu-validation-20261119`.
Local small-model test environment:
`/hdfs/share-data-1/users/hjz/Workspace/artifacts/dualstream-optimization-validation-20261119`.
The remote image uses Python 3.11, torch 2.8.0+cu128, transformers 4.57.6,
DeepSpeed ZeRO-2, and SwanLab 0.10.5 on two RTX PRO 6000 Blackwell GPUs.

### Throughput with gradient checkpointing

Measured on one node with two PRO 6000 GPUs, batch 64 per GPU (global 128),
GC enabled, BF16 and ZeRO-2. Each variant runs 30 optimizer steps; discard
the first five and report the median of the remaining 25. Timing includes
batch acquisition, forward/backward and optimizer updates, excluding startup,
checkpoint writes and logging. Consecutive logged wall times give a similar result.
All variants use identical seeds, data recipe, model and worker counts.

| Cumulative configuration | Seconds/step | Samples/second |
| --- | ---: | ---: |
| Baseline | 3.191 | 40.12 |
| + Skip unused LM head | 3.108 | 41.18 |
| + Batched vision SDPA | 2.696 | 47.49 |
| + Worker preprocessing and CUDA prefetch | 2.530 | 50.60 |

Overall throughput rises by 26.1%; step time drops by 20.7%.
The sequential ablation saves approximately 0.082s
from the LM-head bypass, 0.413s from vision batching,
and 0.166s from the worker/prefetch pipeline.
These marginal contributions depend on ablation order; they are not independent
universal speedup factors. Vision batching is the largest observed contributor.
Peak allocated GPU memory on rank 0 is 61.96 GiB (reserved 70.84 GiB) with all
optimizations; both ranks completed without OOM. Finite losses/gradients were
observed throughout; this short benchmark is not a convergence evaluation.

After removing FA4, all 36 retained regression tests passed both locally and
in the actual torch 2.8 training image (19.29 seconds remotely), without adding
the external FA4 source or DSL runtime to the test import path.

The old four-GPU research-node run is a different baseline: its 52 logged records
through step 260 have median 6.203s/step and maximum 7.980s; the last ten records
have median 7.120s. It used four GPUs/global256 and OMP/MKL/OpenBLAS threads=1;
the new benchmark uses two GPUs/global128 and OMP threads=4 on the new node group.
Both use the same GPU family, image versions, per-GPU batch64 and GC. The reason
for the cross-run gap has not been isolated. Do not attribute the old ~7s to new
~2.53s difference entirely to these code changes or project the 2-GPU timing onto
4 GPUs without measurement.

Sources: `benchmark-summary.json`, per-run `steps.jsonl`,
`sdpa-final-remote-regression.log` in the full-model artifact directory.
SwanLab: https://swanlab.cn/@hejiazhen/dualstream-optimization-validation/runs/2vk95nw1

### Explicit inference recheck

The follow-up tests use `torch.inference_mode()` and cover separate versus packed
scenes, scene permutation, interleaved image-size buckets, differing view counts,
and exact isolation of unchanged scenes. All seven visual tests passed locally
in 4.74 seconds. They include the four-stage forward/backward checks above.

A full-width 493,110,276-parameter vision tower was also tested using all 315 base
VLM visual tensors and newly initialized DualStream fusion (32 tokens, sites
5/11/17/23). Four real LIBERO samples with two views each were evaluated on the
local PRO 6000 GPU, torch 2.10, in BF16 and FP32 with TF32 disabled. In both dtypes,
the original per-image attention loop and batched attention produced exactly
equal merger and all three DeepStack outputs. Scene permutation preserved output
order exactly; perturbing one scene left all other scenes exactly unchanged.

Changing the scene batch size itself is a separate numerical effect: B4 versus
four B1 calls has a maximum feature relative L2 difference of 6.0085% in BF16
and 9.2811e-6 in FP32. The old and new implementations have identical errors in
this diagnostic. Thus the batching optimization preserves the reference at the
same scene batch size, but does not promise bitwise invariance across batch sizes.
This checks pretrained-vision inference with initialized fusion, not LIBERO rollout
success of a trained policy. Prior full-policy B4 comparison also found identical
predicted actions for the optimized SDPA and baseline paths.

Evidence: `vit-inference-recheck.log`, `full-vit-inference-recheck.json`, and
`full-vit-inference-recheck-fp32.json` in the full-model artifact directory.

### Checkpoint retention

The optimized recipe saves weights and full optimizer/scheduler/RNG state every
2,000 updates and retains only the latest checkpoint pair. With
`save_final_full_state: true`, completion also saves a resumable checkpoint at
step 15,000 even though it is outside the periodic interval. This option requires
`save_full_state: true` and defaults to false for other recipes.

The final model export hard-links the latest weights, with a copy fallback across
filesystems. Retention tests cover both PyTorch and safetensors formats, removal
of old weights/state pairs, and repeated finalization without duplicate saves.

The combined pre-commit regression on 2026-11-21 passed all 40 tests, including
the inference and retention additions (27.30 seconds locally). PyTorch emitted
only deprecated pinned-memory device-argument warnings.
