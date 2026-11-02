#!/bin/bash
# Smoke test launch: DualStreamQwenPI, fully random init (load_weights: false),
# libero_object dataset, 30 steps. See starvla_dualstream_libero_smoke.yaml.
#
# Usage (inside the container, repo root):
#   bash examples/simBenchmarks/LIBERO/train_files/run_libero_smoke.sh
# Override process count:  NUM_PROCESSES=1 bash .../run_libero_smoke.sh

# Plain DDP instead of DeepSpeed: fewer variables for a smoke run.
export STARVLA_DISABLE_DEEPSPEED=1
# Skip W&B (the trainer also honors WANDB_DISABLED).
export WANDB_MODE=disabled

config_yaml=./examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero_smoke.yaml
run_root_dir=./playground/Checkpoints
run_id=smoke_dualstream_libero_randinit

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/ 2>/dev/null

NUM_PROCESSES=${NUM_PROCESSES:-2}

accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
