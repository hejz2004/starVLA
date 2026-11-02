#!/bin/bash
# DeepSpeed ZeRO-2 smoke: DualStreamQwenPI, fully random init (load_weights:
# false), libero_object. Same as run_libero_smoke.sh but WITHOUT
# STARVLA_DISABLE_DEEPSPEED, so training runs the repo default path
# Accelerator(deepspeed_plugin=DeepSpeedPlugin()) -> ZeRO stage 2 on 2 PPUs.
#
# Usage (inside the container, repo root):
#   # run A, continuous 30 steps (saves steps_15, steps_30):
#   bash examples/simBenchmarks/LIBERO/train_files/run_libero_smoke_ds.sh
#   # run B phase 1, fresh 15 steps (run B uses its own run_id):
#   RUN_ID=smoke_dualstream_libero_randinit_b \
#     CONFIG_YAML=starvla_dualstream_libero_smoke_b15.yaml \
#     bash examples/simBenchmarks/LIBERO/train_files/run_libero_smoke_ds.sh
#   # run B phase 2, strict resume 15 -> 30:
#   RUN_ID=smoke_dualstream_libero_randinit_b \
#     CONFIG_YAML=starvla_dualstream_libero_smoke_b30.yaml \
#     bash examples/simBenchmarks/LIBERO/train_files/run_libero_smoke_ds.sh
#   # legacy fallback: resume the flat-weights-only run dir 45 -> 60:
#   CONFIG_YAML=starvla_dualstream_libero_smoke_resume.yaml \
#     bash examples/simBenchmarks/LIBERO/train_files/run_libero_smoke_ds.sh
export WANDB_MODE=disabled

tf_dir=./examples/simBenchmarks/LIBERO/train_files
config_yaml=${tf_dir}/${CONFIG_YAML:-starvla_dualstream_libero_smoke.yaml}
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-smoke_dualstream_libero_randinit}

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
