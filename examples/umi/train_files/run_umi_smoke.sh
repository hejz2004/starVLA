#!/usr/bin/env bash
# Single-node smoke with the same WYT data, five stages and pretrained VLM.
# Defaults: 1 PPU, state=True, microbatch=1, accumulation=16, 10 updates, no W&B.
# NUM_PROCESSES=2 doubles the effective batch to 32 unless accumulation changes.
# Example: bash examples/umi/train_files/run_umi_smoke.sh
# State ablation: bash examples/umi/train_files/run_umi_smoke.sh --framework.action_model.use_state false
# Any trainer override is forwarded. Each default run gets a unique directory.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UMI_RESUME=0
export NODE_RANK=0
export NUM_NODES=1
export PROCESSES_PER_NODE="${NUM_PROCESSES:-1}"
export CONFIG_YAML="${CONFIG_YAML:-${script_dir}/starvla_qwenpi_da3_umi_smoke.yaml}"
export RUN_ID="${RUN_ID:-smoke_qwenpi_da3_umi_wyt_$(date +%Y%m%d_%H%M%S)_$$}"
export STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec bash "${script_dir}/run_umi_pretrain_v2.sh" "$@"
