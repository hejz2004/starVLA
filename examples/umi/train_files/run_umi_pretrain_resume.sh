#!/usr/bin/env bash
# Resume an existing WYT five-stage run, using its saved config.full.yaml.
# Preserves the saved state flag, microbatch, accumulation and model settings.
# RUN_ID and the original topology must match; RUN_ROOT_DIR defaults as before.
# Example: RUN_ID=umi_wyt_state_off_v1 bash examples/umi/train_files/run_umi_pretrain_resume.sh
# Trailing overrides are forwarded; the trainer rejects incompatible identities.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUN_ID:?Set RUN_ID to the WYT run to resume}"
export UMI_RESUME=1
exec bash "${script_dir}/run_umi_pretrain_v2.sh" "$@"
