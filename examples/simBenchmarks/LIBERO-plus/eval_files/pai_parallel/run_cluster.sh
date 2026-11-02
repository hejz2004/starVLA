#!/usr/bin/env bash
# Prepared-environment entry. Use launch.sh for explicit fresh-image dependency installation.
set -Eeuo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STARVLA_DIR="${STARVLA_DIR:-$(cd "$BUNDLE/../../../../.." && pwd)}"
: "${LIBERO_HOME:?Set shared LIBERO-plus source root}"
export LIBERO_ASSETS="${LIBERO_ASSETS:-/mnt/workspace/Data/LIBERO_PLUS_EVAL_4976dc3/assets}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-$STARVLA_PYTHON}"
mode="${EVAL_MODE:-full}"
if [[ "$mode" == full ]]; then cards="${NUM_CARDS:-16}"; [[ "$cards" == 16 ]] || { echo 'Full preset requires 16 cards'; exit 2; }
elif [[ "$mode" == smoke ]]; then cards="${NUM_CARDS:-2}"
else echo 'EVAL_MODE must be full or smoke'; exit 2; fi
[[ "$cards" =~ ^[1-9][0-9]*$ ]] || exit 2
workers="${WORKERS_PER_CARD:-20}"
RUN_ID="${RUN_ID:-libero_plus_goal_binary_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$STARVLA_DIR/playground/Evaluation/$RUN_ID}"
CKPT_PATH="${CKPT_PATH:-$STARVLA_DIR/playground/Checkpoints/multiview-umi110k-libero-posttrain/checkpoints/steps_15000_pytorch_model.pt}"
devices="${DEVICES:-$(seq -s, 0 $((cards-1)))}"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT DEBUG
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$STARVLA_DIR${PYTHONPATH:+:$PYTHONPATH}"
cmd=("$STARVLA_PYTHON" -u "$BUNDLE/launchers/evaluation_controller.py"
 --ckpt "$CKPT_PATH" --output-dir "$RUN_ROOT" --devices "$devices"
 --workers-per-device "$workers" --libero-home "$LIBERO_HOME" --libero-assets "$LIBERO_ASSETS"
 --libero-base-assets "${LIBERO_BASE_ASSETS:-}"
 --simulation-site "${LIBERO_SITE_PACKAGES:-$STARVLA_DIR/playground/Artifacts/libero_plus_sim_runtime}"
 --policy-python "$STARVLA_PYTHON" --simulation-python "$LIBERO_PYTHON"
 --base-port "${BASE_PORT:-12883}")
cmd+=(--cpu-budget "${CPU_BUDGET:-$((cards*10))}")
if [[ "$mode" == smoke ]]; then cmd+=(--task-ids 0 1 2 3 4); fi
case "${FORCE_ZERO_GATE:-0}" in
  1) cmd+=(--force-zero-gate) ;;
  0) ;;
  *) echo 'FORCE_ZERO_GATE must be 0 or 1' >&2; exit 2 ;;
esac
exec "${cmd[@]}" "$@"
