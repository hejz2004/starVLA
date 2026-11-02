#!/usr/bin/env bash
# LIBERO 4-in-1 post-training from a selected UMI checkpoint on PAI.
# Submit this command once per node. PAI supplies RANK (node rank),
# WORLD_SIZE (node count), NPROC_PER_NODE, MASTER_ADDR, and MASTER_PORT.
# For a manual launch, set NODE_RANK, NUM_NODES, and PROCESSES_PER_NODE.
# Global VLA batch = total processes x per-device batch(8) x grad accumulation(1).
# Select pretraining weights with PRETRAINED_CHECKPOINT=/path/steps_N.pt or
# pass the path as the first argument. Use a distinct RUN_ID for each comparison.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"

config_yaml="${CONFIG_YAML:-${script_dir}/starvla_qwenpi_da3_libero_action_goals.yaml}"
run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
if (( $# > 1 )); then
    echo "Usage: bash ${BASH_SOURCE[0]} [pretrained_checkpoint_path]" >&2
    exit 2
fi
if (( $# == 1 )) && [[ -n "${PRETRAINED_CHECKPOINT:-}" ]] && [[ "$1" != "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "Checkpoint path argument and PRETRAINED_CHECKPOINT disagree" >&2
    exit 2
fi
pretrained_checkpoint="${1:-${PRETRAINED_CHECKPOINT:-}}"
pretrain_args=()
if [[ -n "${pretrained_checkpoint}" ]]; then
    if [[ ! -f "${pretrained_checkpoint}" ]]; then
        echo "Pretraining checkpoint does not exist: ${pretrained_checkpoint}" >&2
        exit 2
    fi
    pretrain_args+=(--trainer.pretrained_checkpoint "${pretrained_checkpoint}")
fi
# Give different pretraining weights different output directories by default.
# RUN_ID remains available to name each comparison explicitly.
run_prefix=da3_libero_posttrain
if [[ "$(basename "${config_yaml}")" == starvla_qwenpi_da3_libero_action_goals.yaml ]]; then
    run_prefix=da3_libero_goal_binary_posttrain
fi
if [[ -n "${RUN_ID:-}" ]]; then
    run_id="${RUN_ID}"
elif [[ -n "${pretrained_checkpoint}" ]]; then
    pretrain_name="$(basename "$(dirname "$(dirname "${pretrained_checkpoint}")")")"
    checkpoint_name="$(basename "${pretrained_checkpoint}")"
    checkpoint_name="${checkpoint_name%_pytorch_model.pt}"
    checkpoint_name="${checkpoint_name%.safetensors}"
    run_id="${run_prefix}_${pretrain_name}_${checkpoint_name}"
else
    run_id="${run_prefix}_vlm_init"
fi
node_rank="${NODE_RANK:-${RANK:-}}"
num_nodes="${NUM_NODES:-${WORLD_SIZE:-}}"
processes_per_node="${PROCESSES_PER_NODE:-${NPROC_PER_NODE:-}}"
master_addr="${MASTER_ADDR:-}"
master_port="${MASTER_PORT:-29500}"
: "${node_rank:?PAI must inject RANK, or set NODE_RANK manually}"
: "${num_nodes:?PAI must inject WORLD_SIZE, or set NUM_NODES manually}"
: "${processes_per_node:?PAI must inject NPROC_PER_NODE, or set PROCESSES_PER_NODE manually}"
if [[ -z "${master_addr}" ]]; then
    if [[ "${num_nodes}" == 1 ]]; then
        master_addr=127.0.0.1
    else
        echo "MASTER_ADDR is required for multi-node training" >&2
        exit 2
    fi
fi

# RESUME=1 restores this run_id's latest full training state when present;
# the selected pretrain weights are used only if no LIBERO checkpoint exists.
resume="${RESUME:-0}"
if [[ "${resume}" != 0 && "${resume}" != 1 ]]; then
    echo "RESUME must be 0 or 1; got ${resume}" >&2
    exit 2
fi
resume_args=()
if [[ "${resume}" == 1 ]]; then
    resume_args+=(--trainer.is_resume true)
fi

if [[ ! -f "${config_yaml}" ]]; then
    echo "Missing LIBERO training config: ${config_yaml}" >&2
    exit 2
fi

if [[ -z "${WANDB_API_KEY:-}" && -f .env ]]; then
    # shellcheck disable=SC1091
    source .env
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is unset; disabling W&B logging" >&2
    export WANDB_MODE=disabled
fi

unset STARVLA_DISABLE_DEEPSPEED
# torchvision.io.VideoReader emits this deprecation warning in every data
# worker. Suppress only that exact warning; keep all other warnings visible.
video_warning_filter='ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning:torchvision.io._video_deprecation_warning'
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${video_warning_filter}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=100000000

output_dir="${run_root_dir}/${run_id}"
# Never resume a run whose persisted label semantics differ from this config.
if [[ -f "${output_dir}/config.yaml" ]]; then
    python - "${config_yaml}" "${output_dir}/config.yaml" <<'PYCONFIG'
import sys
from omegaconf import OmegaConf
new, old = [OmegaConf.load(p).datasets.vla_data for p in sys.argv[1:]]
for key in ['data_mix', 'action_type', 'normalization_statistics', 'controller_goal_root']:
    if new.get(key) != old.get(key):
        raise SystemExit(f'Refusing to mix action contracts in the same run directory: {key}')
PYCONFIG
fi
mkdir -p "${output_dir}"
cp "${BASH_SOURCE[0]}" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"
echo "LIBERO post-train: ${num_nodes} node(s) x ${processes_per_node} PPU(s), run_id=${run_id}, resume=${resume}"
echo "Pretraining checkpoint: ${pretrained_checkpoint:-YAML default}"

exec python -m torch.distributed.run \
    --nnodes="${num_nodes}" \
    --nproc-per-node="${processes_per_node}" \
    --node-rank="${node_rank}" \
    --master-addr="${master_addr}" \
    --master-port="${master_port}" \
    --max-restarts=0 \
    starVLA/training/train_starvla.py \
        --config_yaml "${config_yaml}" \
        --run_root_dir "${run_root_dir}" \
        --run_id "${run_id}" \
        "${pretrain_args[@]}" \
        "${resume_args[@]}"
