#!/usr/bin/env bash
# WYT UMI: five stages, 5Hz anchors, 30Hz future actions, pretrained VLM.
# Run once per node. PAI provides node RANK/WORLD_SIZE/NPROC_PER_NODE;
# manual topology: NODE_RANK, NUM_NODES, PROCESSES_PER_NODE, MASTER_ADDR.
#
# New run (topology variables are supplied separately):
#   RUN_ID=umi_wyt_state_on_v1 bash examples/umi/train_files/run_umi_pretrain_v2.sh
# State ablation (use a separate RUN_ID):
#   RUN_ID=umi_wyt_state_off_v1 bash examples/umi/train_files/run_umi_pretrain_v2.sh \
#     --framework.action_model.use_state false
# Every trailing KEY=VALUE or --key value override reaches the trainer.
# Set WANDB_MODE=disabled to skip W&B and .env loading.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

# The trainer accepts --key value / --key=value. Normalize bare KEY=VALUE
# here so advertised overrides cannot silently disappear in its CLI parser.
overrides=()
while (( $# )); do
    arg="$1"
    shift
    case "${arg}" in
        --*) key="${arg#--}"; key="${key%%=*}" ;;
        *=*) key="${arg%%=*}"; arg="--${arg}" ;;
        *) echo "Unexpected argument: ${arg}; use --key value or key=value" >&2; exit 1 ;;
    esac
    case "${key}" in
        config_yaml|run_id|run_root_dir|trainer.is_resume)
            echo "Set CONFIG_YAML, RUN_ID, RUN_ROOT_DIR or use the resume launcher; do not override ${key} in trailing arguments" >&2
            exit 1 ;;
    esac
    overrides+=("${arg}")
    if [[ "${arg}" != *=* && $# -gt 0 && "$1" != --* ]]; then
        overrides+=("$1")
        shift
    fi
done

NODE_RANK="${NODE_RANK:-${RANK:-}}"
NUM_NODES="${NUM_NODES:-${WORLD_SIZE:-}}"
PROCESSES_PER_NODE="${PROCESSES_PER_NODE:-${NPROC_PER_NODE:-16}}"
: "${NODE_RANK:?Set NODE_RANK (or PAI RANK)}"
: "${NUM_NODES:?Set NUM_NODES (or PAI WORLD_SIZE)}"
for name in NODE_RANK NUM_NODES PROCESSES_PER_NODE; do
    if [[ ! "${!name}" =~ ^[0-9]+$ ]]; then
        echo "${name} must be a nonnegative integer" >&2
        exit 1
    fi
done
if (( NUM_NODES < 1 || PROCESSES_PER_NODE < 1 || NODE_RANK >= NUM_NODES )); then
    echo "Invalid topology: require NUM_NODES >= 1, PROCESSES_PER_NODE >= 1, NODE_RANK < NUM_NODES" >&2
    exit 1
fi
if (( NUM_NODES > 1 )); then
    : "${MASTER_ADDR:?Set MASTER_ADDR to the master node address for multi-node training}"
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

resume="${UMI_RESUME:-0}"
case "${resume}" in 0|1) ;; *) echo "UMI_RESUME must be 0 or 1" >&2; exit 1 ;; esac
if [[ "${resume}" == 1 ]]; then
    : "${RUN_ID:?Set RUN_ID to an existing WYT staged run}"
    run_root_dir="${RUN_ROOT_DIR:-./playground/Checkpoints}"
    run_id="${RUN_ID}"
    output_dir="${run_root_dir}/${run_id}"
    config_yaml="${CONFIG_YAML:-${output_dir}/config.full.yaml}"
else
    config_yaml="${CONFIG_YAML:-${script_dir}/starvla_dualstream_umi_v2.yaml}"
fi
# Accept a full path, a repo-relative path, or a filename in train_files/.
if [[ ! -f "${config_yaml}" && -f "${script_dir}/${config_yaml}" ]]; then
    config_yaml="${script_dir}/${config_yaml}"
fi
if [[ ! -f "${config_yaml}" ]]; then
    echo "Config not found: ${config_yaml}" >&2
    exit 1
fi
launch_defaults="$(python - "${config_yaml}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
if cfg.datasets.vla_data.dataset_py != "umi_indexed":
    raise SystemExit("This launcher requires the WYT umi_indexed configuration")
print(cfg.run_root_dir)
print(cfg.run_id)
PY
)"
mapfile -t launch_defaults <<< "${launch_defaults}"
if [[ "${resume}" == 0 ]]; then
    run_root_dir="${RUN_ROOT_DIR:-${launch_defaults[0]}}"
    run_id="${RUN_ID:-${launch_defaults[1]}}"
    output_dir="${run_root_dir}/${run_id}"
    if [[ -f "${output_dir}/config.full.yaml" || -f "${output_dir}/config.yaml" ]]; then
        echo "Run already exists: ${output_dir}; choose a new RUN_ID or use run_umi_pretrain_resume.sh" >&2
        exit 1
    fi
else
    if [[ ! -f "${output_dir}/config.full.yaml" ]]; then
        echo "Cannot resume: ${output_dir}/config.full.yaml is missing" >&2
        exit 1
    fi
fi

if [[ "${WANDB_MODE:-}" != disabled && "${WANDB_DISABLED:-}" != true && "${WANDB_DISABLED:-}" != 1 ]]; then
    if [[ -z "${WANDB_API_KEY:-}" && -f .env ]]; then
        source .env
    fi
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "[umi] No WANDB_API_KEY; disabling W&B for this run" >&2
        export WANDB_MODE=disabled
    else
        # A plain KEY=value line sourced from .env is otherwise shell-local.
        export WANDB_API_KEY
    fi
fi
# These flags are useful for local smoke runs, but would disable the formal
# distributed backend or force CPU execution if inherited by this launcher.
case "${STARVLA_DISABLE_DEEPSPEED:-0}:${ACCELERATE_USE_CPU:-false}" in
    1:*|*:true|*:True|*:TRUE|*:1|*:yes|*:Yes|*:YES)
        echo "Formal UMI training requires DeepSpeed on accelerators; unset STARVLA_DISABLE_DEEPSPEED and ACCELERATE_USE_CPU" >&2
        exit 1 ;;
esac
export ACCELERATE_DEEPSPEED_ZERO_STAGE="${ACCELERATE_DEEPSPEED_ZERO_STAGE:-2}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=100000000

mkdir -p "${output_dir}"
if [[ "${resume}" == 0 && "${NODE_RANK}" == 0 ]]; then
    cp -- "$0" "${output_dir}/"
    cp -- "${config_yaml}" "${output_dir}/"
fi
# Resume does not copy over previous configs; the trainer checks identity first.
echo "[umi] config=${config_yaml} run=${output_dir} resume=${resume}"

resume_flag=false
if [[ "${resume}" == 1 ]]; then resume_flag=true; fi

exec python -m torch.distributed.run \
  --nnodes="${NUM_NODES}" \
  --nproc-per-node="${PROCESSES_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --max-restarts=0 \
  starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --trainer.is_resume "${resume_flag}" \
    "${overrides[@]}"
