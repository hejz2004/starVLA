#!/usr/bin/env bash
# PAI A100: eight processes/node. RANK/WORLD_SIZE describe nodes.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"
: "${RUN_ID:?}"
: "${CONFIG_YAML:?}"
node_rank="${RANK:?PAI must supply node rank}"
num_nodes="${WORLD_SIZE:?PAI must supply node count}"
processes_per_node=8
: "${MASTER_ADDR:?PAI must supply master address}"
master_port="${MASTER_PORT:-29500}"
run_root="${RUN_ROOT_DIR:-${repo_root}/playground/Checkpoints}"
output_dir="${run_root}/${RUN_ID}"
mkdir -p "${output_dir}/launch"
exec > >(tee -a "${output_dir}/launch/node${node_rank}.log") 2>&1
echo "$(date -Iseconds) A100 launch: node=${node_rank}/${num_nodes}, master=${MASTER_ADDR}:${master_port}"

# Reuse the runtime used by successful A100 QwenPI job dlc5psc23v46kcqk.
# This extracts a prepared NAS archive into each pod; it runs no pip/apt downloads.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export USE_TF=0 USE_FLAX=0
source /mnt/workspace/Native_Policy/user/wyt/a100/wc/ops/dlc_qwenpi/use_nas_runtime.sh
export PATH="/opt/qwenpi/train/bin:${PATH}"
export PYTHONPATH="${repo_root}"
# Install only the experiment logger from prepared Python 3.10 wheels.
python -m pip install --no-index \
    --find-links "${repo_root}/playground/Artifacts/swanlab_0.10.3_py310_wheels" \
    -r "${script_dir}/requirements-swanlab.txt"
env_dir="/tmp/starvla-a100-${RUN_ID}"
mkdir -p "${env_dir}"
python -m pip freeze > "${output_dir}/launch/node${node_rank}.packages.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Check actual model imports and PyAV decoding before starting 16 processes.
python - "${CONFIG_YAML}" <<'PY'
import sys
from pathlib import Path
import torch,torchvision,transformers,accelerate,deepspeed
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import _auto_import_framework_modules
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
cfg=OmegaConf.load(sys.argv[1])
assert torch.cuda.device_count()==8, torch.cuda.device_count()
assert all('A100' in torch.cuda.get_device_name(i) for i in range(8))
assert torch.cuda.is_bf16_supported()
if cfg.trainer.get("pretrained_checkpoint"):
    assert Path(cfg.trainer.pretrained_checkpoint).is_file()
_auto_import_framework_modules()
torchvision.set_video_backend('pyav')
video=next(Path(cfg.datasets.vla_data.data_root_dir).glob('*/videos/**/*.mp4'))
frame=next(iter(torchvision.io.VideoReader(str(video),'video')))
print('CUDA environment ready:',torch.__version__,torch.version.cuda,
      'transformers',transformers.__version__,'video_frame',tuple(frame['data'].shape),flush=True)
PY

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SWANLAB_MODE="${SWANLAB_MODE:-online}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=100000000
export TORCH_EXTENSIONS_DIR="${env_dir}/torch_extensions"
export TRITON_CACHE_DIR="${env_dir}/triton"
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# NCCL selects RDMA automatically; all pods use the A100 pool VPC.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
unset STARVLA_DISABLE_DEEPSPEED
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
if [[ "${node_rank}" == 0 ]]; then
    cp "${CONFIG_YAML}" "${output_dir}/launch/config.yaml"
    cp "${BASH_SOURCE[0]}" "${output_dir}/launch/"
fi
checkpoint_args=()
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
    checkpoint_args+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi
exec python -m torch.distributed.run \
    --nnodes="${num_nodes}" --nproc-per-node="${processes_per_node}" \
    --node-rank="${node_rank}" --master-addr="${MASTER_ADDR}" \
    --master-port="${master_port}" --max-restarts=0 \
    --rdzv-conf=timeout=1800 \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" --run_root_dir "${run_root}" --run_id "${RUN_ID}" \
    "${checkpoint_args[@]}" --trainer.is_resume false
