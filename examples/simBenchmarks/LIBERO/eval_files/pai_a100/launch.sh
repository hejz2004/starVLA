#!/usr/bin/env bash
set -Eeuo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$BUNDLE/../../../../.." && pwd)
: "${RUN_ROOT:?}" "${CKPT_PATH:?}" "${BENCHMARK:?}"
export EVAL_NODE_RANK="${RANK:?PAI node rank missing}"
[[ "${WORLD_SIZE:?}" == 2 ]]
mkdir -p "$RUN_ROOT/launch"
exec > >(tee -a "$RUN_ROOT/launch/node${EVAL_NODE_RANK}.log") 2>&1
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source /mnt/workspace/Native_Policy/user/wyt/a100/wc/ops/dlc_qwenpi/use_nas_runtime.sh
export PYTHONPATH="$REPO" PATH="/opt/qwenpi/train/bin:$PATH"
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1 MALLOC_ARENA_MAX=2
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT DEBUG
cd "$REPO"
# Match the PPU evaluator and the simulator used to generate action targets.
# Upgrade only the job-local simulator; keep the shared NAS archive unchanged.
if [[ "$BENCHMARK" == libero ]]; then
  /opt/qwenpi/train/bin/python -m pip --python /opt/qwenpi/libero/bin/python install \
    --no-index --no-deps --find-links "$REPO/playground/Artifacts/mujoco_3.2.3_cp310" \
    mujoco==3.2.3
fi
/opt/qwenpi/train/bin/python - <<'PY'
import torch
assert torch.cuda.device_count()==8
assert all('A100' in torch.cuda.get_device_name(i) for i in range(8))
print('A100 evaluator runtime:',torch.__version__,torch.version.cuda,flush=True)
PY
exec /opt/qwenpi/train/bin/python "$BUNDLE/run_node.py"
