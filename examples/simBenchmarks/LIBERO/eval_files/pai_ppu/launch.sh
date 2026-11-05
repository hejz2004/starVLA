#!/usr/bin/env bash
# Standard LIBERO on the verified PPU image; same simulator dependencies as Plus.
set -Eeuo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STARVLA_DIR="${STARVLA_DIR:-$(cd "$BUNDLE/../../../../.." && pwd)}"
: "${RUN_ROOT:?}" "${CKPT_PATH:?}" "${LIBERO_HOME:?}"
export STARVLA_PYTHON="${STARVLA_PYTHON:-/opt/ac2/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-$STARVLA_PYTHON}"
export LIBERO_SITE_PACKAGES="${LIBERO_SITE_PACKAGES:-$("$LIBERO_PYTHON" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled SWANLAB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
bootstrap_log="${BOOTSTRAP_LOG:-${RUN_ROOT}.bootstrap.log}"
mkdir -p "$(dirname "$bootstrap_log")"
exec > >(tee -a "$bootstrap_log") 2>&1
cd "$STARVLA_DIR"
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
"$STARVLA_PYTHON" - "$constraints" <<'PYCONSTRAINTS'
import importlib.metadata as m,sys
names=['torch','torchvision','torchaudio','triton','transformers','accelerate','deepspeed']
with open(sys.argv[1],'w') as f:
    for name in names:
        try:f.write(name+'=='+m.version(name)+'\n')
        except m.PackageNotFoundError:pass
if 'ppu' not in m.version('torch'):raise SystemExit('Use the verified PPU image/runtime')
PYCONSTRAINTS
if [[ "${INSTALL_DEPS:-1}" == 1 ]]; then
    if ! "$LIBERO_PYTHON" -c 'import ctypes;ctypes.CDLL("libOSMesa.so.8");ctypes.CDLL("libMagickWand-6.Q16.so.7")' || ! command -v ffmpeg >/dev/null; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libosmesa6 libgl1 libmagickwand-dev ffmpeg
    fi
    "$STARVLA_PYTHON" -m pip install -c "$constraints" -r "$STARVLA_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/requirements.txt"
    "$STARVLA_PYTHON" -m pip install --no-deps --no-build-isolation -e "$STARVLA_DIR"
fi
rm -f "$constraints"
trap - EXIT
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT DEBUG
export CKPT="$CKPT_PATH" OUTPUT_DIR="$RUN_ROOT"
export DEVICES="${DEVICES:-$(seq -s, 0 $((${NUM_CARDS:-16}-1)))}"
export WORKERS_PER_DEVICE="${WORKERS_PER_CARD:-20}"
exec bash "$STARVLA_DIR/examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh" \
    --suites libero_spatial libero_object libero_goal libero_10 --num-trials-per-task 50 "$@"
