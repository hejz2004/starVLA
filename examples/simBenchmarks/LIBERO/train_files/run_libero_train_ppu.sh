#!/usr/bin/env bash
# PAI PPU: preserve the submitted source/config and the A100 training contract.
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo_root}"
: "${RUN_ID:?}"
: "${CONFIG_YAML:?}"
export PATH="/opt/ac2/bin:${PATH}"
unset PYTHONPATH PYTHONHOME
export PYTHONPATH="${repo_root}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export USE_TF=0 USE_FLAX=0 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-${repo_root}/playground/Checkpoints}"
node_rank="${NODE_RANK:-${RANK:-0}}"
export PROCESSES_PER_NODE="${PROCESSES_PER_NODE:-${NPROC_PER_NODE:-16}}"
output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}/launch"
exec > >(tee -a "${output_dir}/launch/node${node_rank}.log") 2>&1
echo "$(date -Iseconds) PPU launch: source=${repo_root}, node=${node_rank}, processes=${PROCESSES_PER_NODE}"
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
python - "$constraints" <<'PYCONSTRAINTS'
import importlib.metadata as m,sys
names=['torch','torchvision','torchaudio','triton','transformers','accelerate','deepspeed']
assert 'ppu' in m.version('torch'), 'This launcher requires the PPU runtime'
with open(sys.argv[1],'w') as f:
    for name in names:
        f.write(f'{name}=={m.version(name)}\n')
PYCONSTRAINTS
if [[ "${INSTALL_DEPS:-1}" == 1 ]]; then
    python -m pip install --disable-pip-version-check -c "${constraints}" \
        -r "${repo_root}/requirements-ppu.txt" -r "${script_dir}/requirements-swanlab.txt"
    python -m pip install --no-deps --no-build-isolation -e "${repo_root}"
fi
rm -f "${constraints}"
trap - EXIT
python -m pip freeze > "${output_dir}/launch/node${node_rank}.packages.txt"
python - "${CONFIG_YAML}" "${PROCESSES_PER_NODE}" "${repo_root}" <<'PYPREFLIGHT'
import sys
from pathlib import Path
import torch,torchvision,transformers,accelerate,deepspeed,swanlab
import os
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import _auto_import_framework_modules
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
import starVLA.model.framework.VLM4A.QwenPI as qwenpi
cfg=OmegaConf.load(sys.argv[1])
assert torch.cuda.device_count()==int(sys.argv[2]), torch.cuda.device_count()
assert 'ppu' in torch.__version__, torch.__version__
assert torch.cuda.is_bf16_supported()
assert Path(qwenpi.__file__).resolve().is_relative_to(Path(sys.argv[3]).resolve())
assert cfg.framework.name=='QwenPI' and cfg.framework.qwenvl.load_weights
assert cfg.framework.qwenvl.cross_view_attention.enabled
assert cfg.framework.qwenvl.cross_view_attention.start_layer==16
assert cfg.datasets.vla_data.per_device_batch_size==32
assert cfg.trainer.gradient_accumulation_steps==1
assert cfg.trainer.pretrained_checkpoint is None and not cfg.trainer.is_resume
assert (Path(cfg.framework.qwenvl.base_vlm)/'model.safetensors').is_file()
assert Path(cfg.datasets.vla_data.normalization_statistics).is_file()
assert Path(cfg.datasets.vla_data.controller_goal_root).is_dir()
_auto_import_framework_modules()
torchvision.set_video_backend('pyav')
video=next(Path(cfg.datasets.vla_data.data_root_dir).glob('*/videos/**/*.mp4'))
frame=next(iter(torchvision.io.VideoReader(str(video),'video')))
if os.environ.get('REQUIRE_SWANLAB_ONLINE') == '1':
    assert os.environ.get('SWANLAB_MODE') == 'online'
    key = os.environ.get('SWANLAB_API_KEY')
    assert key, 'SWANLAB_API_KEY is required for this run'
    assert swanlab.login(api_key=key, relogin=True, save=False, timeout=20)
    print('SwanLab online authentication preflight passed', flush=True)
print('PPU preflight passed:',torch.__version__,'transformers',transformers.__version__,
      'devices',torch.cuda.device_count(),'micro_batch',cfg.datasets.vla_data.per_device_batch_size,
      'video_frame',tuple(frame['data'].shape),flush=True)
PYPREFLIGHT
if [[ "${STARVLA_PPU_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    exit 0
fi
if [[ "${node_rank}" == 0 ]]; then
    cp "${CONFIG_YAML}" "${output_dir}/launch/config.yaml"
    cp "${BASH_SOURCE[0]}" "${output_dir}/launch/"
fi
exec bash "${script_dir}/run_libero_train.sh"
