#!/usr/bin/env bash
# PAI job entry: explicitly install missing dependencies, then run the prepared evaluator.
set -Eeuo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STARVLA_DIR="${STARVLA_DIR:-$(cd "$BUNDLE/../../../../.." && pwd)}"
export STARVLA_PYTHON="${STARVLA_PYTHON:-/opt/ac2/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-$STARVLA_PYTHON}"
export LIBERO_HOME="${LIBERO_HOME:-$STARVLA_DIR/playground/Artifacts/libero_plus_shared/source}"
# Fresh DLC containers use their installed packages, never the DSW's old overlay.
export LIBERO_SITE_PACKAGES="${LIBERO_SITE_PACKAGES:-$("$LIBERO_PYTHON" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
export RUN_ID="${RUN_ID:-libero_plus_goal_binary_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-$STARVLA_DIR/playground/Evaluation/$RUN_ID}"
bootstrap_log="${BOOTSTRAP_LOG:-${RUN_ROOT}.bootstrap.log}"
mkdir -p "$(dirname "$bootstrap_log")"
exec > >(tee -a "$bootstrap_log") 2>&1
cd "$STARVLA_DIR"
# Keep vendor Torch / torchvision / transformers. Public packages must not replace PPU builds.
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
"$STARVLA_PYTHON" - "$constraints" <<'PYCODE'
import importlib.metadata as m,sys
names=['torch','torchvision','torchaudio','triton','transformers','accelerate','deepspeed']
with open(sys.argv[1],'w') as f:
 for n in names:
  try:f.write(n+'=='+m.version(n)+'\n')
  except m.PackageNotFoundError:pass
if 'ppu' not in m.version('torch'):raise SystemExit('Use the verified PPU image/runtime')
PYCODE
if [[ "${INSTALL_DEPS:-1}" == 1 ]]; then
 if ! "$LIBERO_PYTHON" -c 'import ctypes;ctypes.CDLL("libOSMesa.so.8");ctypes.CDLL("libMagickWand-6.Q16.so.7")' || ! command -v ffmpeg >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libosmesa6 libgl1 libmagickwand-dev ffmpeg
 fi
 "$STARVLA_PYTHON" -m pip install -c "$constraints" -r "$BUNDLE/requirements.txt"
 "$STARVLA_PYTHON" -m pip install --no-deps --no-build-isolation -e "$STARVLA_DIR"
fi
rm -f "$constraints"
trap - EXIT
exec bash "$BUNDLE/run_cluster.sh" "$@"
