#!/usr/bin/env bash
set -euo pipefail
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
: "${CKPT:?Set CKPT to the model checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory, or use --resume}"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
CMD=("${STARVLA_PYTHON:-python}" -u examples/simBenchmarks/LIBERO/eval_files/parallel_eval.py
  --ckpt "${CKPT}" --output-dir "${OUTPUT_DIR}"
  --devices "${DEVICES:-0,1}" --workers-per-device "${WORKERS_PER_DEVICE:-2}"
  --base-port "${BASE_PORT:-11093}"
  --libero-home "${LIBERO_HOME:-${STARVLA_DIR}/../LIBERO}"
  --action-interface "${ACTION_INTERFACE:-anchored_goal_binary}"
  --simulation-python "${LIBERO_PYTHON:-python}"
  --policy-python "${STARVLA_PYTHON:-python}")
if [[ -n "${LIBERO_SITE_PACKAGES:-}" ]]; then
  CMD+=(--simulation-site "${LIBERO_SITE_PACKAGES}")
fi
if [[ -n "${USE_CANONICAL_FORWARD:-}" ]]; then
  [[ "${USE_CANONICAL_FORWARD}" == true || "${USE_CANONICAL_FORWARD}" == false ]] || exit 2
  CMD+=(--config-override "framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}")
fi
exec "${CMD[@]}" "$@"
