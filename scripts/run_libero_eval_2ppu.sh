#!/usr/bin/env bash
# Host entry point for ppu-dev. Generic in-container entry remains available.
set -euo pipefail
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DOCKER="${DOCKER:-/etc/dsw/runtime/export_bin/docker}"
CONTAINER="${CONTAINER:-starvla-pai-check}"
: "${CKPT:?Set CKPT to the checkpoint matching the explicitly selected ACTION_INTERFACE}"
OUTPUT_DIR="${OUTPUT_DIR:-${STARVLA_DIR}/playground/Evaluation/libero_parallel_$(date +%Y%m%d_%H%M%S)}"
# Reuse the pinned simulator dependencies already verified on this machine.
LIBERO_SITE_PACKAGES="${LIBERO_SITE_PACKAGES:-${STARVLA_DIR}/playground/Artifacts/libero_sim_runtime}"
exec "${DOCKER}" exec -w "${STARVLA_DIR}" \
  -e "STARVLA_DIR=${STARVLA_DIR}" -e "CKPT=${CKPT}" -e "OUTPUT_DIR=${OUTPUT_DIR}" \
  -e "DEVICES=${DEVICES:-0,1}" -e "WORKERS_PER_DEVICE=${WORKERS_PER_DEVICE:-2}" \
  -e "ACTION_INTERFACE=${ACTION_INTERFACE:-anchored_goal_binary}" \
  -e "BASE_PORT=${BASE_PORT:-11093}" -e "LIBERO_SITE_PACKAGES=${LIBERO_SITE_PACKAGES}" \
  -e "LIBERO_HOME=${LIBERO_HOME:-${STARVLA_DIR}/../LIBERO}" \
  -e "STARVLA_PYTHON=${STARVLA_PYTHON:-/opt/ac2/bin/python}" \
  -e "LIBERO_PYTHON=${LIBERO_PYTHON:-/opt/ac2/bin/python}" \
  -e "USE_CANONICAL_FORWARD=${USE_CANONICAL_FORWARD:-}" \
  "${CONTAINER}" bash examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh "$@"
