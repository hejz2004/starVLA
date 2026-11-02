#!/usr/bin/env bash
set -Eeuo pipefail
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export STARVLA_DIR
exec bash "$STARVLA_DIR/examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/run_cluster.sh" "$@"
