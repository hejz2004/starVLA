#!/usr/bin/env bash
# Compatibility alias to the current dynamic anchored-goal evaluator.
set -Eeuo pipefail
BUNDLE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$BUNDLE/run_cluster.sh" "$@"
