#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${BAI_WORKSPACE:-$(dirname "$REPO")}"

export BAI_HOME="${BAI_HOME:-$WORKSPACE/astra-sim/bai_multinode_remote_core}"
export BAI_GENERATOR="${BAI_GENERATOR:-$BAI_HOME/bai_multinode_remote_core.py}"
export CHAKRA_CONVERTER="${CHAKRA_CONVERTER:-$WORKSPACE/collectiveapi-046d961/chakra_converter/et_converter.py}"
export BAI_PY="${BAI_PY:-$(command -v python3)}"

test -f "$BAI_GENERATOR"
test -f "$CHAKRA_CONVERTER"

exec "$BAI_PY" "$REPO/scripts/measure_bai_overhead_server.py"
