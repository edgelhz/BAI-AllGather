#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BAI_WORKSPACE="${BAI_WORKSPACE:-$(dirname "$REPO")}"
export ASTRA="${ASTRA:-$BAI_WORKSPACE/astra-sim}"
export CAPI="${CAPI:-$BAI_WORKSPACE/collectiveapi}"
export PY="${PY:-${BAI_PY:-python3}}"
export CONVERTER="${CONVERTER:-$BAI_WORKSPACE/collectiveapi-046d961/chakra_converter/et_converter.py}"
export BASE="${BASE:-$ASTRA/bai_multinode_remote_core}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SCRIPT [ARGS]"
    exit 2
fi

SCRIPT="$1"
shift

case "$SCRIPT" in
    *"/"*|*".."*) echo "Invalid script name"; exit 2 ;;
    *.sh) exec bash "$REPO/scripts/$SCRIPT" "$@" ;;
    *.py) exec "$PY" "$REPO/scripts/$SCRIPT" "$@" ;;
    *) echo "Unsupported script type"; exit 2 ;;
esac
