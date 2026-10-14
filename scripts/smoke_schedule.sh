#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MSCCL_TOOLS:?Set MSCCL_TOOLS}"

PY="${PY:-python3}"
OUT="${BAI_SMOKE_OUT:-${TMPDIR:-/tmp}/bai-smoke-3x4}"

mkdir -p "$OUT"

export PYTHONPATH="$REPO/src:$REPO/src/legacy_reference:$MSCCL_TOOLS${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$REPO/src/bai_multinode_remote_core.py"   --nodes 3   --gpus-per-node 4   --max-local-rounds 1   --freeze-dsl "$OUT/schedule.py"   --report-json "$OUT/report.json"

"$PY" - "$OUT/report.json" <<'CHECK'
import json
import sys

v = json.load(open(sys.argv[1], encoding="utf-8"))

for key, n in [
    ("total", 132),
    ("inter", 24),
    ("intra", 108),
    ("nodes", 3),
    ("gpus_per_node", 4),
]:
    assert int(v[key]) == n

print("BAI_SCHEDULE_SMOKE=PASS")
CHECK
