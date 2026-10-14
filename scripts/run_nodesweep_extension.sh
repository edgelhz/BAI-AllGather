#!/usr/bin/env bash
set -euo pipefail

ASTRA="${ASTRA:-/app/astra-sim/astra-sim}"
BASE="${BASE:-$ASTRA/bai_multinode_remote_core}"
ROOT="${ROOT:-$BASE/perf_star_16MiB}"
CAPI="${CAPI:-/app/astra-sim/collectiveapi}"
PY="${PY:-/opt/venv/astra-sim/bin/python3}"

SPECS="${SPECS:-5x4 6x4 8x4}"
RUN_SIM="${RUN_SIM:-1}"

BAI_RUNNER="${BAI_RUNNER:-/tmp/run_multinode_rc_only.sh}"
HIER_RUNNER="${HIER_RUNNER:-/tmp/run_multinode_hier_matched.sh}"

OLD_HIER="${OLD_HIER:-$BASE/hier_ring_matched_stUubJh9}"

[[ "$RUN_SIM" == 0 || "$RUN_SIM" == 1 ]] || exit 2

for f in "$BAI_RUNNER" "$HIER_RUNNER" \
         "$BASE/bai_multinode_remote_core.py"; do
    [[ -s "$f" ]] || {
        echo "MISSING: $f" >&2
        exit 1
    }
done

for SPEC in $SPECS; do
    [[ "$SPEC" =~ ^[1-9][0-9]*x4$ ]] || {
        echo "INVALID_SPEC=$SPEC" >&2
        exit 2
    }
    K="${SPEC%x*}"
    (( K >= 3 )) || exit 2
done

export PYTHONPATH="$BASE/legacy_reference:$BASE:$CAPI/msccl-tools:${PYTHONPATH:-}"

SWEEP="${SWEEP:-$(mktemp -d "$BASE/nodecount_XXXXXXXX")}"
mkdir -p "$SWEEP"

echo "SWEEP=$SWEEP"

# Reuse the verified hierarchical Ring runner.
# Only parameterize its topology loops.

"$PY" - "$HIER_RUNNER" "$SWEEP/hier_runner.sh" <<'PY_PATCH'
from pathlib import Path
import sys

src, dst = map(Path, sys.argv[1:])
s = src.read_text()

old_loop = "for SPEC in 3x4 4x4; do"
old_summary = "for spec in ('3x4', '4x4'):"

assert s.count(old_loop) == 1
assert s.count(old_summary) == 1

s = s.replace(
    old_loop,
    "for SPEC in $SPECS; do",
    1
)
s = s.replace(
    old_summary,
    "for spec in __import__('os').environ['SPECS'].split():",
    1
)

dst.write_text(s)
PY_PATCH

bash -n "$SWEEP/hier_runner.sh"

# Generate and audit BAI XML.

for SPEC in $SPECS; do
    K="${SPEC%x*}"
    CASE="$BASE/results/$SPEC"
    DSL="$CASE/schedule.py"
    XML="$CASE/schedule.xml"

    mkdir -p "$CASE"

    if [[ ! -s "$DSL" ]]; then
        "$PY" "$BASE/bai_multinode_remote_core.py" \
            --nodes "$K" \
            --gpus-per-node 4 \
            --max-local-rounds 1 \
            --freeze-dsl "$DSL" \
            --report-json "$CASE/report.json"
    else
        echo "REUSE_PLAN=$SPEC"
    fi

    "$PY" "$DSL" 1 --protocol Simple > "$XML"

    "$PY" - "$SPEC" "$XML" <<'PY_AUDIT'
import sys
from collections import Counter
from xml.etree import ElementTree as ET

spec, path = sys.argv[1:]
k = int(spec.split("x")[0])
m = 4
n = k * m

root = ET.parse(path).getroot()

assert root.tag == "algo"
assert root.get("coll") == "allgather"
assert int(root.get("ngpus", -1)) == n

gpus = root.findall("gpu")
assert {int(g.get("id")) for g in gpus} == set(range(n))

moves = Counter()

for gpu in gpus:
    src = int(gpu.get("id"))

    for tb in gpu.findall("tb"):
        dst = int(tb.get("send", "-1"))

        if dst < 0:
            continue

        assert dst != src and 0 <= dst < n

        for step in tb.findall("step"):
            if step.get("type") != "s":
                continue

            a = int(step.get("srcoff"))
            b = int(step.get("dstoff"))
            cnt = int(step.get("cnt", "1"))

            assert a == b
            assert cnt > 0
            assert 0 <= a and a + cnt <= n

            for offset in range(cnt):
                moves[(src, dst, a + offset)] += 1

total = sum(moves.values())
inter = sum(
    v for (s, d, c), v in moves.items()
    if s // m != d // m
)

assert total == n * (n - 1)
assert inter == n * (k - 1)
assert len(moves) == total

expected = {
    (dst, chunk)
    for dst in range(n)
    for chunk in range(n)
    if dst != chunk
}

actual = {(d, c) for s, d, c in moves}
assert actual == expected

print(
    f"XML_PASS {spec}: "
    f"ranks={n} total={total} inter={inter}"
)
PY_AUDIT
done

echo "===== RUN MULTINODE REMOTECORE ====="

BASE="$BASE" ROOT="$ROOT" SPECS="$SPECS" \
RUN_SIM="$RUN_SIM" \
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}" \
bash "$BAI_RUNNER"

echo "===== RUN MATCHED HIERARCHICAL RING ====="

BASE="$BASE" ROOT="$ROOT" SPECS="$SPECS" \
RUN_SIM="$RUN_SIM" \
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}" \
OUT="$SWEEP/hier" \
bash "$SWEEP/hier_runner.sh"

# Aggregate existing and new real simulation results.

if [[ "$RUN_SIM" == 1 ]]; then
"$PY" - "$ROOT" "$OLD_HIER" "$SWEEP/hier" \
         "$SPECS" "$SWEEP" <<'PY_RESULT'
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

root, old, new = map(Path, sys.argv[1:4])
specs = ["3x4", "4x4"] + sys.argv[4].split()
out = Path(sys.argv[5])

rows = []

print()
print(
    "Topology   GPUs    BAI cycles    "
    "Hier Ring cycles    Reduction"
)

for spec in dict.fromkeys(specs):
    source = old if spec in ("3x4", "4x4") else new

    bp = root / spec / "16777216B" / "result.json"
    hp = source / spec / "16777216B" / "result.json"

    if not bp.is_file() or not hp.is_file():
        print(f"{spec}: MISSING_RESULT")
        continue

    b = int(json.loads(bp.read_text())["cycles"])
    h = int(json.loads(hp.read_text())["cycles"])

    assert b > 0 and h > 0

    pct = (
        (Decimal(h) - Decimal(b))
        * 100 / Decimal(h)
    )

    assert abs(
        Decimal(h) * (1 - pct / 100) - Decimal(b)
    ) < Decimal("0.00000001")

    gpus = int(spec.split("x")[0]) * 4

    rows.append([
        spec, gpus, b, h, f"{pct:.6f}"
    ])

    print(
        f"{spec:8} {gpus:5d} "
        f"{b:13,d} {h:18,d} "
        f"{pct:10.4f}%"
    )

csv_file = out / "nodecount_comparison.csv"

with csv_file.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "topology",
        "gpus",
        "bai_cycles",
        "hier_ring_cycles",
        "bai_time_reduction_pct"
    ])
    writer.writerows(rows)

print("CSV=" + str(csv_file))
PY_RESULT
else
    echo "PRECHECK_COMPLETE"
fi

echo "DONE SWEEP=$SWEEP"
