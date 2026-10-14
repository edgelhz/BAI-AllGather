#!/usr/bin/env bash
set -euo pipefail

ASTRA="${ASTRA:-/app/astra-sim/astra-sim}"
CAPI="${CAPI:-/app/astra-sim/collectiveapi}"
NS3="${NS3:-$ASTRA/extern/network_backend/ns-3}"
PY="${PY:-/opt/venv/astra-sim/bin/python3}"
CONVERTER="${CONVERTER:-/app/astra-sim/collectiveapi-046d961/chakra_converter/et_converter.py}"
BIN="${BIN:-$NS3/build/scratch/ns3.42-AstraSimNetwork-default}"
REMOTE="${REMOTE:-$ASTRA/examples/remote_memory/analytical/no_memory_expansion.json}"
LOGICAL="${LOGICAL:-$ASTRA/examples/network/ns3/sample_8nodes_1D.json}"
BASE="${BASE:-$ASTRA/bai_multinode_remote_core}"
FROZEN="${FROZEN:-$CAPI/runs/fullmesh_ring_priority_v2_matched_20260919_083537}"
NETWORK="${NETWORK:-$NS3/scratch/config/config_2node_4x4_1lane_slow_inter.txt}"
TOPO="${TOPO:-$NS3/scratch/topology/8_nodes_2switch_2x4_1lane_slow_inter_topology.txt}"
OUT="${OUT:-$BASE/paper_current_2x4_400_100}"
SIZES="${SIZES:-4KiB 16KiB 64KiB 256KiB 1MiB 4MiB 16MiB}"
RUN_SIM="${RUN_SIM:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
VERIFY_PAPER_HASH="${VERIFY_PAPER_HASH:-1}"

[[ "$RUN_SIM" == 0 || "$RUN_SIM" == 1 ]] || {
    echo "RUN_SIM must be 0 or 1" >&2
    exit 2
}

[[ "$VERIFY_PAPER_HASH" == 0 || "$VERIFY_PAPER_HASH" == 1 ]] || exit 2

for f in \
    "$PY" "$CONVERTER" "$BIN" "$REMOTE" "$LOGICAL" \
    "$NETWORK" "$TOPO" "$BASE/bai_multinode_remote_core.py"; do
    [[ -s "$f" ]] || {
        echo "ERROR: missing $f" >&2
        exit 1
    }
done

# Verify the original paper topology and network.
if [[ "$VERIFY_PAPER_HASH" == 1 ]]; then
    echo 'f63348c7f20702606e4b7e4ec2daa25a95112d00f6b5f51958a3c84e2f7afbd3' \
        " $TOPO" | sha256sum -c -

    echo '8fd7b80ebb8ddcb59d2cf8d6edc57b827f6cebc7f42f0355be280dbbc1a67fc7' \
        " $NETWORK" | sha256sum -c -
fi

mkdir -p "$OUT"

export PYTHONPATH="$BASE/legacy_reference:$BASE:$CAPI/msccl-tools:${PYTHONPATH:-}"

DSL="$OUT/current_multinode_remote_core_2x4.py"
XML="$OUT/current_multinode_remote_core_2x4.xml"

# Generate from the CURRENT MultiNode-RemoteCore source.
if [[ ! -s "$XML" || ! -s "$DSL" ]]; then
    "$PY" "$BASE/bai_multinode_remote_core.py" \
        --nodes 2 \
        --gpus-per-node 4 \
        --mode core \
        --max-local-rounds 1 \
        --freeze-dsl "$DSL" \
        --report-json "$OUT/plan.json"

    "$PY" "$DSL" 1 --protocol Simple > "$XML"
fi

# Audit the actual compiled XML, not merely planner output.
"$PY" - "$XML" <<'PY_AUDIT'
from collections import Counter
from xml.etree import ElementTree as ET
import sys

root = ET.parse(sys.argv[1]).getroot()

assert root.tag == "algo"
assert root.get("coll") == "allgather"
assert int(root.get("ngpus", -1)) == 8

gpus = root.findall("gpu")
assert {int(g.get("id")) for g in gpus} == set(range(8))

moves = Counter()

for gpu in gpus:
    src = int(gpu.get("id"))

    for tb in gpu.findall("tb"):
        dst = int(tb.get("send", "-1"))
        if dst < 0:
            continue

        assert 0 <= dst < 8 and dst != src

        for step in tb.findall("step"):
            if step.get("type") != "s":
                continue

            count = int(step.get("cnt", "1"))
            a = int(step.get("srcoff"))
            b = int(step.get("dstoff"))

            assert a == b
            assert count > 0
            assert 0 <= a and a + count <= 8

            for j in range(count):
                moves[src, dst, a + j] += 1

assert sum(moves.values()) == 56
assert len(moves) == 56

expected = {
    (dst, chunk)
    for dst in range(8)
    for chunk in range(8)
    if dst != chunk
}

assert {(d, c) for s, d, c in moves} == expected

inter = sum(
    v for (s, d, c), v in moves.items()
    if s // 4 != d // 4
)

assert inter == 8

print("CURRENT_2x4_XML_AUDIT=PASS total=56 inter=8")
PY_AUDIT

# Seven sizes used in the manuscript.
for LABEL in $SIZES; do

    case "$LABEL" in
        4KiB)   BYTES=4096 ;;
        16KiB)  BYTES=16384 ;;
        64KiB)  BYTES=65536 ;;
        256KiB) BYTES=262144 ;;
        1MiB)   BYTES=1048576 ;;
        4MiB)   BYTES=4194304 ;;
        16MiB)  BYTES=16777216 ;;
        *)
            echo "ERROR: invalid size $LABEL" >&2
            exit 2 ;;
    esac

    OLD="$FROZEN/$LABEL"
    CASE="$OUT/$LABEL"

    mkdir -p \
        "$CASE/current/et" \
        "$CASE/hier" \
        "$CASE/native"

    for f in \
        "$OLD/priority_inter_v2.json" \
        "$OLD/hierarchical_ring.json" \
        "$OLD/native_ring.json"; do

        [[ -s "$f" ]] || {
            echo "ERROR: missing frozen system $f" >&2
            exit 1
        }
    done

    for ((rank=0; rank<8; rank++)); do
        for f in \
            "$OLD/workload/all_gather.$rank.et" \
            "$OLD/hier_et/custom_allgather.$rank.et"; do

            [[ -s "$f" ]] || {
                echo "ERROR: missing $f" >&2
                exit 1
            }
        done
    done

    CUR_ET="$CASE/current/et/custom_allgather"

    if [[ ! -s "$CUR_ET.0.et" ]]; then
        "$PY" "$CONVERTER" \
            --input_filename "$XML" \
            --output_filename "$CUR_ET" \
            --coll_size "$BYTES" \
            --collective allgather \
            --log_filename "$CASE/current/converter.log" \
            > "$CASE/current/converter.stdout.log" 2>&1 || {

            tail -n 60 "$CASE/current/converter.stdout.log" >&2
            exit 1
        }
    fi

    for ((rank=0; rank<8; rank++)); do
        [[ -s "$CUR_ET.$rank.et" ]] || {
            echo "ERROR: missing ET for rank $rank" >&2
            exit 1
        }
    done

    # Clone exactly matched system/network settings.
    "$PY" - "$OLD" "$CASE" "$CUR_ET" "$NETWORK" <<'PY_CONFIG'
from pathlib import Path
import json
import sys

old, case, prefix, network = map(Path, sys.argv[1:])

a = json.loads((old / "priority_inter_v2.json").read_text())
b = json.loads((old / "hierarchical_ring.json").read_text())

key = "all-gather-implementation-custom"

assert isinstance(a[key], list) and len(a[key]) == 1
assert isinstance(b[key], list) and len(b[key]) == 1
assert set(a) == set(b)

assert {k for k in a if a[k] != b[k]} == {key}, (
    "Original system configs differ beyond custom algorithm"
)

a[key] = [str(prefix)]

(case / "current/system.json").write_text(
    json.dumps(a, indent=2, sort_keys=True) + "\n"
)

lines = network.read_text().splitlines()

for role in ("current", "hier", "native"):
    dest = case / role
    updated = []
    hits = 0

    for line in lines:
        parts = line.split()

        if (
            parts
            and not line.lstrip().startswith("#")
            and parts[0] == "FCT_OUTPUT_FILE"
        ):
            hits += 1
            line = "FCT_OUTPUT_FILE " + str(dest / "fct.txt")

        updated.append(line)

    assert hits == 1

    (dest / "network.txt").write_text(
        "\n".join(updated) + "\n"
    )

print("MATCHED_SYSTEM_CONFIG=PASS")
PY_CONFIG

    # Record exact experiment inputs and reject stale caches.
    (
        sha256sum \
            "$BASE/bai_multinode_remote_core.py" \
            "$DSL" "$XML" "$CONVERTER" \
            "$NETWORK" "$TOPO" "$REMOTE" "$LOGICAL" \
            "$OLD/priority_inter_v2.json" \
            "$OLD/hierarchical_ring.json" \
            "$OLD/native_ring.json" \
            "$OLD"/workload/all_gather.*.et \
            "$OLD"/hier_et/custom_allgather.*.et \
            "$BIN"
    ) | sha256sum | awk '{print $1}' \
        > "$CASE/input.current.sha256"

    if [[ -s "$CASE/input.recorded.sha256" ]]; then
        cmp -s \
            "$CASE/input.current.sha256" \
            "$CASE/input.recorded.sha256" || {

            echo "ERROR: cached experiment inputs changed" >&2
            echo "Choose a fresh OUT directory." >&2
            exit 1
        }
    fi

    cp "$CASE/input.current.sha256" \
       "$CASE/input.recorded.sha256"

    echo "ET_PASS size=$LABEL ranks=8 bytes=$BYTES"

    if [[ "$RUN_SIM" == 0 ]]; then
        continue
    fi

    # Run three algorithms under the same experiment settings.
    for ROLE in current hier native; do

        case "$ROLE" in
            current) SYS="$CASE/current/system.json" ;;
            hier)    SYS="$OLD/hierarchical_ring.json" ;;
            native)  SYS="$OLD/native_ring.json" ;;
        esac

        DIR="$CASE/$ROLE"

        if [[ -s "$DIR/result.json" ]]; then
            echo "REUSE_RESULT size=$LABEL role=$ROLE"
            continue
        fi

        rm -f "$DIR/fct.txt"

        if ! (
            cd "$NS3/build/scratch"

            timeout "$TIMEOUT_SECONDS" "$BIN" \
                --workload-configuration="$OLD/workload/all_gather" \
                --system-configuration="$SYS" \
                --network-configuration="$DIR/network.txt" \
                --remote-memory-configuration="$REMOTE" \
                --logical-topology-configuration="$LOGICAL" \
                --comm-group-configuration=empty

        ) > "$DIR/run.log" 2>&1; then

            echo "SIM_FAILED size=$LABEL role=$ROLE" >&2
            tail -n 60 "$DIR/run.log" >&2
            exit 1
        fi

        "$PY" - "$DIR" <<'PY_RESULT'
import json
import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
text = (p / "run.log").read_text(errors="replace")

matches = re.findall(
    r"sys\[(\d+)\] finished,\s*(\d+) cycles",
    text
)

assert len(matches) == 8

by_rank = {int(r): int(c) for r, c in matches}
assert set(by_rank) == set(range(8))

fct = p / "fct.txt"
assert fct.is_file() and fct.stat().st_size > 0

flows = sum(
    bool(line.strip())
    for line in fct.read_text().splitlines()
)

assert flows > 0

if p.name in ("current", "hier"):
    assert flows == 56

result = {
    "cycles": max(by_rank.values()),
    "ranks": 8,
    "fct_flows": flows
}

(p / "result.json").write_text(
    json.dumps(result, indent=2) + "\n"
)

print("SIM_PASS", p.parent.name, p.name, result)
PY_RESULT

    done
done

if [[ "$RUN_SIM" == 1 ]]; then

    "$PY" - "$OUT" "$SIZES" <<'PY_SUMMARY'
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

out = Path(sys.argv[1])
labels = sys.argv[2].split()
rows = []

print(
    "SIZE        CURRENT_CYCLES    HIER_CYCLES "
    "   NATIVE_CYCLES   REDUCTION_%"
)

for label in labels:
    def cycles(role):
        path = out / label / role / "result.json"
        return int(json.loads(path.read_text())["cycles"])

    current = cycles("current")
    hier = cycles("hier")
    native = cycles("native")

    reduction = (
        (Decimal(hier) - Decimal(current))
        * 100 / Decimal(hier)
    )

    assert abs(
        Decimal(hier) * (1 - reduction / 100)
        - Decimal(current)
    ) < Decimal("0.000001")

    rows.append([
        label, current, hier, native, str(reduction)
    ])

    print(
        f"{label:8s} {current:15,d} {hier:14,d} "
        f"{native:15,d} {reduction:11.4f}"
    )

csv_file = out / "tableIV_current_vs_hier.csv"

with csv_file.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "size",
        "current_multinode_remote_core_cycles",
        "hierarchical_ring_cycles",
        "native_ring_cycles",
        "current_time_reduction_pct"
    ])
    writer.writerows(rows)

print("CSV=" + str(csv_file))
PY_SUMMARY

else
    echo "PREFLIGHT_PASS: XML and all ETs ready."
fi

echo "DONE OUT=$OUT"
