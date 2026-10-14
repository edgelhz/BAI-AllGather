#!/usr/bin/env bash
set -euo pipefail

ASTRA=${ASTRA:-/app/astra-sim/astra-sim}
CAPI=${CAPI:-/app/astra-sim/collectiveapi}
NS3=${NS3:-$ASTRA/extern/network_backend/ns-3}
PY=${PY:-/opt/venv/astra-sim/bin/python3}
BASE=${BASE:-$ASTRA/bai_multinode_remote_core}
ROOT=${ROOT:-$BASE/perf_star_16MiB}

CONVERTER=${CONVERTER:-/app/astra-sim/collectiveapi-046d961/chakra_converter/et_converter.py}
BIN=${BIN:-$NS3/build/scratch/ns3.42-AstraSimNetwork-default}
REMOTE=${REMOTE:-$ASTRA/examples/remote_memory/analytical/no_memory_expansion.json}

SPECS=${SPECS:-"3x4 4x4"}
SIZE=${SIZE:-16777216}
RUN_SIM=${RUN_SIM:-0}

[[ "$RUN_SIM" == 0 || "$RUN_SIM" == 1 ]] || exit 2

export PYTHONPATH="$BASE/legacy_reference:$BASE:$CAPI/msccl-tools:${PYTHONPATH:-}"

for SPEC in $SPECS; do
  CASE="$ROOT/$SPEC/${SIZE}B"
  XML="$BASE/results/$SPEC/schedule.xml"
  mkdir -p "$CASE" "$NS3/scratch/topology"

  "$PY" - "$SPEC" "$SIZE" "$CASE" "$XML" "$NS3" "$CAPI" "$ASTRA" <<'PY_CFG'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from chakra.src.third_party.utils.protolib import encodeMessage
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata, Node, AttributeProto,
    COMM_COLL_NODE, ALL_GATHER
)

spec, size, d, xml, ns3, capi, astra = sys.argv[1:]
k, m = map(int, spec.split("x"))
n, size = k * m, int(size)
d, xml, ns3, capi, astra = map(
    Path, (d, xml, ns3, capi, astra)
)

assert k >= 3 and m > 0 and size > 0

root = ET.parse(xml).getroot()
assert root.tag == "algo"
assert root.get("coll") == "allgather"
assert int(root.get("ngpus", -1)) == n
assert len(root.findall("gpu")) == n

chunks = inter = 0
for gpu in root.findall("gpu"):
    src = int(gpu.get("id"))
    for tb in gpu.findall("tb"):
        dst = int(tb.get("send", "-1"))
        if dst < 0:
            continue
        for step in tb.findall("step"):
            if step.get("type") != "s":
                continue
            cnt = int(step.get("cnt", "1"))
            chunks += cnt
            inter += cnt * (src // m != dst // m)

assert chunks == n * (n - 1)
assert inter == n * (k - 1)

# Build K local switches plus one core switch.
ref = ns3 / "scratch/topology/8_nodes_2switch_2x4_1lane_slow_inter_topology.txt"
rows = [
    s.split() for s in ref.read_text().splitlines()
    if s.strip() and not s.lstrip().startswith("#")
]

assert rows[:2] == [["10", "2", "9"], ["8", "9"]]

host = [r[2:] for r in rows[2:] if r[2] == "400Gbps"]
bridge = [r[2:] for r in rows[2:] if r[2] == "100Gbps"]

assert len(host) == 8 and len(bridge) == 1
assert len({tuple(x) for x in host}) == 1

core = n + k
links = [
    f"{r} {n + r // m} " + " ".join(host[0])
    for r in range(n)
]
links += [
    f"{n + j} {core} " + " ".join(bridge[0])
    for j in range(k)
]

top = ns3 / "scratch/topology" / (
    f"bai_multinode_star_{spec}_400_100.txt"
)

toptext = "\n".join([
    f"{n+k+1} {k+1} {len(links)}",
    " ".join(str(x) for x in range(n, core+1))
] + links) + "\n"

if top.exists():
    assert top.read_text() == toptext
else:
    top.write_text(toptext)

logical = json.loads((
    astra / "examples/network/ns3/sample_8nodes_1D.json"
).read_text())
logical["logical-dims"] = [str(n)]
(d / "logical.json").write_text(
    json.dumps(logical, indent=2) + "\n"
)

system = json.loads((
    capi / "runs/fullmesh_ring_priority_v2_matched_20260919_083537/16MiB/priority_inter_v2.json"
).read_text())

assert len(system["all-gather-implementation-custom"]) == 1
system["all-gather-implementation-custom"] = [
    str(d / "et/custom_allgather")
]
(d / "system.json").write_text(
    json.dumps(system, indent=2) + "\n"
)

netref = ns3 / "scratch/config/config_2node_4x4_1lane_slow_inter.txt"
new = []
seen = set()

for line in netref.read_text().splitlines():
    tok = line.split()
    if tok and not line.lstrip().startswith("#") and tok[0] in (
        "TOPOLOGY_FILE", "FCT_OUTPUT_FILE", "ENABLE_TRACE",
        "TRACE_OUTPUT_FILE", "PFC_OUTPUT_FILE", "QLEN_MON_FILE"
    ):
        key = tok[0]
        assert key not in seen
        seen.add(key)
        value = {
            "TOPOLOGY_FILE": "../../scratch/topology/" + top.name,
            "FCT_OUTPUT_FILE": str(d / "fct.txt"),
            "ENABLE_TRACE": "0",
            "TRACE_OUTPUT_FILE": "/dev/null",
            "PFC_OUTPUT_FILE": "/dev/null",
            "QLEN_MON_FILE": "/dev/null",
        }[key]
        line = key + " " + value
    new.append(line)

assert {"TOPOLOGY_FILE", "FCT_OUTPUT_FILE"} <= seen
(d / "network.txt").write_text("\n".join(new) + "\n")

w = d / "workload"
w.mkdir(exist_ok=True)

for rank in range(n):
    node = Node(id=1, name="All-Gather", type=COMM_COLL_NODE)
    node.attr.append(AttributeProto(name="is_cpu_op", bool_val=False))
    node.attr.append(AttributeProto(name="comm_type", int64_val=ALL_GATHER))
    node.attr.append(AttributeProto(name="comm_size", int64_val=size))

    with (w / f"all_gather.{rank}.et").open("wb") as f:
        encodeMessage(f, GlobalMetadata(version="0.0.4"))
        encodeMessage(f, node)

print(
    f"CONFIG_PASS spec={spec} ranks={n} "
    f"total={chunks} inter={inter} "
    f"chunk_bytes={size} topology={top}",
    flush=True
)
PY_CFG

  mkdir -p "$CASE/et"

  "$PY" "$CONVERTER" \
    --input_filename "$XML" \
    --output_filename "$CASE/et/custom_allgather" \
    --coll_size "$SIZE" \
    --collective allgather \
    --log_filename "$CASE/converter.log" \
    > "$CASE/converter.stdout.log" 2>&1 || {
      tail -n 40 "$CASE/converter.stdout.log"
      exit 1
    }

  N=$(( ${SPEC%x*} * ${SPEC#*x} ))

  for ((i=0; i<N; i++)); do
    test -s "$CASE/et/custom_allgather.$i.et"
    test -s "$CASE/workload/all_gather.$i.et"
  done

  echo "ET_PASS spec=$SPEC ranks=$N"

  if [[ "$RUN_SIM" == 1 ]]; then
    rm -f "$CASE/fct.txt"

    (
      cd "$NS3/build/scratch"
      timeout "${TIMEOUT_SECONDS:-3600}s" "$BIN" \
        --workload-configuration="$CASE/workload/all_gather" \
        --system-configuration="$CASE/system.json" \
        --network-configuration="$CASE/network.txt" \
        --remote-memory-configuration="$REMOTE" \
        --logical-topology-configuration="$CASE/logical.json" \
        --comm-group-configuration=empty
    ) > "$CASE/run.log" 2>&1 || {
      tail -n 50 "$CASE/run.log"
      exit 1
    }

    "$PY" - "$CASE" "$N" <<'PY_RESULT'
import json
import re
import sys
from pathlib import Path

case, n = Path(sys.argv[1]), int(sys.argv[2])
text = (case / "run.log").read_text(errors="replace")
found = re.findall(
    r"sys\[(\d+)\] finished,\s*(\d+) cycles",
    text
)

assert len(found) == n
assert {int(i) for i, _ in found} == set(range(n))

fct = case / "fct.txt"
assert fct.is_file() and fct.stat().st_size > 0

cycles = max(int(t) for _, t in found)
flows = sum(
    bool(line.strip())
    for line in fct.read_text().splitlines()
)

result = {
    "ranks": n,
    "cycles": cycles,
    "fct_flows": flows
}

(case / "result.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("SIM_PASS", json.dumps(result))
PY_RESULT
  fi
done

echo "DONE output=$ROOT"
