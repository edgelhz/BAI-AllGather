#!/usr/bin/env bash
set -euo pipefail

ASTRA="${ASTRA:-/app/astra-sim/astra-sim}"
CAPI="${CAPI:-/app/astra-sim/collectiveapi}"
NS3="${NS3:-$ASTRA/extern/network_backend/ns-3}"
PY="${PY:-/opt/venv/astra-sim/bin/python3}"
BIN="${BIN:-$NS3/build/scratch/ns3.42-AstraSimNetwork-default}"
CONVERTER="${CONVERTER:-/app/astra-sim/collectiveapi-046d961/chakra_converter/et_converter.py}"
REMOTE="${REMOTE:-$ASTRA/examples/remote_memory/analytical/no_memory_expansion.json}"
BASE="${BASE:-$ASTRA/bai_multinode_remote_core}"
ROOT="${ROOT:-$BASE/perf_star_16MiB}"
SIZE=16777216
RUN_SIM="${RUN_SIM:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

[[ "$RUN_SIM" == 0 || "$RUN_SIM" == 1 ]] || {
  echo 'RUN_SIM must be 0 or 1' >&2
  exit 2
}

for f in "$PY" "$CONVERTER" "$REMOTE"; do
  [[ -s "$f" ]] || {
    echo "Missing file: $f" >&2
    exit 1
  }
done

if [[ "$RUN_SIM" == 1 ]]; then
  [[ -s "$BIN" ]] || {
    echo "Missing simulator: $BIN" >&2
    exit 1
  }
fi

if [[ -z "${OUT:-}" ]]; then
  OUT="$(mktemp -d "$BASE/hier_ring_matched_XXXXXXXX")"
fi

mkdir -p "$OUT"
GEN="$OUT/hierarchical_ring_multinode.py"
export PYTHONPATH="$CAPI/msccl-tools:${PYTHONPATH:-}"

echo "RESULT_DIR=$OUT"

# Multi-node extension of the historical 2-node baseline:
# 1. Local ring of original chunks.
# 2. Send each original chunk to each other node,
#    using matching local GPU indices.
# 3. Local rings for all received remote chunks.
#
# These are construction phases, not global barriers.

cat > "$GEN" <<'PY_GEN'
#!/usr/bin/env python3
import argparse
import sys
from collections import Counter


def make_phases(k, m):
    phases = []

    # Stage 1: local original-data ring.
    for hop in range(m - 1):
        phases.append([
            (
                node*m + j,
                node*m + (j+1) % m,
                node*m + (j-hop) % m
            )
            for node in range(k)
            for j in range(m)
        ])

    # Stage 2: original chunks cross to other nodes.
    phases.append([
        (
            src_node*m + j,
            dst_node*m + j,
            src_node*m + j
        )
        for src_node in range(k)
        for dst_node in range(k)
        if src_node != dst_node
        for j in range(m)
    ])

    # Stage 3: local rings of remote-origin chunks.
    for hop in range(m - 1):
        phases.append([
            (
                dst_node*m + j,
                dst_node*m + (j+1) % m,
                src_node*m + (j-hop) % m
            )
            for dst_node in range(k)
            for src_node in range(k)
            if src_node != dst_node
            for j in range(m)
        ])

    return phases


def audit(phases, k, m):
    n = k*m
    held = {r: {r} for r in range(n)}
    moves = Counter()

    for phase_no, phase in enumerate(phases):
        before = {
            r: set(chunks)
            for r, chunks in held.items()
        }
        arrivals = {r: set() for r in range(n)}

        for src, dst, c in phase:
            assert 0 <= src < n
            assert 0 <= dst < n
            assert src != dst
            assert 0 <= c < n

            assert c in before[src], (
                phase_no, src, dst, c
            )
            assert c not in before[dst], (
                phase_no, dst, c
            )
            assert c not in arrivals[dst]

            arrivals[dst].add(c)
            moves[src, dst, c] += 1

        for dst in range(n):
            held[dst].update(arrivals[dst])

    assert all(
        held[r] == set(range(n))
        for r in range(n)
    )

    assert len(phases) == 2*m - 1
    assert sum(moves.values()) == n*(n-1)
    assert len(moves) == sum(moves.values())

    inter = sum(
        v for (s,d,c),v in moves.items()
        if s//m != d//m
    )

    assert inter == n*(k-1)

    assert {
        (d,c) for s,d,c in moves
    } == {
        (d,c)
        for d in range(n)
        for c in range(n)
        if c != d
    }

    return moves, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', type=int, required=True)
    ap.add_argument(
        '--gpus-per-node', type=int, required=True
    )
    ap.add_argument(
        '--verify-only', action='store_true'
    )
    a = ap.parse_args()

    k, m = a.nodes, a.gpus_per_node

    if k < 2 or m < 2:
        ap.error(
            'nodes and gpus-per-node must both be >= 2'
        )

    phases = make_phases(k, m)
    moves, inter = audit(phases, k, m)

    print(
        f'PLAN_PASS nodes={k} gpus_per_node={m} '
        f'phases={len(phases)} '
        f'total={sum(moves.values())} '
        f'inter={inter} '
        f'intra={sum(moves.values())-inter}',
        file=sys.stderr
    )

    if a.verify_only:
        return

    from msccl.language import (
        MSCCLProgram, Buffer, Check, XML, chunk
    )
    from msccl.language.collectives import AllGather
    from msccl.topologies import fully_connected

    n = k*m

    p = MSCCLProgram(
        f'hierarchical_ring_multinode_{k}x{m}',
        fully_connected(n),
        AllGather(n, 1, True),
        1,
        protocol='Simple',
        dependence_nop=True,
        instr_fusion=False
    )

    with p:
        for phase in phases:
            for src, dst, c in phase:
                ref = chunk(
                    src, Buffer.output, c, 1
                )
                if ref is None:
                    raise RuntimeError(
                        f'No chunk {c} on rank {src}'
                    )
                ref.copy(
                    dst, Buffer.output, c, 1, ch=0
                )

        Check()
        XML()


if __name__ == '__main__':
    main()
PY_GEN

for SPEC in 3x4 4x4; do

  K="${SPEC%x*}"
  M="${SPEC#*x}"
  N=$((K * M))

  CASE="$OUT/$SPEC/${SIZE}B"
  BAI="$ROOT/$SPEC/${SIZE}B"
  XML="$CASE/hierarchical_ring.xml"
  ET="$CASE/et/custom_allgather"

  mkdir -p "$CASE/et"

  echo "===== $SPEC: CHECK REFERENCE ====="

  for f in \
    "$BASE/results/$SPEC/schedule.xml" \
    "$BAI/system.json" \
    "$BAI/network.txt" \
    "$BAI/logical.json" \
    "$BAI/result.json"; do

    if [[ "$RUN_SIM" == 0 && "$f" == "$BAI/result.json" ]]; then
      continue
    fi

    [[ -s "$f" ]] || {
      echo "Missing reference: $f" >&2
      exit 1
    }
  done

  for ((r=0; r<N; r++)); do
    [[ -s "$BAI/workload/all_gather.$r.et" ]] || {
      echo "Missing workload ET: rank=$r" >&2
      exit 1
    }
  done

  echo "===== $SPEC: GENERATE RING XML ====="

  "$PY" "$GEN" \
    --nodes "$K" \
    --gpus-per-node "$M" \
    --verify-only \
    2>&1 | tee "$CASE/plan_audit.txt"

  if ! "$PY" "$GEN" \
      --nodes "$K" \
      --gpus-per-node "$M" \
      > "$XML" 2> "$CASE/compiler.log"; then

    tail -n 60 "$CASE/compiler.log" >&2
    exit 1
  fi

  # Verify compiled transfers, compare BAI cut volume,
  # and clone the exact matched simulation settings.

  "$PY" - \
    "$GEN" \
    "$XML" \
    "$BASE/results/$SPEC/schedule.xml" \
    "$BAI/system.json" \
    "$CASE/system.json" \
    "$BAI/network.txt" \
    "$CASE/network.txt" \
    "$CASE/fct.txt" \
    "$BAI/result.json" \
    "$K" "$M" <<'PY_MATCH'
import json
import runpy
import sys
from pathlib import Path
from collections import Counter
from xml.etree import ElementTree as ET

(
    gen, xml, bai_xml, oldsys, newsys,
    oldnet, newnet, fct, result
) = map(Path, sys.argv[1:10])

k, m = map(int, sys.argv[10:])
n = k*m

planned = Counter(
    (s,d,c)
    for phase in runpy.run_path(
        str(gen)
    )['make_phases'](k,m)
    for s,d,c in phase
)


def audit_xml(p):
    root = ET.parse(p).getroot()

    assert root.tag == 'algo'
    assert root.get('coll') == 'allgather'
    assert int(root.get('ngpus', -1)) == n

    gpus = root.findall('gpu')

    assert {
        int(g.get('id')) for g in gpus
    } == set(range(n))

    got = Counter()

    for g in gpus:
        s = int(g.get('id'))

        for tb in g.findall('tb'):
            d = int(tb.get('send', '-1'))

            if d < 0:
                continue

            assert 0 <= d < n and s != d

            for st in tb.findall('step'):
                if st.get('type') != 's':
                    continue

                cnt = int(st.get('cnt', '1'))
                a = int(st.get('srcoff'))
                b = int(st.get('dstoff'))

                assert cnt > 0 and a == b

                for j in range(cnt):
                    got[s,d,a+j] += 1

    assert sum(got.values()) == n*(n-1)
    assert len(got) == sum(got.values())

    inter = sum(
        v for (s,d,c),v in got.items()
        if s//m != d//m
    )

    assert inter == n*(k-1)

    assert {
        (d,c) for s,d,c in got
    } == {
        (d,c)
        for d in range(n)
        for c in range(n)
        if c != d
    }

    return got


ring = audit_xml(xml)

assert ring == planned, (
    'Ring XML differs from verified phase plan'
)

audit_xml(bai_xml)

import os

if os.environ.get('RUN_SIM', '1') == '1':
    ref = json.loads(result.read_text())
    assert int(ref['ranks']) == n
    assert int(ref['cycles']) > 0
    assert int(ref['fct_flows']) == n*(n-1)

source = json.loads(oldsys.read_text())
key = 'all-gather-implementation-custom'

assert isinstance(source[key], list)
assert len(source[key]) == 1

clone = dict(source)
clone[key] = [
    str(newsys.parent / 'et/custom_allgather')
]

assert {
    x for x in source
    if source[x] != clone[x]
} == {key}

newsys.write_text(
    json.dumps(clone, indent=2) + '\n'
)

old_lines = oldnet.read_text().splitlines()
new_lines = []

fct_count = 0
top_count = 0

for line in old_lines:
    parts = line.split()

    if parts and not line.lstrip().startswith('#'):
        if parts[0] == 'FCT_OUTPUT_FILE':
            fct_count += 1
            line = 'FCT_OUTPUT_FILE ' + str(fct)

        if parts[0] == 'TOPOLOGY_FILE':
            top_count += 1
            assert (
                f'bai_multinode_star_{k}x{m}_400_100.txt'
                in line
            )

    new_lines.append(line)

assert fct_count == 1
assert top_count == 1

newnet.write_text(
    '\n'.join(new_lines) + '\n'
)

assert [
    x for x in new_lines
    if not x.startswith('FCT_OUTPUT_FILE ')
] == [
    x for x in old_lines
    if not x.startswith('FCT_OUTPUT_FILE ')
]

print(
    f'MATCH_PASS {k}x{m}: '
    f'total={sum(ring.values())} '
    f'inter={n*(k-1)} '
    f'workload=same '
    f'system=custom-ET-only '
    f'network=FCT-only'
)
PY_MATCH

  echo "===== $SPEC: CONVERT XML TO CHAKRA ET ====="

  if ! "$PY" "$CONVERTER" \
      --input_filename "$XML" \
      --output_filename "$ET" \
      --coll_size "$SIZE" \
      --collective allgather \
      --log_filename "$CASE/converter.log" \
      > "$CASE/converter.stdout.log" 2>&1; then

    tail -n 60 "$CASE/converter.stdout.log" >&2
    exit 1
  fi

  for ((r=0; r<N; r++)); do
    [[ -s "$ET.$r.et" ]] || {
      echo "Missing Ring ET $SPEC rank=$r" >&2
      exit 1
    }
  done

  echo "ET_PASS $SPEC ranks=$N"

  if [[ "$RUN_SIM" == 0 ]]; then
    continue
  fi

  echo "===== $SPEC: RUN ASTRA-sim + ns-3 ====="

  rm -f "$CASE/fct.txt"

  if ! (
    cd "$NS3/build/scratch"

    timeout "$TIMEOUT_SECONDS" "$BIN" \
      --workload-configuration="$BAI/workload/all_gather" \
      --system-configuration="$CASE/system.json" \
      --network-configuration="$CASE/network.txt" \
      --remote-memory-configuration="$REMOTE" \
      --logical-topology-configuration="$BAI/logical.json" \
      --comm-group-configuration=empty

  ) > "$CASE/run.log" 2>&1; then

    echo "Simulation failed for $SPEC" >&2
    tail -n 60 "$CASE/run.log" >&2
    exit 1
  fi

  "$PY" - \
    "$CASE" \
    "$BAI/result.json" \
    "$N" <<'PY_RESULT'
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

case = Path(sys.argv[1])
bai = json.loads(
    Path(sys.argv[2]).read_text()
)
n = int(sys.argv[3])

text = (case / 'run.log').read_text(
    errors='replace'
)

rows = re.findall(
    r'sys\[(\d+)\] finished,\s*(\d+) cycles',
    text
)

assert len(rows) == n, (len(rows), n)

by_rank = {
    int(r): int(t)
    for r,t in rows
}

assert len(by_rank) == n
assert set(by_rank) == set(range(n))

fct_path = case / 'fct.txt'
assert fct_path.is_file()
assert fct_path.stat().st_size > 0

flows = sum(
    bool(row.strip())
    for row in fct_path.read_text().splitlines()
)

assert flows == n*(n-1), (
    flows, n*(n-1)
)

hier = max(by_rank.values())
original = int(bai['cycles'])

assert hier > 0 and original > 0

pct = (
    (Decimal(hier)-Decimal(original))
    * 100 / Decimal(hier)
)

speedup = Decimal(hier)/Decimal(original)

assert abs(
    Decimal(hier)*(1-pct/100)
    - Decimal(original)
) < Decimal('0.000001')

output = {
    'ranks': n,
    'cycles': hier,
    'fct_flows': flows,
    'bai_cycles': original,
    'bai_time_reduction_pct': float(pct),
    'bai_speedup_vs_ring': float(speedup)
}

(case/'result.json').write_text(
    json.dumps(output, indent=2)+'\n'
)

print(
    'SIM_PASS',
    json.dumps(output, ensure_ascii=False)
)
PY_RESULT

done

# Summary for both topologies.

if [[ "$RUN_SIM" == 1 ]]; then
  "$PY" - "$OUT" "$ROOT" "$SIZE" <<'PY_SUMMARY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
size = int(sys.argv[3])

rows = []

for spec in ('3x4', '4x4'):
    bai = json.loads(
        (
            root/spec/f'{size}B'/'result.json'
        ).read_text()
    )

    hier = json.loads(
        (
            out/spec/f'{size}B'/'result.json'
        ).read_text()
    )

    rows.append([
        spec,
        bai['cycles'],
        hier['cycles'],
        f"{hier['bai_time_reduction_pct']:.4f}",
        f"{hier['bai_speedup_vs_ring']:.4f}"
    ])

with (out/'comparison.csv').open(
    'w', newline=''
) as f:
    w = csv.writer(f)
    w.writerow([
        'topology',
        'bai_cycles',
        'hier_ring_cycles',
        'bai_time_reduction_pct',
        'bai_speedup_vs_ring'
    ])
    w.writerows(rows)

print(
    'TOPOLOGY     BAI_CYCLES    '
    'HIER_RING_CYCLES    BAI_REDUCTION_%'
)

for s,b,h,p,sp in rows:
    print(
        f'{s:8s}  {b:12,d}    '
        f'{h:16,d}    {p:>12s}'
    )

print('CSV=' + str(out/'comparison.csv'))
PY_SUMMARY

else
  echo 'DRY_RUN_PASS: XML and ET verified.'
fi

echo "DONE RESULT_DIR=$OUT"
