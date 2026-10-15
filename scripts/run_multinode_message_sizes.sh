#!/usr/bin/env bash
BAI_WORKSPACE="${BAI_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
set -euo pipefail

PY="${PY:-${BAI_PY:-python3}}"
BASE="${BASE:-${BAI_WORKSPACE}/astra-sim/bai_multinode_remote_core}"
ROOT="${ROOT:-$BASE/perf_star_16MiB}"
SCAN="${SCAN:-$BASE/message_sweep_multinode_20261013}"

BAI_RUNNER="${BAI_RUNNER:-/tmp/run_multinode_rc_only.sh}"
HIER_RUNNER="${HIER_RUNNER:-/tmp/run_multinode_hier_matched.sh}"

HIER34="${HIER34:-$BASE/hier_ring_matched_stUubJh9}"
HIERMORE="${HIERMORE:-$BASE/nodecount_yC3RLZ08/hier}"

SPECS="${SPECS:-3x4 4x4 8x4}"
SIZES="${SIZES:-4KiB 256KiB 1MiB 4MiB}"
RUN_SIM="${RUN_SIM:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"

[[ "$RUN_SIM" == 0 || "$RUN_SIM" == 1 ]] || exit 2
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || exit 2

for f in "$PY" "$BAI_RUNNER" "$HIER_RUNNER"; do
    [[ -s "$f" ]] || {
        echo "MISSING: $f" >&2
        exit 1
    }
done

mkdir -p "$SCAN"

# Parameterize the working Hierarchical Ring runner.
# Do not overwrite its original 16 MiB version.

"$PY" - "$HIER_RUNNER" "$SCAN/hier_runner_sizes.sh" <<'PY_PATCH'
from pathlib import Path
import sys

src, dst = map(Path, sys.argv[1:])
s = src.read_text()

replacements = [
    (
        'SIZE=16777216',
        'SIZE="${SIZE:-16777216}"'
    ),
    (
        'for SPEC in 3x4 4x4; do',
        'for SPEC in $SPECS; do'
    ),
    (
        "for spec in ('3x4', '4x4'):",
        "for spec in __import__('os').environ['SPECS'].split():"
    ),
]

for old, new in replacements:
    if s.count(old) != 1:
        raise SystemExit(
            f"PATCH_ABORT: expected exactly one {old!r}"
        )
    s = s.replace(old, new, 1)

# The historical runner must have the earlier dry-run fix.
marker = "if os.environ.get('RUN_SIM', '1') == '1':"

if marker not in s:
    raise SystemExit(
        "PATCH_ABORT: missing previous Hier dry-run fix"
    )

if dst.exists() and dst.read_text() != s:
    raise SystemExit(
        "PATCH_ABORT: runner changed; choose a fresh SCAN"
    )

if not dst.exists():
    dst.write_text(s)

print("HIER_RUNNER_PATCH=PASS")
PY_PATCH

bash -n "$SCAN/hier_runner_sizes.sh"

# Verify that the old 16 MiB experiments exist.
# They will not be rerun.

for SPEC in $SPECS; do

    [[ "$SPEC" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]] || {
        echo "INVALID_SPEC=$SPEC" >&2
        exit 2
    }

    K="${SPEC%x*}"
    M="${SPEC#*x}"

    (( K >= 3 && M >= 2 )) || {
        echo "INVALID_SPEC=$SPEC" >&2
        exit 2
    }

    [[ -s "$BASE/results/$SPEC/schedule.xml" ]] || {
        echo "Missing BAI XML: $SPEC" >&2
        exit 1
    }

    [[ -s "$ROOT/$SPEC/16777216B/result.json" ]] || {
        echo "Missing 16MiB BAI result: $SPEC" >&2
        exit 1
    }

    H="$HIERMORE"

    if [[ "$SPEC" == 3x4 || "$SPEC" == 4x4 ]]; then
        H="$HIER34"
    fi

    [[ -s "$H/$SPEC/16777216B/result.json" ]] || {
        echo "Missing 16MiB Hier result: $SPEC" >&2
        exit 1
    }

    echo "REUSE_16MiB=$SPEC"
done

# Scan missing message sizes.

for LABEL in $SIZES; do

    case "$LABEL" in
        4KiB)   SIZE=4096 ;;
        16KiB)  SIZE=16384 ;;
        64KiB)  SIZE=65536 ;;
        256KiB) SIZE=262144 ;;
        1MiB)   SIZE=1048576 ;;
        4MiB)   SIZE=4194304 ;;
        16MiB)
            echo "SKIP_MEASURED_16MiB"
            continue ;;
        *)
            echo "INVALID_SIZE=$LABEL" >&2
            exit 2 ;;
    esac

    for SPEC in $SPECS; do

        BAI_CASE="$ROOT/$SPEC/${SIZE}B"
        HIER_CASE="$SCAN/hier/$LABEL/$SPEC/${SIZE}B"

        echo "===================================="
        echo "TOPOLOGY=$SPEC SIZE=$LABEL"
        echo "===================================="

        # Run current MultiNode-RemoteCore if missing.
        if [[ "$RUN_SIM" == 0 || ! -s "$BAI_CASE/result.json" ]]; then

            SPECS="$SPEC" \
            SIZE="$SIZE" \
            RUN_SIM="$RUN_SIM" \
            BASE="$BASE" \
            ROOT="$ROOT" \
            TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
            bash "$BAI_RUNNER"

        else
            echo "REUSE_BAI=$SPEC/$LABEL"
        fi

        # Run matched Hierarchical Ring if missing.
        if [[ "$RUN_SIM" == 0 || ! -s "$HIER_CASE/result.json" ]]; then

            SPECS="$SPEC" \
            SIZE="$SIZE" \
            RUN_SIM="$RUN_SIM" \
            BASE="$BASE" \
            ROOT="$ROOT" \
            TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
            OUT="$SCAN/hier/$LABEL" \
            bash "$SCAN/hier_runner_sizes.sh"

        else
            echo "REUSE_HIER=$SPEC/$LABEL"
        fi

    done
done

if [[ "$RUN_SIM" == 0 ]]; then
    echo "PREFLIGHT_PASS SCAN=$SCAN"
    exit 0
fi

# Merge new results with the measured 16 MiB baselines.
# Verify cycles against rank completion logs and FCT counts.

"$PY" - "$ROOT" "$SCAN" "$HIER34" "$HIERMORE" \
        "$SPECS" "$SIZES" <<'PY_SUMMARY'
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import sys

root, scan, hier34, hiermore = map(Path, sys.argv[1:5])
specs = sys.argv[5].split()
sizes = sys.argv[6].split()

labels = list(dict.fromkeys(sizes + ["16MiB"]))

byte_sizes = {
    "4KiB": 4096,
    "16KiB": 16384,
    "64KiB": 65536,
    "256KiB": 262144,
    "1MiB": 1048576,
    "4MiB": 4194304,
    "16MiB": 16777216
}

def read_result(folder, n):
    data = json.loads((folder / "result.json").read_text())
    t = int(data["cycles"])

    assert t > 0
    assert int(data["ranks"]) == n
    assert int(data["fct_flows"]) == n * (n-1)

    log = (folder / "run.log").read_text(errors="replace")

    finished = re.findall(
        r"sys\[(\d+)\] finished,\s*(\d+) cycles",
        log
    )

    assert len(finished) == n
    assert {int(r) for r, _ in finished} == set(range(n))
    assert max(int(t) for _, t in finished) == t

    flows = sum(
        bool(line.strip())
        for line in (folder / "fct.txt").read_text().splitlines()
    )

    assert flows == n * (n-1)

    return t

rows = []

for spec in specs:
    k, m = map(int, spec.split("x"))
    n = k*m

    print("\nTOPOLOGY:", spec)

    for label in labels:
        size = byte_sizes[label]

        bai = read_result(
            root / spec / f"{size}B", n
        )

        if label == "16MiB":
            hroot = (
                hier34
                if spec in ("3x4", "4x4")
                else hiermore
            )
            hdir = hroot / spec / f"{size}B"
            source = "existing"
        else:
            hdir = scan / "hier" / label / spec / f"{size}B"
            source = "new"

        hier = read_result(hdir, n)

        reduction = (
            (Decimal(hier) - Decimal(bai))
            * 100 / Decimal(hier)
        )

        assert abs(
            Decimal(hier) * (1-reduction/100) - bai
        ) < Decimal("0.000001")

        print(
            f"{label:6s} "
            f"BAI={bai:12,d} "
            f"HIER={hier:12,d} "
            f"reduction={reduction:8.3f}%"
        )

        rows.append([
            spec, label, size, bai, hier,
            f"{reduction:.6f}", source
        ])

csv_file = scan / "message_size_comparison.csv"

with csv_file.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "topology",
        "size",
        "bytes_per_rank",
        "bai_cycles",
        "hier_ring_cycles",
        "bai_reduction_pct",
        "source"
    ])
    w.writerows(rows)

print("CSV=" + str(csv_file))
PY_SUMMARY

echo "DONE SCAN=$SCAN"
