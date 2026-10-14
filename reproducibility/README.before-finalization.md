# BAI-AllGather: reproduction artifacts

This repository stages the MultiNode-RemoteCore collective schedule generator,
experiment runners, example simulator input snapshots, and result summaries.

## Layout
- `src/`: generator and its local implementation dependencies
- `tests/`: available regression tests
- `scripts/`: experiment construction and comparison scripts
- `configs/`: snapshots of configurations used in completed experiments
- `results/`: copies of published CSV summaries (see validation report)
- `reproducibility/`: external revisions, hashes, checks, and review report

## Generate a schedule (with MSCCL-tools installed externally)
```bash
REPO=$(pwd)
export PYTHONPATH="$REPO/src:$REPO/src/legacy_reference:/path/to/msccl-tools:${PYTHONPATH:-}"
python3 "$REPO/src/bai_multinode_remote_core.py" \
  --nodes 3 --gpus-per-node 4 --max-local-rounds 1 \
  --freeze-dsl "$REPO/schedule.py" --report-json "$REPO/report.json"
```

## Reproduction caveats
The historical runner scripts and configuration snapshots may contain absolute
`/app/astra-sim` paths and must be adapted to the checkout on a new machine.
Install compatible ASTRA-sim, ns-3, Chakra, and MSCCL-tools revisions first.
See `reproducibility/environment.md` for captured versions and exact hashes.
`configs/two_node_reference` is a frozen reference, not automatically proof of
the 2x4 Current evaluation input.

Before public release: read `reproducibility/release_audit.md`, verify CSV
completeness, review licensing, secrets, local source dependencies, and
portability. No environment binaries or raw packet traces are included.
