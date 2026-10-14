# BAI-AllGather — Reproducibility Artifacts

BAI-AllGather studies scheduling and dependency optimization
for hierarchical GPU AllGather after achieving minimal logical
inter-node data deliveries.

MultiNode-RemoteCore constructs parameterized schedules.
Node-count and GPUs-per-node experiments characterize
performance boundaries.

## Repository Contents

- `src/`: schedule generator and required support modules
- `tests/`: generator tests
- `scripts/`: experiment runners and repeatability tools
- `configs/observed/`: original simulator configurations
- `configs/topology/`: physical network topologies
- `results/`: seven experimental CSV summaries
- `reproducibility/versions.json`: source versions and hashes
- `reproducibility/patches/`: required ASTRA-sim patch
- `reproducibility/r3/`: representative repeatability evidence

## External Dependencies

ASTRA-sim, ns-3, Chakra and MSCCL-tools are external
dependencies, not included in this repository.

Recorded Git revisions:

| Component | Commit |
| --- | --- |
| ASTRA-sim | 28f18ea0d58a96fd812e49b71bdac330a1e3c965 |
| ns-3 backend | 40d9153caefb9ac1a6ec316aedad3068cfc0fbc9 |
| MSCCL-tools | 8d735800a87bf4426816e2c10a9834a42ab64fe6 |

Use `reproducibility/versions.json` for recorded hashes.

The ASTRA-sim custom executor requires the supplied patch.
Apply it to a clean checkout of the pinned ASTRA-sim revision
before rebuilding:

```bash
export BAI_REPO=/path/to/BAI-AllGather-release
export ASTRA=/path/to/astra-sim

git -C "$ASTRA" apply --check   "$BAI_REPO/reproducibility/patches/astra_custom_collectives.patch"

git -C "$ASTRA" apply   "$BAI_REPO/reproducibility/patches/astra_custom_collectives.patch"
```

Do not apply the patch to an already modified worktree.

## Reproduce Schedule Generation

Install MSCCL-tools and its Python dependencies first.

```bash
cd /path/to/BAI-AllGather-release

export BAI_REPO="$PWD"
export MSCCL_TOOLS=/path/to/msccl-tools
export PY=python3

bash scripts/smoke_schedule.sh
```

Expected 3x4 output:

- 11 scheduling groups
- 132 total directed chunk deliveries
- 24 inter-node deliveries
- 108 intra-node deliveries

This validates schedule generation, not full simulator
reproduction on a clean machine.

## Reproduce Network Experiments

Original experiment scripts use Docker paths including
`/app/astra-sim/`, `/opt/venv/` and `/tmp/`.

Check `reproducibility/path_audit.md` first.

Some scripts support environment variables such as:

- ASTRA
- CAPI
- PY
- CONVERTER
- BAI_RUNNER
- HIER_RUNNER

Other paths require manual adaptation.

The historical BASE variable often represents the
experimental workspace rather than this repository.

Configuration files in `configs/observed/` are original
snapshots and should not be modified in place.

Default link configuration:

- GPU access: 400 Gbps, 250 ns per link
- Inter-node: 100 Gbps, 1500 ns per link
- Packet payload: 1000 bytes
- Congestion control: CC_MODE=12

The specific implementation mapping of CC_MODE=12
requires confirmation against the simulator source.

The two-node configuration has one shared inter-switch link.
Multi-node configurations have separate uplinks to a
central switch.

## Results and Reproducibility

Results cover message-size sensitivity, bandwidth,
scalability and controlled Deferred-Inter ablation.

The 21-point multi-node message-size CSV was reconstructed
from previously completed experiments.

A representative 3x4, 16-MiB repeatability run reproduced
14,128,570 cycles and all 132 normalized FCT records.

This does not imply zero variance for all configurations.

End-to-end simulation from a new clean machine has not
yet been verified.

Verify release file integrity:

```bash
sha256sum -c reproducibility/checksums.sha256
```

Review `THIRD_PARTY.md` and
`reproducibility/release_audit.md` before publication.
