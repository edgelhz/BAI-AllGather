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


## Reproduce Network Experiments

Historical configuration snapshots are preserved unchanged. Use `scripts/materialize_sim_inputs.py` for portable runtime inputs and `scripts/run_with_repo_env.sh` for workspace-relative runner paths. Activate the corresponding Python environment before running experiments.

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


A representative 3x4, 16-MiB repeatability run reproduced
14,128,570 cycles and all 132 normalized FCT records.

This does not imply zero variance for all configurations.


Verify release file integrity:

```bash
sha256sum -c reproducibility/checksums.sha256
```

Review `THIRD_PARTY.md` and
`reproducibility/release_audit.md` before publication.

## Reproducibility: 2026 Revision

### Research development

BAI-AllGather was initially developed and analyzed on a
two-node hierarchical GPU network. Its cut-aware remote
seeding and dependency-ready interleaving principles were
subsequently generalized to multi-node configurations.

The multi-node generator uses parameterized node-pair
planning, local forwarding, and remaining-delivery
completion while preserving the original scheduling idea.

See [Algorithm Evolution](reproducibility/ALGORITHM_EVOLUTION.md).

### Complete 2x4 simulator configuration

The source-verified reference model is documented in
[R1 Execution Model](reproducibility/r1/EXECUTION_MODEL.md).

Original experiment inputs:

- `configs/observed/r1_2x4_network.txt`
- `configs/observed/r1_2x4_system.json`
- `configs/topology/r1_2x4_topology.txt`

The physical links are full duplex: 400 Gbps per direction
for GPU access and 100 Gbps per direction for the inter-switch
connection. Multiple active RDMA queue pairs share NIC egress
resources. Custom SEND/RECV operations use zero front-end delay.

Verify the archived inputs:

```bash
sha256sum -c reproducibility/r1/inputs.sha256
```

### Portable runtime inputs

Generate network and system configuration working copies
using the current repository and ET trace locations:

```bash
python3 scripts/materialize_sim_inputs.py \
  --et-prefix "$ET_PREFIX" \
  --output-dir "./work/2x4"
```

`ET_PREFIX` identifies the generated Chakra traces with
suffixes `.0.et` through `.7.et`. The archived configuration
files remain unchanged.

### Measured offline scheduling overhead

Original server-side measurements are available in
[the R4 measurement records](reproducibility/r4/README.md).

| GPUs | Planning (ms) | Total preparation (ms) |
|---|---:|---:|
| 2x4 | 116.287 | 447.912 |
| 2x16 | 416.314 | 947.816 |
| 4x4 | 116.054 | 447.134 |
| 8x4 | 166.330 | 696.454 |

Each configuration uses one warm-up and five timed runs.
Values are medians. These are offline preparation costs,
separate from simulated communication completion times.

Run the measurement tool with:

```bash
BAI_PY=/path/to/python3 \
  bash scripts/run_offline_overhead.sh
```

The generator and converter paths can also be overridden
using `BAI_GENERATOR` and `CHAKRA_CONVERTER`.

Original samples and manifests are in `reproducibility/r4/`.
