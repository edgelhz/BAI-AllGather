# Pre-push inspection

Checked source items: 39

## Blockers
- LICENSE_MISSING: choose and review a license before a public push

## Warnings
- ABSOLUTE_RUNTIME_PATHS scripts/run_multinode_rc_only.sh: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/run_multinode_hier_matched.sh: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/run_nodesweep_extension.sh: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/run_paper_current_2x4.sh: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/run_current_only_reuse_hier.py: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/run_multinode_message_sizes.sh: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/scan_bw_2x4_current.py: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/run_multinode_bandwidth_matched.py: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/run_multinode_remote_core_deferred_inter.py: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/check_astra_variation.py: review and document replacement before public release
- ABSOLUTE_RUNTIME_PATHS scripts/prepare_bai_github_release.py: review and document replacement before public release
- CHECK_SIMULATOR_WORKTREE astra: commit=28f18ea0d58a96fd812e49b71bdac330a1e3c965, tracked_changes=M astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.cc
 M astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh
 M exam
- CHECK_SIMULATOR_WORKTREE msccl: commit=8d735800a87bf4426816e2c10a9834a42ab64fe6, tracked_changes=M examples/mscclang/allreduce_a100_ring.py

## CSV rows (data rows exclude header)
- deferred_inter.csv: 3 rows, expected 3
- multi_node_bandwidth.csv: 9 rows, expected 9
- multi_node_message_sizes.csv: 21 rows, expected 21
- multi_node_scalability.csv: 5 rows, expected 5
- two_node_bandwidth.csv: 12 rows, expected 12
- two_node_gpu_scale.csv: 3 rows, expected 3
- two_node_message_sizes.csv: 7 rows, expected 7

## Review reminder
Absolute paths in config snapshots and runner scripts must
be adapted or documented for new systems. Inspect for tokens, private
data, third-party licenses, and uncommitted simulator patches.
