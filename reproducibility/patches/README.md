# ASTRA-sim custom-collective executor patch

Upstream ASTRA-sim commit:
28f18ea0d58a96fd812e49b71bdac330a1e3c965

The patch modifies:
- CustomAlgorithm.cc
- CustomAlgorithm.hh

Changes:
1. Optional nanosecond-resolution duration_ns support
   for Chakra compute nodes.
2. Execution-depth protection against premature
   algorithm destruction during reentrant callbacks.

Apply this patch to a CLEAN checkout of the pinned
ASTRA-sim commit before rebuilding the ns-3 backend.

Example commands from the ASTRA-sim repository root:

  git apply --check /path/to/astra_custom_collectives.patch
  git apply /path/to/astra_custom_collectives.patch

Do not run git apply against the already modified
experimental worktree.

The upstream AllReduce example-script modification
and unrelated MSCCL allreduce example changes are
not included in this patch.
