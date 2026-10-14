# Schedule-generation smoke test

Environment: existing experimental Docker container.
Generator: released src/bai_multinode_remote_core.py
Node count: 3
GPUs per node: 4
Maximum local matching rounds: 1

Observed output:

PLAN_AUDIT=PASS

Scheduling groups: 11
Total directed chunk deliveries: 132
Inter-node deliveries: 24
Intra-node deliveries: 108

Generated artifacts:
- schedule.py
- report.json

DSL SHA256:
8d18b83b882db49a56b220d4b9f01bffc5ade5222bd3195ae153825971ffbc42

This confirms schedule generation in the tested
environment, not clean-machine end-to-end simulation.
