# R4: Measured Offline Preparation Overhead

Measurements were collected on the original server.
Each configuration has one warm-up and five timed runs.

The workload uses 16 MiB per rank, mode=core,
and max-local-rounds=1.

| GPUs | Planning | XML | Chakra | Total |
|---|---:|---:|---:|---:|
| 2x4 | 116.287 | 216.036 | 115.033 | 447.912 |
| 2x16 | 416.314 | 366.345 | 165.343 | 947.816 |
| 4x4 | 116.054 | 215.264 | 115.170 | 447.134 |
| 8x4 | 166.330 | 365.964 | 165.404 | 696.454 |

All times are milliseconds. Values are medians.

Total is the median of the per-run stage sums, not
the sum of the three independently computed medians.

Measurements include subprocess invocation and the
corresponding planning, compilation or conversion stage.
These are offline preparation costs, separate from
the simulated network completion time.

Original samples and the run manifest are preserved
in this directory.
