# shadow-bench: J:\Shadow\bench\lyg2-snapkernel-varied36-2026-11-16\bench

8 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| snap_kernel | 4 | 54.65 | 54.01 | 54.01 | 54.82 | 0.4% | 1.4% | 8.5% | 14.1 |
| snap_percopy | 4 | 54.79 | 54.64 | 54.64 | 55.27 | 0.4% | 1.3% | 18.6% | 14.0 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| snap_percopy | 140 | -0.4% | -1.1% | +0.7% | -15.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
