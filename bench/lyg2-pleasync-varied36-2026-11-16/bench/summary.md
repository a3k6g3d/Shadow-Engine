# shadow-bench: J:\Shadow\bench\lyg2-pleasync-varied36-2026-11-16\bench

8 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| ple_async | 4 | 55.55 | 55.22 | 55.22 | 55.74 | 0.6% | 1.9% | 7.7% | 14.2 |
| ple_sync | 4 | 54.95 | 54.65 | 54.65 | 55.26 | 0.4% | 1.3% | 14.4% | 14.2 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| ple_sync | 140 | -0.7% | -1.9% | +0.7% | -12.5% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
