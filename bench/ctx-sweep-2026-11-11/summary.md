# shadow-bench: bench/ctx-sweep-2026-11-11

8 records, 4 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| c8192 | 2 | 42.00 | 41.68 | 41.68 | 42.32 | 0.5% | 1.3% | 4.2% | 5.4 |
| c10240 | 2 | 41.94 | 41.81 | 41.81 | 42.06 | 0.5% | 1.7% | 7.9% | 5.4 |
| c12288 | 2 | 44.75 | 36.44 | 36.44 | 53.06 | 40.5% | 88.7% | 99.6% | 5.5 |
| c24576 | 2 | 53.41 | 53.30 | 53.30 | 53.51 | 0.3% | 1.2% | 6.5% | 5.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| c10240 | 70 | -0.5% | -9.4% | +8.6% | -13.7% |
| c12288 | 70 | +1.4% | -30.1% | +45.3% | -52.7% |
| c24576 | 70 | +20.7% | -9.5% | +59.1% | -16.8% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 97 (greedy: expected 0).
