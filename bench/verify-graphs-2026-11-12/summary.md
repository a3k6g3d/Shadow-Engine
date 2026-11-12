# shadow-bench: bench/verify-graphs-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| graphs | 3 | 54.18 | 54.01 | 54.01 | 54.21 | 0.4% | 1.7% | 6.1% | 6.1 |
| eager | 3 | 53.12 | 52.98 | 52.98 | 53.15 | 0.3% | 3.6% | 7.5% | 6.2 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| eager | 105 | -2.2% | -4.0% | -1.2% | -9.2% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
