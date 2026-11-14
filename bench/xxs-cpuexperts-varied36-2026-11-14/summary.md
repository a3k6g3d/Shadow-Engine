# shadow-bench: bench/xxs-cpuexperts-varied36-2026-11-14

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 3 | 58.45 | 56.21 | 56.21 | 60.25 | 8.1% | 29.4% | 33.6% | 6.7 |
| cpu2 | 3 | 53.39 | 52.74 | 52.74 | 54.43 | 5.7% | 22.3% | 29.4% | 7.7 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| cpu2 | 105 | -5.0% | -22.3% | +18.9% | -41.7% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 28 (greedy: expected 0).
