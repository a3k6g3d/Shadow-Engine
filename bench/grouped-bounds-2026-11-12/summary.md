# shadow-bench: bench/grouped-bounds-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| bounds | 3 | 60.98 | 60.34 | 60.34 | 61.08 | 1.2% | 2.4% | 5.6% | 4.9 |
| nobounds | 3 | 59.47 | 59.29 | 59.29 | 59.64 | 0.6% | 1.7% | 7.8% | 4.9 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| nobounds | 105 | -2.4% | -3.3% | -1.2% | -8.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
