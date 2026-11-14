# shadow-bench: bench/xxs-placement-varied36-2026-11-14

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| old | 3 | 55.94 | 54.96 | 54.96 | 56.31 | 6.8% | 30.0% | 47.8% | 6.6 |
| new | 3 | 57.32 | 56.07 | 56.07 | 57.68 | 7.7% | 27.4% | 36.6% | 7.1 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| new | 105 | -2.0% | -16.8% | +31.9% | -31.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 28 (greedy: expected 0).
