# shadow-bench: bench/grouped-unroll-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| unroll | 3 | 59.40 | 59.02 | 59.02 | 59.65 | 0.9% | 2.4% | 3.5% | 4.8 |
| plain | 3 | 58.24 | 58.09 | 58.09 | 58.49 | 0.5% | 1.3% | 2.2% | 4.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| plain | 105 | -1.7% | -2.4% | -0.7% | -3.1% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
