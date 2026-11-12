# shadow-bench: bench/shared-early-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| early | 3 | 56.08 | 56.05 | 56.05 | 56.19 | 0.3% | 1.2% | 3.1% | 5.5 |
| late | 3 | 54.65 | 54.64 | 54.64 | 54.83 | 0.1% | 0.7% | 1.6% | 5.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| late | 105 | -2.2% | -2.6% | -1.8% | -3.9% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
