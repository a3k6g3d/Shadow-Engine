# shadow-bench: bench\batch-2026-10-28

16 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 8 | 18.49 | 16.39 | 16.39 | 18.61 | 7.8% | 12.7% | 15.7% | 10.9 |
| batch | 8 | 18.49 | 17.23 | 17.23 | 18.69 | 4.7% | 8.5% | 9.7% | 10.8 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| batch | 40 | +0.0% | -0.6% | +4.0% | -0.8% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
