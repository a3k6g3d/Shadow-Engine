# shadow-bench: bench\graphs-2026-10-28

16 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 8 | 18.65 | 18.38 | 18.38 | 18.68 | 0.2% | 1.1% | 1.9% | 11.4 |
| graphs | 8 | 19.00 | 17.95 | 17.95 | 19.04 | 0.2% | 3.1% | 14.7% | 11.4 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| graphs | 40 | +2.1% | +0.8% | +2.6% | -10.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
