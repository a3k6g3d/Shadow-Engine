# shadow-bench: bench\overlap-2026-10-28

16 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| graphs | 8 | 18.96 | 18.02 | 18.02 | 19.08 | 1.4% | 6.4% | 12.5% | 11.6 |
| overlap | 8 | 19.47 | 18.18 | 18.18 | 19.55 | 0.4% | 5.6% | 7.7% | 11.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| overlap | 40 | +2.7% | -2.3% | +8.4% | -4.1% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
