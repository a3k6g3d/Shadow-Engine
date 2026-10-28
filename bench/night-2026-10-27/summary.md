# shadow-bench: bench/night-2026-10-27

16 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 8 | 18.53 | 18.38 | 18.38 | 18.61 | 0.3% | 0.9% | 1.9% | 11.5 |
| night | 8 | 18.85 | 18.73 | 18.73 | 18.88 | 0.4% | 1.0% | 2.4% | 11.8 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| night | 40 | +1.5% | +0.9% | +2.2% | +0.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
