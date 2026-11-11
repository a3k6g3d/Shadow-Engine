# shadow-bench: bench/df-keep-ab-v3-2026-11-11

4 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| old8k | 2 | 42.27 | 42.27 | 42.27 | 42.27 | 0.2% | 0.5% | 0.9% | 4.9 |
| new8k | 2 | 53.38 | 53.37 | 53.37 | 53.39 | 0.3% | 1.5% | 4.9% | 4.9 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| new8k | 70 | +0.4% | -0.4% | +72.4% | -5.1% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 14 (greedy: expected 0).
