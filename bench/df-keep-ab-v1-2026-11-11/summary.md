# shadow-bench: bench/df-keep-ab-2026-11-11

6 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| old8k | 2 | 41.78 | 41.18 | 41.18 | 42.38 | 0.3% | 1.8% | 2.9% | 5.3 |
| new8k | 2 | 46.29 | 46.28 | 46.28 | 46.30 | 0.4% | 1.8% | 34.5% | 5.8 |
| new24k | 2 | 45.26 | 44.84 | 44.84 | 45.68 | 0.4% | 1.2% | 185.7% | 4.9 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| new8k | 70 | -0.6% | -20.6% | +54.3% | -30.1% |
| new24k | 70 | -5.8% | -20.6% | +38.0% | -73.2% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 57 (greedy: expected 0).
