# shadow-bench: bench/df-keep-ab-v2-2026-11-11

6 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| old8k | 2 | 42.27 | 42.26 | 42.26 | 42.29 | 0.3% | 1.5% | 5.4% | 5.1 |
| new8k | 2 | 48.52 | 48.50 | 48.50 | 48.54 | 0.3% | 0.8% | 2.3% | 5.3 |
| new24k | 2 | 52.39 | 52.32 | 52.32 | 52.46 | 0.2% | 1.2% | 2.6% | 5.4 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| new8k | 70 | +6.4% | -10.7% | +55.0% | -19.3% |
| new24k | 70 | +5.4% | -11.7% | +62.4% | -15.9% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 56 (greedy: expected 0).
