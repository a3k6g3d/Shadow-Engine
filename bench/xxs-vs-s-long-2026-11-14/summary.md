# shadow-bench: bench/xxs-vs-s-long-2026-11-14

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| s | 3 | 38.73 | 31.58 | 31.58 | 40.74 | 24.7% | 28.6% | 30.4% | 5.6 |
| xxs | 3 | 34.46 | 34.30 | 34.30 | 34.70 | 1.8% | 12.0% | 12.4% | 7.0 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| xxs | 12 | -6.7% | -15.9% | +11.0% | -20.5% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 5 (greedy: expected 0).
