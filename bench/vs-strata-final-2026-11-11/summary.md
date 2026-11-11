# shadow-bench: bench/vs-strata-final-2026-11-11

9 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| shadow | 3 | 53.66 | 53.62 | 53.62 | 53.82 | 0.4% | 1.2% | 4.0% | 5.8 |
| shadow8k | 3 | 42.45 | 42.30 | 42.30 | 42.45 | 0.4% | 1.1% | 1.7% | 5.2 |
| strata | 3 | 50.00 | 49.30 | 49.30 | 50.70 | 5.3% | 16.0% | 72.9% | 9.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| shadow8k | 105 | -17.5% | -37.6% | +10.3% | -44.4% |
| strata | 105 | -10.9% | -25.5% | +8.7% | -43.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 101 (greedy: expected 0).
