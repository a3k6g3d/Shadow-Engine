# shadow-bench: bench/vs-strata-2026-11-11

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| shadow | 3 | 42.37 | 40.99 | 40.99 | 42.49 | 0.6% | 6.9% | 151.1% | 5.2 |
| strata | 3 | 49.10 | 47.00 | 47.00 | 49.90 | 6.9% | 29.1% | 74.1% | 9.1 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| strata | 105 | +9.2% | -17.6% | +60.5% | -33.8% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 77 (greedy: expected 0).
