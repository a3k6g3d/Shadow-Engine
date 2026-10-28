# shadow-bench: bench\strata-skip-2026-10-28

4 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| skip | 2 | 49.80 | 48.60 | 48.60 | 51.00 | 4.3% | 23.9% | 23.9% | 9.2 |
| noskip | 2 | 36.60 | 36.40 | 36.40 | 36.80 | 1.6% | 5.0% | 5.0% | 9.2 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| noskip | 10 | -25.7% | -34.3% | -22.2% | -34.3% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 9 (greedy: expected 0).
