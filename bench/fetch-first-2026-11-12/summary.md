# shadow-bench: bench/fetch-first-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| first | 3 | 56.44 | 56.36 | 56.36 | 56.67 | 0.2% | 0.7% | 2.0% | 5.8 |
| after | 3 | 56.03 | 55.52 | 55.52 | 56.06 | 0.5% | 2.2% | 5.1% | 5.8 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| after | 105 | -1.2% | -2.2% | -0.4% | -5.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
