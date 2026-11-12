# shadow-bench: bench/int8-decode-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| int8 | 3 | 57.68 | 57.25 | 57.25 | 57.70 | 0.9% | 4.5% | 12.0% | 5.6 |
| off | 3 | 56.34 | 56.04 | 56.04 | 56.41 | 0.6% | 2.1% | 28.6% | 5.7 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| off | 105 | -4.8% | -13.8% | +7.2% | -20.9% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 28 (greedy: expected 0).
