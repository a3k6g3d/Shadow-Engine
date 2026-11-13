# shadow-bench: bench/int8-drafter-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| drafter | 3 | 64.44 | 64.14 | 64.14 | 64.63 | 0.4% | 1.1% | 3.5% | 5.1 |
| nodrafter | 3 | 61.07 | 60.94 | 60.94 | 61.12 | 0.4% | 1.4% | 2.3% | 4.9 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| nodrafter | 105 | +0.7% | -10.5% | +12.3% | -20.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 22 (greedy: expected 0).
