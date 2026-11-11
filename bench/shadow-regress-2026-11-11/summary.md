# shadow-bench: bench/shadow-regress-2026-11-11

6 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| head | 2 | 53.70 | 53.69 | 53.69 | 53.70 | 0.5% | 3.2% | 6.2% | 5.9 |
| nov10am | 2 | 53.77 | 53.69 | 53.69 | 53.84 | 0.2% | 0.8% | 6.5% | 7.5 |
| nov8base | 2 | 50.58 | 49.93 | 49.93 | 51.23 | 0.9% | 3.6% | 10.8% | 6.5 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| nov10am | 70 | +0.2% | -0.3% | +1.1% | -5.7% |
| nov8base | 70 | -1.1% | -16.6% | +9.7% | -36.3% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 28 (greedy: expected 0).
