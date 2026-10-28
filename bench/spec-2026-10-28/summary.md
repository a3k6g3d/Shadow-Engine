# shadow-bench: bench\spec-2026-10-28

24 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| nospec | 8 | 18.62 | 18.44 | 18.44 | 18.69 | 0.6% | 1.4% | 3.7% | 10.9 |
| spec2 | 8 | 17.04 | 16.92 | 16.92 | 17.10 | 0.5% | 1.0% | 1.8% | 10.9 |
| spec3 | 8 | 15.22 | 15.05 | 15.05 | 15.24 | 0.6% | 1.3% | 1.8% | 10.9 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| spec2 | 40 | -11.2% | -16.7% | +0.3% | -17.0% |
| spec3 | 40 | -18.3% | -21.1% | -0.0% | -21.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
