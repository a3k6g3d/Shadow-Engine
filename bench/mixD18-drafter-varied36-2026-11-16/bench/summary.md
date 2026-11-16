# shadow-bench: J:\Shadow\bench\mixD18-drafter-varied36-2026-11-16\bench

12 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| stock | 4 | 64.04 | 61.72 | 61.72 | 64.47 | 2.3% | 5.4% | 10.6% | 11.5 |
| tuned | 4 | 64.83 | 64.11 | 64.11 | 64.99 | 0.5% | 1.7% | 3.8% | 11.5 |
| tuned_spec4 | 4 | 60.27 | 59.92 | 59.92 | 60.55 | 0.6% | 2.7% | 15.9% | 11.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| tuned | 140 | +1.4% | -16.0% | +22.1% | -24.6% |
| tuned_spec4 | 140 | -5.1% | -20.1% | +25.2% | -61.1% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 46 (greedy: expected 0).
