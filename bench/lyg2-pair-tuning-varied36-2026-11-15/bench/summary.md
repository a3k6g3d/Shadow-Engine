# shadow-bench: J:\Shadow\bench\lyg2-pair-tuning-varied36-2026-11-15\bench

6 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| split40 | 2 | 40.63 | 39.47 | 39.47 | 41.79 | 0.9% | 5.9% | 7.4% | 22.9 |
| split44 | 2 | 46.36 | 46.16 | 46.16 | 46.55 | 1.0% | 2.1% | 7.3% | 22.8 |
| tier14 | 2 | 43.38 | 43.38 | 43.38 | 43.39 | 0.7% | 2.1% | 8.4% | 21.1 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| split44 | 70 | +2.5% | -20.3% | +127.5% | -40.2% |
| tier14 | 70 | -0.8% | -16.0% | +10.3% | -61.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 40 (greedy: expected 0).
