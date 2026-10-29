# shadow-bench: bench\fast-2026-10-29

40 records, 5 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| exact | 8 | 20.40 | 18.40 | 18.40 | 20.75 | 4.0% | 10.8% | 27.0% | 10.7 |
| t7 | 8 | 33.04 | 31.22 | 31.22 | 34.94 | 3.3% | 7.1% | 11.9% | 11.4 |
| t5 | 8 | 31.79 | 30.76 | 30.76 | 33.41 | 3.2% | 5.7% | 9.1% | 11.6 |
| t4 | 8 | 36.22 | 34.76 | 34.76 | 36.57 | 1.7% | 6.7% | 16.8% | 11.5 |
| t3 | 8 | 31.58 | 30.76 | 30.76 | 32.52 | 2.8% | 5.4% | 6.7% | 11.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| t7 | 40 | +57.9% | +7.1% | +113.3% | +1.9% |
| t5 | 40 | +57.6% | +5.3% | +80.6% | +4.5% |
| t4 | 40 | +66.4% | +30.9% | +110.0% | +23.0% |
| t3 | 40 | +53.9% | +33.5% | +147.0% | +29.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 23 (greedy: expected 0).
