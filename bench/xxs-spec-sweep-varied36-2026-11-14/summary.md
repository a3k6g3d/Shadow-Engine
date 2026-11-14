# shadow-bench: bench/xxs-spec-sweep-varied36-2026-11-14

15 records, 5 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 3 | 56.39 | 55.87 | 55.87 | 59.10 | 10.4% | 31.7% | 39.6% | 6.7 |
| spec4 | 3 | 56.72 | 56.09 | 56.09 | 57.25 | 7.5% | 25.5% | 31.0% | 7.6 |
| spec5 | 3 | 55.16 | 52.68 | 52.68 | 55.34 | 8.1% | 27.9% | 33.4% | 6.7 |
| minp7 | 3 | 54.89 | 53.03 | 53.03 | 55.41 | 4.6% | 30.3% | 33.9% | 7.4 |
| minp3 | 3 | 54.91 | 54.11 | 54.11 | 57.06 | 7.3% | 28.1% | 34.0% | 6.8 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| spec4 | 105 | -0.5% | -18.9% | +38.3% | -38.4% |
| spec5 | 105 | -0.6% | -18.2% | +33.9% | -42.0% |
| minp7 | 105 | -3.6% | -23.4% | +22.8% | -55.1% |
| minp3 | 105 | +0.7% | -26.3% | +37.3% | -45.6% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 99 (greedy: expected 0).
