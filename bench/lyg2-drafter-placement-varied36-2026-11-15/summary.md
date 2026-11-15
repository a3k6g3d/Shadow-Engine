# shadow-bench: bench/lyg2-drafter-placement-varied36-2026-11-15

21 records, 7 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 52.09 | 43.86 | 43.86 | 53.03 | 12.5% | 29.5% | 48.2% | 13.2 |
| spec4 | 3 | 50.00 | 46.30 | 46.30 | 50.16 | 1.5% | 5.0% | 21.6% | 12.9 |
| spec2 | 3 | 49.36 | 41.12 | 41.12 | 51.55 | 15.9% | 29.8% | 41.8% | 13.3 |
| minp03 | 3 | 53.61 | 52.89 | 52.89 | 57.94 | 2.5% | 7.6% | 30.4% | 13.2 |
| place | 3 | 58.37 | 57.41 | 57.41 | 58.42 | 1.1% | 4.6% | 29.7% | 12.9 |
| pin4 | 3 | 60.20 | 57.66 | 57.66 | 60.97 | 1.4% | 10.3% | 32.3% | 16.7 |
| tail4 | 3 | 22.01 | 21.91 | 21.91 | 22.06 | 1.7% | 10.4% | 31.7% | 12.7 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| spec4 | 105 | -4.6% | -20.9% | +27.2% | -70.1% |
| spec2 | 105 | +0.5% | -30.2% | +42.6% | -69.5% |
| minp03 | 105 | +3.7% | -6.2% | +112.4% | -20.4% |
| place | 105 | +6.3% | -4.1% | +106.3% | -17.5% |
| pin4 | 105 | +12.2% | -6.8% | +121.4% | -14.7% |
| tail4 | 105 | -17.9% | -65.5% | +17.2% | -100.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 110 (greedy: expected 0).
