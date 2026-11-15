# shadow-bench: bench/lyg2-confirm-varied36-2026-11-15

16 records, 4 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| default | 4 | 53.36 | 50.81 | 50.81 | 53.45 | 1.3% | 4.9% | 52.3% | 13.7 |
| place | 4 | 57.91 | 48.24 | 48.24 | 58.12 | 5.8% | 27.8% | 36.3% | 13.7 |
| minp03 | 4 | 57.77 | 56.59 | 56.59 | 57.99 | 0.9% | 3.2% | 18.3% | 13.8 |
| pin4 | 4 | 60.31 | 53.74 | 53.74 | 60.79 | 0.9% | 6.3% | 33.8% | 17.2 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| place | 140 | -1.2% | -19.3% | +70.7% | -35.4% |
| minp03 | 140 | +0.1% | -8.0% | +101.6% | -21.5% |
| pin4 | 140 | +3.8% | -11.7% | +104.2% | -31.2% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 52 (greedy: expected 0).
