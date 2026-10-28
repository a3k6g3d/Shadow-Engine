# shadow-bench: bench\tail-2026-10-28

32 records, 4 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| exact | 8 | 21.32 | 21.25 | 21.25 | 21.36 | 0.3% | 0.8% | 1.9% | 10.2 |
| tail7 | 8 | 24.45 | 24.24 | 24.24 | 24.55 | 0.3% | 1.1% | 2.1% | 10.3 |
| tail5 | 8 | 27.78 | 27.67 | 27.67 | 27.86 | 0.2% | 0.7% | 1.5% | 10.5 |
| tail4 | 8 | 29.40 | 29.16 | 29.16 | 29.48 | 0.4% | 1.2% | 3.2% | 10.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| tail7 | 40 | +15.9% | +11.2% | +19.9% | +10.9% |
| tail5 | 40 | +29.8% | +27.3% | +36.9% | +26.5% |
| tail4 | 40 | +36.0% | +34.6% | +45.6% | +33.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 13 (greedy: expected 0).
