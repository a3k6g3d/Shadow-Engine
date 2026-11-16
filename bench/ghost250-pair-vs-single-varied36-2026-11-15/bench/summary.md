# shadow-bench: J:\Shadow\bench\ghost250-pair-vs-single-varied36-2026-11-15\bench

4 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| single | 2 | 45.47 | 44.86 | 44.86 | 46.08 | 0.9% | 5.2% | 9.5% | 8.5 |
| pair44 | 2 | 45.05 | 44.48 | 44.48 | 45.62 | 0.7% | 4.0% | 5.0% | 20.9 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| pair44 | 70 | -13.4% | -32.8% | -0.9% | -36.8% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 20 (greedy: expected 0).
