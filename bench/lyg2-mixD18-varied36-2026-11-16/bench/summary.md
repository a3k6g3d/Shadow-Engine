# shadow-bench: J:\Shadow\bench\lyg2-mixD18-varied36-2026-11-16\bench

8 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| lyg205 | 4 | 55.40 | 55.09 | 55.09 | 56.20 | 0.8% | 2.6% | 6.8% | 14.8 |
| mixD18 | 4 | 64.41 | 63.63 | 63.63 | 64.93 | 0.5% | 2.4% | 6.9% | 9.1 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| mixD18 | 140 | +16.9% | -20.2% | +107.5% | -30.2% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 33 (greedy: expected 0).
