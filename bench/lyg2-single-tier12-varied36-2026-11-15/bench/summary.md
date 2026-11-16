# shadow-bench: J:\Shadow\bench\lyg2-single-tier12-varied36-2026-11-15\bench

4 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| single | 2 | 52.68 | 52.41 | 52.41 | 52.95 | 0.4% | 2.3% | 8.1% | 13.7 |
| single_tier12 | 2 | 50.74 | 50.59 | 50.59 | 50.89 | 2.1% | 4.8% | 7.3% | 26.8 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| single_tier12 | 70 | -5.4% | -14.4% | -2.3% | -23.3% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
