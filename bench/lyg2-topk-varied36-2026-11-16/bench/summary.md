# shadow-bench: J:\Shadow\bench\lyg2-topk-varied36-2026-11-16\bench

8 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| topk_on | 4 | 54.88 | 54.21 | 54.21 | 55.01 | 0.5% | 2.2% | 7.1% | 13.9 |
| topk_off | 4 | 53.09 | 51.93 | 51.93 | 53.47 | 0.5% | 2.4% | 7.5% | 13.8 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| topk_off | 140 | -3.8% | -5.4% | -2.4% | -10.9% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
