# shadow-bench: J:\Shadow\bench\lyg2-decode-all-varied36-2026-11-16\bench

8 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| all_on | 4 | 55.59 | 55.41 | 55.41 | 55.78 | 0.6% | 2.2% | 7.0% | 14.3 |
| all_off | 4 | 53.06 | 52.23 | 52.23 | 53.22 | 0.5% | 2.2% | 6.2% | 14.3 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| all_off | 140 | -5.2% | -6.5% | -3.4% | -10.1% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
