# shadow-bench: J:\Shadow\bench\lyg2-topkstep-varied36-2026-11-16\bench

8 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| step_on | 4 | 54.89 | 54.39 | 54.39 | 55.30 | 0.4% | 1.7% | 4.0% | 14.2 |
| step_off | 4 | 54.81 | 54.60 | 54.60 | 54.98 | 0.3% | 1.7% | 5.7% | 14.1 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| step_off | 140 | -0.1% | -1.3% | +1.3% | -5.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
