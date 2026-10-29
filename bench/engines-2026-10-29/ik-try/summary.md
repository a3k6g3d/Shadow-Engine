# shadow-bench: bench\ik-try

4 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| ik | 2 | 16.41 | 16.38 | 16.38 | 16.43 | 0.3% | 4.9% | 4.9% | 8.5 |
| ikser6 | 2 | 16.03 | 15.96 | 15.96 | 16.10 | 4.0% | 8.7% | 8.7% | 8.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| ikser6 | 10 | -0.9% | -7.6% | +1.3% | -7.6% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 18 (greedy: expected 0).
