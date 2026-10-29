# shadow-bench: bench\ik-try2

4 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| ikcache6 | 2 | 15.97 | 15.92 | 15.92 | 16.02 | 0.9% | 7.8% | 7.8% | 8.4 |
| ikcache4 | 2 | 14.56 | 14.07 | 14.07 | 15.05 | 5.6% | 17.4% | 17.4% | 8.5 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| ikcache4 | 10 | -11.0% | -24.1% | -5.4% | -24.1% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 18 (greedy: expected 0).
