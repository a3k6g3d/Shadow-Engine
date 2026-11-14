# shadow-bench: bench/xxs-fused-iq2-long-2026-11-14

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| off | 3 | 34.13 | 34.12 | 34.12 | 34.22 | 0.7% | 5.1% | 5.3% | 7.1 |
| on | 3 | 35.81 | 35.49 | 35.49 | 35.97 | 1.2% | 22.7% | 23.9% | 7.1 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| on | 12 | -0.6% | -5.3% | +20.9% | -11.5% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 3 (greedy: expected 0).
