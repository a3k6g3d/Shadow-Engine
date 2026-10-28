# shadow-bench: bench/readpool-2026-10-27

16 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 8 | 17.37 | 16.85 | 16.85 | 17.53 | 0.8% | 3.9% | 7.2% | 10.4 |
| readpool | 8 | 18.52 | 18.42 | 18.42 | 18.55 | 0.4% | 0.9% | 2.2% | 10.7 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| readpool | 40 | +5.3% | +3.2% | +12.1% | +2.9% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
