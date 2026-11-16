# shadow-bench: J:\Shadow\bench\lyg2-pair-vs-single-varied36-2026-11-15\bench

6 records, 3 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| single | 2 | 51.65 | 50.89 | 50.89 | 52.40 | 1.6% | 3.0% | 13.1% | 13.4 |
| pair | 2 | 43.50 | 43.41 | 43.41 | 43.59 | 0.6% | 1.7% | 3.0% | 22.6 |
| pair_nodflash | 2 | 33.34 | 33.31 | 33.31 | 33.36 | 0.4% | 1.2% | 8.4% | 22.7 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| pair | 70 | -17.1% | -69.5% | +2.5% | -78.5% |
| pair_nodflash | 70 | -36.7% | -55.0% | +44.7% | -58.8% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 50 (greedy: expected 0).
