# shadow-bench: bench\strata-2026-10-28

16 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| shadow | 8 | 20.98 | 20.25 | 20.25 | 21.12 | 1.0% | 6.4% | 17.6% | 11.1 |
| strata | 8 | 48.55 | 45.40 | 45.40 | 49.40 | 2.5% | 12.3% | 35.1% | 8.1 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| strata | 40 | +135.2% | +104.0% | +145.5% | +76.4% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 18 (greedy: expected 0).
