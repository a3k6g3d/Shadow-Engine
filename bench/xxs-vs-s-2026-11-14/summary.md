# shadow-bench: bench/xxs-vs-s-2026-11-14

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| s | 3 | 48.72 | 48.06 | 48.06 | 49.88 | 8.7% | 14.7% | 21.5% | 7.3 |
| xxs | 3 | 56.02 | 54.88 | 54.88 | 63.27 | 10.1% | 22.6% | 28.3% | 8.9 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| xxs | 15 | +14.2% | -3.2% | +32.3% | -7.9% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 6 (greedy: expected 0).
