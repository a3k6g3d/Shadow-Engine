# shadow-bench: bench\lru-2026-10-28

16 records, 2 arms, decode_warm (turn 0 dropped), tok/s

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| pinned | 8 | 19.53 | 19.49 | 19.49 | 19.57 | 0.2% | 0.6% | 1.5% | 11.9 |
| lru | 8 | 21.06 | 21.03 | 21.03 | 21.09 | 0.2% | 0.5% | 0.8% | 11.4 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| lru | 40 | +11.1% | -1.5% | +14.8% | -1.9% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
