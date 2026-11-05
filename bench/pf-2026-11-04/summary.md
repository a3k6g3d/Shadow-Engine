# shadow-bench: bench/pf-2026-11-04

6 records, 3 arms, decode_warm (turn 0 dropped), tok/s

**1 runs failed** and are left out:

- pg8 run 1: "turn 10: An existing connection was forcibly closed by the remote host. (os error 10054)"

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 2 | 51.67 | 51.17 | 51.17 | 52.17 | 1.1% | 2.6% | 2.7% | 6.1 |
| pg8 | 1 | 17.72 | 17.72 | 17.72 | 17.72 | NaN% | NaN% | NaN% | 3.3 |
| pg16 | 2 | 18.69 | 18.56 | 18.56 | 18.82 | 1.0% | 3.1% | 89.6% | 2.2 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| pg8 | 22 | -65.1% | -69.7% | -61.9% | -69.8% |
| pg16 | 26 | -69.4% | -71.6% | -44.8% | -85.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 25 (greedy: expected 0).
