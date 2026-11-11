# shadow-bench: bench/vs-strata-2026-11-11

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

**3 runs failed** and are left out:

- strata run 0: "the server never became ready"
- strata run 1: "the server never became ready"
- strata run 2: "the server never became ready"

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| shadow | 3 | 42.06 | 41.42 | 41.42 | 42.31 | 1.0% | 2.0% | 3.2% | 4.9 |
| strata | 0 | NaN | NaN | NaN | NaN | NaN% | NaN% | NaN% | 51.3 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| strata | 0 | NaN% | NaN% | NaN% | NaN% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
