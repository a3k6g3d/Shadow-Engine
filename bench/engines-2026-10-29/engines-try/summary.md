# shadow-bench: bench\engines-try

6 records, 3 arms, decode_warm (turn 0 dropped), tok/s

**4 runs failed** and are left out:

- ik43 run 0: "the server never became ready"
- ikcache run 0: "the server never became ready"
- ik43 run 1: "the server never became ready"
- ikcache run 1: "the server never became ready"

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| up43 | 2 | 15.12 | 14.55 | 14.55 | 15.69 | 4.8% | 10.9% | 10.9% | 7.6 |
| ik43 | 0 | NaN | NaN | NaN | NaN | NaN% | NaN% | NaN% | 1.0 |
| ikcache | 0 | NaN | NaN | NaN | NaN | NaN% | NaN% | NaN% | 4.0 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| ik43 | 0 | NaN% | NaN% | NaN% | NaN% |
| ikcache | 0 | NaN% | NaN% | NaN% | NaN% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 6 (greedy: expected 0).
