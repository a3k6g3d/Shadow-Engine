# shadow-bench: bench/mmq-small-2026-11-12

6 records, 2 arms, decode_warm (turn 0 dropped), tok/s

**3 runs failed** and are left out:

- mmq2 run 0: "turn 0: HTTP 400: {\"error\":{\"code\":400,\"message\":\"internal error: called `Option::unwrap()` on a `None` value. The conversation's cache was reset: send the message again, or start a new chat.\",\"type\":\"invalid_request_e"
- mmq2 run 1: "turn 0: HTTP 400: {\"error\":{\"code\":400,\"message\":\"internal error: called `Option::unwrap()` on a `None` value. The conversation's cache was reset: send the message again, or start a new chat.\",\"type\":\"invalid_request_e"
- mmq2 run 2: "turn 0: HTTP 400: {\"error\":{\"code\":400,\"message\":\"internal error: called `Option::unwrap()` on a `None` value. The conversation's cache was reset: send the message again, or start a new chat.\",\"type\":\"invalid_request_e"

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 3 | 56.59 | 56.23 | 56.23 | 56.71 | 0.3% | 0.9% | 2.8% | 5.7 |
| mmq2 | 0 | NaN | NaN | NaN | NaN | NaN% | NaN% | NaN% | 8.8 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| mmq2 | 0 | NaN% | NaN% | NaN% | NaN% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 0 (greedy: expected 0).
