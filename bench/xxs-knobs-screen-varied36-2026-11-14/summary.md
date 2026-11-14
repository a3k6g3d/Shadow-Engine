# shadow-bench: bench/xxs-knobs-screen-varied36-2026-11-14

12 records, 6 arms, decode_warm (turn 0 dropped), tok/s

**2 runs failed** and are left out:

- cpu2 run 0: "turn 0: HTTP 400: {\"error\":{\"code\":400,\"message\":\"internal error: called `Option::unwrap()` on a `None` value. The conversation's cache was reset: send the message again, or start a new chat.\",\"type\":\"invalid_request_e"
- cpu2 run 1: "turn 0: HTTP 400: {\"error\":{\"code\":400,\"message\":\"internal error: called `Option::unwrap()` on a `None` value. The conversation's cache was reset: send the message again, or start a new chat.\",\"type\":\"invalid_request_e"

| arm | runs | median | p90 (slow end) | worst | best | floor median | floor p90 | floor worst | lowest RAM GiB |
|---|---|---|---|---|---|---|---|---|---|
| base | 2 | 56.60 | 54.76 | 54.76 | 58.45 | 11.2% | 29.6% | 36.4% | 6.7 |
| tail4 | 2 | 54.71 | 54.62 | 54.62 | 54.80 | 7.4% | 30.0% | 34.1% | 7.2 |
| vr3 | 2 | 52.00 | 50.79 | 50.79 | 53.22 | 12.1% | 30.2% | 32.6% | 7.4 |
| pin4 | 2 | 56.42 | 56.37 | 56.37 | 56.47 | 3.2% | 6.0% | 32.7% | 8.3 |
| cpu2 | 0 | NaN | NaN | NaN | NaN | NaN% | NaN% | NaN% | 9.6 |
| fuse7 | 2 | 57.74 | 56.84 | 56.84 | 58.64 | 1.2% | 4.6% | 17.5% | 7.6 |

Paired against the first arm, same run and turn (positive = faster than it):

| arm | pairs | median | p10 | p90 | worst |
|---|---|---|---|---|---|
| tail4 | 70 | -4.6% | -21.6% | +25.5% | -48.8% |
| vr3 | 70 | -5.9% | -29.1% | +17.4% | -52.5% |
| pin4 | 70 | -3.3% | -21.9% | +26.2% | -32.9% |
| cpu2 | 0 | NaN% | NaN% | NaN% | NaN% |
| fuse7 | 70 | -0.1% | -29.5% | +33.8% | -58.0% |

A paired difference inside the first arm's floor is a tie.

Answers that differ from the first one seen at the same turn: 109 (greedy: expected 0).
