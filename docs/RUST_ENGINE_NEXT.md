# The Rust engine: what is left, in the order it can be done

Written 2026-10-24. [RUST_HOST.md](RUST_HOST.md) decided the split (2026-10-14) and
[RUST_ENGINE.md](RUST_ENGINE.md) set the order of work (2026-10-23) and it has been walked: **M1-M8 are done on
branch `rust-host-0.1.45`**, each with its gate run and written down. This document is what is left after that, and
it is mostly decisions rather than code.

## Where it stands, from the measurements that were made

- **The seam works and is not to be redesigned.** `STRATA_ABI_VERSION 1`, hand-written `extern "C"`, no bindgen, no
  cxx, both directions parsed by the drift test.
- **The Rust reader is at parity, not ahead** (M5, end of the section): with four issuing threads the ring matches
  the C++ pool at the depth the server uses; end to end it costs -1.7% at the round median, inside the default arm's
  own 10.1% spread. With one issuing thread it had cost -8.2%.
- **The proxy costs nothing measurable and changes nothing in the output** (M8 option A): fingerprints 10 match / 0
  mismatch against the control, 0 rule breaks over a fingerprint arm and three bench launches, decode +2.1% median
  inside a 9.4% spread.
- **The bound is below the program.** Three rounds, C++ pool: one process 4 threads depth 256 ~155k reads/s, 16
  threads depth 1024 ~111k, two processes ~112k total, four processes ~113k total. More processes, each with its own
  reaping thread, never add up to more than one, so no language at the top of the stack can beat this drive (Sandisk
  Optimus 5100 500 GB NVMe). Pushing harder lowers the total - the usual sign of a saturated drive.
- **Nothing in today's code breaks the lifetime rule** (M6 audit, `STRATA_STAGE_AUDIT=1`): 4,980 pool reads checked,
  **0** whose buffer was refilled under them; 32,768 layer steps with 2 recycles one layer after hand-out, and those
  are the startup/prompt paths where the blob is copied on hand-out anyway.

So the honest headline the plan wrote in advance is the one that came true: **Rust bought the safety of the async
tier and bought no speed**, and any milestone that reports speed must clear BENCH_NOISE's floor (10.1% median,
27.0% p90, 51.9% worst) with the standard protocol or be reported as noise.

## The order of what is left

**N1 - the stale `stage_idx_` reset (the only real bug in the list, and it is in today's C++).** Found during the M6
audit and deliberately not fixed there because it is a default-path change and so needs its own decision: the reset
that hands the startup's stage buffers back when the mapped tier takes over (`expert_source.cpp`, after
`close_direct`) clears `stage_buf_` but not `stage_idx_` or the per-buffer events, where the other reset clears all
three. A later `note_async_read` with a pointer matching a stale entry indexes `stage_buf_[v]` before it checks
anything. Fix: clear all three, as the other reset does. Gate: the parity tests (`expert_parity`), and the audit run
again - it should report the same zeros, on a path that can now be reached.

**N2 - decide what M6 and M7's types are for, then stop.** Both are built and both are *not wired in*:
`transient.rs` (a reference into a slot cannot live across a layer advance, E0502 pinned by a `compile_fail` doctest;
a lease kept past its window is refused with `BadSlot`) and `window.rs` (depth 1..4 as a type, capacity, drop impl,
cancelling at the source; the "a prediction only ever warms" invariant tested at every depth). N1's audit says no
consumer breaks the rule today, so wiring them in is a **rail against a future consumer**, which the plan said in
advance and the audit confirmed. Two honest options, and picking one is the work:

- **Rail it now** - wire the borrow and the window into the hot path while the seam is small, pay the hot-path cost
  once, and keep the invariant tests as the thing that fails when a future consumer overreaches.
- **Freeze them as tests** - the types stay where they are, tested, proving the rules; the hot path stays C++ until a
  consumer appears that can actually break one. Cheaper, and defensible given the audit.

Do not do both, and do not wire them "for symmetry".

**N3 - the batch-slot rules are untested in production.** The live config does not use batch slots, so the
`BGEN`/`BT`/`BDONE` rules of the proxy were exercised only by the fake engine's tests, not by the M8 gate. Either run
one gate arm with batch slots on, or write down that those rules are covered by the fake engine alone.

**N4 - the measurement arms the plan already owed** (RUST_HOST.md's item 5): the `pcie_frac` arm the Gen4 correction
created, and the rank-margin gate, read against the floor. M5's sweep already touched `--pcie-frac 0.35` (+3.8% and
-1.3% in its two cold launches, -2.7% in its warm one - no case for changing the live config) with its own too-small
floor of 0.4%; do it properly with >=8 runs, interleaved, `decode_warm`, paired turn-by-turn, or don't write it down.

**N5 - the merge forward, still owed.** `rust-host/`, `include/strata/c/abi.h`, `src/facade/abi.cpp`,
`tools/output_fingerprint.py` and `docs/RUST_HOST.md` are tracked on `speed-work-0.1.43`; `speed-work-0.1.45` carries
none of them. The Rust branch grew off 0.1.45 and is now 725 commits deep, so this is a real move in the other
direction and it should happen before the two trees drift further. Gate as written: `cargo test -p strata-abi -p
strata-host` green on the merged tree, both arms, and the skip path run too.

## What "optimize Strata" means, given all of the above

Not the language. The engine's time is spent where the physics is, and SPEED_BRANCH.md measured it on this class of
machine: of a ~115 ms decode window, ~52 ms is the CPU running RAM-tier experts, ~26 ms waiting on the NVMe tier,
~11 ms waiting for the GPU. The levers, in the order they measured:

| lever | measured | note |
|---|---|---|
| more of the model in pinned RAM | ~0.7 tok/s per GiB (36/40/44/45.5 GiB -> 10.7/13.2/15.5/17.1) | the biggest one, and it is a RAM budget, not code |
| closing programs | ~14 GiB held by other programs on that PC | worth more than any setting |
| `--vram-reserve-mib 640` | +5% | the desktop idles at 567 MiB; +150 expert slots |
| expert cache on | -34.2% decode with it off | the one result in the line that is not marginal |
| link-time optimisation, arch 120-real, tests off | tie | |
| `--pool-workers` 4/6, `--kv q4_0` with freed VRAM | tie | |
| router look-ahead (`STRATA_LOOKAHEAD`) | off with unbuffered reads; tried and reverted | NVMe wait did not shrink, CPU work grew |
| batched DMA | +1.1% | inside the floor, reported as such |

Two hardware notes that outrank any of it: the link first negotiated PCIe 2.0, and set to 3.0 it *measured* 7-10 GB/s
but logged ~230 link replays per second at idle and hard-hung the PC twice under decode load with no bugcheck - check
`nvidia-smi -q` "Replays Since Reset" before trusting a link speed, a link that replays is not stable. And the
page-locked tier cannot be paged out, so the 4 GiB headroom is the only margin for programs that grow after the
engine starts; running it down to 2 GiB bought +10% and froze the PC once free RAM hit 0.4 GiB. Keep the default.

## What this plan does not contain

No kernel in Rust, no Rust CUDA bindings crate - Rust never compiles a kernel, it calls one; the 63 `.cu` files stay
C++. No new default - every step opt-in, the default path byte-identical to the last release. No speed claim that has
not cleared the floor. The desktop shell (`desktop/tauri`, Rust, two source files) is a window around the webui and
stays separate.
