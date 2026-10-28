# The Rust host runtime

Decided 2026-10-14. **Rust owns the host layer; the existing CUDA kernels stay C++ behind a C ABI facade.**
No kernel is rewritten in Rust, and no Rust CUDA bindings crate is used — Rust never compiles a kernel, it calls one.

## Why, stated honestly

The rewrite is **not** for tokens/sec. A clean-slate engine inherits the same physics this machine has: a 2.5 GB/s
NVMe (measured, not labelled), a Gen4 x16 link with ten times the bandwidth the drive can supply, and a routing
spread where only ~9.1% of experts cross the bus. The batched-DMA experiment bought 1.1% — inside the 10.1% noise
floor (BENCH_NOISE.md). So the honest justification for this project is **the safety of the async tier**, which is
where this engine's bugs have always lived, and nothing else. Any milestone that reports a speed win must clear the
noise floor with the standard protocol (interleaved arms, >=8 runs, `decode_warm`, paired turn-by-turn) or it is
reported as noise.

What Rust buys the host layer specifically:

- **Ownership over the buffers that outlive their calls.** The transient-blob lifetime rule (`expert_source.hpp:200`,
  "stays valid for the layer it was asked in and the next one or two") is a comment today. It becomes a borrow.
- **Rings as values.** The IOCP completion ring, the pinned tier, the depth-1..4 prefetch window: each gets a type,
  a capacity and a drop impl, instead of a struct and a convention.
- **No data races between the I/O pool and the expert pool**, checked at compile time rather than by the parity test.

## The split

| Rust owns (the host) | Stays C++ (the device) |
|---|---|
| `platform/direct_file` — the `NO_BUFFERING \| OVERLAPPED` ring, IOCP | the 63 `.cu` files (verify, GDN/QSA, router, head) |
| `core/pinned` — the pinned RAM tier (`--resident-budget-gib`) | `strata_core` (device, graph, weights, layout) |
| `core/expert_source` — the tier decision, the window tables, `warm()` | `strata_kernels` |
| `core/overlap_main`, the planner, the router look-ahead | the MTP draft graph |
| the `GEN` protocol endpoint the Python server already speaks | (linked through the facade, not rewritten) |

`serve/` does not change: the Rust process is the thing the Python server talks to, over the existing
`GEN <max_new> <ids>` line protocol, on the same port. The whole tool layer, the web app and both API dialects keep
working against it from day one — that is the property that makes this project safe to run alongside the current
engine rather than as a bet.

## The FFI shape

`strata_facade.dll` — a small C++ shim, compiled by MSVC, linking `strata_core` + `strata_kernels`, exporting a
hand-written `extern "C"` header (`include/strata/c/abi.h`). Rust binds to that header directly
(`extern "system"`, no bindgen, no cxx). Rules:

- **C ABI only.** No C++ types cross. Buffers are `(ptr, len)`, errors are an int enum, everything is `#[non_exhaustive]`
  on the Rust side so a new variant cannot silently break a match.
- **One version constant, checked from both sides.** `STRATA_ABI_VERSION` in the header, mirrored in
  `rust-host/crates/strata-abi/src/lib.rs`, asserted equal by a test that parses the header. This is the cheapest
  defence against the failure mode that gets these projects: a struct field added on one side and a silent mismatch
  on the other.
- **Ownership crosses explicitly.** Anything Rust hands to the facade must outlive the call, and the facade never
  returns a pointer it owns. The Rust side encodes that as a lifetime, not a comment.

## Milestones, cheapest decision first

1. **Skeleton** (this commit): the workspace, the ABI contract, the ring as a type, tests in pure Rust. No CUDA.
2. **Link test** (done): the facade DLL built from the existing static libs, one Rust test that calls
   `strata_abi_version()` through it. Proves the toolchain question (MSVC + cudart + cargo) on day one, because that
   is the assumption this whole plan rests on.  **It answers yes on this machine, both arms.**
   `strata_facade` is the repo's first shared library; the probes in `src/facade/abi.cpp` call into `strata_spec`
   (and cudart when CUDA is on) rather than returning constants, so the linker has to have actually pulled them in.
   Built and run both ways: the CPU arm answers `choose(3, 4, 6) = lookup, window 4` out of `strata_spec` and
   declines the CUDA call with `DEVICE_UNAVAILABLE`; the CUDA arm (nvcc 13.4) answers the same policy question
   identically and reports a real runtime version.  `cargo test -p strata-abi` is 11 tests, and runs green on a
   machine where the facade has never been built — build.rs is a finder, not a builder, and the tests say so.
   The DLL is loaded by absolute path rather than linked through the import library: the link works, but Windows
   then resolves the DLL at process start from the exe's directory or PATH, and cargo prunes anything in `deps/`
   that it did not put there itself.  `extern "system"` against the header is unchanged either way.
3. **One vertical slice** (done, the data path; the turn itself is not): Rust reads one expert off a file through
   its own ring, hands it down the seam to the existing kernel, and the answer is checked.  What the plan worded as
   "serves a single decode turn … returns the token — with the Python server unchanged" is *not* what is built:
   there is no server in the test, no token out, and the file is a synthetic pack in the temp directory (on this
   machine's NVMe, but not read because it is NVMe).  What is built is the two halves the slice exists to prove:
   the Rust side's own `NO_BUFFERING | OVERLAPPED` read path, on its own IOCP, in a new std-only crate
   (`rust-host/crates/strata-host`), and the first seam call that carries a struct — `strata_facade_expert_apply`,
   which runs the existing `s2_expert_vnni` on bytes the *caller* read.  No Rust kernel code anywhere.  The facade
   now links `strata_kernels_cpu` behind a guard, and built without it the shim declines rather than pretending.
   The checks are two and deliberately different in kind: self-consistency (ring bytes vs an ordinary buffered
   read, same kernel, same answer) and structure (a different blob, a different answer — because a read that always
   returns the wrong window is perfectly self-consistent, the parity lesson from `expert_parity`).  Green on the
   CPU arm with the facade built: 13 tests, and the skip path was run too — a machine with no built facade is told
   so, not passed in silence.

   Four things the machine answered on the way:

   * **A completion port's key cannot tell one read from another.** `GetQueuedCompletionStatus` reports the key
     registered at `CreateIoCompletionPort` — one value for the whole handle — so the OVERLAPPED pointer is what
     identifies a read, and the pending table is keyed by that.  Keying it by the key, which is what reads
     naturally, reaps nothing and looks like a timeout.
   * **The ring's advertised rule was false for its own signature.** `push` took the item by value and reported a
     refusal with a `Status`, so a refused item was *dropped inside `push`* — for a completion, that means its
     aligned buffer is freed, which is exactly the thing the rule says does not happen.  The test that checked the
     rule used `u32`, which has no destructor, so it could not see this.  `push` now hands the refused item back,
     and the rule is tested with a type whose drops are countable.
   * **Answers are compared as bits, not as floats.** Random bytes through a quantized kernel produce plenty of
     NaN, and `NaN != NaN`, so an `assert_eq!` on two runs that agree exactly would fail.  `to_bits` is what the
     machine actually produced.
   * **The facade's last-error buffer is per thread.** Two Rust tests raced on it and one read the other's
     failure.  A host with one ring and one serving thread would not have caught it; the test suite did.
4. **Fingerprint parity**: the existing engine and the Rust host, same prompt, same seed, byte-identical output. This
   is the harness gap BENCH_NOISE flagged (no output hash); it has to be built for this project anyway, and it is the
   test that says a rewrite is *allowed* to exist.  The harness is built (`tools/output_fingerprint.py`, 13 selftest
   checks plus `tools/test_output_fingerprint.py`, 10 pytest tests, both green; `digest()` in `tools/strata_bench.py`
   records `output_sha256` per request now, which is what item 5 of BENCH_NOISE asked for).  What it measured on the
   existing engine before there was any Rust host to compare against (2026-10-19, RTX 5070, engine 0.1.45, the A/B
   harness's own fixed prompts, seed 1234, temperature 0 — which means greedy, since the server does not forward a
   temperature of 0): **the existing engine is not byte-identical to itself.**  Five runs of the same prompt gave 2, 5
   and 4 distinct answers across the three prompts.  So "byte-identical output" cannot be the test as written, on
   these prompts, against this engine — and the tool says which prompts it can be checked on instead of assuming it:
   a short exact-answer prompt was identical five times out of five in an earlier probe, and an open-ended one was
   never identical twice.  The parity gate therefore compares only prompts each arm reproduces on its own, and reports
   the rest as *not comparable* rather than as a pass.  Two more things the shape of the data forced into it: the
   first run of a prompt is a different computation from the later ones (the conversation cache makes run 2 read what
   run 1 wrote), so runs are split cold/warm and only like is compared with like; and an errored run never votes, or
   a run that failed and returned the empty string would agree with every other failure and the test would pass out
   of nothing.  The gate is not vacuous — the tests check that a real difference comes out as a mismatch.
5. Only after 4: the measurement arms (the `pcie_frac` arm the Gen4 correction created, the rank-margin gate), read
   against the floor.

Milestones 1-4 win nothing in speed and are not allowed to claim they do.

## What this repo must not lose

The engine's invariants are the product, not the implementation: **a prediction only ever *warms*** — a guess never
changes what gets computed, only when the bytes arrive. And the build-time guard that keeps the batch cliff out
(`static_assert(MAXT * 10 <= 128)`) has a Rust-side twin in the ABI tests, because a rewrite that quietly moves a
guard is worse than no rewrite.
