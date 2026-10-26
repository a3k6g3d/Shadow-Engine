# The Rust engine: the plan

Written 2026-10-25. [RUST_HOST.md](RUST_HOST.md) decided the split (2026-10-14), [RUST_ENGINE.md](RUST_ENGINE.md)
set the order of the hybrid work and it was walked (M1–M8, branch `rust-host-0.1.45`), and
[RUST_ENGINE_NEXT.md](RUST_ENGINE_NEXT.md) is what was left of *that*. This document is the plan for the thing the
previous three stop short of: **an engine that is mainly Rust**, running the model this PC is built around, fast and
well optimised, with the harness as it stands today.

It is the plan I would run. It is not a promise of speed, and the first section says why.

## What "fast" means here, given what has already been measured

The single most useful result in the whole hybrid effort is this one: **the bound is below the program.** One C++
process, 4 threads, queue depth 256: ~155k reads/s. 16 threads at depth 1024: ~111k. Two processes ~112k total. Four
processes ~113k total. More processes, each with its own reaping thread, never add up to more than one — so **no
language at the top of the stack can beat this drive** (Sandisk Optimus 5100 500 GB NVMe), and pushing harder lowers
the total, which is the usual sign of a saturated drive.

So Rust does not buy speed on the I/O path, and the hybrid run already proved it: the Rust reader came in at parity,
−1.7% at the round median, inside the noise floor. **Rust buys the safety of the async tier and the scheduling layer —
and the speed comes from scheduling, not from the language.** Everything in the plan below is a scheduling or a memory
win, and the language is what makes them expressible without data races.

The measurement protocol is not negotiable, and it is in `BENCH_NOISE.md`: the floor is **10.1% median, 27.0% p90,
51.9% worst**. A change must clear that, on ≥8 runs, interleaved, `decode_warm`, paired turn-by-turn, or it is
reported as a tie and not written up as a win.

## What we are running, from the file itself

`J:\Strata-work\models\swift15\IQ3_S\...-00001-of-00002.gguf` + `-00002-of-00002.gguf`, 41.77 + 36.04 GB,
1,224 tensors, 77.80 GiB of tensor data. Parsed from the GGUF key-value section and from the tensor offsets —
every number below is measured off the file, none of it is taken from the filename.

| | |
|---|---|
| architecture | `qwen4exp`, `general.size_label 512x56B`, quantized by ISTA DASLab, `file_type 26` |
| layers | **48 blocks: 36 recurrent (J) + 12 full-attention (K)**, `full_attention_interval 4` — every 4th is K |
| attention | 24 heads, **2 KV heads** → GQA 12:1, head/key length 256 |
| hidden / MLP | 2,560; **512 routed experts at 640** intermediate; SSM inner 6,144, state 128, groups 16, conv 4 |
| context | **262,144** native; no yarn keys in this file |
| vocab | 248,320, `pre=qwen35`, model `gpt2`; chat template 9,993 bytes, ChatML **with** a system role |
| per-layer, measured | J: **1.032 GiB/layer** (37.14 GiB over 36) — K: **1.086 GiB/layer** (13.03 GiB over 12) |
| per layer, split | routed experts 0.961 GiB (J) / 1.020 GiB (K); attention 0.041; shared expert 0.020 |
| embeddings | `per_layer_token_embd` **26.82 GiB** — 35% of the model, and it is not in any layer |

**The table this section had before was measured on a different file** —
`Qwen3.8-flash-next-swift15-abliterated-iq3s.GGUF`, 20.06 GiB, 40 shards, 256 experts, vocab 151,664, which is not
on this PC. Its rows were off by 2-21x on every geometry line, and its arithmetic did not close either: 4x144 MiB
+ 11x576 MiB is 6.87 GiB, not 11.16, and 256x3x512x2048x3.0625 bits is 294 MiB per layer, not 48.

**The placement answer, recomputed: about 10 of the 48 layers fit in the 11.25 GiB RAM tier** — nine J-layers
greedy at 10.48 GiB, 0.77 GiB spare, and no room left for the 26.82 GiB embedding tier the old table never
counted. The shape of the answer is the same as the plan said (resident set = the J chain, the rest streamed);
only the size changed, and it changed by 21x. What has to be decided before Phase 0 is what the 11.25 GiB is
for: layers, or the embedding.

The J-layers are still the reason this model is fast — a fixed-size state per layer regardless of context — but
the old state figure (8 x 256 x 256 x 4 = 2 MiB) was computed from the old table's geometry and needs measuring
against `ssm.inner_size 6144` / `state_size 128` before anyone quotes it again.

The quant is what it is — **don't requant**. It was measured: that recipe came from a 40-candidate sweep, and
`Q3_K_S` is 10.7 GB and 13 tok/s. The two files on this PC are that quant; the 20 GiB one is not.

## The plan

### Phase 0 — the floor, before any Rust
Build nothing. Run the existing C++ engine cold, warm, and pinned, and write the numbers down. **Rust must beat the
floor, not the previous Rust.**

### Phase 1 — keep the C++ hot path, rewrite around it
Don't rewrite the kernels. **Rewrite around them.** The 63 `.cu` files stay C++ — Rust never compiles a kernel, it
calls one. The already-built seam (`STRATA_ABI_VERSION 1`, hand-written `extern "C"`, no bindgen, no cxx, both
directions parsed by the drift test) is the join and it is **not to be redesigned**; it has been walked eight times.

What goes Rust first, in this order, because each item is independently shippable and independently gated:

1. **The scheduler** — the async tier, the ring, the reaper. `tokio`'s `spawn_blocking`, one ring per drive, not per
   core. This is where the lifetime rules the hybrid work pinned with `compile_fail` doctests actually pay:
   *"a reference into a slot cannot live across a layer advance"*, and a lease kept past its window is refused with
   `BadSlot`.
2. **The placement policy** — which 15 layers get RAM, which experts get touched twice per token, which are dead.
   A policy is pure and testable; a policy in C++ is a policy you can't change at runtime.
3. **The router's bookkeeping** — the counters, the hysteresis, the eviction. Not the score itself: that's a kernel.
4. **The API surface** — the OpenAI-compatible HTTP layer, the session state, the harness loop.

**Gate:** the seam's drift test green both directions on the merged tree, both arms, and the skip path run too.

### Phase 2 — the things that can actually make it faster
These are the levers with a measured number behind them, in the order they measured, from `SPEED_BRANCH.md`:

| lever | measured |
|---|---|
| more of the model in pinned RAM | **~0.7 tok/s per GiB** (36/40/44/45.5 GiB → 10.7/13.2/15.5/17.1) — the biggest one, and it is a RAM budget, not code |
| closing programs | ~14 GiB held by other programs on that PC — worth more than any setting |
| `--vram-reserve-mib 640` | +5% (the desktop idles at 567 MiB; +150 expert slots) |
| expert cache on | **−34.2% decode with it off** — the one result in the line that is not marginal |
| LTO, arch 120-real, tests off | tie |
| `--pool-workers` 4/6, `--kv q4_0` with freed VRAM | tie |
| router look-ahead | tried and reverted — NVMe wait did not shrink, CPU work grew |
| batched DMA | +1.1% — inside the floor, reported as such |

Of a ~115 ms decode window: **~52 ms the CPU running RAM-tier experts, ~26 ms waiting on the NVMe tier, ~11 ms
waiting for the GPU.** So 52 ms is the biggest single piece, and it is the piece Rust can actually move — the
RAM tier's experts are run by CPU code, not by a kernel.

### Phase 3 — the fringe ideas, and what each is worth
Pulled from outside, each one stated with its cost and its honest ceiling, because most of them are small:

- **`io_uring`** (the Linux async I/O interface) — **the one genuinely worth doing.** It is the only way to get real
  async submission depth to the drive without a thread per read, and it is the reason the scheduler goes Rust first.
  On Windows the equivalent is **`IoCompleteCompletion` + `FILE_FLAG_OVERLAPPED` with a single completion port** —
  already the shape of the existing ring, so the port is the work, not the idea. Ceiling: the drive is already the
  bound, so this buys **shape** (fewer threads, no reaper contention) rather than throughput.
- **Huge pages for the pinned tier** — 2 MiB pages for the 11.25 GiB RAM tier means ~6,400 page-table entries instead
  of ~2.9 million. The pinned tier cannot be paged out, so this is pure TLB. Cheap, and Windows has
  `VirtualAlloc(MEM_LARGE_PAGES)`. Worth trying first because it is a day.
- **Prefetch the first expert of the next layer during the current layer's compute** — this is the look-ahead that
  was *tried and reverted* in C++ (NVMe wait didn't shrink, CPU work grew). Re-test it in Rust **only** if the
  scheduler rewrite lands first, because the C++ version couldn't overlap it with anything.
- **Speculative decoding** — the model is 8 active experts of 256, so per-token compute is already tiny; the win
  would be in the ~26 ms of NVMe wait. A small draft model is the standard trick; the fringe version is **self-draft
  with the shared expert alone** (the shared expert is always active, always in VRAM, and its 5120-intermediate MLP
  is already resident — you get a free cheap prediction head). Untested here; the honest expected value is small
  because verification cost is per-layer and you'd pay it 84 times.
- **KV quantisation at long context** — `--kv q4_0` measured a tie at the depths tested, but at 128K the 48 K-layers'
  KV is the thing that stops fitting. It becomes a real lever only at context lengths this PC hasn't been benched at.
- **`mlock` everything** — the pinned tier already can't be paged; the rest of the 11.25 GiB can. Cheap insurance
  against the "free RAM hit 0.4 GiB and froze the PC" failure mode.
- **`MAP_HUGETLB` / `VirtualAlloc` for the GGUF itself** — the 40 shards are read via the ring today; a single
  memory-mapped view with page-fault-driven loading would let the OS do the placement. It fights the pinned tier,
  so it's an either/or, not an addition.

**Explicitly not on the list:** a Rust kernel, a Rust CUDA bindings crate, a new default for anything, and any speed
claim that hasn't cleared the floor. All four are the existing plan's rules and they stand.

### Phase 4 — the harness
The current harness (`tools/pi_harness.py`) suffices, with three changes, none of them a rewrite:

1. **It must run against the new engine through the same OpenAI endpoint**, so the only thing that changes is the
   server URL. It already takes it from `J:\Strata-work\strata.json`.
2. **The guard's no-read list and protected paths must cover the new engine's source tree** — the harness runs with
   no approval gate, so those lists are the only thing enforcing the boundary. They are loaded from `strata.json`
   (50 blocked, 8 protected).
3. **Add a probe that runs the same task on both engines and diffs the answers.** The existing output-fingerprint
   tooling does this job at the byte level (it was 10 match / 0 mismatch across a fingerprint arm and three bench
   launches); the harness version would do it at the task level, which is the one that catches a wrong-but-plausible
   answer.

The harness's job on the new engine is **the same job it does today**: prove the answer is the same one the C++
engine gives, before anything is called a win.

## What has to be true before any of this starts

- **N1 first** — the stale `stage_idx_` reset is the only real bug in the existing list and it is in today's C++.
  Fix it, gate it on `expert_parity` plus the audit run (4,980 pool reads, 0 buffer refills under them), and only
  then move the layer under it.
- **N5, the merge forward, before the trees drift further.** `rust-host/`, `include/strata/c/abi.h`,
  `src/facade/abi.cpp`, `tools/output_fingerprint.py` are tracked on `speed-work-0.1.43`; `speed-work-0.1.45` carries
  none of them and the Rust branch grew off 0.1.45 and is now 725 commits deep. A plan that starts on a third tree
  is a plan that merges three ways.
- **Decide N2** — rail the lifetime types into the hot path now, or freeze them as tests. The plan says pick one;
  doing both is the failure mode. My choice: **freeze them as tests** for the C++ hot path, and **wire them in for
  free** on the Rust side, because in Rust they're the language's own rules and cost nothing.
- **Decide what the engine is for.** If the answer is "run this model on this PC as fast as it will go", the honest
  answer is that the C++ engine is already at the drive's bound and the Rust engine is a **reliability and
  maintainability** project with a scheduling upside — which is a completely good reason, and it's better to say so
  at the start than to discover it at the end.

## The honest shape of it

**The shell is ours** — `desktop/` is 100% ours, `rust-host/` is 100% ours, the web app is 68% ours since it landed
(it started as upstream's), `tools/` and `serve/` are a fifth ours, and the engine is 1% ours.

That is the actual argument for this plan. Not speed — the drive is the bound and the numbers say so. It's that the
one layer everything else sits on is the one layer we don't own, and the plan's first three sections are the parts
where not owning it has started to cost: the lifetime rules that need `compile_fail` doctests to hold, the bug that
can only be fixed by a decision about the default path, the merge that's 725 commits deep.

The first phase costs nothing and tells you whether the rest is worth funding. Start there.
