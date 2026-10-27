# Shadow: the plan

Rewritten 2026-10-26. The plan this replaces (in git history before this commit) kept Strata's C++ engine and
rewrote around it: "Rust never compiles a kernel, it calls one." That is withdrawn. **Shadow is now its own engine,
start to finish**: a Rust host, CUDA kernels we write ourselves and compile at runtime with **NVRTC**, loaded and
launched through the **CUDA driver API**, and nothing from Strata at build time or at run time.

It is the plan I would run. It is not a promise of speed, and the first section says why.

## What changed, and what did not

| | before | now |
|---|---|---|
| the kernels | Strata's 63 `.cu` files, called through the C ABI seam | ours, CUDA C++ source in this repo, compiled by NVRTC when the engine starts |
| the GPU interface | the engine's, behind `include/shadow/c/abi.h` | the CUDA driver API (`nvcuda.dll`), hand-bound, loaded at run time |
| the model math | Strata's | ours, written from a spec this repo owns (`docs/ARCH.md`, stage 0) |
| the engine process | `strata --serve`, driven over its line protocol | `shadow`, one process |
| the build | needed the C++ engine built beside it | `cargo build`. No nvcc, no CMake, no CUDA toolkit at build time |
| Strata | the engine | **not used**: no source, no binary, no tool, no file format, no oracle run |

What did **not** change: the measurement protocol, the RAM safety rule, the hardware, the model file, and the
finding that the drive is the bound. All four are below.

**Why NVRTC and not nvcc, cuTile Rust or a bindings crate.** NVRTC compiles CUDA C++ from a string inside the
running process, so the kernels ship as source in the binary and are compiled for whatever GPU is present. It runs
natively on Windows, so the completion-port I/O ring and large pages stay as they are. cuTile Rust
(NVlabs/cutile-rs, now NVIDIA/cuda-rust) was looked at the same day and set aside for now: Linux only, alpha, and it
has no path for codebook quants like IQ3_S. If cuda-rust gains Windows and a SIMT path that fits, a kernel can move
over one at a time. Nothing below depends on that.

## What "fast" means here, from what has already been measured

**The bound is below the program.** One process, 4 threads, queue depth 256: ~155k reads/s from the Sandisk
Optimus 5100. More threads or more processes never add up to more than that, so **no engine, in any language, beats
this drive** on the NVMe tier. Shadow's ceiling is the same as Strata's. What Shadow buys is a system we own end to
end, safe by construction where Rust can make it so, and scheduling we can change.

The reference figure, measured on this PC on 2026-10-26 with Strata 0.1.45 at the shipped config (36 GiB pinned,
`--spec 4`, `STRATA_STAGE_AUDIT=1` on): **41.75 tok/s median agent decode**, n = 10 turns, 2 runs. It is context,
not a gate. Shadow is not benched against Strata, because benching against it means running it. Shadow is benched
against its own earlier stages and its own noise floor.

**Expect to be slow for a long time.** A scalar CPU forward pass is seconds per token. The first GPU path will be well
under 10 tok/s. Most of Strata's speed is in the memory tiers, speculative decoding and launch-overhead work, and
those come in stages 3 to 5. Say this at the start so nobody reads stage 2 as a regression.

## The machine

| | |
|---|---|
| GPU | GeForce RTX 5070, **12 GiB** (12,227 MiB), compute capability **12.0** (`sm_120`) |
| RAM | 63.8 GiB. Windows plus the desktop take ~10 GiB, so ~50 GiB is usable |
| drive | Sandisk Optimus 5100 500 GB NVMe, ~155k direct 4 KiB-aligned reads/s at best |
| CPU | AVX-512 F/BW/VL/VNNI/VBMI present |
| OS | Windows 11 Pro for Workstations |
| CUDA at run time | `nvcuda.dll` (the driver), `nvrtc64_130_0.dll` + `nvrtc-builtins64_134.dll` (CUDA 13.4, `bin\x64`) |

## The model, from the file itself

`J:\Strata-work\models\swift15\IQ3_S\...-00001-of-00002.gguf` + `-00002-of-00002.gguf`, 41.77 + 36.04 GB,
1,224 tensors, 77.80 GiB of tensor data. Every number here was read from the GGUF key-value section and the tensor
offsets, not from the filename. The file sits in Strata's working folder, but it is a GGUF model file, not a Strata
artifact. Moving it under `J:\Shadow\models\` is a copy, not a dependency.

| | |
|---|---|
| architecture | `qwen4exp`, `general.size_label 512x56B`, quantized by ISTA DASLab, `file_type 26` |
| layers | **48 blocks: 36 recurrent (J) + 12 full-attention (K)**, `full_attention_interval 4`, so K is `layer % 4 == 3` |
| attention | 24 heads, **2 KV heads** (GQA 12:1), head and key length 256 |
| hidden / MLP | 2,560; **512 routed experts at 640** intermediate; SSM inner 6,144, state 128, groups 16, conv 4 |
| context | **262,144** native; no yarn keys in this file |
| vocab | 248,320, `pre=qwen35`, model `gpt2`; chat template 9,993 bytes, ChatML **with** a system role |
| per layer, measured | J: **1.032 GiB/layer** (37.14 GiB over 36). K: **1.086 GiB/layer** (13.03 GiB over 12) |
| per layer, split | routed experts 0.961 GiB (J) / 1.020 GiB (K); attention 0.041; shared expert 0.020 |
| embeddings | `per_layer_token_embd` **26.82 GiB**: 35% of the model, and in no layer |

What this means for 12 GiB of VRAM: attention plus shared experts are ~2.9 GiB across all 48 layers and always
resident. KV, the token embedding, the output head and scratch come next, and stage 0 measures each. Whatever is
left is the expert cache. The ~49 GiB of routed experts live across VRAM, pinned RAM and the drive, and that split
is the engine.

## The protocol, which is not negotiable

From `BENCH_NOISE.md`: **≥ 8 runs per arm, interleaved** (never blocked), **`decode_warm`** (drop each run's first
pass), **paired turn by turn**, and report **median, p90 and worst**. A change that does not clear the floor is a
**tie** and is written up as a tie.

The floor in `BENCH_NOISE.md`, **10.1% median, 27.0% p90, 51.9% worst**, was measured on Strata. Shadow measures
its own the first time it can run a session (stage 4), with the same method: one configuration, 8 interleaved
launches, the pairwise spread at each turn. Until then, the Strata floor is the working figure.

**The RAM rule, from experience.** The pinned tier cannot be paged out. Free RAM once fell to 0.4 GiB and the PC froze
hard. Shadow enforces this in code, not in a doc: **it refuses to start a pinned tier that leaves less than 4 GiB
available**, it re-checks after every allocation, and there is no flag that lowers the floor below 4. (The 2 GiB
setting's +10% is on record as unsafe; it cost a hard hang.)

## The stack

**`crates/shadow-cuda`**: the driver API and NVRTC, bound by hand.
- Loaded with `LoadLibraryW` / `GetProcAddress` at run time, so a machine without a GPU still builds and still runs
  the CPU reference. Driver entry points are resolved by their versioned names (`cuMemAlloc_v2` and so on).
- The surface is small, about 40 functions:
  - context: `cuInit`, `cuDeviceGet`, `cuDeviceGetAttribute`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuMemGetInfo`
  - modules: `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`
  - memory: `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemsetD8Async`, `cuMemHostRegister`, `cuMemHostUnregister`
  - streams and events: `cuStreamCreate`, `cuStreamSynchronize`, `cuEventCreate`, `cuEventRecord`, `cuEventQuery`, `cuEventElapsedTime`
  - graphs, later: `cuStreamBeginCapture`, `cuStreamEndCapture`, `cuGraphInstantiate`, `cuGraphLaunch`
  - NVRTC: `nvrtcVersion`, `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetProgramLog`, `nvrtcGetCUBIN`, `nvrtcGetPTX`, `nvrtcDestroyProgram`
- Safe wrappers own their resources. `DeviceBuffer<T>` frees on drop. A pinned host region is
  `shadow-host::Buffer` (already page-aligned, already large-page capable) registered with `cuMemHostRegister`, and
  unregistered before the `Buffer` can drop. Launch is an `unsafe fn` whose argument list is checked against a
  per-kernel signature (below).
- **No bindgen and no bindings crate.** The same rule as the old seam: a hand-written mirror plus a drift test. The
  test reads the kernel sources as text, pulls out every `extern "C" __global__` signature, and asserts each matches
  its Rust declaration. That is the pattern `shadow-abi/src/header.rs` already uses, pointed at our own `.cu` files.

**`kernels/*.cu`**: CUDA C++ source, compiled in-process.
- Pulled into the binary with `include_str!`, so there is one file per kernel family and the binary carries its
  kernels.
- **Self-contained:** no `#include` of toolkit headers, because a user's PC has the NVRTC DLLs but not the CUDA
  headers. f16 and bf16 conversion goes through inline PTX (`cvt`) in one small header of our own, prepended to
  every program.
- Compile options are fixed and recorded: `--gpu-architecture=sm_<cc>` from the device, `--std=c++17`,
  `-default-device`, **no `--use_fast_math`** (parity matters more than the last few percent; add it per kernel
  later, measured, if ever), and `-lineinfo` in debug builds.
- **Ask for CUBIN, not PTX** (`nvrtcGetCUBIN`), so the driver does no second JIT pass. PTX is the fallback only for a
  device newer than this NVRTC knows about.
- **Cache:** `%LOCALAPPDATA%\shadow\kernels\<sha256 of source + options + NVRTC version + arch>.cubin`. The first
  start compiles, which takes seconds and is measured and logged. Every later start loads the cache. A compile error
  is fatal and prints NVRTC's log in full.
- Redistribution: `nvrtc64_*.dll` and `nvrtc-builtins64_*.dll` are expected to be on the CUDA EULA's redistributable
  list. **Check that clause before shipping a release**; until then the engine finds them in the installed toolkit.

**`crates/shadow-model`**: GGUF, tokenizer, the forward pass.
- A GGUF reader of our own: header, key-values, tensor table, offsets. Weights are read through the I/O ring, never
  memory-mapped, so the tier decides where each byte lives.
- Dequantization for every type in the file: IQ3_S for the routed experts, plus whatever stage 0 finds the other
  1,224 tensors use. Each one is written twice, once in Rust as the reference and once as a kernel, and the two are
  tested against each other.
- A tokenizer of our own: byte-level BPE from the GGUF's vocab and merges, with the `qwen35` pre-tokenizer split
  written by hand rather than as a regex with look-ahead. The chat template is hard-coded ChatML with a system role,
  not a Jinja engine. Only the parts of the 9,993-byte template that requests actually use are implemented, and each
  is tested.

**`crates/shadow-cpu`**: the RAM tier's experts.
- IQ3_S expert matrix-vector products in Rust with `std::arch` AVX-512 (VNNI, VBMI), whose target features are
  stable in Rust 1.89. Detection happens at run time, with a scalar fallback that is also the test oracle.
- This is the ~52 ms of a ~115 ms decode window that was measured in the old engine: the single largest piece, and
  plain Rust.

**`crates/shadow-host`**, kept: the I/O ring (completion port, `FILE_FLAG_OVERLAPPED`, 4 issuing threads),
`Buffer`, the per-drive scheduler thread, placement, the router's bookkeeping and the transient-window rules. They
were written against an interface, and that interface stays.

**`crates/shadow-serve`**: HTTP. **hyper** + **serde_json**, declared in `Cargo.toml` with the reason, as decided
2026-10-26. Streaming `/v1/chat/completions` and `/health` are required; `/v1/models`, `/v1/status`, `/metrics` and
`/props` should exist. The answers are shaped as the old plan recorded them, because the existing clients read those
shapes.

**Retired**, kept in git and not deleted in this step: `crates/shadow-abi`, `include/shadow/c/abi.h`,
`shadow-host/src/ffi.rs` (the DLL the C++ engine loaded) and `shadow-host/src/protocol.rs` (the GEN/T/DONE line
protocol). They are the join to an engine that is no longer in the plan. They are removed in their own commit once
stage 2 runs without them. **Done 2026-10-27** (version 0.1.4): removed with `shadow-host`'s seam tests, its
`shadow-hostd` / fake-engine binaries and its facade-finding `build.rs`; `Status` and `Ring` moved into
`shadow-host/src/status.rs`; `shadow-host` is a plain Rust library.

## The oracle: how we know the math is right without Strata

This is the hardest problem in the plan, so it gets settled first. A forward pass that is subtly wrong still
produces fluent text, and quantization alone guarantees that no two implementations agree bit for bit. In order of
preference:

1. **llama.cpp**, if it loads `qwen4exp`. Its DLLs are already in `bin/` (they serve the memory embeddings). This
   costs half a day to find out. If it loads the model, dump logits for a fixed prompt set once, freeze them as
   fixtures, and compare top-1 token and logit distance. That is an outside reference that is not Strata.
2. **The public reference implementation of the architecture** (the modeling code published with the model family),
   read to write `docs/ARCH.md`, and run on CPU only if the weights it needs can be made from this file.
3. **Self-consistency plus perplexity**, if neither is available. The scalar Rust reference is written straight
   from `ARCH.md`, every kernel is tested against it, and the whole thing must reach a perplexity under a fixed bound
   on a frozen text, with sensible greedy output on the frozen eval prompts. This is weaker: it catches most bugs and
   cannot prove the last few. The writeup says so.

Whichever applies, the fixtures are generated once, checked in under `eval/fixtures/`, and never regenerated
silently. `eval/cases.jsonl` stays frozen, as before.

## The stages

Each stage ends with something that produces tokens, and each has one gate. A stage is not done until its gate is
green on this PC.

### Stage 0: the spec, the file, the tools
- `docs/ARCH.md`: every operation of one J block, one K block, the PLE path, the router, the shared expert and the
  head, written as math with tensor names and shapes taken from the GGUF. Every open question is listed as one.
- The GGUF reader and a `shadow inspect` command that prints the tensor table, the quant type per tensor, and the
  VRAM math above with real numbers.
- `shadow-cuda`: bindings, the NVRTC compile and cache, and one trivial kernel (`vector_add`) compiled, launched and
  checked on the 5070.
- The tokenizer.
- The oracle decision, with its fixtures.

**Gate:** `shadow inspect` accounts for all 1,224 tensors and 77.80 GiB. The tokenizer reproduces the fixture token
ids exactly. `vector_add` passes from a cold cache and from a warm one. The kernel-signature drift test is green.

### Stage 1: the CPU reference forward pass
- Scalar Rust, f32 accumulation, every layer, one token at a time, every weight dequantized on the fly from the
  file through the ring. It is slow on purpose; it is the yardstick everything after it is measured against.
- KV for the 12 K layers and the recurrent state for the 36 J layers, kept in RAM.

**Gate:** the oracle comparison from the section above passes. Greedy decoding of the frozen prompts is identical
across two runs.

### Stage 2: the GPU dense path
- NVRTC kernels for norms, RoPE, attention (with int8 KV), the J-layer recurrence and conv, the router, the shared
  expert, the combine step and the head. Routed experts: an IQ3_S matrix-vector kernel for the ones in VRAM, and
  the stage 1 CPU path for the rest.
- One stream, no graphs yet. Simple first.

**Gate:** every kernel matches its Rust reference within a tolerance written next to the test. Whole-model logits
match stage 1 to the same top-1 token on every fixture position, with the maximum logit distance recorded.

### Stage 3: the tiers
- VRAM expert cache, pinned RAM tier (AVX-512 experts in `shadow-cpu`), NVMe tier through the ring. A placement
  profile in **our own format**, learned from Shadow's own routing traces (Strata's `STRP` profiles are not read).
- The 26.82 GiB per-layer embedding streams from the drive. Pinning it is an arm to measure later, not a default.
- The 4 GiB RAM rule, enforced.

**Gate:** where an expert was served from (VRAM, RAM or drive) does not change the answer: the same top-1 tokens on
the fixtures for every forced placement. A RAM-headroom test lowers available memory artificially and proves the
engine refuses rather than pins. A long run's minimum free RAM is logged.

### Stage 4: the server and the floor
- `shadow-serve` on the engine in-process. The harness (`tools/shadow_harness.py`) and `shadow-eval` point at it.
- A bench of our own, `shadow-bench`, implementing `BENCH_NOISE.md`: interleaving, `decode_warm`, turn-by-turn
  pairing, median/p90/worst, the full launch line recorded byte for byte, and RAM and GPU state sampled during the run.
- **Measure Shadow's own noise floor.**

**Gate:** `shadow-eval` passes against the frozen `eval/cases.jsonl`. Shadow's floor is measured and written into
`BENCH_NOISE.md` beside Strata's.

### Stage 5: speed, one lever at a time, each against the floor
In the order the old engine's measurements say they pay:

| lever | why, from what was measured before |
|---|---|
| more experts in pinned RAM | ~0.7 tok/s per GiB on the old engine (36/40/44/45.5 GiB → 10.7/13.2/15.5/17.1) |
| speculative decoding | draft acceptance was the strongest single factor in decode speed (+0.44 within an arm) |
| CUDA graphs for the decode step | ~48 layers × ~10 kernels is ~500 launches per token without them |
| prefill GEMM | tensor-core kernels (`mma.sync` through inline PTX, via NVRTC) for prompt processing |
| huge pages for the pinned tier | already built in `shadow-host` (`SHADOW_HUGE_PAGES=1`); it needs the lock-pages privilege this PC lacks |
| the expert cache's eviction | adaptive swaps were measured but confounded by the cold region; re-measure here |

The draft model for speculative decoding has to come from somewhere that is not Strata. Strata's MTP draft
(`mtp-basalt-ft1`) is a Strata artifact. Options: an MTP head inside the GGUF, if stage 0 finds one; a small
same-vocab model; or prompt-lookup drafting, which needs no model. Settle it in stage 0's open questions.

Every lever is an arm: ≥ 8 interleaved runs against the configuration without it, judged against Shadow's floor.
A tie is written up as a tie.

## What has to be true before stage 0 starts

- **This document is the plan.** `README.md` still says "the kernels stay C++" and "Rust never compiles a kernel".
  Update it in the same change, or the next session will follow the README.
- **The license line stays honest.** Shadow is MIT, keeps Strata's notice for what came across (the measurements,
  the plan's history, the harness), and copies no Strata source. Writing the kernels from `ARCH.md` rather than from
  Strata's `.cu` files is what keeps that true. Reading MIT code is allowed, but the rule here is independence, not
  only licensing.
- **The model file is the only input.** Copy it under `J:\Shadow\models\`, record its SHA-256, and stop reading
  `J:\Strata-work` for anything.

## The honest shape of it

This is a bigger project than the one it replaces. The old plan wrapped a working engine. This one has to make the
model produce a correct token before it produces a fast one, and stages 0 to 2 are mostly correctness work with
nothing to show on a speed chart. The ceiling is the same drive: when Shadow is done, it should be about as fast as
Strata on this PC, not faster, and anything above that has to clear the floor like everything else.

What it buys is the thing the old plan's last section admitted it lacked: **the engine is ours.** Every kernel is
source we wrote, compiled by a library that ships with the driver, launched by a host whose safety rules are the
language's. Nothing else in the stack needs the C++ toolchain, and nothing runs that we cannot read and change.

Start with stage 0, and settle the oracle first.
