# Status

Where the plan ([SHADOW_PLAN.md](SHADOW_PLAN.md)) stands. Updated 2026-10-27.

Build and run: `cargo build --release -p shadow -p shadow-bench`, then `target/release/shadow.exe` with `inspect`, `kv`,
`stats`, `dump`, `generate` (`--chat`, `--gpu`), `compare` and `serve`. Any process that loads the model runs under
`tools/run_capped.ps1` (hard working-set cap, killed below 4 GiB available).

## Stage 0: done

| item | state | evidence |
|---|---|---|
| `docs/ARCH.md` | written from the file, the published reference and llama.cpp's converter | |
| GGUF reader + `shadow inspect` | all 1,224 tensors, 77.801 GiB, every size from its type | `tests/real_model.rs`, `docs/model-inspect.txt` |
| `shadow-cuda`: driver API + NVRTC by hand, CUBIN cache | green on the RTX 5070 (cc 12.0, NVRTC 13.4) | `crates/shadow-cuda/tests/` |
| kernel-signature drift test | green, both directions | `tests/drift.rs` |
| tokenizer | **46 texts (429 ids) identical to the checkpoint's own tokenizer.json**: Latin, CJK, Hangul, emoji, combining marks, contractions, code, special tokens | `eval/fixtures/tokenizer.json`, `tests/tokenizer_fixture.rs` |
| NFC normalization | the checkpoint's tokenizer normalizes to NFC, the GGUF does not say so; matches Python's unicodedata on 4,000 strings | `nfc.rs`, `tests/nfc_fixture.rs` |
| chat template | **12 conversations identical to the checkpoint's template** | `eval/fixtures/chat.json` |
| the oracle | the published `transformers` `qwen4_exp` reference, run on the BF16 checkpoint read by HTTP range requests (`tools/hf_range.py`, cache on `I:\shadow-ref`) | |

## Stage 1: done for all 48 layers, with one known gap

The CPU forward pass (`forward.rs`, written from ARCH.md) against the reference, 12-token prompt:

- **On the BF16 checkpoint** (`eval/fixtures/reference`): every layer's output stream has cosine >= 0.994 with the
  reference; routing picks 8-10 of the same 10 experts. The weakest stream is the MoE of layers with 2-bit experts
  (0.92-0.98), which is quantization noise.
- **On Shadow's own GGUF, dequantized and mapped back into the reference's layout** (`eval/fixtures/reference_gguf`,
  no quantization difference): **all 23 streams of layers 0-3 have cosine 1.00000, relative error 0.0000, max |diff| <=
  1e-4, and the router picks the same 10 experts at every token of every layer.** That covers the 4-stream
  hyper-connections, the DeltaNet (with the tiled value-head order), gated attention, the layer-1 PLE, the MoE with its
  shared expert, and the final read. The GPU backend passes the same test.
- `crates/shadow/tests/reference.rs` holds those comparisons.
- **All 48 layers** (the same comparison with no download: the GGUF is the source; 8,396 `shadow dump` calls, 14 min,
  48 MB of fixture kept on `I:\shadow-refixtures
eference_gguf48`, regenerate with
  `make_reference_fixture.py --layers 48 --weights gguf`): **243 streams, minimum cosine 1.00000, relative error
  0.0000, max |diff| 1e-4, and the router picks the same 10 experts at every token of all 48 layers** - on the CPU
  backend and on the GPU backend. That covers all 36 DeltaNet and 12 attention layers and every quant type.

**The sparse indexer is gated too** (`eval/fixtures/reference_gguf_long`): a 2,600-token prompt through layers 0-3,
compared at 12 positions around where selection starts (position 2,051, the first with more than 512 complete blocks).
At positions 2,051 to 2,599 the reference drops 4 to 552 of the visible positions in layer 3; **Shadow attends to
exactly the same positions at all 12 kept positions** (0 differing), every stream has cosine 1.00000 and the routing
is 10/10, on the GPU backend (150 s). Positions up to 2,050, where the indexer keeps everything, match as before.

Known gaps: one prompt family (this plan's own text and a 12-token sentence); decoding many tokens is only checked by
greedy runs, not stream by stream; no sampling (greedy only).

## Stage 2: the whole forward pass runs on the GPU, gated; not fast yet

- **`GpuSession`** (`crates/shadow-cuda/src/session.rs`, kernels in `kernels/forward.cu` + `matvec.cu`): the streams,
  every DeltaNet state, every KV cache with the indexer's keys and the n-gram state live on the device; norms,
  hyper-connection glue, the DeltaNet step, attention with the sparse indexer, the MoE glue and the n-gram memory are
  kernels. On the host still: the embedding and n-gram rows (dequantized and uploaded), the router's top-10 (512 logits
  come down), the indexer's top-512 past 2,051 positions (scores come down), and the routed experts, read from the file
  and uploaded per use.
- **Gated against the reference on Shadow's own weights:** 4 layers (`eval/fixtures/reference_gguf`), **all 48
  layers** (243 streams, cosine 1.00000, routing identical at every layer) and the **2,600-token sparse-indexer
  prompt** (identical attended positions, every stream exact). `tests/reference.rs` holds the first and (ignored, a
  minute) the last.
- **Speed: 0.347 s/token** generating (`shadow generate ... --gpu`), against 0.40 with only the matvecs on the GPU and
  2.35 on the CPU. The answer is unchanged ("17 times 23 is 391."). It is not faster because the time is in the 480
  expert slices read from the file and uploaded per token, and in the reference matvec kernels (15-23 GB/s on the
  IQ2/IQ3 experts). Both are the next work: stage 3's VRAM/RAM tiers keep experts resident, and fast matvec kernels.
- Every quant type dequantizes on the GPU bit for bit like the Rust reference (`kernels/quants.cuh`).
- **Fast matvecs (2026-10-27, `kernels/matvec.cu`)**: `matvec_fast_*` decodes 8 consecutive weights per lane with a
  block's scale, grid row and sign byte decoded once (each weight computed with the reference's operations; only the
  f32 summation order differs), and `matvec_batch_*` runs all routed experts of a layer in one launch (blockIdx.y =
  expert, a device table of addresses). The IQ grid tables moved from `__constant__` to global memory: constant memory
  serializes a warp whose lanes read different rows, which every IQ decoder does (that alone took the reference
  IQ2_S kernel from 0.035 to 0.016 ms). `tests/matvec.rs` holds all three kernels to the CPU on real tensors; the
  dequant kernels stay bit-exact. Per expert matrix, batched vs reference: IQ2_S 3.9x, IQ3_XXS 3.7x, IQ3_S 3.1x,
  Q2_0 1.9x, IQ4_NL 1.3x. **Not faster: the fast single-matrix kernel on F32 (0.4x), Q2_0/IQ4_NL (~1.0x) and the
  K-quants** - so single dense matvecs use it only for IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS.
- **8 interleaved runs per arm**, both with the 24 GiB RAM tier, "What is 17 times 23?", 64 tokens: before (308f74f)
  median 0.168 s/token, p90 0.169; batched **0.135**, p90 0.138. Paired: median 19.4%, min 17.9%, max 21.3%; identical
  tokens in all 16 runs. Routed-expert matvec time per token (synchronized profile): 43.8 -> 10.5 ms. Clears Strata's
  10.1% median floor, not its 27.0% p90 (Shadow's own floor is unmeasured).

## Open

1. **Q2_0** (9 down-projections) has no independent reference tool (gguf-py lacks it); it follows upstream ggml's
   definition and is checked by hand and by the end-to-end match above, which runs through it.
2. ~~The sparse indexer~~: implemented and gated (see Stage 1).
3. **Speculative decoding's draft**: the GGUF has no MTP tensors (ARCH.md, question 7).
4. **Stage 3** has two of its three tiers: the VRAM expert cache and the pinned RAM tier with the 4 GiB rule in code
   (see Stage 3 below). The drive is read with plain blocking `seek_read`, not yet through `shadow-host`'s ring.

## Stage 3: VRAM cache + pinned RAM tier, gated for correctness; one interleaved comparison

**Where a token's time went before the RAM tier** (`SHADOW_PROFILE=1`, synchronized at every phase, 86 steps, VRAM
cache 4.57 GiB): expert reads from the file 43%, expert uploads 19%, routed-expert matvecs 20%, mixers/norms/glue 14%,
router top-10 on the host 1%. That is why the RAM tier came before moving the router and indexer onto the GPU.

- **VRAM expert cache**: one arena of 2.54 MiB slots (gate + up + down), LRU, the current token's experts never
  evicted by its own misses; the free VRAM less `VRAM_RESERVE` (1.5 GiB) by default, `--expert-cache-gib G` (`0`: none).
- **Pinned RAM tier** (`crates/shadow-cuda/src/pinned.rs`, `RamTier` in `session.rs`): `VirtualAlloc` +
  `cuMemHostRegister`, experts packed, filled once at the start in placement order, never evicted; a VRAM miss on one
  of them is a DMA copy instead of a file read. `--ram-tier-gib G`, `--placement FILE`.
- **The 4 GiB rule, in code**: `pinned::RAM_FLOOR` (not configurable). Checked before allocating, after the pages are
  locked, every GiB while filling, and at the end. `tests/pinned.rs`: refuses when available RAM is artificially low
  (a fake RAM source, nothing consumed), notices RAM falling after pinning, refuses `available - 2 GiB` on the real
  machine, and the GPU copies pinned memory back byte for byte. `headroom()` unit test includes the overflow case
  (`u64::MAX`), which the first version got wrong.
- **Placement profile, Shadow's own format** (`crates/shadow-model/src/placement.rs`, `SHRC`): per-(layer, expert)
  routing counts; `--record-routing FILE` merges a run into it. `profiles/routing-6prompts.shrc`: 675 tokens from six
  chat prompts (code, explanation, history, translation, arithmetic word problem, poem), none of them the evaluation
  prompts below.
- **Gate (2026-10-27):** greedy, `--chat`, 48 tokens, two prompts ("What is 17 times 23?" and a South American
  rivers question): **identical token ids** for drive only (no cache, no RAM tier), RAM + drive, VRAM + RAM + drive,
  and VRAM + drive. `tests/reference.rs` (with the ignored long-prompt GPU test) passes. Lowest available RAM in a 24 GiB
  pinned run: 18.1 GiB.
- **Speed, 8 interleaved runs per arm** (alternating order, "What is 17 times 23?", 64 tokens, generation s/token):
  VRAM cache only: median 0.217, p90 0.238, worst 0.238. VRAM cache + 24 GiB RAM tier: median **0.171**, p90 0.173,
  worst 0.173. Paired, B is faster in all 8 pairs: median 20.9%, min 12.2%, max 28.2%. **It clears the median noise floor
  (10.1%) but not the p90 floor (27.0%), and both are Strata's floor, not Shadow's** (unmeasured until stage 4). Not
  `decode_warm` (one generation per launch). The 24 GiB fill adds ~20 s at start.
- Before the RAM tier, single runs: 0.371 s/token without the cache, 0.215 at 2 GiB, 0.176 at 4.57 GiB.
- **RAM tier size, 8 rotating runs per arm (2026-10-27, b4c9f8b, batched kernels, 64 tokens):** 24 GiB median 0.136
  s/token (p90 0.148, lowest available RAM 13.9 GiB); 28 GiB 0.118 (p90 0.126, 9.8 GiB); 32 GiB **0.102** (p90 0.108,
  **5.6 GiB**). Paired against 24 GiB: 28 GiB 13.3% (min 11.3%), 32 GiB 25.1% (min 23.5%); identical tokens in all 24
  runs. A first round at 34 GiB (0.099 s/token) left 4.3 GiB available and was stopped as too close to the floor.
- **`--ram-tier-gib auto`** (`pinned::auto_tier_bytes`): the RAM available inside `GpuSession::with_tiers` *after*
  `Model::load` and the dense GPU upload, less `AUTO_MARGIN` (the 4 GiB floor + 6 GiB for what moves after the fill),
  re-checking the 4 GiB floor every 256 MiB during the fill. (In 0.1.7 `shadow serve` evaluated `auto` before
  `Model::load`, which asked for ~37 GiB before the 4.14 GiB of dense weights were loaded and tripped the 4 GiB
  watchdog; fixed in 0.1.8: 42.54 GiB available after dense upload -> 32.54 GiB pinned, lowest available 7.8 GiB.)
- **The drive tier stays on plain buffered reads; two attempts to speed it up lost (2026-10-27, both reverted):**
  1. All of a layer's slice reads at once on scoped threads (buffered `seek_read`, one synchronous handle): correct,
     but slower (p1 0.223 vs 0.167 s/token, no-cache 0.435-0.520 vs 0.29). Windows serializes I/O on a synchronous
     file handle, so the threads only added spawn cost.
  2. `shadow-host`'s direct-access ring (overlapped, unbuffered, a layer's reads in flight together): correct, but
     about **2x slower** in generation (p1 0.360, p2 0.317, no-cache 0.77-0.78 s/token). On random cold offsets it
     wins (`crates/shadow-host/examples/expert_read_bench.rs`: 2.2-3.5 GB/s at depth 4-64 vs 1.4-1.8 buffered), but
     real expert reads repeat, and buffered reads are served from Windows' file cache at memory speed while
     unbuffered ones always go to the NVMe. Untried: overlapped *buffered* reads (queue depth and the cache), and
     reading into pinned staging buffers.

## Stage 4: HTTP server (`shadow-serve`), `shadow-bench`, and Shadow's noise floor

- **`shadow-serve` (`crates/shadow-serve`, `shadow serve`):** `hyper` + `serde_json` OpenAI-compatible server on
  127.0.0.1 (`--port`, `--ctx`, `--expert-cache-gib`, `--ram-tier-gib`, `--placement`). Endpoints:
  `/v1/chat/completions` (streaming SSE and non-streaming, returns `timings.predicted_per_second`, `prompt_n`,
  `predicted_n`, `cache_n`), `/health`, `/v1/models`, `/v1/status`, `/props`, `/metrics` and `/shutdown`. Keeps the
  token history of the last request on the `GpuSession` and reuses the KV + DeltaNet + n-gram state when the next
  conversation's token prefix matches (`cache_n` tokens skipped); otherwise calls `GpuSession::reset()`. Verified by
  3 unit tests and `tools/serve_check.py` (15/15 live checks across streaming, multi-turn prefix reuse, reset and
  metrics).
- **Tool calling + `shadow-eval run` (`0.1.11`):** transcribed the `tools` system preamble and assistant `tool_calls`
  branches of both Qwen3.8-Flash-Next (`<tools>` + `<tool_call><function=...><parameter=...>`) and `RecursiveSI/flint`
  (`<tools>` + `<tool_call>NAME<arg_key>...<arg_value>...`) in `crates/shadow-model/src/chat.rs`, held to `transformers`
  `apply_chat_template` across 14/14 Qwen conversations (`eval/fixtures/chat.json`) and 14/14 Flint conversations
  (`eval/fixtures/chat_flint.json`). `shadow-serve` parses `<tool_call>` blocks into OpenAI `tool_calls` responses (preserving
  argument key order so multi-turn tool sessions keep their token prefix cache warm). Verified by `tools/serve_check.py`
  (16/16 live checks), a multi-turn `tools/shadow_harness.py` tool execution against `shadow serve`, and `shadow-eval run`
  + `shadow-eval answers` (`14/14` correct).
- **`shadow-bench` (`crates/shadow-bench`) and Shadow's noise floor (`bench/floor-2026-10-27`):** launches `shadow serve`
  under `tools/run_capped.ps1`, waits for available RAM to settle before each launch, runs a 6-turn conversation (96
  tokens/turn, greedy), drops turn 0 (`decode_warm`), and computes turn-matched pair differences across runs. Over **8
  launches x 5 warm turns (140 pairs)** with `--ctx 8192 --ram-tier-gib 30 --placement profiles/routing-6prompts.shrc`:
  - **`decode_warm` median: 10.20 tok/s** (p90 slow end 10.04, worst 10.04, best 10.29 tok/s; 98.0 ms/token)
  - **Shadow's noise floor: median 0.8%, p90 2.0%, worst 3.6%** (against Strata's 10.1% / 27.0% / 51.9%)
  - **Lowest available RAM across all 8 launches: 9.6 GiB**; peak GPU memory 10,319-10,422 MiB; **0 answer differences**
    across all 48 turns.

## Stage 5: Speed levers (`0.1.12`)

- **What failed first (`Fast<BF16>::dot8` 8-per-lane uncoalesced loads, reverted for `matvec_pair_bf16` / `hc_down_inject_bf16`):**
  having each warp lane read 8 contiguous `bf16` weights (`16` bytes/lane strided across a 512-byte cache region) dropped
  `BF16` matvec throughput from **455.5 GB/s to 396.5 GB/s** (`tests/matvec.rs`), because `BF16` has no block-scale header
  to amortize and strided `i = lane; i < cols; i += 32` loads coalesce into a single 64-byte cache sector per warp
  instruction. Kept `BF16` on coalesced strided loads (`matvec_bf16`, `matvec_pair_bf16`, `hc_down_inject_bf16`).
- **Register-immediate `iq4nl_val` + `Fast<Q4K/Q5K/Q6K/IQ4XS>` (`kernels/quants.cuh`, `kernels/matvec.cu`):**
  - Replaced `__constant__ signed char kIQ4NL[16]` with a register-immediate byte lookup (`iq4nl_val` over four `uint32_t`
    constants), eliminating 16-way warp serialization on `__constant__` memory across 44 GiB of `IQ4_NL` and `IQ4_XS`
    weights.
  - Added `Fast<Q4K>`, `Fast<Q5K>`, `Fast<Q6K>`, and `Fast<IQ4XS>` specializations (`matvec_fast_*` and `matvec_batch_*`).
    Bit-exact with the reference dequantizers (`crates/shadow-cuda/tests/dequant.rs`); `crates/shadow-cuda/tests/matvec.rs`
    measured:
    - `Q4_K`: **118.5 -> 215.6 GB/s** (`1.8x` fast, `239.5 GB/s` `2.0x` batch)
    - `Q5_K`: **139.6 -> 244.7 GB/s** (`1.8x` fast, `268.2 GB/s` `1.9x` batch)
    - `Q6_K`: **263.6 -> 309.4 GB/s** (`1.2x` fast, `333.3 GB/s` `1.3x` batch)
    - `IQ4_NL` batch: **213.1 GB/s (`2.0x`)** (`0.004 ms`/expert)
    - `IQ4_XS`: **58.3 -> 105.3 GB/s** (`1.8x` fast, `212.1 GB/s` `3.6x` batch)
- **Fused forward-pass kernels (`kernels/forward.cu`, `kernels/matvec.cu`, `crates/shadow-cuda/src/session.rs`):**
  - `hc_down_inject_bf16` (`-288` launches/token), `hc_write_rmsnorm_groups` (`-95` launches/token), `moe_combine_qwen`
    (`-144` launches and `-48` `cuMemsetD8` calls/token), `matvec_pair_bf16` + `l2norm_qk_heads` in `delta_net` (`-72`
    launches/token), and `matvec_pair_bf16` + `head_norm_rope_qgk` + direct `at.v`/`at.raw` writes in `attention` (`-60`
    launches/token). Synchronized GPU compute time (`SHADOW_PROFILE=1`) dropped from **52.5 ms/token to 31.6 ms/token
    (`-40%`)**.
- **Async `copy_stream` DMA + pinned file-miss staging + early shared-expert overlap (`crates/shadow-cuda/src/session.rs`):**
  - Added a dedicated non-blocking `copy_stream` (`CU_STREAM_NON_BLOCKING`) and a `k * slot_bytes` (`25.4 MiB`) pinned
    staging buffer (`PinnedRegion`) for file-tier misses (read in monotonic file-offset order `down -> gate -> up` via
    `expert_into`, zero heap allocations).
  - Launched Qwen's shared expert (`ffn_gate_shexp`, `ffn_up_shexp`, `swiglu`, `ffn_down_shexp`, `ffn_gate_inp_shexp`) on
    stream 0 immediately after downloading router logits so GPU SMs compute the shared expert concurrently with host
    `route()` and `copy_stream` PCIe DMA uploads.
- **Gated & benchmarked (`bench/stage5-kernels-2026-10-27` vs `bench/floor-2026-10-27`):**
  - Passed `cargo test --release -p shadow --test reference` (7/7) and the 48-layer GPU comparison (`shadow compare ... --gpu`:
    all 48 layers `cosine 1.00000`, `10/10` routing in `4.1 s`).
  - Over **8 launches x 5 warm turns (140 pairs)** with the identical `--ctx 8192 --ram-tier-gib 30 --placement profiles/routing-6prompts.shrc`
    protocol:
    - **`decode_warm` median: 15.34 tok/s** vs `10.20 tok/s` baseline (**`+50.4%` (`1.504x`) speedup**, p90 slow end `15.12`,
      worst `15.12`, best `15.38 tok/s`; single-prompt decode `16.95 tok/s` / `0.059 s/token`)
    - **Within-arm noise floor: median 0.3%, p90 1.3%, worst 1.7%**
    - **Lowest available RAM: 12.9 GiB** (up from 9.6 GiB); peak GPU memory 10,012-10,019 MiB (down from 10,422 MiB);
      **0 answer differences** across all 48 turns.

## Second architecture: `RecursiveSI/flint` (`Flint-40B-A2.5B-abliterated-NVFP4.gguf`)

- **Model**: 40 layers (layers 0-1 dense SwiGLU, layers 2-39 128-expert top-6 sigmoid MoE + ungated shared expert),
  GQA 32/8 x 128 (`c_dim = 2048`), periodic no-RoPE full-attention layers (`(l + 1) % 4 == 0`), sliding-window 513 +
  YaRN (`beta_fast = 32`, `beta_slow = 1`, `factor = 8`) on RoPE layers, learned per-head `attn_sinks.weight`, and
  GGUF type `40` (`NVFP4`: 64 weights in 36 bytes = four `ue4m3` sub-block scales + 64 `e2m1` nibbles split low/high 32
  bytes per 16-weight group, paired with per-tensor `.scale` companions).
- **What failed first (`CU_MEMHOSTREGISTER_DEVICEMAP` zero-copy PCIe reads for VRAM cache misses, reverted):** mapping
  the pinned RAM tier into GPU address space and reading cold expert slices directly over PCIe in `matvec_batch_nvfp4`
  (while freezing the VRAM cache once full) dropped decode speed from **51.2 tok/s (`0.0195 s/token`) to 32.3 tok/s
  (`0.0309 s/token`)**. Routed-expert matvec time jumped from 4.1 ms to 16.4 ms/token and VRAM cache hit rate fell from
  68.4% to 51.8%: each warp lane in `Fast<NVFP4>::dot8` gathers 9-byte strided segments across 36-byte blocks (~11 GB/s
  effective PCIe throughput from partial cacheline TLPs), whereas coalesced `cuMemcpyHtoDAsync` streams 1.69 MiB bulk
  DMA transfers at ~52 GB/s and LRU replacement adapts the VRAM cache to the active topic.
- **Proven & gated (`0.1.10`):**
  - `NVFP4` CPU & GPU dequant (`dequant_nvfp4` bit-exact with CPU; `ue4m3_half` branchless IEEE shift + `e2m1_double`
    register LUT in `0xc8643210u`), `matvec_fast_nvfp4` (`229.2 GB/s`), `matvec_fast_nvfp4_s` (fused `.scale`),
    `matvec_qkv_nvfp4` (fused Q+K+V + `.scale`), `matvec_batch_nvfp4` (`290.0 GB/s`): `crates/shadow-model/tests/dequant.rs`,
    `crates/shadow-cuda/tests/dequant.rs`, `crates/shadow-cuda/tests/matvec.rs`.
  - Flint pre-tokenizer (`PreTokenizer::Flint`) and `<|role|>` chat template (`render_flint`): **48/48 texts and 12/12
    chat conversations identical to `RecursiveSI/flint`'s `tokenizer.json` and `chat_template.jinja`**
    (`eval/fixtures/tokenizer_flint.json`, `eval/fixtures/chat_flint.json`, `tests/tokenizer_fixture.rs`).
  - **4-layer CPU and GPU reference gate on Shadow's own GGUF weights** (`eval/fixtures/reference_flint_gguf`,
    `tools/make_reference_fixture_flint.py`, `crates/shadow/tests/reference.rs`): **all 13 streams across layers 0-3
    (covering both dense layers, MoE layers with shared + routed experts, RoPE + sliding-window + sinks, and no-RoPE
    full attention at layer 3) have cosine 1.00000, relative error 0.0000, max |diff| <= 2.3e-5, and 6/6 routing match**
    on both CPU and `GpuSession`.
  - **Speed optimizations**: sequential `(layer, part, expert)` sort in `RamTier::fill` (cuts 20 GiB fill from 59.8 s to
    6.7 s warm / 42.3 s cold); fused `16`-slice `matvec_batch_nvfp4` combining 6 routed experts + shared expert gate/up
    in one launch (`swiglu_scaled` over 4,096 elements); fused `moe_combine_flint`, `head_norm_rope_qk`, and cross-layer
    `add_rmsnorm1`; coalesced single `cuMemcpyHtoDAsync` per expert miss + startup VRAM cache pre-warming from `RamTier`.
    On a 128-token chat generation (`--ram-tier-gib 20 --placement profiles/flint-routing.shrc`): **50.0 tok/s prefill
    (`0.020 s/token`) and 51.2 tok/s decode (`0.0195 s/token`)**.

## Things that went wrong, so they are not repeated

- A claim in ARCH.md that two PLE norm tensors were identical rested on summary statistics and six values; they are
  different (max 0.054). `shadow stats` summaries do not prove equality; compare the vectors.
- git's `eval/** text eol=lf` rule rewrote CRLF byte pairs inside a committed `tensors.bin`; `.gitattributes` now marks
  `eval/fixtures/**/*.bin` binary. After adding any binary fixture: `git show HEAD:path | cmp - path`.
- Uncapped, llama.cpp's mmap load took available RAM from 55 to 0.73 GiB in 49 s. Always `tools/run_capped.ps1`.
- `J:\Strata-work\RUN.md` records the three failed llama-embedding runs (HANDOFF.md's route); nothing there is needed.
