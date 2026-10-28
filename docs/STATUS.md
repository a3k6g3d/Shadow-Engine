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
  3. **Kept (0.1.23):** what lost in (1) was the shared handle, not the threads. `shadow-model::readpool` keeps 8
     worker threads, each with its own buffered handles on every shard; a layer's drive misses are queued by
     `fetch_expert` and read together by `read_drive` (256 KiB pieces, straight into the pinned ring / staging buffer).
     `bench/readpool-2026-10-27`, 8 interleaved launches per arm: `decode_warm` median 17.37 -> **18.52 tok/s**
     (p90 16.85 -> 18.42), paired +5.3% median, +2.9% worst, all 40 pairs faster, 0 answer differences.

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

## Stage 5 (`0.1.13`): aligned word loads, per-layer exact-size VRAM cache + exclusive VRAM pinning, `RamTier` Clock ring, and `matvec_pair_*`

- **Aligned word loads in `kernels/quants.cuh` and `kernels/matvec.cu`:**
  - Replaced 1-byte `ld.global.u8` loads in `qf16`, `BF16::at`, `F32::at`, `Fast<Q4K>`, `Fast<Q5K>`, `Fast<Q6K>`, `Fast<IQ4NL>`,
    `Fast<IQ4XS>`, `Fast<Q20>`, `Fast<IQ3XXS>`, and `Fast<IQ3S>` with aligned `*(const u16*)`, `*(const u32*)`, and
    `*(const float*)` loads (`crates/shadow-cuda/tests/dequant.rs` bit-exact; `crates/shadow-cuda/tests/matvec.rs`):
    - `F32` (`matvec_f32`): **360 -> 788.8 GB/s** (`983.4 GB/s` batch)
    - `BF16` (`matvec_bf16`): **455.5 -> 482.8 GB/s** (`652.9 GB/s` batch)
    - `Q6_K`: **309.4 -> 360.8 GB/s** (`381.6 GB/s` batch)
    - `Q5_K`: **244.7 -> 312.9 GB/s** (`328.9 GB/s` batch)
    - `Q4_K`: **215.6 -> 252.1 GB/s** (`275.0 GB/s` batch)
    - `IQ4_NL` batch: **213.1 -> 236.3 GB/s** (`3.2x`)
    - `IQ4_XS` batch: **212.1 -> 238.3 GB/s** (`4.0x`)
    - `IQ3_S` batch: **180.6 GB/s** (`3.8x`); `IQ3_XXS` batch: **160.8 GB/s** (`3.9x`)
- **General `matvec_pair_*` fusion (`PAIR_KERNELS` in `crates/shadow-cuda/src/kernels.rs`, `mv_pair` in `session.rs`):**
  - Fuses any two projections sharing input `x` (`ffn_gate_inp` + `ffn_gate_inp_shexp` in `F32`, `ffn_gate_shexp` + `ffn_up_shexp`,
    `attn_qkv` + `attn_gate`, `attn_k` + `attn_v`, and `ple_key` + `ple_value`) when they share quant type and column count,
    eliminating ~110-145 kernel launches per token and halving input vector reads.
- **Per-layer exact-size VRAM `ExpertCache` + exclusive VRAM-pinned hot experts + `RamTier` Clock promotion ring (`session.rs`):**
  - Partitioned `ExpertCache` per layer at each layer's exact expert size (`1.44` / `1.88` / `2.08` / `2.54 MiB`, `0` internal
    fragmentation) and lowered `VRAM_RESERVE` to `1024 MiB`, raising VRAM capacity from **1,831 slots (`4.54 GiB`) to 2,654
    slots (`5.07 GiB`, `+44.9%`)**.
  - Pinned the hottest ~half of each layer's VRAM slots (`~1,175` experts) permanently in VRAM (`evict_range[l]`), loaded
    directly into VRAM during `RamTier::fill` (cutting session startup from **45 s to 32 s**, `-28.9%`), and excluded them from
    `RamTier` so the 30 GiB `RamTier` covers **`+1,200` additional unique experts** (`16,989` unique experts across `VRAM U RAM`
    instead of `15,789`).
  - Added an exact-size Clock-replaced promotion ring (`promote_slot`) at the cold tail of `RamTier` (`128` slots) so file-tier
    misses are read directly into pinned `RamTier` and retained across turns.
- **Gated & benchmarked (`bench/stage5-tiers-2026-10-27`):**
  - Passed `cargo test -p shadow --test reference` (`7/7`) and 48-layer GPU comparison (`1.00000` cosine, `10/10` routing).
  - Over **8 launches x 5 warm turns** (`--ctx 8192 --ram-tier-gib 30 --placement profiles/routing-6prompts.shrc`):
    - **`decode_warm` median: 16.92 tok/s** (up from `15.34 tok/s` in `0.1.12`, **`+10.3%`**, and **`+65.9%` (`1.659x`)** over
      the `10.20 tok/s` baseline; p90 `16.62`, worst `16.62`, best `17.02 tok/s`; single-prompt decode `18.5 tok/s` / `0.054 s/token`).
    - **Within-arm noise floor: median 0.8%, p90 2.3%, worst 3.3%**; lowest available RAM `8.7 GiB`; peak GPU `10,628-10,696 MiB`;
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

## Speculative decoding: what it can buy on this PC (2026-10-27, measured before building)

- **The `--mtp` path (0.1.20) is not speculative decoding.** `shadow-serve` drafts a token, then checks it with two
  ordinary one-token `step`s; there is no batched verify pass, so a draft only adds work. The head's KV cache is never
  filled for prompt positions. `SHADOW_MTP_PROBE=1 shadow generate ... --mtp models/mtp-basalt-ft1` (counts drafts
  that match the base model's argmax) panics on the first drafted step: NaN router logits in the base model (`route`,
  forward.rs), so the draft step corrupts state somewhere. Unresolved; `--mtp` stays off by default.
- **The cost side, `SHADOW_ROUTE_LOG` + `tools/spec_probe.py`, 5 prompts (explain, code edit, arithmetic, write
  code, summarise; 53-256 generated tokens each):** checking m tokens in one pass needs the union of their routed
  experts. Per layer, in units of one token's 10 experts: **m2 1.66, m3 2.22, m4 2.74, m5 3.21, m6 3.64, m8 4.45.**
  Expert movement is ~70% of a step here (profile above), so a pass over m positions costs about
  0.3 x (dense, ~1 if batched) + 0.7 x union(m): **~1.5x a step for m=2, ~2.3x for m=4.**
- **Break-even:** a drafter has to land more tokens per pass than that. With per-token acceptance p = 0.8 the best
  case is ~1.25-1.3x (K = 2-3); DFlash-style 5-6x numbers are for compute-bound dense models, not an
  offloaded MoE where every extra position moves more experts.
- **Prompt/history lookup (n-gram) drafting loses:** 1.21-1.24 tokens per pass while verifying 2.0-4.6 positions
  over all 5 prompts; even the code-edit prompt (mostly copying) gets 1.6-1.9 tokens per pass at 2.7-7.3 positions.

## Night of 2026-10-27: CPU experts, pre-gated prefetch, host overhead (all measured; what lost stays opt-in)

Single-prompt runs, "Write a Rust function that parses a comma separated list...", 128 tokens, `--ram-tier-gib 30
--placement profiles/routing-6prompts.shrc`; baseline 0.049-0.051 s/token. Same greedy tokens in every arm below.

- **Where a step goes now (unsynchronized host timers, `host waits per step` in the report):** ~17 ms waiting for
  the GPU to finish a layer (router logits), ~13.5 ms waiting for `copy_stream`, ~5 ms waiting on the drive, and
  ~11-14 ms of host work. `SHADOW_TIMING_NO_COPIES=1` (skips every expert copy, wrong output, timing only) runs at
  **0.028 s/token**: copies cost ~22 ms a token on the critical path.
- **Hybrid CPU experts (`--cpu-experts N --cpu-threads T`, `shadow-model::cpu_expert`): loses.** AVX2+FMA kernels
  for every routed-expert type (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0): weights decoded to the exact int8 x scale
  factors `dequantize` uses (unit-tested), f32 activations, a spinning work-stealing pool. On real experts
  (`cargo run --release -p shadow-model --example cpu_expert_bench`): 1 thread ~1.0 ms/expert (2 GB/s), 8 threads
  0.26 ms, 15 threads 0.20-0.27 ms (~8-11 GB/s) - against ~0.09 ms for a PCIe copy. End to end: 1 expert/layer
  0.051-0.053, 2: 0.058, 4: 0.080 s/token (vs 0.049-0.051), same tokens, cosine 1.000 on the 48-layer compare.
  One CPU expert takes longer than the GPU's whole copy for the layer, and CPU-computed experts never enter the
  VRAM cache. A 3-4x faster kernel (int8 activations, AVX-512) would only reach parity on this i7-11700K/DDR4-2933.
- **Pre-gated prefetch (`--pregate K`): no gain.** Layer l+1's router on layer l's MoE input (`SHADOW_PREGATE_PROBE=1`)
  catches 64-67% of l+1's experts in its top 10, 77-80% in its top 16, ~72% of the misses in its top 16. Copying the
  top K's misses ahead on `copy_stream` raises hits (61.5% -> 69-77%) and cuts the copy wait (13.3 -> 8.6 ms at K=8),
  but the GPU-wait grows by the same amount (16.9 -> 20.2 ms): with the copies skipped and only prefetch DMA running,
  GPU time per step went from 17 to 24 ms. The DMA slows the layer's own work (small per-layer transfers queue behind
  it), so the wait moves instead of disappearing. K=16 also runs a layer's 12-slot LRU dry (now guarded).
- **Kept, default on, exact:** router logits read through mapped pinned memory by a copy kernel (no DtoH memcpy in the
  copy engine); small uploads (embedding, expert tables, n-gram rows) staged in mapped memory and moved by
  `copy_words`; `Gguf::tensor` indexed (it was a linear scan of ~1,500 names per matvec); `route` keeps the top k by
  insertion instead of sorting 512 (unit-tested bit-for-bit against the full sort). Speed-neutral on their own
  (0.050 s/token); they remove copy-engine traffic and host work that a future overlap would otherwise hit.
- **What would move it:** fewer bytes over PCIe per token (a bigger RAM tier only cuts the drive's 5 ms), or
  amortizing a step over several tokens. That is batched forward: prefill (still one token at a time, 0.059
  s/token) and the verify pass speculative decoding needs. Dense tensors are BF16, F32, IQ4_NL, IQ4_XS, Q4_K, Q5_K,
  Q6_K - multi-column versions of those matvecs are the first piece.
- **First piece of batched forward: `matvec_cols_*` (kernels/matvec.cu, all 13 fast types).** y_j = W x_j for up to 8
  inputs per launch. Each `Fast<Q>` has a `dotm<M>` next to its `dot8` (a mechanical copy: decode 8 weights once,
  accumulate into M sums with the same operations in the same order), so the outputs are bit-identical to
  `matvec_fast_*` per input (`tests/matvec.rs::several_inputs_at_once_match_one_at_a_time`, m = 1..8). 8 inputs at once
  vs 8 launches: the expert-sized IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/IQ4_NL/Q2_0 matrices **2.8-4.2x**, BF16 and the F32
  router ~2.2x; the big attention matrices (Q4_K/Q5_K/Q6_K, 6144-10240 x 2560) only 0.9-1.4x, because every warp
  re-reads all 8 inputs from L2 per row (~800 MB of L2 traffic for a 10240-row matrix). Next for those: several rows per
  warp and the inputs staged in shared memory. Timings taken while a CPU test ran; treat them as rough.
  Then 4 rows per warp (`COLS_ROWS`, grid = ceil(rows / 32)), still bit-identical, on a quiet GPU: Q4_K/Q5_K/Q6_K
  attention matrices 1.2x, BF16 1.7x, expert-sized and router 1.9-2.4x for 8 inputs vs 8 launches. One input of a
  10240 x 2560 Q6_K takes 0.06 ms (~360 GB/s); 8 take 0.40 ms, far above the bandwidth or FMA bound, so the kernel is
  limited by something else (register spills from acc[4][8] or occupancy are the first suspects). Profile it before
  building the batched step on it.

## 2026-10-28: session snapshots and the shared-prefix cache

- **`GpuSession::snapshot(with_kv)` / `snapshot_to(with_kv, host)` / `restore`:** every DeltaNet state and conv window,
  the n-gram conv state, and optionally each attention layer's K, V, indexer keys and pooled blocks up to the current
  position; on the device (fast, VRAM) or in host memory. Recurrent state is ~117 MiB (36 x 3 MiB + conv), KV 48 KiB a
  position. Without the KV it only restores onto a session still holding its tokens (a rolled-back speculative pass);
  with it, onto anything. `tests/reference.rs::a_restored_snapshot_continues_exactly`: bit-identical logits after a
  rollback, after a full restore over a different prompt, and through host memory.
- **Shared-prefix cache in `shadow serve`:** when a prompt shares >= 64 leading tokens with the previous request
  (system prompt + tool list), the session is snapshotted there into host memory while it reads; a later conversation
  that starts the same way restores it instead of reading those tokens. Live, 861-token prompts with tools: new chats
  **59.5 s and 56.1 s -> 1.1 s and 1.0 s** (842 tokens restored), identical answers.
- **Web UI:** model-loading state, live prompt-reading progress, readable errors, escape-first Markdown (it used to put
  model text into innerHTML unescaped), NVML VRAM while loading. `tools/serve_check.py` 16/16.

- **`matvec_cols` fixed (2026-10-28): 5.6-6.8x on the big matrices.** The 4-rows-per-warp version still called `dotm`
  per row, reloading every input chunk per row, so the row tiling reused nothing. Now each lane loads its 8-float
  chunk of every input into registers once (float4 loads) and applies it to all 4 rows. Still bit-identical to
  `matvec_fast_*` per input. 8 inputs at once vs 8 launches: Q4_K 6144x2560 6.5x, Q5_K 10240x2560 5.6x, Q6_K 10240x2560
  **6.8x (0.071 ms for 8 inputs vs 0.06 ms for one)**, expert-sized IQ*/Q2_0/IQ4 3.4-6.4x, BF16 2.5x, F32 router 4.9x.
  Dense weights are now close to free to batch: the cost of a multi-token pass is the union of its experts plus the
  per-token small kernels. That is what batched prefill and speculative verification need.
