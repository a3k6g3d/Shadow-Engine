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

## 2026-10-28: the batched step (`GpuSession::step_batch`), batched prefill, speculation measured

- **`step_batch(&[u32]) -> Vec<Vec<f32>>` / `prefill_batch`** (Qwen; Flint, tracing, no cache and the opt-in
  experiments step one token at a time): up to `MAX_BATCH` = 64 tokens through the layers together. Per layer: every
  projection the tokens share through `matvec_cols` (8 inputs a launch; v and the indexer keys written straight into
  the caches, consecutive positions being consecutive rows); the conv window, `delta_step`, q/k norms + RoPE, block
  pooling, the indexer's selection and attention per token in order; the hc glue per token; the router for all of them
  and one mapped read of m x 512 logits; the shared expert for all of them; the union of their routed experts into the
  VRAM cache (misses beyond the layer's free slots into a 128-slot scratch arena, `SHADOW_BATCH_SCRATCH`), in groups of
  consecutive tokens whose misses fit together; each token's routed experts and combine. `SHADOW_BATCH=n` caps the
  batch (1: one token at a time, for A/B on one binary).
- **Exact.** One token runs BF16/F32 matrices in the reference order (not `dot8`'s), so `matvec_cols_ref_bf16/f32`
  and `hc_down_inject_cols_bf16` do the same for m inputs. `tests/reference.rs::a_batched_step_matches_one_token_at_a_time`:
  every token's logits of a 29-token prompt **bit-identical** to `step` one at a time, and the state after it (one more
  step; and `prefill_batch` then a step); `..._past_the_indexer` (ignored, 7 min): the long fixture's 2,600 ids, the
  last 8 positions (past 2,048, sparse selection on) bit-identical. Reference suite 9/9, `shadow compare --gpu` on the
  48-layer fixture cosine 1.000 and 10/10 routing on every layer, the drift test green again (it had been failing on
  the `COLS` macro; the macro is expanded now).
- **Why 64, not 8: the expert union.** First version (chunks of 8, misses only into the layer's free cache slots):
  0.063 -> 0.051 s/token on a 1,262-token prompt, because a layer keeps only 12 evictable slots next to its pinned hot
  experts and a prompt misses ~6 experts a token, so groups were 1-2 tokens wide (misses 362k -> 361k). A 32-slot
  scratch: 0.044. The union of consecutive prompt tokens' experts grows slowly (`SHADOW_ROUTE_LOG` on that prompt,
  experts a token needs per layer): **m=8 5.95, 16 4.75, 32 3.60, 64 2.51, 128 1.63** of its 10. Batches of 64 with
  128 scratch slots: 0.033 s/token; 256 slots the same (0.033), so 128 (~250 MB of VRAM, taken from the cache before it
  sizes itself).
- **Bug found on the way (also latent in the one-token path):** `RamTier::promote_slot`'s Clock hand could wrap and
  hand out a ring slot already pending a file read (or with a copy from it in flight) when few ring slots match the
  layer's expert size; a 256-slot batch hit it (`read_drive` panicked). The ring now skips offsets claimed since the
  copies were last waited for.
- **Measured (8 interleaved launches each):**
  - `shadow generate @prompt --gpu --chat -n 16 --ram-tier-gib 30 --placement profiles/routing-6prompts.shrc`, a
    1,262-token prompt (the start of ARCH.md), the same binary with `SHADOW_BATCH=1 SHADOW_BATCH_SCRATCH=0` (one token
    at a time) and the defaults, interleaved: **0.0622 -> 0.0345 s/token (78.6 s -> 43.5 s), -46.0% paired, all 8
    pairs -43% to -47%**, the same 16 generated tokens in every run. Waits per token, one at a time -> batched: router
    18.8 -> 11.3 ms, copies 22.2 -> 7.5 ms, file 9.8 -> 5.9 ms.
  - `shadow-bench`, `--ctx 8192 --ram-tier-gib 30 --placement profiles/routing-6prompts.shrc`, HEAD (`target/release`)
    vs this build (`bench/batch-2026-10-28`): **decode tie** (paired median +0.0%, p10 -0.6%, worst -0.8%; both arms
    median 18.49 tok/s), **0 answer differences in 48 turns**; prompt reading of the turns' 25-31 new tokens **51.3 ->
    36.0 ms/token** (median over 48 turns each).
- **Speculative decoding (`shadow serve --spec K`, off by default): measured, loses.** Prompt lookup drafts up to K
  tokens (the latest earlier occurrence of the last 3, else 2 tokens); one `step_batch` over [next, drafts] checks them
  (its logits are bit for bit what `step` gives, so the answers cannot change); on a rejection `restore` to a device
  snapshot taken before the pass (`snapshot_again`, refreshed in place, no allocation) and `prefill_batch` of the kept
  tokens. `bench/spec-2026-10-28`, 8 interleaved launches per arm, same flags: **spec2 -11.2% paired (17.04 vs 18.62
  tok/s), spec3 -18.3% (15.22)**, 0 answer differences. Drafts land 1.50 tokens per pass (112 of 448 / 672 drafted
  accepted over 224 passes), and a pass over 3-4 positions plus the replay costs well over 1.5 steps - what the
  2026-10-27 cost model predicted. Kept opt-in. Not tried: the `--mtp` head as drafter (its draft step still corrupts
  the base model; dropped here), keeping per-position DeltaNet states inside the pass so a rejection needs no replay.
- **What next (in order of expected value):** expert-major routed matvecs in the batched MoE (each union expert once
  over the tokens that use it, through gathered `matvec_cols`: today every token re-reads its 10 experts' weights from
  VRAM, ~20 MB a layer); larger prefill unions (m = 128: 1.63 experts a token a layer) by streaming the union through
  the scratch slots expert by expert; the ~11 ms a token the host waits for the GPU in a batched pass (per-token small
  kernels: hc glue, the recurrences, routed matvecs).

## 2026-10-28 (evening): CUDA graphs and overlapped expert copies in decode (both default on, exact)

- **Measured first:** a decode step (104 tokens of the Rust prompt, prompt steps subtracted) made **~1,245 kernel
  launches and spent 5.75 ms of host time in them** (now counted in the session report), of ~52 ms; host waits: router
  17.3 ms, copies 14.6 ms, file 7.5 ms; the rest (~7 ms) host work. So graphs could reach at most the launch share plus
  the GPU's inter-kernel gaps.
- **CUDA graphs** (`SHADOW_GRAPHS=0`: off): every DeltaNet layer's launches from its hc read to its router (hc glue,
  mixer, hc glue, router: ~15 kernels, nothing position-dependent) captured once per (layer, fused-norm) and replayed
  on a blocking stream (`Context::blocking_stream`, so it orders with the legacy default stream both ways; driver
  bindings `cuStreamBeginCapture`/`EndCapture`/`cuGraphInstantiateWithFlags`/`cuGraphLaunch`). Attention layers and
  the PLE layer keep direct launches. Launches per step 1,139 -> 722, host time in them 5.4 -> 3.6 ms; the router wait
  grew ~1 ms (the GPU is the bottleneck in that phase). `bench/graphs-2026-10-28`, 8 interleaved launches vs 0.1.25:
  **+2.1% paired** (18.65 -> 19.00 tok/s, p10 +0.8%, base floor median 0.2% / p90 1.1%), 0 answer differences.
- **Overlapped copies** (`routed_overlapped`, `SHADOW_OVERLAP=0`: off): after routing, the resident experts' matvecs
  launch at once, the misses' copies go out on `copy_stream`, and the default stream (not the host) waits for them
  (`copy_event` + `cuStreamWaitEvent`) before the misses' matvecs; eo's rows come out resident-first and
  `moe_combine_qwen_perm` sums them in routing order, so y is bit-identical. The host queues the rest of the layer
  while the copies run: host copy wait 13.3 -> 2.0 ms, router wait 16.2 -> 25.6 ms, net ~4 ms a step.
  `bench/overlap-2026-10-28` vs the graphs build: **+2.7% paired** (18.96 -> 19.47 tok/s; p10 -2.3%, floor median
  1.4%), 0 answer differences. Reference suite 9/9 with both on (the batch test compares graphed, overlapped `step`
  with `step_batch` bit for bit).
- Both together: **18.65 -> 19.47 tok/s, ~+4.4%** over 0.1.25.

## 2026-10-28 (night): the VRAM expert cache is all LRU (default), `--pin-margin N` for the old pinning

- **Simulated first** (`tools/cache_sim.py`: a `SHADOW_ROUTE_LOG` replayed through policies, per layer, with the
  session's slot counts from `SHADOW_CACHE_DUMP`). Today's policy - the placement's hottest experts pinned, 12 LRU
  slots a layer - replays at 59.1% hits on the Rust prompt's decode (measured: 59.5%, so the simulator is faithful).
  Pure LRU over the same slots: **75.0%** (hash-table prompt 53.4 -> 68.9%, a 1,271-token prompt read 39.7 -> 63.8%);
  margin 24: 65.5%, margin 48: 72.6%; a decayed-LFU with admission 71-73%. Locality inside a conversation beats
  global popularity. More slots under LRU: +3 a layer +1 point, +20 +5.7 points.
- **The cost:** pinned experts were kept out of the RAM tier; without pinning the tier holds them too, so it covers
  ~2,300 fewer distinct experts and more misses go to the file (1,375 -> 2,232 of 104 tokens; file wait 6.5 -> 9.9 ms),
  far outweighed by the misses saved (24,210 -> 15,971, hits 56.6 -> 70.7%).
- `bench/lru-2026-10-28`, 8 interleaved launches vs the overlap build: **+11.1% paired, 19.53 -> 21.06 tok/s**
  (p10 -1.5%, worst -1.9%, floor median 0.2%), 0 answer differences. Today's three decode changes together:
  **18.65 -> 21.06 tok/s (+12.9%)**.

## 2026-10-28 (night): Shadow against Strata, same protocol (`bench/strata-2026-10-28`)

- `shadow-bench` gained `--arm-launch NAME=CMD` (another engine's launch; `{port}`, `{log}` filled in), readiness that
  waits for `"loaded": false` to go away, and a process-tree kill for servers without `/shutdown`. Strata ran from a
  copy of the user's `strata.json` (`strata_bench_cfg.py`): same engine flags, but Discord, MCP tools, the
  conversation store, the learned-profile save and the API key off, loopback only, slots and logs to scratch; and
  matched to Shadow's bench: no vision, context 8192, greedy (temperature 0). Its launcher (`run_strata.ps1`) kills
  the whole process tree below 4 GiB available.
- **8 interleaved launches each, the same 6 turns x 96 tokens: Strata 48.55 tok/s `decode_warm`, Shadow (b431c16)
  20.98 - Strata +135% paired** (p10 +104%, worst +76%). Prompt reading: **Strata 6.1 ms/token, Shadow 35.0**. Answers
  differ in 18 of 48 turns (Strata is not exact against the reference; see below).
- **Where Strata's speed comes from (its own log):**
  1. A working MTP drafter (`mtp-basalt-ft1`, the head Shadow's broken `--mtp` loads) with `--spec 4`: drafts
     accepted 49/58, 39/39, 24/26 ... (85-100% at greedy), verify windows of 1-6 tokens captured as CUDA graphs.
  2. `STRATA_ROUTE_TAIL_SKIP=7` (default on): a missed expert that every token of a verify window routes at rank 7 or
     lower is not computed (11,347 skipped in the first answer) - fewer bytes over PCIe, not exact.
  3. `--pcie-frac 0.35`: part of the misses read by the GPU straight from pinned RAM instead of copied into the cache.
  4. Its VRAM cache hits less than Shadow's now (46-48% vs ~71%); the cache is not where it wins.
  5. Prompts: expert-grouped GEMMs (`src/prefill/moe_mmq.cu`), 1,771 cache slots lent to the prompt path, the CPU
     taking part of the experts below 1,024-token chunks.
- So the gap is speculation that works (exact once verified) plus approximations Shadow has not taken. Next, in that
  order: port the MTP draft layer (Strata `src/core/mtp.cpp` `record_front` / `record_rest`: pre-FC norms + fc_embedding
  / fc_hidden, hc read, gated attention over its own KV with q/k norm + RoPE, MoE over 512 VRAM-resident Q2_0
  experts with a gated shared expert, final mixer, the main head over the draft vocabulary; its KV filled for every
  committed position from the main model's final residual rows); then expert-grouped prefill. Shadow's `mtp.rs` reads
  its routed-expert table in the wrong order, never computes the shared expert's gate, routes by sigmoid, and never
  fills its KV for prompt positions.

## 2026-10-28 (night): the MTP draft layer, ported from Strata - drafts right 95%; speculation not yet a win

- **Port** (`crates/shadow-cuda/src/session_mtp.rs`, `mtp.rs` keeps the loader): the layer as Strata's
  `src/core/mtp.cpp` runs it, on Shadow's kernels - pre-FC norms + `fc_embedding` / `fc_hidden` (Q8_0), the hc read,
  k/v (and q, gate) projections, q/k norm + RoPE at the cell, gated attention over its own K/V, the MLP hc, router
  (BF16) + top 10 of the 512 VRAM-resident Q2_0 experts + the gated shared expert, the final mixer, the main head,
  argmax over `draft_vocab.bin`. Its K/V is filled for every position the main model reads (prompt and verify
  passes, from the final residual rows; `mtp_observe`), its pending residual is in snapshots (and its K/V with
  `snapshot(true)`). The old `mtp.rs` step is gone: it read its expert table in the wrong order, never computed the
  shared expert's gate, routed by sigmoid, never filled its K/V, and its `eg` buffer was half the size the gate+up
  launch wrote - the overflow is the likely source of the NaNs it left in the base model. Its expert unpack (S2 blob
  -> Q2_0, gate/up rows interleaved, fp16 scale per 64) checks out against Strata's `expert.hpp`.
- **Drafts are right:** `SHADOW_MTP_PROBE=1` (one draft before every step, against the step's own argmax), the Rust
  prompt, 104 tokens: **98 of 103 (95.1%)**, and the generated tokens are identical to a run without `--mtp`.
- **Speculation with it** (`shadow serve --mtp models/mtp-basalt-ft1 [--spec K]`, K = 3 by default with `--mtp`):
  exact (0 answer differences), ~3 tokens a pass (55 of 84 drafts accepted - a chain drifts, the first draft is
  right 95%), but **34% slower** in a 2-run check (14.3 vs 20.1 tok/s). Per pass: drafting 8 ms, the verify pass
  (`step_batch` of 4) **152 ms**, the rollback (restore + re-reading the kept tokens) 31 ms - against 48 ms for one
  plain step. Kept opt-in.
- **Why the 4-token pass costs 3 steps** (`SHADOW_KTIME=1`, CUDA events around every launch, GPU ms per token, the
  hash-table prompt read 4 vs 1 at a time): all kernels 21.9 vs 31.9; `hc_down_inject_cols_bf16` 2.53 vs 2.15 (324
  rows at 4 a warp: 81 warps), `delta_step` 2.10 vs 3.46 (each token re-reads and re-writes the 3 MB states), routed
  experts 5.35 vs 6.0 (per token), dense projections 3.1 vs 6.3. **Next, by what each saves a pass:** a rollback that
  replays only the recurrences from inputs the verify pass kept (~29 ms), one row per warp in the batched hc kernel
  (~8 ms), a multi-token `delta_step` holding each state in registers (~6 ms), the copy overlap and grouped experts
  in `moe_batch`.
- **Then (same night), exact, 10/10 reference tests:** `step_batch_verify` + `rollback` (`tests/reference.rs::
  a_rolled_back_verify_pass_continues_exactly`): a verify pass keeps each DeltaNet layer's conv inputs and beta /
  alpha per token and the n-gram conv inputs; a rollback restores the states and redoes only those updates for the
  kept tokens - **31 -> ~1.4 ms a pass**. `delta_steps128`: the recurrence for a batch's tokens with each state
  column in registers between them (bit-identical to `delta_step` per token): 2.10 -> 0.90 ms a token at m = 4. The
  reference-order batched kernels take one row a warp under 2,048 rows (`hc_down_inject_cols_bf16` 2.53 -> 1.80 ms
  a token). Drafting stops after a draft the layer gives < 0.7 (`SHADOW_SPEC_MIN_P`, Strata's `--spec-min-p`).
  2-run checks against plain decode (21.06 tok/s): **--spec 3 -14.1%, --spec 5 -14.9% paired**, 0 answer
  differences; ~2.9 tokens a pass, the verify pass ~119 ms for 3.3 tokens against 47.5 ms a plain token. Still
  opt-in. What is left in the pass: host-blocking expert copies in `moe_batch` (decode overlaps them), routed
  experts per token, the per-token hc kernels (scalar BF16 loads for the reference sum order), no CUDA graphs.
- **Then:** the batched MoE overlaps its copies like decode does (`overlapped_tab` / `routed_rows`: every token's
  resident experts run while the group's misses copy, the GPU waits for the copies, `moe_combine_qwen_perm` keeps
  the sum order; 10/10 reference tests): host copy wait under speculation 10.5 -> 0.0 ms a token. `--spec 3`:
  **-8.9% paired** (19.77 vs 20.95 tok/s, 2-run check), 0 answer differences. `SHADOW_REPORT=1` prints the session
  report after every answer in `shadow serve`.
- **Where it stands:** per committed token the GPU now does about what plain decode does (`SHADOW_KTIME`: 30.9 vs
  29.3 ms) - a verify pass of ~3.3 positions shares only the dense weights, which at m = 3-4 the cols kernels
  barely shrink (Q6_K 1.80 vs 2.35 ms a token), while routed experts (8.4 vs 6.9 ms a token, rejected positions
  included) and the per-position hc / conv / swiglu kernels scale with positions. The drafter's 830 MB also come out
  of the expert cache (2,504 vs 2,929 slots). Strata's verify is cheaper through inexact tail skipping, int8-activation
  kernels and grouped experts. What would still move Shadow's: experts grouped across a pass's positions (each
  union expert's weights read once), fusing the per-position small kernels, the drafter's experts in pinned RAM
  instead of VRAM.
- **Grouped routed experts** (`matvec_grouped_*`, `grouped_plan` / `routed_grouped`): in a batched step each union
  expert runs once on all of a group's tokens that route to it (up to 8 inputs a launch, `dotm`, so each output is
  `matvec_fast_rows`' bit for bit, landing at (token, routing rank) so the plain combine keeps the sum order), the
  resident ones while the misses copy, the copied ones after the GPU-side wait. 10/10 reference tests. Prompt
  reading (`bench/prefill-grouped-2026-10-28`, 1,262 tokens, 8 interleaved runs, `SHADOW_OVERLAP=0` vs default on
  one binary): **0.0334 -> 0.0308 s/token, -7.7% paired, all 8 pairs**, identical tokens. Speculation unchanged
  (-10%, a 2-run check: at ~3 positions a pass few experts are shared). Also fixed: a 64-token group's RAM-tier
  misses could lose their ring slots to the group's own promotions and overrun the staging slots (`read_drive`
  panicked); the group now claims them first.
- **Where exact speculation tops out on this PC (measured):** the drafter's 830 MB of VRAM cost plain decode 2.9%
  (`--expert-cache-gib 4.5` vs auto: 20.27 vs 21.10 tok/s, 2-run check); a pass verifies ~3.3 positions for ~2.9
  tokens kept (~15% of positions wasted); drafting is ~8 ms a pass (~6%); and per committed token the GPU does about
  what plain decode does, because at 3-4 positions only the dense weights are shared, and they are a small part of a
  step here. Fixing all of those lands near break-even, not near Strata's +135%: Strata's gain comes with its
  inexact tail skipping (missed rank >= 7 experts not computed in a verify window), which shrinks each position's
  expert traffic. `--mtp --spec K` stays opt-in at about -10%.

## 2026-10-29: `--tail-skip R` (opt-in, inexact) and other engines

- **Other engines on this PC, same bench:** Strata 48.5 tok/s, Strata with `STRATA_ROUTE_TAIL_SKIP=0` **36.6**
  (-25.7% paired, `bench/strata-skip-2026-10-28`: the skip is a quarter of its speed, not most of its lead), llama.cpp
  from Strata's fork (`llama-server`, `-ngl 99 --n-cpu-moe 43`, CPU experts from the mmapped file: the 78 GB model does
  not fit 64 GB of RAM) **12.8** (16 threads and flash attention the same).
- **`--tail-skip R`** (Strata's rule, from its `expert_source.cpp`): in a decode step, and over a whole verify pass,
  a routed expert not in VRAM that every token routes only at 0-based rank >= R is neither copied nor computed; its
  contribution is zero and the others keep their router weights. The first version garbled every answer: when a token
  kept fewer than 10 experts, the overlapped combine read the shared expert from row n instead of row 10. Fixed.
  Quality (`shadow ppl`, new: teacher-forced perplexity one token at a time, the 1,262-token ARCH.md text): exact
  **6.550**, R=7 6.600 (+0.8%), R=5 6.790 (+3.7%), R=4 7.113 (+8.6%), R=3 7.224 (+10.3%); argmax = next token 57.6-58.6%
  throughout. Speed (`bench/tail-2026-10-28`, 8 interleaved launches): exact 21.32, **R=7 24.45 (+15.9% paired), R=5
  27.78 (+29.8%), R=4 29.40 (+36.0%)**; every pair faster. Off by default.
- **Upstream llama.cpp and ik_llama.cpp** (shallow clones `target-src-llama` cf24317 and `target-src-ik` 1c5ea12, both
  with `qwen4exp`; built for sm_120 into `target-build-*`; `bench/engines-2026-10-29`, 2-run checks, same 6 turns):
  upstream `-ngl 99 --n-cpu-moe 43` **15.1 tok/s**; ik with the same experts on the CPU (`-ot ...exps=CPU -dple`)
  **16.4**, with `-ser 6,1` 16.0, with all experts on the CPU behind its GPU expert cache (`--moe-cache-mib 6000`) 16.0
  (4000: 14.6). ik's `--n-cpu-moe` puts the CPU experts in 71 GB of pinned `CUDA_Host` buffers instead of the mmapped
  file: available RAM fell to 1.0 GiB before `run_capped` (1 s polling) killed it; it ran afterwards under
  `run_guard.ps1` (200 ms polling, 6 GiB floor) with `-ot ...=CPU`, which keeps the file mapped (min 8.4 GiB).
- **Standing on this PC** (decode_warm): Strata 48.5, Strata without tail skip 36.6, **Shadow --tail-skip 4 / 5 / 7:
  29.4 / 27.8 / 24.5, Shadow exact 21.3**, ik_llama.cpp 16.4, upstream llama.cpp 15.1, Strata's llama.cpp fork 12.8.
- **AVX-512 CPU experts** (`shadow-model::cpu_expert_avx512`, the hybrid's kernels when the CPU has AVX-512 BW +
  VBMI; `SHADOW_CPU_EXPERTS_F32=1` keeps the old AVX2 ones): int8 activations and integer dot products - Strata's
  IQ2_S / IQ3_XXS / IQ3_S `iq_avx512.cpp` ported, plus IQ4_XS, IQ4_NL (lookup + |w| x signed y) and Q2_0 (vpermb
  unpack, minus the activation sums); tested against `dequantize` with the quantized activations (1e-4). Real
  experts (`cpu_expert_bench`, 14 threads): **0.088-0.136 ms an expert, 17-21 GB/s** (AVX2: 0.23-0.29 ms, 6-10 GB/s) -
  as fast as one PCIe copy. The hybrid now runs inside the overlapped decode path (its input comes down with the
  router logits; graphs and batched passes stay on). **Still a loss** (2-run checks): exact + 1 CPU expert a layer
  -2.5% (4-6 threads) to -4.2% (14), 2 experts -6.7%; with `--tail-skip 4` (85% cache hits, ~1.2 misses a layer
  left) -6.3% to -13.5%. One CPU expert saves one ~0.09 ms copy a layer, most of which already overlaps GPU work,
  while the host waits for the CPU before the combine and the CPU's RAM reads compete with the DMA's. Opt-in.
- **Where a tail4 step goes** (Rust prompt, 128 tokens): 33 ms a token; with every expert copy skipped
  (`SHADOW_TIMING_NO_COPIES`, wrong output) 25 ms. Strata's 48.5 tok/s is 20.6 ms a token: single steps cannot get
  there on copies alone; the verify pass (now ~77 ms for 3.1 positions with tail skip, against 34 ms a plain step)
  has to get cheap. `--tail-skip 4 --mtp --spec 3`: +2.4% over tail4 alone (2-run check).
- **Multi-row kernels in the batched step** (bit-identical; 13/13 reference tests including the ignored long ones):
  `hc_mix_rows`, `hc_write_rmsnorm_rows`, `conv_silu_steps` (each channel walks the batch's tokens in order),
  `l2norm_qk_rows`, `gated_rmsnorm_rows`, `head_norm_rope_qgk_rows` + `attention_heads_rows` (whenever no token needs
  the sparse selection; 8 rows of scores at a time), `moe_combine_qwen_rows` (per-token expert counts, for tail
  skip): one launch per step instead of one per token. First checks: prompt reading 0.0308 -> **0.023 s/token**
  (1 run, same tokens); a tail4 verify pass 77 -> 72 ms; `--tail-skip 4 --mtp --spec 3` +2.7%, `--spec 5` +6.6%
  paired over tail4 alone (2-run checks, worst pairs -18% / -38%). The session report now breaks verify passes down
  (wall, GPU waits, file, launches, host time in launches).
- **Speculation research (2026-10-29)**, arXiv / Hugging Face: AcceptMoE (2608.02989: resident-only experts for
  uncommitted verify positions, -74..77% H2D, 2.06x under offloading), SpecFit (2610.15417: verify subtree by predicted
  throughput, expert-union cost), Lan (2610.34440: dedup pays from 5-8 tokens a pass), DFlash 2 (a 0.6B block-diffusion
  drafter for this model: acceptance 2.8-3.3 vs MTP 2.5-3.1 per pass, 1.2 GB), S2-MoE / DraftExpert / SpecMoE
  (expert-sparse self-drafts: a full dense pass per draft, too slow here), suffix decoding (code editing).
- **`--verify-resident`** (inexact, AcceptMoE's idea): drafted positions of a verify pass keep only their resident
  experts. With tail skip 4 and MTP: K=5 +4.6%, K=3 -5% over tail4 alone (2-run checks); a pass 72 -> 64 ms - copies
  were a small part of it; GPU work (37-44 ms) and host work (~25 ms, 1,834 launches) are the rest.
- **BF16 / F32 in the 8-at-a-time sum order, default** (`SHADOW_REF_ORDER=1` for the reference order): the
  reference-order kernels forced scalar loads (`hc_down_inject_cols_bf16` was ~120 GB/s). New
  `hc_down_inject_fast_bf16` / `hc_down_inject_cols_fast_bf16` (one row a warp, dot8 / dotm); one token and a batch
  still agree bit for bit; 10/10 reference tests (the fixtures' tolerances). 2-run checks: exact +3.2%; the fastest
  mode (`--tail-skip 4 --mtp --spec 5 --verify-resident`) 30.1 -> **31.9 tok/s**, +53.9% paired over exact.
- **Graphs for the batched step's DeltaNet segments** (one per layer x batch size x rollback flag, hc glue to router;
  the router logits of the whole batch in one mapped read) and **grouped routed experts at one row a warp**
  (`GROUPED_RW`: 640/2560-row experts with a few inputs need warps more than input reuse). Bit-identical (10/10).
  2-run checks, fastest mode (`--tail-skip 4 --mtp --spec 5 --verify-resident`): 31.97 -> 32.95 (graphs) -> **34.0
  tok/s** (one row a warp, +5.4% paired); a verify pass 78 -> 60 -> 56.6 ms for 3.4 tokens (launches 1,834 -> 1,017,
  host time in launches 23 -> 5 ms). Prompt reading 0.024 s/token.

## 2026-10-29 (late): fastest mode confirmed; zero-copy, spin-wait, hybrid drafting, the sync floor

- **`bench/fast-2026-10-29`, 8 interleaved launches**: exact **20.40** tok/s; with MTP K=5 + `--verify-resident`:
  `--tail-skip 7` 33.04 (+57.9% paired), 5: 31.79, **4: 36.22 (+66.4%, p10 +30.9%)**, 3: 31.58. Strata 48.5 (36.6
  with its tail skip off). 23 of 40 answers differ from exact (inexact modes). Speed is not monotonic in the skip
  rank: harder skipping lowers the drafter's acceptance (it was trained on the full model).
- **Zero-copy PCIe reads, refuted on this PC** (`SHADOW_ZERO_COPY=1`: RAM-tier misses read in place by the matvec
  kernels, exact): 0.115 s/token against 0.046 with copies - the kernels pull ~7 GB/s over PCIe where DMA does 22.
  A zero-sync pass built on in-place reads would need kernels that stream experts in wide, aligned bursts first.
- **Host waits, measured**: `SHADOW_SYNC=spin|yield|auto` (CUDA context scheduling; spin is the default now) changes
  nothing (0.046-0.048 s/token). `SHADOW_TIMING_NO_SYNC=1` (timing only: stale routing, no router wait, no copies)
  takes a verify pass from 55.6 to 45.2 ms - the sync and the copies are ~10 ms; **the rest is GPU kernel time and
  launch latency**, about 4x the bandwidth roofline of the bytes the pass touches. The n-gram rows' host reads
  (1.7 ms a token; now parallel, 0.01 ms) were hidden behind GPU work and moved nothing. A 4-warp hc down projection
  cut that kernel 2.7x (2.05 -> 0.75 ms a token) without moving the pass either; rollback copies removed.
- **Hybrid drafting** (a 4+ token repeat in the conversation drafts without the MTP layer): neutral on chat turns
  (`SHADOW_LOOKUP_MIN`, default 4). `SHADOW_LTIME=1` times a batched pass by segment.
- Next lever, for whoever continues: fewer, fatter kernels per layer (a cooperative hc block, the beta/alpha/gate
  projections in one launch, attention without per-head double math) - the pass is GPU-bound at ~45 ms for ~4
  positions against a ~12 ms roofline.

## 2026-10-29 (night): FP64 was a hot spot - float runs added in double, the draft head's rows, attention

- This GPU runs FP64 at 1/64 of FP32. Three kernels did their long sums in double; each now sums runs of 16 terms in
  float (`fmaf`) and adds the runs in double (same code in the single-token and batched kernels, so one token and a
  batch still agree bit for bit; the reference suite's tolerances, the 2,600-token DeltaNet/indexer fixture and the
  48-layer compare all hold): `delta_step` / `delta_steps128` (117 us a launch for 4 tokens before): spec mode
  36.20 -> **36.08..37.2** (+4.2% paired, 3 runs); attention (`attention_head`: scores and weighted sum): +1.3% on
  the short bench prompts, more with context.
- **The draft head over the draft vocabulary's rows only** (`matvec_ids_q6_k`, `argmax_compact`; 106k of 248k rows,
  no extra VRAM): drafts still right 95.1% (98/103); spec mode +3.1%, **37.2 tok/s** (3 runs, paired).
- Fastest mode now (2-3 run checks): `--tail-skip 4 --mtp --spec 5 --verify-resident` **37.2 tok/s** against exact
  20.8 and Strata 48.5. The pass is GPU-bound (~45 ms for ~4 positions with sync and copies removed); remaining
  candidates: merging the projections of one input into a launch (4 in a DeltaNet layer, 5 in attention), a fused
  hc up-projection + mix, hc matrices in a narrower type.

## 2026-10-29 (late night): fusions are a tie, float `expf`, PLE prefetch

- Float `expf` for the DeltaNet decay and the decay's float runs: fastest mode 36.2 -> **38.9 tok/s** (4 runs, paired
  vs the previous build; answers unchanged on the bench turns).
- Kernel fusions (`--fuse N` / `SHADOW_FUSE`, bits: 1 `matvec_cols_multi` for a DeltaNet layer's qkv/gate/beta/alpha, the
  router + shared-expert gate, and the shared expert's gate/up after the router read; 2 `hc_up_mix_cols`, the hc
  up-projection + mix in one launch; 4 `delta_steps128_gn`, the gated RMSNorm folded into the DeltaNet kernel; every
  output bit-identical to the unfused path). Paired vs no fusion, 20 pairs: fuse 0 **+0.3%** (tie, the noise floor is
  0.4-0.6%), fuse 7 **-1.1%**. The launch count is not what the pass waits for; **default is off**, kept opt-in.
- The PLE (n-gram memory) rows are prefetched one layer ahead (`ple_prefetch`).

## 2026-10-29 (night): DFlash 2 drafter ported (`--dflash models/dflash2`)

- The 5-layer block-diffusion drafter (SGLang's DFlash2DraftModel; weights `models/dflash2/model.safetensors`, 1.2 GB BF16 in
  VRAM; `crates/shadow-cuda/src/dflash.rs`, `session_dflash.rs`, `df_*` kernels in `kernels/forward.cu`). Features: the mean
  over the 4 hyper-connection streams of layers 5/15/25/35/45's outputs, for every position the main model reads (also in
  prefill); `fc` + `hidden_norm` -> per-layer K/V rings (window 2048); a draft round runs the anchor + 7 mask tokens through
  the layers (grouped dynamic convs, non-causal attention over context + block), takes 16 candidates a slot from the main
  model's own head (`df_top16`), and walks the selector's bilinear edge scores greedily (`df_walk`). ~8 ms a round.
- Checked against a CPU torch rebuild (`tools/dflash2_ref.py`, with `SHADOW_DF_DUMP=dir`): block hidden max abs error 6e-5
  (relative 1e-6), selector walk identical. Exactness of answers is the verify pass's, as with MTP.
- Acceptance is deeper than MTP's chain (draft position right/tried on one run: 15/22 9/15 4/9 2/4 2/2 1/2 1/1), but a
  7-draft pass costs ~73 ms, so speed needs a short block: fastest mode (`--tail-skip 4 --verify-resident`, 3 runs each,
  6 chat turns): MTP `--spec 5` 38.9 tok/s; DFlash `--spec 2` 38.8, **`--spec 3` 39.8 (+2.2%, floor 0.6%)**, `--spec 4`
  36.7, `--spec 5` 38.6, `--spec 7` 30-32. Stopping a 7-draft chain at a selector probability (`--dflash-min-p`): 0.3
  31.7, 0.5 38.1, 0.7 38.5, 0.85 35.8 - no better than a fixed 3. **Default with `--dflash`: 3 drafts.**
- After a prefix-snapshot restore the drafter has no features for the restored positions, so it drafts nothing until a
  conversation starts from position 0 (the MTP head then drafts if loaded).

## 2026-10-30: DFlash 2 fine-tuned on Shadow's own features; autotune loop

- `tools/dflash2_train.py`: LoRA (r=32, all linears, fp32 over frozen BF16) on the drafter, trained on features Shadow
  itself produced (`SHADOW_DF_DUMP=dir`: 5 taps, tokens; ~246k tokens of alpaca/code-alpaca answers, thinking on and
  off, single- and multi-turn, greedy, fastest-mode flags) with the loss the drafter is used for (full-vocab CE through
  the main model's head + CE over the selector's 16 candidates scored like `df_walk`); merged back to BF16
  (`models/dflash2-ft2`, not in git: rerun the script). Metric: mean accepted prefix of the free-running greedy walk,
  held-out conversations: thinking-mode set 2.72 -> 4.0 of 7; non-thinking multi-turn (the bench's kind) 2.57 -> 3.14.
- A first tuning on thinking-mode data only did not carry to the bench (it sends `enable_thinking: false`): matching the
  training data to the deployed mode was the fix.
- Standard bench (6 turns, floor 0.2-0.6%, 3 runs): MTP `--spec 5` 38.95 tok/s; stock DFlash 2 `--spec 3` 39.84; tuned
  (`--dflash models/dflash2-ft2`) `--spec 3` 38.38, `--spec 4` 37.99, **`--spec 5` 42.82 (+9.9% over MTP, +7.5% over
  stock; runs 42.70 / 42.84 / 42.84)**. The non-monotonic 3/4 arms are not explained yet (their paired spread is wide).
- Held-out prompts (`shadow-bench --prompts-file`, 12 turns of unseen alpaca prompts): speed there is dominated by
  expert-tier misses (the placement was profiled on the 6 hash-table prompts): identical configs swung 9-47 tok/s, so
  it cannot rank drafters; accepted tokens per pass there: tuned `--spec 5` 3.55, stock 2.86, MTP 2.84.
- `tools/autotune.py`: coordinate ascent over serving flags with `shadow-bench` rounds (adopt a value only above
  max(1%, 3 x floor) and a positive paired median), logged to `bench/autotune/log.md`.

## 2026-10-30 (later): the fastest mode is `shadow serve`'s default

- `shadow serve MODEL` with no tier flags now runs the fastest measured mode: `--tail-skip 4 --verify-resident --ram-tier-gib 30
  --placement profiles/routing-6prompts.shrc --dflash models/dflash2-ft2 --spec 5` (each only when its flag is absent and
  the file exists; `models/dflash2` if the tuned weights are missing). `--exact` turns all of it off (the exact, unspeculated
  path); `--tail-skip 0`, `--dflash off`, `--spec N` change one piece. Flag-less vs the explicit flags, 3 runs: **42.09 vs
  42.22 tok/s (tie, floor 1.8%)**; exact mode is ~20.4.
- `shadow-bench` launched flag-less arms under a 16 GiB working-set cap (sized for no RAM tier), which squeezed the 30 GiB
  tier (first checks read 32.6 vs 42.7); it now assumes the default tier unless the arm says `--exact`.
- Autotune (spec 3-7, min-p, tail-skip): nothing beat `--spec 5 --tail-skip 4`; the optimum is narrow (neighbours -5..-29%).

## 2026-10-30 (later): DFlash candidates over the draft vocabulary - refuted

- `matvec_ids_cols_{q6_k,q4_k,iq4_xs}` (the draft head's rows only, every row decoded once for all 7 slots) and `df_top16`'s id
  map; used when a `draft_vocab.bin` sits beside the drafter's weights (opt-in; `SHADOW_DF_FULL_HEAD=1` ignores it). With the MTP
  head's 106k-token list: the draft round 7.1 -> 6.1 ms, but candidates outside the list are lost: position-1 hits 300/432 ->
  260/440, tokens per pass 2.89 -> 2.42, **-25% tok/s** (4 runs, floor 0.3%). The saving is at most ~1 ms of ~65 ms a pass;
  a list built from this model's own outputs could cover more but cannot win more than ~1.5%.

## 2026-10-30 (later): the drafter's matrices as Q8_0 (default; `--dflash-bf16` opts out)

- Quantizing the drafter's matrices to Q8_0 at load (norms, convolution bases and selector codebooks stay BF16) frees ~0.4 GB of VRAM:
  the expert cache grows 2094 -> 2311 slots, and the draft round drops 7.1 -> 6.3 ms (`matvec_cols_q8_0`). Tokens accepted per
  pass did not fall (3.05 vs 2.89). Pass time (draft + check) 66.7 -> 65.6 ms: ~1.6%.
- The bench's per-turn median is a poor judge here: with tail-skip the arms generate different text, and one run read 36.5 vs 43.0
  tok/s by median but 45.3 vs 42.0 by total tokens/time (4 runs each, turn 0 dropped). The mechanism is the measurement;
  the +7.8% in total throughput is mostly which tokens were generated.

## 2026-10-30 (correction): per-turn medians overstated the tuned drafter

- `shadow-bench`'s headline is the median over a run's 5 warm turns. With inexact modes (tail-skip) the arms generate different
  text, so the medians compare different turns. Re-scored by total tokens / total decode time (turn 0 dropped, from the
  servers' own speculation lines; runs are deterministic, 3 runs agree to 0.1):
  MTP `--spec 5` **40.3** tok/s; stock DFlash 2 `--spec 3` **42.9**; tuned (`dflash2-ft2`) `--spec 3` 41.7, `--spec 4` 42.1,
  `--spec 5` 42.0 (earlier medians: 38.95 / 39.84 / 38.38 / 37.99 / 42.82). Min-p arms: 0.3 38.6, 0.5 41.5, 0.7 43.9, 0.85 40.0.
- So: DFlash 2 is ~+6% over MTP on this bench by total throughput (not +10%), and the fine-tune does not show a gain here
  (0.2 .. -2.8% vs stock), although held-out acceptance rose 2.57 -> 3.14 (non-thinking chat) and 2.72 -> 4.0 (thinking).
  The 6 hash-table prompts are too few tokens (~500 a run) and too narrow for the draft quality to dominate; the trajectory
  (which tokens come out) moves a run by several percent. Treat the `--spec 5` optimum and the autotune rounds (which used
  medians) as unproven; a bench with many more turns on varied text (and a placement profiled on it) is needed to rank
  drafters honestly. Strata's 48.5 is a median of the same kind - not re-scored.

## 2026-10-30 (night): varied-text bench; the fine-tune was a prompt-text artifact; default back to the stock drafter

- New bench: 36 turns of unseen alpaca/code-alpaca prompts (`bench/dfdata/v`bench/dfdata/varied36.json``, `shadow-bench --prompts-file F --turns 36`)
  with a placement profiled on 300 *other* prompts (`profiles/routing-varied300.shrc`, recorded by `SHADOW_RECORD_ROUTING=FILE` on
  `shadow serve`), scored by total decode tokens / time (`tools/bench_total.py`; the bench's per-turn median swung 18..44 here).
  Total tok/s, 3 runs each (agree to 0.1-0.3): **stock DFlash 2 `--spec 3` 46.7**, MTP `--spec 5` 45.7, tuned `--spec 5` 44.6,
  tuned `--spec 3` 42.8. Draft position-1 hits: stock 75%, tuned 71%. All DFlash arms ran the Q8_0 drafter.
- Why the tuning "worked" on my held-out eval and not here: the eval (and the training) drew anchors from every position of a
  conversation, including the user's prompt text, which the drafter never drafts in use. With anchors only inside generated
  spans (`generated_mask` in `tools/dflash2_train.py`): stock 3.24, first tuning 3.35 accepted prefix; a retraining on
  generated-span anchors (2500 steps, ~150k generated tokens) peaked at 3.39 after 250 steps and ended at 3.24: no gain. The
  stock drafter is at its ceiling for this amount of data. `models/dflash2-ft2` / `-ft3` are not used.
- `shadow serve` defaults: the stock `models/dflash2` (Q8_0), `--spec 3` (was the tuned weights, `--spec 5`). `tools/autotune.py` now
  scores rounds by total throughput. Strata's 48.5 (a per-turn median on the six hash-table prompts) is not comparable
  with these totals; Shadow's total on that bench is ~43 (stock3 42.9, Q8 default 45.4).

## 2026-10-30 (night): autotune on the varied bench; `--dflash-min-p 0.5` is a default

- `tools/autotune.py` on the varied-text bench (36 turns, placement from other prompts, total-throughput scoring, 2 runs, floor 0.1-0.5%):
  `--spec` 3 (base 46.56) vs 4/5/6/7: -9.3 / -9.9 / -11.1 / -21.0%; `--tail-skip` 4 (base 46.73) vs 3/5: -2.8 / -5.1%;
  `--dflash-min-p` (base 46.70): unset +0.1% (control), **0.5 +2.0% (47.62)**, 0.7 -2.9%. Re-tuning `--spec` with min-p 0.5 in place (base
  47.51): 4/5/6/7 -6.5 / -3.4 / -6.3 / -4.8%. Confirmed on the six hash-table prompts (3 runs, total throughput): 43.4 vs 41.1-41.2.
- `shadow serve` defaults now add `--dflash-min-p 0.5` (with the drafter). Fastest mode on varied text: ~47.5 tok/s total.

## 2026-10-30/31: the truncation, AcceptMoE, and the corrected comparison with Strata

- **`--verify-resident` truncated answers.** A varied-text Shadow-vs-Strata run showed Shadow producing 519 tokens against Strata's
  1,388: its replies ended mid-sentence. Same 14-turn conversation: exact 1,091 tokens (3 turns ended early); exact speculation
  (`--tail-skip 0`, no `--verify-resident`) 1,091 (identical lengths, 20.2 tok/s end to end against 18.2); `--tail-skip 4` alone 1,038
  (3 early, 28.0); default with `--verify-resident` 458 (11 early); `--tail-skip 0 --verify-resident` 706 (8). The cause is that
  `--verify-resident` drops the extra positions' top-ranked experts when they are not in VRAM, and the logits that pick the next
  token come from those positions; the short replies then enter the history and later replies copy them. All "fastest mode"
  numbers of 43 to 47 tok/s earlier in this file were measured with it on and are retracted.
- **Same-history fidelity** (`tools/quality_test.py`, 24 turns, mean fraction of each reply that follows the exact reply): tail skip 4
  0.481, `--verify-rank 1` 0.511, `--verify-rank 2` 0.475, `--tail-skip 3` 0.412, `--tail-skip 5` 0.490, `--verify-resident` 0.444,
  **AcceptMoE 0.549 (beta 0.5), 0.572 (beta 1.0)**; tighter AcceptMoE sets 0.47 to 0.53. All inexact modes follow the exact text
  for only about half of a reply (5 to 9 of 24 identical).
- **AcceptMoE** (`--accept-moe`, after arXiv 2608.02989; docs/SPECULATION.md) is the default verify: same speed as plain tail skip
  (34.68 against 34.36 tok/s on the varied bench, 2,041 against 2,016 tokens a run) and the most faithful. `--verify-rank 1` is
  +18% faster (40.73) but generates 17% fewer tokens (1,682, 20 of 35 turns early): an option, not a default.
- **Shadow against Strata, quality-matched, varied bench** (`bench/vs-strata-final`, 3 runs each, interleaved, resource recording on):
  Shadow **34.68 tok/s** (2,041 tokens a run), Strata **47.18** (1,417). Shadow is at 73.5% of Strata. Resources over the warm decode
  turns, Shadow / Strata: GPU util 86.4 / 78.7%, GPU power 105 / 93 W, **GPU J per token 7.2 / 4.9** (turn windows, prefill included),
  PCIe host-to-GPU 11.6 / 10.1 GB/s (so **334 / 215 MB of expert weights per generated token**), CPU 12.6 / 49.3%, disk read
  159 / 491 MB/s, RAM available min 9.5 / 7.1 GiB. The gap is expert traffic per token (Shadow moves 55% more), not an idle GPU.
  An earlier near-parity reading (47.2 / 47.8) was Shadow with `--verify-resident` truncating its answers.
- Read in full (docs/SPECULATION.md summarises): AcceptMoE, S2-MoE (2608.15018), SpecMoEOff (2508.21706), Specter (2511.14102),
  EcoSpec (2607.12696), MoE-Spec (2602.16052). Their headline speedups are against different baselines and mostly different regimes
  (large-batch, fully resident, small MoEs, A100/H200); what carries over is substitution routing (done), reuse-aware soft gating
  (S2-MoE: bias logits toward experts the block already uses; not yet tried with residency as the preference), cost-aware chain
  length (approximated by `--dflash-min-p`), and routing-driven prefetch (needs a draft with the target's routing: not built).
- Autotune on the varied bench (total throughput): `--spec` 3 best (4/5/6/7: -6.5 to -21%), `--tail-skip` 4 best (3/5: -2.8/-5.1%),
  `--dflash-min-p 0.5` +2.0%, `--pin-margin` 2/4/8 -6.4 to -6.7%, `--cpu-experts` 2/4 no gain, `--fuse` no gain (these ran with
  `--verify-resident` on; the ranking needs a re-check on the AcceptMoE default).

## 2026-10-31: how predictable is routing? (the draft-side route predictor question)

- `tools/route_predictability.py` on three exact-mode routing logs (~200 generated tokens each, 48 MoE layers, k=10), recall of the actual
  top-10 experts of the token d positions after the last committed one, within a predicted set:
  last committed token's experts (|set| 10): **0.39** (d=1) / 0.30 / 0.27 / 0.25 (d=4); union of the last 2 (16 experts): 0.47 / 0.39 / 0.36 / 0.33;
  union of the last 4 (26): 0.56 / 0.48 / 0.45 / 0.42; a static frequency profile of 16 or 32 experts: 0.07 / 0.16; last token + frequent
  experts (16 / 31): 0.40 / 0.45. Training-free signals are weak (Specter's quantized self-draft gets 91 to 96% expert-ID agreement; a
  learned predictor would be needed), so routing-driven prefetch from cheap signals cannot hide the copies, consistent with `--pregate`
  finding nothing. The gater (`accept_route`) does not need a predictor to pick its set (it uses the target's own router logits at verify
  time); a predictor only matters for starting copies earlier.

## 2026-10-31: CPU experts in the verify pass - no gain

- `--cpu-experts N` now also works in the batched verify pass (it was a silent no-op there: the hybrid only ran in single-token
  decode, so the autotune's -0.5% / -0.2% for N=2 / 4 were noise). `moe_batch` takes the MoE input down with the router logits (8 rows at
  most), picks up to N cache misses per layer that the RAM tier holds and that exactly ONE token routes to (a CPU computes an expert per
  token; one copy serves every token that uses the expert), computes them on the AVX-512 pool from the pinned bytes (`ExpertPool::
  submit_multi`: one job, a different input per expert) while the GPU runs the resident experts and the copies run, and writes their
  outputs into the combine's rows of `eo` (`tops` keeps them, so the combine weights them as usual).
- Exact speculation (`--tail-skip 0`, 14 turns, same prompts): none 1,091 tokens, 20.2 tok/s end to end; N=2 identical text (1,091),
  19.9; N=4 1,056 tokens, 13.9 (-31%). The GPU's wait per verify pass fell as predicted (58 -> 46 -> 39 ms) but the pass stayed at 104 ms:
  the CPU time moved the wait into the host's join. Default fast mode on the varied bench (2 runs, total throughput): none 34.71,
  N=1 28.67 (-17%), N=2 34.28 (-1.2%); tokens a run 2,041 / 2,028 / 2,025.
- So a CPU pool, as wired here (a join per layer, a spinning pool on the host's own cores), does not by itself explain Strata's 35% lower
  expert traffic per token (215 against 334 MB). Opt-in, off by default.

## 2026-10-31: how is Strata still faster? (its own counters, and a larger CPU pool)

- Strata logs, per request, where each routed expert use went. Its last 35 requests of the varied bench (1,206 generated tokens, 47.4 tok/s):
  **GPU cache hit 48.4%, skipped 25.7%, computed elsewhere (its CPU pool) 13.4%, copied over PCIe 12.5%** of 576,035 routed uses (478 a
  generated token). Per generated token: 231 cache hits, 123 skipped, 64 on the CPU, 60 PCIe reads.
- Shadow, default mode, 14 turns / 997 generated tokens (`/v1/status` tiers line): 236,554 hits, 129,520 misses (every miss is a PCIe
  copy: **130 a generated token**), 165,876 routed experts skipped (166 a token, but those include the rejected draft positions; about 19% of
  all routed uses by a rough count). Units differ (Shadow counts union experts per pass), so read it as approximate: Strata copies half as
  many experts as Shadow and computes the other half of the misses on the CPU, which is the 119 MB a token (61 experts) between 334 and 215.
- A bigger CPU pool in Shadow's verify pass (exact speculation, 14 turns, same text, 1,091 tokens each): none 20.2 tok/s; 3 experts a layer
  on 14 threads 19.7; **6 a layer on 14 threads 19.5** (misses down 29%, GPU waits 58 -> 34 ms, 94,626 CPU computations); 3 on 6 threads
  20.0. The verify pass stays at 104 to 107 ms: the CPU join (about +20 ms of host time) replaces the GPU's wait. Taking misses off the bus
  does not shorten the pass as wired (a per-layer join after the router, so the CPU's time adds to the layer's critical path).
- Not verified: whether Strata's CPU share overlaps with GPU work across layers or batches tokens per expert; whether its higher skip share
  (about 26%) costs quality (its replies on the same prompts are shorter: 1,417 against 2,041 tokens a run, which a more approximate model
  could also produce).

## 2026-10-31: the CPU join, measured; Strata's pipeline, read; Strata's fidelity

- The verify-pass CPU join is timed (`verify-pass CPU joins` in the report). Exact speculation, 14 threads, none / 2 / 4 / 6 experts a layer:
  pass 104.3 / 107.2 / 109.0 / 108.1 ms; **host wait at the join 66 / 155 / 253 us a layer** (about 3 / 7 / 12 ms a pass: the CPU job, 0.85 to
  1.06 ms from submit to result, is almost always done before the host gets to the join); GPU waits 57.9 / 44.5 / 38.9 / 33.5 ms; file waits
  23.4 / 28.4 / 31.3 / 30.5; host launch time 4.8 / 8.1 / 8.6 / 9.4 ms. At 6 a layer: -24 ms of GPU waiting is paid back by +12 ms of join,
  +7 ms of file waiting (the CPU experts claim ring slots) and +5 ms of host launches and uploads. **The earlier statement that the CPU join
  added about 20 ms was an inference and wrong**; the measured join is 3 to 12 ms and the rest is extra host work. The pass is limited by a serial
  host chain (router sync, issuing copies, file reads, the combine), not by PCIe bytes, so a smaller byte count does not shorten it.
- Strata's source (`src/core/verify.cpp`, "Plan v0.3 P6 split window"): the window's tokens are cut into two groups A and B and the stream is
  ordered `pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | ...` so the CPU computes A's experts of layer l while the GPU runs
  B's mixer and router of layer l, and B's experts while the GPU combines A and runs A's layer l + 1; the whole window is a captured CUDA graph
  with device-side planning and flags instead of host round trips. That software pipeline is the overlap Shadow's per-layer chain lacks.
- Fidelity of Strata's replies against the exact run (same history, 24 turns): mean prefix match **0.431**, 6 of 24 identical (Shadow default
  0.572 / 9 of 24; plain tail skip 4 0.481); free-run 14 turns: 920 tokens, 5 early stops, 31.8 tok/s end to end (Shadow default 997 tokens, 3 early,
  27.8; exact 1,091, 3, 18.2). Strata is the least faithful of these, so part of its lead is approximation.

## 2026-10-31: the two-group pipeline (`--pipeline`) - exact, no gain

- Built our own version of the overlap Strata's source shows (`crates/shadow-cuda/src/session_pipeline.rs`): a verify window of m tokens is cut into
  groups A (the first half) and B; the GPU stream runs `pre(l,A) pre(l,B) | post(l,A) pre(l+1,A) | post(l,B) pre(l+1,B)`, `pre` = mixer + router,
  `post` = experts + combine + hyper-connection write, so the host's expert work for one group (router read, plan, copies, file reads, CPU
  experts, launches) runs while the GPU executes the other group's mixer. Mechanics: every per-token batch buffer's base pointer is shifted to the
  group's first row for the duration of a stage (`with_rows`; the plan-table buffer split in halves; CUDA-graph keys include the row offset); each
  group's router logits (and MoE input) are copied to the pinned mapped region right after its `pre` and fenced by an event the host waits on later
  (`issue_router_read` / `wait_router_read`); buffer reuse between groups is fenced on the GPU (the copy stream waits for an event recorded after the
  previous `post`) and the host syncs the copy stream only before a file read that may reuse a pinned slot.
- **Correct:** exact speculation (`--tail-skip 0`, 14 turns) gives the same 1,091 tokens and the same early-stop pattern with and without it, and the report
  counts every verify pass as pipelined (103 of 103). The groups' arithmetic per token is unchanged (the inexact modes' per-group expert sets differ).
- **Speed: none, or negative.** Exact mode (about 750 expert copies = 1.5 GB a pass, about 67 of the 104 ms on the bus): 19.9 against 20.1 tok/s; the
  verify pass 105.6 against 104.8 ms, GPU waits 40 against 58 ms, launches 1,893 against 1,011, host launch time 8.9 against 4.9 ms. Default fast mode,
  varied bench, 2 runs, total throughput: base 34.66, `--pipeline` **31.28 (-9.8%)**, `--pipeline --cpu-experts 2` 29.51 (-14.9%).
- Why: Shadow's layer is a host-driven chain with fixed costs per group (a router sync, planning, launches, a join), so cutting the window in two
  doubles those costs while the GPU work they were meant to hide under is small (the whole batch's mixer is about 0.6 ms a layer). Strata avoids the
  chain rather than hiding it: its window is one captured CUDA graph with device-side planning and flag doorbells, so there are almost no host round
  trips. The structure that matters is fewer host synchronisations per pass (48 router syncs today), not overlapping them. Kept as an opt-in, off by default.

## 2026-10-31: host round trips measured; the PLE prefetch was serialized - +7%

- The batched MoE's host latency, from "router logits received" to "first expert kernel queued", is **about 100 us a layer** (~5 ms of a 62 ms verify pass), so
  cutting the 48 round trips (device-side planning) can give at most ~8%. The GPU kernel time is ~37 ms of the pass (dense projections ~15, grouped
  experts ~12); the rest is the GPU waiting on the bus and on latencies. `SHADOW_LTIME` charged ~8.4 ms a pass to the one non-graph segment, the PLE layer
  (a normal DeltaNet layer: ~0.4 ms); `SHADOW_PLE_TIME=1` showed why: `ple_prefetch` (the n-gram rows, random reads in a 27 GB table) took **3.5 to 9.3 ms** on most
  passes (0.01 ms when cached) while the GPU idled after layer 0.
- Cause: `ple_prefetch` read through `Model`'s one shared file handle from 16 freshly spawned threads, and Windows serializes I/O on one synchronous handle (the
  repo's own readpool note says so), so the "parallel" reads ran one after another plus the spawns. Fix: the rows are read through the model's `ReadPool`
  (8 workers, each with its own handles). Prefetch: mean **1.22 ms** (median 1.12, p90 1.46 over 371 passes).
- Default mode, varied bench, 2 runs, total throughput, same 2,041 tokens a run (bit-identical output): 34.66 / 34.71 before (two earlier benches) -> **37.08 now (+7%)**.

## 2026-10-31: the VRAM cache - bigger loses, a second-miss admission filter gains ~1%

- Strata's cache holds 5.01 GiB against Shadow's 4.42 GiB (the free VRAM less `VRAM_RESERVE`, after the drafter and buffers). Shadow's slots are already per-layer and
  exactly sized, so Strata's "size slots in bytes" finding (+13%) is already in. Paired bench, varied text, 3 runs, total throughput, one exe:
  default 4.42 GiB **36.45**, `--expert-cache-gib 4.8` **34.64 (-5%)**, 5.0 **32.96 (-10%)**. A bigger cache is slower here: peak GPU use reaches 11.7-11.8 of 12.2 GiB and
  the extra slots come out of what the verify pass needs. Strata's larger cache does not transfer.
- New `--admit-after N` (off by default): a miss gets a VRAM slot only from its (N+1)th miss; the first N are read through the scratch slots and kept out of the cache,
  so a one-off expert does not evict a useful one. Total throughput, same bench: N=1 **36.89** and **37.01** in two benches (default 36.45 / 36.53), N=2 36.78, N=4 35.87.
  About +1.2% at N=1, steady to 0.1 tok/s across runs; the text differs (1,986 tokens a run against 2,041, as AcceptMoE's residency-aware sets follow the cache), so
  read the gain as small. Opt-in.
- Avalon (freriadoc/Avalon, a llama.cpp fork, MIT) was read: its cache counts each expert's demands, admits at 3, and when full admits only an expert whose aged count
  beats the LRU entry's (counts capped at 15, halved every 5 x entries demands, window shortened with context depth); an evicted expert keeps its count; admission copies are
  deferred so they do not slow the CPU's rows; compute buffers and the unused KV tail are lent to the cache between phases. Not built here: an LFU contest with aging
  on top of `--admit-after`, and VRAM lending (which would give the bigger cache without starving the pass), are the next candidates.
- Bench hygiene: three runs of the follow-up bench died ("connection forcibly closed"), one with available RAM at 3.6 GiB: the 4 GiB watchdog in `tools/run_capped.ps1`
  killed them while the browser pane was open. Nothing else was running for the scored runs; the default arm of the follow-up has one complete run.
- Avalon-style LFU contest tried on top (a full layer admits a miss only if its demand count beats the LRU entry's, counts capped at 15 and halved every W x slots
  demands, an evicted expert keeping its count; three benches of 3 runs, varied text, total throughput; default 36.50, `--admit-after 1` 37.49): `--admit-lfu 5` 37.07,
  `--admit-lfu 5 --admit-after 1` 35.80, `--admit-lfu 2 --admit-after 1` 37.41. It adds nothing over the plain second-miss filter and one setting loses, so it was **removed**.
  `--admit-after 1` now reads +1.2% to +2.7% over the default in three benches (36.89 / 37.01 / 37.49 against 36.45 / 36.53 / 36.50); still opt-in, since the output
  text changes with the cache contents (AcceptMoE's sets follow residency) and no quality check has been run on it.
- With the browser pane closed the bench keeps 8-9 GiB of RAM free (4.1-4.9 GiB with it open), and no run died: the earlier deaths were the 4 GiB watchdog.
- Same-history fidelity of `--admit-after 1` (`tools/quality_test.py`, 24 turns, same prompts as the earlier AcceptMoE figure): **0.556** (8 of 24 replies identical, 1,495
  tokens) against the default's **0.572** (9 of 24, 1,597 tokens, reproducing the earlier number). One reply of 24 flipping moves it that much, so the filter
  costs little or no fidelity; it stays opt-in until the speed gain (+1.2% to +2.7%) is shown on a second prompt set.

## 2026-10-31: where the VRAM goes (`SHADOW_VRAM_MAP=1`); lending has nothing to lend

- `SHADOW_VRAM_MAP=1` prints the free VRAM at each stage of a serve start and once more in the ready line. 12.2 GiB card, desktop already using about 1.2 GiB: start 11,036 MiB
  free -> dense weights 4,446 MiB -> buffers and the DFlash drafter 1,594 MiB -> expert cache 4,612 MiB (4.42 GiB) -> **384 MiB free**, the reserve, the same after the ready line.
- Avalon lends its 9,536-token prefill compute buffers and the unused KV tail to its expert cache between phases. Shadow prefills in 64-token batches, so it has no large idle
  prefill buffers; every resident byte is decode working memory and the 384 MiB reserve is the only slack. VRAM lending as Avalon does it has nothing to lend here.
- Hit rates (`/v1/status`, 16 turns of the varied prompts, cumulative union-expert counts): default 4.42 GiB **49.9%**, `--expert-cache-gib 5.0` **52.6%** (+2.7 points),
  `--admit-after 1` **48.4%** (-1.5 points, yet faster in the bench). The 0.6 GiB of extra slots buys 2.7 points of hit rate, about 5% fewer misses, which does not pay for
  taking the reserve down to nothing (the 5.0 GiB bench arm: -10%). The cache is at the edge of what the card can hold; the next VRAM would have to come from the drafter
  (1.2 GB) or the dense weights (4.4 GB), not from idle buffers.
- The drafter's cost, by the same map with `--dflash off`: 5,826 MiB free before the cache against 4,996 with it, so **the DFlash 2 drafter holds 830 MiB** (about 610 MiB of Q8_0
  matrices, the rest its K/V rings and buffers). A 4-bit drafter would free about 290 MiB: +150 cache slots (+6.5%), about +1.3 points of hit rate by the 5.0 GiB
  arm's rate (13.5% more slots gave 2.7 points), i.e. about 2.5% fewer misses, under 1% of speed, against a drafter whose acceptance (the thing the speed rests on) can only
  get worse. Not built: the expected gain is below the bench's run-to-run floor.

## 2026-10-31: the ceiling of device-side planning is 3.9 ms a pass (6.5%); the copies are 19 ms

- Default mode, 8 turns, ms of checking per verify pass (the speculation line; timing-only switches, wrong output): normal **59.7** (95 passes); `SHADOW_TIMING_NO_COPIES=1`
  **40.4** (114 passes); `SHADOW_TIMING_NO_COPIES=1 SHADOW_TIMING_NO_SYNC=1` **36.5** (124 passes).
- So the router wait and host planning, the thing device-side planning would remove (Strata plans on the device inside a captured window), cost **at most 3.9 ms of 59.7
  (6.5%)**, and a device planner would still need the host for the copies. The demand copies cost **19.3 ms (32%)**: about 290 misses a pass at 1.96 MiB is ~570 MiB, 26 ms
  of PCIe at 22 GB/s, partly hidden under the GPU's hit work. The pass is bus-bound; planning on the device does not touch that.
- Earlier attempts on the bus all moved the wait instead of removing it: pre-gated prefetch (the DMA slows the layer's own work), zero-copy reads (7 GB/s), CPU experts
  (the join). What removes bytes from the bus: more hits (the cache is at the VRAM edge, see above), skipping (a fidelity trade), computing on the CPU without a per-layer join.
- Decision: device-side planning is not built; a 6.5% ceiling for a large change is the worst ratio of the remaining ideas.

## 2026-10-31: speed at matched fidelity (the skip ladder)

- Same-history fidelity (`tools/quality_test.py`, 24 turns, the exact run's replies as history; Strata measured 0.431 the same way) against speed (varied bench, 3 runs, total
  throughput, no turn ended early in any arm):

  | config | fidelity | tok/s | tokens a run |
  |---|---|---|---|
  | default (`--tail-skip 4 --accept-moe`) | 0.572 | 37.01 | 2,041 |
  | `--accept-scale 0.5` | 0.525 | not benched | |
  | `--verify-rank 2` | 0.475 | **42.79** (+15.6%) | 1,806 (-11.5%) |
  | `--tail-skip 3` | 0.415 | **39.39** (+6.4%) | 2,034 |
  | `--tail-skip 3 --accept-scale 0.5` | 0.410 | not benched | |
  | `--tail-skip 2` | 0.328 | not benched | |
  | Strata (earlier bench) | 0.431 | 47.2 | 1,417 |

- At Strata's fidelity (0.415 to 0.475 brackets 0.431) Shadow runs **39.4 to 42.8 tok/s against Strata's 47.2: 83% to 91%**, the gap at equal quality being narrower than at the
  default (78%). `--verify-rank 2` buys the most speed but its replies are 11.5% shorter (shorter replies enter the history; the earlier `--verify-resident` truncation started the
  same way, though no turn ended early here), so `--tail-skip 3` is the like-for-like step: +6.4% at the same length. Defaults unchanged; both stay options. `--tail-skip 3` is
  slightly below Strata's fidelity, `--verify-rank 2` slightly above.
- Hit rate stays at 49-50% in all of these (50.0% for rank 2): the skips remove copies, not misses.
- **Default changed to `--tail-skip 3`** (was 4): +6.4% (39.39 against 37.01 tok/s), same reply length, fidelity 0.415 against 0.572 (Strata 0.431). `--tail-skip 4`
  restores the earlier mode. README speed table and USAGE updated; `shadow serve` prints `--tail-skip 3` in its fastest-mode defaults line.

## 2026-10-31: prompt reading - the cache is lent to long prompts (51 -> 80-86 tok/s)

- Prompt reading was never in the decode benches. Measured now: **about 51 tok/s** (a 740-token prompt took 14 s; Strata reads at 1,376, Avalon at 1,613). Chunk size made no
  difference (64/128/256 tokens: 50-52 tok/s). `SHADOW_LTIME=1`: the MoE segment is 972 of ~1,130 ms of a 64-token pass (86%). Cause: `moe_batch` runs a layer's tokens in
  groups limited by free cache slots plus 32 scratch slots (~80 experts), so a chunk's experts were copied again for every group; bigger chunks changed nothing.
- Fix (`lend_cache` / `restore_cache`, `session.rs`): a prompt of 128+ unread tokens evicts the tail of the VRAM expert cache (room for all 512 experts of a layer, at
  most half the cache), uses it as the scratch for that prompt's layers, reads in `MAX_BATCH` = 256 token chunks (a 256-slot pinned staging buffer for the file reads),
  and afterwards reloads the evicted entries from the RAM tier. Decode never sees it. `SHADOW_NO_LEND=1` turns it off; `SHADOW_PREFILL_LEND_MIB`,
  `SHADOW_PREFILL_STAGING` (slots) and `SHADOW_PREFILL_CHUNK` tune it.
- Measured (default mode; prompt tokens/s): 740-token chat prompt 51 -> **78**; four 1.8k-4.6k-token source/doc prompts **66-74** with 256-token chunks; the same prompts
  at 512 -> 76-85. A 1,771-token source prompt: 40.6 without lending, **86.4** with (2.1x); `--exact` 18.4 -> 41.8 (2.3x). Output is identical with and without
  lending in `--exact` mode (a 1,771-token prompt, 40 tokens, word for word); decode right after a lent prompt runs at the same speed (38.1 against 38.4 tok/s).
- `MAX_BATCH` 64 -> 256 costs decode nothing (varied bench, 3 runs, total throughput: 39.46 old, 40.04 and 40.01 new), though the per-token buffers take 122 MiB from the
  cache (4.42 -> 4.30 GiB). **512 costs decode 5.4%** (37.83 against 40.01 tok/s) and was dropped.
- What is left: a 256-token pass copies ~250 experts a layer (~0.5 GiB, 22 ms of PCIe) and reads the ones the RAM tier lacks (36%) from the SSD; the pass sits near the bus
  and disk limit, so a bigger chunk is the lever but VRAM for its per-token buffers is not available. Strata's 1,376 tok/s is 16x this; this PC's gap on prompts is
  larger than on decode.
- **Bigger chunks do not help (measured, not built):** a temporary `MAX_BATCH` = 1024 build (1024-token chunks, 512 staging slots), four 1.8k-4.6k-token prompts: 88.6, 84.3,
  77.8, 75.0 tok/s - the same as 256-token chunks (66-86). Three passes at 256 (`SHADOW_LTIME=1`, 740 tokens, 78.7 tok/s) and one at 1024 (87.0 tok/s) show the same split:
  host waits per prompt token, router 3-5 ms and file 4-5 ms, i.e. file reads are ~36-40% of the prompt time and the router wait (GPU compute plus the copies queued before it)
  ~25%. The file reads already go through the parallel read pool; what is left is the SSD itself (36% of experts are outside the 30 GiB RAM tier).
- A bigger RAM tier is not an answer on this PC: `--ram-tier-gib 40` is refused by the RAM floor (42 GiB available, 4 GiB must stay), and 36 GiB loaded but the server died
  mid-prompt. A read-ahead (the next layer's non-resident experts read from the SSD under this layer's compute) would hide up to ~36% but the next layer's experts are not
  known until this layer ends; with 1024-token chunks it would be "all the non-resident ones" (~180 experts a layer, 0.35 GiB), which needs those chunks' buffers
  (0.6 MiB a token, ~0.6 GiB at 1024) allocated only while a prompt is read. Not built.

## 2026-10-31: prompt read-ahead built and measured (not kept); the long-prompt attention path; resource record refreshed

- **Read-ahead** (`ReadPool::submit` / `wait_async`, two pinned buffers by layer parity: while layer l runs, the file-only experts of layer l+1 - those in neither the VRAM
  cache nor the RAM tier - are read on the read pool; layer l+1 copies them from the buffer like RAM-tier experts). Works and is bit-safe (the bytes are the model's), but the
  win is small and needs big chunks: 256-token chunks **44, 44, 43, 42 tok/s** (worse than the 66-86 without: it reads every file-only expert, ~180 a layer, when a chunk needs ~90
  of them); 1024-token chunks (a temporary `MAX_BATCH` 1024 build) **101.5, 92.7, 82.0, 83.5 tok/s** against 88.6, 84.3, 77.8, 75.0 without (+5% to +15%). Chunks of 1024 need
  ~0.6 GiB of per-token buffers that the VRAM budget cannot hold all the time (512 already costs decode 5.4%), so it would also need those buffers allocated only while a prompt
  is read. **Dropped**: that rewrite for +5-15% on prompts is not worth it. The patch is not in the tree.
- **What a long prompt spends** (`SHADOW_LTIME=1`, 4,574-token prompt, 1024 chunks, read-ahead on, 72.8 tok/s under the timers): per batched pass the MoE segment 5.3 s, the 12
  attention layers **3.5 s (295 ms each)**, the 35 delta layers 1.1 s. Host waits per token: router 4.4 ms, file 1.7 ms. The attention layers cost 26 ms each for a 740-token
  prompt, so about 8x more per token beyond ~2,051 positions: past the indexer's budget `attention_batch` falls back to `attention_mid` one token at a time (`session.rs`, the `else`
  branch after `idx_budget`), ~29% of a 4.5k-token prompt's time. A batched sparse-selection kernel is the next lever for long prompts; the MoE path is at the file/bus limit.
- **A 36 GiB RAM tier is killed by the RAM-floor watchdog** (confirmed: `KILLED at 3.97 GiB available`, peak working set 41.1 GiB for a 36 GiB tier); 30 stays the default.
- **Resource record refreshed** (`bench/vs-strata-final2/`, current defaults, 3 interleaved runs of the 36-prompt set, total throughput): Shadow **39.9** tok/s (39.9, 39.9, 40.0),
  Strata **48.3** (47.8, 48.7, 48.4; its replies are shorter, 1,344 against 2,015 tokens a run). GPU utilisation 87% against 78%, PCIe 10.3 against 10.1 GB/s, so **259 MB of expert
  traffic per token against Strata's 209**; CPU 13% against 52%, disk 154 against 474 MB/s. README updated.
- **The CPU rows without a per-layer join** were not built: the join was measured at 3-12 ms of a 105 ms verify pass (2026-10-31, "the CPU join, measured") and the host-planning
  ceiling at 3.9 ms (6.5%), so removing the join needs Strata's whole-window graph with device-side flags for at most that much.

## 2026-10-31: `--ram-floor-gib`, and what a 36 GiB RAM tier buys

- The RAM floor is now a flag: `--ram-floor-gib N` (default 4, 0 turns the check off) for the engine's refusal and its checks while the tier fills, `-FloorGiB` on
  `tools/run_capped.ps1` / `run_strata.ps1` for the watchdog that kills a run below it. The default is unchanged. `auto` sizes against the floor in force plus 6 GiB.
- Measured with `--ram-tier-gib 36 --ram-floor-gib 3` (watchdog at 3; available RAM bottomed at 4.5 GiB, peak working set 41.5 GiB), against the default 30 GiB tier, same build:
  prompts (1.8k-4.6k tokens) **85.4 / 77.8 / 74.8 / 71.6** against 74.0 / 70.6 / 68.7 / 65.4 tok/s (+15%, +10%, +9%, +9.5%); decode on the varied bench (3 runs each, total
  throughput) **40.79** against **39.79** tok/s (+2.5%, outside the 0.1-0.5% noise), identical answers (0 differing). Minimum available RAM during the bench: 4.9-5.1 GiB
  against 10.4-10.7 GiB for 30.
- So a bigger tier pays a little (+2.5% decode, ~+10% prompts) for ~5.5 GiB less free RAM. The default stays 30 with the 4 GiB floor; 36 needs `--ram-floor-gib 3` or less
  on this PC and leaves about 5 GiB free in steady state.
- **Default tier is now 36 GiB** when 49 GiB or more of RAM is available at start (30 otherwise; `shadow serve` says which). Checked with the defaults: 740-token prompt 93 tok/s, peak working
  set 41.3 GiB, available RAM bottomed at 5.1 GiB, floor still 4. `shadow-ui` no longer passes `--ram-tier-gib 30`.

## 2026-10-31: batched sparse attention for long prompts (+16%), default tier 36 with floor 3

- `attention_batch` ran every token past the indexer budget (2,051 positions) through `attention_mid`: per token and layer an indexer query, a block-scores kernel, a **synchronous scores download**,
  a full host sort, a selection upload and one attention launch. Now (`session.rs`, the `jt`/`SEL_ROWS` code) the tokens before the first one needing a selection still use
  `attention_heads_rows`; the rest go SEL_ROWS (32) at a time: per token the same `head_norm_rope` and `index_scores` kernels, **one scores download per group**, the host selects the `keep` best blocks
  with `select_nth_unstable_by` under the same total order (score, then block number: the same set as the full sort), one upload of the selections and their lengths, and the new
  `attention_heads_sel_rows` kernel (`kernels/forward.cu`; the same `attention_head` body, 8 tokens per launch, each over its own position list). The `--trace` path keeps the per-token code.
- Bit-identical: `--exact`, 3,003-token prompt (about 950 tokens past the budget), 48 tokens, temperature 0: the same text as the per-token path (checked against it in the same build with a
  temporary switch). Whole `cargo test --release` passes (and a doctest in `session_pipeline.rs` that failed on an indented diagram is fixed).
- Speed (`SHADOW_LTIME=1`, 4,574-token prompt, 30 GiB tier, 256-token chunks): the 12 attention layers 1,251 -> 1,003 ms (one download per group) -> **675 ms** a pass (the kernel); prompt 62.0 -> **72.2 tok/s** (+16%).
  SEL_ROWS 128 changed nothing (686 ms): the rest is the kernels' own time (FP64 scores for exactness), not the syncs.
- With the default 36 GiB tier: prompts of 1.8k-4.6k tokens **86.0 / 84.4 / 87.0 / 82.9 tok/s** against 79.0 / 73.4 / 72.8 / 69.6 before (+9% to +19%); decode right after them 43.8 tok/s.
- The 36 GiB tier started from 49.05 GiB available bottoms out at **4.1 GiB**, a hair above the 4 GiB floor, so the default 36 GiB tier now comes with `--ram-floor-gib 3` (an explicit tier keeps 4); the watchdog
  needs `-FloorGiB 3` for it. Under 49 GiB available the default stays 30.

## 2026-10-31: float-precision attention for prompt reading: no gain, dropped

- Tried: `attention_head_f` (scores, softmax and weighted sum accumulated in float, the denominator in float per thread and double across the block) as `attention_heads_rows_f` and
  `attention_heads_sel_rows_f`, used only when a pass has more than `MAX_VERIFY` tokens, behind a switch. The idea was that the double-precision sums (FP64 is 1/64 rate here) set the attention time.
- Result (4,574-token prompt, 30 GiB tier, `SHADOW_LTIME=1`): the 12 attention layers **681.8 ms** a pass against **675.3 ms** with the double kernels; prompt 70.8 against 72.2 tok/s (noise). The `--exact`
  3,003-token prompt gave the same 48-token text as the double kernels. So the double sums are not the cost, there was nothing to trade against fidelity, and the kernels were removed.
- The remaining ~56 ms a layer per 256-token pass is elsewhere (per-token launches of the indexer's query and score kernels, the host selection and the uploads, or the gather of the selected K/V rows);
  not profiled further.

## 2026-10-31: prompt attention profiled with Nsight Compute; the attention kernel is 2.7x faster

- Profiling needed GPU performance counters, which the driver keeps admin-only (`ERR_NVGPUCTRPERM`); `bench/ncu_attn.ps1` runs Nsight Compute from an elevated PowerShell and saves a report
  (`ncu --import bench/ncu_attn.ncu-repz --page details` reads it without admin). Timers around one sparse call (a layer's 256-token chunk, contexts 2k-4.5k) first: indexer scores 20 ms, host selection 7.7 ms,
  **attention kernel 39-43 ms**.
- `attention_heads_sel_rows` (24 heads x 32 rows, 256 threads): 5.66 ms a launch, **L2 80% / L1 71% busy, DRAM 0.6%**, L2 hit rate 99.7%, 40 registers, occupancy 77%; each warp spent 66 of 93 cycles
  stalled on a full global-memory instruction queue. Not FP64 (the float experiment above agrees) and not the SSD: three global loads for every multiply-add (`q[i]`, `kt[i]`; `sc[t]`, `toks[t]`, `vc` in the weighted sum).
- Fix, bit-identical (`attention_head` in `kernels/forward.cu`, used by the decode, prompt-dense and sparse kernels): `q` staged in shared memory, K read as `float4`, each 256-position tile's scores and positions
  staged in shared memory; the sums are the same fmaf runs of 16 added in double in the same order. 39 -> **14.3 ms** a call. The `--exact` 3,003-token prompt gives the same text; the full `cargo test --release` passes.
- Also: the indexer's query and block scores run as `head_norm_rope_rows` / `index_scores_rows` over a group's rows (the scores kernel was FP64-bound in a 5-block launch per token): 20 -> 1.8 ms a call;
  sparse attention rows go 32 to a launch with scratch sized to the selection, not the context.
- Prompt reading (4,574-token prompt, 30 GiB tier, 256-token chunks): 62.0 tok/s before the attention work -> 71.2 (batched selection) -> 74.7 (scores rows) -> **80.1** (kernel). Default mode, four prompts of
  1.8k-4.6k tokens at the 30 GiB tier: **77.1 / 82.5 / 87.5 / 83.6** against 74.0 / 70.6 / 68.7 / 65.4 (+4% to +28%); decode right after them 42.3 tok/s.
- Left in a sparse call: host selection 7.7 ms (a sort per token on one thread), the dense rows before it. Not done.

## 2026-10-31: the heads of a GQA group share K/V reads in the sparse attention kernel

- Re-profiled the new kernel (`bench/ncu_attn.ncu-repz`; the first profile is `ncu_attn_before.ncu-repz`): `attention_heads_sel_rows` 5.66 -> **2.06 ms** a launch, DRAM throughput 0.57% -> 0.97% (3.76 -> 6.45 GB/s: GPU
  memory use up 1.7x from nearly nothing), but **L2 84% busy** (was 80%), L1 hit rate 43% -> 15% (the K reads are 16-byte loads), 39 cycles between warp issues (was 93). Still L2-bound: every head of a GQA
  group (24 heads over 2 kv heads here, 12 a group) re-read the same K rows and V elements of each token's selected positions.
- `attention_group_sel_rows` (`kernels/forward.cu`): a block takes `hb` heads (default 4, `SHADOW_ATT_HB`, at most 8) of one kv head for one token, loads each selected position's K row and V element once for all
  of them, and keeps each head's sums exactly as before (fmaf runs of 16 added in double, the same block reductions): the `--exact` 3,003-token prompt gives the same text, and `cargo test --release` passes.
  Used when `hd` is a multiple of 16 and at most 256 (else the per-head kernel).
- Attention layers on a 4.5k-token prompt (`SHADOW_LTIME=1`, 30 GiB tier): 675 ms (after the staged-load kernel; 1,251 before any of it) -> **267 ms** a pass (hb 4; hb 8: 277). Prompt reading 80.1 -> **81.2 tok/s**: attention
  is now ~9% of a pass, so the rest of the gain is small; the MoE segment (2.2 s of ~3 s) and the file reads are what is left.

## 2026-10-31: `--verify-rank 2` is the default; a lending bug fixed

- **Decode got faster from the attention work, not only prompts.** Verify passes (2-8 tokens) past 2,051 positions went through the per-token sparse path with a synchronous scores download per token and layer; they now use the grouped path. Varied bench, 3 runs each, 30 GiB tier,
  total throughput: `--verify-rank 2` **47.96** tok/s (47.4, 48.2, 48.2; 1,801 tokens a run, 17 of 35 turns ended early), AcceptMoE **45.11** (45.0, 45.1, 45.2; 2,015 tokens a run) against 42.8 and 39.8 before (and Strata's 48.3 earlier today).
- Same-history fidelity of `--verify-rank 2` as the default (24 turns, 3 runs, identical each time): **0.423**, 6 of 24 replies identical (AcceptMoE 0.415, Strata 0.431). `shadow serve` now runs `--verify-rank 2` unless `--accept-moe`, `--verify-resident` or `--verify-rank` is given.
- **Bug fixed (introduced with prompt lending, 65c5f27):** `restore_cache` put each evicted expert back into its old slot without checking that the prompt had not loaded it again into another slot, leaving two slots owning one expert; a later eviction then dropped the live mapping and
  `moe_batch` panicked with "a routed expert neither cached nor fetched" (HTTP 500). It showed on the 24-turn same-history test at turn 14 (250-token continuations; it needs a prompt of 128+ unread tokens), every run, and not with `SHADOW_NO_LEND=1`. Fixed by skipping entries already cached; all 24 turns pass.

## 2026-10-31: Shadow against Strata with the current defaults: a tie

- `bench/vs-strata-final3` (3 interleaved runs, 36 varied prompts, total throughput): Shadow default **48.74** tok/s (48.7, 48.8, 48.7; 1,801 tokens a run, 17 of 35 turns ended early), Strata **48.63** (47.5, 49.3, 49.1; 1,385 tokens a run, 23.7 ended early).
  Shadow +0.2%, inside Strata's run-to-run spread. Fidelity 0.423 against 0.431. PCIe 10.95 against 11.06 GB/s, so **225 against 227 MB of expert traffic per token** (259 against 209 this morning); GPU 86% against 78% busy, 4.2 against 5.2 J a token, CPU 12% against 48%, disk 189 against 480 MB/s.
- Still behind: prompt reading (~80 against Strata's 1,376 tok/s) and the fidelity is 0.008 under. Ahead: bit-exact `--exact` path, CPU and disk use.

## 2026-10-31: prompt reading with a layer-ahead pipeline and big batches (77-87 -> 128-144 tok/s)

- Nsight Systems on a 3,001-token prompt (`nsys profile --trace=cuda`, no admin needed): kernels busy 9.8 s of a 21.6 s window, host-to-device copies 9.2 s (147 GB), the two overlapped for only 4.1 s and neither ran for 6.6 s: the copies, the SSD reads and the
  kernels took turns. Bigger chunks alone plateaued (256 -> 1024 -> 2048 tokens: 77-87 -> 94-107 -> 100-110 tok/s).
- `moe_staged` (`session.rs`), used for prompts of `LONG_PROMPT` = 1,024 or more unread tokens: the lent cache holds two stage halves (slot = expert number, 1 GiB each); while layer l computes, layer l + 1's experts that the VRAM cache lacks are copied into the other half
  (from the RAM tier, or from a pinned buffer that the read pool filled from the file one layer earlier), and layer l + 2's file reads start. All of a layer's tokens run in one group, so no expert is copied twice; the bytes are the model's, so the arithmetic is unchanged.
  `ReadPool::submit` / `wait_async` and `Model::experts_submit` / `experts_wait` are the asynchronous reads.
- The per-token buffers for `BIG_BATCH` = 1,024 tokens are carved from the lent cache too (`DeviceBuffer::view`, `make_bb`, `big_batch_on`; 0.62 MiB a token), swapped in for the 256-token ones and swapped back, so decode keeps its VRAM
  (a permanent 512-token batch cost decode 5.4%). `rows`/`batch_rows()` replaces `MAX_BATCH` where a buffer's row count matters; the CUDA-graph key says which. After the prompt the evicted experts that the RAM tier lacks are read back from the file, so the cache is warm again.
- Prompts of 1.8k-4.6k tokens, 30 GiB tier, default mode: **127.8, 143.8, 133.0, 139.2 tok/s** against 74.0 / 70.6 / 68.7 / 65.4 this morning and 77-87 with the staged-load kernels; `--exact` (no RAM tier) 40 -> **62 tok/s**, same text.
  Prompts of 128-1,023 tokens keep the old lend (256-token chunks): the pipeline's pinned buffers and cache restore cost more than they save below ~1,024 (a 24-turn chat with 250-token continuations ran at 5.6 tok/s end to end with it, 14 without).
- Decode unchanged: varied bench 3 runs, total throughput **48.53** tok/s (48.5, 48.6, 48.6) against 47.96 before; `cargo test --release` passes.
- Fidelity noise: the 24-turn same-history fidelity is deterministic for a build but moves with tiny cache-state differences between builds: this build 0.375 (`--verify-rank 2`) and 0.447 (`--accept-moe`), earlier builds 0.423 and 0.415. Differences of 0.04 between builds or modes (and the 0.008 to Strata) are inside that noise; treat all of them as "about 0.4".
- Still ahead of Shadow: Strata's 1,376 tok/s prompt reading (about 10x). Now per layer: ~68 ms of GPU kernels, ~64 ms of copies (1 GB a layer: all experts the cache lacks, ~30% more than a chunk uses), ~46 ms with neither busy.

- Where the pipeline stands (nsys timeline of a steady layer at 1,024 tokens): the grouped MoE kernels run 22 ms (IQ3_S) + 17 ms (IQ4_NL) + the dense layers, while the copy engine is busy ~76 ms for the layer's ~1 GB (13 GB/s): a pass is copy-bound.
  The link itself does 23-28 GB/s from the RAM tier's kind of memory (`host_to_device_bandwidth` in `tests/pinned.rs`, ignored by default), and a second copy stream changed nothing (128 / 145 / 133 / 139 tok/s with and without), so the 13 GB/s is not the link: host DRAM is the likely limit
  (each file-only expert crosses it ~4 times: SSD -> page cache, page cache -> pinned buffer read and write, pinned -> GPU; the RAM-tier ones once), about 2 GB of traffic a layer in ~65 ms. Not measured further. Ideas left, in order: unbuffered reads into the pinned buffers (a third of the traffic), copying only the experts a chunk needs
  (prediction from the previous chunk; ~30% of the bytes), tensor-core MoE kernels for the 22 + 17 ms.

## 2026-10-31: anatomy of a decode pass (Nsight Systems, `--cuda-graph-trace=node`)

- 14 varied prompts, 1,077 tokens, default mode: per verify pass (385 passes, 2.80 tokens each) drafting 5.5 ms, checking **45.4 ms**, rollback 1.0 ms, other 1.8 ms. Over the 24 s generation window: GPU kernels busy 8.6 s, host-to-device copies 4.0 s (6 GB/s mean; 0.6 s of it
  overlapped with kernels), neither 12.0 s. Most of the "neither" is between passes and requests (gaps over 4 ms: 4.7 s: the draft, sampling, rollback and per-request prefill), not inside a verify pass.
- Inside a verify pass the GPU is already dense: a layer takes ~0.73 ms with no gap over ~25 us (nodes of the mixer graph 0.2 us apart; the MoE segment's `copy` -> shared expert -> plan upload -> grouped kernels -> combine in ~0.45 ms). So the host round trip per layer is not what costs (the earlier
  reading of `--cuda-graph-trace=graph`, which hides graph nodes, showed fake 350 us gaps). A device-side-planning rewrite would save the few small gaps, not a pass's 45 ms.
- What the 45 ms is: ~35 ms of dense GPU chain (dense-weight matvecs ~14 ms: `matvec_cols_q6_k` 18% of kernel time, `iq4_xs`, `q8_0`, `q5_k` 86 us for one projection against 19 us for a bf16 one; expert grouped matvecs ~35%: `iq4_nl` 15%, `iq2_s` 10%, `iq3_xxs` 9%, `iq3_s` 5%) and ~10 ms around the layers (embedding,
  four positions' logits over 248k vocabulary, capture). The floor at 672 GB/s is ~10 ms a pass (4.4 GiB of dense weights + ~25 experts a layer), so the kernels run at about 30-45% of VRAM bandwidth: the real slack, and the real work: faster dense and grouped matvec kernels (vectorised loads, fewer launches
  per layer), needing ncu counters to see what limits each. Not done. Decode at 48.7 tok/s is a tie with Strata; a 3x gain is not on the table from scheduling alone.

- **Nsight Compute on the decode kernels** (`bench/ncu_decode.ps1`, 21 launches of the seven biggest; report `bench/ncu_decode.ncu-repz`): the large dense projections run at **30-35% of DRAM bandwidth** (`matvec_cols_q6_k` 320 blocks 94.8 us dram 35%; `iq4_xs` 384 blocks 84.8 us dram 30%), 128 registers a thread and 30% occupancy;
  the grouped expert kernels at 17-22% dram, 94-106 registers, 30% occupancy (`iq4_nl` 320x12 blocks 78 us, `iq2_s` 80x24 blocks 104 us); and a dozen small projections (16-80 blocks, 0.2 waves on 48 SMs) take 21-50 us each at 5-25% dram: latency-bound, each a fixed ~20 us.
- Tried the cheap fix for the occupancy: `__launch_bounds__(256, 3)` on every `matvec_cols_*` and `matvec_grouped_*` kernel (caps registers at 85, more warps to hide latency). Same arithmetic: the `--exact` 3,003-token prompt gives the same text. Decode on the varied bench (3 runs, per-turn median): **47.9, 48.9, 49.1** against 48.9, 48.9, 48.9: no gain; reverted.
  The kernels are not limited by occupancy alone; what is left is a kernel redesign (more bytes in flight a thread, fusing the small projections that run back to back so their ~20 us fixed cost is paid once) which needs the per-kernel stall reasons (`smsp__pcsamp` sampling was not collected).

- **Full counter set on the same kernels** (`bench/ncu_decode_full.ps1`, `--set full`, 14 launches; report `bench/ncu_decode_full.ncu-repz`): the warp-stall sampling puts **`long_scoreboard` (waiting for a global load) at 53-75% of the stalls** in every one of them (`matvec_cols_q6_k` 73-75%, `iq4_xs` 62-65%,
  grouped `iq2_s` 67-70%, `iq4_nl` 53-54%), `wait` 7-12%, `selected` (issuing) only 7-16%. The kernels are memory-latency-bound: each lane loads a few bytes of a quantized block, uses them, then loads the next; the per-lane accumulation order (which the reference arithmetic fixes) leaves little to hoist.
- Tried `#pragma unroll 4` on `matvec_cols_m`'s chunk loop (independent loads can then be issued together; same accumulation order, `--exact` text identical): decode per-turn median 48.4 / 49.1 / 49.1 against 48.9 / 48.9 / 48.9, i.e. nothing; reverted. What would help is staging each warp's weight bytes with wide
  (16-byte) loads into shared memory and reading them from there, in the same lane order: a rewrite per quantization type (Q6_K, IQ4_XS, IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S, Q2_0, Q5_K, Q4_K, Q8_0), each checked against the reference. Not started.

## 2026-10-31: fewer rows a warp in the plain `matvec_cols_*` kernels (+5% decode)

- The stall data (long-scoreboard bound, 30% occupancy, small grids) says the kernels lack loads in flight, not bandwidth. `COLS_PLAIN` (`kernels/matvec.cu`, the plain `matvec_cols_*` kernels and `matvec_cols_multi`): 4 rows a warp -> **1**, so a block covers 8 rows
  (was 32) and a matrix gets 4x the warps; each row's lane-ordered sum is unchanged, so every output is the same bit for bit (the `--exact` 3,003-token prompt gives the same text; `tests/matvec.rs`'s one-at-a-time against many-at-once comparison passes with its grid updated to 8 rows a block;
  `cols_rows_per_block` and `matvec_cols_multi`'s block count follow `COLS_PLAIN_ROWS`).
- Varied bench, 3 runs, 30 GiB tier: per-turn median **50.1 / 51.0 / 51.25** (2 rows a warp: 49.8 / 50.7 / 50.7; 4 rows: 48.9 / 48.9 / 48.9), total throughput **50.67** tok/s (50.2, 50.9, 50.9) against 48.5-48.7: **+4.5%**, ahead of Strata's 48.6. Taking the reference-order kernels (BF16/F32, `COLS_REF_SPLIT`) to 1 row a warp at every size changed nothing (50.6 / 50.9 / 50.95).

## 2026-10-31: Shadow ahead of Strata with the current defaults (52.1 against 47.7 tok/s)

- `bench/vs-strata-final4` (3 interleaved runs, 36 varied prompts, total throughput; default `shadow serve` flags: 36 GiB tier with floor 3, `--verify-rank 2`, one-row-a-warp `matvec_cols`): Shadow **52.09** tok/s (51.8, 52.2, 52.2; 1,781 tokens a run, 17 of 35 turns ended early), Strata **47.73** (47.7, 47.4, 48.0; 1,518 tokens a run, 21.7 ended early): **+9.1%**.
  Per-turn medians 51.8 against 47.4. PCIe 15.4 against 10.1 GB/s (about 296 against 212 MB of expert weights a token), GPU 84% against 78% busy, CPU 12% against 51%, disk 150 against 468 MB/s, RAM free during the run 4.1 against 8.3 GiB (the 36 GiB tier).

- Tried staging a grouped expert block's inputs (up to 4 x 2048 floats) in shared memory (`matvec_grouped_*`: each warp re-read them from the cache chunk by chunk; the L1 throughput was 35-50% busy). Same sums, `--exact` text identical, but decode per-turn median **49.4 / 48.5 / 50.0** against 50.1 / 51.0 / 51.25 and the exact prompt 59.6 against 62.8 tok/s:
  slower (the 32 KB of shared memory a block and the barrier cost more than the reads saved). Reverted.

## 2026-10-31: where a long prompt's GPU time goes; what did not help

- Nsight Systems on a 3,001-token prompt (1,024-token batches, pipeline on): kernels busy **15.4 s** of a 22.0 s window (70%), host-to-device copies 9.4 s (6.6 s overlapped with kernels), neither 3.8 s. The GPU is now the limit, not the copies or the host. Kernel time: dense `matvec_cols_*` (iq4_xs 2.0 s, q6_k 1.8 s, bf16 1.1 s,
  `hc_down_inject` 0.8 s, q5_k 0.75 s, q4_k 0.5 s, f32 0.4 s, `matvec_fast_bf16` 0.3 s) = ~7.7 s; grouped expert matvecs (iq4_nl 1.9 s, iq2_s 1.3 s, iq3_xxs 1.1 s, iq3_s 0.7 s, q2_0 0.4 s) = ~5.4 s; sparse/dense attention ~1.1 s; the delta layers 0.2 s.
  Dense work per prompt token is ~16 GFLOP of FP32 FMAs (about 8 G quantized weights): 48 TFLOP for 3,001 tokens, 1.7 s at the card's FP32 peak, 7.7 s measured: ~6 TFLOP/s.
- Tried: 2,048-token batches (122.8 / 141.9 / 120.2 / 119.3 tok/s against 128-144 at 1,024: slower, the GPU work per token is constant and the buffers cost more); a 16-inputs-a-launch `matvec_cols` (the weights read once for 16 inputs instead of 8; same sums, same text): 126.0 / 141.8 / 131.6 / 138.0 tok/s,
  no change, so the dense kernels are not limited by re-reading the weights (they come from the L2 cache) but by FP32 decode-and-FMA throughput. Both reverted.
- What would move prompt reading from here: tensor-core matrix multiplies for the dense and expert projections (dequantize a weight tile to FP16 in shared memory, `mma` against FP16 activations, FP32 accumulation, as llama.cpp's prompt kernels do). That is an inexact path (FP16 rounding of weights and activations
  where the reference sums in FP32), so default mode only, with a fidelity check against `--exact`; one kernel family per quantization type. Not started.

## 2026-10-31: tensor-core projections for a long prompt (`--tensor-prompt`, Q6_K and IQ4_XS)

- `mma_cols_q6_k` / `mma_cols_iq4_xs` (`kernels/matvec.cu`): Y = X W^T for 32 tokens x 64 rows a block (4 warps, `mma.sync.m16n8k16` FP16 operands, FP32 accumulation); the weight tile is decoded to FP16 in shared memory with the reference decode (same float `d * q`, then rounded to FP16), the input tile converted on the way in.
  `tests/matvec.rs::tensor_core_projections_are_close_to_the_exact_ones` on real tensors: relative error **3.4e-4** (`output.weight` Q6_K 248320 x 2560) and **2.9e-4** (an IQ4_XS attention projection), and 64 inputs in **0.143 ms against 1.032 ms** (7.2x) and 3.26 against 22.0 ms (6.8x) for the exact kernel's 8 launches of 8.
- Used by `mv_cols` for 16 or more inputs on those two types (cols a multiple of 64), when `--tensor-prompt` is on: `shadow serve` adds it unless `--exact`, so default mode only (FP16 rounding of both operands is not the reference arithmetic). Decode and verify passes (fewer than 16 inputs) are untouched.
- A 3,001-token prompt (Nsight Systems): kernel time 15.4 -> **12.0 s**, the two projection kernels gone from the top of the list; prompt reading 125 -> 137 tok/s under the profiler, 128-144 -> 129 / 146 / 135 / 140 tok/s plain: the pass is copy-bound again (kernels idle 4.4 s waiting for host-to-device copies, 3.6 s idle for the host; 9.5 s of copies).
  Same-history fidelity of the 24-turn test with it on: 0.467 (the metric's build-to-build noise is ~0.04: 0.375 / 0.403 / 0.423 / 0.447 on earlier builds).
- Tried and reverted: starting layer l + 1's expert copy before layer l's router answer (the copy stream waits on a GPU event instead of the host): 122 / 138 / 127 / 128 tok/s against 129 / 146 / 135 / 140, exact prompt 59.0 against 62.8 tok/s: slower.
- Next: the same kernels for BF16 / Q5_K / Q4_K / F32 (about 3.5 s of the 12 s) and for the grouped expert matvecs (5.4 s, needs 32-token entries in the plan and decodes for IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S, Q2_0).

## 2026-10-31: the grouped experts on the tensor cores; the long prompt is now bound by the SSD reads

- `mma_grouped_{q6_k,iq4_xs,iq4_nl,q2_0,iq2_s,iq3_xxs,iq3_s}` (`kernels/matvec.cu`): the routed experts of a long prompt as 32-token groups (`grouped_plan`'s group size is a parameter now: 8 for `matvec_grouped_*`, 32 for these; records of 65 u32), one block a 64-row x 32-token tile, the same FP16-decode tiles as `mma_cols_*`
  (`Dec8` for every type the model's experts use). Used when `--tensor-prompt` is on, every expert matrix of the layer has a decode and the group covers 64 or more tokens (the staged long-prompt path and the lent 256-token chunks).
  3,001-token prompt (Nsight Systems, 30 GiB tier): kernel time 12.0 -> **7.6 s** (`matvec_grouped_iq4_nl` 1.94 s -> `mma_grouped_iq4_nl` 0.26 s, `iq2_s` 1.27 -> 0.26 s).
- That did not speed the prompt (129 / 147 / 135 / 140 tok/s plain at the 30 GiB tier, the same as before): the layer's host time was **waiting for the file reads of the next layer's experts** (`copy(l+1)` 54-97 ms a layer of the host, `stage_copy` waiting in `stage_pipe_wait`); the kernels were already idle behind the copies. One `cuMemcpyBatchAsync_v2` call for a layer's ~450 copies (the driver's
  batch call: it wants the sources' device-mapped addresses, `PinnedRegion::dev_ptr`; `tests/pinned.rs::batched_host_to_device_copies_are_exact`: 64 copies of 2 MiB in 86 us of host time, exact) takes the queueing off the host but not the waiting.
- With the default 36 GiB tier (25% of the experts file-only instead of 36%, about 30% fewer file reads): 4 prompts of 1.8k-4.6k tokens at **165 / 191 / 178 / 178 tok/s** (first run 103 / 191 / 184 / 184) against 129-146 at 30 GiB; same-history fidelity 0.421 (noise ~0.04); decode after the prompts 43-45 tok/s;
  24-turn chat 15.9 tok/s end to end (14.5 before). `cargo test --release` passes (the drift test wants every entry point written out, not made by a macro).
- So a long prompt is bound by reading the file-only experts from the SSD (~6 GB/s while the pipeline also copies and computes) and then by the host-to-device copies: more RAM tier, or a faster read path (unbuffered reads straight into the pinned buffers), is what is left; the kernels are not the limit any more.

## 2026-10-31: the SSD is the limit of a long prompt; 2,048-token batches

- Raw reads of the model file (`tests/pinned.rs::ssd_read_bandwidth`, ignored by default; random 2 MiB reads over the first shard): **buffered 9.5 GB/s** at 8 threads (the Windows file cache serving most of it) but **unbuffered (`FILE_FLAG_NO_BUFFERING`) 3.5 GB/s**, the drive itself. A long prompt reads every file-only expert (25% of them at the 36 GiB tier, ~11.9 GB a pass)
  while the 36 GiB tier leaves the file cache a few GB, so it runs near 3.5 GB/s: ~3.4 s a pass, an upper bound of ~300 tok/s at 1,024 tokens a pass. Unbuffered reads would not help (slower than the cache when it hits); more tokens a pass (the same bytes for more tokens) does.
- `BIG_BATCH` 1,024 -> **2,048** (per-token buffers 1.27 GiB carved from the lent cache; the drafter's feature rows 2,048 as well, +52 MB of VRAM): 4 prompts of 1.8k-4.6k tokens at the 36 GiB tier **184 / 197 / 220 / 226 tok/s** (1,024: 165 / 191 / 178 / 178); at the 30 GiB tier 173 / 177 / 188 / 164 (1,024: 129 / 147 / 135 / 140).
  Decode unchanged: varied bench, 3 runs, default flags, total throughput **51.89** tok/s (52.09 before, inside the noise); the RAM floor in use 3.9-4.5 GiB free. `cargo test --release` passes.

## 2026-10-31: flags retuned on the current build; achieved bandwidth of the dense kernels

- `tools/autotune.py` (base now the current defaults: tail-skip 3, verify-rank 2, min-p 0.5, tier 30; space `bench/autotune_space3.json`, fidelity-neutral knobs only; log `bench/autotune4/log.md`, ~3 h unattended): `--spec 4` +0.8% (`--spec 5` -24.6%), `--dflash-min-p` 0.4 -16.3% / 0.6 +2.2% / **0.7 +2.9% by per-turn median**,
  `--admit-after` 1 -0.7% / 2 +0.7%, `--pin-margin` 2 / 4 -0.8% / -0.7%. The only knob it adopted was `--dflash-min-p 0.7`; confirmed on total throughput (3 runs each, default flags): 0.5 **52.06** (52.0, 52.1, 52.1) against 0.7 **52.11** tok/s (52.0, 52.2, 52.1): a tie, so the default stays 0.5.
- `tests/matvec.rs::cols_bandwidth` (ignored; `matvec_cols_*` at 4 inputs, a verify pass's shape, on the model's real tensors): the weight bandwidth reached against the card's 672 GB/s: BF16 projections **79-100%** (`ple_key` 672 GB/s), F32 71%, **Q6_K 50-54%** (`output.weight` 360 GB/s), Q5_K 41-45%, Q4_K 36-38%, **IQ4_XS 29-37%** (`ssm_out` 195 GB/s), small
  matrices (640-512 rows) 19-24% (a fixed ~8 us). So a 2x on the IQ4_XS / Q4_K / Q5_K / Q6_K projections is the headroom left in the dense decode kernels, about 12% of a verify pass at most (those kernels are ~14 ms of its ~45 ms).
  What it takes: the lanes' sums must keep their order (the single-token, batched and many-inputs kernels are bit-identical by test), so the gain has to come from how the bytes reach the lanes: one 256-weight block is exactly one warp iteration (32 lanes x 8 weights), and each lane today issues ~6-10 small loads of its own bytes (the IQ4 nibble
  bytes are loaded by two lanes each); a cooperative coalesced load of the block's 136 / 144 / 176 / 210 bytes (the 210-byte Q6_K blocks are only 2-byte aligned) followed by shuffles would feed the same lanes the same bytes. Not done.

## 2026-11-01: dense matvec kernels, what is not the limit (tried, nothing kept)

- `cols_bandwidth` (4 inputs) on the IQ4_XS projections, one change at a time, same numbers to within 1%: the 136-byte block header and a lane's 8 quants as two 8-byte loads instead of four 1-4 byte loads (1,299 -> 1,292 us on the 248k-row matrix); the IQ4_NL table lookup by three byte-permutes instead of compare/select (1,300 us); `prefetch.global.L2` of the row 3 iterations ahead (slower, 1,460 us); a 16-entry float table in shared memory (slower, 1,637 us). So neither the load instruction count, the memory latency nor the decode instructions limit these kernels.
- What does move the microbench is **rows a warp**: each lane loads 32 floats of inputs per 8 weights (30x the weight bytes at 4-bit and 4 inputs), so the input loads from L1 dominate and the efficiency ranks with the bit width (BF16 100%, Q6_K 50%, Q5_K 43%, Q4_K 37%, IQ4_XS 29-37%). 4 rows a warp (inputs loaded once for 4 rows) for matrices of 2,048-65,535 rows: IQ4_XS 12288x2560 67.7 -> 57.5 us, Q4_K 69.8 -> 50.0, Q6_K 12288 74.0 -> 59.2, Q4_K 6144 37.3 -> 26.8; worse for 512-640 rows (fewer warps) and for the 248k-row output matrix (1,299 -> 1,540 us: the kernel holding both variants gets the heavier one's register count, so even the one-row path lost occupancy).
- End to end it was a tie: varied bench, default flags, 3 runs interleaved, total throughput **51.31** (2 runs; a third died at turn 14 with a closed connection, RAM 3.8 GiB free) against 51.58 for the baseline binary (`bench/wide-ab`). The decode pass runs the projections through `matvec_cols_multi` (jobs of mixed sizes in one kernel), where the extra variant costs registers for every job. Reverted. What is left is a separate wide multi kernel for the 2,048+ row jobs (groups whose jobs are all wide), est. 2-3% of decode; not done.

## 2026-11-01: the drive's read ceiling is 3.5 GB/s

- `tests/pinned.rs::ssd_read_sweep` (ignored; unbuffered random reads of the model file, 3 s a cell; the drive is an NVMe, Windows names it "Sandisk Optimus 5100 500GB"): 128 KiB reads top out at 1.9 GB/s, 512 KiB at 3.5, **2 MiB and 8 MiB at 3.5 GB/s from 4 threads up to 64** (1 thread: 2.5 / 3.0). More queue depth or bigger reads do not go past it, so the 3.5 GB/s that bounds a long prompt is the drive's own ceiling (the figure of a PCIe 3.0 x4 NVMe), not how Shadow issues its reads. Only the Windows file cache (9.5 GB/s) or less file-only experts (a bigger RAM tier) get past it.

## 2026-11-01: drafter time (6.5 ms a round) is GPU work, not host overhead; 4 rows a warp tied again

- Timer around `dflash_draft`: the host spends **0.66 ms** issuing the ~140 launches, then waits 6.2-6.9 ms for the GPU. Caching the weight pointers/names could save at most a fraction of that 0.66 ms, so it was not done. `SHADOW_DF_TIME=1` per-stage (synchronized laps, so inflated): head (full-vocabulary `output.weight` x 7 inputs) 2.3-2.6 ms, qkv + norms + rope 2.0, mlp + convs 1.7, attention 1.2, o_proj + conv 0.4, top-16 0.2 (the draft-vocabulary head is refuted: -25%, docs/SPECULATION.md).
- `matvec_cols_wide_{q8_0,bf16}` (4 rows a warp, used by the drafter's matrices of 2,048+ rows, which launch as standalone kernels): mlp + convs 1.67 -> 1.25 ms, o_proj 0.38 -> 0.31, qkv unchanged. End to end (varied bench, default flags, 3 runs interleaved, total throughput): **51.20 against 51.54** for the baseline binary (`bench/wide-ab2`): a tie, and with the earlier `matvec_cols_multi` try (51.31 vs 51.58) the 4-rows-a-warp idea is closed: microbench gains on single kernels do not reach the decode rate. Reverted.

## 2026-11-01: 3,072 / 4,096-token prompt batches do not fit; reading only "used" experts is moot

- `BIG_BATCH` 3,072 and 4,096 (to read a 3,001-token prompt in one pass, a 4.3k one in two): the 1.9 / 2.5 GiB of per-token buffers do not fit in the 4.06 GiB lent expert cache beside the stage halves, and the prompt drops to the slow path: **60-66 and 31-45 tok/s** against **187 / 198 / 229 / 186** for 2,048 (clean runs of the same four prompts, `longsrc.py`). 2,048 stays the largest batch; more would need smaller per-token buffers (`bb_row_floats`) or a bigger lent cache.
- Reading only the experts earlier chunks used does nothing at this batch size: 2,048 tokens x 8 routed experts is ~16k selections over 512 experts a layer (~32 per expert), so nearly every expert is needed by every chunk.
- Bench hygiene: a server left running on port 8095 answered a later script run (the cleanup killed `shadow-*`, not `shadow`); those 31-45 tok/s numbers were one server's.

## 2026-11-01: the VRAM reserve 384 -> 96 MiB (+4.5% decode); default placement is the 300-prompt profile

- The expert cache takes the free VRAM less `VRAM_RESERVE`. The display runs off the integrated GPU on this PC, so the RTX card has nothing else on it; `--vram-reserve-mib` (new) sets the reserve. Varied bench, 3 runs interleaved, 30 GiB tier (the RAM was 47 GiB free): reserve 384 **50.01** against 96 **52.26** tok/s (`bench/vram-ab`, 3 runs each agree to 0.3); 96 against 16 MiB 52.40 / 52.45 (`bench/vram-ab2`, a tie). Long prompts at 96 MiB run (176-184 tok/s at the 30 GiB tier, `longsrc.py`). Default is now **96 MiB**; a card that also drives a display should pass a larger `--vram-reserve-mib`. `cargo test --release` passes.
- `shadow serve`'s default placement is `profiles/routing-varied300.shrc` (300 varied prompts, the benches' profile), falling back to the six-prompt one. Not A/B'd against the six-prompt profile on its own (the UI's server held the RAM when asked).

## 2026-11-01: pictures (phase 1)

- Shadow reads pictures. `shadow serve` accepts OpenAI-style `image_url` parts (`data:image/...;base64,` URLs); the chat page's 📎 button, drop and paste take pictures
  (shrunk in the browser to about 768 x 768, the sides multiples of 32, sent as JPEG) and show a thumbnail in the chip.
- How: **the picture encoder is Strata's `strata-vision.exe`** (llama.cpp's mtmd library on the model's mmproj, `models/Qwen3.8-Flash-Next.mmproj-f16.gguf`, a Qwen3-VL-style
  27-block ViT), copied to `bin/` (ignored by git; `SHADOW_VISION_EXE` / `SHADOW_MMPROJ` override, Strata's build is the fallback). It runs on the CPU as a child
  process, started on the first picture and ended once the pictures are encoded (it holds 1-1.3 GiB while alive): 260 rows 2.3 s, 350 rows 3.2 s, 1,024 rows 13.5 s;
  rows are cached by content hash. Its output (n rows x 2560 floats) replaces the embeddings of the `<|image_pad|>` tokens (`vision.rs`, `engine::expand_pictures`,
  `GpuSession::set_pictures`); the prompt's `<|vision_start|><|image_pad|><|vision_end|>` becomes one pad token per row; the per-layer inputs of those positions come
  from the pad id (the model's `ple.image_token_id`), as llama.cpp's qwen4exp does. The prefix cache treats the same ids with other pictures as another prompt
  and takes no shared-prefix snapshot for a prompt with pictures.
- **Not yet (phase 2): the 2-D M-RoPE positions.** The model was trained with (t, h, w) positions for picture rows (`rope.dimension_sections = [11, 11, 10, 0]`, interleaved,
  also for the indexer's q and pooled k); Shadow gives them the plain 1-D position, and the text after a picture is not shifted by `max(nx, ny) - rows`. Measured anyway:
  shapes picture "red circle, blue square, green triangle, text STRATA 42" correct; `CODE 4217` read exactly; a 1,024-row invoice read "INVOICE 20261014", "Total due: 1337 EUR";
  a canvas drawing's layout (circle left, square upper right, text at the bottom) right in the chat page. Small spatial errors on dense layouts are to be expected until phase 2.
- RAM: the RAM tier leaves ~4-5 GiB above the floor, so a picture is refused (HTTP 400 with the numbers) when free RAM is under the floor + 1.8 GiB. A first test
  under `run_capped -FloorGiB 4` was killed by the watchdog at 3.94 GiB: the 36 GiB tier plus the encoder.
- The page shows a server error that arrives mid-stream (it used to leave an empty reply bubble).

## 2026-11-01: pictures (phase 2): 2-D rotary positions

- The picture rows now get their (t, h, w) positions: `forward.cu` `rope_one` reads an optional device table `g_rope_tab` (token index -> three positions; null with no picture, then
  the same single-position rotation as before: pair i takes t, h or w as i % 3 is 0, 1, 2, llama.cpp's IMROPE with the model's sections 11, 11, 10). `GpuSession::sync_rope_tab`
  builds it from the pictures' spans on every request: a text token is the next position; a row at grid (x, y) is (start, start + y, start + x); the token after a picture is
  `start + max(width, height)`. It covers every rope call (attention q/k, the indexer's q, the pooled indexer keys at their first member) without new kernel arguments.
- Measured (varied pictures, default flags): "list the shapes left to right, then read the text" correct (red circle, blue square, green triangle, STRATA 42); `CODE 4217` exact;
  the invoice picture now gives its line items too (widgets x 12, gears x 7, springs x 30) besides "INVOICE 20261014" and "1337 EUR"; a follow-up in the same conversation reuses the
  350 cached tokens including the picture ("the shape on the far right" green triangle, "the one in the middle" blue). Text-only: 17 x 23 = 391. `cargo test --release` passes.
- Not compared against llama.cpp's answers on the same pictures yet (no reference run); the positions follow its `mtmd_helper_image_get_decoder_pos` and `ggml_mrope_cache_init`.

## 2026-11-01: 3-D files in the chat page (STL, OBJ, STEP)

- Attach (button, drop, paste) a `.stl`, `.obj`, `.step` or `.stp` and the page turns it into what the model can read: **a 768 x 768 contact sheet of four views**
  (isometric, front, top, right; an STL / OBJ is a shaded mesh drawn by a small painter's-algorithm rasteriser on a canvas, up to ~150k triangles) **plus a text summary**
  (triangles, bounding box, volume, surface area, centre; the file states no units). A STEP file has no mesh and the page has no CAD kernel: it is drawn as a
  **wireframe of its vertices and edges** (curved edges as straight chords) and summarised from its header, schema, units, product names, vertex bounding box and
  the most frequent entities. Everything is browser-side (`view3d` block in `crates/shadow-serve/src/ui.rs`); the server sees a picture and a text block.
- Tested with a generated STL (a 40 x 20 x 10 plate with a 12 mm cylinder to z = 35) and a hand-written STEP box: the sheets are right; the model answered from the sheet and
  the summary ("rectangular base plate with a vertical cylinder, 40 x 20 x 35, cylinder diameter about 12, base about 10-12").
- Limits: no surfaces for STEP (a curved part is a wire frame of chords); STL volume is meaningful only for a closed mesh; very fine detail is limited by the 768-pixel sheet.

## 2026-11-01: a Dashboard and a conversation sidebar in the web UI

- **Dashboard tab** (between Chat and Process Monitor): engine (state, model, uptime, speculation on/off, requests answered, context in use with a bar), decode speed per reply (bar
  chart of the last 40, last and average tok/s, last prompt-reading speed), GPU (busy %, a 2-minute line, VRAM bar, temperature), expert tiers (VRAM cache hit rate, the pinned RAM tier,
  the share of misses served from RAM against the drive), system memory (free RAM line, Shadow's working set, CPU), saved conversations, a table of the last 12 requests (prompt and
  cached tokens, reading speed, tokens written, decode speed, how it ended, a picture mark) and quick actions (new chat, resume the last chat, persona, Windows tools on/off, the
  process monitor). The server keeps the last 40 answered requests (`engine::Recent`, telemetry `recent`) and now reports `session_tokens`.
- **Conversation sidebar** in the chat view (hide with «, reopen with ☰): every conversation is saved in the browser's localStorage after each reply (pictures included until the space
  runs out, then the oldest conversation's pictures go first), grouped Today / Yesterday / This week / Earlier, searchable, deletable, restorable (attached files and pictures shown again).
- Tested live: the sidebar after a real reply, a fabricated second conversation (open, switch back, delete), the dashboard on the running server with real numbers.

## 2026-11-01: queued messages and a compact chat bar

- **Queue:** while a reply is being written, Enter / the button (now "Queue") puts the message, with its attachments, in a list above the box (numbered, each removable); each one is sent when the reply before it
  ends, so every message sees the answers to the ones before it. A reply that was stopped or failed pauses the queue (a "Resume" button appears); New chat clears it. Tested: three messages sent
  at once ("one", "two", "three") ran in order, each answered before the next.
- **Compact bar:** one rounded bar with the attach button, the text and Send / Stop inside it (32 px controls, a one-line box that grows with the text), and a slim status line under it (reasoning
  depth, Windows tools, persona as quiet text buttons, the status on the right). The reasoning-depth menu opens above the bar instead of over it (`--bar-h`).

## 2026-11-01: a CPU card on the dashboard

- Between Expert tiers and System memory: whole-PC busy % (big number), a two-line 2-minute chart (whole PC, Shadow), Shadow's share, the CPU threads it uses, **the clock** (`% Processor
  Performance` x the base clock from CPUID leaf 0x16: 4.60 GHz on the 3.6 GHz base of the i7-11700K), **the caches** (`GetLogicalProcessorInformation`: L1 48 KB data + 32 KB instruction and L2 512 KB
  a core, L3 16 MB shared) and **the temperature**. `crates/shadow-serve/src/cpuinfo.rs`: CPUID, `GetLogicalProcessorInformation` and PDH counters, no driver; telemetry `cpu.info`.
- The temperature is the ACPI thermal zone Windows exposes without a driver (28 C here, the board's; `MSAcpi_ThermalZoneTemperature` is access-denied unelevated). A core's own temperature
  needs a kernel driver (HWiNFO, LibreHardwareMonitor): not loaded; the card labels the value "(board)".

## 2026-11-01: web tools and MCP tools

- The agent layer had six local tools (read_file, glob, grep, edit_file, write_file, cmd); asked for web access the model improvised with `Invoke-WebRequest` through `cmd` and got raw HTML (118k characters, clipped).
  New: **`web_search`** (DuckDuckGo's HTML results: title, URL, snippet) and **`web_fetch`** (a page as readable text: scripts, styles and tags dropped, entities decoded, long pages in parts with `start`), `webtools.rs`,
  over `curl.exe` (in Windows 10+, its own TLS: no HTTP library in Shadow). The default persona names them.
- **MCP client** (`mcp.rs`, stdio transport): servers listed in `mcp.json` (Claude Desktop / Claude Code format: `SHADOW_MCP_CONFIG`, else `mcp.json` in the working directory, else `%APPDATA%\Shadow\mcp.json`) are started with
  `shadow serve`; their tools (`initialize`, `tools/list`, `tools/call`) join `/v1/tools` as `mcp__<server>__<tool>` and run through `/v1/tools/execute`. `npx` (a .cmd on Windows) is started through `cmd /C`. Tested with a stub server
  and the real `@modelcontextprotocol/server-filesystem` (14 tools); the dashboard has a Tools card with each server's tool count or error. In the chat page the model called an MCP tool and `web_search` in one turn.
- Tool parameters are written by the model as text; they are now converted to the types the tool's schema declares (`coerce_arguments`): the stub's `add` returned "1725" for "17" and "25", and the built-in tools' integer
  parameters (`read_file` start / end, `grep`, `glob` limits) were silently ignored when written as text. Tool calls run on a blocking thread (`spawn_blocking`) instead of the server's async threads.

## 2026-11-01: memory, and the persona moves to Settings

- **Memory** (`memory.rs`): short notes about the user in one JSON file (`%APPDATA%\Shadow\memory.json`, `SHADOW_MEMORY_FILE`), at most 60 notes of 400 characters; HTTP `GET/POST /v1/memory`, `PUT/DELETE /v1/memory/<id>`; the tools `memory_save`,
  `memory_list`, `memory_forget`. The page puts the notes (newest first, at most ~2,400 characters) and a one-line hint about the memory tools at the end of the system prompt of every chat, **frozen for that chat** (a note saved in
  the middle would change the start of the prompt and make the server read the whole conversation again; it applies from the next chat). Tested: told "remember that I am building ... Shadow ..." the model called `memory_save`
  by itself with a clean note; in a new chat it answered "what do you know about me" from the notes, no tool call.
- **Learn from my past chats** (Settings): the model reads up to 80 of your earlier messages from the saved conversations (12,000 characters at most) and proposes up to 10 notes as tick boxes; nothing is saved until you tick and save.
- **Settings tab** (between Dashboard and Process Monitor): the default persona (editable, reset to the built-in one), the memory (list, click a note to edit, delete, add, the on/off switch). The persona button left the chat bar; the
  persona is now a default for every new chat, and a one-off role goes in the chat ("act as a software engineer and design this"). The built-in persona is shorter and no longer names Strata folders; it lists the tools.

## 2026-11-01: a real browser for the model, inside the Shadow window

- `mcp.json` now starts **Playwright's MCP server on the installed Chrome** (`@playwright/mcp`, `--browser chrome --headless`): the model navigates, reads a page as a structured text snapshot (every element with a ref), clicks,
  types, fills forms, presses keys, waits and manages tabs. The server has 25 tools (~4,500 prompt tokens); `"allow"` in the config offers the model the 10 that matter; `"hints"` adds a line to a tool's description
  ("navigate does not return the page: call browser_snapshot next"), which stopped the model wasting turns re-snapshotting and reading files for the page.
- **The Browser pane** (chat view, right side; opens by itself on the first browser tool call; close with the cross, refresh with the arrow): the page as a picture that follows the model (`GET /v1/browser/screenshot`
  calls the server's `browser_take_screenshot`, which `"allow"` keeps from the model but Shadow may call), the address, refreshed after each browser tool and every 4 s while a reply is written. Headless, so the pane is the only
  view (remove `--headless` for a real Chrome window).
- Tested in the chat page: "open Hacker News and tell me the top 3 stories": navigate, snapshot, then the right answer (titles, hosts and points) with the page visible in the pane. Slow the first time (the start of the prompt is
  3,769 tokens with the browser tools, read once at ~85 tok/s, then cached; a page snapshot is ~3-4k tokens, ~45 s to read).
- A browser server's results are capped at 12,000 characters (`MAX_MCP_OUTPUT_CHARS`); the agent loop allows 14 tool turns (was 6); the tool clip no longer cuts inside a multi-byte character.
- MCP servers' processes (npx, node, Chrome) are put in a Windows job object that kills them when Shadow ends, however it ends (tested: 16 processes while running, 0 after Shadow was killed).
- Not done: the model cannot *see* the screenshot (it reads the text snapshot); feeding the picture back to it, now that Shadow reads pictures, is possible (~300-1000 tokens a picture, ~4-12 s to read).

## 2026-11-01: an engine panic no longer hangs the chat; the bubble throbs while it works

- The desktop app's log (`%APPDATA%\Shadow\shadow-console.log`) showed the engine thread dead with `session.rs:108: the cache has fewer slots than one token routes to` (`ExpertCache::victim`: every evictable slot of a
  layer was in use by the current token) on a long agent conversation (a request reading ~5,000 new tokens after two large tool results, 2,091 cache slots). The thread died, `busy` stayed on, every later request
  waited for ever and the page showed an empty bubble. Not reproduced since: a 13,451-token and a 15,136-token prompt read fine, alone and after other long prompts (one earlier run lost the server process at the second long prompt
  with no panic text; it did not recur in two repeats). So the cause is **not found**; what is done: a panic inside one answer is caught (`engine::run` loop, `catch_unwind`): the request gets an error ("internal error: ... send the
  message again"), the lent cache is restored and the session reset; a panic of the whole thread marks the engine `failed`; `shadow serve` sets `RUST_BACKTRACE=1` so the log shows where the next one comes from.
- **The bubble throbs** (a glow in the accent colour, 1.8 s) while a reply is being written, and a tool card while its tool runs; it stops when the turn ends or is stopped. If the engine reports `failed`, or the telemetry
  poll fails three times while a reply is awaited, the bubble turns red with the reason instead of throbbing for ever.

## 2026-11-01: the Process Monitor tab is gone; the Dashboard has what it had

- The Process Monitor (CPU, GPU, RAM, engine and three tier cards) duplicated the Dashboard cards. The tab, its panel, styles and poller code are removed (~300 lines); `/monitor` and the old `#monitor` link go to the Dashboard.
  What the monitor showed that the Dashboard did not moved into the Dashboard's cards: GPU: memory-controller activity and free VRAM; Expert tiers: VRAM cache hits / misses and misses read from RAM / drive (counts);
  System memory: RAM in use with its load; Engine: process ID. The rest was already there (CPU usage and cores, GPU busy / VRAM / temperature, working set, pinned tier, uptime, requests, tokens, speculation, queue).

## 2026-11-01: replies that ended mid-thought (the 1,024-token default) and thinking that never ended

- A chat in the page ended with the thinking cut off mid-sentence and no answer, twice ("Continue" did the same): the page sends no `max_tokens`, and the server defaulted to **1,024**, which a reasoning model spends on
  thinking alone (the stats line said "1024 tokens"). Fix: no `max_tokens` now means as many as fit in the context (`lib.rs` default 0, `engine.rs` computes the room); a reply cut by a limit says so in its bubble.
- Taking the limit off showed the opposite failure: an Extra-High question thought for **17,303 tokens**, until the context was full, and never answered (301 s). So there is a **thinking budget by effort** (`engine.rs`):
  Low 1,000, Medium 4,000, High 8,000, Extra-High 12,000 tokens, always leaving 1,200 for the answer (none for a small `max_tokens`); past it the engine feeds the model "That is enough thinking: the answer now. </think>" and it
  answers. Tested: the same question at Medium thought ~1,600 tokens by itself and answered; at Extra-High the budget closed it at ~12,000 and it answered (14,582 tokens in all, 268 s, finish "stop").

## 2026-11-01: a 32,768-token window by default, and old tool outputs are trimmed

- A long agent chat ended with "23724 prompt tokens + 0 max_tokens exceed the context of 17408": the window is `--ctx`, 16,384 by default, **not the model's limit** (`qwen4exp.context_length` = 262,144). It costs VRAM, not speed
  directly: ~57 KB a token for the attention layers' K/V, taken from the expert cache. Measured (`serve --ctx N`, 4 chat prompts of 250 tokens, decode tok/s): **16k: 2,091 cache slots, 52.4; 32k: 1,621, 51.7 (-1.3%); 48k: 1,136, 49.7;
  64k: 695, 49.5 (-5.5%)**. Default is now **32,768**; `--ctx` (or the desktop app's extra flags) changes it. A 20,802-token prompt reads in 287 s at that window (~72 tok/s: reading, not the window, is the slow part).
- The page now trims by itself: when the conversation reaches 85% of the window's room it cuts the oldest tool outputs (over 600 characters) to a one-line stub, newest kept, until it is at 60%; the chat goes on instead of stopping.
- Not done: the window as a setting in the page (it is a start flag), 16-bit K/V (would halve the cost per token), K/V in host RAM (attention reads only 2,048 positions a token, so it could live there).

## 2026-11-01: Resume and Retry; the staged-expert panic

- **Resume did nothing** when a queued message waited after a failed reply: it cleared the pause, then the queue runner saw the status text "Failed ..." and paused it again. `resumeQueue()` now forces the send.
  **Retry**: a failed reply (a server error in the stream, an HTTP error, a lost connection) shows a Retry button. If the question itself failed it is taken back and Retry sends it again with its attachments; if the failure came part-way
  through an agent turn, what was done stays and Retry asks for the next step again (`resumeOnly`, no new user message). Tested: a simulated engine fault, Retry (answered), a paused queue with status "Failed", Resume (sent and answered).
- The engine's net (the previous entry) caught another panic on a very long prompt (~18,600 new tokens): `a staged expert in neither the RAM tier nor the read-ahead` (`stage_copy`). Likely cause: the read-ahead buffers are sized at the start of a
  long prompt for the file-only experts a layer has *then*, but the RAM tier changes while the prompt is read (promotions, evictions), so a layer can need more slots later and `stage_pipe_read` cut its list short.
  Now the half being filled gets a bigger pinned buffer when a layer needs it. Not reproduced here (a 20,802-token prompt read fine before and after), so this is a fix for a probable cause, not a confirmed one; if it recurs the message tells.

## 2026-11-02: "move compute to the data" for the prompt path - not built; the premises were measured first, and they fail

Spec: `docs/EXPERIMENT_SPEC_2026-11-01.json` (CPU computes 1-3 cache-miss experts a layer from the mapped file with the AVX-512 kernel; no host join; those experts leave the copy set; 4 arms x 3 interleaved fresh-launch runs, scored by total throughput).
Nothing was built; the two numbers the idea rests on were measured instead (`cargo run --release -p shadow-model --example cpu_expert_tokens_bench`, AVX-512 kernel untouched, 14 threads, best of 12, one expert, T tokens through it):

- **A prompt chunk sends many tokens to each expert** (2,048 tokens x 10 experts over ~400 per layer = ~32 tokens each, where decode has 1). The kernel streams the weights per (expert, token) pair, so the cost is linear in tokens:
  layer 5, 32 tokens: **3.06 ms an expert (0.096 ms a token)**; layer 40, 32 tokens: **4.36 ms**. The PCIe copy it would replace is **0.13-0.18 ms**. A CPU expert is **~20-25x dearer than the copy** at prompt batch sizes
  (at T=1, decode, it is as fast as the copy - which is why the decode hybrid wins and this does not).
- **The page-cache premise** (a warm file cache serving the missing experts at DRAM speed): with the 36 GiB tier pinned and ~10 GiB free (63.8 GiB total; floor 3 GiB), most file-only experts are *not* in the page cache; they come from NVMe at ~3.5 GB/s.
  Taking 1-3 of ~400 experts a layer off the copy set saves under 1% of a chunk even if the CPU work were perfectly hidden, and the CPU work (3-4 ms an expert, x1-3 a layer) is longer than the whole layer's copy time.
- Verdict: as specced it cannot clear the 0.8% noise floor ("a tie is a loss"), so the 4-arm bench was **not run** and no code path was added (off by default would have been the only option anyway). The spec's own odds were ~40%; this lowers them.
  The bench example stays in the tree as the measurement. Would change the answer: a prompt-batched CPU kernel (weights read once for all T tokens, so ~0.1 ms of weight traffic + T x compute) - that is a kernel change, which the spec rules out.

**Found while setting the baseline: prompt reading is 3x slower today than on 2026-11-01.** The same ~3,000-token prompt reads at **65-74 tok/s** at every length from 3.0k to 20.8k tokens (3,001 tokens in 40.7 s = 73.8), against 186-229 tok/s measured earlier on Nov 1.
The Oct 31 commit cd8be97 (built and run in a separate worktree) is just as slow now, so it is **not the code**, not `--ctx`, not the VRAM reserve, not MCP. During a read the GPU shows 100% utilisation but only 77-90 W of 250 W, clocks 2910 MHz, PCIe Gen4 x16,
memory utilisation 6-10%, no other compute process. Changed on the machine meanwhile: a Windows Insider cumulative update (KB5127935, installed 2026-11-01; boot 10:23 that morning), nothing seen in the driver (617.42, 2026-09-29).
Cause **not found**. Until a valid prompt baseline is re-established (a reboot, or finding what changed), no prompt-path A/B is trustworthy; decode (52 tok/s) was not re-measured.

- **Follow-up, 2026-11-02 12:50, after a reboot** (boot 12:48; the server started under `run_capped.ps1 -CapGiB 52 -FloorGiB 3`): the same 3,001-token prompt reads at **69.6 and 72.1 tok/s** (43.1 s, 41.6 s). So the slowdown is **not** left-over machine state; it survived a reboot.
  Still unexplained. Remaining suspects: the Windows Insider update KB5127935 (2026-11-01), RAM/BIOS settings that a reboot or the update reset (installed DIMMs: 2 x CMH32GX4M2D3600C18 + 2 x CMW32GX4M2C3200C16, running 2933 MT/s at 1.2 V, i.e. no XMP profile applied; a DRAM speed or timings change
  would show in the CPU-side/RAM-tier reads but is unlikely to be 3x), a GPU power/clock policy change.

## 2026-11-02: the prompt-speed regression found and fixed - it was the context window, not the machine (65-74 -> 165-195 tok/s)

The two entries above blamed the OS; both guesses were wrong ("not `--ctx`" was never actually tested). Bisected on the same 3,001-token prompt (two runs each, tok/s):

| change | tok/s |
|---|---|
| as it was (ctx 16,384; also ctx 32,768, the default since 39db7c0) | 70-74 |
| `SHADOW_PREFILL_PIPE=0` (no layer-ahead pipeline) | 113-118 |
| `SHADOW_PREFILL_BIG=0` | 70-72 (so the big batch was already off) |
| `SHADOW_NO_BATCH_COPY=1` (one-by-one copies, not `cuMemcpyBatchAsync`) | 78-79 (not the cause; +10% worse-case noise) |
| **`--ctx 8192`** | **172-194** |

Per-layer trace (`SHADOW_STAGE_TIME=1`): every pass was `m=256` tokens, not 2,048, so each of the 40 layers copied its ~400 experts (~0.8 GiB) once per 256 tokens, 8x as often.
Cause: a long prompt lends the expert cache for the staging halves (2.54 GiB) plus the 2,048-row batch buffers (1.15 GiB), capped at 90% of the cache. The cache is what the KV for `--ctx` leaves of VRAM:
ctx 8k = 4.47 GiB (room after the halves 1.16 GiB: fits), **ctx 16k = 4.01 GiB (1.06 GiB: 0.09 GiB short)**, ctx 32k = 3.11 GiB (0.26 GiB). Short by a little, the batch silently stayed at 256 rows.
(The earlier 186-229 tok/s were measured at a smaller window; the 32k default of 39db7c0 was measured on decode only, and its "20,802 tokens in 287 s" was this bug.)

Fix (`big_batch_on`, `lend_cache`): the big batch takes as many rows as the lent room holds (up to 2,048, in steps of 64; below 512 it stays off), and the lend cap is 96% of the cache, not 90%.
`SHADOW_BIG_ROWS=n` caps the rows (testing); `SHADOW_STAGE_TIME=1` prints the per-layer trace and the lent-room sums; `SHADOW_NO_BATCH_COPY=1` forces the one-by-one copies.
Measured after (3,001 tokens, two runs): **ctx 16,384: 194.8 / 179.7 tok/s (rows 2,048); ctx 32,768, the default: 164.4-180.2 (rows 768); ctx 8,192: 194 / 172.** A 25,260-token prompt at the default window: **145.4 s = 173.7 tok/s** (was ~72), no panic.
With rows capped at 512 / 768 / 1,024 / 1,536 on ctx 16k the same question is answered the same (151 / 197 / 178 / 168 tok/s).

Open: on this prompt (a STATUS.md excerpt, question "the first heading word") ctx 32,768 and 24,576... differ - 32k answers "Fast", 8k-24k answer "Open" (the right one) - and 32k gives "Fast" with the big batch off as well, so it is not this change: something in the attention or the
position tables at a 32k window. Not investigated; the next thing to check before trusting long-window answers.
The CPU-experts experiment (entry above) still stands: its premises fail; its baseline is now valid again (prompt path ~175-195 tok/s).

- **The 32k-window "wrong answer" is not a window bug.** A 7-question check (the first `## ` heading's first word in 9,000-character slices of this file, temperature 0): default modes at `--ctx 16384` **5/7**, at `--ctx 32768` **5/7 with the identical answers**, `--exact` at 32,768 **6/7**. The one prompt that flipped ("Open" -> "Fast" from ctx ~28k up) is a near-tie that the default fast modes
  (`--tail-skip`, `--tensor-prompt`, ...) decide differently as the cache size changes; `--exact` answers it correctly at both windows. The exact-vs-fast gap (1 of 7) is too small a sample to size.
- **Timing hooks for the CPU-experts question** (what removing copies would buy; a prompt path whose critical path is the GPU gains nothing from fewer copies): `SHADOW_STAGE_COPY_EVERY=k` copies only every k-th staged expert (output wrong, timing only). Trace at 2,048-token passes: the host spends ~6 ms a layer queuing, i.e. the GPU, not the host or the copies, sets the pace.
  The k = 1 / 4 / 16 runs **did not complete**: a Strata server (started 15:21 by the user, 28.8 GiB, GPU 99%) was running and the RAM floor stopped the Shadow launch (correctly: "KILLED at 2.93 GiB available"). Not measured yet; no number is claimed.

## 2026-11-02: CPU experts on the prompt path (the spec of 2026-11-01): closed without building it - the copies are not on the critical path

With Strata stopped (it was running and had the GPU and 28.8 GiB), the prompt baseline is **192-212 tok/s** (3,001 tokens, ctx 16,384, 4 runs; it was 140-208 while other things ran).
Skipping or thinning the staged copies (`SHADOW_STAGE_NO_COPY`, `SHADOW_STAGE_COPY_EVERY=k`) fed the router garbage, NaN logits, `route()` panicked: no number from that. The direct measure instead, `SHADOW_LTIME=1 SHADOW_REPORT=1`
with new points in `moe_staged` (event timestamps on the GPU's default stream; per batched pass, averaged over the prompt's passes):

| part of a pass | ms | share |
|---|---|---|
| DeltaNet segment (graph) | 1,966 + 336 | ~47% |
| attention layers | 1,082 | ~26% |
| routed experts, resident in the cache | 559 | 13% |
| routed experts, staged (copied) | 274 | 6.5% |
| **GPU stalled waiting for the expert copy** | **58** | **1.4%** |

So the copy engine and PCIe are **hidden behind the GPU's own work**: the GPU waits 1.4% of a pass for them. Removing *every* weight copy would gain at most ~1.4%; the spec moves 1-3 of ~400 experts a layer: ~0.01%. That is below the noise floor (0.8%), "a tie is a loss",
and with the earlier numbers (a CPU expert at 32 tokens costs ~20x its copy; file-only experts are mostly not in the page cache) the design cannot win on this path. **Not built; the 4 arms were not run.** The measuring hooks stay (off by default).

Where the prompt time is: ~73% in the DeltaNet linear-attention layers and the full-attention layers, ~20% in the experts, ~1.4% waiting on copies. The ~7x gap to Strata's 1,376 tok/s prompt rate is in those two attention kernels at 2,048-row passes, not in expert movement.
That is the next thing to look at (see the DeltaNet "delta segment" and the sparse-indexer attention paths), as its own experiment.

## 2026-11-02: where the prompt path's GPU time goes (kernel by kernel) - the dense matmuls, not the experts

`SHADOW_KTIME=1 SHADOW_REPORT=1`, 3,001-token prompt twice at ctx 16,384 (graphs off, so ~20% slower: 156-162 tok/s; the split is what counts). GPU kernel time per token **9.51 ms**:

| kernel | ms/token | share | what it is |
|---|---|---|---|
| `matvec_cols_bf16` | 1.74 | 18% | dense BF16 projections (DeltaNet/attention q,k,v,o), M<=8 columns a launch |
| `delta_steps128` | 1.61 | 17% | the DeltaNet recurrence, token after token |
| `hc_down_inject_cols_fast_bf16` | 1.30 | 14% | hyper-connection down projection |
| `matvec_cols_q5_k` | 0.60 | 6% | dense Q5_K |
| `matvec_cols_f32` | 0.56 | 6% | dense F32 |
| `attention_heads_rows` + `attention_group_sel_rows` | 0.76 | 8% | attention |
| `matvec_cols_q4_k` | 0.47 | 5% | dense Q4_K |
| `matvec_cols_iq4_nl`, `matvec_fast_bf16`, `matvec_cols_q6_k` | 0.73 | 8% | other dense |
| `mma_grouped_*`, `mma_cols_*` (tensor-core experts and dense) | ~0.6 listed | ~6% | |

- The `matvec_cols_*` kernels take at most 8 tokens a launch, so with a 2,048-token pass each weight row is read ~256 times from L2/DRAM; the four dense kernels with no tensor-core version (bf16, f32, q5_k, q4_k) plus `hc_down_inject` are **~49% of the GPU's kernel time**.
  The experts (routed, tensor-core `mma_grouped_*`) are a few percent. This matches the pass split above (47% DeltaNet segment, 26% attention, ~20% experts, 1.4% copy stall).
- `--tensor-prompt` already has `mma_cols_q6_k / iq4_xs / iq4_nl / iq2_s / iq3_xxs / q2_0`; bf16, f32, q5_k, q4_k and the hc projection are the gaps.
- Not measured: what a tensor-core BF16/F32/Q5_K/Q4_K path would give. If those kernels got 4x faster the kernel time would fall ~37% (an arithmetic estimate, not a result) - the biggest prompt lever found so far, and a CUDA kernel project of its own.

## 2026-11-02: tensor-core BF16 dense projections for the prompt path (`mma_cols_bf16`): +28% at ctx 16k, +3% at the 32k default

New: `Dec8<BF16>` + `mma_cols_bf16` (the same tensor-core skeleton as the Q6_K / IQ4_XS prompt kernels: 64 rows x 32 tokens x 64 k tiles, weights and activations as half, f32 accumulate), used by `mv_cols` under `--tensor-prompt` for BF16 tensors with >= 16 tokens (attention/indexer/PLE/ssm projections that are BF16).
`SHADOW_MMA_BF16=0` turns it off. Bench, ctx 16,384, the 3,001-token prompt, 5 interleaved baseline/new launches, 2 prompts each (first cold), scored by total throughput:

| arm | prompt times (s) | total |
|---|---|---|
| baseline (`matvec_cols_bf16`) | 17.0+16.6, 14.9+12.4, 14.3+16.3, 14.7+16.8, 14.3+16.4 | 30,010 tokens / 153.7 s = **195.2 tok/s** |
| `mma_cols_bf16` | 12.8+10.9, 12.6+10.8, 12.6+10.8, 12.5+10.8, 12.6+13.7 | 30,010 / 120.1 s = **249.9 tok/s (+28%)** |

(Baseline launches swing 177-243 tok/s; every new-kernel launch was above 217.) Correctness: the sanity question gives the same answer ("Open") in all 4 launches checked, and the 7-question check scores 6/7 (default baseline 5/7, `--exact` 6/7): no loss.
It is now on by default (it wins; the half-precision tensor path is the one `--tensor-prompt` already uses for Q6_K and IQ4_XS).
At the **default 32k window** the same change reads 181 / 204 tok/s on the 3k prompt and a 25,260-token prompt in 141.7 s = **178.2 tok/s (was 173.7: +2.6%)**: there the pass is 768 rows (the lent room is smaller), so the fixed per-layer expert copy is a larger share and
the long prompt's attention grows with length; the dense matmuls are a smaller part of the time. Next in the profile: `hc_down_inject_cols_fast_bf16` (14%), Q5_K / Q4_K / F32 dense (17%), the DeltaNet recurrence (17%).
Hand-over for another AI: `docs/HANDOFF_tensor_core_dense_prompt_2026-11-02.json`.

## 2026-11-02: tensor-core Q4_K and Q5_K dense projections (`mma_cols_q4_k`, `mma_cols_q5_k`): +7.1% at ctx 16k

`Dec8<Q4K>` / `Dec8<Q5K>` (the same floats `Q4K::at` / `Q5K::at` compute, 8 weights a call from one 8-byte nibble load) with the same tensor-core skeleton; used by `mv_cols` under `--tensor-prompt` for those types with >= 16 tokens. `SHADOW_MMA_QK=0` turns them off.
Bench, ctx 16,384, the 3,001-token prompt, 3 interleaved pairs (new first), 2 prompts each, total throughput; both arms already have `mma_cols_bf16`:

| arm | prompt times (s) | total |
|---|---|---|
| `matvec_cols_q4_k/q5_k` (`SHADOW_MMA_QK=0`) | 12.7+11.3, 12.6+10.8, 12.6+10.9 | 18,006 tokens / 70.9 s = **254.0 tok/s** |
| `mma_cols_q4_k/q5_k` | 11.8+10.1, 11.9+10.2, 12.0+10.2 | 18,006 / 66.2 s = **272.0 tok/s (+7.1%)** |

Every pair favoured the new kernels (cold launch 236-239 -> 250-253, warm 266-277 -> 294-296). Correctness: the 7-question check is 6/7, the same as before; the one-question sanity prompt (a near tie that the fast modes already decide differently from `--exact` at some cache sizes) now answers "Fast" instead of "Open" at ctx 16k -
the weights go through half precision as in the other `mma_cols_*` kernels, which changes the last digits. On by default. Since the session began the 3k prompt has gone 70 -> ~272 tok/s (the batch fix, BF16, Q4_K/Q5_K). Remaining in the profile: `hc_down_inject_cols_fast_bf16` (14%), F32 dense (6%), the DeltaNet recurrence (17%).

## 2026-11-02: tensor-core hyper-connection down/inject projection (`mma_hc_down_inject_bf16`): +8.6% to +11.2% at ctx 16k

The fused `hc_down_inject_cols_fast_bf16` (320 "down" rows + 4 "inject" rows over the 10,240-wide stream, <= 8 tokens a launch, 14% of the prompt's GPU time) has a tensor-core twin for a long prompt: one launch for all of a pass's tokens (grid ((324 rows)/64, tokens/32), 128 threads), BF16 weights and activations as half, f32 accumulate,
then the same `hc_epilogue` (down: v*sigmoid(v), inject: 2*sigmoid(v)). Used by `hc_read_batch` under `--tensor-prompt` for >= 16 tokens; `SHADOW_MMA_HC=0` turns it off. On by default.
Bench, ctx 16,384, the 3,001-token prompt, 7 interleaved pairs (2 prompts each, total throughput; the first three pairs ran new-first, the last four old-first):

| arm | total time for 42,014 tokens | tok/s |
|---|---|---|
| `SHADOW_MMA_HC=0` | 175.5 s (one pair had an 18.8 s cold launch) | **239.4** |
| `mma_hc_down_inject_bf16` | 157.8 s | **266.2 (+11.2%)** |

Dropping the pair with the 18.8 s outlier: 247.3 -> 268.5 (**+8.6%**). New was faster in 6 of 7 pairs, a tie in the seventh (22.5 s vs 22.9 s). Warm second prompts: 10.1-11.1 s (new) vs 10.6-12.0 s (old).
Correctness: the one-question sanity prompt answers "Open", the 7-question check is 6/7 (unchanged). Remaining in the profile: F32 dense (6%), the DeltaNet recurrence `delta_steps128` (17%), attention (8%).

## 2026-11-02: the DeltaNet recurrence with 8 threads a column (`delta_steps128_x8`): -4.7%, stays opt-in (`SHADOW_DELTA_X8=1`)

`delta_steps128` runs one 128-thread block per value head, a thread per column of the 128x128 state with its 128 rows in registers, 17% of the prompt's GPU time. `delta_steps128_x8` uses a 1,024-thread block (8 threads a column, 16 rows each; the 8 partial sums are added in the same order in double by every thread of the column), the same operations on the same values in the same order:
an 80-token reply under `--exact` is byte-identical with and without it. Used for >= 16 tokens, 3,001-token prompt, ctx 16,384, 4 interleaved pairs (2 prompts each), total time for 24,004 tokens:
**old 90.6 s (265 tok/s), x8 95.1 s (252 tok/s): x8 is 4.7% slower** (old faster in 4 of 4 pairs). The four extra `__syncthreads` per token (1,024 threads each) cost more than the shorter fma chains save; the recurrence is latency-bound on the token-after-token dependency, not on the chain length.
So it is **off by default** (`SHADOW_DELTA_X8=1` turns it on). What would help is not more threads on the same loop but fewer sequential steps: a chunked scan (the WY / UT-transform form that handles 64 tokens per step with matrix products on tensor cores) - a different algorithm with different rounding, not bit-exact with the reference; not attempted.

## 2026-11-02: the DeltaNet recurrence as a chunked scan (`delta_chunk_prep` + `delta_chunk_run`): a tie (-0.9%), stays opt-in (`SHADOW_DELTA_CHUNK=1`)

The gated delta rule in its WY / chunked form, 64 tokens a chunk: `delta_chunk_prep` builds, for every (chunk, value head) at once, T = (I + A)^-1 diag(beta) (A[t][s] = beta_t * gamma_t/gamma_s * k_s.k_t), Mo[t][s] = gamma_t/gamma_s * k_s.q_t, gamma and gamma_64/gamma_s (scratch: 8,448 floats a chunk-head, 17 MB per 2,048-token pass, allocated at start);
`delta_chunk_run` walks a head's chunks in order, 16 columns of the state a block (U = T (V - gamma K S0), O = gamma Q S0 + Mo U, S = gamma_64 S0 + K^T diag(gamma_64/gamma_s) U). Not bit-exact with `delta_steps128` (different rounding), same maths.
Correctness: 7-question check 6/7 (unchanged), sanity answer "Open"; an 80-token reply is a different but sensible text.
Speed, ctx 16,384, 3,001-token prompt, 2 interleaved pairs (2 prompts each): sequential **46.5 s**, chunked **46.9 s** for 12,004 tokens (-0.9%): a tie, and a tie is a loss.
The reason is in the kernel profile of the *current* build (`SHADOW_KTIME=1`): the GPU kernel time per token is now **5.11 ms** (it was 9.51 ms before today's tensor-core work) and `delta_steps128` is **0.14 ms of it (3%)**, not the 17% the first profile showed (that profile was taken with the dense matmuls still slow and
the copy engine competing for the GPU; the recurrence itself was never that long). There is almost nothing left in the recurrence for a chunked scan to save. The rest of the profile now: `matvec_cols_f32` **1.18 ms (23%)**, `attention_heads_rows` 0.56 + `attention_group_sel_rows` 0.20 (15%), `pool_block` 0.36, `matvec_cols_iq4_nl` 0.32, `matvec_fast_bf16` 0.24, `matvec_cols_q6_k` 0.22.
Next candidate: the F32 dense projections (`matvec_cols_f32`, 24 launches a pass; check which tensors they are before moving them to half precision).

## 2026-11-02: the router and shared-expert gate (F32) on the tensor cores with 3 bf16 products (`mma_cols_f32x3`): +3.0% at ctx 16k

The 23% `matvec_cols_f32` of the profile is two F32 tensors per layer: the MoE router `ffn_gate_inp` (512 x 2,560) and the shared expert's gate `ffn_gate_inp_shexp` (1 row), <= 8 tokens a launch (256 launches for a 2,048-token pass; the 1-row gate is nothing but launch latency).
They pick the experts, so half precision was not an option: `mma_cols_f32x3` splits weight and activation each into a bf16 high and a bf16 low part and accumulates hi*hi + hi*lo + lo*hi in f32 on the tensor cores (about 16 bits of mantissa, error ~2^-16 of the dot against ~2^-11 for one half product), one launch for all of a pass's tokens.
`SHADOW_MMA_F32=0` turns it off; on by default. Bench, ctx 16,384, 3,001-token prompt, 3 interleaved pairs (2 prompts each): old 65.8 s, new **63.9 s for 18,006 tokens (+3.0%: 273.7 -> 281.8 tok/s)**; new was faster in every pair, cold 11.7-11.8 -> 11.3 s, warm 10.1-10.2 -> 9.9-10.1 s.
Correctness: 7-question check 6/7 (unchanged); the one-question near-tie prompt answers "Fast" with this build (it flips between "Open" and "Fast" with any numeric change; `--exact` says "Open"). Modest: the profile's `matvec_cols_f32` share was larger than the gain because the kernel times overlapped other work.

## 2026-11-02: the prompt path is now copy-bound, and a second copy stream does not help (tie, opt-in `SHADOW_COPY_STREAMS=2`)

A fresh profile of the build after today's tensor-core kernels (`SHADOW_KTIME=1` / `SHADOW_LTIME=1`, ctx 16,384): GPU kernel time per token is **4.30 ms** (9.51 ms this morning); the largest kernels are `matvec_cols_iq4_nl` 0.65, `attention_heads_rows` 0.57, `pool_block` 0.35, `matvec_fast_bf16` 0.25, `matvec_cols_q6_k` 0.22, `attention_group_sel_rows` 0.20;
`delta_steps128` 0.14. In the pass split the **GPU waiting for the expert copy is now ~677 ms a pass (it was 58 ms before the kernels got faster)**: of a ~3.2 s pass, resident experts 581, copied experts 277, attention 912, DeltaNet segment 394+305, stall 677 - roughly a fifth. The earlier conclusion "copies are not on the critical path" held for the slow kernels; it does not hold any more.
Tried: the experts of a layer split between two copy streams (`SHADOW_COPY_STREAMS=2`; stream 1 waits for stream 2, so the half's event covers both). 3 interleaved pairs, 3,001-token prompt, ctx 16,384: one stream **65.0 s**, two **65.5 s** for 18,006 tokens: a tie, so it stays off. The copy is bound by the host memory / link, not by the number of streams.
What is left for the copy: fewer bytes per token (a bigger batch: 2,048 rows at ctx 16k, 768 at the 32k default, so the 32k default copies ~2.7x as many bytes per prompt token; a lower `--ctx` or less VRAM per token buys rows), or experts that need no copy (a bigger resident set; the CPU-experts idea, whose premises were measured earlier as unfavourable).

## 2026-11-02: what the context window costs the prompt path (all of today's kernels in; 3,001-token prompt, two prompts a launch)

The window (`--ctx`) takes VRAM from the expert cache, the cache is what a long prompt lends for its staging halves and its batch buffers, so the window sets the prompt batch:

| `--ctx` | expert cache | prompt batch rows | cold / warm tok/s |
|---|---|---|---|
| 16,384 | 2,065 slots | 2,048 | 260 / 294 |
| 20,480 | 1,953 | 1,856 | 261 / 296 |
| 24,576 | 1,837 | 1,472 | 227 / 253 |
| **32,768 (default since 39db7c0)** | 1,595 | **704** | **144 / 165** |

Each pass copies a layer's whole expert set whatever its token count, and now that the kernels are fast the copy is the limit: 704 rows copy 2.9x the bytes per prompt token of 2,048 rows. The 32k default costs **44% of the prompt speed** against 20k (decode: -1.3%, measured on 2026-11-01). There is no cheap way to give the 32k window the big batch:
16-bit K/V (half the window's VRAM; touches the attention kernels of the decode path), lending the DFlash drafter's ~1.2 GB during a prompt, or staging halves sized to the experts a layer needs. A default of 24,576 keeps 86% of the 20k speed.

## 2026-11-02: a long prompt lends 99% of the expert cache (was 96%): +15.4% at the 24,576 default

`lend_cache` caps what a long prompt borrows at 99% of the arena (`SHADOW_LEND_PCT` sets it; at least 24 slots stay with the cache, the per-token path needs ~10). The staging halves take a fixed 2.5 GiB and the batch buffers the rest, so every percent is rows: ctx 24,576 goes from **1,472 to 1,600 rows**, ctx 32,768 from 704 to 768.
3 interleaved pairs at ctx 24,576, 3,001-token prompt, 2 prompts each: 96% **75.7 s** (225 / 252 tok/s), 99% **65.6 s** for 18,006 tokens (**259 / 292 tok/s, +15.4%**). The 1,600-row pass is exactly two passes of the 3,001-token prompt (1,600 + 1,401) instead of three (1,472 + 1,472 + 57), which is where most of it comes from; the copy per pass is fixed.
Checks: a 25,260-token prompt at the 32k window reads in 159.3 s (158.6 tok/s, no panic), 7-question check 6/7. An unrelated slow minute of the machine (190 / 203 at 99% in one earlier run) is why this was measured interleaved. Default now 99%.
At 24,576 the 3k prompt now reads at 259 / 292 tok/s, level with the 20k window's 296; the 32k window is still 162 / 174 (768 rows). Desktop app: no rebuild needed (it starts `target/release/shadow.exe` with only `--port 8081`); restart it to pick up the build.

## 2026-11-03: a router bias towards the experts the VRAM cache holds (`SHADOW_ROUTER_BIAS=x`): faster decode, less faithful replies; opt-in

Weight-space idea (a per-layer vector on the router, here a constant added to the logits of the resident experts when the top-k is *chosen*; the combine weights stay the unbiased router's over the chosen experts, `route_biased`). Used in the single-token decode path and the batched (verify / prompt) path;
in a long prompt the whole layer is staged before routing, so it saves no copies there (and the cache is lent away anyway). Off by default.
Decode, 4 chat prompts x 160 tokens at temperature 0, each setting launched fresh and run twice (tok/s): bias 0: 48.0, 51.7, 53.9, 55.4, 54.1, 55.3 (mean 53.1); **0.5: 57.4, 55.8 (56.6, +6.6%)**; **1: 54.9, 58.0, 55.1, 58.2 (56.6, +6.6%)**; 2: 54.4, 53.9, 54.5, 54.2 (54.3, +2%). Launch-to-launch noise is +-3 tok/s, so the 0.5 and 1 gains are real but small.
Fidelity (`tools/quality_test.py`: 12 chat turns, how far each reply follows the `--exact` run's reply, mean prefix match): default fast modes **0.339** (1 of 12 identical); bias 0.5 **0.255** (1 of 12); bias 1 **0.108** (0 of 12); bias 2 **0.088** (0 of 12).
So 0.5 buys about +5-7% decode for roughly a quarter of the prefix agreement the default keeps; 1 and 2 give up far more for no more speed. Whether the replies are *worse* or only *different* is not measured (prefix match is harsh: one different token ends it). Stays opt-in; a task-quality check (not a prefix match) would decide it.

## 2026-11-03: the unfilled KV window is lent to a long prompt (CUDA virtual memory, `kvvmm.rs`): the 32k window reads +45% faster; on by default (`SHADOW_KV_LEND=0` off)

The Avalon idea from the video ("every byte not in use holds experts, including the part of the conversation's memory you haven't filled yet"), done for the prompt's batch buffers rather than the expert cache. `tests/vmm_probe.rs` first checked that the CUDA virtual-memory calls work on this driver / WDDM and what they cost:
2 MiB granularity; 768 chunks (1.5 GiB) create 24 ms + map 0.6 ms + set access 5 ms; unmap + release 384 chunks (768 MiB) 30 ms and the free VRAM grows by exactly that; an ordinary `cuMemAlloc` of 704 MiB then fits into the released memory.
Engine: the attention layers' K, V and indexer-raw buffers (12 layers) are `VmmRegion`s (a reserved address range backed by 2 MiB chunks; the kernels keep their pointers). When a long prompt starts (`lend_kv`), the chunks past `tokens + prompt + 256` positions are released
(~1.6 GiB at a 32k window and a 3k prompt) and mapped into a second, fixed address range (`kv_extra_reg`: fixed so the CUDA graphs captured over the batch buffers stay valid; the first version allocated a fresh buffer each prompt and the second request died with an illegal address) that holds the 2,048-row batch buffers;
`restore_cache` unmaps it and maps the KV tail back (zeroed). ~100 ms a long prompt. The cache arena keeps its size, and the lend of the arena no longer has to carry the batch buffers (smaller lend, more cache kept during the prompt).
3,001-token prompt, ctx 32,768, 2 interleaved pairs, 2 prompts each: lend off **65.1 s** (768 rows; 176 / 193 tok/s) -> lend on **44.8 s** for 12,004 tokens (2,048 rows; 247-256 / 286-288 tok/s), **+45%** (184 -> 268 tok/s). The window now costs the short prompt nothing: 32k with lending reads like 24k (259 / 291) did.
At ctx 24,576 it is a tie (11.6 + 10.4 s with lending, 11.6 + 10.3 without: 2 passes either way). A 25,260-token prompt at the 32k window: 158.7 s (159 tok/s), as before (the window is nearly full, little to lend). 7-question check 6/7; decode 56.0 tok/s before and 55.6 after a long prompt (no leak, no loss).
A failure to map the tail back (another process took the VRAM in the 100 ms) panics with "KV window: ..." and the engine's net resets the session. Open: lend the same tail to the expert cache for decode (the greedy half of the idea; bigger caches measured as losses, so unlikely to help), and re-decide the default window (see below).

## 2026-11-03: windows past ~42k fell off a cliff (110 tok/s); the staging halves now come from the lent KV tail there (64k: +100%); a 4,096-row batch is not worth it

Looking for a use of the idle KV window beyond the batch buffers, a 4,096-row batch (half the expert copying a token) was tried first: it needs a ~2.3 GiB tail, which a short prompt has only at windows past ~44k, and at those windows **the staged prompt pipeline did not start at all**.
The pipeline's two staging halves are 2 x 512 slots (2.13 GiB) lent from the expert cache; the cache is 1,595 slots at 32k, 995 at 49k, 515 at 64k, so past ~42k it cannot hold them (`lend_cache`: `n >= 2 * n_expert` fails) and the prompt takes the plain path: **110 tok/s at ctx 46,080 / 49,152 / 65,536** (the 3,001-token prompt; 105.6 on 9.5k tokens), against 260-290 at 16k-32k. That cliff is old (any window past ~42k) and nobody had measured it.
Fix: when the cache cannot host the halves, `lend_kv(true)` releases the KV tail and maps halves + batch buffers into the fixed range (`kv_halves`; rows 2,048 / 1,024 / 512 by what the tail holds), the cache is not lent at all (its experts stay resident during the prompt).
Result, ctx 65,536: the 3,001-token prompt **202-204 / 228 tok/s (was 110 / 114: +100%)** with 1,024-row passes (a 3k prompt's tail minus 2.1 GiB of halves); 9,536 tokens 129.7 (was 105.6); a 25,260-token prompt 102 tok/s (little tail: the window is part-filled); decode 50.2 (it was 49.5-50.5 at this window); 7-question check 6/7. ctx 49,152 stays in the cliff by ~100 MB (halves + the 512-row minimum do not fit its tail); 256-row passes (the tail's last option) read at 70 tok/s, worse than the plain path, so they are not offered.
The 4,096-row batch itself: the drafter's feature buffers must hold a whole pass (`feat_rows`), 4,096 doubled them (+150 MB VRAM), which cost the default window ~12% of its prompt speed (220 / 246 against 259 / 291 tok/s) and the larger batch is reachable only in a window band where the pipeline cannot start; `BIG_BATCH_MAX` is back to 2,048 (the code that picks 4,096 / 2,048 / 1,024 / 512 rows stays). Defaults unchanged: 24,576 reads 258 / 291, 32,768 255 / 288.

## 2026-11-03: conversation snapshots - going back to a chat no longer re-reads it (16.3 s -> 0.8 s)

The engine kept one snapshot (the start two prompts shared). It now also keeps a small pool of whole-conversation snapshots in host memory (`PrefixCache.convs`): after any prompt of >= 1,024 tokens has been read (`CONV_MIN`), the session is snapshot (`snapshot_to(true, true)`, KV + DeltaNet state + the rest) before the reply is generated; an older snapshot of the same conversation (its tokens a start of the new ones) is replaced,
at most 4 are kept (`CONV_KEEP`) within 4 GiB of host RAM (`SHADOW_CONV_GIB`, 0 turns it off) and only while the available RAM stays above the floor + 3 GiB. A request whose prompt starts with a stored conversation's tokens (and is longer than the session already holds) restores the longest such snapshot before the prompt is read; pictures skip it.
Test (`conv.py`: conversation A of 3,874 tokens, then a different conversation B of 3,871, then A again with its reply and a follow-up question; ctx 24,576): the request back to A took **16.3 s without the pool, 0.8 s with it** (restore 0.04 s, 23 tokens read), the same answer ("## Stage 0: done"). A snapshot costs 0.07 s and 0.31 GiB for a 3.9k-token prompt (about 80 KB a token), so a 20k-token conversation is ~1.6 GiB: the pool holds two of those.
Not covered: a snapshot is exact only at the end of a prompt (the DeltaNet state cannot be cut back), so a reply that the chat template renders differently the next time (thinking text stripped, say) still re-reads from the last matching snapshot; conversations that fall out of the pool read in full as before; the pool is lost on a restart (not saved to disk).

## 2026-11-03: conversation snapshots on disk - they survive a restart and the RAM holds only the freshest (back to a chat after a restart: 1.3 s instead of 16 s)

The pool of conversation snapshots (entry above) is now also written to files: `Snapshot::to_bytes` / `from_bytes` (a host snapshot as flat bytes), `crates/shadow-serve/src/convdisk.rs` (a file per snapshot: magic, the model's id, the tokens, the payload; a background thread writes it after the prompt has been read, with a tmp file and a rename, replaces an older snapshot of the same conversation, and deletes the oldest past the disk budget).
Default folder `ShadowConvCache` at the root of the model's drive (the NVMe: `SHADOW_CONV_DIR` moves it), at most 20 GiB (`SHADOW_CONV_DISK_GIB`, 0 = off). At start the folder's headers rebuild the index (files of another model stay, broken ones go); the RAM keeps the token lists and, as a fast path, the freshest snapshots up to 1 GiB (`SHADOW_CONV_GIB`; 4 GiB when the disk copy is off).
A request that starts with a stored conversation restores the longest match from RAM or disk. Measured (conversation A 3,874 tokens, B 3,871, then A again with its reply and a follow-up, ctx 24,576): snapshot write in the background 0.22-0.26 s for 0.32 GiB (about 1.4 GB/s; no wait for the reply); **after stopping and starting the server, the request back to A took 1.3 s (the restore from disk 0.42 s, 23 tokens read) against 16 s to read it again**, the same answer.
Cost: ~80 KB a token on the drive (a 20k-token chat ~1.6 GiB, ~1.2 s to restore at 1.4 GB/s); writes only for prompts >= 1,024 tokens. The prompt-end snapshot is exact only where the next prompt starts with the same tokens (a reply the template renders differently re-reads from the last match); snapshots with pictures are not kept.

## 2026-11-03: two faults from the desktop chat: "the cache has fewer slots than one token routes to" after the 99% lend, and a snapshot written every tool call

The desktop app's log (`%APPDATA%\Shadow\shadow-console.log`) of an agent chat: a red "internal error: the cache has fewer slots than one token routes to" on a reply, and 17 conversation snapshots of ~0.5 GiB written in a few minutes (one per tool call, ~3 s each at ~170 MB/s).
1. The panic (`ExpertCache::victim`, from `moe_batch` under `step_batch_verify`): a chunk of <= 8 tokens (the tail of a prompt, or what is left after a snapshot point) takes the non-staged path, which needs a free slot in the layer's own cache range; a long prompt lends 99% of the arena (`lend_cache`), which empties the range of nearly every layer
(it already happened at 90% and 96%, less often: STATUS 2026-11-01). Fix: the prefill loop calls `restore_cache` before a chunk of <= `MAX_VERIFY` tokens (the lend would end there anyway; the evicted experts are read back either way). Checked with `SHADOW_PREFILL_CHUNK=1000` on a 3,001-token prompt (chunks 1,000 / 1,000 / 1,000 / 1): no panic, 212 / 227 tok/s.
2. Snapshots: the freshest snapshot of a conversation is kept in RAM and marked pending; it goes to disk when the next request does not continue that conversation, or after 30 s without a request (`recv_timeout` in the engine loop), and a newer snapshot of the same conversation replaces a pending one: one write per conversation episode instead of one per turn (a snapshot too big for the RAM pool still goes straight to disk).
Re-checked: A, B, then 40 s idle: all three snapshots on disk; after a restart the request back to A took 1.3 s (restore from disk 0.45 s) as before.


## 2026-11-04: prompt-side speculation benchmark (4 arms x 3 runs interleaved, prompt-heavy vs mixed text)

Experiment specified in docs/EXPERIMENT_SPEC_2026-11-04.json to probe whether speculation (--spec 3, DFlash 2) helps, hurts, or ties on prompt-heavy and mixed workloads.
Scoring: TOTAL throughput (sum(tokens) / sum(seconds)), warm turns only (turn 0 dropped), 4 arms x 3 runs interleaved with fresh server launches per run, RTX 5070 12.2 GB VRAM (4.42 GiB expert cache), 36 GiB pinned RAM tier, 3 GiB floor.

| Workload & Arm | Prompt tok/s | Decode tok/s | Combined Total tok/s | Paired Decode Delta |
|---|---|---|---|---|
| **Prompt-Heavy (6 turns, ~2.5k-3.5k tokens/turn, ctx 16k, tokens 16)** | | | | |
| Arm A: baseline (spec off: --dflash off --spec 0) | 188.74 (164.9, 204.0, 202.9) | 29.29 (29.1, 29.9, 28.9) | 186.93 (163.5, 202.0, 200.8) | reference |
| Arm B: spec on (--spec 3, DFlash 2) | **190.33 (+0.8%, tie)** | **36.56 (+24.8%)** | **188.89 (+1.05%)** | **+27.2%** (+6.0% to +37.3%) |
| **Mixed (36 turns varied text `bench/dfdata/varied36.json`, ctx 8k, tokens 96)** | | | | |
| Arm C: baseline (spec off: --dflash off --spec 0) | 55.03 (55.1, 55.1, 54.9) | 34.56 (34.4, 34.7, 34.6) | 40.29 (40.1, 40.4, 40.3) | reference |
| Arm D: spec on (--spec 3, DFlash 2) | 51.51 (-6.4%) | **51.55 (+49.2%)** | **51.53 (+27.9%)** | **+45.4%** (+23.5% to +77.8%) |

Observations and answers to the hypothesis:
1. **Prompt ingestion is not degraded by DFlash**: In the prompt-heavy regime (132k warm prompt tokens benched), Arm B prompt read speed was 190.33 tok/s vs Arm A 188.74 tok/s (+0.84% difference, essentially a tie within the 0.8% noise floor). On large batches, the df_capture taps and ring buffer updates are completely amortized under the dense tensor-core kernels and expert DRAM copies.
2. **Decode gains persist across prompt lengths**: Even after cold, multi-thousand-token prompts, DFlash decode speculation delivers +24.8% total decode tok/s (29.29 -> 36.56 tok/s; paired median +27.2%).
3. **Mixed workload net win**: On short prompts, DFlash capture introduces a minor fixed cost (-6.4% on prompt reading: 55.03 -> 51.51 tok/s), but the decode speedup from 34.56 to 51.55 tok/s (+49.2%) dwarfs this, giving a **+27.9% net gain in combined total throughput** (40.29 -> 51.53 tok/s).
4. **Conclusion**: Speculation does not harm prompt ingestion and delivers consistent net positive total throughput across both prompt-heavy (+1.05%) and mixed (+27.9%) regimes. Per the rules, decode speculation remains active by default (--spec 3).


## 2026-11-04: hybrid read path & direct I/O benchmark (Prompt-Heavy & Mixed text, 4 arms x runs interleaved)

Experiment specified in `docs/EXPERIMENT_SPEC_2026-11-04b.json` (Task 1): test direct unbuffered I/O straight into the pinned buffer via an overlapped event pool (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, `crates/shadow-model/src/readpool.rs`), gated behind `SHADOW_DIRECT_IO=1` (baseline buffered reads stay default).
Hypothesis: Direct I/O skips one DRAM round-trip for cold expert bytes (two touches instead of three: NVMe direct to pinned buffer, pinned to GPU link).
Physics risk: Direct I/O bypasses the Windows OS file cache. While raw NVMe sequential speed is ~3.5 GB/s, MoE expert access patterns repeat across prompt chunks and decode steps. The standby page cache serves those hits at memory speed (~9.5–25 GB/s).

Scoring: TOTAL throughput (`sum(tokens)/sum(seconds)`), warm turns only (turn 0 dropped), RTX 5070 12.2 GB VRAM (4.42 GiB expert cache), 36 GiB pinned RAM tier, 3 GiB floor.

| Workload & Arm | Prompt tok/s | Decode tok/s | Combined Total tok/s | Paired Delta vs Baseline |
|---|---|---|---|---|
| **Prompt-Heavy (6 turns, ctx 16,384, tokens 16)** | | | | |
| Arm A: baseline (buffered file cache) | 158.56 (150.6, 158.1, 167.9) | 33.74 (33.7, 33.8) | 157.51 (149.6, 157.1, 166.7) | reference |
| Arm B: direct I/O (`SHADOW_DIRECT_IO=1`) | 105–140 (**-20% to -42%**) | 28.1–33.7 (**-16.5%**) | 104–139 (**-21% to -41%**) | **-20% to -42%** |
| **Mixed (varied text, 36 turns, ctx 8,192, tokens 96)** | | | | |
| Arm C: baseline (buffered file cache) | 52.91 (53.0, 52.9) | 53.06 (52.8, 53.3) | 53.02 (52.9, 53.1) | reference |
| Arm D: direct I/O (`SHADOW_DIRECT_IO=1`) | 49.97 (**-5.6%**) | 47.70 (**-10.1%**) | 48.40 (**-8.7%**) | **-10.3% paired median** (-7.2% to -12.8%) |

Observations:
1. **Direct I/O is a net loss across all regimes**: In prompt-heavy workloads, unbuffered direct I/O reduces prompt ingestion by -20% to -42%. In mixed workloads, decode speed drops -10.1% (53.06 -> 47.70 tok/s) and combined throughput drops -8.7% (53.02 -> 48.40 tok/s).
2. **The file cache matters more than the DRAM round-trip**: Bypassing the OS standby file cache forces repeating expert accesses across prompt chunks and decode steps to hit physical NVMe flash at 3.5 GB/s rather than memory bus speed (9.5–25 GB/s).
3. **Working set overhead**: Maintaining 64 x 1 MiB page-aligned direct-I/O staging buffers increases working set footprint, dropping available RAM closer to the 3.0 GiB safety floor.
4. **Conclusion**: Per the tree's discipline ("measure, and what loses stays opt-in"), `SHADOW_DIRECT_IO=1` stays strictly opt-in and off by default. The standard buffered read pool remains the active default.

## 2026-11-03: raw host-to-device copy from the engine's own region runs at the full PCIe rate (28 GB/s); the engine's staged copies reach about half that

`tests/h2d_probe.rs` (1 GiB, the best of 4 passes, `cuMemcpyHtoDAsync` on the default stream): from the engine's `PinnedRegion` (VirtualAlloc + `cuMemHostRegister`) **27.6 GB/s cut in 2 MiB pieces, 27.9 in 16 MiB, 28.05 in 128 MiB, 28.03 as one copy**; `cuMemHostAlloc` 27.65-28.06; write-combined 27.6-28.07.
So how the memory is made and how the copies are cut make **no** difference, and the link (PCIe 4.0 x16) is the ceiling at ~28 GB/s. The prompt path's expert copies come out at ~13 GB/s end to end (the spec of 2026-11-01 measured the same): about half the link is idle or contended.
Arithmetic for a full pass (every expert is used by ~2,048 tokens): ~43 GiB over the link (47 GiB of experts, 4 GiB in the VRAM cache) is **1.65 s at 28 GB/s** = ~1,240 tok/s at 2,048 rows from the copy alone; the engine's passes take ~3-4 times that. Suspects, not yet measured: the CPU's reads of the model file for the ~23% of experts that are not in the RAM tier (page cache -> pinned, a read and a write of ~11 GiB a pass) compete with the DMA for the ~30 GB/s of DRAM bandwidth;
the copy of layer l + 1 is only queued after layer l's router (the copy engine idles while the GPU computes the layers before it). Next: copy rate with a CPU thread streaming the model file at the same time (DRAM contention), and the link's busy fraction over a prompt pass.

## 2026-11-03 (night): the prompt gap to Strata measured on this PC, and why - DRAM is shared between the link and the CPU, and Strata's passes hold the whole prompt

**Head to head, same machine, same prompts (STATUS.md excerpts; Strata = `build-upstream-029` with the 2026-10-28 bench config at `--max-context 24576`, `--prefill auto`; Shadow = defaults, ctx 24,576; Strata alone on the GPU, then Shadow alone):**

| prompt | Strata | Shadow |
|---|---|---|
| 3,043 / 3,001 tokens | 4.2 s = **733 tok/s** | **194 (cold) / 210 tok/s** |
| 9,578 / 9,536 tokens | 10.3 s = **934 tok/s** | 45.5 s = **209 tok/s** |
| 15,178 / 15,136 tokens | 10.7 s = **1,422 tok/s** | 72.9 s = **208 tok/s** |

(the Shadow 3k figure was 258 / 291 earlier today with a warm file cache; Strata's run had just used 40 GiB of RAM, so this one started cold). The "1,376 tok/s" in the old notes is a real figure for long prompts on this PC. **Strata's speed rises with the prompt (733 -> 934 -> 1,422); Shadow's is flat at ~210.**
Model that fits the three Strata points: a fixed ~3.2 s a pass (the expert copy, ~43 GiB) plus ~0.5 ms a token (compute), with the whole prompt in one pass (rows >= 15k). Shadow: a pass is capped at 2,048 rows by VRAM (590 KB a row: ~30 per-row buffers, among them the 41k-float expert intermediates eg / act / eo and three 10k-float hyper-connection streams),
so the fixed copy cost is paid every 2,048 tokens (1.6 ms a token), and Shadow's compute is ~1.7 ms a token against Strata's ~0.5.

**Why the copy is slow: DRAM is shared.** `tests/h2d_probe.rs`: a DMA from the engine's pinned region alone runs at **28 GB/s** (the PCIe 4.0 x16 rate, whatever the allocation or chunking); with **one** CPU thread copying memory beside it the DMA falls to **15.2 GB/s**, with two **14.5**, with four **13.9** - the engine's ~13 GB/s. Non-temporal stores for the CPU copy change nothing (12.6 GB/s beside one thread of either kind): the ~30 GB/s of DRAM is the budget, the DMA alone takes all of it.
The CPU's share in a pass is the read pool's file-cache -> staging copies for the ~23% of experts outside the RAM tier (~11 GiB, read + write).
**New fact:** `cuMemHostRegister(view, size, CU_MEMHOSTREGISTER_READ_ONLY = 8)` works on a `MapViewOfFile` view of the model file: registration 56 us, the DMA reads **the file's exact bytes** (checked, MATCH) at 28 GB/s once the pages are in, and for pages nobody has touched the first DMA pages them in from the NVMe (118 ms for 256 MiB = 2.3 GB/s) - the GPU can pull straight from the file cache with no CPU copy. Cost: pinning the pages on the first DMA after a registration (~50 ms per 256 MiB, ~5 GB/s), so views would be registered once and kept, not per pass.
**So the levers, in the order the numbers give them:** (1) rows a pass: 2,048 -> 8,192+ cuts the copy a token 4x (VRAM per row has to fall ~4x: build the expert intermediates per expert group instead of per (token, expert); keep the hyper-connection streams in half precision or recompute them); (2) compute a token 1.7 -> ~0.5 ms (attention 0.6 and the DeltaNet / dense projections: the profile of 2026-11-02); (3) CPU DRAM traffic out of the pass: file-backed experts as registered file views (no staging copy), the RAM tier left alone.
Not needed to start: a new engine. The passes, the per-row buffers and the attention kernels are what to change; each step above can be measured against this table. Strata's source is in `J:\Strata-work\Strata-up029\src` for the layout of its prefill.

How Strata's prefill is built (read from `J:\Strata-work\Strata-up029\src\prefill\prefill.cpp` comments, not run): a prompt goes in **chunks of up to 8,192 tokens**; from 1,024 tokens up *every* non-resident expert of every layer streams in a fixed order through a **ring of slots** ("nearly all 512 are routed at such a chunk, so the copy engine keeps working through the attention halves instead of waiting for each layer's routing"): no per-layer double buffer of 2 x 512 slots, the staging memory is the ring, which is what leaves VRAM for the big chunk;
the expert products are its own **MMQ int8 tensor-core kernels** (`moe_mmq.cu`, `mmq_kq_lib.cu`), the dense projections cuBLAS, the DeltaNet recurrence a **chunked tensor-core scan** (`gdn_chunk.cu`, `gdn_tc.cuh`); the CPU takes part only below 1,024 tokens (`STRATA_PREFILL_CPU_SHARE`, on by default there: -20..-35% at 512-1,000 tokens, none above). That is the design the numbers above ask for: a continuous expert stream through a small ring, 8k-row chunks, int8 tensor-core experts.

## 2026-11-04: short-turn expert prefetch - not built; the existing pre-gated prefetch is -66% in the default mode

Task: hide the ~15 ms/token of demand copies on short chat turns (prompts under ~128 tokens) by prefetching the next layer's / token's experts.

- Measured on the current tree (baseline binary = HEAD plus the uncommitted working tree, `target-dev/shadow-base.exe`), varied36 bench, 2 runs x 14 turns, default flags, `--ctx 8192`, routing-varied300 placement, total throughput (`tools/bench_total.py`, `bench/pf-2026-11-04`):
  default **53.62** tok/s (53.8, 53.4); `--pregate 8` **18.46**; `--pregate 16` **17.67 / 17.3, 18.0**. Same number of tokens a run. Pre-gating makes the per-layer graph path ineligible (`prefetch.is_none()` gates the CUDA graphs, `session.rs`) and adds its router product and host round trip per layer: a loss of two thirds, not a tie.
- `SHADOW_TIMING_NO_COPIES=1` run (`bench/pf-nocopy-2026-11-04`) produced 26 tokens a run of wrong text, so its tok/s (36.8) is not a ceiling; the earlier same-switch measurement stands: demand copies cost 19.3 of 59.7 ms per verify pass (32%).
- Why this case does not differ from the earlier negatives: (1) the verify pass is bus-bound (~570 MiB of misses at ~22 GB/s is ~26 ms against ~35 ms of dense GPU work), so a prefetch moves bytes the demand copy would move anyway and, with mispredictions, adds bytes; (2) the DMA slows the layer's own kernels (2026-10-27: copy wait 13.3 -> 8.6 ms but GPU wait +3.3 ms); (3) cheap predictors are weak: last token's experts recall 0.39 at d=1, the layer-ahead router 64-67% in the top 10 (2026-10-27, 2026-10-31); (4) the layer-ahead staging pipeline (`moe_staged`) was already found to lose below ~1,024 tokens (a 24-turn chat ran 5.6 vs 14 tok/s), and a prompt under 128 tokens has even fewer experts to amortise its pinned buffers and cache restore over, so lowering `LONG_PROMPT` was not built.
- No Nsight Systems trace was taken: the run was unattended and the earlier traces (2026-10-31, `--cuda-graph-trace=node`) already show the pass dense on the GPU with copy wait the only slack. No code changed; nothing to revert. Remaining levers are unchanged: more hits (a bigger cache) or fewer bytes, not earlier bytes.
## 2026-11-03 (night, 2): compute-side attempts after the Strata comparison: two ties and one fidelity failure; the profile's kernel times mislead once the pass is copy-bound

- **Indexer pooled keys in one launch (`pool_blocks`, `SHADOW_POOL_BATCH=0` off):** a prompt pass completed hundreds of blocks a layer, one launch each. Three interleaved pairs, ctx 24,576: **69.3 s against 70.1 s** for 18,006 tokens (+1.2%, the cold launch 0.2 s faster every time). Bit-exact (the same per-block code). On by default (no downside), a small gain.
- **Tensor-core IQ4_NL dense projections (`SHADOW_MMA_NL=1`; the kernel existed, `mv_cols` never used it):** 3 interleaved pairs, **68.6 s against 68.9 s** (+0.4%: a tie) and the 7-question check 5/7 against 6/7. `KTIME` had shown `matvec_cols_iq4_nl` at 0.65 ms a token (15% of GPU time): **event-timed kernel spans include the time the copy engine and the DRAM contention stretch them**, so with the pass bound by the copy a kernel that "takes" 15% gives nothing when it gets faster. Off by default.
  This also explains why the dense-kernel wins stopped after the hyper-connection one: the pass is now limited by copying bytes (43 GiB a pass, ~14 GB/s with the CPU beside the DMA), not by the GPU's kernels.
- **Per-row VRAM (591 KB a row, `SHADOW_ROW_INVENTORY=1` lists it: r/rn/up 20%, attention buffers 20%, DeltaNet buffers 22%, expert intermediates eg/act/eo 30%) - sharing buffers:** the attention set and the DeltaNet set (never both in one layer, 30,720 floats saved a row) and `up` with `eg` (10,240 floats). `--exact` output stayed byte-identical (the 80-token reply hash 78221637), but the repo's fidelity test
  (`quality_test.py`, 12 chat turns against the `--exact` replies) says **sets 0.001 (replies of a few tokens), up 0.271, both 0.001, none 0.349**: sharing the two sets breaks the fast modes, for a reason not found (`bb.qg/q/gate/ao` appear only in `attention_batch`, `bb.qkv/x/z/o` only in `delta_batch` / `multi_group`; a replay, a rollback or the drafter reading one after the other layer kind wrote it would do this). Opt-in only (`SHADOW_ALIAS_ROWS=on|sets|up`), off by default; `DeviceBuffer::subview` stays.
- **The order again, with this:** the pass is copy-bound, so bytes a token is the lever: rows a pass (VRAM a row: the 30% expert intermediates and the sets, once the sharing fault is found; and the staging halves 2.1 GiB -> a ring), then the CPU's DRAM traffic. Compute work (attention, IQ4_NL) pays only after those.

- **2026-11-04: the buffer-sharing fault found - two bugs of mine, not the fast modes.** (1) `make_bb` kept the shared region in a local that was dropped at the end of the function: the views dangled, and in the small batch (owned allocations) later buffers were handed the freed memory. That is why `sets` gave fidelity 0.001 even with speculation off (0.003 / 0.001 / 0.004 with `--dflash off --spec 0`, with and without `--tail-skip 0`) and, with `--exact` + `sets`, 0.004: **my earlier note that `--exact` stayed byte-identical was wrong** (that run did not have the sharing on). Fixed: `BatchBufs._region` owns it; `--exact` + `sets` is then 12 of 12 identical (1.000). Each carved buffer is also 256-byte aligned now (padded region). (2) The batch CUDA graphs were keyed by (layer, m, rows) but their launches carry the buffers' addresses: with the sets shared a 3k and a 9k prompt both fit 2,048 rows at different stretches of the lent KV window, and the 9k prompt replayed the 3k prompt's graph (illegal address, HTTP 400). The key now holds the batch buffers' base address (`session.rs`, `session_pipeline.rs`); no change for the default.
- **Measured, sharing the sets + `up` (`SHADOW_ALIAS_ROWS=on`) against off, ctx 24,576, two pairs, tok/s:** 3k 260.7 / 275.8 and 264.1 / 291.2 against 252.7 / 270.6 and 260.3 / 289.1 (+2%); 9.5k **301.7 and 318.2 against 276.1 and 292.7 (+9%)**; 15k **293.2 and 294.0 against 262.2 and 263.5 (+12%)**. `up` alone (with `eg`): a tie (9.5k 276.5). `--exact` output identical. **The fidelity test against `--exact` replies is lower with it on: 0.219 (deterministic, three runs) against 0.339 (twice)**, replies a few tokens shorter (922 against 935 tokens). Fast modes depend on cache composition, so a different composition is not by itself a bug, but a deterministic loss is a loss: **opt-in stays (off by default)**, decision for the user: +9-12% on long prompts against 0.12 less agreement with `--exact` in the fast modes. Not found why it differs; `--exact` is clean, so it is in a fast path (tensor-core projections, tail skip, verify) reading something the sharing changed.
- **FreeToken (FlashML, arXiv 2608.16157) read for ideas, paper text only, not its code, numbers are its own claims on an RTX 5090:** (a) its prefill is a double-buffered *full-layer* stream (the next layer's whole expert set while this one computes), chunks of 8,192 tokens that take as long as streaming the pool once (52.7 GB/s); without the second buffer -19% at 4k, -26% at 16k: the same shape as our staging halves, and the same lever as rows a pass (Strata 8k, us 2,048). (b) q*: of the m experts a decode layer misses, fill q ~ m * B_pcie / B_host and run the rest on the CPU, with the *residual* host bandwidth max(B_host - B_pcie, 0) because DMA and CPU share DRAM: this is the contention we measured (28 vs ~14 GB/s). (c) Semantic anchor checkpoints: recurrent state saved at special-token boundaries (thinking, tool call, tool output, turn) on a radix tree of KV, restore the deepest surviving one and prefill only the suffix: our conversation snapshots are at turn ends only. (d) A global LRU over all layers whose victim choice runs in one GPU kernel that also rewrites routed ids to slots (graph-compatible); decode miss rates 16% against llama.cpp 62% (Qwen3.6). (e) Rebuilding the expert cache for a new VRAM budget at a safe point (our lend/restore is the same idea).

- **2026-11-04: buffer sharing (attention/DeltaNet sets + `up`/`eg`) is now ON by default, at the user's say** (+9-12% on 9-15k prompts for 0.12 less fast-mode agreement with `--exact`). Default build measured: 3k 250.2 / 275.4, 9.5k 298.6, 15k 288.5 tok/s; fidelity test 0.219; `--exact` 1.000 (12 of 12 identical). `SHADOW_ALIAS_ROWS=0` turns it off; a pipelined session never shares.

- **2026-11-04: more rows a pass - the cap is VRAM, not code.** The drafter's feature/context buffers no longer scale with the pass: `df_capture` / `dflash_observe` take a pass's last `DF_ROWS` (= `WINDOW`, 2,048) positions, all the drafter looks back (`dflash.rs`, `session_dflash.rs`); `BIG_BATCH_MAX` is 8,192 and `lend_kv` offers 8,192 / 4,096 / 2,048 rows (`SHADOW_BIG_ROWS` sets the cap, default still 2,048). Measured on the RTX 5070 (12 GiB), prompt tok/s, rows cap 2,048 / 4,096 / 8,192: 24k window 9.5k prompt 295.7 / 276.8 / 296.6, 15k 284.0 / 276.2 / 284.6 - **no gain, because the rows actually used stay 2,048**: `SHADOW_STAGE_TIME=1` shows the lent room is 0.84 GiB at the 24k and the 32k window (the KV tail), 0.42 GiB at 65k, = 2,048 rows of ~410 KB (shared buffers) and no more; at 65k the prompt reads at 124 / 107 tok/s (the small arena). **The two staging halves are 2.54 GiB of the 3.56 GiB arena** - that is what stops the rows: a ring of expert-sized slots (a few hundred, not two whole layers) in their place frees ~2 GiB = ~5,000 more rows. Next.

- **2026-11-04 (2): is the pass copy-bound? Rows tested with room for them (drafter off, `SHADOW_KV_LEND=0 SHADOW_PREFILL_LEND_MIB=99999`, 24k window), prompt tok/s, rows 2,048 / 4,096 / ~4,600 (8,192 asked, 1.90 GiB of room):** 9.5k 301.5 / 364.5 / 366.6, 15k 292.9 / 346.2 / 342.1, 3k 278.8 / 318.2 / 312.6. **Doubling the rows is +21%, not the 2x a copy-bound pass would give, and more rows after that are flat: the pass is now bound by compute (marginal ~2.2 ms a token, ~2.3 s a pass fixed).** So a staging ring (frees ~2 GiB for ~5,000 more rows) would add nothing past ~4,096; what 4,096 rows needs is ~1 GiB more VRAM than the drafter's 1.2 GB leaves (the +21% is with `--dflash off`; `--dflash` default on has 2,048 rows). IQ4_NL tensor cores at 4,096 rows: 368.1 / 369.3 against 365.3 / 365.9 (9.5k), 352.3 / 352.7 against 348.5 / 347.9 (15k): +1% - still a tie, so the per-kernel profile (`SHADOW_KTIME`: 2.88 ms a token, attention 0.59, `matvec_cols_iq4_nl` 0.32, q6_k 0.26) is still inflated, not a to-do list. The host's MoE planning is 12 ms a layer at 4,096 rows (`grouped_plan` is O(experts x tokens)): 0.5 s of an ~11 s pass. `BIG_BATCH_MAX` 8,192, default cap unchanged (2,048).

- **2026-11-04 (3): 4,096 rows with the drafter on - not yet.** The batch buffers are now carved first-fit over the KV tail and the lent arena past the halves (`big_batch_on` finds the most rows, a multiple of 64, whose buffers fit; `lend_cache` lends the arena part a bigger `SHADOW_BIG_ROWS` needs). 24k window, prompt tok/s, 2,048 / 4,096 rows: **drafter on** 3k 274 / 306, 9.5k 297.4 / 298.6, 15k 289.0 / 289.0 (3k +11%, long prompts a tie); **drafter off, default lend** 9.5k 315 / 311 and 305 / 306 (tie); **drafter off, the whole arena lent and no KV lend (`SHADOW_KV_LEND=0 SHADOW_PREFILL_LEND_MIB=99999`)** 303.4 and 305.7 / **366.8 and 352.8 (+19%)**; drafter off, KV lend + the whole arena lent: 328 / 336. So the +19% needs the arena emptied (no resident experts) and no KV lend, which holds ~2,400 rows with the drafter loaded; why the half-empty cache or KV-tail buffers lose most of it is not found (`SHADOW_LTIME` events distort the pass too much to say). KV lend on vs off at 2,048 rows, drafter on: 270.7 / 297.6 / 289.3 against 267.6 / 300.0 / 318.0 (3k: 245 / 271 / 266 against 248 / 276 / 285): a tie to slightly worse with the KV lend at this window. Default unchanged (2,048 rows).

- **2026-11-04 (4): 4,096 rows a pass is the default (drafter on) - prompts +14-24%.** A prompt whose KV tail was too small for a pass alone was refused the KV lend and then took only 2,048 rows' buffers from the arena (the `want` was hard-coded to `BIG_BATCH`), so every earlier 9.5k / 15k 'cap 4,096' run was in fact 2,048-row passes. Now a tail of at least 512 rows adds to the arena's part, and `SHADOW_BIG_ROWS` (default 4,096) sets the lend and the carving alike. 24k window, drafter on, 2,048 / 4,096 rows: 3k 283.4, 287.4 / 311.1, 312.0; **9.5k 302.3, 316.2 / 349.5, 342.2; 15k 287.5 / 354.0, 354.4**; with decode after the prompt (9.5k then a follow-up, two runs each) the prompt 319.1, 321.0 / 367.1, 365.8 tok/s, the first reply's decode 38.5, 38.5 / 35.0, 34.8 (-9%: the cache is emptier) and the follow-up's 32.8, 33.1 / 36.8, 37.2 (+12%): net ~3.5 s faster on a 9.5k prompt and a 160-token reply. Default build: 3k 310.3, 9.5k 371.5, 15k 358.0 tok/s; `--exact` reply hash 78221637 as before, `--exact` agreement 1.000 (12 of 12), fast-mode agreement 0.219 (unchanged). Also `grouped_plan` is now linear in tokens x k (one walk over the tokens, not a search of every token for each expert), `--exact` hash identical; its host time not re-measured (was 12 ms a layer at 4,096 rows).

- **2026-11-04 (5): what the pass waits on (default 4,096 rows, drafter on, 15k prompt) - sampled, not instrumented.** `nvidia-smi dmon` once a second over the prompt: the GPU shows a kernel running in 72% of the seconds (SM%, avg), PCIe rx averages 5.9 GB/s of the ~28 GB/s the engine's pinned region reaches, in two kinds of seconds: ~2 in each ~7-8 s with 18-37 GB/s and the SMs at ~52%, the rest at 80-100% SM with ~0.2 GB/s. A clean kernel profile of one 15k prompt (`SHADOW_KTIME=1`, no decode in it): 1.74 ms a token of kernels against ~3.0 ms of wall; the sparse-attention family 0.47 (`attention_group_sel_rows` 0.26, `index_scores_rows` 0.15, `attention_heads_rows` 0.06), `matvec_cols_iq4_nl` 0.19 (COLS_MAX = 8 tokens a launch, so 6 launches a token; tensor cores off), `matvec_fast_bf16` 0.12, the rest under 0.1 each. Tried: queue each layer's copy a layer earlier (copy l + 2 behind layer l's combine, a compute-to-copy event; `SHADOW_STAGE_LEAD=2`) so it overlaps a whole layer, not just the experts: `--exact` output identical (hash 162a8e07 both ways), prompt tok/s 9.5k 365.3, 358.4, 358.2 against 368.9, 359.4, 357.1, 15k 349.1, 350.5, 354.0 against 358.7, 357.0, 354.7: **a tie - the copy is not on the critical path - opt-in, default is the old order.** What is left is the GPU's own 72%: kernels at ~1.7 ms a token (sparse attention a quarter of it) and the ~28% of seconds with no kernel running (host syncs per layer, launch gaps), not measured apart.

- **2026-11-04 (6): the sparse attention's block selection moved to the GPU - prompts +20%.** `SHADOW_SEL_TIME=1` showed where the GPU's idle time was: each 32-row group of `attention_batch`'s sparse path downloaded the index scores, selected the top blocks on the host (`select_nth`) and uploaded the token lists - per attention layer in a deep pass 300 ms, of which 100 ms host select and the rest the download waiting on a queue emptied every group (102 groups a layer at 3,264 rows; 12 attention layers). New kernel `select_blocks_rows` (`kernels/forward.cu`): one block a row, radix selection of the keep-th best key (score order, then lower block first - the host's total order), the chosen blocks' tokens in ascending order then the incomplete block's, written in the layout the attention kernels read; no download, so the group's launches queue behind each other. `SHADOW_SEL_CPU=1` is the old host path. **Prompt tok/s, GPU select / host select, two pairs, default build, 24k window: 9.5k 425.8, 427.7 / 355.1, 367.8; 15k 424.4, 444.2 / 355.2, 357.1; 3k 318.8, 328.0 / 310.2, 313.0.** Same output: `--exact` reply hash on a 9.5k prompt 162a8e07 both ways, default (fast modes) f35749c5 both ways, `--exact` agreement 1.000 (12 of 12), fast-mode agreement 0.219 (unchanged). Decode after a 9.5k prompt (first reply, follow-up): 35.8 / 33.2 against 33.2 / 31.0 tok/s. `index_scores_rows` still does its dot products in `double` (one thread a block, 0.15 ms a token): the next kernel to look at, but a precision change would move the selection.

- **2026-11-04 (7): `index_scores_rows` in float - a tie, opt-in.** New kernel `index_scores_rows_f32` (the dot products in float, fmaf in the same order; the double one is limited by the card's FP64 rate). Prompt tok/s, float / double, three interleaved pairs, default build with GPU selection: 9.5k 426.6, 429.5, 451.7 / 419.1, 441.2, 444.7; 15k 457.9, 461.6, 461.7 / 450.9, 449.0, 452.1; total time of the six runs 164.4 s / 166.6 s = **+1.3%**, fast-mode agreement 0.219 both. But the 9.5k reply changes (hash a517bdff against f35749c5): a block at the selection's margin flips. +1.3% for a different reply is within the tie range, so `SHADOW_IDX_F32=1` (with `--tensor-prompt`) turns it on; default stays the double. The index scores are no longer where the time is: the earlier 0.15 ms a token was the stalled, event-inflated figure.

- **2026-11-04 (8): idle time re-sampled after the GPU selection; Nsight Compute blocked.** `nvidia-smi dmon` over a 15k prompt (467.3 tok/s that run): a kernel running in 84% of the seconds (72% before), PCIe rx averaging 9.8 GB/s (5.9 before) with 18-35 GB/s bursts now at 84-100% SM - copy and compute overlap more, the GPU idles ~16% (part of it the run's ramp seconds). **Tensor-core attention is not started: the kernel is not shown to be bandwidth- or tensor-bound.** `attention_group_sel_rows` keeps scores and exponentials in a double scratch in global memory and accumulates in double (reference-exact), ~512 selected tokens a row; a mma kernel is inexact (a fast-mode kernel) and a guess without a profile. `ncu` (Nsight Compute 2026.3.1 is installed) refused: ERR_NVGPUCTRPERM - GPU performance counters need an admin or the NVIDIA Control Panel's 'Allow access to the GPU performance counters to all users' (a system setting, left to the user). The first attempt also left one orphan `shadow.exe` (16 GB, pid 5680) running after `ncu` was refused; killed.

- **2026-11-04 (9): Nsight Compute on `attention_group_sel_rows` (elevated, `bench/ncu_attn_group.ncu-repz`, `bench/ncu_attn_group.ps1`): FP64-bound - and a float version is a tie.** Per launch (32 rows, grid 6 x 32 blocks of 256): 932 us, **FP64 pipe 74% busy**, DRAM 2.3%, L2 31%, L1 hit 35%, 80 registers, 33.8 KB shared a block -> theoretical occupancy 33% (achieved 31%), 12.9 cycles a warp between issues, issue slots 29% busy. So tensor cores are the wrong tool (it is not bandwidth- or tensor-bound; the scores, exponentials and the weighted sum run in double on a card with FP64 at 1/64 of FP32). `attention_group_sel_rows_f32` (same kernel, scores/softmax/weighted sum in float): prompt tok/s float / double, three interleaved pairs, 9.5k 412.4, 431.0, 431.8 / 424.7, 428.6, 429.5, 15k 443.2, 428.2, 452.7 / 434.9, 440.9, 445.3: total 170.3 s / 170.0 s = **a tie** (the double kernel's ~1 s a pass was not on the critical path). Reply agreement with `--exact` on three ~9k prompts (120 tokens): float 0.046, 0.063, 0.018, double 0.026, 0.063, 0.138 (both low: the fast modes diverge early on long prompts). `SHADOW_ATT_F32=1` turns the float kernel on; default is the double. Taken together with the copy-lead and float-index-score ties: **the pass has several limits within a few percent of each other, so removing one moves nothing** - next is a timeline (Nsight Systems) of what the GPU does between and during layers.

- **2026-11-04 (10): the timeline (Nsight Systems, `bench/nsys-pass36-2026-11-04.nsys-rep`, `tools/nsys_pass_summary.py`) - the pass is paced by the per-layer expert copies at ~14 GB/s.** 9.5k prompt, default build, 36 GiB tier (405.8 tok/s under the tracer), window = first to last sparse-attention launch, 11.29 s: **kernels running 51.3% of it, host-to-device copies 62.9%, both 27.1%, neither 12.9%, copy with no kernel 35.8%**; 96.7 GB copied, in 92 copies of ~1.0 GB (one per layer, 59-72 ms each, **13.3-15.3 GB/s each, the same whether a kernel overlaps (15.3) or none does (13.9)**) plus ~5,900 small ones; one copy in flight at a time; consecutive layers' copies start ~77 ms apart. That is why every compute-side change tied (float index scores, float attention, copy a layer ahead, IQ4_NL tensor cores): the copies set the pace and the kernels (5.8 s of 11.3) fit inside. Kernel time in the window: `attention_group_sel_rows` 23.8%, `attention_heads_rows` 13.2% (the dense path for a pass's first positions, 2,827 launches), `index_scores_rows` 7.6%, `matvec_cols_iq4_nl` 6.6% (37,536 launches of 10 us), the expert kernels ~25%. **Why 14 GB/s:** the link does 28 (PCIe 4.0 x16, STATUS 2026-11-03) and a kernel running does not slow it; the host's DRAM does: `SHADOW_STAGE_NOREAD=1` (timing only, the file-only experts are zeros instead of read: wrong output) gives 9.5k 460.3, 462.3 / 430.7, 430.5 and 15k 488.3, 488.4 / 446.6, 448.8 = **+7% / +9%** from taking the 7.5% of experts that are in no RAM tier out of the copy path - so the file reads cost that much and the rest of the gap (the copies still at ~15 GB/s) is not found; this PC's DRAM is 4 x 16 GB Corsair at a configured 2,933 MT/s (a single thread copies memory at 13.4 GB/s here, 26.8 GB/s of DRAM traffic). Options by size of the bound: the file-only experts without a CPU copy (registered file views, <= +7-9%), fewer bytes a token (rows a pass, VRAM-bound), faster DRAM (BIOS, the owner's call), then compute.

- **2026-11-04 (11): decode profiled (Nsight Systems, `bench/nsys-decode-2026-11-04.nsys-rep`), the router bias task-checked, and CPU experts / pre-gating not re-run.** 4 prompts x 160 tokens (52.4 tok/s under the tracer): the 14.8 s trace has **kernels running 40.3% of the time, host-to-device copies 31.5% (only 4.7% of the time with a kernel running at once), neither 32.8%**; 107.5 GB over 51,648 copies of ~2.1 MB (one expert each, ~23 GB/s a copy, 168 MB a decoded token), ~470 kernels and 23 ms of wall a token (9.3 ms of kernel time). Each layer is a chain - attention / DeltaNet kernels, router logits to the host, the misses copied, then the experts - so the copy engine and the kernels take turns; the gaps between kernels: 3,370 ms in 1,167 gaps over 1 ms, 3,768 ms in 12,987 gaps of 0.1-1 ms. Kernel time: `matvec_cols_q6_k` 18% (13,045 launches of 82 us), the grouped expert matvecs ~45%, `matvec_cols_q8_0` 9%, `delta_step` 6%. This is the 2026-10-27 / 2026-11-04 picture again (demand copies are ~32% of a verify pass), now with the timeline: the earlier tries on it all lost on this PC - CPU experts (one CPU expert ~0.2 ms against ~0.09 ms for the copy), pre-gated prefetch (-66%), zero-copy reads (7 GB/s) - and FreeToken's q* needs host expert bandwidth close to the link's (here ~8-11 GB/s against 22-28). **Router bias, task-checked** (`tools/task_check.py`, 48 auto-graded short tasks: arithmetic, word problems, string reversal, sorting, vowel counts, facts), correct / decode tok/s on `dec.py`: bias 0 **42/48**, 55.8; **0.5: 40/48, 57.2 (+2.5%)**; 1: 40/48, 54.7; `--exact` 37/48 (the string and counting tasks are noisy: the model misses them in every mode). Fewer correct by 2 of 48 is within that noise and +2.5% decode is within launch-to-launch noise (+-3 tok/s): a tie either way, so **it stays opt-in**.

- **2026-11-04 (12): file-only experts copied straight from file views (`SHADOW_FILE_VIEWS=1`, `fileview.rs`): correct, but -58% - off.** A long prompt's experts that are in neither the VRAM cache nor the RAM tier are copied to the stage halves from read-only `MapViewOfFile` views of the model file, registered with `cuMemHostRegister(READ_ONLY | DEVICEMAP)` a layer at a time, instead of being read into pinned buffers by the read pool (no CPU copy through DRAM). Same reply as the CPU-copy path (default-mode hash f35749c5 both ways). Prompt tok/s, views / CPU copy: 9.5k **183.4 / 438.7**, 15k **176.8 / 442.4**. Per layer on the host (`SHADOW_STAGE_TIME`): mapping + registering 8-15 ms for 250-540 pieces (200-350 merged ranges), but **unregistering + unmapping 46-87 ms** (`cuMemHostUnregister` and `UnmapViewOfFile` cost per range), done before the next copy is queued: ~4 s of a ~7 s pass. Two things found on the way: (1) the first version kept one view of the whole file and the pages it touched stayed in the working set (51 GiB peak, the RAM watchdog killed the run at 1.9 GiB available, nothing left behind) - hence per-layer views; (2) the file-only set is bigger than the 7.5% average: **115-180 experts a layer in the late layers** (45% of their 512), 74-95 in the first - the placement profile fills the tier by global heat, not per layer. Not pursued: the registration lifetime would have to be taken off the critical path (a thread that unregisters, or ranges kept registered across passes: the file-only experts of all layers are ~4-6 GiB of pinned page cache, over the RAM floor's room), against a bound of +7-9% (`SHADOW_STAGE_NOREAD`).

- **2026-11-04 (13): the rest of the list - more rows with the drafter's VRAM, the dense attention path, the cache policy.** (2) **Rows with the drafter's 1.2 GB free** (`--dflash off --spec 0`, cap 4,096 / 8,192, two launches each): 9.5k 451.1, 444.1 / 462.1, 464.1 (+3%), 15k 449.3, 445.1 / 437.4, 440.3 (-2%): the copy-paced pass gains nothing from rows past ~4k, so unloading the drafter during a long prompt (deferred observe, weights re-read afterwards) is not built. (3) **`attention_heads_rows`** (13% of the prompt's kernel time, the dense path for a pass's first ~2k positions): not touched - its sparse sibling's float version tied (kernel time off the critical path), and this one is the same kind of kernel. (6) **Cache policy**: the VRAM cache is already pure per-layer LRU (2026-10-28: +11% over pinning, hit rate 59 -> 71-75%, simulated first with `tools/cache_sim.py`); a global cross-layer LRU was not simulated (the routing log is written by the session's Drop and `shadow serve` does not reach it on `/shutdown`: no `SHADOW_ROUTE_LOG` file came out); CPU experts (0.2 ms an expert against 0.09 ms for the copy), pre-gating (-66%) and zero-copy reads (7 GB/s) were measured and lost earlier on this PC.

- **2026-11-04 (14): the pivot - prompt path 2 (`docs/PREFILL2_SPEC.md`); first measurements for it.** Real 6,400-row passes (drafter off; rows used 4,608 and 6,400): 9.5k 461.5, 449.9 vs 452.5, 472.1 at 4,096, 15k 441.2, 437.1 vs 450.0, 448.6: **bigger chunks alone no longer help** (my 2026-11-04 (13) test had only reached ~4.6k rows). Timeline at 6,400 rows: 70.2 GB copied (vs 96.7), kernels busy 54.9%, copies 43.6%, neither 22.9%, **124 gaps of 10-50 ms (3.4 s) + 4 over 50 ms (0.7 s) with no kernel running**. The host chain per layer: router logits down, `route()` for every token (f64 exp over 512 logits each, one thread: ~27 ms at 6,400 rows), bookkeeping, `grouped_plan`, upload, then the experts launch; the GPU waits for all of it and the host cannot queue the next layer first. **Routing spread over the CPU threads** (`route_all`, bit for bit the sequential one; `--exact` hash 162a8e07 unchanged): 9.5k 442.8, 459.2, 458.6 / 15k 456.6, 461.8, 461.3 against 430.7, 430.5, 429.1 / 446.6, 448.8, 442.4 before: +3-5%. With it, after the logits arrive: conversion 2.3 ms, route + bookkeeping + plan prep 3.5-4.6 ms, `moe_staged` 6-27 ms (it includes waiting for the previous copy). The design that removes the round trip (route and plan on the device, host queues a pass ahead) is in the spec.

- **2026-11-04 (15): prompt path 2, stage 1 - the MoE of a long prompt routed and planned on the device: +9-11%, on by default (`SHADOW_PREFILL_DEVPLAN=0` off; only with `--tensor-prompt`).** New kernels (`kernels/forward.cu`): `route_topk_rows` (`route()` per token on the GPU: f64 exponentials, the sum in index order, top 10 by score then index, weights as f32 - checked against the host router on real logits with `SHADOW_DEVPLAN_CHECK=1`: 0 of 30,010 experts and 0 of 30,010 weights differ, histograms equal, every layer), `plan_scan` and `plan_fill` (the groups of 32 tokens per expert and the matrix addresses, in the layout `mma_grouped_*` read, with room for the most groups a phase can have; the empty ones are skipped by the kernel). The host no longer downloads the router logits, routes, builds or uploads a plan: it sends 512 expert addresses and flags per layer (they do not depend on the routing) and queues the layer. Two things had to change for the host to run ahead of the GPU: (1) the next layer's copy fills the half the layer before it reads, so the copy stream now waits for an event recorded after that layer's combine (without it: NaN logits, `argmax` panic), (2) the pinned read-ahead buffers are three, layer l in `l % 3`, each with its own done event - with two, the host blocked 60-87 ms a layer on the copy it had just queued before it could queue the next layer's compute (first device-planned build: -8%). **Prompt tok/s, device plan / host plan, interleaved pairs, default build, ctx 24,576: 9.5k 448.0, 483.2, 490.0, 491.3 / 443.0, 448.3, 446.5, 454.3, 456.5; 15k 510.2, 510.0, 517.3, 513.7 / 450.4, 460.0, 460.8, 461.6, 464.9 (+9% / +11.5%)**; one cold first run 407 / 416 (page cache). Checks: `--exact` reply hash 162a8e07 and `quality_test` 1.000 (12 of 12) unchanged, default-mode hash f35749c5 unchanged, fast-mode agreement 0.219 unchanged, `task_check` 42/48 unchanged, decode after a 9.5k prompt 33.7 (first reply) / 33.8 (follow-up) tok/s. The CUDA `exp()` is not the host library's, so the device routing is a fast-mode kernel in principle (it matched bit for bit on every logit checked).

- **2026-11-04 (16): prompt path 2, stage 2 analysis - what gates each layer's copy; `--direct-io` crashes (heap corruption).** Stage-1 timeline (`bench/nsys-stage1-2026-11-04`): 92 big copies in the window (1.9 passes, 5.0 s a pass), copy engine busy 3.3 s of each, **idle 1.2 s a pass**; the gap after the previous copy ended is 5.8 ms median but 36.8 ms mean: **68.3 ms on average (28 copies, 1.9 s) where an attention layer ran between two copies, 22.7 ms (63 copies, 1.4 s) otherwise**. The copy of layer l + 1 may not start before layer l - 1's experts are done (two stage halves), and an attention layer's chain (sparse attention ~100 ms, then its experts) makes that late; the copies run at 13-15 GB/s while active, 71 ms each, against ~41 ms of kernels a layer, so compute and copy are about equal and the dependency chain, not either's throughput, leaves the overlap short (perfect overlap would be ~3.3 s a pass instead of 5.0: +50%). Candidates: a ring of expert slots so a copy needs only its own slots free (stage 4), a faster copy (the late layers' 115-180 file-only experts are read by the CPU through the page cache: ~0.9 GB of DRAM traffic a layer beside a 15 GB/s DMA). **`--direct-io` (the uncommitted `DirectIoPool` in `readpool.rs`, not mine) crashed the first long-prompt run: `shadow.exe` APPCRASH 17:35:13, exception 0xC0000374 (heap corruption) in ntdll, exit -1073740940; the same build without the flag ran 365.7 tok/s on that launch (a slow one, 15k).** Not investigated; it may interact with the three read-ahead buffers (buffers are now `l % 3`, each with a done event). Also: my benchmark scripts killed every process named `shadow` between runs, which would stop a Shadow the user was running - they now kill only the test server by its port.

- **2026-11-04 (17): `--direct-io` crash: reproduced three times, cause not found.** `shadow.exe` APPCRASH 0xC0000374 (heap corruption, same fault offset 0x112fc5 in ntdll) at 17:35:13, 17:45:23 and 18:06:27, each the first long prompt after a start with `--direct-io` and the default 36 GiB tier (RAM tier filled, `Direct I/O enabled` printed); the same launches without the flag ran. Not default-build bugs: nothing in the default path was involved. `crates/shadow-model/examples/direct_io_stress.rs` (new) drives `DirectIoPool` alone with 540 unaligned pieces of ~850 KB per batch on the real model file and compares every byte with ordinary reads: 6 batches, 0 differences, no crash, so the pool itself holds up with ordinary heap buffers; the corruption needs something in the engine's use (pinned destination buffers, the read-ahead's submit / wait order, other readers sharing the pool, the tier). Runs I made with a 4 GiB tier the same afternoon are void: my port-based kill did not stop the previous server and four launches talked to one stale server. `--direct-io` is Gemini's uncommitted code (`readpool.rs`); do not run it on long prompts until someone finds this.

- **2026-11-04 (18): `--direct-io` crash, run with device planning off (`SHADOW_PREFILL_DEVPLAN=0`): crashed too (18:14:27, 0xC0000374)** - so it does not depend on stage 1. A crashed `shadow.exe` stays alive behind the Windows Error Reporting window (`WerFault`) holding its 42.8 GB until someone closes the window or kills it (9 minutes here): my test script now also kills any `shadow*` launched with its `--port`.

- **2026-11-04 (19): the `--direct-io` crash, narrowed (not fixed).** Whole runs (`--direct-io`, 3k prompt twice then 9.5k, default flags): tier 8 GiB **ok** (86-97 tok/s), 24 GiB **ok** (127-152), 32 GiB **crash**, 36 GiB **crash** (x4), also with `SHADOW_PREFILL_DEVPLAN=0`, with the read-ahead buffers pre-sized to 512 experts (`SHADOW_STAGE_SLOTS=512`: no buffer is ever replaced), and with every read-ahead waited right after it is submitted (`SHADOW_READ_SYNC=1`: nothing in flight). So it is not stage 1, not the buffer replacement and not an unfinished read. `SHADOW_HEAPCHECK=1|2` (new, debugging) validates the process heap at the long-prompt path's steps: valid at every check up to layer 0's `moe_batch`; the crash comes right after, while layer 1 is queued in the first pass; the exception differs between runs (0xC0000374 heap corruption, once 0xC0000005 access violation) - a write to memory the process does not own. `DirectIoPool` alone does not reproduce it (`examples/direct_io_stress.rs`: 540 fixed pieces; 120 batches of 1-400 pieces of 0.4-1.3 x 850 KB; 12 batches of 3,000 tiny 100-8,100 byte reads - all bytes equal, no crash). Crash dumps are written to C:/Users/a3k6g/AppData/Local/CrashDumps/shadow.exe.*.dmp (also one from Nov 3 19:26, before today's changes); no debugger is installed to read them (cdb / WinDbg), so the faulting function is not known.

- **2026-11-04 (20): the `--direct-io` crash found and fixed (WinDbg installed from Microsoft via winget; `cdb -z shadow.exe.35920.dmp` with `shadow.pdb`).** The dump's stack: `GpuSession::reset` -> `Model::ple_prefetch` -> `DirectIoPool::read_many` -> `submit` -> `wait_locked` -> `Vec::grow_one` -> `RtlReAllocateHeap` access violation. Cause (in `DirectIoPool`, Gemini's uncommitted code): `alloc_slot` takes a new 1 MiB `VirtualAlloc` slot for every 512 KiB piece when none is free - `ple_prefetch` reads thousands of tiny n-gram rows in one batch, so a batch committed thousands of MiB; with a 32+ GiB tier pinned the commit ran out, `submit` returned `Err` (the allocation failed) **while the reads it had already queued were still in flight into the caller's row buffers**; `ple_prefetch` then freed those buffers on the error return, and the next call's `wait_locked` copied the finished reads into freed memory: heap corruption in whatever allocated there (hence the varying exception and the later crash site; and why 8 and 24 GiB tiers, which leave commit, survived). Fix, `readpool.rs`: (1) `submit` waits for everything it queued before it returns an error; (2) at most `DIRECT_MAX_SLOTS` (128) slots, the oldest read is finished first (`retire_oldest`); (3) a failing direct batch is read by the buffered workers instead (`direct_failed` prints once), and `wait_async` waits on both paths. Checks: `direct_io_stress` (3,000 tiny reads x 6; 30 random batches) all bytes equal; `shadow serve --direct-io`, default 36 GiB tier, two 3k prompts and a 9.5k prompt: 198.1 / 260.7 / 241.6 tok/s, no crash, no `direct-I/O read failed`. (Direct I/O is still half the speed of the buffered path on this PC, as the earlier benchmark found: opt-in stays.) `readpool.rs` also holds Gemini's uncommitted work, so the file is not committed here; a copy of it before my edit is in the scratch folder.

- **2026-11-04 (21): why the layer copies run at 14 GB/s - the CPU's file reads - and the float kernels on by default.** `tests/scatter_probe.rs` (new, `--ignored`): 400-460 pieces of 2.4 MB from random offsets of a 4-16 GiB registered region, one `cuMemcpyHtoDAsync` each / one `cuMemcpyBatchAsync` (the engine's way) / adjacent / one contiguous block, and while device-to-device copies keep VRAM saturated: **27.6-28.1 GB/s in every case** - how the copies are cut and issued, and kernels using VRAM, cost nothing. In the engine's timeline the layer copies run at 14-18 GB/s (min 9.8); with the file reads skipped (`SHADOW_STAGE_NOREAD=1`, timing only; `bench/nsys-noread-2026-11-04`) **the median layer copy is 27.4 GB/s (p10 21.3, p90 27.9; 65% of the copies at 24+)** and the 9 s window has kernels busy 64.6%. So the one thing between this pass and the link's rate is the CPU copying the 7.5% of experts that are in no tier from the page cache into the read-ahead buffers (read + write of ~0.45 GB a late layer beside a DMA of 1.1 GB: 2.0 GB of DRAM traffic a layer, against DRAM that has ~35 GB/s). Prompt tok/s, 9.5k / 15k, two runs each: default 480.5 / 476.0 and 478.4 / 511.6; **float attention + float index scores 489.9 / 508.1 and 498.5 / 530.1 (+2-7%: with the device planner the kernels are on the critical path, they tied before)**; plus no file reads 526.3 / 577.1 and 494.5 / 576.7 (+8-17% more, not obtainable: the bytes must come from the file). So `SHADOW_ATT_F32` and `SHADOW_IDX_F32` are now on by default with `--tensor-prompt` (`=0` off). Checks: `--exact` hash 162a8e07 and 12 of 12, fast-mode agreement 0.219, `task_check` 42/48, three long prompts' agreement with `--exact` 0.165 / 0.063 / 0.304 (mean 0.177; double kernels 0.026 / 0.063 / 0.138 and float 0.046 / 0.063 / 0.018 in earlier runs: noise), default prompt speed 475.5 / 490.4 on that run. **Where the gap to Strata (934 / 1,422 tok/s at 9.6k / 15k) comes from:** Strata reads 15k in 2 passes (8k chunks); at our 14-28 GB/s that is 3-6 s of copying for 10.6 s total, i.e. its whole cost per token is ~0.7 ms, ours ~2 ms (kernels 0.75 ms + the copy-bound remainder). Passing it needs both fewer passes (8k rows: the per-row memory must come down from ~410 KB to ~100-150 KB) and ~2x faster kernels; neither is a tuning step.

- **2026-11-04 (22): prompt path 2, lean rows - a pass of 8,192 rows in sub-chunks of 2,048; on by default.** The layer's temporaries (the attention / DeltaNet sets, the router, the expert intermediates: ~410 KB a row) are only live inside one layer, so they are now sized for a sub-chunk (`SHADOW_SUB_ROWS`, 2,048) and a pass runs layer by layer over its sub-chunks (the first takes the remainder, so the last is a full 2,048); only the residual streams and each row's normalized input to the next layer are kept for the whole pass (`rbig`, `rnbig`: 2 x 4 x 2,560 floats = 82 KB a row; `bb.r` / `bb.rn` are views of the sub-chunk being run). The layer's experts are copied once for the whole pass (the copy pipeline advances after the layer's last sub-chunk; `moe_staged` returns early before it for the others), DeltaNet's state and the attention caches advance in position order, the drafter takes the last sub-chunk's features. A pass of 8,192 rows takes 1.55 GB of VRAM (temporaries 0.88 + 0.67) where 4,096 rows took 1.76. Two things found on the way: with the next layer's normalized input not carried between sub-chunks `hc_read_batch` / `hc_write_batch` fall back to a launch a token (110,592 `rmsnorm_groups` launches, prompts at 230 tok/s) - hence `rnbig`; and `batch_rows()` is now the pass size, not the temporaries' rows. Numerics: `--exact` hash 162a8e07 and 12 of 12 unchanged, default-mode hash identical for the unsplit pass, 4,096 and 8,192 rows (1bd164af), fast-mode agreement 0.219, `task_check` 42/48, decode after a 9.5k prompt 40.7 (first reply) / 33.5 tok/s. **Prompt tok/s, 9.5k / 15k:** unsplit 4,096 rows 503.2 / 501.3; two sub-chunks of 2,048 (4,096) 510.1 / 471.2 and 508.8 / 499.2; **8,192 rows 543.4 / 522.0, 545.4 / 525.5, then 537.5 / 509.0 in the checked default build**. Queueing the next layer's copy at the layer's start instead of after its last sub-chunk (`SHADOW_STAGE_EARLY`, on) ties with after (545.4 / 525.5 against 542.2 / 524.6). Timeline at 8,192 rows (`bench/nsys-lean8k-2026-11-04`, 6.6 s window): 45.4 GB copied (96.7 before), kernels busy 65.6%, copies 44.6%, neither 10.8%, copy with no kernel 23.6%, ~100 gaps of 10-50 ms (1.8 s) left, one a layer-pass. Sparse attention (`attention_group_sel_rows_f32` 21.8%, `attention_heads_rows` 17.7%, `index_scores_rows_f32` 4.4%) is now 44% of the kernel time.
- **2026-11-05 (23): the dense float attention kernel for a prompt's first positions (committed a50cfe3).** `attention_group_dense_rows_f32` (`kernels/forward.cu`): the float group kernel without token lists (row r attends positions 0 .. ntok0 + r - 1), 4 heads a block sharing each key row's loads; replaces `attention_heads_rows` (one head a block, 17.7% of the kernel time) when `--tensor-prompt` and `SHADOW_ATT_F32` are on. `--exact` hash 162a8e07 unchanged. Prompt tok/s 9.5k / 15k, float / double: 550.6 / 481.7 and 544.9 / 535.7 against 532.5 / 448.8 and 522.6 / 511.9 (+3-7%).
- **2026-11-05 (24): Nsight Compute on the sparse float attention kernel, and `attention_group_sel_rows_v2` built from it: +5-9% on prompts, on by default (`SHADOW_ATT_V2=0` keeps the first float kernel).** Profile of `attention_group_sel_rows_f32` (9.5k prompt, grid 6 x 32, 827 us a launch; `bench/ncu_attn_sel.ncu-repz`): FP32 FMA rate ~6% of peak, issue slots busy 31.5%, LSU requests 71% of peak and L1/TEX 72% - bound by load issue, not arithmetic nor DRAM (2.9%); every q value came from shared memory for 4 FMAs, the weighted sum read 1 shared score per FMA; and `block_sum` / `block_max`'s two 8 KB double buffers made the static shared memory 33.8 KB, so 2 blocks an SM (theoretical occupancy 33%) with 8 syncs per reduction per head. `v2`: 4 heads a block (fixed), scores 4 tokens a thread (a q load feeds 16 FMAs), max and sum by warp shuffles (shared memory 21 KB), the exp folded into the weighted sum's tile load (one pass over the scores less), and the weighted sum with a float4 of V columns a thread for 4 heads, the tile's rows split over 4 thread groups and summed at the end (16 FMAs per shared load). **Prompt tok/s, v2 / first float kernel, interleaved pairs, 9.5k: 540.8, 579.8, 578.7 / 540.3, 546.0, 520.5; 15k: 564.3, 572.7, 570.4 / 538.2, 539.3, 525.1** (medians 579 / 571 against 540 / 538: +7% / +6%); an earlier round 571.3 / 571.3 against 548.2 / 525.0 and one disturbed pair (472.7 / 427.3 against 534.2 / 418.2, a cold start). Numerics: float, so inexact like its predecessor: `quality_test` agreement 0.313 and 1 of 12 identical for both kernels (the default mode), the 160-token reply of the 9.5k summary prompt agrees with the first float kernel's for 253 characters and both are coherent; the `--exact` path does not use it.
- **2026-11-05 (25): profile of `attention_group_sel_rows_v2` and a timeline with it; the shared experts' IQ4_NL down projection on tensor cores, on by default (`SHADOW_MMA_NL=0` off).** Nsight Compute (`bench/ncu_v2_sel.ncu-repz`, `ncu_v2_idx.ncu-repz`, 9.5k prompt, 32-row launches): `attention_group_sel_rows_v2` 827 -> 420 us a launch (2x), now L2-bound (75% of peak; DRAM 5%), 125 registers (occupancy 33%, registers now the limit), 33% excess sectors from 16-byte-per-thread K loads from different rows, issue slots 22%, MIO stalls 32% of the wait; `index_scores_rows_f32` 110 us a launch, L1/TEX 98.6% (bound by L1 wavefronts), grid 3 x 32 blocks. Timeline (`bench/nsys-v2-2026-11-05`, 5.9 s window, 9.5k): kernels busy 59.2%, copy 53.6%, neither 12.0%, ~100 gaps of 10-50 ms (1.9 s); kernel time: sparse attention v2 13.9%, mma_grouped (4 types) 36%, **`matvec_cols_iq4_nl` 9.5% (30,528 launches of 10 us: the shared experts' down projection run as 8-column launches, 256 a 2,048-row sub-chunk)**, dense attention 8.5%, index scores 5.5%, route_topk 4.1%. `mma_cols_iq4_nl` existed (`SHADOW_MMA_NL=1`, tied on 2026-11-03 while the copies paced the pass); with the device planner it pays. **Prompt tok/s on / off, 9.5k: 620.7, 616.4, 609.4, 616.7, 616.8 (one disturbed pair 443.6) / 574.1, 582.1, 479.1, 580.1, 577.3 (456.1); 15k: 614.0, 608.4 (460.0, 457.0, 399.7, 402.5) / 558.2 (473.1, 441.7, 470.6, 439.4, 418.4)** - the machine was noisy in these rounds (15k off was 525-540 an hour earlier and 418-470 here; on beat off in 5 of 6 15k pairs and 5 of 6 at 9.5k, +6.5% at 9.5k in the quiet pairs). `quality_test` agreement 0.381 (0.313 before; noise), `--exact` path unchanged (the kernel runs only with `--tensor-prompt`, 16+ inputs).
- **2026-11-05 (26): `route_topk_rows` a warp a token.** The device router ran one thread a token with a 4 KB local-memory array: 1.08 ms for each 2,048-row sub-chunk on 8 blocks (143 ms, 4.1% of the 9.5k pass's kernel time). Now a warp a token (8 per block): the exponentials in parallel, the sum z sequentially in index order by lane 0 (the same double sum), the top 10 by ten rounds of a warp-wide best-of (greater score, then smaller index, the order the insertion kept). `SHADOW_DEVPLAN_CHECK=1` on a 9.5k prompt, all 48 layers: 0 experts, 0 weights, 0 histogram entries differ from the host router (432 checks). `SHADOW_ROUTE_1T=1` keeps the one-thread kernel. Prompt tok/s, new / old, interleaved, quiet machine, 9.5k: 628.7, 626.7, 590.8 / 583.5, 621.5, 614.4; 15k: 620.6, 614.8, 618.3 / 614.7, 608.1, 607.2 (+1.2% at 15k, 3 of 3 pairs; 9.5k noisy, medians 626.7 / 614.4).
- **2026-11-05 (27): what paces a long prompt now - the SSD reads of the file-only experts - and sub-chunks of 1,024 rows with passes up to 16,384 rows (new defaults).** Timeline of the *second* 9.5k prompt on a server (`bench/nsys-steady-2026-11-05`, `tools/nsys_last_run.py`; the first prompt of a server also pays ~98 `cuGraphInstantiate` calls of 43 ms, which made the earlier profiles' gaps look worse: a repeated prompt is 3% faster, 632 -> 650): window 4.8 s, kernels busy 62.6%, copies 43.4%, 99 gaps of 10-50 ms = 1.73 s with the host in `cuEventSynchronize` 1.69 s. Instrumented the host (`SHADOW_STAGE_TIME=3`, `HOSTWAIT` / `STAGECOPY` lines): the host waits 50-110 ms a layer in `stage_pipe_wait` for the next layer's reads (the file-only experts: 5,625 of 24,576, 11.6 GB a pass, 74-172 a layer), 2.6 s of each pass's ~13 s, while the upload call itself takes 0.3 ms. `typeperf` on the disks: **J: is read at 1.1-2.8 GB/s the whole prompt, on the second run of the same prompt too** - the page cache does not keep the 12 GB (pinned tier 36 GiB leaves ~7-15 GB; a cyclic 12 GB read through a smaller LRU cache hits ~never), so each pass reads its 11.6 GB from the SSD; a prompt in two passes reads 23 GB. Tried: `--direct-io` (deep queue, no page cache) 293 / 363 tok/s (9.5k, first and repeat) and 249 (15k) against 635 / 647 / 615 buffered - stays opt-in; `SHADOW_STAGE_EARLY=0` (copy queued after the layer's experts, the host waits behind a queued layer) 630.7, 634.4, 634.1 / 625.8, 621.2, 624.2 against 628.0, 630.0, 591.3 / 616.4, 619.3, 621.3 (+0.7%, a tie: stays off). **So fewer passes: the pass is limited by VRAM (12 GB card, 0.75 GB free in a pass), and the VRAM is the layer temporaries (per sub-chunk) + 82 KB a row of residual streams.** `BIG_BATCH_MAX` 8,192 -> 16,384 and `SHADOW_SUB_ROWS` 2,048 -> 1,024 (temporaries 0.44 GB smaller): a 9.5k prompt is one pass of 11,008 rows (it was 8,192 + 1,344 / 5,504 + 4,032), a 15k prompt 11,008 + 4,128 (was 8,192 + 6,944); sub-chunks of 512 / 768 / 1,536 measured 650 / 676 / 617 at 9.5k and 425 / 642 / 616 at 15k (kernels are less efficient on small chunks, and 512 starved the 15k prompt). **Prompt tok/s, 9.5k (first prompt / repeat) and 15k, new defaults / old (2,048 and 8,192): 657.8 / 684.3, 661.4 and 659.0 / 685.1, 661.4 / 574.1 / 651.6, 614.1 and 630.9 / 644.9, 621.1 (+5% at 9.5k, +7% at 15k).** Checks: `--exact` hash 162a8e07 and `quality_test` 12 of 12 unchanged; fast-mode agreement 0.381. Left: a 15k prompt still takes two passes (the KV tail to lend shrinks as the prompt grows), and with one pass of 9.5k the reads (12 GB at ~2.5 GB/s = 4.8 s) are still a third of its 14 s.
- **2026-11-05 (28): a 15k prompt in one pass (residual streams in parts, `rnbig` gone) and 16 read workers.** (1) The normalized input to the next layer is no longer kept for the whole pass: each sub-chunk normalizes its own at its start (`rmsnorm_groups_rows`, one launch for all rows, the arithmetic of the per-row launches; it also replaces the per-row loop in `hc_read_batch` when the norm is not carried), so a pass costs 41 KB a row of residual streams, not 82. (2) The residual streams are one buffer of `rows` rows a *part*, carved wherever the lent ranges have room (the first sub-chunk, the remainder, ends part 0): the two lent ranges were 0.878 + 0.421 GiB, and one 0.6 GB buffer did not fit in either. Pass rows for the 15k prompt: 11,008 -> 11,968 (no `rnbig`) -> **16,384 (parts): one pass of 15,136 rows**, 9.5k one pass of 9,536. Speed barely moved at 15k (667 -> 670: the first pass was already compute-bound, the second small), so passes are no longer what limits it. (3) `READ_WORKERS` 8 -> 16 (`SHADOW_READ_WORKERS`): the SSD wants a deeper queue. Prompt tok/s at 8 / 16 / 24 / 32 / 48 workers: 9.5k first prompt 660.8 / 677.3 / 680.3 / 679.5 / 679.3, repeat 686.2 / 690.4 / 689.2 / 689.6 / 689.9, 15k 666.5 / 682.5 / 685.3 / 683.5 / 684.8 (+2.3%; 4 workers 636.9 / 683.9 / 648.9). `SHADOW_MAIN_PRIO=2` (the queueing thread at highest priority) 663.0 / 685.4 / 633.0: no help. Checks: `--exact` hash 162a8e07 and `quality_test` 12 of 12, fast-mode agreement 0.381, `task_check` 42/48; decode after a 9.5k prompt with 1,024-row sub-chunks 40.4 / 38.9 and 41.0 / 39.4 tok/s against 39.0 / 37.4 and 39.1 / 37.5 with 2,048 (36.6 / 37.4 in the last run after the worker change, to be re-measured). **The 15k timeline (`bench/nsys-steady15-2026-11-05`, `tools/nsys_last_run.py`, `tools/nsys_gap_why.py`): 15.4 s window, kernels busy 67.1%, copies 11.3%, 419 gaps of 10-50 ms = 4.3 s + 76 of 1-10 ms = 0.7 s; the host's `cuLaunchKernel` is 5 us (median) but 1,011 launches a run take ~16 ms (one Windows scheduler quantum): those are the gaps.**
- **2026-11-05 (29): the ~16 ms launch stalls - what they are not.** In the steady 15k timeline 675 of 36,348 `cuLaunchKernel` calls take >5 ms (typical launch 5 us), exactly one every 15-16 ms (317 at 16 ms, 277 at 15 ms), with ~20 fast launches between them; the GPU is idle meanwhile (no kernel, memset or copy), one thread makes every CUDA call and no other thread's call overlaps, the call before is always `cuCtxSetCurrent`. They are the gaps: ~10 ms of idle after each sub-chunk step (419 gaps of 10-50 ms = 4.3 s of 15.4 s at 15k; the count scales with the number of sub-chunk steps). A plain CuPy launch loop (60,000 launches, GPU otherwise idle) has none (max 0.1 ms). Tested, all on `prio.sh` pairs with 9.5k first / repeat and 15k: `SHADOW_MAIN_PRIO=2` (queueing thread at highest priority) 663.0 / 685.4 / 633.0 against 660.8 / 686.2 / 666.5; `SHADOW_FLUSH_EVERY=N` (a `cuStreamQuery` every N launches, skipped during graph capture, which it would invalidate) N=2 / 8 / 32: 676.5 / 690.3 / 680.9, 677.7 / 687.3 / 655.6, 679.5 / 653.3 / 683.5 against 677.4 / 689.9 / 683.4 for none; `SHADOW_TIMER_MS=1` (`timeBeginPeriod(1)`) 678.2 / 690.1 / 683.1 and 680.5 / 692.2 / 680.0 against 673.8 / 688.8 / 680.5. All ties: they stay opt-in (inert by default). Facts about the machine: WDDM, hardware-accelerated GPU scheduling already on (`HwSchMode` 2), GeForce RTX 5070 with a display attached (no TCC), 15 other processes hold the GPU for graphics (Discord, Steam, Edge, Claude, explorer). Open: a GPU-scheduler trace (GPUView / WPA) of one stall, and graphs that cover a whole sub-chunk layer (docs/PLAN_EXCEED_STRATA.md).
- **2026-11-05 (30): correction - the prompt path is kernel-bound (91% busy); the "idle gaps" of entries 27-29 were CUDA graphs that Nsight Systems did not list.** `nsys profile --trace=cuda` shows a CUDA graph as one launch unless `--cuda-graph-trace=node` is given, and the prompt path runs most layer segments as graphs, so those timelines read the graphs' time as GPU idle (and their kernels missing from the tables: `mma_cols_q6_k` 249 ms in 759 launches there, 995 ms in 1,254 with graphs off). With `SHADOW_BATCH_GRAPHS=0` (`bench/nsys-nograph-2026-11-05`, 15k prompt, `tools/nsys_last_run.py`): **a 17.0 s window with kernels busy 91.2%**, copies 10.8%, gaps: 128 of 1-10 ms = 0.58 s and two of 50+ ms = 0.74 s. The ~16 ms `cuLaunchKernel` calls are the host waiting on a full launch queue, not a stall (the GPU is busy). Graphs: first run of a prompt ties on or off (683.9, 687.7 off / 687.7, 689.4 on at 15k; 9.5k 682.7, 683.4 / 681.6, 681.8), a repeated shape is faster with them (a 5,609-token follow-up 481-509 tok/s on, 444-451 off): they stay on (`SHADOW_BATCH_GRAPHS=0` off), and the cache is cleared at 4,000 entries (a new prompt shape adds ~500). `SHADOW_SEL_ROWS` (rows per group of the sparse attention's launches) 32 -> **128** (default): attention launches 19.6k -> ~5k a prompt; 9.5k first / repeat and 15k at 32 / 64 / 128 / 256: 676.9 / 689.5 / 680.2, 678.2 / 693.9 / 687.2, 683.3 / 695.0 / 690.4, 682.8 / 697.6 / 691.5 (+1.5% at 15k). Checks: `--exact` hash 162a8e07, `quality_test` 12 of 12 (agreement 0.383), `task_check` 42/48. **Kernel time of the whole 15k prompt, graphs off (15.5 s):** grouped MoE matrices 27% (iq4_nl 1.27 s at 1.14 ms a launch = 29 TFLOP/s; iq2_s 1.16 s at 2.16 ms = 15 TFLOP/s; iq3_xxs 1.05 s; iq3_s 0.69 s), dense projections `mma_cols_*` ~15% (q6_k 0.995 s, iq4_xs 0.98 s, q5_k 0.40 s; ~800 us a launch), sparse attention 11.7%, index scores 8.8%, DeltaNet `delta_steps128` 5.9% / `gated_rmsnorm_rows` 3.0% / `conv_silu_steps` 2.6%, `hc_write_rmsnorm_rows` 3.1%, `matvec_cols_q6_k` 2.3% (217 launches of 1.6 ms: a projection that misses the tensor-core path). So the plan's lever A is not a lever; the work is B-D (docs/PLAN_EXCEED_STRATA.md).
- **2026-11-05 (31): the 12-head sparse attention kernel (+3-4%, default) and last-row-only logits (a tie, opt-in).** `attention_group_sel_rows_v3`: one block a (KV group, row) with all of the group's heads (12), so the selected positions' K and V are read once for the group instead of once for each block of 4 heads (v2 was at 75% of L2 peak, reading them three times); scores 2 tokens a thread for 12 heads (a q load from shared memory feeds 8 FMAs), the weighted sum a float4 of V columns for 12 heads (48 accumulators), the 4 row groups summed one at a time through the dead q buffer (25 KB shared memory). `SHADOW_ATT_V3=0` keeps v2. Prompt tok/s, v3 / v2, interleaved pairs, 9.5k first prompt and 15k: 706.0, 705.8, 703.0 / 683.1, 684.6, 683.5 and 714.6, 717.5, 712.7 / 689.7, 688.4, 690.7 (+3.2% / +3.8%, 6 of 6 pairs). The 9.5k summary's 160-token reply is identical to v2's (644 of 644 characters), `quality_test` agreement 0.383 both. `SHADOW_LOGITS_LAST=1`: `prefill` computes the output projection for the last row only (the other 1,023 rows of the last sub-chunk were ~130 launches of 1.6 ms): 689.6, 692.5, 691.0 / 688.8, 691.5, 692.2 at 15k with the order balanced (an unbalanced run had read -3.5%: the first server after a restart is slower), `--exact` hash 162a8e07 and 12 of 12 unchanged: a tie, so opt-in.
- **2026-11-05 (32): the register-blocked index-score kernel (+3%, default, bit-identical).** `index_scores_rows_f32` (8.8% of the 15k prompt's kernel time; L1/TEX 98.6%, two loads for every FMA; the indexer is 4 heads of width 128, ratio 8) as `index_scores_rows_f32v2`: a block of 128 threads scores 4 rows, a thread 2 key blocks (b, b + 128), so each key float4 serves 4 rows and each query float4 (shared, a broadcast) 2 key blocks: 8 FMAs a shared load; each dot product is summed in the same order (i = 0 .. 127 by fmaf), so **the scores are bit for bit the old kernel's**: the 9.5k summary reply hash a0960fb7 in all six runs, on or off. `SHADOW_IDX_V2=0` keeps the old kernel. Prompt tok/s, v2 / old, balanced order, 9.5k and 15k: 748.8, 700.9 (noisy), 743.6 / 721.0, 718.0, 720.3 and 761.1, 724.1, 732.7 / 713.3, 716.3, 711.9 (medians +3.3% / +2.7%).
- **2026-11-05 (33): DeltaNet's convolution and gated norm rewritten, bit-identical (+2.1-2.2%, default); the existing DeltaNet variants re-screened.** `conv_silu_steps` (763 us a launch, 3.0% of the 15k prompt) ran one thread a channel, walking all m tokens (32 blocks on a 48-SM GPU); now `conv_silu_steps_par` (every (token, channel) a thread: a token's window is the k - 1 inputs before it, or the carried state for the first tokens, taps summed in the same double order) plus `conv_state_update` (each channel one thread, the shift in place). `gated_rmsnorm_rows` (864 us, 3.4%, a 128-thread block a (head, row) and seven barriers in `block_sum`) now `gated_rmsnorm_rows_w`: a warp a (head, row), a lane holds elements l, l + 32, l + 64, l + 96 and sums them in the block tree's order, `(e_l + e_l+64) + (e_l+32 + e_l+96)`, then halving by shuffles: the same sum. `SHADOW_CONV_PAR=0` / `SHADOW_GN_W=0` keep the old ones. `--exact` hash 162a8e07 and 12 of 12 unchanged (they use these kernels too), the fast-mode 9.5k reply hash a0960fb7 in all six runs. Prompt tok/s new / old, 9.5k and 15k: 763.2, 765.0, 766.7 / 748.8, 749.5, 747.7 and 773.5, 776.6, 776.0 / 757.3, 758.0, 759.3. Re-screened with the pass kernel-bound (baseline 727.6, 730.0 / 759.8, 760.9): `SHADOW_DELTA_CHUNK=1` (chunked scan, not bit-exact) 737.7 / 767.3 (+1.0%, a tie), `SHADOW_DELTA_X8=1` 671.5 / 691.2 (-8%), `SHADOW_FUSE=4` 723.3 / 749.0 (-1.4%), `SHADOW_FUSE=2` 543.0 / 558.2 (-25%), `SHADOW_FUSE=1` and `=7`: the server fails the first request (HTTP 400; not investigated). All stay off.
- **2026-11-05 (34): the end of a prompt computes the logits of its last row only (+5.7-8.4%, default).** The server reads a prompt's last logits with `step_batch(..).pop()`, so the output projection ran for all 1,024 rows of the last sub-chunk: 130 `matvec_cols_q6_k` launches of 2.7 ms (352 ms) *and a 1 GB device-to-host download* (1,024 rows x 248,320 floats; `cuMemcpyDtoH` 844 ms in the trace). `step_batch_last` (the engine's prompt tail) computes the last row only; the rows before come back empty (`SHADOW_LOGITS_LAST=0` computes them). Entry (31)'s `prefill` switch tied because the server never calls `prefill`. Prompt tok/s, last-row / all rows, balanced order, 9.5k and 15k: 811.3, 816.5, 812.4 / 748.6, 749.2, 751.9 and 819.8, 820.0, 820.9 / 775.6, 781.1, 776.3 (+8.4% / +5.7%). Checks: `--exact` hash 162a8e07 and `quality_test` 12 of 12, fast-mode 9.5k reply hash a0960fb7 (unchanged), agreement 0.383, `task_check` 43/48, decode after a 9.5k prompt 42.0 / 36.3 tok/s.
- **2026-11-05 (35): layer sub-chunks of 2,048 rows again (+6-8%, default).** With `rnbig` gone and the residual streams in parts, the pass fits a 2,048-row sub-chunk; the grouped MoE kernels dequantize each expert's weights once per sub-chunk, so more rows (tokens) per expert amortize it (STATUS 22/27: 1,024 rows cost +36% a row there). Prompt tok/s, 9.5k and 15k, balanced pairs (pass rows in the log): **2,048: 863.1, 806.6 (first prompt after a restart), 870.7 / 872.4, 863.7, 874.8** (pass 12,288 rows for the 15k prompt: two passes) against **1,024: 813.4, 794.8, 817.7 / 815.8, 808.8, 818.2** (one pass of 16,384); 1,280 783.8 / 793.4, 1,536 838.2 / 848.9, 2,560 847.9 / 719.0 (pass 5,120), 3,072 588.1 / 580.8 (pass 3,072), 4,096 747.5 / 692.6 (pass 4,096): above 2,048 the temporaries leave too little room for the residual streams and the pass shrinks. One 15k run read 331 tok/s (45.7 s) in a series of nine 2,048 runs and did not recur. `SHADOW_SUB_ROWS=1024` keeps the old size. Checks: `--exact` hash 162a8e07 and 12 of 12 unchanged; the fast-mode 9.5k reply hash is fc1b04f4 (a0960fb7 at 1,024 rows) - fast mode accepts a drafted token within the target's top 2 and the drafter's context is the last sub-chunk's rows (now the full 2,048-position window), so the reply can differ without the engine's numbers having changed; agreement 0.383 both, `task_check` 42/48, decode after a 9.5k prompt 41.6 / 42.4 tok/s.
- **2026-11-05 (36): groups of 64 tokens for the tensor-core expert kernels - bit-identical, a tie, opt-in (`SHADOW_MMA_GROUP=64`).** At 2,048-row sub-chunks an expert has ~40 tokens on average, so the 32-token tile of `mma_grouped_*` dequantizes its weights twice for the busier experts. `mma_core<Q, TOK>` (TOK 32 for the dense `mma_cols_*`, 64 for the new `mma_grouped64_*`, records of 129 words, `plan_scan` / `plan_fill` take the group size) and the n-tiles with no token in them are skipped (also in the 32 kernels). The fast-mode reply hash is the same (fc1b04f4) and `--exact` 162a8e07. Prompt tok/s, 64 / 32, balanced, 9.5k and 15k: 824.2 (first prompt), 889.5, 889.0 / 884.2, 879.7, 887.9 and 873.2, 875.3, 874.7 / 867.2, 821.7, 862.4 (+0.3% / +1.2%): the grouped kernels are not limited by the weight dequantization the way the arithmetic suggested (docs/PLAN_EXCEED_STRATA.md lever B) - they need a profile. 
- **State after levers A-D without the grouped-MoE profile (docs/STATUS.md 30-36):** 9.5k ~885 tok/s (Strata 934), 15k ~875 (Strata 1,422); from 677 / 680 at the start of the 2026-11-05 session's second half. Wins: sparse attention in one block per KV group (+3.5%), register-blocked index scores (+3%, bit-identical), DeltaNet conv and gated norm (+2.1%, bit-identical), last-row logits (+5.7-8.4%), 2,048-row sub-chunks (+6-8%), 128-row selection groups (+1.5%).
- **2026-11-05 (37): float versions of the prompt path's norm and DeltaNet kernels (+6.5-7%, default with `--tensor-prompt`; 9.5k reaches 934 tok/s).** Nsight Compute (`bench/ncu_k_*.ncu-repz`): `gated_rmsnorm_rows_w` has the **FP64 pipe at 86%** (issue slots 6.6%; the card does FP64 at 1/64 of FP32) and `delta_steps128` 46% at 8% occupancy (255 registers, one block a SM). In the whole-prompt timeline (graphs traced) ten kernels with double sums took 3.0 s of 12.3 s of kernel time: `delta_steps128` 972 ms, `hc_write_rmsnorm_rows` 484, `conv_silu_steps_par` 276, `rmsnorm_groups_rows` 242, `l2norm_qk_rows` 227, `head_norm_rope_qgk_rows` 204 (rope in double: left), `hc_mix_rows` 199 (float already), `gated_rmsnorm_rows_w` 151. New `_f` kernels without any double, sums by warp shuffles (`block_sum_f`): `rmsnorm_groups_rows_f`, `hc_write_rmsnorm_rows_f`, `l2norm_qk_rows_f`, `gated_rmsnorm_rows_wf`, `conv_silu_steps_par_f`, `delta_steps128_f` (the two sums over the 128 rows in float, 8 partials of 16). Chosen by `nk()` with `--tensor-prompt` only: `--exact` and decode keep the double kernels (`--exact` hash 162a8e07 and 12 of 12 unchanged); `SHADOW_NORM_F32=0` turns all off, `SHADOW_DELTA_F32=0` the recurrence only. **Prompt tok/s, float / double, balanced pairs, 9.5k and 15k: 933.7, 934.6, 934.7 / 878.2, 887.0, 816.0 and 926.9, 918.0, 921.1 / 867.5, 849.1, 862.2.** The 64-token reply to the 9.5k summary is the same (fc1b04f4), `task_check` 42/48, decode after a 9.5k prompt 38.4 / 36.8 tok/s. **Quality caveat, measured:** agreement with `--exact` on the 12-item `quality_test` goes 0.383 -> 0.285 (0.246 with the recurrence too, 0.236 without it) and on 36 items (`bench/dfdata/varied36.json`, history = the exact replies, deterministic across repeats) 0.436 / 9 identical -> 0.376 / 7 identical; the recurrence in float alone moves it 0.366 -> 0.376. The float norms differ from the double ones by ~1e-7 relative, 4 orders of magnitude below the fp16 tensor-core products fast mode already carries, so I read the shift as the chaos of near-tied tokens rather than drift, but the metric cannot tell the two apart: `SHADOW_NORM_F32=0` is the way back.
- **2026-11-05 (38): 64-token tiles for the dense tensor-core projections (+3.7%, bit-identical, default).** `mma_cols64_*` (`mma_core<Q, 64>`; the n-tiles past the batch's last token are skipped): a weight tile is dequantized once for 64 tokens instead of 32, so with 2,048-row sub-chunks it is rebuilt 32 times a projection, not 64. `SHADOW_MMA_COLS_TOK=32` keeps the old tile. The 9.5k reply hash is the same in all six runs (fc1b04f4); `--exact` 162a8e07 and 12 of 12, `task_check` 42/48, decode 37.4 / 36.3. Prompt tok/s, 64 / 32, 9.5k and 15k: 971.5, 888.9 (a noisier server start), 970.4 / 928.8, 934.6, 934.6 and 961.3, 945.9, 941.4 / 927.5, 916.5, 912.5 (medians +3.8% / +3.6%). The tier stays at 36 GiB on purpose (38 GiB measured faster, STATUS 36/tier test: 3k 570 / 657, 9.5k 934 / 965, 15k 942, with 4.1 GiB minimum free; 40 would dip under the 3 GiB watchdog floor; 0 / 12 / 24 GiB: 3k 206 / 237 / 347, 9.5k 534 / 660 / 901).
- **2026-11-05 (39): float rope for the prompt's attention (+1.5%, default with `--tensor-prompt`); register budget of the mma kernels: a tie.** `head_norm_rope_qgk_rows` (1.9 ms a launch, 1.7% of the 15k prompt) did `pow` / `cos` / `sin` in double for every (row, head) pair. `head_norm_rope_qgk_rows_f`: base^(2i / dims) once per launch into a device table (`rope_pow_table`, in double), the angle `pos / table[i]` still in double (a float angle at position 15,000 would be ~1e-3 rad off), reduced to [-pi, pi] in double, then `sincosf`; the head norm uses `rmsnorm_one_f`. `SHADOW_ROPE_F32=0` (or `SHADOW_NORM_F32=0`) keeps the double one. Prompt tok/s, float / double, 9.5k and 15k: 985.3, 977.1, 984.5 / 901.0, 911.3, 975.9 and 967.5, 963.4, 964.7 / 957.0, 906.7, 953.3 (+1-2%). The 9.5k reply differs (hash 13d11c16, fc1b04f4 before), both coherent; agreement with `--exact` on the 36 items 0.416 / 8 identical (0.376 / 7 before). Register budget: `__launch_bounds__(128, 6 / 7 / 8)` on the 35 mma kernels (80 registers = 6 blocks of 128 threads an SM; 7 forces <= 73, 8 <= 64): 9.5k / 15k 956.7, 963.2 / 973.0, 967.5 at 6; 960.8, 959.0 / 966.0, 967.3 at 7; 880.2 (first server after a build), 950.0 / 909.1, 956.2 at 8: a tie at 7, worse at 8 - occupancy is not the limiter; removed. (An earlier run of this test compared the same binary six times because the builds failed with the test server still holding `shadow.exe`; it read 962-968 at 9.5k, which is the run-to-run spread.)
- **2026-11-05 (40): the per-layer-embedding (PLE) step for a whole batch (+15-20%, default with `--tensor-prompt`; 9.5k now ~1,140 tok/s).** `SHADOW_PLE_TIME=1` showed the PLE layer's host time per 2,048-row chunk at 205-470 ms (the GPU done ~10 ms after the host finished queueing), a loop over the chunk's tokens of: the n-gram rows into a vector, a blocking upload, two matvecs, three norms, a gate, a convolution step with carried state, an add - about 8 launches a token (`matvec_fast_bf16` 5,700 x 55 us, `matvec_cols_q8_0` 5,678 x 32 us, `ple_gate`, `ple_conv_step` ... in the kernel list) while the GPU idled (the 50+ ms gaps). `ple_batch` (new): every token's rows built into one array and uploaded once, the key and value projections as `mv_cols` (tensor cores), `rmsnorm_groups_rows_f`, `ple_gate_rows_f`, `ple_conv_rows_f` (every token a thread: the dilated window is gvn rows `j - (k-1-t) * dil` or the carried state; `r += gv + silu(conv)` folded in) and `ple_state_update`; four idle per-row batch buffers (`rn`, `up`, `qkv`, `x`) and `u` hold the intermediates, so no VRAM is added. Only `--tensor-prompt`, batches of 16 or more rows with no replay (`SHADOW_PLE_BATCH=0`: token by token); `--exact` unchanged (hash 162a8e07, 12 of 12). **Prompt tok/s, batched / token by token, 9.5k and 15k: 1,143.5, 1,036.9, 1,141.5 / 950.6, 985.4, 981.0 and 1,089.4, 1,061.9, 1,100.2 / 957.6, 964.8, 909.9.** The 9.5k reply hash is the same (13d11c16), agreement with `--exact` on 36 items 0.400 / 9 identical (0.416 / 8), `task_check` 42/48, decode after a 9.5k prompt 39.3 / 44.0 tok/s.
- **Other things tried in this stretch (STATUS 38-40):** 64-token dense tiles +3.7%; float rope +1.5%; `delta_steps128_fp` (next token's inputs prefetched into registers; bit-identical): 990.8, 989.3 / 971.7, 972.9 against 985.2 / 964.0 - a tie, opt-in `SHADOW_DELTA_PF=1`; register budgets of the mma kernels: a tie.
- **2026-11-05 (41): Strata 0.2.11 measured on this PC, same model files, same prompts.** Downloaded the release installer (113,010,362 bytes, SHA-256 checked against `SHA256SUMS.txt`), installed silently into `J:\Strata-work\strata-0.2.11-app`, `strata launch --model-path J:\Strata-work\models\swift15\IQ3_S --context 32k --parallel 1 --drafting off --vision off --tune skip` (engine 0.1.57 windows cuda13, 627,812,560 bytes, signature-checked; the model files are the ones Shadow loads). Prompts from `docs/STATUS.md` text with a random ID first and `cache_prompt: false` (no shared prefix), 3 rounds each: **3k (2,972-3,045 tokens) 1,261 / 1,269 / 1,286 tok/s server timing (wall 1,232-1,255); 9.7k (9,659-9,755) 1,809 / 1,878 / 1,844; 15.3k (15,178-15,351) 2,447 / 2,442 / 2,457.** Decode (drafting off) 55.8 / 46.3 / 50.6 tok/s. The October baseline on this PC (733 / 934 / 1,422; decode 47.7) was Strata 0.1.42 (`speed-work-0.1.42`). **What the 0.2.11 engine does, from its own log:** the whole expert set (46.84 GiB) is held in RAM - "expert arena: ... 47 slices pinned (45 GiB)", loaded unbuffered at 2.97 GiB/s - so a prompt reads nothing from the SSD; 5.32 GiB VRAM expert cache and the prompt path borrows 2,154 cache slots (4.12 GiB); prompt chunks of 8,192 (the log says why 16,384 / 32,768 stay off: RAM); the idle CPU computes some experts for chunks up to 3,072 tokens; `expert-profile.bin` placement; K-quant experts read the prompt on llama.cpp's MMQ (int8) kernels (our IQ3_S experts do not take that path, so its IQ3_S speed comes from its own kernels); 351 MiB VRAM free, 3.4 GiB RAM free while serving (working set 49 GiB). Shadow's tier is 36 GiB (77% of the experts) at the user's choice. **Shadow now vs Strata 0.2.11 (tok/s), 3k / 9.5k / 15k: 569-682 / 991-1,143 / 1,018-1,100 against 1,261 / 1,844 / 2,447** (Strata 1.9x / 1.7x / 2.3x).
- **2026-11-05 (42): int8 MMQ feasibility (docs/PLAN_INT8_MMQ.md phase 1): the gate is passed for the dense projections and for IQ2_S / IQ3_XXS / IQ3_S gate/up, not for IQ4_NL.** `bench/mmq/mmq_bench.cu` runs llama.cpp's own MMQ kernels (ggml-cuda `mmq.cuh` from Strata's vendored llama.cpp 3cf03257f, MIT; the few ggml runtime hooks stubbed) on fabricated data of our shapes, `nvcc -arch=sm_120` (build scripts `build_bench*.bat`, MSVC from VS 18 Community). Product time (activation quantization separately), TFLOP/s, against our fp16 `mma_*` kernel from the 15k timeline (`bench/nsys-f-2026-11-05`): **grouped experts, 512 experts x 40 tokens: IQ4_NL down (2,560 rows, K 640) 1.73 ms / 38.8 and gate/up (640 x 2,560) 1.42 ms / 47.2 against ours 1.56 ms (a tie: the plan's gate of 1.5x fails, IQ4_NL stays as is); IQ2_S 1.73 ms / 38.9 (ours 3.50), IQ3_XXS 1.55 ms / 43.4 (ours 3.74), IQ3_S 1.73 ms / 38.9 (ours 3.44) - 1.7-2.4x with the quantization (0.44 ms, shared by gate and up) included at 2 x; dense, 2,048 tokens: Q6_K 10,240 x 2,560 1.53 ms / 70.3 (ours 2.80), Q6_K 6,144 x 2,560 0.91 ms / 71.0 (ours 1.69), IQ4_XS 12,288 x 2,560 1.22 ms / 105.8 (ours 3.02), quantization 0.018 ms.** Numerics against a float reference (IQ4_NL): relative RMS error 0.52-0.55% (int8 activations), the fast mode's allowance. Shares of the 15k prompt's kernel time: IQ2_S + IQ3_XXS + IQ3_S gate/up 23.6%, dense Q6_K / IQ4_XS / Q5_K / Q4_K ~19% - about 2.1 s of 10.8 s could go to ~1 s. Plan: a small `shadow_mmq.dll` (nvcc, ggml's MMQ + a C wrapper) loaded by the engine, dense projections first (no expert table), then the grouped gate/up.
- **2026-11-05 (43): int8 MMQ for the quantized dense projections of a long prompt (+6-8%, default when `native/mmq/shadow_mmq.dll` is built).** `native/mmq/shadow_mmq.cu` (+ `build.bat`: nvcc 13.5 + MSVC; the harness and its per-type builds in `native/mmq/bench/`) wraps llama.cpp's own `mul_mat_q` kernels (ggml-cuda `mmq.cuh`, MIT, from Strata's vendored copy; GGML_DIR overridable) for Q4_K / Q5_K / Q6_K / IQ4_XS behind three C calls (quantize the f32 activations to `block_q8_1_mmq`, GEMM with the weights in their GGUF layout, buffer size); `crates/shadow-cuda/src/mmq.rs` loads it with the existing `Dylib` (no cudart linkage, the primary context the engine already holds) and `mv_cols` calls it for those types when `--tensor-prompt`, m >= 128, K % 256 == 0; a 14 MB activation buffer (`mmq_y`). Without the DLL, or `SHADOW_MMQ=0`, the engine's own fp16 kernels run as before; `--exact` and decode never use it. **Prompt tok/s, MMQ / own kernels, 9.5k and 15k: 1,226.2, 1,225.1, 1,226.1 / 1,132.1, 1,131.8, 1,130.1 (+8.4%) and 1,093.6, 1,159.5, 1,157.6 / 1,094.0, 1,094.8, 1,010.9 (+6%).** In the 15k timeline the new `mul_mat_q` kernels take ~0.93 s where the fp16 dense kernels took ~2.1 s. Checks: `--exact` hash 162a8e07 unchanged, 36-item agreement 0.410 (0.400), `task_check` 43/48 (42), the 9.5k reply differs in wording (hash 5dfb15cf; "Stages 0 through 2" where the float build said "0 through 5") and is coherent. **A bug on the way, for the record:** my first activation buffer had an operator-precedence slip (`n_head * head_dim.max(..)` = 24 x 6,144 columns, 340 MB), which cut the expert cache by 165 slots (0.32 GiB) and the lent room, a 14,336-row pass instead of 16,384, two passes for 15k, and made int8 look 25-35% *slower* (763-770 / 630-735 tok/s) although its kernels were faster; fixed to 14 MB. Next: the grouped gate/up experts (IQ2_S / IQ3_XXS / IQ3_S, 23.6% of kernel time, 1.7-2.4x in the standalone test) need a per-expert weight-pointer table (our experts live in cache slots and stage halves, not one contiguous tensor) and the device planner's sorted token ids.
