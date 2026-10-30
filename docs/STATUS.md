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

- New bench: 36 turns of unseen alpaca/code-alpaca prompts (`bench/dfdata/varied36.json`, `shadow-bench --prompts-file F --turns 36`)
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
