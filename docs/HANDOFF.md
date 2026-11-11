# Handoff

Updated 2026-11-10 (`v0.1.36`). Read this section first, then [STATUS.md](STATUS.md) (entries 1–95, chronological evidence) and [ARCH.md](ARCH.md) (the model as math).

---

## 2026-11-10 Agent Handoff & Current State (`v0.1.36`)

### 1. Current Performance Scoreboard (Shadow `v0.1.35` vs Pre-Opt `v0.1.32` and Strata `0.2.11`)

Measured via interleaved A-B-B-A on fresh servers (`--ctx 24576 --ram-tier-gib 45 --ram-floor-gib 3.5`, `tools/ab_prompt_opt.py`, `bench/prompt-opt-2026-11-10/results_r2.json` & `results_r3.json`, all reply SHA-256 hashes bit-identical across arms):

| Workload | Tokens | Pre-Opt (`0.1.32`) | Round 2 (`0.1.33`) | Round 3 (`0.1.35`) | Strata `0.2.11` | Shadow `0.1.35` vs Strata |
|---|---:|---:|---:|---:|---:|---|
| **`3k_a` (first prompt)** | 2,867 | 3.42 s (838 tok/s) | 3.12 s (919 tok/s) | **2.99 s (958 tok/s)** | ~2.40 s (1,210 tok/s) | `-19%` (remaining ~200 ms is KV/cache lend + restore) |
| **`3k_b` (warm prompt, 45 GiB)** | 2,622 | 2.72 s (962 tok/s) | 2.62 s (1,001 tok/s) | **2.50–2.62 s (~1,001–1,048 tok/s)** | ~2.40 s (1,210 tok/s) | Within ~0.1–0.2 s |
| **`3k_b` (at 43 GiB trimmed tier)** | 2,622 | 3.82 s (687 tok/s) | 3.05 s (860 tok/s) | **2.93 s (896 tok/s)** | — | `+23.4%` vs `0.1.32` |
| **`9k6` (long prompt)** | 9,673 | 5.89–6.09 s (1,588–1,643 tok/s) | 5.73 s (1,689 tok/s) | **5.68–5.71 s (1,694–1,702 tok/s)** | 5.50 s (1,735 tok/s) | Parity (within 3%) |
| **`15k` (long prompt)** | 15,034 | 8.70–9.05 s (1,661–1,729 tok/s) | 8.57 s (1,754 tok/s) | **8.52–8.54 s (1,761–1,764 tok/s)** | 9.90 s (1,543 tok/s) | **Shadow +16.2% faster (`-1.38 s`)** |
| **`decode200` (speculative decode)** | 200 | 53.2 tok/s | 52.9 tok/s | **52.9–53.1 tok/s** | 47.7 tok/s | **Shadow +11.1% faster** |

---

### 2. Where Prompt Wall-Clock Time Goes (`SHADOW_PHASES=1` Breakdown in `v0.1.35`)

With all Round 2 and Round 3 optimizations active (`bench/prompt-opt-2026-11-10/r3_r2.err`):

- **3k prompt (`2,867` tokens, `2.992 s` total)**:
  - `lend_cache`: **`0.040 s`** (KV tail VMM remap from pool + device drain; down from `0.316 s` in `0.1.32` and `0.091 s` in `0.1.33`).
  - `step_batch` (`m = 2867`, 2 sub-chunks of `819 + 2,048`): **`2.786 s`** (`embed_rows` `0.015 s`, 48 layers queued by `2.539 s`, GPU drained at `2.671 s`, DFlash observer `0.098 s`).
  - `restore_cache`: **`0.166 s`** (`unlend_kv` VMM remap `0.031 s`, overlapped cold-file + `3.2 GiB` RAM-tier `cuMemcpyBatchAsync` DMA `0.135 s`; down from `0.259 s` in `0.1.32` and `0.205 s` in `0.1.33`).
- **9.6k prompt (`9,673` tokens, `5.685 s` total)**:
  - `lend_cache`: **`0.027 s`** (down from `0.124 s` in `0.1.32`).
  - `step_batch` (`m = 9673`, 5 sub-chunks): **`5.503 s`** (`embed_rows` `0.043 s`, layers queued by `5.118 s`, GPU drained at `5.360 s`, DFlash observer `0.098 s`).
  - `restore_cache`: **`0.154 s`** (`unlend_kv` `0.018 s`, overlapped cold + RAM-tier DMA `0.136 s`; down from `0.246 s` in `0.1.32`).
- **15k prompt (`15,034` tokens, `8.524 s` total)**:
  - `lend_cache`: **`0.020 s`** (down from `0.113 s` in `0.1.32`).
  - `step_batch` (`m = 15034`, 8 sub-chunks of `698 + 7 × 2,048` in a single pass): **`8.352 s`** (`embed_rows` `0.071 s`, layers queued by `8.020 s`, GPU drained at `8.210 s`, DFlash observer `0.070 s`).
  - `restore_cache`: **`0.151 s`** (`unlend_kv` `0.016 s`, overlapped cold + RAM-tier DMA `0.135 s`; down from `0.220 s` in `0.1.32`).

---

### 3. Active Optimizations Inventory (Defaults & Opt-Out Switches)

All prompt and decode optimizations below are **on by default** in `shadow serve` (fast mode, `--tensor-prompt`):

1. **Chunked 45 GiB Pinned RAM Tier + Watchdog (`crates/shadow-cuda/src/pinned.rs`, `session.rs`, STATUS 81–85, 90, 93, 94)**:
   - `fastest_defaults` sizes `--ram-tier-gib` up to `45 GiB` (`23,643` of `24,576` experts pinned in 1 GiB `VirtualAlloc` + `cuMemHostRegister` chunks) with `--ram-floor-gib 3.5`.
   - **Important (`v0.1.34`–`v0.1.35`)**: All hard RAM-floor refusals during runtime (`PinnedRegion::new`, `PinnedRegion::new_chunked`, `sp_spare` retention, `expand_pictures`) were converted to log warnings (`RAM warning: ...`). The background watchdog (`pinned::start_watchdog`, polling every 250 ms) automatically asks `GpuSession::ram_pressure_poll` to decommit 1 GiB chunks from the tail of the RAM tier whenever available RAM dips below `ram_floor()` for 2 consecutive samples.
2. **Fused INT8 Tensor-Core MoE Kernels on All 48 Layers (`native/fused/shadow_fused.dll`, `crates/shadow-cuda/src/fused.rs`, STATUS 86–89)**:
   - Covers all 48 MoE layers (`IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS` gate/up × `IQ4_NL`, `Q2_0` down). `-8.8%` on 15k prompts.
   - Opt-out: `SHADOW_FUSED_MOE=0`.
3. **Fused-Expert Fast Device Planner (`crates/shadow-cuda/src/session.rs` `dev_plan_buckets`, `fused_experts`, STATUS 93)**:
   - Skips `zero_words` (`2.9 GB` of memset per 15k prompt), `plan_scan_w`, and `plan_fill_w`, uploads only 4 KB of `dp_eaddr`, quantizes `bb.u` once across phases 0 and 1, and skips `fu.mask` when `!any_res`.
   - Opt-out: `SHADOW_FUSED_FAST_PLAN=0` or `--no-fused-fast-plan`.
4. **Sub-Chunk (`si > 0`) Fast Path & Layer-0 Pipeline Ordering (`crates/shadow-cuda/src/session.rs` `moe_staged`, `dev_plan_buckets`, STATUS 94)**:
   - On sub-chunks `si > 0`, skips rebuilding/uploading `dp_eaddr` and `dp_copied`, skips `stream_wait_event(sp.events[p])`, and runs a single unmasked `fused_experts` phase (since all 512 experts of layer `l` are already in VRAM after `si == 0` waited on `sp.events[p]`).
   - On `first_of_pass` (`l == 0`), defers `early` (`stage_copy(1)` + `stage_pipe_read(2)`) until after sub-chunk 0's `moe_combine_qwen_rows` so Layer 0 sub-chunk 0 runs on the GPU while `stage_pipe_read(1)` finishes on the CPU read pool.
   - Opt-out: `SHADOW_SUBCHUNK_FAST=0`.
5. **VMM Physical Handle Pool & Scratch Zero-Skip (`crates/shadow-cuda/src/kvvmm.rs`, `session.rs` `lend_kv`/`unlend_kv`, STATUS 94)**:
   - Pools the ~650 2 MiB `CUmemGenericAllocationHandle`s in `GpuSession::vmm_pool` across `lend_kv` and `unlend_kv` instead of calling `cuMemRelease` and `cuMemCreate` ~2,600 times per prompt, and skips `cuMemsetD8` when mapping `kv_extra_reg`. Saves `70–90 ms` per prompt.
   - Opt-out: `SHADOW_VMM_POOL=0`.
6. **Startup Pre-Allocated & Retained `StagePipe` and `staging` Buffers (`crates/shadow-cuda/src/session.rs`, STATUS 93)**:
   - Pre-allocates `staging` (`502 MiB`) and `sp_spare` (3 pinned buffers, `546–990 MiB`) during `GpuSession::with_tiers` and retains `sp_spare` across prompts in `restore_cache`. Saves `~225 ms` on the first prompt and `~130 ms` on every subsequent prompt.
   - Opt-out: `SHADOW_KEEP_PIPE=0`.
7. **Batched & Overlapped `restore_cache` (`crates/shadow-cuda/src/session.rs` `restore_cache`, STATUS 93, 94)**:
   - Uses `cuMemcpyBatchAsync` (`SHADOW_RESTORE_SINGLE=1` reverts to single copies) and queues the ~3.2 GiB RAM-tier `upload_batch` onto `copy_stream` *before* reading cold file-only evicted experts into `sp.bufs[0]`, overlapping the CPU/NVMe read with PCIe DMA (`SHADOW_RESTORE_OVERLAP=0` reverts to sequential).
8. **Deduplicated `shadow_mmq_quantize` on `bb.u` (`crates/shadow-cuda/src/mmq.rs`, `session.rs` `mv_cols`, STATUS 94)**:
   - Caches `(x_ptr, k, m, xs, ds_layout)` in `GpuSession::mmq_y_tag` (invalidated when `bb.u` is written) so consecutive MMQ projections on `bb.u` (`ffn_gate_shexp` + `ffn_up_shexp`, `attn_qkv` + `attn_gate`, `attn_q` + `attn_k` + `attn_v`) quantize `bb.u` once instead of 2–3 times (`~576–768` fewer launches per 15k prompt).
   - Opt-out: `SHADOW_MMQ_REUSE=0`.
9. **Parallel Host Token-Embedding Dequantization (`crates/shadow-cuda/src/session.rs` `batch`, STATUS 93)**:
   - Parallelizes `Q6_K` host embedding dequantization across CPU threads via `std::thread::scope`.
   - Opt-out: `SHADOW_EMBED_PAR=0`.
10. **Direct Mapped Table Reads in Decode (`crates/shadow-cuda/src/session.rs` `stage_slice`, `routed_overlapped`, `routed_qwen`, STATUS 94)**:
    - Reads the per-layer expert pointer/weight/permutation table directly from mapped `self.stage.dev_ptr + at` instead of launching `copy_words` (`-48` kernel launches per single-token decode step).
    - Opt-out: `SHADOW_DIRECT_TAB=0`.
11. **Selective `"cache_prompt": false` Snapshot Bypass (`crates/shadow-serve/src/lib.rs`, `engine.rs`, STATUS 93)**:
    - Honors `"cache_prompt": false` in `/v1/chat/completions` by skipping host prefix/conversation snapshots (`sess.snapshot_to(true, true)`), while keeping default `"cache_prompt": true` for interactive chat/agent sessions (`SHADOW_FORCE_CONV_SNAP=1` forces snapshots even when `false`).
12. **Combined Benchmark Flags (`crates/shadow/src/main.rs`)**:
    - `--no-prompt-opt-r3`: disables the 5 Round 3 switches (`SHADOW_VMM_POOL=0`, `SHADOW_SUBCHUNK_FAST=0`, `SHADOW_MMQ_REUSE=0`, `SHADOW_RESTORE_OVERLAP=0`, `SHADOW_DIRECT_TAB=0`).
    - `--no-prompt-opt`: disables all 10 Round 2 + Round 3 switches at once.

---

### 4. Closed / Falsified Hypotheses (Do NOT Re-Run)

1. **Avalon CPU-Experts Prompt Port (`docs/EXPERIMENT_SPEC_2026-11-10.json`, STATUS 92, `bench/avalon-2026-11-10/summary.md`) — CLOSED**:
   - Tested via a clairvoyant zero-cost oracle (`SHADOW_AVALON_ORACLE=N`, which drops `N` non-resident experts per layer from both file read-ahead and H2D DMA with 0 CPU compute, 0 DRAM traffic, and 0 host sync) and a router-sync probe (`SHADOW_AVALON_ORACLE_SYNC=1`).
   - At `N = 3`, even the 0-cost oracle is `+0.11%` to `+0.65%` (inside the 0.8% noise floor, because `3 / 474 = 0.63%` of the staged copy).
   - Synchronizing per layer so the host can read router top-k indices before submitting CPU work costs **`-5.31%` prompt throughput (`-6.22%` combined)** by breaking the device planner's host-ahead overlap.
2. **Right-Sizing `lend_cache` to `prompt_hint` (`right_size_lend`, STATUS 93, `bench/prompt-opt-2026-11-10/results.json`) — CLOSED**:
   - Lending fewer VRAM cache slots on a 3k prompt (`1,026` of `1,843` instead of `1,825`) was a net loss (`2.72 s -> 2.75 s`) and changed the reply hash because `slot_offset` is ordered by layer (`0..48`): lending the tail left 100% of layers `0..20` resident (`any_res == true`, forcing two-phase `fused_experts` launches on sub-chunk 0) and 0% of layers `22..47` resident while saving only 7.4% of the H2D copy.
3. **Larger Sub-Chunk Rows (`SHADOW_SUB_ROWS=4096`, STATUS 45 & 88) — CLOSED**:
   - Larger temporaries take VRAM away from `rparts`, splitting a 15k prompt into 2 passes instead of 1 (`9.85 s -> 13.40 s`, `+36%`). Default `2,048` is optimal on 12–16 GB VRAM.
4. **Prompt-Path CUDA Graphs (`SHADOW_BATCH_GRAPHS=1`, STATUS 58) — CLOSED (opt-in only)**:
   - Keyed by buffer addresses and row counts; instantiates ~500 graphs on a new prompt (`-10%` at 15k). Only helps repeated identical-shape prompts.

---

### 5. What Remains & Where the Hardware Ceiling Is

- On a **15k prompt (`8.52 s`, `1,764 tok/s`)**, fixed costs outside `step_batch` are now down to **`171 ms` total** (`20 ms` `lend_cache` + `151 ms` `restore_cache` = **2.0% of the prompt**), and the GPU is busy for `8.21 s` of the `8.35 s` pass (`98.3%` GPU utilization; host finishes queuing all 48 × 8 sub-chunk launches `190 ms` before the GPU finishes layer 47).
- On a **3k prompt (`2.62–2.99 s`, `958–1,001 tok/s`)**, fixed costs are **`206 ms` total** (`40 ms` `lend_cache` + `166 ms` `restore_cache`, dominated by the `135 ms` PCIe 4.0 x16 DMA to restore `3.2 GiB` of evicted experts back into the 3.53 GiB VRAM cache so speculative decode immediately starts at `53 tok/s`).
- The remaining levers if further prompt speed is ever needed are **Phase E** items (`docs/PLAN_RAM_TIER_AND_LEVERS_2026-11-09.md`): INT8 KV cache (`SHADOW_KV_INT8`, saving VRAM and bandwidth on the 12 attention layers at 15k–24k context) or fusing the hyper-connection norms into adjacent kernels.

---

## Historical Handoff (2026-10-27, Stages 0–2)

Written 2026-10-27 by the session that built stages 0-1 and began stage 2. Read this, then
[SHADOW_PLAN.md](SHADOW_PLAN.md) (the plan), [STATUS.md](STATUS.md) (what is proven) and [ARCH.md](ARCH.md) (the model
as math). This file is what is *not* in those: what is in flight, how to run things, and what bit us.

## What Shadow is now

A from-scratch Rust engine for `Qwen3.8-Flash-Next` (48 layers: 36 DeltaNet + 12 gated attention, 512-expert MoE,
4-stream hyper-connections, hashed n-gram memory at layer 1) and `RecursiveSI/flint` (`Flint-40B-A2.5B-abliterated-NVFP4.gguf`,
40 layers: 2 dense + 38 128-expert top-6 sigmoid MoE, GQA 32/8 with learned attention sinks, sliding window 513 + YaRN
on 3/4 layers, `NVFP4` quantization). CUDA kernels are source in `kernels/`, compiled at
run time by NVRTC through hand-bound driver calls (`crates/shadow-cuda`). **Nothing from Strata** at build or run time:
the user decided that on 2026-10-26 and cancelled the Strata benchmark plan ("Phase 0"). Strata's folder
`J:\Strata-work` holds only the model file, which is a plain GGUF. Do not propose Strata shortcuts.

Crates: `shadow-model` (GGUF reader, tokenizer, NFC, chat template, dequant, the CPU forward pass), `shadow-cuda`
(driver + NVRTC + kernels registry + `GpuBackend`), `shadow` (the CLI and the reference-comparison code), and the older
`shadow-host` / `shadow-eval` (I/O ring and tier bookkeeping, the memory yardstick).

## Where things stand

HEAD is a few commits past `7db46c9` on `main`, **nothing is pushed**. Proven (details in STATUS.md): tokenizer, NFC and chat template
identical to the checkpoint's; the CPU forward pass equals the published `transformers` reference on Shadow's own
weights for **all 48 layers** (cosine 1.00000, identical routing) on CPU and GPU, and agrees with the BF16 weights at
cosine >= 0.994 per layer; greedy decoding answers correctly; the GPU matvec backend runs 0.40 s/token vs 2.35.

### The sparse indexer: done and gated

Implemented in `forward.rs` and held to the reference by `eval/fixtures/reference_gguf_long` (2,600 tokens, layers 0-3,
12 kept positions): Shadow attends to exactly the reference's positions, every stream cosine 1.00000, routing 10/10, on
the GPU backend (150 s). The CPU backend run and `cargo test --release -p shadow --test reference -- --ignored`
(minutes) are the same comparison; regenerate the fixture with the command in `tools/make_reference_fixture.py`'s
docstring plus `--prompt-file J:\Shadow\docs\SHADOW_PLAN.md J:\Shadow\docs\ARCH.md J:\Shadow\docs\STATUS.md --tokens
2600 --positions 5,1000,2047,2050,2051,2052,2053,2054,2055,2056,2400,2599 --out evalixtures
eference_gguf_long`
(~10 minutes, `--weights gguf`, no download, capped at 30 GiB). Note that the prompt is made of the docs, so editing
them changes the prompt; the fixture records the ids, so it stays valid as a fixture.

### Then, in the plan's order

- **Stage 2 is done for correctness**: `GpuSession` runs the whole forward pass on the GPU and matches the reference
  (48 layers and the long prompt). `shadow generate --gpu` uses it; `--gpu-matvec` is the older matvec-only path. What
  is left in stage 2 is speed: fast matvec kernels (the reference ones decode per weight) and CUDA graphs.
- **Stage 3, the tiers**: the VRAM expert cache and the pinned RAM tier (4 GiB rule in `pinned.rs`) are done and
  gated (STATUS.md, Stage 3). The drive tier stays on buffered reads: a threaded version and `shadow-host`'s direct-access ring were both measured slower (STATUS.md says why - the file cache), and an evictable/adaptive RAM tier is untried. Placement profiles are
  Shadow's own `SHRC` routing counts (`profiles/`, `--record-routing`).
- **Stage 2 speed**: batched expert matvecs are in (0.1.3). Left: the mixers/norms/glue (~30 ms/token, 21%) and
  launch overhead (CUDA graphs). `SHADOW_PROFILE=1` prints where a token's time goes.
- **Stage 4**: `shadow-serve` (`crates/shadow-serve`, `shadow serve`, 16/16 live checks in `tools/serve_check.py`,
  including `<tool_call>` parsing and `tools/shadow_harness.py` + `shadow-eval run`) and `shadow-bench`
  (`crates/shadow-bench`) are in (0.1.7-0.1.11). Shadow's own 8-launch noise floor is measured in
  `bench/floor-2026-10-27`: **median 0.8%, p90 2.0%, worst 3.6%** at **10.20 tok/s** `decode_warm` (`--ram-tier-gib 30`),
  lowest available RAM 9.6 GiB, 0 answer differences across 48 turns.
- **Stage 5 (`0.1.12`-`0.1.13`)**:
  - `0.1.12`: Register-immediate `iq4nl_val`, `Fast<Q4K/Q5K/Q6K/IQ4XS>` kernels, fused forward-pass kernels
    (`hc_down_inject_bf16`, `hc_write_rmsnorm_groups`, `moe_combine_qwen`, `matvec_pair_bf16`, `l2norm_qk_heads`,
    `head_norm_rope_qgk`), and `copy_stream` DMA + pinned file-miss staging + early shared-expert overlap cut GPU compute
    from 52.5 to 31.6 ms/token (`-40%`) and raised 8-launch `decode_warm` from **10.20 tok/s to 15.34 tok/s (`+50.4%`)**
    (`bench/stage5-kernels-2026-10-27`).
  - `0.1.13`: Aligned `u16`/`u32`/`float` loads across `quants.cuh` + `matvec.cu`, general `matvec_pair_*` (`mv_pair`),
    per-layer exact-size VRAM `ExpertCache` (`1,831 -> 2,654` slots, `5.07 GiB`), exclusive VRAM-pinned hot experts (`16,989`
    unique experts across `VRAM U RAM`, startup `45 s -> 32 s`), and `RamTier`'s Clock promotion ring raised 8-launch
    `decode_warm` to **16.92 tok/s (`+65.9%` over baseline, `+10.3%` over `0.1.12`)** (`bench/stage5-tiers-2026-10-27`,
    floor `0.8%` median / `2.3%` p90 / `3.3%` worst, single-prompt decode `18.5 tok/s`).
  - `0.1.23`: drive-tier expert reads in parallel on `shadow-model::readpool` (8 workers, own handles, buffered):
    `decode_warm` **17.37 -> 18.52 tok/s, +5.3% paired, all 40 pairs faster** (`bench/readpool-2026-10-27`). Profile
    before it (synchronized, 99 steps, 70.6 ms/step): file reads 27%, pinned-RAM uploads 26% (~22 GB/s, the PCIe 4 x16
    limit on this RTX 5070), mixers/norms/glue 25%, routed matvecs 9%. Uploads can only shrink by moving fewer bytes.
  - Night of 2026-10-27 (docs/STATUS.md, last sections): speculation measured before building (verifying m tokens
    moves the union of their experts, m2 1.66x, m4 2.74x: ~1.25-1.3x ceiling here; n-gram drafting loses; the `--mtp`
    path never batches verification and its draft step leaves NaNs in the base model). Hybrid CPU experts
    (`--cpu-experts`) and pre-gated prefetch (`--pregate`) built, exact, and both measured as losses or ties; kept
    opt-in. Default-on and exact: mapped router reads, mapped staging for small uploads, `Gguf::tensor` index, top-k by
    insertion. `matvec_cols_*` + `dotm<M>`: bit-identical multi-input matvecs, the first piece of batched forward.
  - 2026-10-28: `matvec_cols` is fast (5.6-6.8x for 8 inputs on the big matrices, bit-identical); session
    snapshots (`snapshot`/`snapshot_to`/`restore`, bit-exact) and the server's shared-prefix cache (new chats with the
    same system prompt + tools: ~60 s -> ~1 s); web UI fixes.
  - 2026-10-28 (later): **the batched step is in** (`step_batch` / `prefill_batch`, up to 64 tokens, bit-identical
    to `step`; STATUS.md last section): prompt reading **0.062 -> 0.035 s/token** (-46%, 8 pairs), decode a tie,
    speculation (`--spec K`) measured as a loss (-11% / -18%) and kept opt-in. Next there: expert-major routed matvecs
    in the batched MoE, bigger prefill unions. What follows is the design as it was planned.
  - **Next: the batched step**, `GpuSession::step_batch(&[u32]) -> logits per token` (m <= 8), layer-major. Per layer:
    the mixer's input projections for all m tokens through `matvec_cols` (inputs/outputs laid out [m][dim]); the
    sequential parts (`delta_step` recurrence, conv window, KV write, attention over positions <= its own) per token
    in order; the output projection through `matvec_cols`; hc glue per token (small kernels); the router for all m
    through `matvec_cols`, one download of m x 512 logits; one fetch of the union of the m tokens' experts; the routed
    expert matvecs per token (or `matvec_batch` over (token, expert) pairs); combine per token. Gate: prefill of a
    prompt through `step_batch` in chunks of 8 gives bit-identical last-token logits... or within the f32 sum-order
    noise of the token-by-token path (the cols kernels are bit-identical per input, so aim for identical). Then use it
    for prefill in `shadow serve` (today 0.059 s/token), measure, and only then speculative decoding: draft (MTP head
    or n-gram), `snapshot(false)` before the verify pass, `step_batch(draft)`, keep the accepted prefix, `restore` +
    re-step on rejection (or restore and replay the accepted tokens), with the draft length chosen from the measured
    union cost (`tools/spec_probe.py`).
- Cleanup done (0.1.4): `shadow-abi`, `include/shadow/c/abi.h`, `shadow-host/src/ffi.rs` and `protocol.rs` are
  removed. Still owed: copy the model under `J:\\Shadow\\models\\` and record its SHA-256 (the plan
  says the model file is the only input; it is still read from `J:\Strata-work\models\swift15\IQ3_S\`).

## Commands

```powershell
cargo build --release -p shadow            # target\release\shadow.exe
cargo test --workspace                     # model-dependent tests print "SKIPPED, not passed" without the model
cargo test --release -p shadow --test reference            # the stage 1 gate, ~10 s
shadow inspect  MODEL [--tensors] [--kv]   # headers only; docs/model-inspect.txt is its output
shadow kv       MODEL KEY                  # one key whole (the chat template, the vocabulary)
shadow stats    MODEL TENSOR...            # min/mean/max of float tensors (summaries do NOT prove equality)
shadow dump     MODEL TENSOR OUT [--rows a,b | --expert E]   # dequantized f32; prints "ROWS COLS"
shadow compare  MODEL FIXTURE_DIR [--gpu]  # stream-by-stream against a reference fixture
shadow generate MODEL "prompt" [-n N] [--chat] [--gpu]
    [--expert-cache-gib G] [--ram-tier-gib G|auto --placement profiles/routing-6prompts.shrc] [--record-routing FILE]
    # a RAM tier of G GiB raises the working set by G: give run_capped a cap of G + 8 or more (auto: -CapGiB 44)
    # best measured (8 runs): --ram-tier-gib auto (~32.7 GiB here) + the placement: ~0.100 s/token
```
MODEL is `J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf`
(`SHADOW_MODEL` overrides it in the tests). Regenerating fixtures: `tools/make_reference_fixture.py`
(`--weights checkpoint` needs network and the `I:` cache; `--weights gguf` needs only the built `shadow.exe`),
`tools/make_tokenizer_fixture.py`, `tools/gen_dequant_fixtures.py`, `tools/gen_quant_tables.py`,
`tools/gen_unicode_table.py`, `tools/gen_nfc_table.py`. The reference runs in `.venv-ref` (git-ignored; Python 3.11, the
machine's torch 2.11 CPU via system-site-packages, `transformers==5.21.0`; `tools/_refenv.py` stubs the machine's broken
torchaudio, import it before transformers). Checkpoint reads are cached under `I:\shadow-ref` (`I:` has ~700 GB free;
the user said to use it if more space is needed).

## Rules and how this machine bites

- **RAM**: 63.8 GiB; a pinned tier cannot be paged and the PC has frozen at 0.4 GiB free. Every process that loads the
  model runs under `tools/run_capped.ps1` (hard working-set cap, killed below 4 GiB available). Call it **in
  process with `&`**, with `-Exe` and one `-Arguments` string: a nested `powershell -File` splits quoted arguments.
  Never `mlock` or `--mmap off` this model; never start a second pinned engine beside a live one. The user's live
  Strata server (port 8080) was not touched tonight; ask before stopping it.
- **Bash tool**: a command with several heredocs, or with Python containing `'''` and apostrophes, can fail to parse.
  Write scripts to files with the Write tool and run them. MSYS paths (`/c/...`) do not work in Windows Python; use
  `C:/...`. `sleep` chains are blocked; use Monitor with an until-loop, or run in the background.
- **git**: `.gitattributes` forces `eval/** text eol=lf` and now `eval/fixtures/**/*.bin binary`. A binary under
  `eval/` once lost 34 bytes to the eol rule; after adding one, `git show HEAD:path | cmp - path`. `core.autocrlf=true`
  prints a CRLF warning on every add; harmless. Commit messages end with the `Co-Authored-By` line the harness asks for.
- **The 48-layer fixture** (`I:\shadow-ref\fixtures\reference_gguf48`, 48 MB) is deliberately not in the repo.
- Do not trust: `shadow stats` for equality; the plan's old numbers measured on Strata (floor, 41.75 tok/s); `STATUS.md`
  and this file after a week.

## Things settled that look unsettled

- DeltaNet value heads are **tiled** (`v*16 + k`) in the GGUF, the reference groups them; Shadow's key head is `h % 16`.
- All RMS norm weights in the file carry the +1 (verified exactly against the checkpoint for the PLE norms).
- `ple_norm_key` and `ple_norm_query` differ (max 0.054) in the file and the checkpoint.
- Type 42 is upstream ggml's Q2_0 (`(q - 1) * d`, 64 weights per 18 bytes).
- The checkpoint's tokenizer NFC-normalizes; Shadow's `nfc.rs` does too.
- The chat template: with thinking on and the default effort, a system block is always emitted.

## Still open

Q2_0 has no independent fixture; the GGUF has no MTP tensors (speculative decoding needs another draft: quantize the
checkpoint's own MTP layer, or prompt-lookup drafting); a prompt other than the one 12-token sentence has not been
compared stream by stream apart from the 2,600-token one above; sampling (temperature, top-p) is not implemented
(greedy only); vision is out of scope until asked for.

## The user

Works on Windows 11, RTX 5070 (12 GiB), 63.8 GiB RAM, Sandisk NVMe. Wants an engine they own end to end, in Rust, with
measured claims (BENCH_NOISE.md's protocol: >= 8 interleaved runs, a tie is reported as a tie). Prefers short, direct
reports with what failed said first. Said running out of tokens overnight is fine: the limit resets.
