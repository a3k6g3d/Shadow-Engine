# Handoff

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
  - **Next, in order:** (1) row-tiled `matvec_cols` for the big K-quant matrices; (2) a layer-major batched step
    (`step_batch(&[u32])`): dense matvecs through `matvec_cols`, per-token mixer kernels, one MoE pass over the union
    of the batch's experts; gate it on prefill against token-by-token (identical logits) - prefill is 0.059 s/token
    today; (3) DeltaNet/conv/PLE state snapshots so a verify pass can roll back; (4) only then a drafter, with
    adaptive draft length from the measured union cost.
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
