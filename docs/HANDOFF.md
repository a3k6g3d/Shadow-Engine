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
(greedy only); `chat.rs` refuses `tools` definitions (needed before `shadow-eval run` can run `tools/shadow_harness.py`
against `shadow-serve`); vision is out of scope until asked for.

## The user

Works on Windows 11, RTX 5070 (12 GiB), 63.8 GiB RAM, Sandisk NVMe. Wants an engine they own end to end, in Rust, with
measured claims (BENCH_NOISE.md's protocol: >= 8 interleaved runs, a tie is reported as a tie). Prefers short, direct
reports with what failed said first. Said running out of tokens overnight is fine: the limit resets.
