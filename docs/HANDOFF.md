# Handoff

Written 2026-10-27 by the session that built stages 0-1 and began stage 2. Read this, then
[SHADOW_PLAN.md](SHADOW_PLAN.md) (the plan), [STATUS.md](STATUS.md) (what is proven) and [ARCH.md](ARCH.md) (the model
as math). This file is what is *not* in those: what is in flight, how to run things, and what bit us.

## What Shadow is now

A from-scratch Rust engine for `Qwen3.8-Flash-Next` (48 layers: 36 DeltaNet + 12 gated attention, 512-expert MoE,
4-stream hyper-connections, hashed n-gram memory at layer 1). CUDA kernels are source in `kernels/`, compiled at
run time by NVRTC through hand-bound driver calls (`crates/shadow-cuda`). **Nothing from Strata** at build or run time:
the user decided that on 2026-10-26 and cancelled the Strata benchmark plan ("Phase 0"). Strata's folder
`J:\Strata-work` holds only the model file, which is a plain GGUF. Do not propose Strata shortcuts.

Crates: `shadow-model` (GGUF reader, tokenizer, NFC, chat template, dequant, the CPU forward pass), `shadow-cuda`
(driver + NVRTC + kernels registry + `GpuBackend`), `shadow` (the CLI and the reference-comparison code), and the older
`shadow-host` / `shadow-abi` / `shadow-eval` (I/O ring, retired C seam, the memory yardstick).

## Where things stand

HEAD is `7db46c9` on `main`, **nothing is pushed**. Proven (details in STATUS.md): tokenizer, NFC and chat template
identical to the checkpoint's; the CPU forward pass equals the published `transformers` reference on Shadow's own
weights for **all 48 layers** (cosine 1.00000, identical routing) on CPU and GPU, and agrees with the BF16 weights at
cosine >= 0.994 per layer; greedy decoding answers correctly; the GPU matvec backend runs 0.40 s/token vs 2.35.

### In flight: the sparse indexer gate (do this first)

The indexer is **implemented but not gated** (commit `7db46c9`). The gate is a 2,600-token fixture whose positions
2,051+ make layer 3 select blocks. Its generation was running in the background when this was written; the output
folder is `eval/fixtures/reference_gguf_long`. If that folder has a `manifest.json`, skip to step 2.

1. Generate it (about 10-15 minutes, ~30 GiB cap, needs no download, `target/release/shadow.exe` built first):
   ```powershell
   cd J:\Shadow
   cargo build --release -p shadow
   & .\tools\run_capped.ps1 -CapGiB 30 -Out run.txt -Exe 'J:\Shadow\.venv-ref\Scripts\python.exe' -Arguments '-u J:\Shadow\tools\make_reference_fixture.py --layers 4 --weights gguf --prompt-file J:\Shadow\docs\SHADOW_PLAN.md J:\Shadow\docs\ARCH.md J:\Shadow\docs\STATUS.md --tokens 2600 --positions 5,1000,2047,2050,2051,2052,2053,2054,2055,2056,2400,2599 --out J:\Shadow\eval\fixtures\reference_gguf_long'
   ```
   (If the docs have changed, the prompt text changes; that is fine, the fixture records the ids. If a file has fewer
   than 2,600 tokens in total the script says so.) Its first attempt failed on an expert-group size, now fixed.
2. Compare: `target\release\shadow.exe compare <model.gguf> eval\fixtures\reference_gguf_long`. Expected: every stream
   cosine 1.00000 and routing 10/10 at the kept positions, and `layer3: attended positions that differ ...` all **0**.
   Then `cargo test --release -p shadow --test reference -- --ignored` (minutes) is the same as a test.
3. If the selection differs by whole blocks (multiples of 4) at a few positions, suspect a score tie or the f32
   summation order of the block scores (Shadow sums the dot in f64 and rounds to f32; the library uses an f32 matmul);
   if it differs everywhere, the semantics in ARCH.md ("The indexer only changes anything past 2,051 tokens") or the
   RoPE position of a block key (its first position, `4b`) is wrong. The reference code is in
   `.venv-ref\Lib\site-packages\transformers\models\qwen4_exp\modeling_qwen4_exp.py` (`Qwen4ExpTextQSAIndexer`).
4. When green, commit the fixture (check `git show HEAD:path | cmp - path` for the `.bin`), update STATUS.md (the known
   gap) and remove the "not gated" wording.

### Then, in the plan's order

- **Stage 2, finish the GPU path** (checked against the 48-layer comparison each step): norms, the DeltaNet
  recurrence, attention (+ the indexer), routing, and the hyper-connection glue on the GPU; today only the matrix-vector
  products are there and every expert is uploaded per use (58k uploads per 41 tokens). The reference matvec kernels are
  slow (15-23 GB/s on the IQ2/IQ3 experts); the fast ones are measured against them.
- **Stage 3, the tiers**: VRAM expert cache, pinned RAM (the 4 GiB headroom rule enforced in code, not in a doc), the
  drive through `shadow-host`'s ring. Needs a placement profile in Shadow's own format.
- **Stage 4**: the HTTP server (hyper + serde_json, decided), `shadow-bench`, and Shadow's own noise floor (the
  10.1% / 27.0% / 51.9% in BENCH_NOISE.md is Strata's).
- Cleanup owed: remove `shadow-abi`, `include/shadow/c/abi.h`, `shadow-host/src/ffi.rs` and `protocol.rs` once
  nothing needs them (SHADOW_PLAN.md says so); copy the model under `J:\Shadow\models\` and record its SHA-256 (the plan
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
(greedy only); no streaming server; vision is out of scope until asked for.

## The user

Works on Windows 11, RTX 5070 (12 GiB), 63.8 GiB RAM, Sandisk NVMe. Wants an engine they own end to end, in Rust, with
measured claims (BENCH_NOISE.md's protocol: >= 8 interleaved runs, a tie is reported as a tie). Prefers short, direct
reports with what failed said first. Said running out of tokens overnight is fine: the limit resets.
