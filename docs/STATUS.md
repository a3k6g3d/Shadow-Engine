# Status

Where the plan ([SHADOW_PLAN.md](SHADOW_PLAN.md)) stands. Updated 2026-10-27.

Build and run: `cargo build --release -p shadow`, then `target/release/shadow.exe` with `inspect`, `kv`, `stats`, `dump`,
`generate` (`--chat`, `--gpu`) and `compare`. Any process that loads the model runs under `tools/run_capped.ps1`
(hard working-set cap, killed below 4 GiB available).

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
  48 MB of fixture kept on `I:\shadow-refixtureseference_gguf48`, regenerate with
  `make_reference_fixture.py --layers 48 --weights gguf`): **243 streams, minimum cosine 1.00000, relative error
  0.0000, max |diff| 1e-4, and the router picks the same 10 experts at every token of all 48 layers** - on the CPU
  backend and on the GPU backend. That covers all 36 DeltaNet and 12 attention layers and every quant type.

Known gap: attention was only exercised below 2,048 positions, where the sparse indexer keeps every key. Past that the
indexer selects blocks and is **not implemented** (`forward.rs` refuses more than 2,048 positions). Also untested:
a different prompt (one prompt, 12 tokens, so far) and decoding many tokens (the DeltaNet and KV state across steps is
exercised only by the 41-token greedy runs, which are correct but not compared stream by stream).

## Stage 2: begun

- Every quant type dequantizes on the GPU bit for bit like the Rust reference (`kernels/quants.cuh`, `dequant.cu`).
- A reference GPU matvec for all 11 types (`kernels/matvec.cu`), relative error ~3e-8 on real tensors; not fast yet
  (264 GB/s Q6_K, 15-23 GB/s for the IQ2_S/IQ3 expert slices).
- `shadow generate --gpu` runs the matrix-vector products on the GPU: same tokens, **0.40 s/token against 2.35** on the CPU.
- Still on the CPU: norms, the DeltaNet recurrence, attention, routing, and every expert upload.

## Open

1. **Q2_0** (9 down-projections) has no independent reference tool (gguf-py lacks it); it follows upstream ggml's
   definition and is checked by hand and by the end-to-end match above, which runs through it.
2. **The sparse indexer** (positions > 2,048): not implemented.
3. **Speculative decoding's draft**: the GGUF has no MTP tensors (ARCH.md, question 7).
4. **Stage 3** (VRAM / pinned RAM / drive tiers) and everything speed-related has not started; the plan's floor
   protocol applies from there.

## Things that went wrong, so they are not repeated

- A claim in ARCH.md that two PLE norm tensors were identical rested on summary statistics and six values; they are
  different (max 0.054). `shadow stats` summaries do not prove equality; compare the vectors.
- git's `eval/** text eol=lf` rule rewrote CRLF byte pairs inside a committed `tensors.bin`; `.gitattributes` now marks
  `eval/fixtures/**/*.bin` binary. After adding any binary fixture: `git show HEAD:path | cmp - path`.
- Uncapped, llama.cpp's mmap load took available RAM from 55 to 0.73 GiB in 49 s. Always `tools/run_capped.ps1`.
- `J:\Strata-work\RUN.md` records the three failed llama-embedding runs (HANDOFF.md's route); nothing there is needed.
