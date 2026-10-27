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

## Open

1. **Q2_0** (9 down-projections) has no independent reference tool (gguf-py lacks it); it follows upstream ggml's
   definition and is checked by hand and by the end-to-end match above, which runs through it.
2. ~~The sparse indexer~~: implemented and gated (see Stage 1).
3. **Speculative decoding's draft**: the GGUF has no MTP tensors (ARCH.md, question 7).
4. **Stage 3** has its first tier only: the VRAM expert cache (see Stage 3 below). Pinned RAM (with the 4 GiB rule in
   code) and the drive through the ring have not started; the plan's floor protocol applies from there.

## Stage 3: the VRAM expert cache, gated for correctness; speed unmeasured

- `GpuSession` keeps routed experts in one VRAM arena of 2.54 MiB slots (gate + up + down), LRU, the current token's
  experts never evicted by its own misses. It takes the free VRAM less `VRAM_RESERVE` (1.5 GiB) by default;
  `shadow generate --gpu --expert-cache-gib G` sets it (`0`: no cache).
- **Gate (2026-10-27):** `tests/reference.rs` (4 passed), `shadow compare ... reference_gguf48 --gpu` (all 48 layers,
  cosine 1.000, routing 10/10 everywhere) and the long-prompt GPU indexer test pass with the cache on. Greedy
  "What is 17 times 23?" (`--chat`, 24 tokens) gives **identical token ids** with no cache, 2 GiB and auto (4.57 GiB,
  1,842 slots).
- **One run each, not the protocol** (no 8 interleaved runs): 0.371 s/token without the cache, 0.215 at 2 GiB (45.5%
  hits), 0.176 at 4.57 GiB (61.3% hits). Indicative only until stage 4's bench measures it against a floor.

## Things that went wrong, so they are not repeated

- A claim in ARCH.md that two PLE norm tensors were identical rested on summary statistics and six values; they are
  different (max 0.054). `shadow stats` summaries do not prove equality; compare the vectors.
- git's `eval/** text eol=lf` rule rewrote CRLF byte pairs inside a committed `tensors.bin`; `.gitattributes` now marks
  `eval/fixtures/**/*.bin` binary. After adding any binary fixture: `git show HEAD:path | cmp - path`.
- Uncapped, llama.cpp's mmap load took available RAM from 55 to 0.73 GiB in 49 s. Always `tools/run_capped.ps1`.
- `J:\Strata-work\RUN.md` records the three failed llama-embedding runs (HANDOFF.md's route); nothing there is needed.
