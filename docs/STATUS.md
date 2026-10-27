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

## Things that went wrong, so they are not repeated

- A claim in ARCH.md that two PLE norm tensors were identical rested on summary statistics and six values; they are
  different (max 0.054). `shadow stats` summaries do not prove equality; compare the vectors.
- git's `eval/** text eol=lf` rule rewrote CRLF byte pairs inside a committed `tensors.bin`; `.gitattributes` now marks
  `eval/fixtures/**/*.bin` binary. After adding any binary fixture: `git show HEAD:path | cmp - path`.
- Uncapped, llama.cpp's mmap load took available RAM from 55 to 0.73 GiB in 49 s. Always `tools/run_capped.ps1`.
- `J:\Strata-work\RUN.md` records the three failed llama-embedding runs (HANDOFF.md's route); nothing there is needed.
