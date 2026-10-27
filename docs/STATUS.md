# Status

Where the plan ([SHADOW_PLAN.md](SHADOW_PLAN.md)) stands. Updated 2026-10-27, after the first overnight run.

The binary is now its own crate: `cargo build --release -p shadow`, then `target/release/shadow.exe` (inspect, kv,
stats, generate).

## Stage 0: done, with one gate item open

| item | state | evidence |
|---|---|---|
| `docs/ARCH.md` | written; 6 open questions left, each with how to settle it | from the GGUF, transformers' `qwen4_exp` (Apache-2.0) and the base model's `config.json` |
| GGUF reader + `shadow inspect` | **gate green**: 1,224 tensors, 77.801 GiB, every size from its type and equal to the offsets | `crates/shadow-model/tests/real_model.rs`, `docs/model-inspect.txt` |
| `shadow-cuda`: driver API + NVRTC by hand, CUBIN cache | **green on the RTX 5070** (cc 12.0, NVRTC 13.4): `vector_add` cold 20.5 ms compile, warm from cache, exact | `crates/shadow-cuda/tests/gpu.rs` |
| kernel-signature drift test | **green**, both directions | `crates/shadow-cuda/tests/drift.rs` |
| f16 / bf16 conversions (GPU prelude) | **bit-exact** against the Rust reference: 2 x 65,536 widenings, 2 x 925,630 narrowings | `crates/shadow-cuda/tests/convert.rs` |
| tokenizer + chat template | 247,098 / 247,100 vocabulary tokens re-encode to one token; 78 KB of real text round-trips | `real_model.rs`; **gate item open: no fixture ids from an outside tokenizer yet** |
| the oracle | **decided**: the public reference implementation + the base model's BF16 weights (ARCH.md, "The fixture"). llama.cpp is out (the only build here is Strata's fork; three load attempts failed, `J:\Strata-work\RUN.md`) | |

## Stage 1: running, not gated

- **Dequantization for all 12 types the model uses.** 11 are bit-exact against ggml's reference dequantizer on 84
  real blocks (`eval/fixtures/dequant.txt`, `tests/dequant.rs`). Q2_0 (9 tensors) follows upstream's definition and is
  sanity-checked only, because no reference tool here implements it.
- **The CPU reference forward pass** (`crates/shadow-model/src/forward.rs`, `shadow generate`), written from
  ARCH.md: every layer, f64 accumulation, dense weights in RAM (4.14 GiB), experts and the n-gram table read per use.
  Peak working set 4.4 GiB, ~2.5 s/token on the CPU. Greedy outputs:
  - raw "The capital of France is" -> " Paris, which is often located"
  - chat (template, thinking off) "What is 17 times 23? Answer in one short sentence." -> **"17 times 23 is 391."**
    then `<|im_end|>`: right answer, clean stop, 41 positions.
- **The bug that took the night:** the DeltaNet value heads are stored *tiled* (`v*16 + k`), not grouped as in the
  reference, because llama.cpp's converter reorders them for ggml's broadcast. With the grouped mapping the model
  still said "Paris" and then fell apart within ~30 tokens. Now in ARCH.md.
- **Not gated yet.** Coherent text means most of the math is right, not all of it. The stage 1 gate is the fixture
  from the BF16 reference, and making it needs a download (HTTP range reads of a few safetensors shards from
  huggingface.co/Qwen/Qwen3.8-Flash-Next), which needs your OK.
- Limit: at most 2,048 positions (the sparse indexer is not implemented; below 2,048 it keeps every key, so attention
  is exact there).

## Stage 2: begun

- **Every quant type dequantizes on the GPU bit for bit** like the Rust reference (all fixture blocks of 9 types +
  BF16, 25,600 real Q2_0 blocks), from one decoder per format (`kernels/quants.cuh`).
- **A reference GPU matvec for all 11 types** (`kernels/matvec.cu`), within 3.2e-8 relative of an f64 CPU dot product
  on real tensors. Not fast yet: 264 GB/s for Q6_K, 15-23 GB/s for the IQ2_S/IQ3 expert slices.
- **The forward pass runs its matvecs on the GPU** (`shadow generate ... --gpu`; `Backend` trait in forward.rs,
  `GpuBackend` in shadow-cuda): 3.78 GiB of dense weights resident in VRAM, experts uploaded per use. Same tokens as
  the CPU ("17 times 23 is 391."), **0.40 s/token against 2.35** on the CPU. `crates/shadow/tests/backends.rs` holds
  the two backends to the same top token and logits within 7.3e-6 (of ~16) at every position.
- What is still on the CPU: the norms, the DeltaNet recurrence, attention, the hyper-connection glue, the MoE routing,
  and every expert upload (58,361 per 41 tokens). Those are the next kernels, and then the memory tiers (stage 3).

## What needs you

1. **OK to download** the parts of the BF16 checkpoint the first fixture needs (ARCH.md, "The fixture": the
   safetensors headers, then layers 0-3, the embedding and the head, by range read; a few GB, not 360) - and to install
   `torch` (CPU) into Python to run the reference once.
2. `J:\Strata-work\RUN.md` replaces HANDOFF.md's plan: llama-embedding was tried three times, capped, and failed with a
   tensor read out of bounds in Strata's fork. Nothing to do unless you want that path.
3. I did not start or stop the Strata server tonight; it was not running when the probes started.

## Safety notes for whoever runs this next

- Any process that maps or reads this model runs under `tools/run_capped.ps1` (hard working-set cap + kill below
  4 GiB available). Uncapped, llama.cpp's mmap load took available RAM from 55 to 0.73 GiB in 49 s.
- `shadow generate` itself needs ~4.4 GiB and was run capped at 16 GiB.
