# Plan: beat Strata on prompt speed (2026-11-07)

Supersedes the numbers of `PLAN_EXCEED_STRATA.md` (2026-11-05) and `PLAN_INT8_MMQ.md`; their rules still apply: **measure, and what loses stays opt-in; every experiment gets a dated STATUS entry with its numbers, wins and losses alike; a tie is a loss; every model load under `tools/run_capped.ps1`; the tier stays 36 GiB unless the user says otherwise; the DRAM is at its stable max.**

## 1. Where we are (measured, same PC, same model files)

| prompt | Strata 0.2.11 | Shadow now (mean of 4 servers) | gap |
|---|---|---|---|
| 3k tokens | 1,261 tok/s (2.4 s) | 535 (5.6 s) | 2.4x, 3.2 s |
| 9.6k | 1,844 (5.3 s) | 1,068 (9.0 s) | 1.7x, 3.7 s |
| 15k | 2,447 (6.3 s) | 1,038 (14.6 s) | 2.4x, 8.3 s |

Strata's own cost model fits all three: **T = 1.8 s x chunks of 8,192 tokens + 0.18 ms x tokens** (1.8 s is its expert set, 45 of 46.8 GiB pinned in RAM, going over PCIe once a chunk; 0.18 ms is 5,500 tok/s of compute). Shadow's 15k prompt is one pass whose 49 GB of expert copies hide under the compute, so Shadow is **ahead structurally** at 15k (no 3.6 s of copying); it loses on compute: its kernels need **0.57 ms a token (8.7 s)** against Strata's 0.18 ms, 3.2x.

Shadow's 15k prompt (Nsight Systems, graphs off, STATUS 44-50), kernel share of the 8.7 s: MoE products 39% (gate/up IQ2_S + IQ3_XXS + IQ3_S 27%, IQ4_NL down 11%, Q2_0 2%), attention 14% (sparse 8%, dense 3%), dense int8 projections 10% + bf16 3%, DeltaNet 5%, hyper-connection / norm / swiglu kernels 11%. **Outside the layer loop: ~2.3 s (20% of the wall time)**, and the default (graphs on) build reads **~2 s slower than the same prompt with graphs off** (14.0-14.6 s against 11.7-12.0 s in the profiled runs; the A/B in STATUS 54's neighbourhood saw +4%): both are unexplained and cheap to look at.

## 2. What has to be true

- **15k:** wall <= 6.3 s: kernels <= ~5.3 s (-40%), everything else <= 1 s (now ~2.3 s + graph cost).
- **9.7k:** <= 5.3 s. **3k:** <= 2.4 s.
- **3k cannot be won with kernels.** A 3k prompt touches most experts; the 11 GiB that are not in the 36 GiB tier come off the SSD at 2.4 GB/s (4-5 s) before any compute; Strata reads nothing. It needs one of: a bigger tier (the user's call: Strata runs with 3-4 GiB of RAM free), CPU compute of the cold experts (Strata does it for chunks <= 3,072 tokens), a better placement of the hot experts for short prompts, or a faster read path (direct I/O crashed, STATUS 19-20). **Decision for the user at the end of Phase 2; not assumed.**

## 3. Phases (each ends with its gate; a lever below 3% of kernel time, or a tie, stays opt-in)

**Phase 0: remove Flint (user's decision 2026-11-07; it will be tested on tandem.cpp later).** The audit: `arch == "flint"` paths in `shadow-model` (`forward.rs`: CPU `step_flint`, `attention_flint`, `mlp_flint`, `moe_flint`, sigmoid routing; `chat.rs`: `render_flint`, its tool-call format; `tokenizer.rs`), `shadow-cuda` (`session.rs`: `step_flint_opt`, `attention_flint`, `mlp_flint`, ~26 `is_flint` branches in the MoE path; `moe_combine_flint`, `attention_sinks_heads`, the NVFP4 kernels `matvec_*_nvfp4`, `dequant_nvfp4`), tests (`crates/shadow/tests/reference.rs`, `tokenizer_fixture.rs`, the NVFP4 dequant tests), fixtures (`eval/fixtures/*flint*`, `reference_flint_gguf`, `profiles/flint-routing.shrc`, `tools/make_reference_fixture_flint.py`), README / docs mentions (history stays). Keep the GGUF type table so `shadow inspect` still reads a Flint file. Everything stays in git history (tag `pre-flint-removal`). *Gate:* build with no new warnings, drift + workspace tests green, `--exact` hash 162a8e07 and `quality_test` 12/12, prompt speed unchanged (a deletion), `shadow serve` on a Flint file refuses with a clear message.

**Phase 1: learn where Strata's 0.18 ms a token comes from (no guessing).**
1. Read Strata's prefill sources on disk (`J:\Strata-work\Strata-0144\src\prefill`, v0.1.44, MIT): `prefill.cpp` (the chunk loop, 321 KB), `moe_fused_iq.cu` (its own fused IQ-type MoE kernels, 86 KB: the source of its IQ3_S speed), `moe_mmq.cu` / `mmq_resident_sort.hpp` (resident / copied split, sorted ids), `gdn_rec_parity.cu` (its DeltaNet recurrence), `gemm.cu`, `kernels.cu`. Note per stage: the chunk size, what is batched across the whole chunk, which kernels are fused, which precision.
2. Profile a running Strata 0.2.11 on the same 15k / 9.7k / 3k prompts with Nsight Systems (`--cuda-graph-trace=node`): kernels by category with times, the copy timeline, the CPU's share. **Needs a runnable Strata: the user's own install (path to be given) or a new download (ask first; the 0.2.11 copy of 2026-11-05 was deleted).** Shadow stopped, RAM free (Strata pins 47 GiB).
3. *Gate:* a **budget table**, per category: Strata ms, Shadow ms, ratio, and a one-line reason. It replaces the guesses of section 1 and orders everything below.

**Phase 2: the cheap, measured wins (can start before Phase 1 ends).**
1. **CUDA graphs on the prompt path: on vs off** (`SHADOW_BATCH_GRAPHS`), A-B-B-A, 3k / 9.6k / 15k, new prompt each time (the graph key now carries the buffer layout, so a repeated shape reuses less than before). Expected -1 to -2 s at 15k if the ~500 captures at ~18 ms each are on the critical path. *Gate:* default off for prompts that do not repeat, or per prompt length; STATUS entry.
2. **Instrument the ~2.3 s outside the layer loop** (`answer`: tokenization and `fit_context`'s repeated encodes, `lend_cache` (eviction, sync), the pass, `restore_cache` (the lent experts read back, from the file for cold ones), the prefix and conversation snapshots (0.9 GiB host copy), first logits). Fix the largest. Expected -0.5 to -1.3 s at 15k.
3. *Gate:* timeline of one 15k prompt with every phase named; the numbers go to STATUS.

**Phase 3: the structural lever, layer-wide MoE groups (Strata's chunk of 8,192).** Today each sub-chunk (1,536-2,048 rows) runs its own MoE: ~40 tokens an expert, weights re-read and re-unpacked 4-8 times a layer, groups 38% empty (STATUS 46, 48). The MoE's own temporaries are small (gate/up/act/out 180 KB a row, 12.8k + 6.4k + 25.6k floats; fused gate/up+SwiGLU 25 KB a row: `act` only), the big per-row buffers belong to attention / DeltaNet (591 KB a row). So run each layer in two sweeps: (1) the mixers sub-chunk by sub-chunk, keeping each row's normalized MoE input `u` (10 KB a row) and the routing for the whole pass; (2) **one MoE over all rows of the pass** (planner already sized for 16,384 rows), ~160 tokens an expert at 8k rows, groups of 128 (the MMQ kernels take J up to 128), then the hyper-connection write per sub-chunk. First an experiment, no restructuring: the MoE kernel time per token against the sub-chunk size (1,024 / 1,536 / 2,048 and a synthetic 4,096 on the standalone `native/mmq/bench` harness with 160 tokens an expert) to read the potential before building. Expected -1.0 to -1.7 s at 15k, and it re-opens the fused gate/up kernel, which lost at 40 tokens an expert. *Gate:* MoE kernel time -25% or better in Nsight Systems, `--exact` hash and quality unchanged, needle tests 8/8, `prefix_restore_test` 3/3, ABBA speed pairs.

**Phase 4: the rest, ordered by the Phase 1 budget table** (each only if the table says it matters): attention weighted sum with V staged through shared memory (`cp.async`, `ldmatrix`; the direct version lost, STATUS 51) and the softmax passes of the dense kernel; the hyper-connection up + mix fusion (2%); dense int8 tile choice and fusing the activation quantization; Strata's IQ fused-MoE ideas if Phase 1 shows them; CPU compute for short prompts.

**Phase 5: short prompts (3k): the user's decision** on tier / CPU experts / placement / read path, with Phase 1's numbers for how much each would buy.

## 4. Honest projection (15k, ranges, to be replaced by the Phase 1 table)

| after | wall | tok/s | share of Strata |
|---|---|---|---|
| now | 14.6 s | 1,040 | 42% |
| Phase 2 (graphs, overhead) | 11-12 s | 1,300-1,400 | ~55% |
| + Phase 3 | 9.5-10.5 s | 1,450-1,600 | ~62% |
| + Phase 4 | 8.5-9.5 s | 1,600-1,800 | ~70% |

**Beating Strata at 15k needs another ~2.5-3 s of kernel time beyond what is visible today: that is what Phase 1 is for.** Checkpoint after Phases 1-3: if the budget table shows a closable 3x compute gap, go on to beat it at 9.7k and 15k; if it shows Strata doing less work (a smaller model, fewer attention blocks, skipped layers), say so and set the target as "within X%" with the reason, not a number picked to look good.

## 5. Do not retry (measured, STATUS 36, 46-48, 51, 52)

fp16 tensor-core vs int8 vs fused gate/up+SwiGLU vs 48/64-token groups for the MoE products *at 40 tokens an expert* (all within 1-2%), the tensor-core weighted sum written with scalar V loads, dense-attention scores on tensor cores, `SHADOW_DELTA_PF`, bigger passes alone (STATUS 14).

## 6. Protocol for every step

Machine idle (no `strata`, no other server, free RAM about 49 GiB); A-B-B-A pairs on fresh servers, each server one prompt of each size with a random id and no prompt cache; kernel time from Nsight Systems (graphs off) for anything under 5%; export to a fresh `.sqlite` name; gates before a default changes: `--exact` hash 162a8e07, `quality_test` 12/12, `task_check` >= 42/48, `tools/needle_test.py` (single and `multi`), `tools/prefix_restore_test.py`, drift test, workspace tests; commit per step with its STATUS entry.
