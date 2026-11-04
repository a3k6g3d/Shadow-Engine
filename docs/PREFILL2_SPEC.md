# Prompt path 2 ("prefill2") - spec and staged plan, 2026-11-04

Goal: read long prompts (9-15k tokens) at Strata's rate on this PC (733 / 934 / 1,422 tok/s at 3k / 9.6k / 15k) instead of ~310 / ~450 / ~460 now.
Rust for everything except the CUDA kernels (NVRTC compiles CUDA C; nothing else can).  Built beside the current path, opt-in until it wins on the same prompts;
`--exact` stays the reference (reply hash 162a8e07 on `hash9k.py`, default-mode hash f35749c5 on a 9.5k prompt, `tools/quality_test.py`, `tools/task_check.py`).

## What the measurements say (all in docs/STATUS.md 2026-11-04, entries 1-13)

- Model: 48 layers (36 DeltaNet, 12 attention), hidden 2560, 4 hyper-connection streams, 512 experts / top 10 / expert_ff 640, shared expert, sparse indexer attention.
  RTX 5070 (12 GiB, sm_120, PCIe 4.0 x16 = 28 GB/s), 64 GB DDR4-2933 (at its stable maximum: never a lever), 36 GiB pinned expert tier.
- A pass copies ~43 GiB of experts whatever its size.  The copies run at ~14 GB/s (13.3-15.3 per layer copy, whether or not a kernel runs): the host's DRAM is
  shared with the CPU's file reads for the experts that are in neither the VRAM cache nor the RAM tier (115-180 per late layer).
- Timeline of a 9.5k prompt (Nsight Systems, `tools/nsys_pass_summary.py`): kernels busy 51-55%, copies busy 44-63%, both only 21-27%, **neither 13-23%**; gaps of
  10-50 ms with no kernel running: 124 of them, 3.4 s of an 11 s window (one a layer).
- What did NOT help (each a tie, so off): copy a layer earlier, float index scores, float sparse attention (the kernel is FP64-bound, 74% of the pipe, and still off the
  critical path), IQ4_NL tensor cores, file views (-58%), rows past ~4k (4,096 -> 6,400 real rows: 452/472 vs 462/450 tok/s).
- What helped: rows 2,048 -> 4,096 (+14-24%), sparse-attention selection on the GPU (+20%), parallel host routing (+3-5%), buffer sharing (+9-12%).
- **The pass is a chain with a host round trip in every layer**: attention / DeltaNet kernels -> router logits down -> the host routes ~4,000 tokens (f64 exp over 512
  logits each), builds the grouped plan, uploads it -> experts launch -> next layer.  The GPU idles ~8-10 ms of host work a layer after the logits arrive, and cannot
  run ahead because the host cannot queue layer l + 1 before layer l's routing is back.  Strata plans on the device inside a captured window (Strata prefill
  source: J:\Strata-work\Strata-up029\src\prefill\prefill.cpp: 8k-token chunks, a ring of streamed experts, int8 MMQ experts, cuBLAS dense, chunked GDN).

## Design

1. **Device-side routing and planning for a long prompt's MoE** (stage 1, the main idea).  In the staged path a layer's whole expert set is copied whatever the tokens
   route to (the host's routing only decides the grouped plan and the cache's LRU touches), so nothing the host does depends on the routing.  New kernels:
   - `route_topk_rows`: per token, `shadow_model::forward::route` bit for bit (f64 exponentials, the sum in index order, top 10 by (score desc, index asc), weights
     `(ex/z)/tz`): one thread a token is enough (6,400 tokens x 512 exps is nothing for the GPU).
   - `plan_groups_rows`: counting sort of the (token, rank) pairs by expert (histogram, scan, scatter), chunks of 32 tokens a group (the mma kernels' `chunk_n`),
     written in the table layout `routed_grouped` / `mma_grouped_*` read (addresses per group from a per-layer expert -> address table the host fills: 512 u64s that
     do not depend on routing; resident experts are phase 0, copied ones phase 1 via a flag array).
   - the grouped kernels get the group count from a device word and early-exit past it (grid = the upper bound ne + m*k/32), so no download.
   Then the host queues a whole pass ahead of the GPU: the layer-ahead copies run continuously, the CUDA graph of a layer can include the experts.
   Costs to handle: `routing.add` (profile recording) and `cache.touch` (LRU) become approximations or are dropped in this path; `--record-routing` falls back.
2. **Leaner per-row memory** (stage 3): the biggest rows are `eo` (17%, k x hidden fp32 expert outputs: accumulate weighted into `y` instead), `r/rn` (20%, 4 streams x 2560
   fp32: rn can be recomputed), the two attention/DeltaNet sets (already shared), `eg/act` (13%).  Target ~250 KB a row (from ~410) = 8k rows in ~2 GiB.
3. **A ring of expert slots instead of two whole-layer halves** (stage 4, only if stage 1 leaves copies unhidden): ~250-300 slots, frees ~2 GiB.
4. **Int8 tensor-core expert matmul** (stage 5, only if the kernels become the limit): quantized weights decoded to int8 in the kernel, int8 activations (llama.cpp's MMQ).

## Stages and acceptance (each measured as the earlier entries were: interleaved A/B, 3 pairs, default build, ctx 24,576, 3k / 9.5k / 15k, dated STATUS entry)

| Stage | What | Accept when |
|---|---|---|
| 1a | `route_topk_rows` kernel + a test against `route()` on real router logits (`tests/`), bit for bit | all tokens identical on 10 prompts |
| 1b | `plan_groups_rows` + grouped kernels reading a device count; `SHADOW_PREFILL_DEVPLAN=1` | `--exact` hash 162a8e07 and default hash f35749c5 unchanged; speed >= the host path |
| 1c | no logits download, no host routing in the staged path; host queues ahead | prompt tok/s up (the 8-10 ms a layer gone), timeline "neither" down |
| 2 | graph the layer incl. experts (the shape is now data-independent) | launch gaps gone from the timeline |
| 3 | lean rows | more rows without new VRAM; rows 6k+ real |
| 4-5 | ring, int8 MMQ | only on evidence |

## Where the code is

`crates/shadow-cuda/src/session.rs`: `moe_batch` (router download, `route_all`, tail skip / accept, then `moe_staged`), `moe_staged` (copy pipeline, `grouped_plan`, `routed_grouped`),
`stage_copy`, `stage_pipe_read`, `big_batch_on`, `make_bb`.  Kernels: `kernels/forward.cu` (`select_blocks_rows` is a model for a block-per-row device kernel), `kernels/matvec.cu`
(`mma_grouped_*`, `matvec_grouped_*`).  Host router: `crates/shadow-model/src/forward.rs` `route`.  Tools: `tools/nsys_pass_summary.py`, `tools/task_check.py`, scratch
scripts in %TEMP% (ab.sh, qt.sh, hash9k.py, p9k.py, p15k.py - recreate from STATUS if gone).  Rules: every model load under `tools/run_capped.ps1` (floor 3 GiB with the 36 GiB tier);
commit only files you changed (Gemini edits the same tree, uncommitted: CHANGELOG.md, Cargo.toml, README.md, shadow-bench, readpool.rs, real_model.rs, shadow-serve engine.rs / ui.rs,
shadow main.rs, USAGE.md); DRAM is at its stable maximum; a tie is a loss and what loses stays opt-in.


## Status (2026-11-04, end of the day)

- **Stage 1 (1a-1c) done and on by default** (`SHADOW_PREFILL_DEVPLAN=0` off): routing and planning on the device, host runs ahead, three read-ahead buffers, copy waits for the layer reading its half. 9.5k 446-456 -> 490 tok/s, 15k 461-465 -> 514-517 (+9-11%), all exact/fidelity checks unchanged. Timeline after it (`bench/nsys-stage1-2026-11-04`, 9.65 s window): kernels busy 60.6%, copies busy 64.6% (15.5 GB/s while active), both 34.1%, **neither 9.0%, copy with no kernel 30.5%**; the 10-50 ms gaps left are the GPU waiting on a layer's copy.
- Rows past ~4k still do not help (drafter off: 4,096 vs 6,400 real rows 519/497 and 428/427 vs 495/481 and 506/484, noisy).
- **Next**: the copy engine is idle 35% of the time and the GPU waits on it 30%: the copy of layer l + 1 starts only after the previous copy ends and the layer before the last has combined (two halves), so a finer pipeline (the ring: stage 4) or a copy that does not need the whole layer first is where the overlap is; kernel time is ~0.75 ms a token (Strata's whole cost per token is ~0.7 ms), so the kernels need to shrink too to pass ~800 tok/s.
