# Plan: pass Strata on prompt speed (2026-11-05)

Target (this PC, `docs/STATUS.md`): Strata 733 / 934 / 1,422 tok/s at 3k / 9.6k / 15k prompt tokens. Decode is already ahead (52-56 against 47.7 tok/s) and is not part of this plan.

## Where we are (default build, ctx 24,576, all checks green: `--exact` hash 162a8e07, `quality_test` 12 of 12, `task_check` 42/48)

| prompt | ours now | Strata | time now | time to beat |
|---|---|---|---|---|
| 9.5k | 677 first / 690 repeat | 934 | 13.8 s | < 10.2 s (-26%) |
| 15k | 680 | 1,422 | 22.2 s | < 10.6 s (-52%) |

Every number below is measured (STATUS entries 25-29), 15k steady-state timeline `bench/nsys-steady15-2026-11-05` (read with `tools/nsys_last_run.py`, `tools/nsys_gap_why.py`):

* 15.4 s sparse-attention window: **kernels busy 67%** (10.3 s), copies 11%, **GPU idle 33% (5.0 s)** in ~500 gaps.
* Kernel time: grouped MoE matrices (`mma_grouped_*`) 40.5%; sparse attention `attention_group_sel_rows_v2` 17.6%; `index_scores_rows_f32` 14.3% (300 us a launch at 15k); `mma_cols_*` 8%; `hc_write_rmsnorm_rows` 3%; dense attention 2.7%; the rest small.
* The file-only experts (5,625 of 24,576, 11.6 GB a pass) are read from the SSD at 1-3 GB/s every pass (the page cache cannot hold them beside the 36 GiB pinned tier); a pass is now the whole prompt (9.5k and 15k are one pass), so that is 4.8 s of reads, hidden under the compute as long as the compute is longer than that.
* The idle gaps are ~16 ms stalls of the host's `cuLaunchKernel` (one every 15-16 ms) with the GPU idle; thread priority, a stream-query flush, the 1 ms timer, 16-48 read workers do not move them (STATUS 29).

## The arithmetic

Time = kernels + idle. With the idle gone and kernels at today's speed: 9.5k ~12 s (790 tok/s), 15k ~17 s (890). Both Strata numbers need the kernels faster too. Savings are estimated from the measured shares above; each lever has a gate (measure, keep only if it wins by more than the ~1% run-to-run spread on interleaved pairs, `--exact` hash and `quality_test` unchanged, STATUS entry).

| # | lever | what | estimated saving (15k / 9.5k) | cost / risk |
|---|---|---|---|---|
| A | **kill the idle gaps** | (1) one CUDA graph per layer that works for every sub-chunk: stage each sub-chunk's residual streams through one fixed buffer (a 42 MB device copy, ~0.1 ms) so graph keys (today: layer, rows, `bb.r` base pointer) repeat, and launches fall from ~36k to a few thousand; (2) if stalls survive, a GPU-scheduler trace (GPUView / WPA) of one stall to see who holds the GPU; (3) user-side diagnostic: hardware-accelerated GPU scheduling off, GPU apps closed, one run | up to 5.0 s / 1.8 s | graphs: medium (capture per layer segment exists); trace: needs the user's tools |
| B | **grouped MoE kernels** | Nsight Compute on `mma_grouped_iq4_nl / iq2_s / iq3_xxs / iq3_s` (needs the elevated script, as for attention); today ~10x off the tensor peak for the FLOPs (0.94 TFLOP a layer at 9.5k); likely dequantization-bound: stage dequantized tiles through shared memory with a software pipeline, then int8 MMA for the integer quants (PREFILL2 stage 5) | 1.4-2 s / 0.9-1.3 s (1.5-2x on 40% of kernel time) | high; the largest single kernel prize |
| C | **attention family** | `attention_group_sel_rows_v2`: all 12 heads of a KV group in one block (K/V re-read /3: L2 was 75% of peak) ~-30%; `index_scores_rows_f32` (L1-bound, 98.6%): tile the keys through shared memory, 2x | 1.3 s / 0.8 s | medium; profile exists (`bench/ncu_v2_*.ncu-repz`) |
| D | small kernels | fuse `hc_write_rmsnorm_rows` with its neighbors, `mma_cols_*` tuning, `swiglu` into the down projection | 0.3 s / 0.2 s | low |
| E | reads stay hidden | after A-D the compute at 9.5k is ~9 s against 4.8 s of reads: still hidden; if a faster compute exposes them, read-ahead of two layers with per-layer tickets in `ReadPool` | 0 now | watch item |

## Projection (honest)

* After A (best case): 9.5k 12 s, 15k 17 s.
* After A-D: 9.5k ~9.8 s = **~970 tok/s (passes Strata's 934)**, 15k ~13.3 s = ~1,140 tok/s (Strata's 1,422 needs 10.6 s: not reached).
* 15k needs another ~25% beyond A-D: the open candidates are attention's growth with context (index scores and sparse attention are 32% of kernel time at 15k and scale with it), a second look at the grouped kernels at int8 peak, and fewer layers' worth of dequantization by caching dequantized expert tiles across the sub-chunks of a layer (the same experts run in every sub-chunk: today each sub-chunk of 1,024 rows re-reads and re-dequantizes the layer's whole expert set; larger sub-chunks amortize it but cost VRAM). **This last one is the first thing to measure in B**: at 1,024-row sub-chunks every expert matrix is dequantized 15 times a layer at 15k. Already visible in the traces: `mma_grouped_iq4_nl` costs 1.68 ms a launch at 2,048 rows (0.82 us a row) and 1.15 ms at 1,024 (1.12 us a row), +36% a row, which the single pass and the fewer idle gaps more than paid for; sub-chunks that are bigger without more VRAM (the temporaries are what the sub-chunk size costs) would take that back.

## Order of work

1. **A(1)** first: it removes the largest measured loss (5 s at 15k) and shrinks launch count for everything after; verification is the timeline (gaps of 10-50 ms from 419 to near zero) and prompt speed.
2. **B** next, with the user running the elevated Nsight Compute script once (same as for attention); then C, then D.
3. After each: STATUS entry, interleaved pairs at 3k / 9.5k / 15k, `--exact` hash, `quality_test`, decode check.

## Constraints that stay (the user's)

Measure; what loses stays opt-in; a tie is a loss; every model load under `tools/run_capped.ps1` (floor 3 GiB with the 36 GiB tier); the RAM tier is not enlarged and DRAM stays at its stable 2,933 MT/s; the user's own Shadow process is never killed; no destructive git.


## Update 2026-11-05 evening (STATUS 30-36)

* **Lever A is not a lever.** The "33% idle GPU" of the first timelines was CUDA graphs that Nsight Systems did not list (`--cuda-graph-trace=node` shows them). With graphs traced or off the prompt path has the GPU **91% busy**: it is kernel-bound. The ~16 ms `cuLaunchKernel` calls are the host waiting on a full launch queue.
* Done, all checked (`--exact` hash 162a8e07, `quality_test` 12 of 12, `task_check` 42-43/48): 12-head sparse attention kernel (+3.5%), register-blocked index scores (+3%, bit-identical), DeltaNet conv + gated norm in parallel (+2.1%, bit-identical), logits of the last row only at the end of a prompt (+5.7-8.4%: the old path also downloaded 1 GB of logits), sub-chunks of 2,048 rows (+6-8%), selection groups of 128 rows (+1.5%).
* **Now: 9.5k ~885 tok/s (Strata 934, -5%), 15k ~875 (Strata 1,422, -38%).**
* Tried and not adopted (ties or losses, STATUS 28-36): thread priority, stream-query flush, 1 ms timer, direct I/O (-50%), early/late next-layer copy, prompt-graph on/off (tie on a first run; graphs win on a repeated shape), `SHADOW_DELTA_CHUNK` (+1%), `SHADOW_DELTA_X8` (-8%), `SHADOW_FUSE` bits, 64-token expert groups (+0.3% / +1.2%).
* **Remaining kernel time, 15k prompt (nsys, graphs traced):** grouped MoE matrices ~31% (iq4_nl 1.27 s, iq2_s 1.16 s, iq3_xxs 1.05 s, iq3_s 0.69 s), dense projections `mma_cols_*` ~18% (q6_k 0.99 s, iq4_xs 0.98 s, q5_k 0.40 s, bf16 0.33 s), `delta_steps128` 6.8%, sparse attention 7.7%, `hc_write_rmsnorm_rows` 3.6%, `mma_hc_down_inject_bf16` 2.6%. **Next: Nsight Compute on the grouped MoE kernels and the dense projections** (`bench/ncu_attn_group.ps1`, six profiles, needs an elevated PowerShell): the dequantization-bound theory failed its test, so the real limiter has to be read from the counters before changing the kernels again.
* 15k needs a further ~1.6x beyond today; the MoE and dense-projection kernels (49% of the time) are the only place that much can come from.
