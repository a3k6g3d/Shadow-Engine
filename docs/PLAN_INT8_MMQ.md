# Plan: int8 tensor-core kernels for the prompt path (2026-11-05)

Why: Strata 0.2.11 measured on this PC with the same model files is 1,261 / 1,844 / 2,447 tok/s at 3k / 9.7k / 15.3k; Shadow is 569-682 / 991-1,143 / 1,018-1,100 (STATUS 41). A fit of Strata's three timings is `T = chunks x 1.8 s + tokens x 0.18 ms`: its compute is 4-5x faster than ours per token (0.18 ms against 0.7 ms), and the 1.8 s per 8,192-token chunk is the time to push its 47 GiB of experts over PCIe. The rest of Shadow's kernel time (10.8 s at 15k): grouped expert matrices 30% (IQ4_NL / IQ2_S / IQ3_XXS / IQ3_S / Q2_0), dense projections 24% (Q6_K / IQ4_XS / Q5_K / Q4_K / BF16), attention 11%, DeltaNet 8%, norms / mixes / misc ~27%.

Nsight Compute (STATUS 37, `bench/ncu_k_*.ncu-repz`): both families are balanced fp16 kernels at ~45% of the tensor pipe's peak (issue slots 36-65%, 80 registers, occupancy 50%), and fp16 with fp32 accumulation runs at half rate on a GeForce part. Dequantizing to fp16 costs ALU and shared-memory traffic on top of that. llama.cpp's MMQ keeps the weights as int8 (a table lookup or shift per value), quantizes the activations to int8 once per 32 values with a scale, and multiplies with `mma.sync ... s8` into int32 accumulators, applying the scales per 32-value block: twice the tensor rate and no fp16 dequant. Strata's NVIDIA prompt path is built on it ("K-quant experts read the prompt on llama.cpp's MMQ kernels").

## Sources we may use

* llama.cpp (MIT): `J:\Strata-work\build-0145\_deps\strata_llamacpp-src\ggml\src\ggml-cuda\` - `mmq.cuh` (1,605 lines: tile loaders per quant type, int8 mma dot products, the MoE-capable `mul_mat_q` with `ids_dst` / `expert_bounds`), `quantize.cu` (`quantize_mmq_q8_1`), `mma.cuh`. Pinned at 3cf03257f (2026-09-20).
* Strata (MIT, `J:\Strata-work\Strata-up029\src\prefill\`): `moe_mmq.cu`, `moe_mmq_tab.cuh`, `mmq_expert_bench.cpp` (how it drives MMQ for experts), `moe_fused_iq.cu`, `moe_ws_*.cu` (dp4a "weight-stationary" kernels, aimed at Pascal: hard-coded 1280 x 2560 shapes; not the model for a 5070).
* The memory rule: borrowing from Strata is allowed; Shadow stays its own engine.

## What our experts and projections look like (the integration constraints)

* Experts live in two places per layer: the VRAM cache arena (arbitrary slot addresses) and the stage half (`half + e * slot_bytes`); a grouped launch reads a per-group table of addresses (`wtab`) and a record of up to 32 tokens (`gtab`: count, input rows, output rows). ggml's MoE MMQ assumes one contiguous `[expert][row][k]` tensor with a channel stride, so the per-expert base pointer must come from our table (a small edit in the kernel's copy).
* Activations today are f32 rows (`bb.eo`, `bb.act`, ...); MMQ needs them as `block_q8_1_mmq` (int8 x 128 + scales per 128 values), quantized once per layer per matrix input (a cheap elementwise kernel: it reads f32, writes ~28% of the bytes).
* Output stays f32. Fast-mode only (`--tensor-prompt`): the int8 activations change the numbers (~0.4% per element); `--exact` and decode keep the existing kernels.

## Options for getting the kernels

* **A. nvcc-built cubin of ggml's MMQ**, loaded with `cuModuleLoadData` like our NVRTC modules, with `extern "C"` wrappers that call `mul_mat_q_process_tile`-style device code, and the expert base pointers read from our table. Least new kernel code, but pulls ggml-cuda's headers (`common.cuh`, `mma.cuh`, `vecdotq.cuh`, ...) into a build step (nvcc + MSVC; both are installed: nvcc 13.5, VS 18 Community, `sm_120` compiles and runs here). The cubin is committed or built by a script; the engine loads it at start.
* **B. Our own int8 kernels in NVRTC**, written from ggml's tile loaders (`load_tiles_iq4_nl`, `..._iq3_s`, ...) and `mma.sync.m16n8k32.s8`: no new toolchain, full control over the grouped-table launch, but all the tuning is ours.
* Decision rule: try A's standalone harness first (one afternoon's bounded effort, three compile attempts). If ggml's headers drag in more than they are worth, go to B for IQ4_NL.

## Phases and gates

1. **Feasibility (IQ4_NL grouped, standalone).** A harness with fabricated data: the shape of one expert product (gate / up: rows 1,280 x K 2,560; down: rows 2,560 x K 640 at expert_ff 640 / hidden 2,560 as in the profile), 32-64 tokens a group, 512 groups, timed against our `mma_grouped_iq4_nl` (1.14-1.56 ms a launch at 1,024-2,048 rows). **Gate: >= 1.5x faster, and its output within int8-activation tolerance (~1e-2 relative) of our fp16 kernel's.** Below 1.2x: stop, write down why, and say so.
2. **Integrate IQ4_NL** behind `SHADOW_MMQ=1`: activation quantization kernel, grouped launch from the device plan (`gtab` / `wtab`), same outputs layout. Gate: 9.5k and 15k prompt tok/s up by more than the spread (~3%) in balanced pairs; the 9.5k reply coherent; `quality_test` agreement not below the float-norm level (0.37-0.42 on 36 items); `--exact` unchanged (hash 162a8e07, 12 of 12).
3. **The other expert types** in order of kernel time: IQ2_S (9.5%), IQ3_XXS (9.0%), IQ3_S (5.1%), Q2_0 (1.9%).
4. **Dense projections** (`mma_cols_*`: Q6_K, IQ4_XS, Q5_K, Q4_K) on the same machinery (they are the ungrouped case).
5. **Re-profile** (Nsight Systems with `--cuda-graph-trace=node`, Nsight Compute on what is left) and take the next measured item: attention (11%: the sparse kernel is at 75% of L2 peak), DeltaNet (8%), the memory-bound norms and mixes (fusion).

## Expected result (honest)

If IQ4_NL reaches 1.5-2x on its 9% share and the others follow, grouped experts + dense projections (54% of kernel time) shrink by a third to a half: 10.8 s -> ~7.5-8.5 s of kernels at 15k, i.e. ~1,500-1,900 tok/s at 15k and 9.5k near 1,400-1,700. That is 60-75% of Strata at 15k and ~80-90% at 9.5k; 3k stays bound by the SSD until the tier changes (the user's decision, 36 GiB for now; 38 GiB measured +7% at 3k). Closing the rest needs the same kind of whole-engine tuning Strata has had for months.

## Rules that stay

Measure; a tie is a loss; what loses stays opt-in; every model load under `tools/run_capped.ps1` (floor 3 GiB); the tier stays at 36 GiB; DRAM is at its stable maximum; never kill the user's own processes; no stash / reset / branch switches; a dated STATUS entry for every experiment, wins and losses alike.
