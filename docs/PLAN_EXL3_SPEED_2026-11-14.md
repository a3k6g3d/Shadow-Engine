# Plan: EXL3 speed (written 2026-11-14, after STATUS 143)

Standing instruction from the user: carry this out without asking for approval at each step. Stop and ask only for: pushing, deleting data, downloads, anything outside
`J:\Shadow` and the model folders, or a change that makes `--exact` or a default less faithful than measured here. Every step ends the same way: tests green, an A-B(-A) on the
server with the off switch, a numbered STATUS entry with the measured numbers (wins and losses), a commit of only my files.

## Where it stands (2.05 bpw base checkpoint on J:, default fast mode, one fresh server per arm)

| | EXL3 2.05 | IQ3_XXS GGUF |
|---|---|---|
| prompt 2k / 9.5k / 20k (tok/s) | 665 / 935 / 978 | 481 / 975 / 1,001 |
| decode (tok/s) | 37-45 | 39-53 |

Decode round (Nsight, 50.4 ms): kernels 33.4 ms (dense `matvec_cols_exl3` 10.6 ms = 300 launches of 35 us; grouped experts 9.6 ms = 153 launches of 63 us), host stalls 14 ms
(`cuEventSynchronize` 2.5, `cuMemcpyDtoH` 1.8). Prompt pass (9.5k): kernels 66%, GPU idle 32% (n-gram row reads and host planning).

## A. Decode (largest gap; verify rounds of m <= 8 tokens)
1. **Grouped expert kernel specialised on the group size** (`m2s2` 184/112 us against 200/161 for gate+up / down at 2 tokens a group; the plan knows the largest group): pick M = 2 / 4 / 8 per launch.
2. **Tile-word prefetch in the multi-input kernel** (load a round's 8 tile rows before the rotation + barrier; helped s2, hurt s4 through registers): keep only where it wins per M and KS.
3. **Split-K across blocks for the dense shapes with few blocks** (J out, K o: 20 blocks; k/v: 4; shared expert: 5): partial sums to a scratch, a combine kernel does the sum in slice order, the output rotation and `svh`. Deterministic.
4. **Same for the single-token `matvec_exl3` / `matvec_batch_exl3`** (the step with no drafter) and for the expert batch path of m = 1.
5. **Round host stalls** (14 ms): find why a verify round waits twice (event sync, DtoH of logits) and overlap the drafter with the next verify's launch.
6. **Drafter acceptance on EXL3** (50.7% in STATUS 139's run): check that the DFlash drafter / tail skip / verify rank defaults are the best for this model's routing; the placement profile (`profiles/routing-*.shrc`) is the GGUF's, make one for EXL3.

## B. Prompt
7. **Group fill**: a 2,048-row sub-chunk gives ~40 tokens an expert for a 64-token group; try `SHADOW_SUB_ROWS=4096` and a 128-token group variant.
8. **Single-buffered tensor-core kernels**: `cp.async` double buffering of the rotated rows and the trellis words in `mma_cols_exl3_pre` / `mma_grouped_exl3_pre`.
9. **N-gram rows** (GPU idle ~1 s a 9.5k pass on J:, 3.5 s on a SATA SSD): start the next sub-chunk's reads while the previous sub-chunk's layer 1 runs, keep the rows of the last pass, and put the table on the fastest drive (a loader option for a separate table path).
10. Attention / DeltaNet / hyper-connection kernels are shared with the GGUF path: out of scope unless EXL3-specific.

## C. Both
11. **Half-integer and mixed-K layers** (the abliterated 2.05 / 2.50 bpw repos): per-layer slot size is uniform today; support per-matrix K inside a layer by giving experts the layer's largest size, or sort experts by K.
12. **`shadow ppl` of EXL3 2.05 / 3.05 against the GGUFs** (quality is the point of EXL3; speed means nothing if 2 bpw is poor), and the 8-prompt `long_agree` against an `--exact` run.
13. MTP head and the vision tower (not read yet); `--exact` for EXL3 = the CPU decoder to rounding.
14. Second card: when it is installed run `tools/bringup_two_gpus.ps1` and the stage split on real hardware.

## Order and expected gains
A1-A2 (small, +5% decode) -> A3-A4 (+10-15% decode) -> B7-B9 (+5-10% long prompts, +15% 2k) -> A5-A6 (+10% decode) -> C12 (quality numbers) -> C11, C13, C14.
Out of the 33.4 ms of kernels in a decode round the dense and expert kernels are 20 ms; getting them to ~60% of the memory roofline is about 8 ms, i.e. 50 -> ~42 ms a round (+19%).
