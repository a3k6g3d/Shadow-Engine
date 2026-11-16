# Plan: a self-tuned, uncensored EXL3 for Shadow (2026-11-15)

Goal: the best quality that still decodes at the single-card speed (~61 tok/s on the varied-36 bench, STATUS 154) on this PC, from an abliterated source, with quantization
choices made for this engine and this machine instead of taken from someone else's recipe.  "Self-tuned" here means *quantization* tuned (bit allocation, calibration data,
per-expert Hessians); fine-tuning a 180 B-parameter model is not possible on 12 + 16 GiB.

## What the measurements fix

| Fact | Number | Where |
|---|---|---|
| Speed does not depend on bpw while the experts stay in RAM + VRAM | 60.9 tok/s at 2.0 bpw and at 2.5 bpw, single card | STATUS 154 |
| Past the fast tier it falls | Lygodactylus 3.05 bpw: 45.1 tok/s (-26%), ~43 GiB of experts against ~37 GiB of RAM tier (33) + VRAM cache (4), 160 s load | `bench/lyg3-single-varied36-2026-11-15`, STATUS 156 |
| The fast tier | RAM tier up to 33 GiB (default, free RAM 46 GiB, floor 3.5) + ~4-5 GiB VRAM cache on the 5070 | `main.rs`, bench logs |
| Slot size | 1.21 MiB at 2.05 bpw, 1.78 MiB at 3.05 bpw, 512 experts x 48 layers | bench logs |

So the target is **about 2.5-2.6 bpw for the experts** (35-37 GiB resident): ghost-actual's size.  The possible gain over ghost-actual is therefore *quality at equal size*
(calibration, allocation), not capacity.  Anything above ~2.7 bpw loses speed on one card unless the tier grows (the pair's second-adapter limit, STATUS 155, blocks that for now).

## Tools on this PC

* `I:/tools/exl3-venv`: Python 3.11, **torch 2.14.1+cu130**, sees the 5070 and the 5060 Ti (compute capability 12.0), a matmul runs.
* `I:/tools/exllamav3`: v1.6.7, has `Qwen4ExpForConditionalGeneration` (`architecture/qwen4_exp.py`), `convert.py` with `-cd` (calibration file), `-mae` (a Hessian per expert),
  `-rcp` (per-tensor recipe), `-cb mul1` (what Shadow's kernels need), `util/recompile.py --override` (take tensors from several quants of the *same* source).
  Installing the extension: `build_exl3.bat` (vcvars64 + `DISTUTILS_USE_SDK=1`, `pip install --no-build-isolation .`), log in `%TEMP%\exl3_build.log`.
* Sources: `orcarouter/Qwen3.8-Flash-Next-Uncensored` (BF16, 180e9 parameters, ~360 GB), the base of both Lygodactylus checkpoints (`base_model` in their model cards).
  Not the base of ghost-actual (not confirmed).  Free space: F: 2.7 TB, G: 1.2 TB, I: 430 GB, J: 115 GB.

## Phase 1: no download, minutes to hours: a mix of the Lygodactylus 2.05 and 3.05 bpw quants

`util/recompile.py --override mix.yaml` builds a checkpoint taking each tensor from whichever source the YAML names.  Both quants are of the same weights, so they mix.
* Rank every expert tensor by how far its 2.05 bpw version is from the 3.05 bpw one (`||W2 - W3|| / ||W3||` from the dequantized tensors: the 2.05 bpw error, as the 3.05 bpw
  one is much smaller); upgrade the worst tensors, and the first and last layers and the attention / shared-expert / dense tensors, until the experts reach ~2.5-2.6 bpw.
* Output on J: or I: (62-68 GB each; free J: 115 GB).  Measure it exactly as ghost-actual was measured: speed (`bench_arms.ps1`, 36 prompts x 2 runs), then quality
  (`tools/quality_test.py`, `tools/task_check.py`, `tools/long_agree.py`, `tools/needle_test.py`, the perplexity text of STATUS 147), against Lygodactylus 3.05 bpw as the reference.
* Pass criteria: speed within the ~3% floor of 60.9 tok/s, and quality better than ghost-actual 2.50 (and than Lygodactylus 2.05) on the same suites.
* Risk: low (files only); the mix of K and K+1 tensors in one layer needs Shadow's per-tensor rates (it has them: `Rate{bits, half}`).

## Phase 2: a real conversion, needs the BF16 download (asks the user first)

1. Download the source to F: with `hf download orcarouter/Qwen3.8-Flash-Next-Uncensored` (~360 GB; time depends on the line, earlier downloads here ran 1.5 MB/s to tens of MB/s).
   **Needs an explicit yes (a large download).**
2. Calibration data from Shadow itself: serve the best resident model, feed it a set of prompts of the kind this PC is used for (agent tool use, code, chat, the languages used) and
   keep the model's own replies; pack the token rows into the safetensors format `convert.py -cd` reads (check `sc_trace.py` / `convert.py` for the layout first).  The prompt part is
   free (prompt speed ~1,100 tok/s); 250 rows x 2,048 tokens = 512 k tokens, a few hours of decode if all of it were generated, so most rows are prompts + shorter replies.
3. Convert layer by layer (it streams; the 64 GB of RAM never holds the whole model):
   `python convert.py -i SRC -o OUT -w WORK -b 2.55 -hb 4 -ngb 4 -mb 2 -vb 4 --out_scales always -cb mul1 -cr 250 -cc 2048 -cd trace.safetensors -mae 0.01 -d 0,1`
   (the Lygodactylus recipe plus our calibration file and a Hessian per expert).  Time is unknown: the author used four GPUs; do a dry run on a few layers first and extrapolate
   before committing days.  The sensitivity pipeline (`sc_measure`, `sc_optimize`) holds the unquantized model in memory (~360 GB) and is documented as untested on sparse models:
   **not used**; the allocation is the Phase 1 ranking plus `-rcp`, if the converter accepts it for this architecture.
4. Evaluate against the Phase 1 mix, ghost-actual 2.50 and Lygodactylus 3.05 on the Phase 1 suites; keep it only if it wins at equal speed.

## Other constraints

* A model is never loaded outside `tools/run_capped.ps1`; the conversion is Python, so it gets the same RAM floor by running under the same wrapper's rule (kill below 3.5 GiB free).
* Disk: the work directory of a conversion can hold tens of GB of Hessians and partial layers: put it on F:.
* The exllamav3 build must be finished and smoke-tested (import the extension, load a small EXL3 model on one card) before Phase 2 is planned further.
