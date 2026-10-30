# Speculative decoding in Shadow

How a pass checks several tokens at once, where the drafts come from, and what was tried. The measurements behind each
statement are in [STATUS.md](STATUS.md); how they were taken is in [BENCHMARKING.md](BENCHMARKING.md).

## Why it helps here, and what limits it

Decoding one token streams every active expert of 48 layers through the GPU, and most of those experts are not in VRAM:
the model's 24,576 routed experts are ~47 GiB, VRAM holds ~4.4 GiB of them, a 30 GiB pinned RAM tier holds most of the
rest, and a VRAM miss is a PCIe copy at ~22 GB/s (a miss in both is a read from the GGUF). A pass over several tokens at once reads the *union* of their experts once, so four
tokens cost far less than four passes. That is the whole opportunity: tokens that are checked together share their expert
traffic. The limit is the same traffic: a verify pass of 4 to 6 positions takes 50 to 60 ms, about one exact single-token step
(~49 ms at 20 tok/s), and each extra position adds ~4 ms, so it must be accepted often enough to pay for itself (the
fastest setting is 3 drafts, not 7).

## The verify pass

`GpuSession::step_batch` runs up to 64 tokens through the layers *layer-major*: for each layer, all tokens' projections
(`matvec_cols_*`, bit-identical per token to the single-token kernels), then the union of the experts they route to, grouped
so an expert's weights are read once for all its tokens (`matvec_grouped_*`). The same code is the prompt prefill.

For speculation the engine does, per round:

1. take the model's next token `t` (the argmax of the last logits) and ask a drafter for up to K tokens after it;
2. snapshot the DeltaNet state (a few MB; attention KV is append-only), run `step_batch_verify([t, d1 .. dK])`;
3. accept drafts while `argmax(logits[i]) == d(i+1)`; the pass already holds the logits for the first rejected position,
   so even a round with no accepted draft produces one token;
4. if some were rejected, roll back: restore the snapshot and replay only the accepted positions' recurrence updates from
   inputs the verify pass saved (no second forward pass).

So speculation by itself never changes an answer: a draft is kept only if it is exactly the token the model would have
chosen. Two opt-in-by-default *inexact* modes change the model's own computation and are the reason answers can differ
from `--exact`:

* `--tail-skip R`: a routed expert of rank >= R (by router weight) is skipped unless it is already in VRAM. Strata's
  `STRATA_ROUTE_TAIL_SKIP` does the same; it is about a quarter of Strata's speed.
* `--verify-resident`: positions after the first in a verify pass use only experts already in VRAM.

Neither is part of the exact path; `--exact` removes both.

## Where drafts come from

| source | how | when it is used |
|---|---|---|
| prompt lookup | the longest recent n-gram (8, 6, then `SHADOW_LOOKUP_MIN`=4 tokens) that repeats earlier in the conversation proposes what followed it | tried first; used when it finds >= 2 tokens |
| DFlash 2 (`--dflash`) | a 5-layer block-diffusion drafter reading features from the main model; one pass drafts a block | the default drafter |
| MTP head (`--mtp`) | a one-layer draft model (`models/mtp-basalt-ft1`, borrowed from Strata), chained token by token | when `--mtp` is given |

### DFlash 2

A port of SGLang's `DFlash2DraftModel` (`crates/shadow-cuda/src/dflash.rs`, `session_dflash.rs`, the `df_*` kernels in
`kernels/forward.cu`; the model is `models/dflash2/model.safetensors`, 81 BF16 tensors, 1.2 GB).

* **Features.** After layers 5, 15, 25, 35 and 45 of the *main* model, the residual (4 hyper-connection streams) is averaged
  over the streams; the five 2560-vectors are concatenated (12800 floats) for every position the main model reads, in
  prefill and decode. `fc` (12800 -> 2560) and `hidden_norm` turn that into the drafter's context row.
* **Context K/V.** For each of the drafter's 5 layers, the context row goes through `k_proj`/`v_proj`, K is RMS-normed per head
  and RoPE'd (NeoX, theta 1e4), and both are written to a ring of 2064 positions (window 2048). Rolled-back positions are
  simply overwritten when they are read again.
* **A draft round.** The block is 8 rows at positions `n .. n+7` (n = the anchor's position): row 0 the token about to be
  stepped (the anchor), rows 1..7 the mask token (248077), embedded with the main model's own embedding table. Each layer:
  RMSNorm, a *grouped dynamic 2-tap convolution* along the block (a 640-wide projection of the input gives per-group
  coefficients on top of learned bases; groups of 16 channels), q/k/v, per-head norms, RoPE, **non-causal** attention of the
  8 rows over the ring plus the block itself, `o_proj`, the same convolution on the output, then the MLP with its own
  convolution. The result is normed once more.
* **Candidates and the walk.** Rows 1..7 go through the *main model's* output head (`output.weight`, never loaded twice);
  the 16 largest logits of each row are the candidates (`df_top16`). A low-rank bilinear selector scores each candidate
  as `logit + <pred[prev] * W h, succ[candidate]>` (two 248320 x 256 codebooks), and the greedy walk (`df_walk`) picks one
  candidate per row left to right, each conditioned on the previous pick. The walk also returns the chosen candidate's
  probability among the 16; `--dflash-min-p` cuts the chain after a draft below it.
* **Cost.** A round is ~6.3 ms (matrices Q8_0): the output head 2.0 ms, the MLPs ~1.3, the projections ~2, attention 0.4. It
  adds 217 expert-cache slots compared with BF16 because it frees ~0.4 GB of VRAM.
* **Checked.** `tools/dflash2_ref.py` recomputes a draft in CPU torch from the features Shadow fed it: block hidden state
  max abs error 6e-5 (relative 1e-6), selector picks identical (use `SHADOW_DF_DUMP=dir`).

### Tuning the draft length

Accepted-token statistics (`draft position right/tried` is printed after every answer) show why 3 is the optimum: with the
stock drafter on varied text, draft position 1 is right ~75% of the time, 2 ~75% of those, 3 ~83% of those, and so on, while
each added position costs ~4 ms of verify time on top of the pass. `--dflash-min-p 0.5` trims the tail when the selector is
unsure: +2.0% on varied text, +5% on the six-prompt bench. The autotune loop (`tools/autotune.py`) re-checks both whenever
something underneath changes.

## What did not work (so nobody repeats it)

| idea | result |
|---|---|
| fine-tune DFlash 2 on Shadow's own features (LoRA, `tools/dflash2_train.py`) | looked like +25% accepted tokens on a held-out set, but the set counted positions inside the *user's prompt*; on generated text only, stock 3.24 vs tuned 3.35 and a retrain ended at 3.24: no gain with ~150k generated tokens. Total throughput on varied text: stock 46.7 tok/s, tuned 42.8 to 44.6. |
| candidates from only the 106k-token draft vocabulary (the MTP head's list), saving ~1 ms/round | position-1 hits 69% -> 59%, -25% tok/s. Opt-in only via a `draft_vocab.bin` beside the weights. |
| longer blocks (4 to 7 drafts) | each is slower than 3 on both benches; the verify pass grows faster than acceptance |
| merging projections into one launch, fusing the hc up-projection and delta norm | bit-identical, measured a tie (off) to -1.1% (all on): the pass is not waiting on launches. `--fuse` keeps them. |
| zero-copy PCIe reads, FP64 removal in the long sums | see STATUS.md: FP64 removal was kept (FP64 is 1/64 rate on this GPU); zero-copy refuted |
| MTP head vs DFlash 2 | DFlash 2 by total throughput ~+2% on varied text, ~+6% on the six-prompt bench |

## Training pipeline (kept for the next attempt)

`SHADOW_DF_DUMP=dir shadow serve ...` writes every position's feature row and token. `tools/dflash2_train.py` loads those,
the main model's embedding and head (`shadow dump MODEL token_embd.weight FILE`, `... output.weight FILE`), trains LoRA
adapters over the frozen BF16 weights with the loss the drafter is used for (cross entropy through the main model's head
plus cross entropy over the selector's 16 candidates), scores the *free-running accepted prefix* of the greedy walk on held-out
conversations, and merges the result back to BF16. **Anchors must be drawn from generated spans only**
(`generated_mask`): drafting never starts inside a prompt.
