# Plan: making the two cards faster together (RTX 5070 + RTX 5060 Ti), the hop and the pipeline

Written 2026-11-17, after the v0.1.37 pair work (STATUS 169) landed but **before it is
committed**. The pair already works and is fast on prompts; this plan is about the part that
is still slow — single-token decode — and about the one structural thing left in the stage
handover ("the hop"). Numbers marked *estimate* are arithmetic from the measurements below,
to be replaced by a measurement at the gate named beside them.

## 0. Do this first: the pair baseline is uncommitted

The whole v0.1.37 pair change (STATUS 169) is sitting in the working tree, not committed
(head is `b3344bd`, 14 files modified: `session.rs`, `engine.rs`, `pinned.rs`, `kernels.rs`,
`main.rs`, `matvec.cu`, `forward.cu`, `exl3.cu`, `CHANGELOG.md`, `Cargo.toml`, `STATUS.md`,
plus the UI/telemetry files). **Commit that pair baseline before layering anything from this
plan on top of it**, or an A-B-A below cannot be attributed to one change. Commit only the
files each step touches; leave the unrelated UI/APK edits alone.

## 1. What the hardware allows (read first, it decides the design)

| | RTX 5070 (second stage) | RTX 5060 Ti 16 GB (first stage) |
|---|---|---|
| chip / SMs / VRAM | GB205, 48 SM, 11.9 GiB | GB206, 36 SM, 15.9 GiB |
| own memory bandwidth | 672 GB/s | 448 GB/s |
| slot / link (measured today) | PCI 01:00, **PCIe 4.0 x16** | PCI 03:00, **PCIe 3.0 x4** (chipset slot) |
| host <-> device, measured | 8.74 / 8.15 GB/s | 2.83 / 3.48 GB/s |
| driver | 621.24 | 621.24 |

Consequences, all re-measured on the live machine this morning (`shadow gpus`, no model load):

* **No peer-to-peer, both ways.** `cuDeviceCanAccessPeer` returns 0 in both directions
  (`probe.rs:81`); GeForce under WDDM, and the second card sits behind the chipset anyway.
  There is **no** `EnablePeerAccess` / `MemcpyPeer` anywhere in `shadow-cuda`. So every byte
  that crosses cards goes **device -> pinned host -> device**. That is the only path today,
  and the hop is built around it.
* **The slow card runs the cheap half on purpose.** `engine.rs:380-419` swaps the order so the
  5060 Ti (36 SM, 16 GiB) is the **first** stage with layers `0..15` fully VRAM-resident — no
  RAM-tier reads on the Gen3 x4 card — and the 5070 (48 SM, 12 GiB) is the **second** stage:
  layers `15..48`, `lm_head`, DFlash, the 23 GiB RAM tier, all over the fast Gen4 x16 link.
  That split is right; do not move streaming onto the slow card.

## 2. What the hop costs, measured today (µs, median / p99 / max)

| hop | bytes | what it is | [0]->[1] | [1]->[0] |
|---|---|---|---|---|
| single token | 40 KiB | one token's 4 residual streams | 54.2 / 138.7 / 141 | 57.0 / 124.0 / 131.8 |
| same card, 40 KiB | — | the floor without the link | 32.1 | — |
| verify pass m=2..4 | 160 KiB | four tokens' streams | 143.4 / 335.8 / 352 | 115.4 / 216.1 / 242.6 |
| prompt chunk | 80 MiB | 2,048 rows (HOP_CHUNK_ROWS) | 39,087 | 30,668 |

Reading: a 40 KiB hop on the same card is 32 µs; across the Gen3 x4 link it is 54 µs, so the
**link itself adds ~22 µs** and the host bounce + two context binds add the rest. The 80 MiB
prompt chunks run at ~2.6 GB/s effective (link + bounce), close to the 2.83 GB/s measured
host->device ceiling — the bounce is not wasting bytes on big transfers, only on small ones.

## 3. Where the pair is fast and where it is slow (STATUS 169, the honest split)

The pair is **ahead** on: prompt throughput **+56.8%**, varied multi-turn combined **+19.5%**,
~9.5k prompts **+33%** vs the pre-optimisation pair, free RAM **+10.5 GiB**. It is **behind** on
one thing only: **warm single-turn decode, ~5% below the 5070 alone** (`bench/v0137-mixD18-pair-vs-single-opt`).
The reason is stated in STATUS 169 and is **not** the hop: Stage 0 (the 5060 Ti at 448 GB/s)
executes **serially ahead of** Stage 1, so a token pays Stage 0 then Stage 1 back to back.

The 40 µs decode hop is ~0.2% of a ~20 ms token — **irrelevant to decode speed**. The hop is
only a real cost on **prompts**, where 80 MiB chunks cross it. So this plan splits cleanly:

* the **hop** work (items 1-2, 5) is a **prompt** lever,
* the **pipeline** work (item 3) is the **decode** lever, and it is the headline.

## 4. The one structural thing left in the hop

`hop_pull_on` (`session.rs:3453-3470`) uses a **single** pinned buffer (`hop_buf`, allocated
once in `hop_ensure`, `session.rs:3432`). The down leg is a **blocking** `cuMemcpyDtoH`
(`session.rs:3467`) that first runs `hop_done.synchronize()` — so the next chunk's **down cannot
start until the previous chunk's up has run**, because both share one buffer. The up leg is
already async (`cuMemcpyHtoDAsync`, on `copy_stream2` when given). The prompt path already
two-half-pipelines the *compute* (`step_batch_opt`, `use_two_halves`, `rparts`, `copy_stream2`,
`SHADOW_PAIR_PIPE`, `session.rs:5688-5755`) — that is where the +56.8% came from — **but it still
hands over through one buffer**, so the two halves' hops serialise against each other.

The decode path (`session.rs:5925-5934`) does one synchronous 40 KiB `hop_pull` per pass before
Stage 1 runs, two `ctx.bind()` per hop; with m=1 there is nothing to overlap it against, and the
n-gram rows are already prefetched on a background thread (`ple_async`).

## 5. The plan, ranked by (gain x confidence) / effort

### Item 0 — Probe peer-to-peer once, cheaply (a gate, not a build)
The whole host-bounce design rests on `cuDeviceCanAccessPeer == 0`. Before building anything
on top of that assumption, spend one experiment trying to *break* it: with **Resizable BAR
enabled in the BIOS** (check `nvidia-smi -q | Select-String 'Resizable\|BAR'` first — if BAR is
not the full 16 GiB, enable it and reboot), call `cuCtxEnablePeerAccess` and time
`cuMemcpyPeerAsync` for the three sizes in §2 against the host-bounce numbers. Ada/Blackwell
GeForce under a recent driver (621.24 is current) sometimes reports peer access once ReBAR is
on, where the old GeForce rule said never.

* **If peer access works:** the 80 MiB prompt hops go direct at PCIe line rate instead of
  device->host->device — *estimate* roughly 2x on the hop, and on a ~9.5k prompt where each
  chunk hops 39 ms that is a real slice of the chunk time. Items 1-2 then become "use
  `cuMemcpyPeerAsync` instead of the bounce", not "double-buffer the bounce".
* **If it still says no** (likely, chipset slot + WDDM): record the attempt as the one failure
  of this task and carry on with the host-bounce items below.

Gate: `bench/pair-p2p-probe-2026-11-17/`, a throwaway harness, no engine change. Off-switch:
none (it is a probe). **This decides the shape of items 1-5, so do it first.**

### Item 1 — Ping-pong the hop buffer and make the down leg async (prompt lever)
Give the hop **two** pinned buffers and alternate them per chunk, so chunk *k+1*'s down runs
while chunk *k*'s up is still in flight on `copy_stream2`. Replace the blocking
`cuMemcpyDtoH` (`session.rs:3467`) with `cuMemcpyDtoHAsync` on that copy stream, and move the
`hop_done.synchronize()` from "before every down" to "before reusing a buffer that is still
being read" (the ping-pong makes that rare). This removes the serial chain in §4 so the two
halves' hops overlap the way their compute already do.

* Files: `session.rs` `hop_ensure` (3432), `hop_pull_on` (3453-3470), the `down`/`up` closures;
  `driver.rs` already has `cuMemcpyDtoHAsync`.
* Off-switch: `SHADOW_HOP_PINGPONG=0` (off) / `1` (on). Memory cost: one more pinned buffer of
  `HOP_CHUNK_ROWS * hidden * streams * 4` (~40 MiB) — cheap, and the 16 GiB card has room.
* Gate: the ~9.5k prompt bench `bench/v0137-mixD18-pair-prompt-opt2/prompt.txt`, A-B-B-A, plus
  `nsys_prompt_client.py`; and the `varied36` prompt set. *Estimate*: on multi-chunk prompts the
  down of chunk k+1 (39 ms) currently waits for up of chunk k; overlapping them recovers up to
  ~half the hop time per chunk boundary — a few percent on the longest prompts, near zero on
  short ones. Verify exactness (same tokens, same logits) before keeping.
* Risk: low. It is the standard double-buffer; the only trap is reusing a buffer whose upload
  has not run, guarded by the event.

### Item 2 — Ping-pong the verify hop too (m=2..4)
Once item 1 exists, the DFlash verify passes (m=2..4, 160 KiB, 143 µs) can pass *k+1*'s down
over pass *k*'s up. Decode hops are small against the token budget, so this is a **minor**
decode gain — do it only because item 1 made the machinery free, not as its own project. Same
off-switch, same gate family (`varied36` decode).

### Item 3 — The decode headline: pipeline the two cards across *tokens* with DFlash (decode lever)
Prompts got a two-half pipeline (STATUS 169). **Decode did not**: with m=1, Stage 0 runs the
whole head of a token, then the hop, then Stage 1 runs the tail — fully serial, which is the
~5% deficit. The fix is the decode analog of the prompt pipeline: while Stage 1 (the fast 5070)
finishes token *N*, Stage 0 (the 5060 Ti) runs ahead on **DFlash-drafted** token *N+1*, and the
draft is verified when token *N*'s real id arrives. The two cards then work **concurrently**
instead of back-to-back, which is what the pair architecture is for.

* This is the **biggest** decode win available and the **riskiest**: it needs an
  accept/reject protocol across the hop (the draft is speculative; a rejected draft wastes
  Stage 0's work but costs nothing Stage 1 did not already pay). DFlash already exists and
  already drafts on the second stage (`lm_head` + DFlash live there, `engine.rs:380-419`), so
  the drafter is in the right place to feed Stage 0 ahead.
* Files: `session.rs` decode path (`5925-5934`, the hop call and the pass loop), `step_batch_opt`
  (`SHADOW_PAIR_PIPE`), the DFlash verify plumbing; `engine.rs` for where the drafter sits.
* Off-switch: `SHADOW_PAIR_SPEC=0` (off, today's serial decode) / `1`. Start with a *single*
  speculative token ahead (depth 1); depth > 1 is a later experiment, not v1.
* Gate: `bench/v0137-mixD18-pair-vs-single-opt` decode, A-B-A, **and** the held-out DFlash
  conversations (STATUS 167: the drafter gains +8.9% on its own training kind, nothing on the
  benchmark — so the accept rate on benchmark text is the thing to measure first, before
  building the pipeline). *Estimate*: recovering the ~5% serial deficit, plus whatever the
  accept rate adds; if accept is low on benchmark text the gain is small and the complexity is
  not worth it — **measure the accept rate before building this.**
* Risk: high. Correctness (a wrong accept changes the output) and wasted Stage 0 work on
  rejects. This is the last item to start, after 1-2 are banked.

### Item 4 — Trim the per-hop fixed overhead (small, both paths)
Each hop pays two `ctx.bind()` (`session.rs:3465,3469`) and an event synchronize. On the 40 KiB
decode hop those fixed costs are most of the 54 µs (the bytes themselves are ~14 µs of link).
Options to measure, not to assume: keep one context bound and use the peer's device pointer
through the unified addressing path if the driver allows it; drop the `hop_time` instrumentation
from the hot path (compile it out unless `SHADOW_HOP_TIME=1`). *Estimate*: single-digit µs per
hop, so ~0.1% on decode — worth folding into item 1, not its own project.

### Item 5 — Resizable BAR and the desktop offload (startup and headroom, not steady state)
ReBAR (item 0's BIOS change) mostly helps the **startup** load of the 16 GiB resident set
(~6 s today) and item 0's P2P probe, not per-token speed. Separately, running the monitors from
the Intel UHD 750 (the motherboard output) takes the desktop's ~200 MB of VRAM off both cards,
which is headroom for the expert cache, not a hop change. Both are cheap; do them as a pair of
small, separately-committed steps with their own STATUS notes.

## 6. Order and bookkeeping

Run item 0 first (it can change items 1-5). Then 1, then 2 and 4 folded in, then 5 whenever,
and start 3 only after 1-2 are banked **and** the DFlash accept rate on benchmark text is known.
Each step: an off-switch defaulting off until its gate passes, an A-B-A (or A-B-B-A for the
prompt bench) on the named `bench/` directory, a STATUS entry with the real number replacing the
*estimate*, and a commit of **only the files that step touched**. The live instance holds the
GPU and the 36 GiB tier, so every gate here is a handoff spec for the **next** instance (or an
external harness), the same as STATUS 169's checks were.

## 7. What this plan deliberately does not touch

The layer split at 15 (`auto_split_first`) and the RAM-tier-on-the-fast-card layout are correct
and measured; this plan does not move them. The 5060 Ti's Gen3 x4 slot is a board limit (the
ASUS TUF H470-PRO's second slot is x4 electrically), so no software change makes the link
wider — only narrower hops and fewer of them.
