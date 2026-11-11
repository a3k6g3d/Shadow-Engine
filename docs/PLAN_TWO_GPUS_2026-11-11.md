# Plan: Shadow on two graphics cards (RTX 5070 12 GB + RTX 5060 Ti 16 GB)

Written 2026-11-11, before the second card is installed. Nothing here is built yet; every number marked *estimate* is
arithmetic from earlier measurements, to be replaced by a measurement at the gate named beside it.

## 1. What the hardware allows (read first, it decides the design)

| | RTX 5070 | RTX 5060 Ti 16 GB |
|---|---|---|
| chip / compute capability | GB205, sm_120 (12.0, measured) | GB206, sm_120 (same kernels, same NVRTC cache) |
| VRAM / bandwidth | 12 GiB / 672 GB/s | 16 GiB / 448 GB/s ([spec](https://www.guru3d.com/review/nvidia-geforce-rtx-5060-ti-16gb-review/)) |
| power | 250 W | ~180 W |
| slot on this board (ASUS TUF GAMING H470-PRO) | PCIEX16_1: CPU, **PCIe 4.0 x16** (measured: gen 4, width 16) | PCIEX16_2: **chipset, PCIe 3.0 x4** ([ASUS spec](https://asus.com/motherboards-components/motherboards/tuf-gaming/tuf-gaming-h470-pro/techspec): "x16 slot, x4 mode") |
| host link, peak | ~25 GB/s in practice (what the engine saturates today) | **~3.5 GB/s**, and shared with the chipset's DMI link (NVMe, SATA, USB) |

Consequences:

* **The second card cannot stream experts.** Today's decode is bound by expert bytes crossing PCIe (~296 MB a token at
  ~15-20 GB/s, README resource record). A card on a ~3.5 GB/s link adds only ~15% to that bandwidth. Its value is its
  **16 GiB of VRAM**, used as permanent storage, loaded once at start (16 GiB at ~3 GB/s is ~6 s), never refilled.
* **The 5070 stays in the CPU slot** (x16 gen 4) and keeps doing the streaming. Do not swap them.
* **No peer-to-peer.** GeForce cards on Windows normally report no peer access, and the second card sits behind the
  chipset anyway. Anything that crosses cards goes through pinned host memory (device -> host -> device). Plan for that;
  use P2P only if `cuDeviceCanAccessPeer` says yes (probed at the first gate).
* The Intel UHD 750 is present. Running the monitors from the motherboard output would take the desktop's VRAM
  (~200 MB or more) off both cards. Optional, free.
* Check before install: PSU wattage and the 5060 Ti's power connector (5070 250 W + 5060 Ti ~180 W + an 11700K that can
  pull 125-250 W: ~750 W or more is the safe size), and the clearance for a second card under the 5070.
  *Not known; please tell me the PSU.*

## 2. Why a layer split

What the card is for: the engine is PCIe-bound because VRAM is full (12 GiB: dense weights 4.4 GiB, buffers and drafter
~1.6, expert cache ~3.5-4.6 of the model's 44.75 GiB of experts: 1,843-2,317 slots, hit rate ~48-50%; STATUS.md, "2026-10-31:
where the VRAM goes"). One more GiB of cache buys only ~4.5 points of hit rate (the same section: 0.6 GiB bought 2.7
points), so "more cache" is inefficient; **whole layers whose
experts are all resident cost no copies at all** (100% hits).

Design (**layer split, "pipeline" in the sense of GPUs taking turns, not overlapping**): the 48 layers are cut at layer
*k*. GPU 0 (5070) runs layers 0..k-1 as today (RAM tier, streaming, VRAM cache). GPU 1 (5060 Ti) runs layers k..47 and the
output head with **every expert of those layers resident in its VRAM**. Per token one hop: the residual streams (4 streams x
2,560 floats = 40 KB a token, 160 KB for a 4-token verify pass) go GPU 0 -> pinned host -> GPU 1; the next token's id comes
back, a few bytes. Hop cost *estimate*: 30-100 us a pass against ~40 ms: under 0.3%. A 2,048-row prompt sub-chunk is 80 MB
through a 3 GB/s link: ~25 ms a hop against ~1.4 s a chunk (the link is only used once a chunk).

*Estimate of the gain, for the first gate:* GPU 1's ~12-14 layers (16 GiB less ~1.3 GiB of their dense weights, the head
0.5, the DFlash drafter 1.2 if it moves there, buffers; ~0.93 GiB of experts a layer) never miss: that is 25-29% of the
layers' expert traffic gone, and GPU 0's cache grows by the dense weights it no longer holds (~+1.3 GiB, +5 points of hit
rate on its layers). About -33% bytes over PCIe; if decode is ~70% PCIe-bound that is ~-23% pass time, less the slower card's
compute on its layers (448 against 672 GB/s: +3 ms): **~+15-25% decode (55 -> ~65 tok/s)**. Also: more resident experts
means fewer "skipped unless in VRAM" tail experts (`--tail-skip`), so fidelity should rise a little. Prompt reading gains
the same way on GPU 1's layers (no stage copies).

Rejected, and why:
* **GPU 1 as a remote expert cache for GPU 0** (copy GPU 1 -> GPU 0 on a miss): the bytes still enter GPU 0's PCIe link,
  which is the bottleneck. No gain.
* **Expert parallel with the hot set on GPU 1** (every layer split between the cards, outputs merged each layer): the best
  ceiling (~65% fewer PCIe bytes, maybe 1.6x) but 96 host-staged hops a token (~5 ms, 12% of a pass) and a planner that
  splits experts across devices. Kept as the **stretch phase 5**, gated on the measured hop latency.
* **Drafter and head only on GPU 1**: ~+3% (frees 1.2 GiB of cache). A stepping stone, not the goal.

## 3. How it fits the engine (what the survey found)

* `GpuSession<'c, 'm>` (`crates/shadow-cuda/src/session.rs`, 6,290 lines) owns one `&Context`, ~127 `DeviceBuffer<'c>`
  uses, per-layer weights, KV, DeltaNet state, the expert cache, the RAM tier, CUDA graphs, the stage pipeline, the drafter.
  It is single-device throughout, and a single refactor to many devices inside it would touch everything.
* **What makes the split tractable:** nothing global is bound to a device (the driver API and NVRTC are process-wide
  statics; each `Context` loads its own module, `kernels.rs:345`). So **two ordinary `GpuSession`s, one per device in one
  process, each told to own only a range of layers, need no multi-device refactor.** The new work is (a) a layer range
  in a session, (b) a hop between two sessions, (c) the engine treating the pair as one session.
* (a) **Layer range**: the weight upload, the per-layer state allocation, the expert cache sizing and the layer loops
  (about ten places: `session.rs` 1016, 1459, 1505, 1571-1675, 2499, 4266, 4448) take `[lo, hi)`. The first stage owns the
  token embedding / n-gram rows (host side already), the last owns `output.weight`, `output_norm` and the logits.
* (b) **Hop**: `Session::run_layers(lo..hi)` between two residual-stream hand-offs; the hop is a device -> pinned -> device
  copy ordered by events (`Stream` / `Event` wrappers exist). The RAM tier belongs to GPU 0's layers only (34 layers x 0.93
  GiB = ~31 GiB: **it then holds 100% of GPU 0's experts; NVMe is never read**). Pinned host memory used by two contexts
  must be registered with the portable flag, or kept separate (the plan keeps it separate: one tier, one staging area
  each).
* (c) **Engine**: `engine.rs` calls 28 distinct session methods (`step`, `step_batch`, `prefill_batch`, `snapshot`,
  `restore_conv`, `rollback`, `dflash_draft`, `lend_cache`, `restore_cache`, `set_pictures`, ...). The session on GPU 0 becomes the *main* session and holds an optional `peer`
  (the GPU 1 stage); the stateful methods (reset, snapshot, restore, rollback, set_pictures, hint_prompt, lend / restore of
  the cache, tokens) forward to the peer, and the two layer loops call it at layer *k*. That is about 15 forwarding
  points among the 28, not a second API; the rest (reports, stats, capacity) read the main session only.
* **DFlash**: the drafter reads feature taps at layers 5, 15, 25, 35, 45 and uses the model's head for its candidates, so
  it lives with the head on GPU 1; the taps from layers on GPU 0 hop with the residuals (2 x 10 KB a token).
* **Not in scope:** `--mtp` (unused), pictures on two cards (stay single-card until the rest works), the CPU hybrid experts.

## 4. Phases, gates, what comes out

Effort is honest, not hopeful: this is a 10k-line CUDA/Rust engine and a new execution mode. **Nothing in phases 1-5 will
run on both cards by Thursday.** Thursday is the probes (phase 0b); the first two-card token is the end of phase 2.

**Phase 0a (before the card arrives, no hardware): the probe tool and the baseline.**
`shadow gpus`: list devices (name, UUID, compute capability, VRAM, driver, PCIe gen/width now and max), and per device:
pinned host -> device and device -> host bandwidth alone, both cards at the same time (does the chipset path hurt the CPU
slot?), a small-copy round trip (the hop: 40 KB and 160 KB device -> pinned -> device), peer access. Runs on one card
today (a one-device report), so it is tested now. Pin the device order by UUID or `CUDA_DEVICE_ORDER=PCI_BUS_ID`, never by
ordinal. Baseline: re-measure the single-card bench (decode 55.7 tok/s default, STATUS 97/99) so the gain is a same-day
pair.

**Phase 0b (Thursday, card in): bring-up and the first gate.**
Install in PCIEX16_2; BIOS: Above 4G Decoding on, check the CPU slot still negotiates gen 4 x16 with both cards in.
`shadow gpus` on both. **Gate G1** (decides the design): hop round trip under ~150 us; GPU 0's CPU-slot bandwidth
unchanged with GPU 1 loading at the same time; GPU 1's host bandwidth ~3 GB/s or better; no peer access is acceptable.
If the hop is slow (over ~300 us) the layer split still works (one hop a token) but the stretch phase 5 is dead. If the
CPU slot drops to x8 or the 5070 loses bandwidth, stop and re-plan (the board's slot sharing is then different from the spec).

**Phase 1: layer ranges, same GPU, exact mode (testable before Thursday, and the correctness proof).**
Two `GpuSession`s on the one 5070, small caches and `--ctx 2048`, layers 0..24 and 24..48 with a hop through host memory
(the real hop code, a fake second card). **Gate G2: `--exact` hash identical to the single session (162a8e07), `quality_test`
12/12, and the layer ranges' memory is the sum of the whole.** The exact path has no drafter, no tail skip and no
speculation, so any difference is the split. Off switch: `SHADOW_GPUS=1` (single session, today's behaviour).

**Phase 2: two cards, exact mode, decode and prompt.** Same test with the 5060 Ti as the peer: all of its layers' experts
uploaded and resident (the cache is sized to hold every expert of the range; startup reads those experts once). **G2
again on the real pair**, then a prompt of 3k / 15k and a 200-token decode. The prompt runs its sub-chunks through GPU 0
then GPU 1 one after the other (no overlap yet).

**Phase 3: fast mode.** The per-session tiers (GPU 0: RAM tier + streaming + cache; GPU 1: resident), CUDA graphs per
session, the device planner and `lend_cache` per session, DFlash on GPU 1 with its feature taps, speculation and rollback
across both, `snapshot` / `restore_conv` forwarding (STATUS 98-99: a restore must leave both drafters' and both layers'
state coherent), `--accept-moe` and `--tail-skip` per stage. **Gate G3: decode >= the single-card figure on the same day
(the estimate says +15-25%); replies' fidelity not worse (`tools/quality_test.py`, `needle_test.py`, `long_agree.py`);
the 36-prompt bench and the Strata comparison re-run, three interleaved runs each.** Choose *k* by measurement: sweep it
(k = 30..38) against decode and prompt; the model's 48 layers, the 16 GiB, the drafter's 1.2 GiB and ~0.93 GiB of experts a
layer give ~12-14 resident layers; add the RAM tier size to the sweep (all of GPU 0's 34 layers fit).

**Phase 4: the prompt path overlapped.** Two threads: GPU 0 starts the next sub-chunk while GPU 1 runs the previous one
(a prompt of 15k is 8 sub-chunks: up to ~2x on the part GPU 1 owns). Per-session stage pipelines already exist.

**Phase 5 (stretch, only if G1 showed a hop under ~50 us): expert-parallel hot set.** GPU 1 keeps the hottest experts of
*all* layers (by `profiles/routing-varied300.shrc`) and computes them while GPU 0 computes the rest of the layer; per layer
a hop each way. Ceiling ~1.6x; cost ~12% of a pass in hops. Decided by the phase 3 numbers, not by hope.

## 5. Risks and how each is caught

| risk | caught by |
|---|---|
| the chipset slot's real width / bandwidth differs from the spec (shared lanes: the manual may disable x1 slots, not CPU lanes) | G1, `shadow gpus` |
| device-staged hop slower than ~150 us (WDDM scheduling, two drivers' queues) | G1; fallback: one hop a token is still fine |
| exact-mode difference introduced by the split | G2: hash against 162a8e07, in phase 1 on one card first |
| fast mode is slower: the 5060 Ti's layers compute 1.4x slower (448 vs 672 GB/s) and sit on the critical path | G3 and the *k* sweep (fewer layers on the slow card) |
| the drafter / feature taps across cards, rollback and snapshot state | phase 3 tests: replies identical to single card in exact mode, restore test (`tools/df_restore_test.py`) |
| both cards at full power on the PSU; two cards' heat | the PSU question; `nvidia-smi` power and clocks in the bench's per-second records |
| the RAM floor (pinned tier 39-45 GiB of 64) with a second CUDA context's own host buffers | the black box `rambox.log` and the tier sizing `tier_for_available`: GPU 1's contexts need ~1-2 GiB of host memory |
| startup time: 16 GiB through a 3 GB/s link and the 5070's tier at once | G3 reports `ready in` (now 38 s); the loads can overlap |

## 6. What I need from you

1. The power supply (model / wattage, free 8-pin or 12V-2x6 connectors) and whether the 5060 Ti fits beside the 5070.
2. A yes to start **phase 0a and phase 1 now** (they need no card): the `shadow gpus` probe and layer ranges on one card.
3. The card's exact model when you have it (it decides the connector and the power limit).
