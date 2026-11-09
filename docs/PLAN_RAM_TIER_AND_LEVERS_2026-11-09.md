# Plan: a bigger RAM tier, and the compute levers Strata already has (2026-11-09)

Asked for by the user: "raise the RAM tier and improve the remaining levers 10x; if it can be done on Strata we can do it too." This plan keeps the second half (what Strata does, Shadow can do - Strata is MIT and borrowing is allowed) and is honest about the first: **10x of the whole prompt is not available**, and the sections below say which parts can move by how much, from measurements already in this repository.

## 1. What 10x can and cannot mean

The ceilings are physical and measured here:

| ceiling | value | source |
|---|---|---|
| the model's expert set crossing PCIe once per pass | ~1.8 s (46.8 GiB at ~26 GB/s; Strata's own `1.8 s x chunks`) | PLAN_BEAT_STRATA 2026-11-07, STATUS 44-50 |
| compute (Strata, int8 tensor cores) | 0.18 ms a token = 5,500 tok/s | same |
| the NVMe (Sandisk Optimus 5100, J:) | **3.5 GB/s** at any read size or depth; direct I/O was a loss (-20..-42%, STATUS 2026-11-04) | STATUS 2026-11-01, 2026-11-04 |
| DRAM | 4 x 16 GiB at 2,933 MT/s, the stable maximum | memory note |

So a 15k prompt cannot go below ~1.8 s (one pass of experts over PCIe) + 15k x 0.18 ms (2.7 s) = **~4.5 s**; Shadow reads it in 10.8 s now, Strata in 9.9 s (6.3 s on 2026-11-05). The most any plan can promise is **~2.4x at 15k, ~1.9x at 3k** - and only the SSD term has an order-of-magnitude in it: **at tier 45 the file-only part of a pass shrinks from ~11 GB (3.3 s at 3.5 GB/s) to ~1.9 GB (0.55 s), about 6x**, and the share of expert-cache misses that go to the SSD in live decode from 16% to ~4%.

## 2. The facts the plan rests on

- RAM 63.8 GiB. The live server (tier 36 GiB): **working set 40.7 GiB, so 4.7 GiB beyond the tier**; 8.1 GiB available with the user's other programs running (they use 11-15 GiB). The floor is 3 GiB by default in the launcher, 4 GiB by the rule in memory. **Where the floor comes from, exactly:** the user once saw the PC freeze at about 0.4 GiB free, and the cause was never recorded (it may have been an outside program, Windows itself, or an engine); the only low-RAM event documented in this repository is 2026-10-26, when an uncapped llama.cpp mmap load of the older, larger model took available RAM from 55 to 0.73 GiB in 49 s and the watchdog killed it before any freeze. So the repository does not show that 1-3 GiB free is unsafe on this PC, only that 0.4 GiB once was and that a pinned tier cannot be paged out. Strata at tier 45 dipped to 3.61 GiB available when measured (3 GiB floor, user-approved test), with no freeze.
- The tier is **one `PinnedRegion`** (one `VirtualAlloc` + `cuMemHostRegister`, `RamTier::fill`, `session.rs`): it cannot give memory back while the server runs.
- Live chat (uptime 992 s): VRAM expert-cache hit rate 79.8%; of the misses 1.65 M came from the RAM tier and **0.32 M (16.3%) from the SSD**.
- A 3k prompt: 4.5 s, of which ~2.4 s is the SSD term (tier 36); Strata 2.4 s reads nothing from disk.
- Strata's design (docs/PHASE1_STRATA_SOURCE_NOTES_2026-11-08.md): tier 45 of 46.8 GiB pinned; 8,192-token chunks streamed through a ring in a fixed order; **its own fused int8 tensor-core MoE kernels** (int8 MMA measured ~3.1x fp16 on sm_120; weights decoded in shared memory to int8 per 64 values behind the MMA; 64-128 routed rows per weight tile); a **CPU share** (idle CPU threads compute the cold experts a short chunk routes few tokens to, instead of streaming them: -20..-35% at 512-1,000 tokens).
- Shadow's MoE products are 39% of kernel time; the profile (STATUS 68) shows them latency-bound (28% issue slots, 41% occupancy), not tensor- or DRAM-bound.

## 3. The levers

| # | lever | what Strata does | expected gain (range, not a promise) | effort | risk |
|---|---|---|---|---|---|
| L1 | **RAM tier 36 -> 43-45 GiB** | 45 | 3k: 4.5 -> ~3.0 s; 9.6k: -0.3..-0.8 s; live decode: SSD misses 16% -> ~4%; first pass after a restart | medium | the RAM floor (section 2: the one reported freeze has no recorded cause) |
| L2 | the 4.7 GiB of RAM beyond the tier -> <= 2 GiB | (its overhead is ~1.5 GiB) | pays for ~2.7 GiB of tier | small-medium | none |
| L3 | **int8 fused MoE** on the tensor cores (IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL; Q2_0 stays ours) | `moe_fused_iq.cu` | MoE products -40..-55% of their time = 15k: -1.4..-1.9 s (of 10.8) | large | numerics (int8 is an inexact fast mode: fidelity gates) |
| L4 | CPU share of cold experts for short chunks | on since Strata 0.1.41 | agent steps (100-1,000 tokens): -20..-35% **only while experts are cold**; mostly moot at tier 45 | large | CPU contention |
| L5 | attention (14%), hyper-connection norms (11%), fixed costs (~0.15 s) | int8 KV (fp16 MMA) | 15k: -0.5..-1 s together | medium | low |
| L6 | decode: drafter (DSpine-style +10% acceptance), speculation depth | n/a | +3..+8% | large | training needs |

## 4. Phases and gates (each ends with a STATUS entry; tie or better goes default; every model load under `tools/run_capped.ps1`, floor never below the user-approved value)

**Phase A - make a bigger tier safe (L2 + the safety half of L1).**
- A0. **A RAM black box**: while serving, log system available RAM every second and, whenever it falls under 3 GiB, the five biggest processes by working set (Shadow's own working set and pinned tier included) to a rolling file next to the tool-output log. Cheap, and it settles the question above: the next low-RAM event is recorded with its cause (an outside program, Windows, or Shadow), so the floor can be chosen from evidence.
- A1. Audit the 4.7 GiB: for each component switch it off or measure it (stage pipeline regions 1.33 GiB, conversation snapshot host buffers, drafter weights if a host copy stays, the embedding table, mapped file pages, KV-lend host side) and write the table. Gate: a table that adds up to within 0.3 GiB.
- A2. Cut it to <= 2 GiB without a speed loss (A-B-B-A at 9.6k / 15k). Gate: working set minus tier <= 2.0 GiB.
- A3. **Build the tier from 1 GiB chunks, each registered on its own**, so a watchdog can release the tail chunk(s) when available RAM falls under floor + 0.5 GiB for 2 s (unregister, free, mark those experts file-only, refill later when RAM returns). Gate: a test that allocates a 6 GiB hog while serving: the tier shrinks, nothing freezes, available never drops below the floor, the answers stay correct.
- A4. Sizing at start: tier = min(requested, available - floor - overhead - 1 GiB), replacing the fixed 36 / 30 rule.

**Phase B - raise the tier** (needs A). Ladder 36, 40, 42, 44, 45 on fresh servers (A-B-B-A, same prompts, min available recorded). Gates: 3k <= 3.3 s at >= 43; no decode loss; min available >= the user's floor throughout. Expected stop point on this PC with the user's usual programs open: **~42-43 GiB**; 45 needs the other programs under ~11 GiB.

**Phase C - int8 fused MoE (L3).** C1 standalone harness (`bench/mmabench`) on the real tensors: Strata-style fused kernel versus llama.cpp's `mul_mat_q` with expert ids (MIT, already vendored behind `shadow_mmq.dll`; STATUS 43 measured 1.7-2.4x for gate/up in isolation). C2 the per-expert weight-pointer table and the planner's sorted ids (named in STATUS 43). C3 all four formats. C4 engine behind `SHADOW_MMQ_MOE`, default only when it is a tie or better on kernel time and passes `quality_test`, the needles and the hash gates. Gate: MoE kernel time -40% at 15k.

**Phase D - attention / norms / fixed costs (L5)**, taken in the order of the profile after C. **Phase E - the CPU share (L4)** only if B leaves cold-expert stalls in agent steps. **Phase F - decode (L6)** last.

## 5. Where this should land

| prompt | Strata (0.2.11) | Shadow now | after A+B | after A+B+C | after A..D |
|---|---|---|---|---|---|
| 3k | 2.4 s | ~4.5 s | ~3.0 s | ~2.7 s | ~2.5 s |
| 9.6k | 5.5 s | 7.0 s | ~6.4 s | ~5.4 s | ~5.0 s |
| 15k | 9.9 s (6.3 s on 11-05) | 10.8 s | ~10.3 s | ~8.7 s | ~7.8 s |

Parity with Strata's 11-08 numbers is reachable at every length; matching its 11-05 figure at 15k (6.3 s) is not shown to be reachable with these levers. These are estimates from the measured shares above; each phase's gate replaces its row with a measurement.

## 6. Decisions needed from the user

1. **Floor policy at a high tier**: keep 4 GiB (tier ~42-43 with the usual programs open), or approve 3.5 GiB for tier 45 once the watchdog (A3) exists; or decide after the black box (A0) has run for a while and shown what actually consumes RAM on this PC. The watchdog reacts to a drop whatever its cause, so the plan does not depend on knowing it.
2. Whether to close memory-heavy programs while Shadow serves (needed only for 45).
3. Go-ahead for Phase C (the largest piece of engineering; several sessions).
4. Accept the target as "parity with Strata, then ahead where its design allows", not 10x.
