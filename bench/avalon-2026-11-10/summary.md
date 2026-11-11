# Avalon Port (`EXPERIMENT_SPEC_2026-11-10`) — Zero-Cost Oracle Bound & Sync Cost Benchmark

## Overview
- **Spec**: `docs/EXPERIMENT_SPEC_2026-11-10.json` (Task 2 of `docs/EXPERIMENT_SPEC_2026-11-04b.json`, executing `docs/EXPERIMENT_SPEC_2026-11-01.json`)
- **Hardware**: RTX 5070 12.2 GB VRAM, i7-11700K (8C/16T, AVX-512), DDR4-2933 (~30 GB/s total), 64 GB RAM (36 GiB pinned tier, 3 GiB floor)
- **Method**: Gated zero-cost oracle hook (`SHADOW_AVALON_ORACLE=N` / `--avalon-oracle N`) in `crates/shadow-cuda/src/session.rs` (`stage_pipe_read`, `stage_copy`, `moe_staged`) that leaves `N` non-resident experts per layer out of both the file read-ahead and the staged H2D DMA copy (slots zeroed once at pass start), plus `SHADOW_AVALON_ORACLE_SYNC=1` (`--avalon-oracle-sync`) which records and synchronizes an event at `moe_staged` entry to measure the mandatory host wait for layer `l`'s router tops before submitting a CPU job.

## Suite 1: Zero-Cost Clairvoyant Oracle Curve (`bench/avalon_oracle_prompt`)
5 arms (`N = 0, 3, 12, 25, 50`) x 3 runs interleaved, fresh server launch per run, `bench/dfdata/prompt_heavy.json` (`--ctx 16384 --ram-tier-gib 36 --ram-floor-gib 3`).

### Clean Single-Pass Warm Turns (Turns 1–3: 5,013 + 3,756 + 4,298 = 13,067 tokens/run, before 16k context compaction)
| Arm | `N` Skipped / Layer | Theoretical Copy Reduction (`N / 474`) | Turns 1–3 Prompt tok/s (per run) | vs Warm Baseline (`1,037.2 tok/s`) | Turns 1–2 Prompt tok/s | vs `oracle0` Turns 1–2 |
|---|---:|---:|---|---:|---:|---:|
| `oracle0` | 0 | 0.00% | 1,018.30 (991.8*, 1,029.0, 1,035.2) | — | 1,109.57 | reference |
| `oracle3` | 3 | 0.63% | 1,038.36 (1,038.6, 1,036.4, 1,040.1) | **+0.11%** (tie) | 1,116.79 | **+0.65%** (tie) |
| `oracle12` | 12 | 2.53% | 1,050.21 (1,045.5, 1,050.3, 1,054.9) | **+1.25%** | 1,130.37 | **+1.87%** |
| `oracle25` | 25 | 5.27% | 1,068.47 (1,064.3, 1,065.6, 1,075.6) | **+3.01%** | 1,152.14 | **+3.84%** |
| `oracle50` | 50 | 10.55% | 1,105.32 (1,102.0, 1,107.2, 1,106.7) | **+6.57%** | 1,193.81 | **+7.59%** |

*\*Note: `oracle0` run 0 turn 3 was the very first pass over turn 3's tokens after server shutdown and paid a one-time cold OS page-cache fill (5.23 s vs 4.79 s / 4.77 s in runs 1–2).*

### Full 5-Warm-Turn Aggregate (Turns 1–5, ~44,151 tokens/run; includes Turns 4–5 after `fit_context` compaction)
| Arm | Prompt tok/s (per run) | Decode tok/s (garbage output on `N>0`) | Combined Total tok/s |
|---|---|---|---|
| `oracle0` | 1,140.41 (1,167.4, 1,132.1, 1,122.7) | 47.22 (46.9, 47.9, 46.9) | 1,094.58 |
| `oracle3` | 1,195.60 (1,178.6, 1,191.5, 1,217.4) | 40.71 (41.4, 41.5, 39.4) | 1,139.33 |
| `oracle12` | 1,178.82 (1,158.5, 1,162.4, 1,217.4) | 46.36 (47.0, 45.4, 46.7) | 1,133.73 |
| `oracle25` | 1,199.32 (1,155.9, 1,186.1, 1,260.7) | 36.99 (36.1, 37.0, 37.9) | 1,155.77 |
| `oracle50` | 1,249.56 (1,240.1, 1,241.3, 1,267.7) | 37.74 (37.7, 37.7, 37.9) | 1,203.22 |

*(At Turns 4 and 5, cumulative history exceeds `--ctx 16384`, triggering rolling compaction and a 15,678-token two-pass re-read where zeroing `N > 0` experts perturbs hidden states across 48 layers and alters downstream routing density.)*

## Suite 2: A/A Control & Per-Layer Router Sync Cost (`bench/avalon_oracle_sync`)
3 arms (`baseA`, `baseB` identical baseline, `oracle3_sync`) x 2 runs interleaved, Turns 0–3 (26,134 warm prompt tokens across Turns 1–3).

| Arm | Prompt tok/s (per run) | vs `baseA` | Decode tok/s | Combined Total tok/s | vs `baseA` Combined |
|---|---|---:|---:|---:|---:|
| `baseA` (baseline) | **1,037.72** (1,040.2, 1,035.2) | reference | 43.55 | **957.70** (959.6, 955.8) | reference |
| `baseB` (A/A control) | **1,036.69** (1,037.8, 1,035.6) | **-0.10%** (noise floor) | 43.77 | **957.22** (957.6, 956.8) | **-0.05%** |
| `oracle3_sync` (`N=3` + router sync) | **982.61** (972.9, 992.5) | **-5.31% (LOSS)** | 36.82 | **898.17** (889.0, 907.6) | **-6.22% (LOSS)** |

## Conclusion
1. **Even a clairvoyant 0-cost oracle at `N = 3` experts/layer is a tie (+0.11% to +0.65%)**: Removing 3 of ~474 non-resident experts per layer saves 0.63% of staged DMA bytes, which cannot clear the 0.8% noise floor.
2. **Waiting for router tops on the host turns `N = 3` into a -5.31% prompt loss (-6.22% combined)**: Today's 1,037–1,140 tok/s prompt pipeline (`dev_plan_buckets` + `early` stage copy) runs completely on the device so the host stays a full layer ahead of the GPU, queuing `stage_copy(l + 1)` and `stage_pipe_read(l + 2)` before layer `l + 1`'s router even executes. Synchronizing on layer `l`'s router to learn which cache-miss experts were chosen forfeits that host-ahead overlap (~4.5 ms/layer = ~216 ms/pass), losing -5.31% even when the CPU work and DRAM reads themselves cost zero.
3. **Verdict**: Per the agreed protocol, the full CPU-expert prompt port is not built; the timing-only oracle hooks (`SHADOW_AVALON_ORACLE`, `SHADOW_AVALON_ORACLE_SYNC`) stay gated and off by default.
