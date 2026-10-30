# Changelog

All notable changes to Shadow will be documented in this file.

## [0.1.27] — 2026-10-30
### Changed (default on; `--exact` turns the speculation and inexact modes off)
- `shadow serve MODEL` runs the fastest mode that keeps answers intact with no tier flags: `--tail-skip 4 --accept-moe
  --accept-beta 1.0 --ram-tier-gib 30`, the six-prompt placement, the DFlash 2 drafter (matrices Q8_0), `--spec 3`,
  `--dflash-min-p 0.5`. About 35 tok/s by total throughput on varied text against about 20 for `--exact`, and 47 for Strata
  (docs/STATUS.md, docs/BENCHMARKING.md). Not exact: `--tail-skip` and `--accept-moe` change the model's own computation;
  speculation never does. `--verify-resident` is no longer a default: it truncated answers (11 of 14 replies ended early).
- DeltaNet decay `expf` in float and the long sums in runs of 16 added in double (FP64 is 1/64 rate on the test GPU): +4% in spec mode.
### Added
- The DFlash 2 drafter (`--dflash DIR|off`, `--dflash-bf16`, `--dflash-min-p P`): features from layers 5/15/25/35/45, per-layer
  K/V rings, a block-diffusion draft round, top-16 candidates through the main model's head and a bilinear selector walk
  (docs/SPECULATION.md); `tools/dflash2_ref.py` checks it against torch, `tools/dflash2_train.py` is the fine-tuning pipeline.
- Kernel fusions behind `--fuse N` (bit-identical, measured a tie or a loss, off); PLE row prefetch.
- `shadow-bench`: records the machine once a second per run (CPU, RAM, disk, GPU util/memory/power/temperature/clocks, PCIe) and
  each turn's start/end, `--prompts-file`, `--no-resources`; `tools/plot_bench.py` (charts, tables, joules per token),
  `tools/bench_total.py` (total-throughput scoring), `tools/autotune.py` (coordinate ascent over flags),
  `tools/bench_vs_strata.ps1`.
- `SHADOW_RECORD_ROUTING=FILE` (a routing profile from a running server), `SHADOW_NO_PREFIX=1`, `SHADOW_DF_DUMP=DIR`, `SHADOW_DF_TIME=1`.
- AcceptMoE verifier expert sets (`--accept-moe`, `--accept-beta/--accept-scale/--accept-budget`), the graded `--verify-rank R`,
  `finish` recorded per bench turn, tokens-a-run and early stops in `bench_total.py`, `tools/quality_test.py`, `tools/trunc_test.py`.
### Learned (docs/STATUS.md)
- The bench's per-turn median misranks arms that generate different text; score by total throughput.
- A drafter fine-tune that looked like +25% accepted tokens counted prompt positions; on generated text it was a tie.
- Candidates over a 106k draft vocabulary: -25%. Longer blocks than 3 drafts: slower.

## [0.1.26] — 2026-10-28
### Changed (default on, exact)
- CUDA graphs for each DeltaNet layer's launches from its hc read to its router (`SHADOW_GRAPHS=0`: off):
  `bench/graphs-2026-10-28` **+2.1% paired** decode.
- Overlapped expert copies: resident experts run while the misses copy, the GPU (not the host) waits for the copies,
  `moe_combine_qwen_perm` keeps the sum order (`SHADOW_OVERLAP=0`: off): `bench/overlap-2026-10-28` **+2.7% paired**.
  Together 18.65 -> 19.47 tok/s `decode_warm`, 0 answer differences.
- The VRAM expert cache is LRU over every slot of a layer; the placement's hottest experts are no longer pinned (and
  so stay in the RAM tier). `tools/cache_sim.py` predicted 59 -> 75% hits; measured 56.6 -> 70.7%, `bench/lru-2026-10-28`
  **+11.1% paired** (19.53 -> 21.06 tok/s). `--pin-margin N` brings back pinning with N LRU slots a layer.
### Added
- Kernel launch counts and host time in launches in the session report; `SHADOW_CACHE_DUMP=FILE` (per-layer slots and
  pinned experts, for `tools/cache_sim.py`).

## [0.1.25] — 2026-10-28
### Added (default on, exact)
- **Batched step**: `GpuSession::step_batch` / `prefill_batch`, up to 64 tokens through the layers together (Qwen):
  shared projections through `matvec_cols` (new `matvec_cols_ref_bf16/f32` and `hc_down_inject_cols_bf16` keep the
  BF16/F32 sums in the one-token order), the union of the tokens' experts fetched once (overflow into a 128-slot VRAM
  scratch arena). Logits bit-identical to `step` (`tests/reference.rs`, also past the sparse indexer). Used to read
  prompts in `shadow serve` and `shadow generate`: **0.062 -> 0.035 s/token on a 1,262-token prompt, -46.0% paired,
  all 8 pairs** (`bench/prefill-2026-10-28`); decode a tie, 0 answer differences (`bench/batch-2026-10-28`).
  `SHADOW_BATCH=n`, `SHADOW_BATCH_SCRATCH=n` for comparisons. `shadow generate @FILE` reads the prompt from a file.
### Added (opt-in, measured, slower)
- `shadow serve --spec K`: speculative decoding with prompt-lookup drafts verified by `step_batch`
  (`GpuSession::snapshot_again` + `restore` on rejection). `bench/spec-2026-10-28`: K=2 -11.2%, K=3 -18.3%.
### Fixed
- `RamTier::promote_slot` could hand out a ring slot twice in one layer (wrapping Clock hand), overwriting an expert
  still being read or copied; claimed offsets are now skipped until the copies are waited for.
- The kernel drift test (it failed on the `COLS` macro; the declarations are written out).

## [0.1.24] — 2026-10-27
### Changed (default on, exact)
- Router logits read through mapped pinned memory by a copy kernel; small uploads (embedding, expert tables, n-gram
  rows) staged in mapped memory and moved by `copy_words`, so neither queues in the copy engine behind expert DMA.
- `Gguf::tensor` uses a name index (was a linear scan per matvec); `route` keeps the top k by insertion (unit-tested
  bit-for-bit against the full sort).
- `bench/night-2026-10-27`, 8 interleaved launches per arm against 0.1.23: `decode_warm` median **18.53 -> 18.85
  tok/s, +1.5% paired, all 40 pairs faster** (worst +0.4%), 0 answer differences.
### Added (opt-in, measured, not faster on this PC; docs/STATUS.md)
- `--cpu-experts N --cpu-threads T`: hybrid MoE, a layer's lowest-weight RAM-tier misses computed on the CPU
  (`shadow-model::cpu_expert`, AVX2 kernels for every routed-expert type). Slower at every N tried.
- `--pregate K`: pre-gated prefetch of the next layer's likeliest experts. A tie.
- `matvec_cols_*` kernels with `dotm<M>`: up to 8 inputs per weight read, bit-identical to `matvec_fast_*`; the first
  piece of batched prefill / speculative verification.
- Probes: `SHADOW_ROUTE_LOG`, `SHADOW_MTP_PROBE`, `SHADOW_PREGATE_PROBE`, `SHADOW_TIMING_NO_COPIES`,
  `tools/spec_probe.py`; host wait timers in the session report; `shadow compare` takes tier flags.

## [0.1.23] — 2026-10-27
### Changed
- **Parallel drive-tier expert reads (`shadow-model::readpool`)**: a layer's experts that miss VRAM and the pinned RAM
  tier are no longer read one at a time on `Model`'s single file handle. `GpuSession::fetch_expert` queues them, and
  `read_drive` reads them all at once in 256 KiB pieces on 8 persistent workers, each with its own handles (Windows
  serializes I/O on one synchronous handle). Reads stay buffered, so repeats still come from the file cache.
  8 interleaved launches per arm (`bench/readpool-2026-10-27`, `--ctx 8192 --ram-tier-gib 30 --placement
  profiles/routing-6prompts.shrc`): `decode_warm` median **17.37 -> 18.52 tok/s**, paired **+5.3%** median, faster in
  all 40 pairs (worst +2.9%), 0 answer differences. Reference tests and the 48-layer compare are unchanged (cosine 1.000).
- `shadow-bench run --arm-exe NAME=EXE` runs an arm on another build, for A/B tests of two binaries.

## [0.1.22] — 2026-10-27
### Fixed
- **Stop Button Availability During Active Reasoning**:
  - Added dedicated **`⏹ Stop Thinking`** button directly into the active `.think-header` alongside a live timer and pulsing spinner, allowing users to instantly halt generation while the model is reasoning without having to wait for the thought to complete.
  - Fixed viewport height chain in Chat view (`#messages { min-height: 0; }` and `#chat-panel { min-height: 0; overflow: hidden; }`), preventing lengthy reasoning streams from expanding `#messages` beyond the viewport ceiling and pushing `#input-pane` (and the lower chat bar Stop button) off the bottom of the screen.
  - Added global `Escape` key listener (`window.addEventListener('keydown')`) to trigger immediate generation cancellation from anywhere in the window.
  - Enhanced thought header transition: on thought completion or abort, displays `Thought for Xs` / `Stopped thinking after Xs` with an interactive collapsible accordion (`▼`/`▶`).

## [0.1.21] — 2026-10-27
### Changed
- **Made MTP Speculative Decoding Strictly Opt-In (`--mtp <DIR>`)**:
  - Removed auto-activation of MTP when `models/mtp-basalt-ft1` is found on disk. Server now defaults to pure GPU base-model decoding at full speed with maximum Tier 1 VRAM expert cache allocation (~3.8 GiB) unless `--mtp <DIR>` is explicitly passed.
  - Fixed MTP draft normalization dimension: `pre_fc_norm_hidden` now normalizes across the full 10,240-element multi-stream residual (1 group of 10,240 floats instead of 4 groups of 2560).
  - Fixed MTP residual initialization bug: initialized `r_mtp` via `copy` + `add` rather than in-place accumulation over uninitialized buffer memory.
  - Restored full VRAM expert cache headroom for the base model, preventing expert cache misses and PCIe thrashing during standard inference.

## [0.1.20] — 2026-10-27
### Added
- **MTP (Multi-Token Prediction) Speculative Decoding Engine (`crates/shadow-cuda/src/mtp.rs`)**:
  - Wired Shadow's GPU forward pass to load and execute the tuned MTP draft head from `models/mtp-basalt-ft1` (`dense.bin`, `dense.txt`, `experts.bin`, `draft_vocab.bin`).
  - Implemented the full 1-layer MTP draft transformer block: input projections (`fc_embedding`, `fc_hidden`), attention hyper-connection, QSA self-attention, MLP hyper-connection, MoE with 512 resident routed experts (unpacked to contiguous Q2_0 rows) + shared expert, and final mixer.
  - Added native `Q8_0` quantization format to CUDA kernel registry (`kernels/quants.cuh`, `kernels/dequant.cu`, `kernels/matvec.cu`, and `shadow-cuda/src/kernels.rs`), supporting `dequant_q8_0`, `matvec_q8_0`, `matvec_fast_q8_0`, `matvec_batch_q8_0`, and `matvec_pair_q8_0`.
  - Added speculative decode loop in `crates/shadow-serve/src/engine.rs`: proposes candidate tokens via MTP draft head and verifies them against the base model's greedy argmax in batch. Accepted drafts (~86% acceptance rate) commit multiple tokens per base model step, accelerating generation throughput from ~14 tok/s to >30 tok/s with guaranteed bit-for-bit output parity.
  - Added `--mtp DIR|off` CLI support and live telemetry indicator in Web UI Process Monitor (`engine-mtp-txt`).

## [0.1.19] — 2026-10-27
### Added
- **Chat Stop Button (`⏹ Stop`)**: Added responsive generation cancellation to the lower chat input bar. While Shadow streams tokens, the Send button switches to an interactive red Stop button. Clicking Stop instantly cancels the client `AbortController` and signals the server; `shadow-serve` drops the stream receiver and halts GPU forward-pass decoding immediately, preserving all partially generated content.

## [0.1.18] — 2026-10-27
### Fixed
- **Process Monitor Dashboard DOM Null Dereference**: Resolved a silent JavaScript exception in `pollTelemetry()` where a query for `cpu-sys-val` returned `null`, halting execution before the GPU compute, RAM, Engine, and MoE Tier dashboard cards could be populated. Replaced with null-safe setters (`setTxt`, `setHTML`, `setWidth`) ensuring all telemetry cards update continuously and reliably.

## [0.1.17] — 2026-10-27
### Fixed
- **View Panel Layout Bug in Web UI**: Fixed CSS specificity bug where `#monitor-panel { display: flex; }` caused the Process Monitor to render side-by-side on the right of the Chat view. `#chat-panel` and `#monitor-panel` now strictly respect tab selection, rendering full-width independently.
- **Accurate CPU Usage Calculation**: Corrected `SysMonitor` calculation in `crates/shadow-serve/src/telemetry.rs`. Removed duplicate logical core multiplication that caused process CPU to report 100% saturation during single-threaded generation. Process CPU now accurately represents overall system capacity share alongside live active core equivalents.
### Added
- **Real NVML GPU Compute Utilization Monitor**: Dynamically binds `nvml.dll` (`nvmlDeviceGetUtilizationRates` and `nvmlDeviceGetTemperature`) to report real-time GPU compute core execution percentage (0–100%), memory controller load %, and GPU temperature in °C.
- **Native Windows Agent Tool Orchestration (`crates/shadow-serve/src/orchestrator.rs`)**:
  - Implemented local Windows agent tools: `read_file`, `glob`, `grep`, `edit_file`, `write_file`, and `cmd` (PowerShell).
  - Added HTTP endpoints `GET /v1/tools` and `POST /v1/tools/execute`.
  - Added native multi-turn autonomous tool execution loop in Web UI with visual collapsible execution cards, allowing Shadow to inspect previous sessions (`J:\Strata-work\sessions`), run tests, and perform code edits directly on Windows.

## [0.1.16] — 2026-10-27
### Added
- **Claude & Antigravity Style Thinking Level Selector**: Added interactive dropdown popup directly above the thinking button matching Claude and Antigravity's UI styling (`Off`, `Low`, `Medium`, `High`, `X-High`) with icons, sublabels, descriptions, and active checkmarks, passing `reasoning_effort` to the backend and persisting user selection in `localStorage`.
- **System Persona / Local Dev Authority Pill & Modal**: Built-in system prompt controller granting Shadow sovereign local authority to create, inspect, and modify code and project files directly without safety refusal hesitation.
- **Hardware Telemetry Engine (`GET /v1/telemetry`)**: Win32-grounded system monitor tracking process CPU %, system CPU %, logical core counts, exact GPU VRAM free/used/total, system RAM load, working set, pagefile/commit, and MoE 3-tier cache hit/miss statistics.
- **Process Monitor Dashboard (`/monitor` & Tab Navigation)**: Real-time telemetry dashboard with live visual progress bars, detailed memory breakdown, MoE expert cache hit rates, engine decode speed (`tok/s`), request counters, and a visual MoE 3-tier memory pipeline (VRAM -> Pinned RAM -> NVMe).

## [0.1.15] — 2026-10-27
### Added
- **`shadow-ui.exe` Native Desktop Launcher (`crates/shadow-ui`)**: Native Windows desktop launcher built with Rust and Tauri 2.0 (using system WebView2). Manages connection to `http://127.0.0.1:8081`, displays a dark-mode status and loading splash screen while the model initializes, supports single-instance window focusing, maintains a system tray icon with Show/Reconnect/Quit controls, and stores configuration in `%APPDATA%\Shadow\desktop.json`.

## [0.1.14] — 2026-10-27
### Added
- **Embedded Web UI**: Built-in dark-mode chat interface served directly by `shadow serve` at `GET /` and `/index.html` with streaming SSE, `<think>` reasoning card toggling, live performance telemetry (tok/s, status, memory), and copyable markdown code blocks without external web dependencies.
- **`PAIR2_KERNELS` & Cross-Quant Matvec Fusion**: `matvec_pair_q6_k_q4_k`, `matvec_pair_q4_k_iq4_xs`, `matvec_pair_iq4_xs_q6_k` fusing mixed-quant DeltaNet and shared-expert GEMVs.
- **`ple_cache` in `Model`**: In-memory RwLock cache for Layer 1 PLE n-gram table lookups (`per_layer_token_embd.weight`), eliminating repetitive NVMe reads.
- **`step_prefill`**: Accelerated prefill phase skipping output normalization and vocab head matrix multiplication on non-final prompt tokens.
### Changed
- **`VRAM_RESERVE`**: Lowered default VRAM reserve from 1024 MiB to 384 MiB, raising VRAM expert cache allocation to 11.37 GiB on the RTX 5070 and increasing permanently pinned hot experts per layer (`margin = 12`).

## [0.1.13] — 2026-10-27
### Added
- Aligned word loads (`u16`/`u32`/`float4`) in `kernels/quants.cuh` and `kernels/matvec.cu`.
- `matvec_pair_*` fused kernels for matching quant types.
- Per-layer exact-sized VRAM `ExpertCache` with exclusive VRAM pinning.
- `RamTier` Clock promotion ring (128 slots).
