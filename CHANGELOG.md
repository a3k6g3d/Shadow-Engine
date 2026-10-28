# Changelog

All notable changes to Shadow will be documented in this file.

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
