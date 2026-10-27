# Changelog

All notable changes to Shadow will be documented in this file.

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
