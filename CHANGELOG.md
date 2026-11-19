# Changelog

All notable changes to ShadowZero will be documented in this file.

## [0.1.69] — 2026-11-19
### Added
- **4-Way Trigram `SuffixDrafter` + `propose_after` Lookup Chaining + Online Cost-Per-ms `DraftPolicy` (`crates/shadowzero-serve/src/engine.rs`, `crates/shadowzero-cuda/src/session.rs`, `docs/charts/basalt-exl3-optimizations-v0163-to-v0169.svg`):**
  - Replaced linear `O(|hist|)` prompt-lookup scanning with an `O(1)` 4-way open-addressing trigram `SuffixDrafter` (`SUFFIX_WAYS = 4` most recent positions per 3-token FNV-1a key, backward match extension up to `32` tokens, and `propose_after` lookup chaining after neural draft sequences; `SHADOWZERO_SUFFIX_DRAFT=0` off).
  - Added an online cost-per-millisecond `DraftPolicy` (`SHADOWZERO_DRAFT_POLICY=0` off, `SHADOWZERO_LOOKUP_CHAIN=0` off, `SHADOWZERO_DYN_NEURAL_K=1` optional adaptive neural `k`) across 4 match-length buckets (`<6`, `6..11`, `12..23`, `>=24`) that tracks empirical round latency `cost_ms(t)` for verify window sizes `t = 1..=8` and expected tokens `mtp_tokens(t)`, dynamically choosing between `DFlash 2` / `MTP` neural drafting, wide lookup windows up to `t = 8` (`7` drafts), and lookup-chained neural windows (`propose_after`) with stale-size probing.
  - Fixed `top_buf` slice offsets in `session.rs` (`top_ready`, `head_top_at`, `topk_rows_at`) to allocate and index `64` max slices per row (`COLS_MAX * 64 * 64 * 4` bytes) instead of `vocab.div_ceil(4096) = 61`, preventing `nb_sp = 64` (`n = 8`, `max_sel = 1024`) from overflowing `slices_v` into `slices_i` during 8-token sparse verify passes.
  - Measured on `mixD18` (`--ctx 24576`): improved dual-GPU `varied36` warm median decode by **`+6.92%`** (`68.37 -> 73.10 tok/s`, `+3.4%` paired, `p90/worst: 67.84 -> 72.62 tok/s`), aggregate decode by **`+8.68%`** (`71.16 -> 77.34 tok/s`), and combined throughput by **`+9.76%`** (`85.55 -> 93.90 tok/s`); improved single-GPU (`RTX 5070 12 GiB`) `varied36` warm median decode to **`74.98 tok/s`** (`78.94 tok/s` aggregate decode, `+1.90%`, runs up to `80.2 tok/s`, `98.81 tok/s` combined).
  - Added before-and-after SVG performance chart (`docs/charts/basalt-exl3-optimizations-v0163-to-v0169.svg`) summarizing the full `v0.1.63 -> v0.1.69` progression (`65.10 -> 74.98 tok/s` single-GPU warm median `+15.2%`, `62.57 -> 73.10 tok/s` dual-GPU warm median `+16.8%`, `1,461.8 tok/s` 9.6k prefill, and `2,880 -> 4,009` single-GPU resident EXL3 expert slots `+39.2%`).

## [0.1.68] — 2026-11-19
### Added
- **Blackwell Programmatic Dependent Launch (`PDL`, `kernels/{prelude.cuh, exl3.cu, matvec.cu, forward.cu}`, `crates/shadowzero-cuda/src/{driver.rs, kernels.rs, session.rs}`):**
  - Added `cuLaunchKernelEx` (`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6`) in `driver.rs` (`pdl_enabled()`, `SHADOWZERO_PDL=0` off; enabled on `sm_90+` / `sm_120`) and `pdl_wait()` (`asm volatile("griddepcontrol.wait;");`, `#define PDL_NO_GRIDDEP 1` when `SHADOWZERO_PDL=0`) across hot sequential decode, verify, and CUDA-graph kernels (`exl3.cu`, `matvec.cu`, `forward.cu`).
  - Kept hardware end-of-kernel dependent launch signaling by default (`pdl_launch_dependents()` gated behind `#ifdef PDL_EARLY_SIGNAL`) so successor kernels that do not execute `griddepcontrol.wait` never overlap before predecessor global stores finish.
  - Verified bit-for-bit parity across `two_stage` (`6/6`, `212.54s`), `--exact` (`p2k2 9fc6a009`, `p3k 7bc06952`, `short 94ef35a6`), and `SHADOWZERO_PDL=0` vs `1` reply hashes (`g_short 8895ae93`, `g_2k2 bc3002ce`, `s_60 ac4f6e8d`); tightened p90/worst tail latency and reached **74.92 tok/s** warm median (`77.58 tok/s` aggregate, `88.65 tok/s` prompt, `81.24 tok/s` combined) on single-GPU `varied36` and **68.57 tok/s** warm median (`71.26 tok/s` aggregate, `85.69 tok/s` combined, `+0.1%` paired, `0` differing answers) on dual-GPU `varied36`.

## [0.1.67] — 2026-11-19
### Added
- **2-bit `Q2_P` `lm_head` proxy screen + exact sparse EXL3 128-row-block rescore + fused sparse top-K (`kernels/{exl3.cu, forward.cu}`, `crates/shadowzero-cuda/src/{kernels.rs, session.rs, session_dflash.rs}`, `crates/shadowzero-cuda/tests/head_proxy.rs`):**
  - Added GPU startup builder (`exl3_build_q2_proxy_{2,2h,3,3h,4,5,6,8}`, `3.84 ms` on RTX 5070) that decodes the pre-rotated EXL3 `output.weight` (`248,320 x 2,560`) into a 2-bit symmetric Lloyd-Max (`{-1.5, -0.5, +0.5, +1.5} * delta`, `delta = 0.9957 * std`) per-128-column proxy table (`161 MiB`, `100%` top-1 recall and `98.91%` top-64 recall at `128 / 1,940` blocks).
  - Added `proxy_q2_quant_x` (per-128 `int8` activation quantization + group sum), `proxy_q2_dp4a_bmax_{m1,m4,m8}` (`dp4a.s32.u32` via two `u32` LUTs at `~590 GB/s`, 88% of RTX 5070 peak), `proxy_pick_rows` (1024-item bitonic top-`k_blk` block selector per row), `proxy_compact_seen` (unique block union across verify batch rows with self-clearing `g_seen`), `matvec_cols_exl3_pre_sparse_m{4,8}s1_{2..8}` (bit-exact sparse EXL3 evaluation on selected 128-row blocks), and `topk_slices_blks` (`2048`-item sort) + `topk_merge_blks` (`tk_sort_n`) + `df_top16_blks`.
  - Achieved **`2.17x`** (`m=1: 1,070.08 -> 493.51 us`), **`2.35x`** (`m=3: 1,614.63 -> 687.72 us`), and **`2.16x`** (`m=4: 1,696.93 -> 787.25 us`) full `lm_head + top-K` kernel speedups on RTX 5070.
  - Enabled by default for `serve` (`--tensor-prompt` when `vocab >= 131,072 && hidden % 128 == 0`; `SHADOWZERO_HEAD_PROXY=0` to disable); never used in `--exact` (`head: exl3`) or full-logit paths (`ppl`).
  - Improved dual-GPU `varied36` warm median decode by **`+5.9%`** (`64.50 -> 68.32 tok/s`, `+3.7%` paired, `69.14 -> 71.31` aggregate `+3.14%`, `83.16 -> 85.91` combined `+3.31%`) and single-GPU `RTX 5070` `varied36` warm median decode by **`+6.1%`** (`70.33 -> 74.62 tok/s`, `+4.7%` paired, `75.06 -> 77.22` aggregate `+2.88%`), while passing `two_stage` (`6/6`), `needle_test.py` (`8/8` single & `8/8` multi), and bit-exact `--exact` parity.

## [0.1.66] — 2026-11-19
### Added
- **Packed per-32 symmetric `Q8` hyper-connections (`hc8`, `kernels/matvec.cu`, `crates/shadowzero-cuda/src/{kernels.rs, session.rs}`, `crates/shadowzero-cuda/tests/hc_q8.rs`):**
  - Added CPU BF16 -> `hc8` SOA quantizer (`quantize_bf16_hc8`: row-major `i8` weights + per-32 `fp16` scales) for all layer and output hyper-connection projections (`*.hc_attn_{down,inject,up}.weight`, `*.hc_ffn_{down,inject,up}.weight`, `output_hc_{down,up}.weight`) while keeping `fp32` activations and BF16 norms untouched.
  - Added coalesced 1-word-per-lane / 4-weight-per-lane kernels specialized by batch size (`hc8_down_inject_{m1,m4,cols}`), L1-broadcast fused up+mix kernels with zero shared-memory bank conflicts (`hc8_up_mix_{m1,m4,cols}`), and `uint4` + `mul.rn.f16x2` tensor-core prompt kernels (`mma_hc8_down_inject`, `mma_hc8_cols`), hitting `1.37x–1.69x` (`m=1..8`) and `5.99x` (`m=512`) kernel speedups over `BF16` (`mma_hc8_cols` `13.43x`).
  - Enabled by default for `serve` (`--tensor-prompt` when `(hidden * streams) % 32 == 0 && hc_rank % 32 == 0`; `SHADOWZERO_HC_Q8=0` to disable); `--exact` remains `hc: bf16`.
  - Freed **+530 resident EXL3 expert slots (+0.76 GiB; 3,479 -> 4,009)** on single-GPU RTX 5070 and **+381 slots (+0.52 GiB; 12,128 -> 12,509)** on dual-GPU (`--ctx 24576`), improving single-GPU `varied36` warm median decode **`+5.1%`** (`65.93 -> 69.32 tok/s`, `+7.4%` paired, `70.75 -> 75.97` aggregate `+7.38%`, `82.28 -> 88.01` combined `+6.96%`), dual-GPU `varied36` warm median **`+3.5%`** (`64.54 -> 66.80 tok/s`, `+3.21%` combined `96.39 -> 99.48`), and perplexity (`9.1066 -> 9.0106`, `-1.054%`), while passing `two_stage` (`6/6`), `needle_test.py` (`8/8` single & `8/8` multi), and bit-exact `--exact` parity.

## [0.1.65] — 2026-11-19
### Added
- **Packed 16-byte-aligned per-32 symmetric `Q8_KV` cache (`kernels/forward.cu`, `crates/shadowzero-cuda/src/{kernels.rs, session.rs}`, `crates/shadowzero-cuda/tests/kv_q8.rs`, `crates/shadowzero-serve/src/engine.rs`):**
  - Implemented per-32 symmetric `int8` + `fp16` scale KV storage (`272` bytes/head at `head_dim = 256`, zero wasted padding, 3.76× smaller than `f32`) with fused RMSNorm + RoPE + Q8 write (`head_norm_rope_qgk_rows_{q8,fq8}`), `kv_quant_q8_rows`, direct Q8-to-FP16 Tensor Core `m16n8k16` GQA attention (`attention_group_{sel,dense}_rows_tc_q8`), and VMM `kv_lend` / `ConvDisk` snapshot compatibility (auto-rejecting legacy `f32` snapshot stride on mismatch).
  - Enabled by default for `shadowzero serve` (`--tensor-prompt` when `head_dim % 32 == 0`; `SHADOWZERO_KV_Q8=0` to disable, `=1` to force); `--exact` remains bit-exact `f32`.
  - Freed **+0.84 GiB (+599 resident experts, +20.8% cache)** on single-GPU RTX 5070 and **+0.63 GiB (+459 resident experts)** on dual-GPU (`--ctx 24576`), improving dual-GPU prompts `+1.14%..+2.30%` (crossing `1,001.8 tok/s` at 3k, `1,461.8 tok/s` at 9.6k, `1,457.4 tok/s` at 15k), dual-GPU `varied36` decode `+2.9%` paired / `+1.75%` aggregate (`+17.8%` combined), single-GPU prompts `+0.76%..+1.87%` (`1,337.1 tok/s` at 15k), and `9.6k` decode `+4.2%..+5.0%`, while passing `ppl` (`+0.456%`), `needle_test.py` (`8/8` single & `8/8` multi), `two_stage` (`6/6` in both `f32` and `q8`), and bit-exact `--exact` parity.

## [0.1.64] — 2026-11-19
### Changed
- **EXL3 decode & prompt kernel optimizations (`kernels/exl3.cu`, `crates/shadowzero-cuda/src/{kernels.rs, session.rs}`):**
  - Added warp-coalesced `exl3_load_tile` for `K=2` (`R::WORDS == 16`, `SHADOWZERO_EXL3_COALESCE=0` off), packed `m<=4` split-K `matvec_cols_exl3_sk4_m4_*` (`SHADOWZERO_EXL3_SK4_M4=0` off), and pre-rotated `lm_head` (`exl3_rotate_rows_f32` + `matvec_cols_exl3_pre_m4s1_*` for `rows >= 131072, n in 3..=4`, `SHADOWZERO_EXL3_HEAD_PRE=0` off).
  - Measured on `mixD18` (`--ctx 24576`): dual-GPU `varied36` decode `+1.1%` paired (`62.57 -> 63.12 tok/s` warm median, `66.88 -> 67.66` aggregate) and prompt `+0.55%..+1.87%` across `2k2..15k`; single-GPU (`RTX 5070`) `varied36` decode `+1.2%` paired (`66.87 -> 67.69 tok/s` warm median, `72.41 -> 73.33` aggregate) and prompt `+0.38%..+1.49%`; `0` differing answers and bit-exact `--exact` parity.

## [0.1.63] — 2026-11-18
### Added
- **Opt-in Windows PC Control with always-on-top takeover notification & `Ctrl+Alt+Esc` kill-switch (`crates/shadowzero-serve/src/{pc_control.rs (new), orchestrator.rs, lib.rs, ui.rs}`, `tools/test_ui_logic.js`):**
  - Added server-gated native Windows PC Control (`OFF` by default, loopback-only `GET/POST /v1/pc-control`, mirrored in `/v1/telemetry` and `/v1/events`) exposing `pc_screenshot`, `pc_mouse`, `pc_keyboard`, `pc_window`, and `pc_wait` tools only when explicitly enabled.
  - Built a native Win32 always-on-top floating takeover banner (`WS_EX_TOPMOST | WS_EX_NOACTIVATE | WDA_EXCLUDEFROMCAPTURE`) and 4-edge click-through glowing screen border visible across all apps whenever PC Control is ON/active, complete with a clickable **Stop (Ctrl+Alt+Esc)** button and a global **`Ctrl+Alt+Esc`** emergency hotkey that immediately revokes control and stops active turns.
  - Added a **PC Control** toggle pill in the chat bar, a **PC Control** setting card, a native `<dialog closedby="any">` confirmation modal (with `autofocus` on **Cancel**), a live in-app `#pc-control-banner`, and multimodal `image_url` attachment on `pc_screenshot`.

## [0.1.62] — 2026-11-18
### Added
- **Auto-restore last open conversation & project on startup (`crates/shadowzero-serve/src/ui.rs`, `tools/test_ui_logic.js`):**
  - Persisted `shadow_last_chat_id` and `shadow_last_project_id` in `localStorage` across `openChat`, `saveCurrent`, `newChat`, `assignChatToProject`, and `deleteChat` / `deleteProjectFromModal`.
  - On startup (`restoreLastSession` / `resolveInitialRestore`), automatically reopens the last active conversation and expands its project (or restores the last active project context / newest active conversation on first run) without overriding explicit `#dash` / `#settings` hash navigation.

## [0.1.61] — 2026-11-18
### Fixed
- **Literal `</tool_response>` / XML control tokens inside tool outputs (`crates/shadowzero-serve/src/lib.rs`):**
  - Added `disarm_control_tags` when parsing `role: "tool"` and `role: "user"` messages in `/v1/chat/completions`: unwraps one outer `<tool_response>...</tool_response>` wrapper (avoiding double-wrapping with `chat.rs::render`) and inserts `\u{200b}` after `<` on any inner `<tool_response>`, `</tool_response>`, `<tool_call>`, `</tool_call>`, `<think>`, or `</think>` literals. Prevents reading or grepping source files that contain `</tool_response>` (such as `ChatModel.kt`) from prematurely closing Qwen's tool response block and causing 8-token empty replies on Retry.

## [0.1.60] — 2026-11-18
### Fixed
- **Literal `<tool_call>` / `</tool_call>` tags in thinking and written code files (`crates/shadowzero-serve/src/{engine.rs, ui.rs}`, `crates/shadowzero-model/src/chat.rs`, `tools/test_ui_logic.js`):**
  - Added `<tool_call>` lookahead in `engine.rs::Emitter` so mentioning `<tool_call>` inside `<think>` (e.g., when planning code that parses XML tool tags) stays inside reasoning instead of prematurely exiting `<think>` and leaking thought text into the reply; genuine `<tool_call>\n<function=...>` blocks opened inside `<think>` still transition cleanly into content.
  - Hardened `parse_tool_calls` (`chat.rs`) and `parseToolCallsFromText` / `stripToolCalls` / `hasPendingToolCall` / `classifyEmptyStop` / `unfinishedTurnNudge` (`ui.rs`) to match genuine `<tool_call>\s*<function=...> ... </function>\s*</tool_call>` spans, preventing files containing literal `<tool_call>` / `</tool_call>` strings (such as `MainActivity.kt`) from truncating mid-file or triggering false `incomplete or unrecognized tool call` auto-nudges.

## [0.1.59] — 2026-11-18
### Added
- **Sidebar Projects tab, project/job folders, newest-to-oldest ordering & conversation archiving (`crates/shadowzero-serve/src/{sessions.rs, lib.rs, ui.rs}`, `tools/test_ui_logic.js`):**
  - Added persistent project/job storage (`projects.json` alongside sessions) and REST endpoints (`GET/POST /v1/projects`, `PUT/DELETE /v1/projects/<id>`, `POST /v1/projects/<id>/archive-older`), plus extended session metadata (`project_id`, `archived`) via `PUT /v1/sessions/<id>` without disturbing chronological timestamps; archived sessions are preserved during pruning.
  - Added an accessible **Chats | Projects** tab bar (`role="tablist"` + `focusgroup="tablist nomemory"`, keyboard arrow/Home/End navigation, `hidden="until-found"` with `beforematch` sync) in the left sidebar.
  - Added native `<dialog closedby="any">` modals to create/edit/delete projects and move conversations between projects, quick `+` new chat inside any project (with an active project pill in the composer bar and rollover continuity), newest-to-oldest conversation ordering inside every project, per-conversation **Archive / Restore** actions, one-click **Archive oldest N** per project, and collapsible **Archived** sections in both tabs.

## [0.1.58] — 2026-11-18
### Fixed
- **Clean command termination & UI abort recovery (`crates/shadowzero-serve/src/{ui.rs, orchestrator.rs, lib.rs}`, `tools/test_ui_logic.js`):**
  - Passed `stepSignal` into `executeLocalTool` / `executeLocalToolRemote` so clicking **Stop** or **Send now** aborts in-flight `/v1/tools/execute` calls cleanly, records paired `Stopped` `<tool_response>` history entries, marks remaining calls `Skipped`, clears `stepBusy`, and dispatches any queued `Sending next` steering prompt without stalling.
  - Replaced blocking `cmd.output()` in `orchestrator.rs` with streaming stdio capture + null stdin + per-command Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`), configurable command timeout (`timeout_secs` / `SHADOWZERO_CMD_TIMEOUT_SECS`, default `300s`), `500ms` post-shell-exit pipe grace so background daemons holding stdout/stderr handles (such as `GradleDaemon`) cannot hang tool calls indefinitely, and `CancelOnDrop` HTTP disconnect propagation in `/v1/tools/execute`.

## [0.1.57] — 2026-11-18
### Changed
- **Editorial chat timeline & uncollapsed tool steps (`crates/shadowzero-serve/src/ui.rs`, `tools/test_ui_logic.js`):**
  - Replaced heavy bordered/throbbing assistant `.bubble` cards and purple `.think-block` boxes with an open, borderless dark canvas matching the `820px` reading column (keeping compact right-aligned rounded user pills and continuous `.msg-row.assistant + .msg-row.assistant` spacing).
  - Fixed chronological streaming & saved-chat rendering so intermediate model prose stays visible directly in the timeline instead of being pushed inside `.tool-group-body`.
  - Switched `.tool-group-head` and `.think-header-left` to accessible `<button type="button" aria-expanded>` disclosures, and kept completed tool step lists **open by default** with a new **Settings > Display** toggle (`Auto-collapse completed tool steps into one summary line`, `shadow_collapse_tools`, plus a Command Palette toggle).

## [0.1.56] — 2026-11-18
### Added
- **Remote pairing token, auth gate and CORS for LAN peers (`crates/shadowzero-serve/src/{auth.rs (new), lib.rs, ui.rs}`, `docs/PLAN_APK_2026-11-13.md` Phase 1):** Added `auth::Pairing` (four-digit one-time pin with five-minute TTL and five wrong-try lockout, constant-time SHA-256 hash comparisons, a 32-byte hex bearer token whose hash only is persisted to `%LOCALAPPDATA%\ShadowZero\pair.json`), a middleware in `lib.rs` rejecting unpaired non-loopback peers on every `/v1*` route (`Authorization: Bearer`, with a `?token=` fallback for SSE) while loopback stays exempt for the desktop UI and benches, the routes `GET /pair/pin` (loopback only) / `GET /pair/status` / `POST /pair` / `DELETE /pair` (a remote device must present its own token; an empty token is honoured for this machine only), reflected-Origin CORS with OPTIONS preflight for browser clients, and a **Remote access** card in the Settings panel of the embedded UI showing the pin with an expiry countdown, the paired state, and a forget-device button (five new tests in `auth.rs`; `cargo test -p shadowzero-serve` = 66 passed, `node tools/test_ui_logic.js` green including the zero-JS-syntax-errors check).
- **On-demand GPU VRAM park & reload (`crates/shadowzero-serve/src/{engine.rs, lib.rs}`, `crates/shadowzero-ui/src/main.rs`):** Added loopback-only `POST /v1/engine/park` and `POST /v1/engine/resume` endpoints, split `engine::run` into reloadable `run_cycle` lifecycles via `Cmd::{Chat, Park, Resume}`, updated `/health` (`503` while "parked") and `shadowzero-ui` readiness polling, and auto-woke parked servers when new chat turns arrive.
- **Media HTTP serving & configurable MCP timeouts (`crates/shadowzero-serve/src/{lib.rs, mcp.rs}`):** Added `GET`/`HEAD /v1/media/<file>` (`png`, `jpg`, `jpeg`, `webp`, `mp4`, `webm`) with path-traversal protection and HTTP `206` byte-range video streaming, propagated `SHADOWZERO_SERVE_URL` and `SHADOWZERO_MEDIA_DIR` to MCP children, and added `timeout_secs` / `SHADOWZERO_MCP_TIMEOUT_SECS` support in `mcp.json`.
- **Inline chat media cards (`crates/shadowzero-serve/src/ui.rs`, `tools/test_ui_logic.js`):** Added `.media-strip` / `.media-card` styling, `renderMediaCardsHtml`, friendly `toolStep` labels, `"parked"` status badge/send-guard updates, and saved-conversation media hydration.
- **Local HD photo, HD video & 4K AI upscaling MCP server (`tools/media_mcp.py`, `tools/test_media_mcp.py`, `mcp.json`):** Implemented stdio MCP server exposing `generate_photo` (`Z-Image-Turbo`), `generate_video` (`Wan2.2-TI2V-5B-Diffusers` text-to-video & image-to-video with FP8 layerwise casting + VAE tiling/slicing), and `upscale_media` (`Real-ESRGAN` `x4`/`x2` + NVENC H.264 up to `3840x2160` 4K UHD) running in isolated child workers on the highest-VRAM GPU with guaranteed `finally` engine resume.

### Fixed
- **CUDA drop diagnostics (`crates/shadowzero-cuda/src/{driver.rs, pinned.rs}`):** Logged first-occurrence non-zero `cuMemFree` / `cuMemFreeHost` / context bind errors during teardown and included free/total VRAM in `cuMemHostGetDevicePointer` errors.

## [0.1.55] — 2026-11-18
### Fixed
- **Image/vision encoding on EXL3 & safetensors model directories (`crates/shadowzero-serve/src/vision.rs`)**: Added `resolve_vision_model(...)` (with `SHADOWZERO_VISION_MODEL` override and recursive companion `.gguf` discovery inside the model directory, sibling folders such as `J:\Models\`, and repo `models/`, filtering out `mmproj`, embedders, and secondary `00002-of-*` shards). Previously `Encoder::start(...)` passed the EXL3 directory path directly to `strata-vision.exe --model`, failing with `ERR cannot open the text model`.

### Added
- **Restored `High` level in Reasoning Depth menu (`crates/shadowzero-serve/src/ui.rs`, `tools/test_ui_logic.js`)**: Re-added `High` (`~8k` token thinking budget, between `Medium` and `X-High`) to `#thinking-dropdown` and the command palette (`palCmds`), and removed the `high -> medium` coercion on page load so saved `High` preferences persist.

## [0.1.54] — 2026-11-17
### Improved
- **Web UI clarity, navigation & accessibility (`crates/shadowzero-serve/src/ui.rs`, `tools/test_ui_logic.js`)**:
  - Hid the duplicate header `+ New chat` button (`#newchat-btn`) when the chat sidebar is open so only one `+ New chat` action renders at a time (still shown when the sidebar is collapsed or on Dashboard/Settings).
  - Made mid-reply interruption visible and cancellable: `#send-btn` displays `Send now` during generation, `interruptAndSend(...)` shows a `Sending next` steering banner inside `#queue-bar` with a `Cancel` button (`cancelInterrupt()`) that restores the draft and attachments cleanly if canceled.
  - Added a live context-window usage pill (`#ctx-pill`, `formatCtxUsage(...)`) in the chat controls bar showing token percentage/window, warning styling at `>= 75%` / `>= 90%`, auto-rollover state tooltip, and one-click jump to Settings (`openRolloverSettings()`).
  - Added a sidebar `⌘` command palette trigger and listed `Ctrl+Shift+P` Commands in `#welcome .kbd-hints`; migrated `#persona-modal` and `#palette` to native top-layer `<dialog closedby="any">` modals with `overscroll-behavior: contain`, `autofocus`, automatic focus restoration, progressive pointer light-dismiss fallback, ARIA live regions/pressed/expanded states, and `>= 24px` touch/focus target sizes.
  - Added `chatEpoch` guarding in `sendMessage()`, `newChat()`, `openChat()`, and `deleteChat()` so starting or switching chats mid-stream never leaks aborted output or triggers stale `saveCurrent()` / `runQueue()`.

## [0.1.53] — 2026-11-17
### Changed
- **Dual-GPU prompt & decode speedups (`crates/shadowzero-model`, `crates/shadowzero-cuda`, `kernels/`, `docs/STATUS.md` 170–173)**:
  - **Exp 1 (`SHADOWZERO_PLE_FIFO=0`, `SHADOWZERO_PLE_PROMPT_ASYNC=0`)**: Added FIFO eviction at 65,536 entries via `ple_extend_rows` and overlapped host PLE row prefetching across sub-chunks in `step_batch_opt` (`+4.4%` at `9.6k`, `+5.5%` at `15k` dual-GPU prompts).
  - **Exp 2 (`SHADOWZERO_EXL3_M4=0`, `SHADOWZERO_RES_SMALL_BITMASK=0`)**: Added packed `m<=4` EXL3 `matvec_cols_exl3_m4s4_*` kernels, `m=1 -> mv2` dispatch, and `O(1)` bitmask group planning in `route_plan_res_small` (`+0.94%` decode total / `+1.2%` paired median on `varied36`).
  - **Exp 3 (`SHADOWZERO_PAIR_SINGLE_PASS=0`)**: Added single-pass Stage 0 -> Stage 1 `rparts` handoff when `subs.len() <= 4`, keeping `use_two_halves` above 4 sub-chunks (`+13.1%` at `2.1k`, `+11.7%` at `2.6k`, `+2.5%` at `3.1k` prompt ingestion).
  - **Exp 4 (`SHADOWZERO_MOE_HOST_FAST=0`)**: Cached `Arc<Vec<MoeLayerMeta>>` and `&'static str` EXL3 kernel names, replaced heap routing/group-plan allocations with stack `route_into` / `MoeScratch` / `grouped_plan_into` / `[u64; 8]` bitmask scan / `download_mapped_into`, reducing Stage 1 host overhead from `86` to `73 us/layer` (`+0.8%` paired decode / `+1.1%` prompt / `+0.85%` combined on `varied36`; rejected `GROUPED_DIRECT`, `GTAB_H2D`, and `PAIR_DF_STEP` reverted after measurements).

## [0.1.52] — 2026-11-17
### Removed
- **Four UI features removed at the user's request (`crates/shadowzero-serve/src/ui.rs`, `crates/shadowzero-serve/src/lib.rs`)**: the diff/edit card (the `.diff-card` CSS, the `renderDiffCard`/`renderDiffBody`/`applyDiff`/`loadDiffContext` JS, the diff mount in the tool card, and the `POST /v1/diff` route + `diff()` handler); the workspace file tree (`tree.rs` deleted, the `GET /v1/tree` + `GET /v1/file` routes and their handlers removed, the sidebar `#tree-panel` and `#file-modal` HTML, the `renderTree`/`toggleTree`/`openFile` JS and their Settings and command-palette entries); the `/v1/events` SSE client (`connectEvents`/`EventSource`/`eventsConnected` in `ui.rs` — the poll of `/v1/telemetry` is the only source again; the server `/v1/events` route and `events.rs` stay for other consumers); and the permission pill (the `.perm-*` CSS, the `PERM_MODES`/`permissionGate`/`permissionReason`/`commandIsSafe`/`selectPermMode` JS, the `#perm-btn` dropdown HTML, the gate in tool execution — tools run directly as they did before this button — and the palette's permission-mode commands). The command palette (Ctrl+P), the server-side `/v1/sessions` endpoints and the client session wiring used by auto-rollover (0.1.49) are kept.
## [0.1.51] — 2026-11-17
### Changed
- **Sending while ShadowZero is working skips the queue cleanly (`crates/shadowzero-serve/src/ui.rs`)**: Sending a message while `isGenerating || stepBusy` is active now calls `interruptAndSend(...)`, aborting the in-flight stream/permission gate cleanly, recording any active or skipped `<tool_response>` entries in `history` so no orphan `<tool_call>` remains (and pushing only `stripToolCalls(rawContent)` on aborted streams), and immediately sending the new message once the step settles. Added `stepBusy` tracking to prevent overlapping `sendMessage()` loops if Stop/Send is clicked while a tool call is finishing.

## [0.1.50] — 2026-11-17
### Fixed
- **Catch hallucinated `#### N tool calls` headings & imperative `"Now <verb>..."` pauses (`crates/shadowzero-serve/src/ui.rs`)**: Updated `unfinishedTurnNudge(...)` to detect and strip trailing `#### N tool calls` markdown headings (where the model imitated the UI summary instead of emitting `<tool_call>`), and expanded the mid-task intent matcher to catch `"Now read/check/edit..."` and `"see"` pauses.

## [0.1.49] — 2026-11-17
### Added
- **Automatic conversation & notes rollover (`crates/shadowzero-serve/src/ui.rs`)**: Added `shouldAutoRollover(...)`, `buildRolloverPrompt(...)`, `performRollover(...)`, and the Settings toggle (`#set-auto-rollover`, on by default). When a reply hits the context limit (`cutOff`), receives a context-too-long (`413`) error, or reaches `>= 90%` of `serverCtx` during an active tool loop, ShadowZero saves the current chat to `/v1/sessions`, carries the pinned `Notes` scratchpad and a compact handoff summary into a fresh conversation, and continues automatically (with anti-loop guards).

## [0.1.48] — 2026-11-17
### Fixed
- **Dropped tool-call auto-nudge no longer disabled when `finishActive()` is false (`crates/shadowzero-serve/src/ui.rs`)**: Removed the `&& o.finishActive` gate on `dropped_tool_call` in `unfinishedTurnNudge(...)`, expanded the intent match to cover period-terminated forward-looking steps (`"Now let me edit lib.rs."`), and made `loadTools()` retry automatically when `availableTools` is empty at startup or before `sendMessage()`.

## [0.1.47] — 2026-11-17
### Fixed
- **Server-side conversation list not appearing in sidebar (`crates/shadowzero-serve/src/ui.rs`, `sessions.rs`)**: Fixed `refreshChats()` reading `(j && j.sessions) || []` against `GET /v1/sessions` (which returns a top-level JSON array `[...]`), guarded `migrateChats()` so `localStorage` is only cleared when `POST /v1/sessions/import` returns `200 OK` and sequenced before `refreshChats()`, fixed `saveCurrent()` snapshot/ID race across `newChat()` / `openChat()`, and fixed `chat_title()` whitespace normalization in `sessions.rs`.

## [0.1.46] — 2026-11-17
### Fixed
- **Reset auto-nudge counter on real tool calls (`crates/shadowzero-serve/src/ui.rs`)**: Made `autoNudges` consecutive rather than per-turn total by resetting `autoNudges = 0` whenever the model executes a real tool call, so long multi-step coding tasks don't exhaust the 3-nudge budget over dozens of steps, and added a visible stop notice + `Retry` button if 3 consecutive nudges still fail.

## [0.1.45] — 2026-11-17
### Fixed
- **Auto-resume when the model drops a `<tool_call>` mid-task (`crates/shadowzero-serve/src/ui.rs`)**: Added `unfinishedTurnNudge(...)` (capped at `3` automatic nudges per user turn using ephemeral `nudgeMsgs` so saved `history` stays clean) to automatically continue when an active tool loop stops on a malformed `<tool_call>`, an empty mid-task reply, or a mid-step trailing intent (e.g. `"Let me read the exact chat-store block:"` ending without emitting `<tool_call>`).

## [0.1.44] — 2026-11-17
### Fixed
- **Header `Speed:` pill now matches Dashboard `DECODE SPEED` (`crates/shadowzero-serve/src/ui.rs`)**: Added `latestDecodeSpeed(d)` so the top header pill displays the latest reply's decode speed from `d.recent` (matching `#d-tps` instead of drifting behind the session-wide lifetime average `d.process.tok_per_sec`, while keeping the session average in the pill tooltip and falling back to it when `recent` is empty).
- **Clear reason instead of silent stops on empty/unparsed replies (`crates/shadowzero-serve/src/ui.rs`)**: Added `classifyEmptyStop(...)`, trailing SSE buffer flush, and premature-EOF detection (`!streamFinished && !finishReason`) so a turn that ends after thinking with an empty reply, an incomplete/unrecognized `<tool_call>` stripped to empty, context cut-off, or an unexpected stream end displays an explicit stop reason and `Retry` button instead of leaving an empty bubble with `Completed in ...s`, and does not append an empty assistant turn to `history`.

## [0.1.43] — 2026-11-17
### Fixed
- **Post-folder-rename path cleanup (`J:\Shadow` -> `J:\ShadowZero`)**: Updated remaining path references in `mcp.json`, `tools/swap_and_relaunch.ps1` (including `Get-Process shadowzero`), `tools/make_headless_shortcut.ps1`, `crates/shadowzero-serve/src/ui.rs` (chat suggestion chip and `%APPDATA%\ShadowZero` help text), `docs/USAGE.md`, `CLAUDE.md`, and `tools/` scripts, and refreshed Start Menu shortcuts.

## [0.1.42] — 2026-11-17
### Renamed project to ShadowZero
- **Full codebase, binary & native DLL rename**: Renamed workspace crates (`shadowzero`, `shadowzero-model`, `shadowzero-cuda`, `shadowzero-serve`, `shadowzero-bench`, `shadowzero-eval`, `shadowzero-host`, `shadowzero-ui`), binaries (`shadowzero.exe`, `shadowzero-ui.exe`), native CUDA DLLs (`shadowzero_mmq.dll`, `shadowzero_fused.dll`, rebuilt with `nvcc`), runtime env vars (`SHADOWZERO_*`), caches (`ShadowZeroConvCache`, `%LOCALAPPDATA%\shadowzero\kernels`), `%APPDATA%\ShadowZero`, UI branding, tools (`shadowzero_harness.py`, `shadowzero_memory.py`), Start menu shortcuts, and GitHub repository (`a3k6g3d/ShadowZero`).
- **Fixes along the way**: Fixed format/lifetime compile issues in `crates/shadowzero-serve/src/sessions.rs` and `lib.rs`, kept `shadowExe` config alias and `localStorage` compatibility, and added `ShadowZeroConvCache` to `semsearch.rs` skip list.

## [0.1.41] — 2026-11-16
### Faster dual-GPU prompts & DFlash drafting
- **Better overlap across both GPUs (`HopPipe` + `tap_feat`)**: Switched the stage handoff to double-buffered async copies on a dedicated upload stream, packed DFlash's capture features 5x smaller (`taps_below` only), and moved Stage 0's second macro-half inside the worker scope so Stage 0 and Stage 1 crunch long prompts at the same time.
- **Graphed DFlash drafter**: Captured the 5-layer DFlash draft loop and head walk into CUDA graphs using a tiny dynamic metadata buffer, uploaded the mask token embeddings once, and trimmed the head/walk to the requested `--spec 3` slots instead of all 7 — cuts drafting overhead ~20% (`~4.8 ms` down to `3.1–3.9 ms`) with identical drafts.
- **Cleaner verify planning**: Enabled the single-kernel `m4s4` small-batch MoE planner on resident layers and swapped the `--tail-skip` heap map for stack bitmasks.
- **Numbers (`mixD18`, `ctx 24576`)**: `72.32 tok/s` decode (`72.26` median, up from `63.23` / `60.00` in `0.1.37`), `125.19 tok/s` short prompts, `1298.5 tok/s` on 9.7k prompts (`+11.6%` over single GPU), and `1397.0 tok/s` on 15.2k prompts (`+16.0%` over single GPU). `two_stage` (`6/6`), `shadowzero-cuda` (`16/16`), and `shadowzero-serve` (`57/57`) all pass.

## [0.1.40] — 2026-11-16
### Chat UI polish
- Kept multi-step tool turns inside a single assistant bubble in `ui.rs`, rolling intermediate thinking blocks and tool calls into one collapsible group above the final reply with a single aggregated tok/s footer (both live and after reloading saved chats).

## [0.1.39] — 2026-11-16
### Streaming tool calls fix
- Fixed `ui.rs` to accumulate streamed `delta.tool_calls` chunks properly, run the requested local/MCP tools, and keep them in chat history so multi-step tool conversations continue instead of stalling after the first thinking block.

## [0.1.38] — 2026-11-16
### Multi-GPU dashboard & telemetry
- Updated NVML telemetry (`telemetry.rs`) to report every GPU (`gpu.devices`), and updated the Dashboard card, sparkline, and top header pill to display both cards side by side (utilization, VRAM, memory controller, temperature).

## [0.1.37] — 2026-11-16
### Two-GPU resident prefix & pinned RAM tier fixes
- **Lifted the two-card pinned RAM cap**: Switched pinned RAM allocation to 1 GiB `cuMemHostAlloc` chunks (`SHADOWZERO_PIN_ALLOC=1`) so WDDM doesn't double-count registered pages across both GPUs.
- **Auto card ordering & layer split**: When two GPUs are passed with `--gpus`, ShadowZero puts the larger-VRAM card first (e.g., `5060 Ti 16 GiB` holding layers `0..15` 100% resident with full-layer DeltaNet CUDA graphs) and the faster card second (`5070 12 GiB` holding layers `15..48`, head, DFlash, and RAM-tier misses).
- **2D kernels & macro-half prefill**: Collapsed multi-row `mv_cols` and `hc_write` loops into single 2D launches, added a two-half prompt pipeline (`step_batch_opt`) to cut expert reloads on `mixD18`, and verified all `two_stage` parity tests (`6/6`).

## [0.1.36] — 2026-11-10
### Added / Changed
- **Web UI & Desktop Shell Modernization (`crates/shadowzero-serve/src/ui.rs`, `crates/shadowzero-ui/ui/index.html`)**:
  - **Tabular numerals (`font-variant-numeric: tabular-nums`)** across the live header telemetry pills, Dashboard cards, recent-requests table, message timing metadata, and Tauri launcher status timer so 1 Hz `/v1/telemetry` updates never jitter horizontally.
  - **Auto-sizing form controls (`field-sizing: content`)** on `#prompt-input`, `#notes-text` (pinned scratchpad), and `#set-mem-kind` with explicit `width: 100%` / `min-block-size` / `max-block-size` bounds and a feature-detected JS fallback (`CSS.supports('field-sizing', 'content')`).
  - **IME-safe Enter-to-submit (`!e.isComposing && e.keyCode !== 229`)** on `#prompt-input`, `#set-mem-new`, and inline memory editing (`.mem-edit`), plus accessible `aria-label` attributes on form inputs.
  - **Single-line overflow resilience (`min-inline-size: 0`, `overflow: hidden`, `text-overflow: ellipsis`, `table-layout: fixed`)** across Dashboard `.kv` rows, sidebar conversation items (with `outline-offset: -2px` so scroll containers never clip focus rings), and the recent-requests table.
  - **Ergonomic polish & keyboard shortcuts**: floating `↓ Jump to latest` button (`#scroll-bottom-btn`) when scrolled up during long conversations, `Copied ✓` feedback on code blocks, global shortcuts (`Ctrl+K` / `Ctrl+/` to search conversations, `Ctrl+Shift+O` for new chat, `Esc` to close modals/dropdowns or stop active generation), keyboard shortcut badges on the Welcome card, and updated `DEFAULT_PERSONA` tool description (`search finds files and earlier outputs by keyword, ranked by BM25 and path match`, with automatic `localStorage` migration from the old `by meaning` wording).

## [0.1.35] — 2026-11-10
### Added / Changed
- **Prompt-Path & Decode Round 3 Optimizations (`crates/shadowzero-cuda/src/kvvmm.rs`, `crates/shadowzero-cuda/src/mmq.rs`, `crates/shadowzero-cuda/src/session.rs`, `crates/shadowzero-cuda/src/pinned.rs`, `crates/shadowzero/src/main.rs`)**:
  - **VMM physical handle pool & scratch zero-skip (`SHADOWZERO_VMM_POOL`, default `1`)**: `VmmRegion::release_tail_to` and `remap_tail_from` now pool the 2 MiB `CUmemGenericAllocationHandle` allocations across `lend_kv` and `unlend_kv` (`GpuSession::vmm_pool`, released on `Drop`), avoiding `~2,600` `cuMemCreate` / `cuMemRelease` driver calls per prompt, and `map_bytes_from(..., false)` skips the redundant `1.5 GiB` `cuMemsetD8` on scratch `kv_extra`. Cuts `lend_cache` + `restore_cache` VMM overhead by `70–90 ms` per prompt (`lend_cache`: `91 ms → 40 ms` at 3k, `45 ms → 19 ms` at 15k; `unlend_kv`: `53 ms → 31 ms` at 3k, `26 ms → 16 ms` at 15k).
  - **Sub-chunk (`si > 0`) fast path & Layer-0 pipeline ordering (`SHADOWZERO_SUBCHUNK_FAST`, default `1`)**: On sub-chunked passes (`m > 2,048`), `dev_plan_buckets` skips rebuilding and re-uploading `dp_eaddr` and `dp_copied` on sub-chunks `si > 0` (since `cache.map` is invariant across sub-chunks of layer `l` and `si == 0` already waited on `sp.events[p]`), `moe_staged` skips the redundant `stream_wait_event(sp.events[p])` and runs a single unmasked `fused_experts` phase on `si > 0`, and on `first_of_pass` (`l == 0`) defers `early` (`stage_copy(1)` + `stage_pipe_read(2)`) until after sub-chunk 0's MoE launches so the GPU executes Layer 0 sub-chunk 0 while `stage_pipe_read(1)` finishes on the CPU read pool.
  - **Deduplicated `shadowzero_mmq_quantize` on `bb.u` (`SHADOWZERO_MMQ_REUSE`, default `1`)**: Caches `(x_ptr, k, m, xs, ds_layout)` in `GpuSession::mmq_y_tag` (invalidated whenever `bb.u` or `mmq_y` is written in `hc_read_batch`, `ple_batch`, `big_batch_on`, `big_batch_off`) so back-to-back MMQ projections sharing `bb.u` (`ffn_gate_shexp` + `ffn_up_shexp` on all 48 layers, `attn_qkv` + `attn_gate` on 36 DeltaNet layers, `attn_q` + `attn_k` + `attn_v` on 12 attention layers) quantize `bb.u` once instead of 2–3 times (`~576–768` fewer quantization kernel launches per 15k prompt).
  - **Overlapped cold file reads with RAM-tier DMA in `restore_cache` (`SHADOWZERO_RESTORE_OVERLAP`, default `1`)**: Queues the RAM-tier `upload_batch` (`~3.2 GiB` over PCIe) onto `copy_stream` asynchronously *before* reading cold file-only evicted experts into `sp.bufs[0]`, overlapping the CPU/NVMe read with the GPU copy engine's RAM-tier transfer (`restore_cache`: `205 ms → 166 ms` at 3k, `180 ms → 154 ms` at 9.6k, `170 ms → 151 ms` at 15k).
  - **Direct mapped table reads in `routed_overlapped` and `routed_qwen` (`SHADOWZERO_DIRECT_TAB`, default `1`)**: `stage_slice` writes per-layer routing pointer/weight/permutation tables into the mapped staging buffer and passes `stage.dev_ptr + at` directly to `mv_batch` and `moe_combine_qwen_perm`, saving 48 `copy_words` kernel launches per single-token decode step.
  - **Warning-only RAM floor check in `PinnedRegion::build` (`crates/shadowzero-cuda/src/pinned.rs`)**: `PinnedRegion::new` and `PinnedRegion::new_chunked` now log a `RAM warning:` instead of hard-failing when `available_ram() < len + ram_floor()`, relying on the chunked-tier watchdog (`pinned::start_watchdog`) to trim 1 GiB chunks if free RAM dips below `ram_floor()`.
  - **CLI switch `--no-prompt-opt-r3`** (and updated `--no-prompt-opt`) to disable all five Round 3 optimizations for clean A-B-B-A benchmarking.
  - **Measured (`bench/prompt-opt-2026-11-10/results_r3.json`, interleaved A-B-B-A, bit-identical hashes on all 5 workloads)**:
    - Warm `3k_a` (2,867 tok): **`2.99 s` (`958.2 tok/s`)** vs `3.08 s` (`931.1 tok/s`) in Round 2 (**+2.9%**, and **+15.1%** vs pre-opt `3.52 s`)
    - Warm `3k_b` (2,622 tok): **`2.93 s` (`896.2 tok/s`)** vs `3.05 s` (`860.4 tok/s`) in Round 2 (**+4.2%**, and **+23.4%** vs pre-opt `3.82 s` at 43 GiB tier)
    - `9k6` (9,673 tok): **`5.68 s` (`1,701.6 tok/s`, mean `5.71 s` / `1,693.7 tok/s`)** vs `5.95 s` (`1,627.2 tok/s`) in Round 2 (**+4.1%**, and **+6.6%** vs pre-opt `6.09 s`)
    - `15k` (15,034 tok): **`8.52 s` (`1,763.8 tok/s`, mean `8.54 s` / `1,761.0 tok/s`)** vs `8.60 s` (`1,748.4 tok/s`) in Round 2 (**+0.7%** over Round 2, **+6.0%** vs pre-opt `9.05 s`, and **+16.2% faster than Strata 0.2.11's `9.90 s` / `1,543 tok/s`**)
    - `decode200` (200 tok): **`52.9 tok/s` (`53.1 tok/s` warm, +11.1% faster than Strata's `47.7 tok/s`)**.

## [0.1.34] — 2026-11-10
### Changed
- **Removed Remaining Hard RAM Guards (`crates/shadowzero-cuda/src/session.rs`, `crates/shadowzero-serve/src/engine.rs`)**:
  - Removed the leftover `available_ram() >= ram_floor() + ...` retention guard on `sp_spare` and `staging` in `GpuSession::with_tiers` and `restore_cache` (`PinnedRegion::new` checks the floor at allocation time and the chunked tier watchdog `pinned::start_watchdog` handles low-RAM pressure thereafter).
  - Changed the hard error refusal in `expand_pictures` (`not enough free RAM to encode a picture`) to a log warning (`shadowzero serve: warning: encoding a picture with ... GiB free`), matching `answer()`'s chunked-tier release behavior from STATUS 90.
  - Updated the in-RAM conversation snapshot check in `chat()` from the hardcoded `RAM_FLOOR + 3 GiB` (`7 GiB`) to the configured `ram_floor()`.

## [0.1.33] — 2026-11-10
### Added / Changed
- **Prompt Path Fixed-Cost & Fused-Planner Optimizations (`crates/shadowzero-cuda/src/session.rs`, `crates/shadowzero-serve/src/engine.rs`, `crates/shadowzero-serve/src/lib.rs`, `crates/shadowzero/src/main.rs`)**:
  - **Pre-allocated & retained `StagePipe` and `staging` buffers (`SHADOWZERO_KEEP_PIPE`, default `1`)**: `GpuSession::with_tiers` now pre-allocates the 256-slot pinned staging buffer (`502 MiB`) and the three `StagePipe` read-ahead buffers (`546 MiB` at tier 45) during session creation, and removes the old `available_ram() >= ram_floor() + 4 GiB` drop guard in `restore_cache` (relying on `PinnedRegion::new`'s floor check and the chunked tier watchdog `pinned::start_watchdog`). Eliminates `~225 ms` on the first prompt (`3k_a`) and `~68–70 ms` of `cuMemHostRegister` / `cuMemHostUnregister` overhead on every subsequent prompt (`lend_cache` + `restore_cache`).
  - **Fused-expert fast device planning (`SHADOWZERO_FUSED_FAST_PLAN`, default `1`)**: When `shadowzero_fused.dll` covers a layer (`fused_covers`), `dev_plan_buckets` skips zeroing the `7.6 MB` width-class record tables (`2.9 GB` of GPU memset per 15k prompt), skips `plan_scan_w` and `plan_fill_w`, uploads only the 4 KB base-address part of `dp_eaddr` (instead of 12 KB + `dp_lay`), quantizes `bb.u` once per sub-chunk across phases 0 and 1, and bypasses `fu.mask` when a layer has no resident VRAM-cache experts (`ph == 1 && !any_res`).
  - **Batched DMA in `restore_cache` (`SHADOWZERO_RESTORE_SINGLE=0` default)**: Replaces `1,775–1,809` individual `cuMemcpyHtoDAsync` driver calls when restoring evicted experts from the RAM tier and `StagePipe` with a single `cuMemcpyBatchAsync` batch upload.
  - **Parallel host token-embedding dequantization (`SHADOWZERO_EMBED_PAR`, default `1`)**: Parallelizes the `Q6_K` host embedding dequantization loop across CPU threads (`std::thread::scope`) in `batch()`.
  - **Honor `"cache_prompt": false` in `/v1/chat/completions`**: `parse_chat` / `chat` now parses `"cache_prompt"` (default `true`) and skips taking host shared-prefix and conversation snapshots (`sess.snapshot_to(true, true)`) when `"cache_prompt": false` is requested (`SHADOWZERO_FORCE_CONV_SNAP=1` forces snapshots).
  - **CLI switch `--no-prompt-opt`**: Disables all five optimizations at once for clean A-B-B-A benchmarking (`tools/ab_prompt_opt.py`).
  - **Measured (interleaved A-B-B-A at tier 45, floor 3.5, `bench/prompt-opt-2026-11-10/results_r2.json`, bit-identical hashes across all 5 workloads)**:
    - `3k_a` (2,867 tok, first prompt): `3.42 s → 3.12 s` (**+9.7%**, `837.9 → 919.1 tok/s`)
    - `3k_b` (2,622 tok, warm prompt): `2.72 s → 2.62 s` (**+4.0%**, `962.2 → 1,000.9 tok/s`, breaking `1,000 tok/s` at 2.6k tokens)
    - `9k6` (9,673 tok): `5.89 s → 5.73 s` (**+2.8%**, `1,642.9 → 1,689.1 tok/s`)
    - `15k` (15,034 tok): `8.70 s → 8.57 s` (**+1.4%**, `1,728.7 → 1,753.6 tok/s`, **15.5% faster than Strata 0.2.11's 9.90 s / 1,543 tok/s**)
    - `decode200` (200 tok): `53.2 → 52.9 tok/s` (unchanged, **+11% faster than Strata's 47.7 tok/s**).

## [0.1.32] — 2026-11-10
### Added
- **Avalon Zero-Cost Oracle & Router-Sync Probes (SHADOWZERO_AVALON_ORACLE=N, SHADOWZERO_AVALON_ORACLE_SYNC=1, crates/shadowzero-cuda/src/session.rs, crates/shadowzero/src/main.rs)**:
  - Added timing-only oracle hooks (--avalon-oracle N and --avalon-oracle-sync, off by default) to measure the exact upper bound of offloading N non-resident experts per layer to the CPU during staged prompt passes (docs/EXPERIMENT_SPEC_2026-11-10.json, docs/STATUS.md 92).
  - Benchmarked across 5 arms x 3 runs (ench/avalon_oracle_prompt) and 3 arms x 2 runs (ench/avalon_oracle_sync), summarized in ench/avalon-2026-11-10/summary.md: N = 3 is a tie (+0.11% to +0.65%) even for a clairvoyant 0-cost oracle, while synchronizing per layer to read router tops on the host costs -5.31% prompt throughput (-6.22% combined). Closed without building the full CPU prompt port; baseline defaults unchanged.

## [Unreleased] — 2026-11-05 and 2026-11-06
Prompt reading with `--tensor-prompt` (the default fast mode) is about 5% / 11% / 23% faster than at the start of 2026-11-06 for 3k / 9.6k / 15k-token prompts (535 / 1,068 / 1,038 tok/s, mean of four server runs each); decode and `--exact` are unchanged (exact hash 162a8e07, `quality_test` 12 of 12).
### Fixed
- Chat page / server: one long agent run (a single user message, dozens of steps) no longer fills the window with its own history: the oldest steps fold into one line each (messages, call, output numbers, the model's note), the newest stay whole - the page does it with message numbers, the server as a fallback; before, 85 steps left the reply 560 tokens ("the conversation has no room left") (2026-11-09, STATUS 79).
- Server: with thinking on, a request that sets no token limit now keeps room for the thinking budget plus a 2,048-token answer (at most a third of the window) instead of only 2,048 tokens, so a reply that thinks for ~4,000 tokens and goes on no longer runs into the end of the window ("the conversation has no room left") (2026-11-09, STATUS 74).
- Tools: `read_file` ignored the line range a model asked for under the names `start_line` / `end_line` (and `file_path`), always answering with lines 1-400, so a model reading a big file kept re-reading its top; those names are now taken as `start` / `end` / `path`, and any argument a native tool does not have is reported in its result (2026-11-09, STATUS 73).
- Server: in one long agent run (a single user message) the tool outputs of the active step are now shrunk to previews too, oldest first and the newest whole, when the prompt would otherwise leave the reply no room ("the conversation has no room left" after 66 tokens) (2026-11-09, STATUS 73).
- Server: loop guard in the fast modes: a reply that falls into repeating the same stretch while thinking gets its think block closed at once (instead of at the 4,000-token Medium budget), and one that repeats in its answer is stopped with a note "[Stopped: the reply was repeating itself.]" instead of running until the context is full; `SHADOWZERO_LOOP_GUARD=0` turns it off, `--exact` is untouched (2026-11-08, STATUS 72).
- Chat page: no limit on agent steps per message any more (it was 14, and ended the turn silently after the last tool result); the rolling context keeps the request bounded and Stop ends a run; a model that makes the same tool call 4 times in a row is stopped with a note (2026-11-08, STATUS 71).
- Server: a long prompt whose shared-prefix snapshot point fell 1-8 tokens past the cached start was read through the non-staged path in one huge chunk and looked hung ("Reading your message: 6 / 16,119", GPU at full load for minutes); the tiny snapshot split is now skipped while the VRAM-cache lend is on (2026-11-08, STATUS 70).
- CUDA: kernel modules load eagerly (`CUDA_MODULE_LOADING=EAGER` unless set): lazy loading took device memory for a kernel's code in the middle of a 20k-token prompt, where ~100 MiB of VRAM is free, and a slightly larger module made the 20k gates fail with CUDA_ERROR_OUT_OF_MEMORY (2026-11-08, STATUS 70).
- **Garbled replies in a chat that started from the stored shared-prefix snapshot** (default since 2026-11-05 14:27): a prompt-path CUDA graph recorded for one prompt was replayed over big-batch buffers that another prompt had placed elsewhere; the graph key now covers every buffer's address (`GpuSession::bb_sig`). Regression test: `tools/prefix_restore_test.py`. (docs/STATUS.md 53)
- **Tool calls opened while the model is still thinking** (no `</think>` first) now end the thinking and run; before, the turn stopped with the calls shown inside the reasoning.
- **`glob` tool**: real patterns (`*`, `**`, `?`, folders); it used to match a substring of the file name, so `**/*.html` found nothing. **`grep` tool**: regexes as its description says (`busy|telemetry`); literal text still matches.
- **"N pictures sent but the prompt has M picture slots"**: literal special-token text in a message (a fetched page that mentions `<|image_pad|>`) is defanged with a zero-width space, only for the model's real special tokens, so it no longer becomes a picture slot or a fake turn.
- Kernel-signature drift test: the parser skips `__launch_bounds__(N)`; it had been failing since 2026-11-05.
### Changed (2026-11-08, prompt path)
- Tools: `replace_lines` (path, start, end, newText, optional expect) edits a file by the line numbers `read_file` shows, so a model does not have to repeat the old text; an empty newText deletes, start one past the end appends, `expect` refuses a stale line number, line endings are kept, the answer shows the new numbers. `read_file` now returns 120 lines from `start` (it used to end at line 400 whatever `start` was) and `recall` 100 (2026-11-09, STATUS 80).
- Chat page / server: the newest tool outputs stay whole within a budget (30% of the room; the last step's always) instead of only the last step's, so one edit can use several reads; after 8 read-only calls in a row without notes or an edit the result carries a note asking for notes and an edit, and again from the 14th (2026-11-09, STATUS 80).
- Tools: `search` - find files of the project and earlier tool outputs of the session by meaning (a sentence, not exact text): the text is cut into chunks, embedded on the CPU with the repo's `llama-embedding.exe` and `all-MiniLM-L6-v2` (`models/embed-minilm/`, 25 MB; BGE-small is the fallback), stored as a quantized index on disk (`%LOCALAPPDATA%\Shadow\toolcache\semindex-*.bin`), built in the background at low priority, recent files first, and refreshed file by file; a changed file embeds only its changed chunks (2026-11-09, STATUS 78).
- Chat page: pinned notes - a `notes` tool the model uses to keep the plan, decisions and key facts of a long task (shown to it at the end of every request, never shortened, kept with the chat, editable by the user in the Notes box beside the tools button); `recall` can now read any earlier message of the conversation (`message=N`) or search all messages, and the one-line summaries of folded turns name their messages (`m3-m9`) (2026-11-09, STATUS 77).
- Tools: every long output of a native tool is kept in a per-session log on disk (`%LOCALAPPDATA%\Shadow	oolcache\s<start time>\<n>.txt`, the newest 6 sessions kept) and starts with `(output #n)`; the new `recall` tool reads one again by number (a line range, or only the lines containing a text) or searches every earlier output of the session, and the shortened preview of an old output in a long conversation names its number and says to use `recall` (2026-11-09, STATUS 76).
- Chat page / server: help the model through a long agent run in a small window: the default persona now tells it to write down the facts it will need (file, line numbers, exact text) before its next tool call because earlier tool outputs are shortened; a tool call repeated exactly gets a note in its result saying so; and a shortened tool output now says it was shortened, how it began, and that the tool can be called again (2026-11-09, STATUS 75).
- Prompt path: the grouped MoE kernels (`iq2_s`, `iq3_xxs`, `q2_0`, `iq4_nl` for the widths where it measured faster) load the next weight tile while the current one multiplies; bit-identical output, about 0.8% of a 15k prompt's kernel time (2026-11-08, STATUS 69).
- The DeltaNet recurrence of a prompt pass runs the three value heads of a key head in one thread with the inputs copied ahead (`delta_steps128_kh2`, bit-identical to the previous kernel, 1.08x standalone; `SHADOWZERO_DELTA_KH=0` keeps the previous one; docs/STATUS.md 67).
- A long prompt's embedding rows go up in groups of 512 (new kernel `broadcast_rows`; `SHADOWZERO_EMBED_ROWS=0` keeps the per-token loop), and the layer-ahead pipeline's three pinned read-ahead buffers (1.33 GiB) are kept between prompts while RAM is plentiful (`SHADOWZERO_KEEP_PIPE=0` frees them as before): about 0.45 s less per prompt from the second one on, +6% at 9.6k and 15k in A-B-B-A, output bit-identical (docs/STATUS.md 66).

### Fixed (2026-11-08)
- The grouped embedding upload (earlier the same day) allocated a 5 MB device scratch at the start of a pass and ran out of VRAM on a ~20k-token prompt (HTTP 400, session reset); it now uses a buffer the pass already owns (docs/STATUS.md 67).
- The width-class MoE planner (2026-11-07) failed with "internal error: width-class plan: ... bytes of tables" on a small batch buffer (768 rows) and reset the conversation's cache; its table buffer is now sized for its fixed part, and a pass whose tables do not fit takes the fixed-width planner instead of failing (docs/STATUS.md 62).

### Changed (2026-11-07, prompt path)
- The MoE products of a long prompt run in groups of 16 / 32 / 64 tokens matched to each expert's count (device planner `plan_fill_w`, `SHADOWZERO_MMA_BUCKETS=0` off): the MoE kernel time -17%, a 9.6k prompt's kernel time -9%, 15k prompts +15% and 9.6k +3.6% in A-B-B-A; output bit-identical (docs/STATUS.md 60-61).
- Prompts of 512-1,023 tokens now use the single big pass (`LONG_PROMPT` 1,024 -> 512, `SHADOWZERO_LONG_PROMPT=n`): a 1,000-token prompt reads in ~4.1 s instead of ~7.7 s, 700 tokens in 4.0 s instead of 5.3 s; below 512 nothing changes.
- CUDA graphs for the prompt path are off by default (`SHADOWZERO_BATCH_GRAPHS=1` on): 15k-token prompts +10% (1,292 against 1,174 tok/s, A-B-B-A), 0.5k-9.6k ties.
- `SHADOWZERO_PHASES=1` prints the wall-clock phases of a prompt (rows, layers, lend, restore, snapshot) on stderr.

### Removed
- **Flint** (`RecursiveSI/flint`, NVFP4) is no longer part of this build: its CPU and GPU forward paths, chat template, pre-tokenizer, NVFP4 kernels, tests and fixtures. A Flint file is refused with a clear message; `shadowzero inspect` still reads it. It stays in git history (tag `pre-flint-removal`) and will be tested on tandem.cpp later (docs/STATUS.md 57).
### Changed
- Prompt path (fast mode only): 15k-token prompts run in one pass (sub-chunk size fitted to the prompt, `SHADOWZERO_ONE_PASS=0` off), DeltaNet recurrence on 512 threads (bit-identical, `SHADOWZERO_DELTA_F4=0` off), sparse-attention scores on tensor cores with fp16 inputs (`SHADOWZERO_ATT_TC=0` off), int8 MMQ dense projections when `native/mmq/shadowzero_mmq.dll` is built (`SHADOWZERO_MMQ=0` off), float norm/DeltaNet/rope kernels, batched per-layer-embedding step. STATUS entries 31-52.
- Web UI: thin, quiet scroll bars.
- Desktop launcher (`shadowzero-ui.exe`): closing the window now quits it and stops the server it started (its 36+ GiB of pinned RAM comes back); it used to hide to the tray. `"closeToTray": true` in `%APPDATA%\Shadow\desktop.json` brings the old behaviour back.
### Added
- Start Menu entry "ShadowZero" (the desktop launcher `shadowzero-ui.exe`, which starts the server and opens the chat).
- Tools: `tools/needle_test.py` (long-prompt retrieval), `tools/replay_snapshot.py` (replay a stored conversation), `tools/prefix_restore_test.py`, `tools/nsys_last_run.py`, `tools/nsys_gap_why.py`.
- Opt-in experiments, all measured ties or losses (docs/STATUS.md 36, 46-48, 51, 52): `SHADOWZERO_MMQ_MOE=1` (gate/up experts on int8 kernels), `SHADOWZERO_GU_SWIGLU=1` (gate+up+SwiGLU in one kernel), `SHADOWZERO_MMA_GROUP=48|64`, `SHADOWZERO_ATT_TC=2` (tensor-core weighted sum), `SHADOWZERO_ATT_TCD=1` (dense attention scores on tensor cores).

## [0.1.31] — 2026-11-04
### Added
- **Opt-in Direct I/O Read Pool (SHADOWZERO_DIRECT_IO=1, crates/shadowzero-model/src/readpool.rs)**:
  - Implemented Windows unbuffered overlapped direct-I/O (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) via a preallocated pool of 4KB-aligned staging slots and heap-pinned OVERLAPPED events (win::GetOverlappedResult).
  - Added CLI flags --direct-io and --no-direct-io across serve, generate, and kv.
  - Added unit test direct_io_matches_buffered_on_real_expert_slices in crates/shadowzero-model/tests/real_model.rs verifying bit-identical reads against authentic model tensor slices.
  - Benchmarked across prompt-heavy and mixed regimes per docs/EXPERIMENT_SPEC_2026-11-04b.json: confirmed that bypassing the OS file cache costs -8.7% on mixed and -20% to -42% on prompt-heavy. Per project rules, direct I/O remains opt-in and off by default.

## [0.1.30] — 2026-11-03
### Changed
- **Establish Winning Speculation & Engine Settings as ShadowZero Defaults (`crates/shadowzero/src/main.rs`)**:
  - Validated by the 4-arm prompt-side speculation benchmark (`docs/EXPERIMENT_SPEC_2026-11-04.json`), which proved +49.2% decode speedup (34.56 -> 51.55 tok/s), +27.9% combined total throughput gain, and zero regression on prompt ingestion (+0.84% on massive prompts).
  - Defaults locked in via `fastest_defaults`:
    - Speculative decoding: `--spec 3` with DFlash 2 drafter (`models/dflash2`) and selector threshold `--dflash-min-p 0.5`.
    - Routing placement profile: `profiles/routing-varied300.shrc` (34,990 tokens of empirical varied routing).
    - Tiering: `--ram-tier-gib 36` (pinned DRAM tier) with guaranteed default `--ram-floor-gib 3` (protecting against watchdog false-positives under memory pressure).
    - Verification & prompt acceleration: `--verify-rank 2`, `--tail-skip 3`, and tensor-core dense prompt projections (`--tensor-prompt`).
  - Added multi-path resolution in `found()` (`current_exe` ancestors and manifest root) so models, drafters, and profiles resolve regardless of launch working directory.
  - Wired `generate` to use `fastest_defaults` as well, bringing identical performance to CLI generation.

## [0.1.29] — 2026-11-03
### Fixed
- **Allow Scrolling Up During Thinking & Streaming (`crates/shadowzero-serve/src/ui.rs`)**:
  - Replaced unconditional `messagesEl.scrollTop = messagesEl.scrollHeight` on every streamed reasoning/content token with smart `autoScroll` tracking (`scrollToBottom()`), which immediately pauses auto-scrolling when you wheel up or drag the scrollbar away from the bottom (`dist >= 48px`) and resumes when you scroll back to the bottom or send a new message.
  - Replaced `scroll-behavior: smooth` with `overflow-anchor: none` on `#messages` so high-frequency `scrollTop` updates (~45–52 tok/s) do not restart overlapping smooth-scroll animations that fought user scroll input.

## [0.1.28] — 2026-11-03
### Added
- **Unlimited Rolling Context Window (`0 B` extra RAM/VRAM)**:
  - Server-side `fit_context()` in `crates/shadowzero-serve/src/engine.rs` guarantees multi-turn and agent conversations never fail with `"exceed the context of ... positions: the conversation is full"`. Leading `Role::System` messages are kept untouched so `prefix.snap` (system prompt + tool schemas) always hits in ~15 ms, while older tool outputs are compacted to 1-line digests, older turn groups fold into a bounded `[Earlier conversation summary]` block, and oversized single messages are middle-truncated to leave guaranteed reply headroom inside the fixed GPU `n_ctx` arena.
  - Non-destructive client-side `buildContextMessages()` in `crates/shadowzero-serve/src/ui.rs` keeps the full unabridged `history` in the UI and `localStorage` scrollback while sending a rolling summary + recent turns to `/v1/chat/completions`.
### Fixed
- **Strip Raw `<tool_call>` XML from Chat Bubbles (`crates/shadowzero-serve/src/ui.rs`)**:
  - Added `stripToolCalls()`, `hasPendingToolCall()`, and `pendingToolCallName()` so `<tool_call><function=...>` XML blocks are stripped from assistant prose both during live streaming (replaced with a clean `⚡ Preparing tool call...` status indicator) and after completion.
  - Updated `renderSaved()` when reopening saved conversations from the sidebar to reconstruct collapsible `.tool-call-block` cards paired with their corresponding `role: 'tool'` outputs instead of dumping raw `<tool_call>` XML into the chat bubble, and skipped empty aborted assistant turns.

## [Unreleased]
### Added
- `/v1/events`: a server-sent-events stream pushing telemetry once a second and status changes, so the web UI's header gauges update by push instead of polling (`EventSource`, with the 1 s poll kept as fallback).
- Server-side chat sessions (`/v1/sessions` CRUD + `/v1/sessions/import`): the sidebar list and every conversation now live on the server instead of the browser's localStorage, so chats survive a browser change and are readable by other clients; the first load migrates any old localStorage chats onto the server automatically.
- **Diff/edit card in tool-call blocks (`crates/shadowzero-serve/src/ui.rs`)**: `edit_file` / `replace_lines` / `write_file` tool cards now render a side-by-side-free unified diff (`diffLines` prefix/suffix-trim with 2-line context, "N unchanged" skip rows, 140-row cap) instead of the raw JSON arguments, with a `+n −n` stat and `diff` tag in the collapsed card header; `replace_lines` shows its line range and `expect` guard, `write_file` shows added-only lines; replayed history from `/v1/sessions` rebuilds the same diff cards.
### Changed
- Tensor-core prompt kernels (`--tensor-prompt`, default mode): the Q6_K / IQ4_XS projections and the grouped experts run on `mma.sync` FP16 tiles (3e-4 relative error, 7x faster kernels); one batched copy call per layer: prompts of 1,024+ tokens read 185-225 tok/s with the default tier in 2,048-token batches (was 130-145).
- Prompt reading of 1,024+ tokens: a layer-ahead pipeline (next layer's experts copied and its file reads started while this one computes) and 1,024-token batches from buffers carved out of the lent cache: 77-87 -> 128-144 tok/s, `--exact` 40 -> 62; decode unchanged.
- **Fix:** ending a prompt's cache lending could leave an expert owned by two cache slots (loaded again during the prompt, then restored into its old slot), which crashed a later chat turn with "a routed expert neither cached nor fetched" (HTTP 500) - seen at turn 14 of a 24-turn history with 250-token continuations.
- `shadowzero serve` defaults to `--verify-rank 2` instead of `--accept-moe`: decode 47.96 against 45.11 tok/s (varied bench), replies 10.6% shorter, same-history fidelity 0.423 (AcceptMoE 0.415, Strata 0.431). `--accept-moe` restores the old default.
- Attention kernel 2.7x faster (shared-memory q, 16-byte K loads, staged score tiles; same arithmetic order, bit-identical) and the indexer's scores in row launches: prompts of 1.8k-4.6k tokens read 4-28% faster.
- Long prompts: past the indexer budget (2,051 positions) attention selects each token's positions SEL_ROWS at a time and runs them in one launch per 8 tokens, instead of one synchronous round trip per token and layer; bit-identical, prompt reading +16% on a 4.5k-token prompt.
- `shadowzero serve` defaults to a 36 GiB RAM tier when 49 GiB of RAM is available at start (30 GiB otherwise): +2.5% decode, ~+10% prompt reading; `--ram-floor-gib N` makes the 4 GiB RAM floor a flag.
- The default tail skip is **3** (was 4): 39.4 tok/s by total throughput on varied text against 37.0, same reply length, same-history fidelity 0.415 against
  0.572 (Strata measured 0.431 the same way). `--tail-skip 4` restores the earlier mode. Added `--admit-after N` (VRAM-cache admission filter, opt-in, about +1%) and the
  `SHADOWZERO_VRAM_MAP=1` startup probe (docs/STATUS.md has the cache, VRAM and bus measurements behind both).

## [0.1.27] — 2026-10-30
### Changed (default on; `--exact` turns the speculation and inexact modes off)
- `shadowzero serve MODEL` runs the fastest mode that keeps answers intact with no tier flags: `--tail-skip 4 --accept-moe
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
- `shadowzero-bench`: records the machine once a second per run (CPU, RAM, disk, GPU util/memory/power/temperature/clocks, PCIe) and
  each turn's start/end, `--prompts-file`, `--no-resources`; `tools/plot_bench.py` (charts, tables, joules per token),
  `tools/bench_total.py` (total-throughput scoring), `tools/autotune.py` (coordinate ascent over flags),
  `tools/bench_vs_strata.ps1`.
- `SHADOWZERO_RECORD_ROUTING=FILE` (a routing profile from a running server), `SHADOWZERO_NO_PREFIX=1`, `SHADOWZERO_DF_DUMP=DIR`, `SHADOWZERO_DF_TIME=1`.
- AcceptMoE verifier expert sets (`--accept-moe`, `--accept-beta/--accept-scale/--accept-budget`), the graded `--verify-rank R`,
  `finish` recorded per bench turn, tokens-a-run and early stops in `bench_total.py`, `tools/quality_test.py`, `tools/trunc_test.py`.
- The n-gram (PLE) row prefetch reads through the model's read pool (own file handles: one shared synchronous handle serializes reads on Windows): prefetch
  3.5-9 ms -> 1.2 ms a verify pass, +7% in the default mode, output unchanged (34.7 -> 37.1 tok/s on the varied bench).
- Opt-in experiments, all exact and none a gain: `--pipeline` (two token groups per verify pass), `--cpu-experts N` in the verify pass, `--verify-rank R`.
### Learned (docs/STATUS.md)
- The bench's per-turn median misranks arms that generate different text; score by total throughput.
- A drafter fine-tune that looked like +25% accepted tokens counted prompt positions; on generated text it was a tie.
- Candidates over a 106k draft vocabulary: -25%. Longer blocks than 3 drafts: slower.

## [0.1.26] — 2026-10-28
### Changed (default on, exact)
- CUDA graphs for each DeltaNet layer's launches from its hc read to its router (`SHADOWZERO_GRAPHS=0`: off):
  `bench/graphs-2026-10-28` **+2.1% paired** decode.
- Overlapped expert copies: resident experts run while the misses copy, the GPU (not the host) waits for the copies,
  `moe_combine_qwen_perm` keeps the sum order (`SHADOWZERO_OVERLAP=0`: off): `bench/overlap-2026-10-28` **+2.7% paired**.
  Together 18.65 -> 19.47 tok/s `decode_warm`, 0 answer differences.
- The VRAM expert cache is LRU over every slot of a layer; the placement's hottest experts are no longer pinned (and
  so stay in the RAM tier). `tools/cache_sim.py` predicted 59 -> 75% hits; measured 56.6 -> 70.7%, `bench/lru-2026-10-28`
  **+11.1% paired** (19.53 -> 21.06 tok/s). `--pin-margin N` brings back pinning with N LRU slots a layer.
### Added
- Kernel launch counts and host time in launches in the session report; `SHADOWZERO_CACHE_DUMP=FILE` (per-layer slots and
  pinned experts, for `tools/cache_sim.py`).

## [0.1.25] — 2026-10-28
### Added (default on, exact)
- **Batched step**: `GpuSession::step_batch` / `prefill_batch`, up to 64 tokens through the layers together (Qwen):
  shared projections through `matvec_cols` (new `matvec_cols_ref_bf16/f32` and `hc_down_inject_cols_bf16` keep the
  BF16/F32 sums in the one-token order), the union of the tokens' experts fetched once (overflow into a 128-slot VRAM
  scratch arena). Logits bit-identical to `step` (`tests/reference.rs`, also past the sparse indexer). Used to read
  prompts in `shadowzero serve` and `shadowzero generate`: **0.062 -> 0.035 s/token on a 1,262-token prompt, -46.0% paired,
  all 8 pairs** (`bench/prefill-2026-10-28`); decode a tie, 0 answer differences (`bench/batch-2026-10-28`).
  `SHADOWZERO_BATCH=n`, `SHADOWZERO_BATCH_SCRATCH=n` for comparisons. `shadowzero generate @FILE` reads the prompt from a file.
### Added (opt-in, measured, slower)
- `shadowzero serve --spec K`: speculative decoding with prompt-lookup drafts verified by `step_batch`
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
  (`shadowzero-model::cpu_expert`, AVX2 kernels for every routed-expert type). Slower at every N tried.
- `--pregate K`: pre-gated prefetch of the next layer's likeliest experts. A tie.
- `matvec_cols_*` kernels with `dotm<M>`: up to 8 inputs per weight read, bit-identical to `matvec_fast_*`; the first
  piece of batched prefill / speculative verification.
- Probes: `SHADOWZERO_ROUTE_LOG`, `SHADOWZERO_MTP_PROBE`, `SHADOWZERO_PREGATE_PROBE`, `SHADOWZERO_TIMING_NO_COPIES`,
  `tools/spec_probe.py`; host wait timers in the session report; `shadowzero compare` takes tier flags.

## [0.1.23] — 2026-10-27
### Changed
- **Parallel drive-tier expert reads (`shadowzero-model::readpool`)**: a layer's experts that miss VRAM and the pinned RAM
  tier are no longer read one at a time on `Model`'s single file handle. `GpuSession::fetch_expert` queues them, and
  `read_drive` reads them all at once in 256 KiB pieces on 8 persistent workers, each with its own handles (Windows
  serializes I/O on one synchronous handle). Reads stay buffered, so repeats still come from the file cache.
  8 interleaved launches per arm (`bench/readpool-2026-10-27`, `--ctx 8192 --ram-tier-gib 30 --placement
  profiles/routing-6prompts.shrc`): `decode_warm` median **17.37 -> 18.52 tok/s**, paired **+5.3%** median, faster in
  all 40 pairs (worst +2.9%), 0 answer differences. Reference tests and the 48-layer compare are unchanged (cosine 1.000).
- `shadowzero-bench run --arm-exe NAME=EXE` runs an arm on another build, for A/B tests of two binaries.

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
- **MTP (Multi-Token Prediction) Speculative Decoding Engine (`crates/shadowzero-cuda/src/mtp.rs`)**:
  - Wired ShadowZero's GPU forward pass to load and execute the tuned MTP draft head from `models/mtp-basalt-ft1` (`dense.bin`, `dense.txt`, `experts.bin`, `draft_vocab.bin`).
  - Implemented the full 1-layer MTP draft transformer block: input projections (`fc_embedding`, `fc_hidden`), attention hyper-connection, QSA self-attention, MLP hyper-connection, MoE with 512 resident routed experts (unpacked to contiguous Q2_0 rows) + shared expert, and final mixer.
  - Added native `Q8_0` quantization format to CUDA kernel registry (`kernels/quants.cuh`, `kernels/dequant.cu`, `kernels/matvec.cu`, and `shadowzero-cuda/src/kernels.rs`), supporting `dequant_q8_0`, `matvec_q8_0`, `matvec_fast_q8_0`, `matvec_batch_q8_0`, and `matvec_pair_q8_0`.
  - Added speculative decode loop in `crates/shadowzero-serve/src/engine.rs`: proposes candidate tokens via MTP draft head and verifies them against the base model's greedy argmax in batch. Accepted drafts (~86% acceptance rate) commit multiple tokens per base model step, accelerating generation throughput from ~14 tok/s to >30 tok/s with guaranteed bit-for-bit output parity.
  - Added `--mtp DIR|off` CLI support and live telemetry indicator in Web UI Process Monitor (`engine-mtp-txt`).

## [0.1.19] — 2026-10-27
### Added
- **Chat Stop Button (`⏹ Stop`)**: Added responsive generation cancellation to the lower chat input bar. While ShadowZero streams tokens, the Send button switches to an interactive red Stop button. Clicking Stop instantly cancels the client `AbortController` and signals the server; `shadowzero-serve` drops the stream receiver and halts GPU forward-pass decoding immediately, preserving all partially generated content.

## [0.1.18] — 2026-10-27
### Fixed
- **Process Monitor Dashboard DOM Null Dereference**: Resolved a silent JavaScript exception in `pollTelemetry()` where a query for `cpu-sys-val` returned `null`, halting execution before the GPU compute, RAM, Engine, and MoE Tier dashboard cards could be populated. Replaced with null-safe setters (`setTxt`, `setHTML`, `setWidth`) ensuring all telemetry cards update continuously and reliably.

## [0.1.17] — 2026-10-27
### Fixed
- **View Panel Layout Bug in Web UI**: Fixed CSS specificity bug where `#monitor-panel { display: flex; }` caused the Process Monitor to render side-by-side on the right of the Chat view. `#chat-panel` and `#monitor-panel` now strictly respect tab selection, rendering full-width independently.
- **Accurate CPU Usage Calculation**: Corrected `SysMonitor` calculation in `crates/shadowzero-serve/src/telemetry.rs`. Removed duplicate logical core multiplication that caused process CPU to report 100% saturation during single-threaded generation. Process CPU now accurately represents overall system capacity share alongside live active core equivalents.
### Added
- **Real NVML GPU Compute Utilization Monitor**: Dynamically binds `nvml.dll` (`nvmlDeviceGetUtilizationRates` and `nvmlDeviceGetTemperature`) to report real-time GPU compute core execution percentage (0–100%), memory controller load %, and GPU temperature in °C.
- **Native Windows Agent Tool Orchestration (`crates/shadowzero-serve/src/orchestrator.rs`)**:
  - Implemented local Windows agent tools: `read_file`, `glob`, `grep`, `edit_file`, `write_file`, and `cmd` (PowerShell).
  - Added HTTP endpoints `GET /v1/tools` and `POST /v1/tools/execute`.
  - Added native multi-turn autonomous tool execution loop in Web UI with visual collapsible execution cards, allowing ShadowZero to inspect previous sessions (`J:\Strata-work\sessions`), run tests, and perform code edits directly on Windows.

## [0.1.16] — 2026-10-27
### Added
- **Claude & Antigravity Style Thinking Level Selector**: Added interactive dropdown popup directly above the thinking button matching Claude and Antigravity's UI styling (`Off`, `Low`, `Medium`, `High`, `X-High`) with icons, sublabels, descriptions, and active checkmarks, passing `reasoning_effort` to the backend and persisting user selection in `localStorage`.
- **System Persona / Local Dev Authority Pill & Modal**: Built-in system prompt controller granting ShadowZero sovereign local authority to create, inspect, and modify code and project files directly without safety refusal hesitation.
- **Hardware Telemetry Engine (`GET /v1/telemetry`)**: Win32-grounded system monitor tracking process CPU %, system CPU %, logical core counts, exact GPU VRAM free/used/total, system RAM load, working set, pagefile/commit, and MoE 3-tier cache hit/miss statistics.
- **Process Monitor Dashboard (`/monitor` & Tab Navigation)**: Real-time telemetry dashboard with live visual progress bars, detailed memory breakdown, MoE expert cache hit rates, engine decode speed (`tok/s`), request counters, and a visual MoE 3-tier memory pipeline (VRAM -> Pinned RAM -> NVMe).

## [0.1.15] — 2026-10-27
### Added
- **`shadowzero-ui.exe` Native Desktop Launcher (`crates/shadowzero-ui`)**: Native Windows desktop launcher built with Rust and Tauri 2.0 (using system WebView2). Manages connection to `http://127.0.0.1:8081`, displays a dark-mode status and loading splash screen while the model initializes, supports single-instance window focusing, maintains a system tray icon with Show/Reconnect/Quit controls, and stores configuration in `%APPDATA%\Shadow\desktop.json`.

## [0.1.14] — 2026-10-27
### Added
- **Embedded Web UI**: Built-in dark-mode chat interface served directly by `shadowzero serve` at `GET /` and `/index.html` with streaming SSE, `<think>` reasoning card toggling, live performance telemetry (tok/s, status, memory), and copyable markdown code blocks without external web dependencies.
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
