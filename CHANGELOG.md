# Changelog

All notable changes to Shadow will be documented in this file.

## [Unreleased] — 2026-11-05 and 2026-11-06
Prompt reading with `--tensor-prompt` (the default fast mode) is about 5% / 11% / 23% faster than at the start of 2026-11-06 for 3k / 9.6k / 15k-token prompts (535 / 1,068 / 1,038 tok/s, mean of four server runs each); decode and `--exact` are unchanged (exact hash 162a8e07, `quality_test` 12 of 12).
### Fixed
- Chat page / server: one long agent run (a single user message, dozens of steps) no longer fills the window with its own history: the oldest steps fold into one line each (messages, call, output numbers, the model's note), the newest stay whole - the page does it with message numbers, the server as a fallback; before, 85 steps left the reply 560 tokens ("the conversation has no room left") (2026-11-09, STATUS 79).
- Server: with thinking on, a request that sets no token limit now keeps room for the thinking budget plus a 2,048-token answer (at most a third of the window) instead of only 2,048 tokens, so a reply that thinks for ~4,000 tokens and goes on no longer runs into the end of the window ("the conversation has no room left") (2026-11-09, STATUS 74).
- Tools: `read_file` ignored the line range a model asked for under the names `start_line` / `end_line` (and `file_path`), always answering with lines 1-400, so a model reading a big file kept re-reading its top; those names are now taken as `start` / `end` / `path`, and any argument a native tool does not have is reported in its result (2026-11-09, STATUS 73).
- Server: in one long agent run (a single user message) the tool outputs of the active step are now shrunk to previews too, oldest first and the newest whole, when the prompt would otherwise leave the reply no room ("the conversation has no room left" after 66 tokens) (2026-11-09, STATUS 73).
- Server: loop guard in the fast modes: a reply that falls into repeating the same stretch while thinking gets its think block closed at once (instead of at the 4,000-token Medium budget), and one that repeats in its answer is stopped with a note "[Stopped: the reply was repeating itself.]" instead of running until the context is full; `SHADOW_LOOP_GUARD=0` turns it off, `--exact` is untouched (2026-11-08, STATUS 72).
- Chat page: no limit on agent steps per message any more (it was 14, and ended the turn silently after the last tool result); the rolling context keeps the request bounded and Stop ends a run; a model that makes the same tool call 4 times in a row is stopped with a note (2026-11-08, STATUS 71).
- Server: a long prompt whose shared-prefix snapshot point fell 1-8 tokens past the cached start was read through the non-staged path in one huge chunk and looked hung ("Reading your message: 6 / 16,119", GPU at full load for minutes); the tiny snapshot split is now skipped while the VRAM-cache lend is on (2026-11-08, STATUS 70).
- CUDA: kernel modules load eagerly (`CUDA_MODULE_LOADING=EAGER` unless set): lazy loading took device memory for a kernel's code in the middle of a 20k-token prompt, where ~100 MiB of VRAM is free, and a slightly larger module made the 20k gates fail with CUDA_ERROR_OUT_OF_MEMORY (2026-11-08, STATUS 70).
- **Garbled replies in a chat that started from the stored shared-prefix snapshot** (default since 2026-11-05 14:27): a prompt-path CUDA graph recorded for one prompt was replayed over big-batch buffers that another prompt had placed elsewhere; the graph key now covers every buffer's address (`GpuSession::bb_sig`). Regression test: `tools/prefix_restore_test.py`. (docs/STATUS.md 53)
- **Tool calls opened while the model is still thinking** (no `</think>` first) now end the thinking and run; before, the turn stopped with the calls shown inside the reasoning.
- **`glob` tool**: real patterns (`*`, `**`, `?`, folders); it used to match a substring of the file name, so `**/*.html` found nothing. **`grep` tool**: regexes as its description says (`busy|telemetry`); literal text still matches.
- **"N pictures sent but the prompt has M picture slots"**: literal special-token text in a message (a fetched page that mentions `<|image_pad|>`) is defanged with a zero-width space, only for the model's real special tokens, so it no longer becomes a picture slot or a fake turn.
- Kernel-signature drift test: the parser skips `__launch_bounds__(N)`; it had been failing since 2026-11-05.
### Changed (2026-11-08, prompt path)
- Tools: `search` - find files of the project and earlier tool outputs of the session by meaning (a sentence, not exact text): the text is cut into chunks, embedded on the CPU with the repo's `llama-embedding.exe` and `all-MiniLM-L6-v2` (`models/embed-minilm/`, 25 MB; BGE-small is the fallback), stored as a quantized index on disk (`%LOCALAPPDATA%\Shadow\toolcache\semindex-*.bin`), built in the background at low priority, recent files first, and refreshed file by file; a changed file embeds only its changed chunks (2026-11-09, STATUS 78).
- Chat page: pinned notes - a `notes` tool the model uses to keep the plan, decisions and key facts of a long task (shown to it at the end of every request, never shortened, kept with the chat, editable by the user in the Notes box beside the tools button); `recall` can now read any earlier message of the conversation (`message=N`) or search all messages, and the one-line summaries of folded turns name their messages (`m3-m9`) (2026-11-09, STATUS 77).
- Tools: every long output of a native tool is kept in a per-session log on disk (`%LOCALAPPDATA%\Shadow	oolcache\s<start time>\<n>.txt`, the newest 6 sessions kept) and starts with `(output #n)`; the new `recall` tool reads one again by number (a line range, or only the lines containing a text) or searches every earlier output of the session, and the shortened preview of an old output in a long conversation names its number and says to use `recall` (2026-11-09, STATUS 76).
- Chat page / server: help the model through a long agent run in a small window: the default persona now tells it to write down the facts it will need (file, line numbers, exact text) before its next tool call because earlier tool outputs are shortened; a tool call repeated exactly gets a note in its result saying so; and a shortened tool output now says it was shortened, how it began, and that the tool can be called again (2026-11-09, STATUS 75).
- Prompt path: the grouped MoE kernels (`iq2_s`, `iq3_xxs`, `q2_0`, `iq4_nl` for the widths where it measured faster) load the next weight tile while the current one multiplies; bit-identical output, about 0.8% of a 15k prompt's kernel time (2026-11-08, STATUS 69).
- The DeltaNet recurrence of a prompt pass runs the three value heads of a key head in one thread with the inputs copied ahead (`delta_steps128_kh2`, bit-identical to the previous kernel, 1.08x standalone; `SHADOW_DELTA_KH=0` keeps the previous one; docs/STATUS.md 67).
- A long prompt's embedding rows go up in groups of 512 (new kernel `broadcast_rows`; `SHADOW_EMBED_ROWS=0` keeps the per-token loop), and the layer-ahead pipeline's three pinned read-ahead buffers (1.33 GiB) are kept between prompts while RAM is plentiful (`SHADOW_KEEP_PIPE=0` frees them as before): about 0.45 s less per prompt from the second one on, +6% at 9.6k and 15k in A-B-B-A, output bit-identical (docs/STATUS.md 66).

### Fixed (2026-11-08)
- The grouped embedding upload (earlier the same day) allocated a 5 MB device scratch at the start of a pass and ran out of VRAM on a ~20k-token prompt (HTTP 400, session reset); it now uses a buffer the pass already owns (docs/STATUS.md 67).
- The width-class MoE planner (2026-11-07) failed with "internal error: width-class plan: ... bytes of tables" on a small batch buffer (768 rows) and reset the conversation's cache; its table buffer is now sized for its fixed part, and a pass whose tables do not fit takes the fixed-width planner instead of failing (docs/STATUS.md 62).

### Changed (2026-11-07, prompt path)
- The MoE products of a long prompt run in groups of 16 / 32 / 64 tokens matched to each expert's count (device planner `plan_fill_w`, `SHADOW_MMA_BUCKETS=0` off): the MoE kernel time -17%, a 9.6k prompt's kernel time -9%, 15k prompts +15% and 9.6k +3.6% in A-B-B-A; output bit-identical (docs/STATUS.md 60-61).
- Prompts of 512-1,023 tokens now use the single big pass (`LONG_PROMPT` 1,024 -> 512, `SHADOW_LONG_PROMPT=n`): a 1,000-token prompt reads in ~4.1 s instead of ~7.7 s, 700 tokens in 4.0 s instead of 5.3 s; below 512 nothing changes.
- CUDA graphs for the prompt path are off by default (`SHADOW_BATCH_GRAPHS=1` on): 15k-token prompts +10% (1,292 against 1,174 tok/s, A-B-B-A), 0.5k-9.6k ties.
- `SHADOW_PHASES=1` prints the wall-clock phases of a prompt (rows, layers, lend, restore, snapshot) on stderr.

### Removed
- **Flint** (`RecursiveSI/flint`, NVFP4) is no longer part of this build: its CPU and GPU forward paths, chat template, pre-tokenizer, NVFP4 kernels, tests and fixtures. A Flint file is refused with a clear message; `shadow inspect` still reads it. It stays in git history (tag `pre-flint-removal`) and will be tested on tandem.cpp later (docs/STATUS.md 57).
### Changed
- Prompt path (fast mode only): 15k-token prompts run in one pass (sub-chunk size fitted to the prompt, `SHADOW_ONE_PASS=0` off), DeltaNet recurrence on 512 threads (bit-identical, `SHADOW_DELTA_F4=0` off), sparse-attention scores on tensor cores with fp16 inputs (`SHADOW_ATT_TC=0` off), int8 MMQ dense projections when `native/mmq/shadow_mmq.dll` is built (`SHADOW_MMQ=0` off), float norm/DeltaNet/rope kernels, batched per-layer-embedding step. STATUS entries 31-52.
- Web UI: thin, quiet scroll bars.
- Desktop launcher (`shadow-ui.exe`): closing the window now quits it and stops the server it started (its 36+ GiB of pinned RAM comes back); it used to hide to the tray. `"closeToTray": true` in `%APPDATA%\Shadow\desktop.json` brings the old behaviour back.
### Added
- Start Menu entry "Shadow" (the desktop launcher `shadow-ui.exe`, which starts the server and opens the chat).
- Tools: `tools/needle_test.py` (long-prompt retrieval), `tools/replay_snapshot.py` (replay a stored conversation), `tools/prefix_restore_test.py`, `tools/nsys_last_run.py`, `tools/nsys_gap_why.py`.
- Opt-in experiments, all measured ties or losses (docs/STATUS.md 36, 46-48, 51, 52): `SHADOW_MMQ_MOE=1` (gate/up experts on int8 kernels), `SHADOW_GU_SWIGLU=1` (gate+up+SwiGLU in one kernel), `SHADOW_MMA_GROUP=48|64`, `SHADOW_ATT_TC=2` (tensor-core weighted sum), `SHADOW_ATT_TCD=1` (dense attention scores on tensor cores).

## [0.1.31] — 2026-11-04
### Added
- **Opt-in Direct I/O Read Pool (SHADOW_DIRECT_IO=1, crates/shadow-model/src/readpool.rs)**:
  - Implemented Windows unbuffered overlapped direct-I/O (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) via a preallocated pool of 4KB-aligned staging slots and heap-pinned OVERLAPPED events (win::GetOverlappedResult).
  - Added CLI flags --direct-io and --no-direct-io across serve, generate, and kv.
  - Added unit test direct_io_matches_buffered_on_real_expert_slices in crates/shadow-model/tests/real_model.rs verifying bit-identical reads against authentic model tensor slices.
  - Benchmarked across prompt-heavy and mixed regimes per docs/EXPERIMENT_SPEC_2026-11-04b.json: confirmed that bypassing the OS file cache costs -8.7% on mixed and -20% to -42% on prompt-heavy. Per project rules, direct I/O remains opt-in and off by default.

## [0.1.30] — 2026-11-03
### Changed
- **Establish Winning Speculation & Engine Settings as Shadow Defaults (`crates/shadow/src/main.rs`)**:
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
- **Allow Scrolling Up During Thinking & Streaming (`crates/shadow-serve/src/ui.rs`)**:
  - Replaced unconditional `messagesEl.scrollTop = messagesEl.scrollHeight` on every streamed reasoning/content token with smart `autoScroll` tracking (`scrollToBottom()`), which immediately pauses auto-scrolling when you wheel up or drag the scrollbar away from the bottom (`dist >= 48px`) and resumes when you scroll back to the bottom or send a new message.
  - Replaced `scroll-behavior: smooth` with `overflow-anchor: none` on `#messages` so high-frequency `scrollTop` updates (~45–52 tok/s) do not restart overlapping smooth-scroll animations that fought user scroll input.

## [0.1.28] — 2026-11-03
### Added
- **Unlimited Rolling Context Window (`0 B` extra RAM/VRAM)**:
  - Server-side `fit_context()` in `crates/shadow-serve/src/engine.rs` guarantees multi-turn and agent conversations never fail with `"exceed the context of ... positions: the conversation is full"`. Leading `Role::System` messages are kept untouched so `prefix.snap` (system prompt + tool schemas) always hits in ~15 ms, while older tool outputs are compacted to 1-line digests, older turn groups fold into a bounded `[Earlier conversation summary]` block, and oversized single messages are middle-truncated to leave guaranteed reply headroom inside the fixed GPU `n_ctx` arena.
  - Non-destructive client-side `buildContextMessages()` in `crates/shadow-serve/src/ui.rs` keeps the full unabridged `history` in the UI and `localStorage` scrollback while sending a rolling summary + recent turns to `/v1/chat/completions`.
### Fixed
- **Strip Raw `<tool_call>` XML from Chat Bubbles (`crates/shadow-serve/src/ui.rs`)**:
  - Added `stripToolCalls()`, `hasPendingToolCall()`, and `pendingToolCallName()` so `<tool_call><function=...>` XML blocks are stripped from assistant prose both during live streaming (replaced with a clean `⚡ Preparing tool call...` status indicator) and after completion.
  - Updated `renderSaved()` when reopening saved conversations from the sidebar to reconstruct collapsible `.tool-call-block` cards paired with their corresponding `role: 'tool'` outputs instead of dumping raw `<tool_call>` XML into the chat bubble, and skipped empty aborted assistant turns.

## [Unreleased]
### Added
- Long prompts (128+ unread tokens) borrow the tail of the VRAM expert cache as staging and are read in 256-token chunks; the cache is restored from the RAM tier afterwards.
  Prompt reading 51 -> 78-86 tok/s in the default mode (2.3x in `--exact`), output unchanged, decode unaffected. `SHADOW_NO_LEND=1` turns it off (docs/STATUS.md).
### Changed
- Tensor-core prompt kernels (`--tensor-prompt`, default mode): the Q6_K / IQ4_XS projections and the grouped experts run on `mma.sync` FP16 tiles (3e-4 relative error, 7x faster kernels); one batched copy call per layer: prompts of 1,024+ tokens read 185-225 tok/s with the default tier in 2,048-token batches (was 130-145).
- Prompt reading of 1,024+ tokens: a layer-ahead pipeline (next layer's experts copied and its file reads started while this one computes) and 1,024-token batches from buffers carved out of the lent cache: 77-87 -> 128-144 tok/s, `--exact` 40 -> 62; decode unchanged.
- **Fix:** ending a prompt's cache lending could leave an expert owned by two cache slots (loaded again during the prompt, then restored into its old slot), which crashed a later chat turn with "a routed expert neither cached nor fetched" (HTTP 500) - seen at turn 14 of a 24-turn history with 250-token continuations.
- `shadow serve` defaults to `--verify-rank 2` instead of `--accept-moe`: decode 47.96 against 45.11 tok/s (varied bench), replies 10.6% shorter, same-history fidelity 0.423 (AcceptMoE 0.415, Strata 0.431). `--accept-moe` restores the old default.
- Attention kernel 2.7x faster (shared-memory q, 16-byte K loads, staged score tiles; same arithmetic order, bit-identical) and the indexer's scores in row launches: prompts of 1.8k-4.6k tokens read 4-28% faster.
- Long prompts: past the indexer budget (2,051 positions) attention selects each token's positions SEL_ROWS at a time and runs them in one launch per 8 tokens, instead of one synchronous round trip per token and layer; bit-identical, prompt reading +16% on a 4.5k-token prompt.
- `shadow serve` defaults to a 36 GiB RAM tier when 49 GiB of RAM is available at start (30 GiB otherwise): +2.5% decode, ~+10% prompt reading; `--ram-floor-gib N` makes the 4 GiB RAM floor a flag.
- The default tail skip is **3** (was 4): 39.4 tok/s by total throughput on varied text against 37.0, same reply length, same-history fidelity 0.415 against
  0.572 (Strata measured 0.431 the same way). `--tail-skip 4` restores the earlier mode. Added `--admit-after N` (VRAM-cache admission filter, opt-in, about +1%) and the
  `SHADOW_VRAM_MAP=1` startup probe (docs/STATUS.md has the cache, VRAM and bus measurements behind both).

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
- The n-gram (PLE) row prefetch reads through the model's read pool (own file handles: one shared synchronous handle serializes reads on Windows): prefetch
  3.5-9 ms -> 1.2 ms a verify pass, +7% in the default mode, output unchanged (34.7 -> 37.1 tok/s on the varied bench).
- Opt-in experiments, all exact and none a gain: `--pipeline` (two token groups per verify pass), `--cpu-experts N` in the verify pass, `--verify-rank R`.
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
