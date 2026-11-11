# Using Shadow

How to build it, run it, and what each switch does. Every claim about speed in this file is a pointer; the numbers,
how they were measured and what did not work are in [STATUS.md](STATUS.md) and [BENCHMARKING.md](BENCHMARKING.md).

## What you need

| | tested on | notes |
|---|---|---|
| OS | Windows 11 | the engine uses Win32 APIs (pinned memory, overlapped file reads, CUDA driver DLL) |
| GPU | NVIDIA GeForce RTX 5070, 12 GB (compute 12.0) | any CUDA GPU the NVRTC in use can target; 12 GB is what the defaults are sized for |
| CPU | Intel i7-11700K (AVX-512) | the hybrid CPU-expert path uses AVX-512 when present |
| RAM | 64 GB | the default 36 GiB pinned expert tier needs ~49 GiB free (else the default is 30 GiB, ~44 GiB free); see "The RAM rule" |
| Disk | NVMe | expert reads that miss both VRAM and the RAM tier come from the GGUF |
| Rust | stable (edition 2021+) | `cargo build` is the whole build: no nvcc, no CMake, no CUDA toolkit to compile |
| CUDA runtime | the driver (`nvcuda.dll`) and an `nvrtc64_*.dll` | NVRTC is found via `SHADOW_NVRTC` (full path), else `%CUDA_PATH%\bin\x64` / `\bin` |
| Model | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S` GGUF | a plain GGUF; Shadow also runs the `Flint-40B-A2.5B` NVFP4 GGUF (see HANDOFF.md) |
| Drafters (optional) | `models/dflash2/model.safetensors`, `models/mtp-basalt-ft1/` | not in git (size); see [SPECULATION.md](SPECULATION.md) for where they come from |

## Build

```bash
cargo build --release            # shadow, shadow-bench, shadow-eval
cargo test --release -p shadow-cuda --test drift   # kernel signatures in Rust match the .cu files
```

The CUDA kernels are source files in `kernels/`, compiled when first used by NVRTC for the GPU that is present and cached
as CUBIN (`SHADOW_KERNEL_CACHE` moves the cache). Every kernel entry point's parameter list is written out in
`crates/shadow-cuda/src/kernels.rs`, each launch is checked against it, and `tests/drift.rs` checks that list against the
`.cu` text, so the two cannot drift.

## Run the server

```bash
shadow serve MODEL.gguf
```

**Server only, from the Start menu (no window, no chat page):** the Start menu entry **Shadow (server only)** runs `shadow-ui.exe --headless` - the same server as the **Shadow** entry (same `%APPDATA%\Shadow\desktop.json`, so the same model and port 8081) but no window. A tray icon stays: hover shows the state (starting / loading / ready), the menu has *Open Chat in Browser*, *Restart Server*, *Open Console Log*, *Open Settings* and *Quit* (Quit stops the server it started; a server that was already running is left alone). Point another harness (Hermes, pi, ...) at `http://127.0.0.1:8081/v1`. It is one instance with the normal launcher: start only one of the two. `tools/make_headless_shortcut.ps1` (re)creates the entry (run it again after the exe moves).

With no tier flags this is the **fastest measured mode**, an OpenAI-compatible server on `http://127.0.0.1:8081`:

| default | flag it stands for | what it is |
|---|---|---|
| inexact tail skip | `--tail-skip 3` | a routed expert past the 3rd by router weight is skipped unless it is already in VRAM |
| AcceptMoE verify | `--accept-moe --accept-beta 1.0` | a verify pass routes its drafted positions through one block-wide expert set (the anchor's top-k plus the experts the likely-to-be-accepted positions demand); each token still takes k experts, re-picked inside the set |
| RAM expert tier | `--ram-tier-gib 36` | 36 GiB (30 when under 49 GiB of RAM is available at start) of pinned host memory holds the likeliest experts, copied to the GPU by DMA |
| placement | `--placement profiles/routing-varied300.shrc` | which experts the RAM tier holds (a routing profile) |
| drafter | `--dflash models/dflash2` | the DFlash 2 drafter, matrices as Q8_0 |
| draft length | `--spec 3` | up to 3 drafted tokens checked per pass |
| draft cutoff | `--dflash-min-p 0.5` | a draft chain stops after a draft the selector gives under 0.5 |

Each default applies only when you did not pass that flag and the file exists. **`--exact`** turns all of them off: the
exact, unspeculated path, bit for bit the reference (about 20 tok/s on the test machine against about 35 in the
default mode on varied text). The default mode is **not exact**: `--tail-skip` changes which experts run, so answers can differ from the
exact path. The verify pass still checks every drafted token against the model it is running, so speculation itself never
changes an answer; only the tail skip does.

Pictures: the chat page's 📎 button (or drop / paste) attaches a picture or a text/code file; the API takes `image_url` parts (`data:image/...;base64,`). The picture
encoder is `bin/strata-vision.exe` with `models/Qwen3.8-Flash-Next.mmproj-f16.gguf` (`SHADOW_VISION_EXE` / `SHADOW_MMPROJ` to point elsewhere); it needs ~1.8 GiB of
free RAM above the floor while it runs (docs/STATUS.md, 2026-11-01).

Open `http://127.0.0.1:8081/` for the chat UI and its Dashboard tab (`/#dash`; `/monitor` goes there) for the telemetry (CPU, GPU load, temperature,
RAM, expert-cache hit rates, memory tiers).

## Flags

### `shadow serve MODEL [flags]`

| flag | default | meaning |
|---|---|---|
| `--host H`, `--port P` | `127.0.0.1`, `8081` | where to listen |
| `--ctx N` | 24576 | positions the session is built for (KV and drafter rings scale with it) |
| `--exact` | off | no speculation, no inexact modes, none of the defaults above |
| `--ram-tier-gib G` or `auto` | 36 (30 when under 49 GiB of RAM is available) | pinned host memory for the expert tier; `auto` sizes it from the RAM available now less 10 GiB |
| `--ram-floor-gib N` | 4 | RAM that must stay available after pinning (0: no check); see "The RAM rule" |
| `--placement FILE` | the six-prompt profile | a routing profile (`SHRC`) ranking which experts the tier holds |
| `--expert-cache-gib G` | all free VRAM less 384 MiB | the VRAM expert cache (LRU per layer) |
| `--admit-after N` | 0 | a VRAM-cache miss gets a slot only from its (N+1)th miss (N=1: about +1%; STATUS.md) |
| `--pin-margin N` | off | pin each layer's hottest experts in VRAM, keeping N evictable slots a layer |
| `--tail-skip R` | 3 | routed experts of rank >= R are skipped when not in VRAM (0: exact; 4: the earlier default, fidelity 0.572 and 6% slower) |
| `--accept-moe`, `--accept-beta B`, `--accept-scale S`, `--accept-budget F` | on, 1.0, 1, 1 | AcceptMoE verifier expert sets (docs/SPECULATION.md); B sharpens the commitment weights, S scales the set, F the pruning budget |
| `--verify-resident` | off | **can truncate answers**: a verify pass's extra positions use only the experts already in VRAM; on varied text 11 of 14 replies ended early against 3 |
| `--verify-rank R` | off | like `--verify-resident` but each extra position keeps its top R experts even when they must be copied; R=1: about +18% tok/s and about 17% fewer generated tokens; R=2 (`shadow serve MODEL --verify-rank 2`, replaces AcceptMoE): 42.8 tok/s against the default's 39.4 on the varied bench, replies 11.5% shorter, same-history fidelity above Strata's (STATUS.md) |
| `--dflash DIR` or `off` | `models/dflash2` | the DFlash 2 drafter |
| `--dflash-bf16` | off | keep the drafter's matrices BF16 instead of Q8_0 (about 0.4 GB more VRAM) |
| `--dflash-min-p P` | 0.5 | stop a draft chain after a draft the selector gives under P |
| `--mtp DIR` or `off` | off | the MTP draft head instead of DFlash (`models/mtp-basalt-ft1`) |
| `--spec K` | 3 with a drafter | drafted tokens checked per pass; without a drafter, prompt-lookup drafting |
| `--cpu-experts N`, `--cpu-threads T` | 0, 8 | compute up to N RAM-tier misses a layer on the CPU instead of copying them, in decode and in verify passes (opt-in: measured a tie at N=2 and a loss at 1 and 4; STATUS.md) |
| `--direct-io`, `--no-direct-io` | off | unbuffered overlapped Direct I/O straight into pinned staging (`SHADOW_DIRECT_IO=1`; measured -8.7% to -42% loss due to losing OS file cache; opt-in) |
| `--pregate N` | 0 | experimental router lookahead |
| `--pipeline` | off | a verify pass as two token groups with one group's expert work under the other's mixer (exact; measured -10% in the default mode, a tie in exact mode; STATUS.md) |
| `--fuse N` | 0 | kernel fusions, bits 1/2/4 (measured a tie or a loss; opt-in) |

`shadow generate MODEL "prompt" | @FILE [-n N] [--gpu] [--chat] [tier flags]` decodes greedily from the command line,
`--record-routing FILE` saves the experts a run routed to (a placement profile). `shadow inspect|kv|stats|dump|compare|ppl`
are the model tools: header and tensor listing, a metadata key, routing statistics, a tensor dequantized to f32, the
reference comparison, perplexity.

### Two graphics cards (experimental, `docs/PLAN_TWO_GPUS_2026-11-11.md`)

`shadow serve MODEL --gpus 0,1 [--split N] [--peer-cache-gib G]`: the model's first `N` layers on the first card, the rest on the second, whose VRAM then holds those layers' experts (`--split` omitted: as many layers as fit; `--peer-cache-gib` caps the second card's expert cache). The engine sees one session. No drafter across two cards yet (it is dropped, speculation falls back to prompt lookup); conversation snapshots are not written to disk in this mode. **With one card, with `--gpus` absent or naming one card, with `--gpus` naming a card the driver does not see (a warning, then card 0 alone), or with `SHADOW_GPUS=1`, the whole model runs on one card exactly as before.** `shadow gpus` lists the cards and measures what the host can move through each (bandwidth, the hop between them). `--gpus 0,0` runs both stages on one card (a test mode: give both a small expert cache).

### Environment variables

Defaults are right for normal use; these are for measurement and diagnosis.

| variable | effect |
|---|---|
| `SHADOW_NVRTC`, `SHADOW_KERNEL_CACHE` | where NVRTC and the CUBIN cache are |
| `SHADOW_REPORT=1` | print the session report after each answer (tier hits, verify pass times, launches) |
| `SHADOW_LTIME=1`, `SHADOW_KTIME=1`, `SHADOW_PROFILE=1` | time a batched pass by segment / each kernel / the single-token step (these synchronize: they slow the run) |
| `SHADOW_DF_TIME=1` | time the phases of the DFlash draft round |
| `SHADOW_DF_KEEP=0` | the old behaviour: any conversation-snapshot restore switches the DFlash drafter off for the rest of the session (decode -17% to -38%, STATUS 98-99). Default on: the drafter keeps the K/V it still has after a restore |
| `SHADOW_DF_DUMP=DIR` | write the drafter's features and tokens (`feats.bin`) for training and the reference check; `SHADOW_DF_DUMP_DRAFTS=1` also each draft |
| `SHADOW_RECORD_ROUTING=FILE` | `serve` rewrites a routing profile of everything it has read after each answer |
| `SHADOW_NO_PREFIX=1` | no shared-prefix snapshots (every new conversation reads its whole prompt) |
| `SHADOW_SPEC_MIN_P`, `SHADOW_LOOKUP_MIN` | the MTP chain cutoff; the shortest repeat that prompt-lookup drafts |
| `SHADOW_GRAPHS=0`, `SHADOW_OVERLAP=0`, `SHADOW_BATCH=n`, `SHADOW_REF_ORDER=1` | switch off CUDA graphs / overlapped copies / batching / use the reference sum order |
| `SHADOW_FUSE`, `SHADOW_ZERO_COPY`, `SHADOW_SYNC`, `SHADOW_HUGE_PAGES` | opt-in experiments (all measured; see STATUS.md) |

## The RAM rule

Shadow never pins memory that would leave less than **4 GiB** of RAM available, checks again after the pages are locked and
again as the tier fills: a pinned tier cannot be paged out, and the test machine has frozen hard at 0.4 GiB free.
The default 36 GiB tier runs with `--ram-floor-gib 3` (available RAM bottoms out at 4.1-5.1 GiB from a start of ~49); any tier you give keeps the 4 GiB floor unless you pass the flag. `--ram-floor-gib N` changes the floor (0 turns the check off); `tools/run_capped.ps1 -FloorGiB N` and `tools/run_strata.ps1` do the same for the
watchdog that kills a run below it. Lower it only to try a bigger `--ram-tier-gib`, and measure that it pays. The default 36 GiB tier peaks around 41 GiB resident (the 30 GiB one around 35) for the process. On a smaller
machine pass `--ram-tier-gib` explicitly (or `auto`), or `--exact` with a small tier.

## The API

OpenAI-compatible: `POST /v1/chat/completions` (streaming and not; `reasoning_effort`, `chat_template_kwargs.enable_thinking`),
`GET /v1/models`, `/health`, `/metrics` (Prometheus), `/v1/status` (throughput, queue, memory tiers), `/v1/telemetry`
(CPU, RAM, GPU, expert-cache stats), `/v1/tools` and `/v1/tools/execute` (the native Windows agent tools). The response
carries llama.cpp-style `timings` (`predicted_n`, `predicted_ms`, `predicted_per_second`, `prompt_n`, ...), which
`shadow-bench` reads.

## MCP tools

`mcp.json` in the Shadow folder lists the MCP servers `shadow serve` starts (the format of Claude Desktop / Claude Code, so a list can be copied). Shipped on: `time` (current time, time zones) and `thinking`
(sequential thinking). Shipped off, set `"disabled": false` or delete the line to use them: `git` (`uvx mcp-server-git`) and `files` (the filesystem server on `J:/Shadow`). They need `uvx` / `npx` (installed) and the network
the first time (the packages are fetched once). Each tool is offered to the model as `mcp__<server>__<tool>` and costs prompt tokens in every chat's start (the first chat reads them once, then the server caches them);
`mcp.json` is looked for in the working directory, beside the exe, in the checkout Shadow was built from, then `%APPDATA%\Shadow` (`SHADOW_MCP_CONFIG` names one). The Dashboard's Tools card shows each server and its tool count or error.
