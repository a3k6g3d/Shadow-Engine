# ShadowZero

An engine for running Qwen3.8-Flash-Next mixture-of-experts models, built mainly in Rust.

**Based on [Strata](https://github.com/Niko1221/strata)** by Niko1221 and the Strata contributors, MIT licensed —
see [LICENSE](LICENSE) and [NOTICE](NOTICE). ShadowZero keeps that license and the upstream notice. It is not a copy
of Strata: its engine is its own. It does borrow parts, from Strata and from other projects, where they make ShadowZero
better, faster or more efficient, and [NOTICE](NOTICE) lists each one and where it came from.

## What it is

The plan is in [docs/SHADOW_PLAN.md](docs/SHADOW_PLAN.md), rewritten 2026-10-26. The short version:

- **ShadowZero is its own engine, start to finish.** A Rust host and CUDA kernels we write ourselves; the engine is not
  Strata's. Borrowed parts are welcome when they earn their place (the optional MTP draft head is one), and each is
  credited in NOTICE. The only required input is the GGUF.
- **The kernels are CUDA C++ source in this repo, compiled at run time by NVRTC** for whatever GPU is present, cached
  as CUBIN, and launched through the CUDA driver API (`nvcuda.dll`), hand-bound with no bindgen and no bindings
  crate. `cargo build` is the whole build: no nvcc, no CMake, no CUDA toolkit needed to compile ShadowZero.
- **The drive is the bound, not the language.** One process, 4 threads, queue depth 256: ~155k reads/s, and more
  threads or processes never add up to more. ShadowZero's ceiling on this PC is the same as any engine's. What it buys
  is an engine we own, safe by construction where Rust can make it so, with scheduling we can change.
- **Correctness before speed.** The stages each end in tokens: a scalar CPU reference first, the GPU dense path
  checked against it, then the memory tiers, the server, and only then speed, one lever at a time.

## Speed

Measured on the project test box (RTX 5070 12 GiB @ 200W + RTX 5060 Ti 16 GiB @ 150W, i7-11700K, 64 GB RAM, NVMe, Windows 11, `--ctx 24576`) running mixed EXL3 Qwen3.8-Flash-Next (`mixD18`), greedy decoding, with DFlash 2 speculation (`--spec 3`):

| setup | varied36 decode (median) | short prompts | 3.3k prompt | 9.7k prompt | 15.2k prompt |
|---|---|---|---|---|---|
| Pair (`--gpus 0,1`, v0.1.37) | 63.23 (60.00) tok/s | 115.08 tok/s | 755.2 tok/s | 1095.9 tok/s | — |
| **Pair (`--gpus 0,1`, v0.1.41)** | **72.32 (72.26) tok/s** | **125.19 tok/s** | **836.0 tok/s** | **1298.5 tok/s** | **1397.0 tok/s** |
| Single (`--gpus 0`, v0.1.41) | 75.76 (76.21) tok/s | 75.65 tok/s | 878.5 tok/s | 1163.6 tok/s | 1204.0 tok/s |

- **One GPU vs. two GPUs on this board:** Because the second PCIe slot on this motherboard is chipset-wired PCIe 3.0 x4 (`~3.4 GB/s` vs `27.4 GB/s` on the main slot) and WDDM disables P2P, single-card RTX 5070 keeps a ~4–5% decode edge while **two cards load faster (`39.6s` vs `45.6s`) and pull well ahead on prompts** (+65.5% on short prompts, +11.6% at 9.7k, +16.0% at 15.2k) because layers `0..15` stay 100% resident on the 5060 Ti.
- **Earlier IQ3_S GGUF baseline vs. Strata:** On single-card IQ3_S (`v0.1.35–v0.1.36`), ShadowZero hit **55.7 tok/s** decode against Strata's **49.5 tok/s** (`+12.5%`, [docs/COMPARISON_STRATA_2026-11-11.md](docs/COMPARISON_STRATA_2026-11-11.md)) and up to **1,764 tok/s** on 15k prompts (`+16%`). Exact mode (`--exact`) turns off speculation and fast approximations (~20 tok/s on IQ3_S) to match the scalar reference bit for bit.

![Decode and short-prompt progression from v0.1.37 to v0.1.41](docs/charts/decode-progression.svg)

![Long-prompt prefill scaling across 3.3k, 9.7k, and 15.2k tokens](docs/charts/prompt-prefill-scaling.svg)

![Dual-GPU layout and layer split sweep](docs/charts/dual-gpu-split-and-pipeline.svg)

## Quick start

```bash
cargo build --release
shadowzero gpus                                # list cards and measure host <-> GPU / hop bandwidth
shadowzero serve path/to/model --gpus 0        # single GPU, fastest defaults (add --exact for bit-exact)
shadowzero serve path/to/model --gpus 0,1      # dual GPU, auto-splits layers across both cards
```

That opens the web chat UI at `http://127.0.0.1:8081/` (with a live multi-GPU Dashboard at `/#dash`) and an OpenAI-compatible API at `/v1/chat/completions`.

What every flag and environment switch does is in [docs/USAGE.md](docs/USAGE.md). ShadowZero keeps at least 4 GiB of system RAM free when sizing the pinned host expert tier, so on smaller machines pass `--ram-tier-gib` explicitly.

## Examples

Standalone, zero-dependency examples live in [`examples/`](examples/README.md):

- [`examples/quickstart_single_and_dual_gpu.ps1`](examples/quickstart_single_and_dual_gpu.ps1) — GPU/bandwidth probe and ready-to-run single-card, dual-card, and `--exact` launch commands
- [`examples/chat_and_streaming.py`](examples/chat_and_streaming.py) — non-streaming and SSE streaming `/v1/chat/completions` calls (including `reasoning_content` and `timings`)
- [`examples/agent_tool_loop.py`](examples/agent_tool_loop.py) — minimal local tool-calling agent loop using `/v1/tools`, `/v1/tools/execute`, and `finish`
- [`examples/watch_telemetry.py`](examples/watch_telemetry.py) — live terminal monitor for per-GPU utilization/VRAM, expert cache hit rate, RAM tier activity, and DFlash acceptance

## Documentation

| | |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | requirements, build, run, flags, environment variables, the API |
| [docs/SPECULATION.md](docs/SPECULATION.md) | the verify pass, the drafters (prompt lookup, MTP, DFlash 2), AcceptMoE, what did not work |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | the protocol, total-throughput scoring, resource charts, comparing with another engine, autotune, pitfalls |
| [docs/TOOLS.md](docs/TOOLS.md) | every tool and binary |
| [docs/STATUS.md](docs/STATUS.md) | what is measured, in order, including the wrong turns |
| [docs/ARCH.md](docs/ARCH.md) | the model as math, from the file and the reference |
| [docs/SHADOW_PLAN.md](docs/SHADOW_PLAN.md), [docs/HANDOFF.md](docs/HANDOFF.md) | the plan and the state handed over |
| [CHANGELOG.md](CHANGELOG.md) | release notes |

ShadowZero runs one model family: Qwen3.8-Flash-Next (everything above). Flint (`RecursiveSI/flint`, NVFP4) was removed from this
build on 2026-11-07 (it stays in git history, tag `pre-flint-removal`; it will be tested on tandem.cpp later), and a Flint file
is refused with a clear message. It is not a general LLM runner: a dense Llama or Qwen3 GGUF has no forward pass here.

## Server & Web UI

* **Desktop app (`shadowzero-ui.exe`)**: Windows launcher built with Tauri 2 (and headless server-only tray mode via `shadowzero-ui.exe --headless`), pointing at `http://127.0.0.1:8081`.
* **Chat UI & Dashboard**: Streaming chat with collapsible thinking blocks, reasoning-effort control, and local file/command/MCP tools folded into a single collapsible bubble per turn. Rolling context compaction keeps long agent sessions inside your VRAM budget, while the Dashboard (`/monitor` or `#dash`) tracks CPU, per-GPU load/VRAM/temps across both cards, expert-cache hit rates, and RAM/disk tier activity.
* **API**: `POST /v1/chat/completions` (streaming or non-streaming), `GET /v1/models`, `/health`, `/metrics` (Prometheus), `/v1/status`, `/v1/telemetry`, `/v1/tools`, `/v1/tools/execute`.

## Layout

    crates/          the engine (shadowzero-model, shadowzero-cuda, shadowzero-serve, shadow), shadowzero-bench, shadowzero-eval
    kernels/         the CUDA C++ compiled at run time with NVRTC
    examples/        quick PowerShell & Python examples (chat, streaming, tools, live GPU telemetry)
    docs/            USAGE, SPECULATION, BENCHMARKING, TOOLS, STATUS, plans, and docs/charts/ SVGs
    bench/           shadowzero-bench runs, one directory per comparison
    profiles/        routing profiles for the RAM tier's placement
    tools/           the harness and shadowzero_memory.py
    bin/ models/     the embedder the memory tool drives
    memory/          the store — committed, so it travels with the repo
    eval/            the frozen yardstick: cases.jsonl, baseline.jsonl, prompts/, answers/

`tools/shadowzero_memory.py` is stdlib-only: `add`, `index`, `search`, `stats`. Embedding is not done in Python —
the tool shells out to `bin/llama-embedding.exe` with `models/bge-small-en-v1.5-q8_0.gguf` (384 dims). Search
is cosine over the whole store, no ANN, since the store is a few thousand records at most.

## The yardstick

`eval/cases.jsonl` is 14 questions this project actually asked, each with the record that should answer it. It is
**frozen**: a self-improvement loop may edit its skills, its prompts and its index parameters, but not this file —
a ruler the measured thing is allowed to move stops being a ruler. Grading is substring matching, not an LLM
judge, because a judge inside the loop is slow and gets gameable the moment the loop learns what it likes.

The tool is `crates/shadowzero-eval` — Rust, because the project is Rust. Two arms, split on purpose:

- **`cargo run -p shadowzero-eval -- rank`** — retrieval only. It touches no model, so it is deterministic and it can be
  run from inside the session being improved. Baseline 2026-10-26, 16 records: **12/14 top-1, MRR 0.911**, recorded
  in `eval/baseline.jsonl`; `verify` re-ranks and fails loudly if it has moved.
- **`cargo run -p shadowzero-eval -- run`** — the answer arm. It needs a live model, and it spawns a **fresh** harness
  session per case (one call, no tools, no skills) so the answer comes from the retrieved notes alone. Baseline, same
  day: **14/14**, graded with `answers`. Two of the fourteen are answered correctly *despite* the retriever ranking
  them 2nd and 4th — the answer arm passes where the rank arm does not, which is the honest reason to keep both arms
  and not just the convenient one.

The rank arm reproduces exactly; the answer arm does not, because it is a live model — re-running it reproduces the
verdict, not the wording. That is the whole reason the deterministic arm is the one with a recorded baseline.

The loop this enables is one edit at a time — a skill file, a prompt, an index parameter — re-scored, kept only if
it beats baseline by more than the noise floor in [BENCH_NOISE.md](docs/BENCH_NOISE.md), and reverted otherwise.
The model's weights are frozen, so every gain has to land as an artifact: a skill, a record, a tool. That is
accumulation with a keep/revert gate, and it compounds; it is not the weights getting smarter, and it is not
recursive past the point where the evaluator is fixed.

## License

MIT, with both copyright holders. [NOTICE](NOTICE) says which parts came from where.
