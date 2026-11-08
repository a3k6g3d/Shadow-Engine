# Shadow

An engine for running Qwen3.8-Flash-Next mixture-of-experts models, built mainly in Rust.

**Based on [Strata](https://github.com/Niko1221/strata)** by Niko1221 and the Strata contributors, MIT licensed —
see [LICENSE](LICENSE) and [NOTICE](NOTICE). Shadow keeps that license and the upstream notice. It is not a copy
of Strata: its engine is its own. It does borrow parts, from Strata and from other projects, where they make Shadow
better, faster or more efficient, and [NOTICE](NOTICE) lists each one and where it came from.

## What it is

The plan is in [docs/SHADOW_PLAN.md](docs/SHADOW_PLAN.md), rewritten 2026-10-26. The short version:

- **Shadow is its own engine, start to finish.** A Rust host and CUDA kernels we write ourselves; the engine is not
  Strata's. Borrowed parts are welcome when they earn their place (the optional MTP draft head is one), and each is
  credited in NOTICE. The only required input is the GGUF.
- **The kernels are CUDA C++ source in this repo, compiled at run time by NVRTC** for whatever GPU is present, cached
  as CUBIN, and launched through the CUDA driver API (`nvcuda.dll`), hand-bound with no bindgen and no bindings
  crate. `cargo build` is the whole build: no nvcc, no CMake, no CUDA toolkit needed to compile Shadow.
- **The drive is the bound, not the language.** One process, 4 threads, queue depth 256: ~155k reads/s, and more
  threads or processes never add up to more. Shadow's ceiling on this PC is the same as any engine's. What it buys
  is an engine we own, safe by construction where Rust can make it so, with scheduling we can change.
- **Correctness before speed.** The stages each end in tokens: a scalar CPU reference first, the GPU dense path
  checked against it, then the memory tiers, the server, and only then speed, one lever at a time.

## Speed

Measured on one PC (RTX 5070 12 GB, i7-11700K, 64 GB RAM, NVMe, Windows 11) with the Swift IQ3_S GGUF of Qwen3.8-Flash-Next,
greedy, 96 tokens a turn, a 36-turn conversation of unseen prompts, scored by **total decode throughput** (tokens / decode
time; the bench's per-turn median misranks arms that generate different text, see
[docs/BENCHMARKING.md](docs/BENCHMARKING.md)):

| | tok/s | note |
|---|---|---|
| Shadow `--exact` | ~20 | no speculation, no inexact modes; bit for bit the reference |
| Shadow default (`shadow serve MODEL`) | **52.1** | DFlash 2 drafter, `--tail-skip 3`, `--verify-rank 2`, 36 GiB RAM expert tier (30 when under 49 GiB of RAM is free); same-history fidelity about 0.4 (Strata 0.43; the metric moves by about 0.04 between builds) |
| Strata (its tuned config, same model, prompts, context) | **47.7** | interleaved with Shadow in the same session |

**Shadow is about 9% faster than Strata on this bench** (52.1 against 47.7 tok/s by total throughput, 3 interleaved runs of 36 prompts each: Shadow 51.8, 52.2, 52.2, Strata 47.7, 47.4, 48.0). Shadow's replies are longer (1,781 tokens a run against Strata's 1,518) and its fidelity to the exact model
is about the same. With AcceptMoE verification (`--accept-moe`) Shadow runs about 3% slower. Prompt reading is slower: on this PC with the default 36 GiB tier (2026-11-06, mean of four server runs each) about 535 tok/s for a 3k-token prompt, 1,070 for 9.6k and 1,040 for 15k, against Strata 0.2.11's 1,261, 1,844 and 2,447 on the same files (docs/STATUS.md entries 41 and 54); Strata pins nearly all of the model's experts in RAM (45 of 46.8 GiB), Shadow's tier holds 77% of them and reads the rest from the SSD. The resource record
(`bench/vs-strata-final4/charts/`) shows where the time goes: Shadow's GPU is busier (84% against 78%), it moves more over PCIe (15.4 against 10.1 GB/s, about 296 against 212 MB of expert weights per generated token: the bigger RAM tier serves more of its misses) and uses the CPU and disk far less
(12% against 51% CPU, 150 against 468 MB/s of disk), but it keeps only about 4-5 GiB of RAM free (Strata 8-9 GiB). Earlier figures in this repository's history that put Shadow at 43 to 47 tok/s came from a mode that cut
answers short (`--verify-resident`) and from per-turn medians; both are documented in
[docs/STATUS.md](docs/STATUS.md). Shadow's noise floor is 0.8% median on the exact engine
(`bench/floor-2026-10-27`), about 0.1 to 0.5% for total throughput in the fastest mode.

## Quick start

```bash
cargo build --release
shadow serve path/to/model.gguf            # the fastest mode by default; add --exact for the exact path
# http://127.0.0.1:8081/ chat UI, Dashboard (#dash) and Settings (#settings) tabs, /v1/chat/completions OpenAI API
```

What the defaults are, every flag and every environment variable: [docs/USAGE.md](docs/USAGE.md). Shadow never pins memory
that would leave less than 4 GiB of RAM available, and the default 36 GiB expert tier (30 GiB when under 49 GiB is free) peaks around 41 GiB resident, so on a
smaller machine pass `--ram-tier-gib` explicitly.

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

Shadow runs one model family: Qwen3.8-Flash-Next (everything above). Flint (`RecursiveSI/flint`, NVFP4) was removed from this
build on 2026-11-07 (it stays in git history, tag `pre-flint-removal`; it will be tested on tandem.cpp later), and a Flint file
is refused with a clear message. It is not a general LLM runner: a dense Llama or Qwen3 GGUF has no forward pass here.

## Server & Web UI

* **Desktop UI (`shadow-ui.exe`)**: Windows launcher built with Rust and Tauri 2 (WebView2); connects to `http://127.0.0.1:8081`.
* **Web chat and process monitor**: streaming chat with collapsible `<think>` blocks, a reasoning-effort selector, native tool
  orchestration (`read_file`, `glob`, `grep`, `edit_file`, `write_file`, `cmd`) rendered as collapsible `.tool-call-block` cards,
  and an **unlimited rolling context window** (`0 B` extra RAM/VRAM) that preserves the system prompt prefix snapshot while
  compacting older tool outputs and folding older turns into a bounded summary; the Dashboard tab charts CPU, GPU load, temperature,
  RAM, expert-cache hit rates and the three memory tiers (VRAM, pinned RAM, NVMe).
* **API**: `POST /v1/chat/completions` (streaming or not), `GET /v1/models`, `/health`, `/metrics` (Prometheus), `/v1/status`,
  `/v1/telemetry`, `/v1/tools`, `/v1/tools/execute`.

## Layout

    crates/          the engine (shadow-model, shadow-cuda, shadow-serve, shadow), shadow-bench, shadow-eval
    kernels/         the CUDA C++ the engine compiles with NVRTC at run time
    docs/            USAGE, SPECULATION, BENCHMARKING, TOOLS, STATUS (what is measured), the plan, HANDOFF; docs/history/ holds the retired designs
    bench/           shadow-bench runs, one directory per comparison
    profiles/        routing profiles for the RAM tier's placement
    tools/           the harness and shadow_memory.py, both Python for now
    bin/ models/     the embedder the memory tool drives
    memory/          the store — committed, so it travels with the repo
    eval/            the frozen yardstick: cases.jsonl, baseline.jsonl, prompts/, answers/

`tools/shadow_memory.py` is stdlib-only: `add`, `index`, `search`, `stats`. Embedding is not done in Python —
the tool shells out to `bin/llama-embedding.exe` with `models/bge-small-en-v1.5-q8_0.gguf` (384 dims). Search
is cosine over the whole store, no ANN, since the store is a few thousand records at most.

## The yardstick

`eval/cases.jsonl` is 14 questions this project actually asked, each with the record that should answer it. It is
**frozen**: a self-improvement loop may edit its skills, its prompts and its index parameters, but not this file —
a ruler the measured thing is allowed to move stops being a ruler. Grading is substring matching, not an LLM
judge, because a judge inside the loop is slow and gets gameable the moment the loop learns what it likes.

The tool is `crates/shadow-eval` — Rust, because the project is Rust. Two arms, split on purpose:

- **`cargo run -p shadow-eval -- rank`** — retrieval only. It touches no model, so it is deterministic and it can be
  run from inside the session being improved. Baseline 2026-10-26, 16 records: **12/14 top-1, MRR 0.911**, recorded
  in `eval/baseline.jsonl`; `verify` re-ranks and fails loudly if it has moved.
- **`cargo run -p shadow-eval -- run`** — the answer arm. It needs a live model, and it spawns a **fresh** harness
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
