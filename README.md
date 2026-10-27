# Shadow

An engine for running Qwen3.8-Flash-Next mixture-of-experts models, built mainly in Rust.

**Based on [Strata](https://github.com/Niko1221/strata)** by Niko1221 and the Strata contributors, MIT licensed —
see [LICENSE](LICENSE) and [NOTICE](NOTICE). Shadow keeps that license, keeps the upstream notice, and copies no
Strata source code: what came across is the plan, the measurements, the seam's contract and the harness, all of
which were written by us on a Strata checkout.

## What it is

The plan is in [docs/SHADOW_PLAN.md](docs/SHADOW_PLAN.md), rewritten 2026-10-26. The short version:

- **Shadow is its own engine, start to finish.** A Rust host, CUDA kernels we write ourselves, and nothing from
  Strata at build time or at run time: no source, no binary, no tool, no file format. The only input is the GGUF.
- **The kernels are CUDA C++ source in this repo, compiled at run time by NVRTC** for whatever GPU is present, cached
  as CUBIN, and launched through the CUDA driver API (`nvcuda.dll`), hand-bound with no bindgen and no bindings
  crate. `cargo build` is the whole build: no nvcc, no CMake, no CUDA toolkit needed to compile Shadow.
- **The drive is the bound, not the language.** One process, 4 threads, queue depth 256: ~155k reads/s, and more
  threads or processes never add up to more. Shadow's ceiling on this PC is the same as any engine's. What it buys
  is an engine we own, safe by construction where Rust can make it so, with scheduling we can change.
- **Correctness before speed.** The stages each end in tokens: a scalar CPU reference first, the GPU dense path
  checked against it, then the memory tiers, the server, and only then speed, one lever at a time.

## The floor

[docs/BENCH_NOISE.md](docs/BENCH_NOISE.md): **10.1% median, 27.0% p90, 51.9% worst.** A change must clear that, on
≥8 runs, interleaved, `decode_warm`, paired turn-by-turn — or it is reported as a tie and not written up as a win.
That rule is not a formality and it has killed more ideas than any reviewer. That floor was measured on Strata;
Shadow measures its own once it can run a session (stage 4), and until then Strata's is the working figure.

## Server & Web UI

Shadow runs a built-in OpenAI-compatible HTTP server and embedded web chat UI:

```text
shadow serve <path-to-gguf> [--port 8081] [--host 127.0.0.1] [--ram-tier-gib 30]
```

* **Web Chat UI**: Visit `http://127.0.0.1:8081/` in any browser for the built-in dark-mode chat interface with real-time SSE streaming, collapsible `<think>` reasoning blocks, and live performance metrics (tok/s, VRAM, and tier stats).
* **API Endpoints**:
  * `POST /v1/chat/completions`: OpenAI-compatible streaming and non-streaming chat completions.
  * `GET /v1/status`: Live throughput, queue status, and memory tier breakdown.
  * `GET /v1/models`, `/health`, `/metrics` (Prometheus).

## Layout

    crates/          the Rust side, including shadow-eval, the yardstick
    include/shadow/  the old seam's contract (C ABI) - retired with the C++ engine, removed once stage 2 runs
    docs/            the plan, and the measurements it is built on
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
