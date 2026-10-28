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

## The floor

Shadow measures its own noise with `shadow-bench` (`bench/floor-2026-10-27`): **0.8% median, 2.0% p90, 3.6% worst**
on `decode_warm`. A change must clear that on ≥8 interleaved launches, paired turn by turn, or it is reported as a
tie. Strata's floor (10.1% / 27.0% / 51.9%, [docs/BENCH_NOISE.md](docs/BENCH_NOISE.md)) is history. Current speed and
every lever tried, kept or not, are in [docs/STATUS.md](docs/STATUS.md).

## Server & Web UI

Shadow runs a built-in OpenAI-compatible HTTP server and embedded web chat UI:

```text
shadow serve <path-to-gguf> [--port 8081] [--host 127.0.0.1] [--ram-tier-gib 30] [--mtp models/mtp-basalt-ft1]
```

* **Desktop UI (`shadow-ui.exe`)**: Native Windows desktop launcher built with Rust and Tauri 2 (WebView2). Connects to `http://127.0.0.1:8081`, provides system tray controls, and manages single-instance window focusing.
* **MTP draft head (experimental, `--mtp <DIR>`, off by default)**: loads the borrowed `models/mtp-basalt-ft1` draft head (824 MB VRAM). It does **not** speed decoding up yet: the server checks a draft with ordinary one-token steps (Shadow has no batched verify pass), so every draft is extra work, and the head's KV cache is never filled for prompt positions. `tools/spec_probe.py` measures what speculation could buy here (docs/STATUS.md).
* **Web Chat & Process Monitor UI**: Visit `http://127.0.0.1:8081/` in any browser:
  * **Chat View**: Full SSE streaming, collapsible `<think>` blocks with direct inline "Stop Thinking" buttons, live reasoning elapsed timer, chat-bar stop button, global `Escape` cancellation shortcut, Claude & Antigravity style reasoning effort selector (`Off`, `Low`, `Medium`, `High`, `X-High`), native Windows tool orchestration (`read_file`, `glob`, `grep`, `edit_file`, `write_file`, `cmd`), and customizable system persona allowing sovereign local file inspection, test execution, and code modifications.
  * **Process Monitor View (`/monitor` or Tab Switcher)**: Strictly separate full-screen dashboard tracking accurate CPU % (process & system), real NVML GPU compute load %, GPU temperature, RAM working set & commit, MoE expert cache hit rates, and 3-tier memory distribution (VRAM -> Pinned RAM -> NVMe).
* **API Endpoints**:
  * `POST /v1/chat/completions`: OpenAI-compatible streaming and non-streaming chat completions (supporting `reasoning_effort` and `chat_template_kwargs.enable_thinking`).
  * `GET /v1/tools`: List of available native Windows agent tools.
  * `POST /v1/tools/execute`: Execute a local tool natively on Windows.
  * `GET /v1/telemetry`: Real-time hardware telemetry JSON (CPU, RAM, GPU utilization %, VRAM, and MoE cache stats).
  * `GET /monitor`: Direct access to the standalone system telemetry dashboard.
  * `GET /v1/status`: Live throughput, queue status, and memory tier breakdown.
  * `GET /v1/models`, `/health`, `/metrics` (Prometheus).

## Layout

    crates/          the engine (shadow-model, shadow-cuda, shadow-serve, shadow), shadow-bench, shadow-eval
    kernels/         the CUDA C++ the engine compiles with NVRTC at run time
    docs/            the plan, STATUS (what is measured), HANDOFF; docs/history/ holds the retired Strata-era designs
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
