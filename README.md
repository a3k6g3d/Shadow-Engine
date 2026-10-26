# Shadow

An engine for running Qwen3.8-Flash-Next mixture-of-experts models, built mainly in Rust.

**Based on [Strata](https://github.com/Niko1221/strata)** by Niko1221 and the Strata contributors, MIT licensed —
see [LICENSE](LICENSE) and [NOTICE](NOTICE). Shadow keeps that license, keeps the upstream notice, and copies no
Strata source code: what came across is the plan, the measurements, the seam's contract and the harness, all of
which were written by us on a Strata checkout.

## What it is

The plan is in [docs/SHADOW_PLAN.md](docs/SHADOW_PLAN.md). The short version, since it is a short version of a
plan that has already been walked once:

- **The drive is the bound, not the language.** One process, 4 threads, queue depth 256: ~155k reads/s. Two
  processes ~112k total. Four ~113k. No language at the top of the stack beats this, and the Rust reader in the
  previous round came in at parity, −1.7% at the median, inside the noise floor.
- **So Rust buys the safety of the async tier and the scheduling layer**, and the speed comes from scheduling —
  52 ms of a ~115 ms decode window is the CPU running RAM-tier experts, which is CPU code and not a kernel.
- **The kernels stay C++.** Rust never compiles a kernel, it calls one, through
  [include/shadow/c/abi.h](include/shadow/c/abi.h) — hand-written, no bindgen, no cxx, and the crate's test suite
  parses the file and asserts the version constant matches its own.

## The floor

[docs/BENCH_NOISE.md](docs/BENCH_NOISE.md): **10.1% median, 27.0% p90, 51.9% worst.** A change must clear that, on
≥8 runs, interleaved, `decode_warm`, paired turn-by-turn — or it is reported as a tie and not written up as a win.
That rule is not a formality and it has killed more ideas than any reviewer.

## Layout

    crates/          the Rust side
    include/shadow/  the seam's contract (C ABI, hand-written)
    docs/            the plan, and the measurements it is built on
    tools/           the harness, shadow_memory.py, and the eval
    bin/ models/     the embedder the memory tool drives
    memory/          the store — committed, so it travels with the repo
    eval/            the frozen yardstick: cases.jsonl, prompts/, answers/

`tools/shadow_memory.py` is stdlib-only: `add`, `index`, `search`, `stats`. Embedding is not done in Python —
the tool shells out to `bin/llama-embedding.exe` with `models/bge-small-en-v1.5-q8_0.gguf` (384 dims). Search
is cosine over the whole store, no ANN, since the store is a few thousand records at most.

## The yardstick

`eval/cases.jsonl` is 14 questions this project actually asked, each with the record that should answer it. It is
**frozen**: a self-improvement loop may edit its skills, its prompts and its index parameters, but not this file —
a ruler the measured thing is allowed to move stops being a ruler. Grading is substring matching, not an LLM
judge, because a judge inside the loop is slow and gets gameable the moment the loop learns what it likes.

Two arms, split on purpose:

- **`shadow_eval.py rank`** — retrieval only. It touches no model, so it is deterministic and it can be run from
  inside the session being improved. Baseline 2026-10-26, 16 records: **12/14 top-1, MRR 0.911**.
- **`shadow_eval_run.py`** — the answer arm. It needs a live model, and it spawns a **fresh** harness session per
  case (one call, no tools, no skills) so the answer comes from the retrieved notes alone. Baseline, same day:
  **14/14**. Two of the fourteen cases are answered correctly *despite* the retriever ranking them 2nd and 4th —
  the answer arm passes where the rank arm does not, which is the honest reason to keep both arms and not just the
  convenient one.

The loop this enables is one edit at a time — a skill file, a prompt, an index parameter — re-scored, kept only if
it beats baseline by more than the noise floor in [BENCH_NOISE.md](docs/BENCH_NOISE.md), and reverted otherwise.
The model's weights are frozen, so every gain has to land as an artifact: a skill, a record, a tool. That is
accumulation with a keep/revert gate, and it compounds; it is not the weights getting smarter, and it is not
recursive past the point where the evaluator is fixed.

## License

MIT, with both copyright holders. [NOTICE](NOTICE) says which parts came from where.
