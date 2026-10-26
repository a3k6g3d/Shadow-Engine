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
    tools/           the harness, and shadow_memory.py
    bin/ models/     the embedder the memory tool drives
    memory/          the store — committed, so it travels with the repo

`tools/shadow_memory.py` is stdlib-only: `add`, `index`, `search`, `stats`. Embedding is not done in Python —
the tool shells out to `bin/llama-embedding.exe` with `models/bge-small-en-v1.5-q8_0.gguf` (384 dims). Search
is cosine over the whole store, no ANN, since the store is a few thousand records at most.

Its honest weakness, measured: on an 8-record store a paraphrased query put the answer to a *different*
question first (0.67) and the directly relevant memory third (0.62). It is a 35 MB quantised embedder and it
leans on phrasing overlap. Good enough to be worth having, not good enough to be trusted on one result.

## License

MIT, with both copyright holders. [NOTICE](NOTICE) says which parts came from where.
