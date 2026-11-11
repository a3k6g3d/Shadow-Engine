# Tools

Everything under `tools/` and the other binaries of the workspace, what it is for and when to reach for it. Python tools
are stdlib-only unless noted; the ones that need torch run in a venv (`python -m venv .venv-train`, then torch from the
CUDA index, `safetensors`, `numpy`, `matplotlib`).

## Binaries (cargo)

| binary | what |
|---|---|
| `shadow` | the engine's CLI: `serve`, `generate`, `inspect`, `kv`, `stats`, `dump`, `compare`, `ppl` ([USAGE.md](USAGE.md)) |
| `shadow-bench` | the interleaved, paired benchmark runner with resource recording ([BENCHMARKING.md](BENCHMARKING.md)) |
| `shadow-eval` | the memory-retrieval yardstick (`rank`, `run`, `verify`) described in the README |
| `shadow-ui` | the desktop launcher (Tauri 2 / WebView2) for the server |

## Measuring

| tool | what |
|---|---|
| `tools/run_capped.ps1` | runs a program under a hard working-set cap and a free-RAM watchdog (kills below 4 GiB). **Every model load goes through it** (`shadow-bench` does this itself). |
| `tools/make_headless_shortcut.ps1` | creates the Start menu entry "Shadow (server only)" = `shadow-ui.exe --headless` (server + tray icon, no window) for running another harness against the API |
| `tools/bench_total.py DIR...` | total decode throughput per arm from a `shadow-bench` directory (the honest headline) |
| `tools/plot_bench.py DIR...` | charts and tables of the machine's CPU, RAM, disk, GPU, power, PCIe during each run, per run and per arm |
| `tools/autotune.py` | coordinate ascent over serving flags with paired bench rounds; adopts only what clears the noise |
| `tools/bench_vs_strata.ps1`, `run_strata.ps1`, `strata_bench_cfg.py` | the same bench against a Strata install, as an interleaved arm, with its server under the RAM floor |
| `tools/cache_sim.py` | replays a routing log through VRAM expert-cache policies and prints each one's hit rate (`SHADOW_ROUTE_LOG`, `SHADOW_CACHE_DUMP`) |
| `tools/spec_probe.py` | what speculation could buy from routing logs: expert-union size per verify pass, accepted tokens per pass |
| `tools/serve_check.py [URL]` | live checks of a running `shadow serve`: every route, streamed vs whole answers identical, prefix reuse, refusals, concurrent requests |

## The DFlash 2 drafter

| tool | what |
|---|---|
| `tools/dflash2_ref.py DIR` | recomputes the GPU drafter's block from the features it was given, in CPU torch, and compares hidden states and the selector walk (`SHADOW_DF_DUMP_DRAFTS=1`) |
| `tools/dflash2_train.py` | LoRA fine-tuning of the drafter on Shadow's own features; reports the free-running accepted prefix on held-out conversations ([SPECULATION.md](SPECULATION.md); the gain it showed once was an artifact, kept as a pipeline) |

## Research prototypes

| tool | what |
|---|---|
| `tools/route_predictability.py` | recall of a token's actual top-k experts per layer within cheap predicted sets (the last committed token's experts, unions of the last 2 and 4, a static frequency profile), by distance ahead: how predictable routing is without a learned predictor (`SHADOW_ROUTE_LOG` logs) |
| `tools/moe_speculative_gater.py` | PyTorch module `MoESpeculativeGater`: expert capacity gating for speculative verification of an offloaded MoE (commitment-weighted demand from a draft route predictor and the target router prior, effective-rank set size under a hard `k_max_experts` cap, residency-aware pruning, substitution routing `route_with_mask`); self-test with `python tools/moe_speculative_gater.py`. The same selection as `accept_route` in `session.rs`, for experiments outside the engine. |

## Reference and fixtures (correctness)

| tool | what |
|---|---|
| `tools/make_reference_fixture.py`, `make_reference_fixture_flint.py` | run the published reference implementation (transformers / the Flint model code) in float32 on the first N layers, write every layer's streams as a fixture; the GPU and CPU forward passes are held to it (`shadow compare`, `crates/shadow/tests/reference.rs`) |
| `tools/gen_dequant_fixtures.py`, `gen_quant_tables.py`, `gen_nfc_table.py`, `gen_unicode_table.py`, `make_tokenizer_fixture.py` | tables and fixtures the Rust tests are checked against |
| `tools/hf_range.py` | reads tensors from a Hugging Face safetensors checkpoint by HTTP range requests (no 360 GB download) |
| `tools/_refenv.py` | import before transformers in the reference scripts (a stand-in for a broken global torchaudio) |

## The agent harness and memory

`tools/shadow_harness.py` (a small command-line agent against the local server, six tools) and `tools/shadow_memory.py`
(semantic search over what it logged, embedding with `bin/llama-embedding.exe`), with `eval/` as their frozen yardstick.
See the README.
