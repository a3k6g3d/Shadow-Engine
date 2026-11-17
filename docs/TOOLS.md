# Tools

Everything under `tools/` and the other binaries of the workspace, what it is for and when to reach for it. Python tools
are stdlib-only unless noted; the ones that need torch run in a venv (`python -m venv .venv-train`, then torch from the
CUDA index, `safetensors`, `numpy`, `matplotlib`).

## Binaries (cargo)

| binary | what |
|---|---|
| `shadow` | the engine's CLI: `serve`, `generate`, `inspect`, `kv`, `stats`, `dump`, `compare`, `ppl` ([USAGE.md](USAGE.md)) |
| `shadowzero-bench` | the interleaved, paired benchmark runner with resource recording ([BENCHMARKING.md](BENCHMARKING.md)) |
| `shadowzero-eval` | the memory-retrieval yardstick (`rank`, `run`, `verify`) described in the README |
| `shadowzero-ui` | the desktop launcher (Tauri 2 / WebView2) for the server |

## Measuring

| tool | what |
|---|---|
| `tools/run_capped.ps1` | runs a program under a hard working-set cap and a free-RAM watchdog (kills below 4 GiB). **Every model load goes through it** (`shadowzero-bench` does this itself). |
| `tools/make_headless_shortcut.ps1` | creates the Start menu entry "ShadowZero (server only)" = `shadowzero-ui.exe --headless` (server + tray icon, no window) for running another harness against the API |
| `tools/bench_total.py DIR...` | total decode throughput per arm from a `shadowzero-bench` directory (the honest headline) |
| `tools/plot_bench.py DIR...` | charts and tables of the machine's CPU, RAM, disk, GPU, power, PCIe during each run, per run and per arm |
| `tools/autotune.py` | coordinate ascent over serving flags with paired bench rounds; adopts only what clears the noise |
| `tools/bench_vs_strata.ps1`, `run_strata.ps1`, `strata_bench_cfg.py` | the same bench against a Strata install, as an interleaved arm, with its server under the RAM floor |
| `tools/cache_sim.py` | replays a routing log through VRAM expert-cache policies and prints each one's hit rate (`SHADOWZERO_ROUTE_LOG`, `SHADOWZERO_CACHE_DUMP`) |
| `crates/shadowzero-cuda/tests/grouped_bench.rs` | times the grouped expert kernels on real expert tensors with a verify pass's group table (cold weights); `GROUPED_SRC=variant.cu` times a variant of matvec.cu beside the repo's and compares the outputs byte for byte (`GROUPED_VARIANT_ROWS`, `GROUPED_ONLY=IQ4_NL`); STATUS 114 |
| `tools/bringup_two_gpus.ps1` | the one-command bring-up of a second card: `shadowzero gpus`, the pair tests and the int8 kernels on card 1, exact and fast conversations on the pair, a decode bench (single / pair / pair without the drafter), power and temperature logged; `-Peer 0` is a dry run on one card (STATUS 113) |
| `tools/ncu_decode.ps1` | Nsight Compute on one verify pass's biggest decode kernels (SpeedOfLight, stall, occupancy, memory sections), elevated PowerShell; uses `SHADOWZERO_NCU_RANGE` + `--profile-from-start off` because `--launch-skip` hangs (STATUS 108) |
| `tools/nsys_decode_rounds.py X.sqlite` | decode only from an Nsight Systems export (`--cuda-graph-trace=node`): per DFlash round the kernel, copy and idle time, the kernels by name, the gaps by size and by the CUDA call running in them |
| `tools/nsys_decode_capture.ps1 -Tag NAME [-Flags "..."]` + `tools/nsys_decode_client.py` | nsys capture of a served conversation (`--cuda-graph-trace=node`, the first six `varied36` prompts) of any flag set, the shadowzero.exe child capped like `run_capped`; export with `nsys export --type sqlite` (STATUS 155) |
| `tools/nsys_pair_rounds.py X.sqlite` | per DFlash round on one or two cards: busy time per device, both busy at once, idle, copies by kind and device, kernels by name per device, and the segment timeline of a typical round (STATUS 155) |
| `tools/nsys_prompt_bound.py X.sqlite` + `tools/nsys_prompt_client.py` | long prompts under nsys (`nsys_decode_capture.ps1 -Client tools/nsys_prompt_client.py`): per prompt burst the wall time, kernel-busy, H2D-copy-busy, both, neither, GiB copied and the kernel time by name (STATUS 157) |
| `tools/mix_exl3.py BASE OTHER OUT 0,1,..[,dense]` | an EXL3 checkpoint from two quants of the same weights (streams tensor bytes: ~0.7 GiB of RAM, 4 minutes for 40 GiB): BASE with the routed experts of the listed layers (and with `dense` the non-expert tensors and the head) from OTHER; the n-gram table is hard-linked (STATUS 165) |
| `tools/expert_mask.py PROFILE.shrc K OUT.bin [--scores S.npy]` | a `SHADOWZERO_EXPERT_MASK` file keeping the K most-used (or highest-scored) experts of every layer (STATUS 168) |
| `tools/exl3_export_tables.py MODEL OUT [cuda_device]` | an EXL3 checkpoint's token embedding and dequantized output head as float32 files (`token_embd.f32`, `output.f32`) for `tools/dflash2_train.py`, which `shadowzero dump` cannot make for a directory; run in the exllamav3 venv (STATUS 166) |
| `tools/spec_probe.py` | what speculation could buy from routing logs: expert-union size per verify pass, accepted tokens per pass |
| `tools/serve_check.py [URL]` | live checks of a running `shadowzero serve`: every route, streamed vs whole answers identical, prefix reuse, refusals, concurrent requests |

## The DFlash 2 drafter

| tool | what |
|---|---|
| `tools/dflash2_ref.py DIR` | recomputes the GPU drafter's block from the features it was given, in CPU torch, and compares hidden states and the selector walk (`SHADOWZERO_DF_DUMP_DRAFTS=1`) |
| `tools/dflash2_train.py` | LoRA fine-tuning of the drafter on ShadowZero's own features; reports the free-running accepted prefix on held-out conversations ([SPECULATION.md](SPECULATION.md); the gain it showed once was an artifact, kept as a pipeline) |

## Research prototypes

| tool | what |
|---|---|
| `tools/route_predictability.py` | recall of a token's actual top-k experts per layer within cheap predicted sets (the last committed token's experts, unions of the last 2 and 4, a static frequency profile), by distance ahead: how predictable routing is without a learned predictor (`SHADOWZERO_ROUTE_LOG` logs) |
| `tools/moe_speculative_gater.py` | PyTorch module `MoESpeculativeGater`: expert capacity gating for speculative verification of an offloaded MoE (commitment-weighted demand from a draft route predictor and the target router prior, effective-rank set size under a hard `k_max_experts` cap, residency-aware pruning, substitution routing `route_with_mask`); self-test with `python tools/moe_speculative_gater.py`. The same selection as `accept_route` in `session.rs`, for experiments outside the engine. |

## Reference and fixtures (correctness)

| tool | what |
|---|---|
| `tools/make_reference_fixture.py`, `make_reference_fixture_flint.py` | run the published reference implementation (transformers / the Flint model code) in float32 on the first N layers, write every layer's streams as a fixture; the GPU and CPU forward passes are held to it (`shadowzero compare`, `crates/shadowzero/tests/reference.rs`) |
| `tools/gen_dequant_fixtures.py`, `gen_quant_tables.py`, `gen_nfc_table.py`, `gen_unicode_table.py`, `make_tokenizer_fixture.py` | tables and fixtures the Rust tests are checked against |
| `tools/hf_range.py` | reads tensors from a Hugging Face safetensors checkpoint by HTTP range requests (no 360 GB download) |
| `tools/fetch_exl3_subset.py` | `REPO REV OUT.safetensors REGEX...`: pulls the tensors of an EXL3 (or any safetensors) repo branch whose names match, by range requests, into one small valid safetensors file; what the EXL3 tests and fixtures are made from |
| `tools/gguf_tensor_types.py` | `FILE.gguf [SUBSTRING...]`: every tensor of a GGUF (the ones whose name has a substring) with dims and quant type, then a count per type; reads the header only |
| `tools/find_exl3.py` | lists every EXL3 repo of Qwen3.8-Flash-Next on the Hub (several searches), marks the abliterated / uncensored / heretic ones and the gated ones |
| `tools/analyze_exl3.py` | `REPO [REVISION]`: reads the safetensors headers of an EXL3 repo by range requests: total size, n-gram table size, the bit rate K (halves included) of every routed expert per layer and of the dense, MTP and vision tensors; what decides whether ShadowZero can load a repo |
| `tools/fetch_base_experts.py` | `OUT LAYERS EXPERTS`: slices of the fused expert tensors of the BF16 base checkpoint (`Qwen/Qwen3.8-Flash-Next`) for chosen layers and experts, by range requests, under per-expert names; what `tests/exl3_vs_base.rs` compares the EXL3 decode with |
| `tools/_refenv.py` | import before transformers in the reference scripts (a stand-in for a broken global torchaudio) |

## The agent harness and memory

`tools/shadowzero_harness.py` (a small command-line agent against the local server, six tools) and `tools/shadowzero_memory.py`
(semantic search over what it logged, embedding with `bin/llama-embedding.exe`), with `eval/` as their frozen yardstick.
See the README.
