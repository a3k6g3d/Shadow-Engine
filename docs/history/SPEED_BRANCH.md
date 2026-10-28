# Speed branch: Q4_0 KV, decode attention, adaptive drafts, tool-call streaming

Changes on top of this fork's `best`. Measured on an RTX 5070 12 GB, i5-10600K, 64 GB RAM, CUDA 13.4; the kernel
numbers come from the repo's parity tests (`qsa_prompt_attn_parity`, `kv_q4_parity`, `draft_policy_test`).

## What changed

- **`--kv q4_0` prompt attention** ([`qsa_prompt_attn.cu`](../src/kernels/cuda/qsa_prompt_attn.cu)). Q4_0 now runs a
  pipelined kernel like int8's (each warp gathers its two q4_0 blocks per row with `cp.async`, one chunk ahead):
  10.2 -> 5.6 ms per 2,048-query chunk at 64K context (int8: 6.5 ms). The queries' and the output's Hadamard rotations
  run inside the kernel (`qsa_prompt_attn_batch_rot`, bit-identical to rotate -> attend -> rotate, 3-7% faster).
  `STRATA_PROMPT_ATTN_V1=1` brings back the old kernel, `STRATA_PROMPT_ATTN_ROT_FUSED=0` the separate rotation passes.
- **Decode attention on tensor cores** (`qsa_decode_attn_tc`, int8 and q4_0 KV, sm_80+; `STRATA_DECODE_ATTN_TC=0` turns
  it off). A verify window's few queries are split over their selections to fill the GPU: a 5-token window's attention
  takes 0.033 ms instead of 0.098 ms per layer. FP32-level accuracy, not bitwise equal to `qsa_decode_attn_batch`.
  Safe inside CUDA graph capture (tested); non-resident KV blocks are masked like the old kernel's (tested).
- **Verify windows with Q4_0 KV** take the batched path (it was kept off Q4_0 from before the batched queries were
  rotated), and a window's Q4_0 cells are appended in one launch (`kv_append_q4_steps`, same bytes, tested). Not in a
  batch of slots (`batch_rec_`), where every row has its own K/V state.
- **`--spec-adaptive`** (opt-in). The MTP window's length by expected committed tokens per millisecond from the
  measured round costs and the acceptance by draft probability (`DraftPolicy::choose_mtp`), instead of
  `--spec-min-p`'s threshold. **Not a win on the PC above with an SSD-bound model:** simulated +3% to +11%
  (`draft_policy_test`, steady round costs), but Qwen3.8-Flash-Next Q4_K_M with 38 GiB of experts in RAM and the rest
  read from a SATA SSD gave a median 3.2 tok/s against 4.6 tok/s for `--spec-min-p 0.5` (round costs swing 300-700 ms
  with the SSD, the policy learns that and drafts too little). Leave it off unless every expert is in RAM or VRAM.
- **Tool-call streaming in the server** (`serve/frontend.py`): `call_end` resumes where the previous call stopped, so
  scanning a long call is linear. A 53K-token, 212 KB file write: 7.4 s -> 0.46 s of server time; the events are
  identical to the old parser's over 6,000 randomized streamed parses (`LinearToolCallScan` in
  `serve/test_server.py`).
- **`tools/mtp_from_gguf.py`**: the MTP draft block from a llama.cpp GGUF that carries it (finetunes and abliterations
  whose head ships only in the GGUF), in the form `tools/mtp_fetch.py` writes, so `mtp_pack.py` and `mtp_rt.py` build
  the runtime files from it.

## Where a decode window's time goes, and what moved it (RTX 5070 12 GB, i5-10600K, 64 GB DDR4-2666, NVMe, IQ4_XS)

`STRATA_DECODE_TIMING=1` splits a ~115 ms window (about 2.2 tokens) into: CPU running the RAM-tier experts 52 ms, waiting for
the NVMe-tier experts 26 ms ("jobs"), waiting for the GPU 11 ms, the rest ~25 ms. Held-out prompts, three passes per
variant on a fresh server, the first pass dropped; the control repeated at the end drifted +1%:

| Change | tok/s | Verdict |
| --- | ---: | --- |
| baseline (`--vram-reserve-mib 1022`, 4 GiB RAM headroom) | 14.9 | |
| `--vram-reserve-mib 640` (the desktop idles at 567 MiB; +150 expert slots) | 15.6 | +5% |
| `STRATA_RESIDENT_HEADROOM_GIB=2` with `--resident-budget-gib 48` (pins 45.8 GiB, not 43.5) | 17.1 | +10%, **unsafe**: froze the PC once free RAM fell to 0.4 GiB; keep the default 4 GiB |
| pinned RAM 36 / 40 / 44 / 45.5 GiB (NVMe wait 69 / 42 / 23 / 16 ms per window) | 10.7 / 13.2 / 15.5 / 17.1 | about 0.7 tok/s per GiB |
| link-time optimisation + `CMAKE_CUDA_ARCHITECTURES=120-real` + tests off | 14.8 vs 14.8 | tie |
| `--pool-workers` 4 / 6 (default 5) | 14.7 / 14.6 vs 14.8 | tie |
| `--kv q4_0` with the freed VRAM | 15.5 vs 15.6 | tie (about 80 more slots) |
| `--resident-budget-gib` above what the free RAM allows (50, 54) | no change | the engine caps the pin at *available RAM minus the headroom* |

The router lookahead (`STRATA_LOOKAHEAD`) is off with unbuffered reads (`FileExpertSource::warms()`); filling the stage
buffers from it instead of warming pages was tried and reverted: the NVMe wait did not shrink (25.6-27.7 ms vs 25.7) and
the CPU work grew (55-56 ms vs 52), so K=6/10/14 gave 15.4 / 14.6 / 13.4 tok/s against 15.0 with it off.

The remaining lever is more of the model in RAM: every GiB of pinned experts removes 3-5 ms of NVMe wait per window.
With about 14 GiB held by other programs, closing them is worth more than any setting here. The page-locked tier
cannot be paged out, so the headroom is the only margin for programs that grow after the engine starts.

On this PC the link first negotiated PCIe 2.0. Set to 3.0 in the BIOS, it measured 7-10 GB/s but logged about 230
link replays per second at idle (`nvidia-smi -q`, "Replays Since Reset"), and the PC hard-hung twice under decode
load, with no bugcheck. Check that counter before trusting a link speed. A link that replays is not stable.

## Chat and tools

The MCP permission modes, the no-read list, the exec tools, long-chat summarizing and queued messages are described in
[DETAILS.md](DETAILS.md#tools-from-mcp-servers); what changed and when is in [CHANGELOG.md](../CHANGELOG.md).

## Not measured

No end-to-end tokens/s for the kernel changes: the model on this PC is SSD-bound (about 30 GB of reads per 256-token
reply), which hides GPU-side gains. Kernel timings are the evidence.
