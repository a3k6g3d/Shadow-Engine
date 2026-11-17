# ShadowZero Examples

A few small, standalone examples showing how to run ShadowZero on one or two GPUs and talk to its HTTP API from Python or PowerShell. Everything here uses only the Python standard library — no `openai` package or extra dependencies required.

## Files

- [`quickstart_single_and_dual_gpu.ps1`](quickstart_single_and_dual_gpu.ps1) — Probe your GPU/PCIe bandwidth (`shadowzero gpus`) and launch `shadowzero serve` on one card (`--gpus 0`), two cards (`--gpus 0,1`), or bit-exact mode (`--exact`).
- [`chat_and_streaming.py`](chat_and_streaming.py) — Send normal and SSE-streaming requests to `/v1/chat/completions`, toggle `<think>` reasoning off with `chat_template_kwargs`, and read per-request `timings`.
- [`agent_tool_loop.py`](agent_tool_loop.py) — Run a small local tool-calling agent loop using ShadowZero's built-in tools (`/v1/tools` + `/v1/tools/execute`) and the `finish` closing tool.
- [`watch_telemetry.py`](watch_telemetry.py) — Live terminal monitor polling `/v1/telemetry` to print per-GPU utilization and VRAM for both cards, expert-cache hit rate, RAM tier activity, and tokens/sec.

## Running any Python example

First start ShadowZero (either via `shadowzero-ui.exe`, Start menu **ShadowZero (server only)**, or CLI):

```powershell
.\target\release\shadowzero.exe serve J:\Models\your-model --gpus 0,1 --port 8081 --power-limit 5070=200,5060=150
```

Then run whichever script you want:

```powershell
python examples\chat_and_streaming.py
python examples\agent_tool_loop.py
python examples\watch_telemetry.py
```

You can point any script at a different host/port by setting `SHADOWZERO_URL` (defaults to `http://127.0.0.1:8081`).
