# Benchmarking Shadow

How speed is measured here, what the numbers mean, how the machine's resources are recorded and charted, and every way a
benchmark on this PC has misled us. [BENCH_NOISE.md](BENCH_NOISE.md) is the original study of the machine's noise;
[STATUS.md](STATUS.md) holds the results.

## The rules (`shadow-bench`)

`crates/shadow-bench` runs arms of `shadow serve` configurations the way a measurement has to be made on this machine:

* **Interleaved.** Run i of every arm before run i+1 of any arm, the order rotated each round, so no arm owns the warm or
  the cold end of the session.
* **One fresh launch per run**, under `tools/run_capped.ps1` (a hard working-set cap and a free-RAM watchdog that kills the
  process below 4 GiB available). The full launch line is recorded in `results.jsonl`.
* **A fixed conversation.** The same user turns in the same order; the server keeps the history, so turn t of every run faces
  the same prompt.
* **Warm turns only.** Turn 0 is dropped from every figure.
* **The floor.** Within an arm, the difference between runs at the same turn. A difference between arms smaller than the
  floor is a tie. Measured on this PC: 0.8% median, 2.0% p90, 3.6% worst for the exact engine
  (`bench/floor-2026-10-27`); about 0.1 to 0.5% for total throughput in the fastest mode.

```bash
shadow-bench run --model MODEL.gguf --out bench/NAME --runs 3 --turns 6 --tokens 96 \
    --arm "a=--ctx 8192 --spec 3" --arm "b=--ctx 8192 --spec 5" \
    [--arm-exe "a=path/to/other/shadow.exe"] [--prompts-file turns.json] [--no-resources]
shadow-bench summary bench/NAME
```

`--arm-launch "name=POWERSHELL"` runs another engine as an arm (`{port}` and `{log}` are filled in): it must serve `/health` and
`/v1/chat/completions` with llama.cpp-style `timings`, and enforce the RAM floor itself (`tools/run_strata.ps1` does).
`--prompts-file F` replaces the six built-in turns with a JSON array of strings. The bench sends `enable_thinking: false`.

## Score by total throughput, not the median

`summary.md` prints the median over a run's warm turns. **That is the wrong headline whenever the arms generate different
text**, which is every comparison involving `--tail-skip` or a different drafter: the arms' turn t is then a different
number of tokens of different content, and a few slow turns swing the median by 20% or more (on the varied bench the
median read 18 to 44 tok/s for arms whose totals were 43 to 47). Use

```bash
python tools/bench_total.py bench/NAME     # sum(predicted_n) / sum(predicted_ms) over the warm turns, per run and per arm
```

which is deterministic run to run (runs agree to 0.1 to 0.3%). `tools/autotune.py` scores this way.

## Two benches

| bench | turns | what it is for | caveat |
|---|---|---|---|
| standard | 6 hash-table follow-ups, 96 tokens each | the project's historical yardstick; short; low noise | ~500 tokens a run on one topic, with a routing profile (`routing-6prompts.shrc`) from other prompts but similar content; which tokens come out moves a run by several percent |
| varied | 36 unseen alpaca / code-alpaca prompts (`bench/dfdata/varied36.json`) | the more representative one | needs its own placement (`profiles/routing-varied300.shrc`, recorded from 300 *other* prompts); with the six-prompt placement its expert misses dominate and identical configs swung 9 to 47 tok/s |

Build a placement for any prompt set with `SHADOW_RECORD_ROUTING=profile.shrc shadow serve ...` and post the prompts (a profile
only helps text like the text it was recorded on, so the test prompts must not be in it).

## Recording the machine

Every run, unless `--no-resources`, samples the PC once a second from launch to shutdown:

| file | source | columns |
|---|---|---|
| `ARM-runN.sys.csv` | PowerShell performance counters | `time, cpu_pct, avail_mb, disk_read_bps, disk_write_bps, pages_per_s` |
| `ARM-runN.gpu.csv` | `nvidia-smi --query-gpu` | timestamp, GPU and memory utilisation, memory used, power draw, temperature, SM and memory clocks, PCIe link gen / width, clock-throttle reasons |
| `ARM-runN.pcie.csv` | `nvidia-smi dmon -s t` | host->GPU (`rxpci`) and GPU->host (`txpci`) MB/s |

`results.jsonl` carries `t_launch`, `t_ready`, `t_done` and each turn's `t_start` / `t_end` (unix seconds), so the samples can
be lined up with the decode windows. Then

```bash
python tools/plot_bench.py bench/NAME        # needs matplotlib
```

writes `bench/NAME/charts/`:

* `ARM-runN.png`: seven stacked panels on one time axis (GPU utilisation, GPU power, GPU memory, PCIe, RAM available, CPU,
  disk read), the warm decode turns shaded, a dashed line where the server became ready;
* `compare.png`: the arms side by side over the warm decode turns of all their runs;
* `resources.csv` and `resources.md`: the same numbers as a table. Beyond means: GPU utilisation p95, **GPU joules per token**
  (mean power x decode time / tokens), SM clock, the fraction of time the GPU was power- or thermal-limited, PCIe rx mean and
  p95 (the expert-copy traffic: the bound this engine lives under), CPU p95, minimum RAM available, disk read, paging.

Reading them: a faster arm should show *why*: higher GPU utilisation with the same PCIe traffic is a better-overlapped
pass; lower PCIe rx for the same tokens is a better expert cache; a lower minimum RAM available is the cost of a bigger tier.
GPU power is board power from `nvidia-smi`, not wall power.

## Shadow against another engine

```bash
powershell -File tools/bench_vs_strata.ps1 -Out bench/vs-strata-varied -Model MODEL.gguf \
    -StrataConfig path/to/strata-bench.json -PromptsFile bench/dfdata/varied36.json -Turns 36 \
    -Placement profiles/routing-varied300.shrc -Runs 3
```

`tools/strata_bench_cfg.py DIR` writes a bench copy of a Strata config: the same engine flags, but Discord, MCP tools, the
conversation store and learned-profile saving off, slots and log to scratch, loopback only, no vision, context 8192, greedy.
Both arms get *their own* best configuration (Shadow its default fastest mode with the placement for the prompts, Strata its
tuned flags and expert profile), the same model file, prompts, context and token budget, and are interleaved. Report the
total throughput and the charts, not the medians.

## Autotune

```bash
python tools/autotune.py --exe PATH/shadow.exe --bench PATH/shadow-bench.exe --out bench/autotune \
    --runs 2 --passes 1 --placement profiles/routing-varied300.shrc \
    "--bench-args=--turns 36 --prompts-file bench/dfdata/varied36.json" "--knobs=--spec,--tail-skip,--dflash-min-p"
```

Coordinate ascent: for each knob, one bench round of the current best against every other value; a value is adopted only
if its total throughput beats the best's by more than max(1%, 3 x the best's own run-to-run spread). A control arm (the
knob left unchanged) is a free noise check. It logs to `OUT/log.md`, writes `OUT/best.json`, and resumes (a finished round
is not rerun). `--space F.json` replaces the knob list. Only flags can be tuned this way (a bench arm cannot set
environment variables).

## Ways a benchmark here has misled us

1. **Medians across arms that generate different text** (above). The "tuned drafter is +10%" result was this.
2. **Anchors in the prompt.** A drafter evaluation that counts positions inside the user's prompt measures something
   drafting never does.
3. **A placement profiled on the test prompts.** The six hash-table prompts' placement makes that bench unusually kind to the
   RAM tier.
4. **The launcher's memory cap.** Arms with no `--ram-tier-gib` flag were launched under a 16 GiB working-set cap and ran
   25% slow against identical explicit flags (fixed: the cap follows the default tier).
5. **Another GPU job.** Never run anything else on the GPU while a bench runs; a data-generation server next to a bench
   skews both (VRAM and PCIe are shared).
6. **A cold first run.** The first arm after a long idle has read slow and noisy (33 against 41 tok/s); interleaving and 3+ runs average it out.
7. **Thinking mode.** Training and testing a drafter on thinking-mode text while the bench sends `enable_thinking: false`
   (or the reverse) measures the wrong distribution.
