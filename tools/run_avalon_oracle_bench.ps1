# Avalon zero-cost oracle bound (EXPERIMENT_SPEC_2026-11-10, step 0)
# Arms differ ONLY by --avalon-oracle N (N experts a layer left out of the staged copy: timing only, output wrong).
# 4 arms x Runs, interleaved, one fresh launch per run, warm turns only.  Upper bound for any "compute the miss on the CPU" design.
param(
  [string]$Model = "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
  [string]$Placement = "J:/Shadow/profiles/routing-varied300.shrc",
  [string]$Bench = "target/release/shadow-bench.exe",
  [string]$Out = "bench/avalon_oracle_prompt",
  [int]$Runs = 3,
  [int]$Turns = 6,
  [int]$Port = 8092,
  [int[]]$Ns = @(0, 3, 25, 50)
)
$ErrorActionPreference = "Stop"
$base = "--ctx 16384 --ram-tier-gib 36 --ram-floor-gib 3 --placement $Placement"
$benchArgs = @("run", "--model", $Model, "--out", $Out, "--runs", "$Runs", "--turns", "$Turns", "--tokens", "16",
               "--port", "$Port", "--prompts-file", "bench/dfdata/prompt_heavy.json")
foreach ($n in $Ns) {
  $flags = if ($n -eq 0) { $base } else { "$base --avalon-oracle $n" }
  $benchArgs += @("--arm", "oracle$n=$flags")
}
& $Bench @benchArgs
python tools/bench_total.py $Out
python tools/bench_prompt_and_decode.py $Out
