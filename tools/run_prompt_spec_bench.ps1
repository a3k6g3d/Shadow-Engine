# Prompt-Side Speculation Benchmark (EXPERIMENT_SPEC_2026-11-04)
# 4 arms x 3 runs, interleaved, fresh launches, warm turns only
param(
  [string]$Model = "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
  [string]$Placement = "J:/Shadow/profiles/routing-varied300.shrc",
  [string]$Bench = "target/release/shadow-bench.exe",
  [int]$Port = 8092
)

$ErrorActionPreference = "Stop"

Write-Host "=========================================================="
Write-Host "STARTING BENCHMARK 1: PROMPT-HEAVY WORKLOAD (Arms A & B)"
Write-Host "=========================================================="
$heavyOut = "bench/prompt_spec_heavy"
$heavyArgs = @(
  "run",
  "--model", $Model,
  "--out", $heavyOut,
  "--runs", "3",
  "--turns", "6",
  "--tokens", "16",
  "--port", "$Port",
  "--prompts-file", "bench/dfdata/prompt_heavy.json",
  "--arm", "armA_spec_off=--ctx 16384 --ram-tier-gib 36 --ram-floor-gib 3 --dflash off --spec 0 --placement $Placement",
  "--arm", "armB_spec_on=--ctx 16384 --ram-tier-gib 36 --ram-floor-gib 3 --spec 3 --placement $Placement"
)
& $Bench @heavyArgs

Write-Host "=========================================================="
Write-Host "STARTING BENCHMARK 2: MIXED WORKLOAD (Arms C & D)"
Write-Host "=========================================================="
$mixedOut = "bench/prompt_spec_mixed"
$mixedArgs = @(
  "run",
  "--model", $Model,
  "--out", $mixedOut,
  "--runs", "3",
  "--turns", "36",
  "--tokens", "96",
  "--port", "$Port",
  "--prompts-file", "bench/dfdata/varied36.json",
  "--arm", "armC_spec_off=--ctx 8192 --ram-tier-gib 36 --ram-floor-gib 3 --dflash off --spec 0 --placement $Placement",
  "--arm", "armD_spec_on=--ctx 8192 --ram-tier-gib 36 --ram-floor-gib 3 --spec 3 --placement $Placement"
)
& $Bench @mixedArgs

Write-Host "=========================================================="
Write-Host "SCORING BENCHMARKS"
Write-Host "=========================================================="
python tools/bench_total.py $heavyOut $mixedOut
python tools/bench_prompt_and_decode.py $heavyOut $mixedOut
Write-Host "ALL ARMS COMPLETE!"
