# Shadow against Strata on the same bench, same prompts, same machine, one interleaved run: scored by total throughput
# (tools/bench_total.py) and charted with the machine's resource use (tools/plot_bench.py).  See docs/BENCHMARKING.md.
#
#   powershell -File tools/bench_vs_strata.ps1 -Out bench/vs-strata-varied -Runs 3 `
#       -Model J:/Strata-work/models/.../model.gguf -StrataConfig path/to/strata-bench.json `
#       -PromptsFile bench/dfdata/varied36.json -Turns 36 -Placement profiles/routing-varied300.shrc
param(
  [Parameter(Mandatory = $true)] [string]$Out,
  [Parameter(Mandatory = $true)] [string]$Model,
  [Parameter(Mandatory = $true)] [string]$StrataConfig,
  [int]$Runs = 3,
  [string]$PromptsFile = '',
  [int]$Turns = 6,
  [string]$Placement = 'profiles/routing-6prompts.shrc',
  [string]$ShadowFlags = '',
  [string]$ShadowExe = 'target/release/shadow.exe',
  [string]$Bench = 'target/release/shadow-bench.exe'
)
Set-Location (Split-Path $PSScriptRoot -Parent)
$L = "& '$PSScriptRoot\run_strata.ps1' -Port {port} -Out '{log}' -Config '$StrataConfig'"
$a = @('run', '--model', $Model, '--out', $Out, '--runs', "$Runs", '--turns', "$Turns")
if ($PromptsFile) { $a += '--prompts-file'; $a += $PromptsFile }
$a += '--arm'; $a += "shadow=--ctx 8192 --placement $Placement $ShadowFlags"
$a += '--arm'; $a += 'strata=-'
$a += '--arm-exe'; $a += "shadow=$ShadowExe"
$a += '--arm-launch'; $a += "strata=$L"
& $Bench @a
python tools/bench_total.py $Out
python tools/plot_bench.py $Out
