# Runs the Strata server (its own engine, the bench config from tools/strata_bench_cfg.py) under the RAM floor: the whole
# process tree is killed if available RAM falls below 4 GiB.  Prints one closing line like run_capped.ps1.  shadow-bench
# starts it as an `--arm-launch` ({port} and {log} filled in):
#
#   $L = "& 'tools/run_strata.ps1' -Port {port} -Out '{log}' -Config 'path/to/strata-bench.json'"
#   shadow-bench run ... --arm "strata=-" --arm-launch "strata=$L"
param([double]$FloorGiB = 4, [int]$Port = 8091, [string]$Out = 'strata.txt', [string]$Config, [string]$StrataDir = 'J:\Strata-work\Strata-0140')
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$t0 = Get-Date
$p = Start-Process -FilePath python -ArgumentList @('-u', '-m', 'serve.server', '--engine', 'strata', '--config', $Config, '--port', "$Port") `
    -WorkingDirectory $StrataDir -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -WindowStyle Hidden -PassThru
$min = 999
while (-not $p.HasExited) {
  $av = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
  if ($av -lt $min) { $min = $av }
  if ($FloorGiB -gt 0 -and $av -lt $FloorGiB) { taskkill /T /F /PID $p.Id | Out-Null; "KILLED at {0:N2} GiB available" -f $av; break }
  Start-Sleep -Milliseconds 1000
}
"exit $($p.ExitCode), wall {0:N0} s, min available {1:N1} GiB (strata, tree floor 4 GiB)" -f ((Get-Date) - $t0).TotalSeconds, $min
