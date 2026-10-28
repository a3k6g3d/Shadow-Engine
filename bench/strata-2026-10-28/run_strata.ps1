# Runs the Strata server (its own engine, the bench config) under the RAM floor: the whole process tree is killed if
# available RAM falls below 4 GiB.  Prints one closing line like run_capped.ps1.
param([int]$Port = 8091, [string]$Out = 'strata.txt', [string]$Config)
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$t0 = Get-Date
$p = Start-Process -FilePath python -ArgumentList @('-u', '-m', 'serve.server', '--engine', 'strata', '--config', $Config, '--port', "$Port") `
    -WorkingDirectory 'J:\Strata-work\Strata-0140' -RedirectStandardOutput $Out -RedirectStandardError "$Out.err" -WindowStyle Hidden -PassThru
$min = 999
while (-not $p.HasExited) {
  $av = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
  if ($av -lt $min) { $min = $av }
  if ($av -lt 4) { taskkill /T /F /PID $p.Id | Out-Null; "KILLED at {0:N2} GiB available" -f $av; break }
  Start-Sleep -Milliseconds 1000
}
"exit $($p.ExitCode), wall {0:N0} s, min available {1:N1} GiB (strata, tree floor 4 GiB)" -f ((Get-Date) - $t0).TotalSeconds, $min
