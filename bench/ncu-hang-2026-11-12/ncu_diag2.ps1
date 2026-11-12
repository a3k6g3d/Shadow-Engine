# ELEVATED, round 2.  Round 1: profiling the FIRST matvec_cols_q6_k launch works (9 passes, 10-14 s), ncu attached with nothing to profile works, but profiling a LATER launch
# (skip 400) hangs with ncu spinning on one thread and the engine and GPU idle.  Which launch is the first bad one, and is it STATUS 107's reordering (shared expert queued
# between the logits' event record and the host's wait)?  The engine is stopped alone at the end (ncu then writes its report).  Nothing here changes a system setting.
Set-Location J:\Shadow
$sp = 'C:\Users\a3k6g\AppData\Local\Temp\claude\c2eb5aa3e3d3334d\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$log = "$sp\ncu_diag2.log"
function Say($t) { "$([DateTime]::Now.ToString('HH:mm:ss')) $t" | Out-File -Append -Encoding utf8 $log }
Remove-Item "$sp\ncu_diag2.done", $log -ErrorAction SilentlyContinue
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe'
$m = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Ws5 {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
}
'@
$env:SHADOW_VERIFY_GRAPHS = '0'; $env:SHADOW_GRAPHS = '0'
function Stop-All { Get-Process ncu, shadow -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; Start-Sleep 4 }

function Experiment($name, $serverExtra, $ncuExtra, $client, $waitSec, $envs) {
  Say "=== ${name}: server extra [$serverExtra], env [$(($envs.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')]"
  Stop-All
  foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
  $nargs = @('-o', "$sp\ncu_diag2_$name", '-f') + $ncuExtra + @('J:\Shadow\target-dev\release\shadow.exe', 'serve', $m, '--port', '8095', '--ctx', '4096', '--ram-tier-gib', '4') + $serverExtra
  $t0 = Get-Date
  $p = Start-Process -FilePath $ncu -ArgumentList $nargs -PassThru -WindowStyle Hidden -RedirectStandardOutput "$sp\ncu_diag2_$name.out" -RedirectStandardError "$sp\ncu_diag2_$name.err"
  foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
  $child = $null
  for ($i = 0; $i -lt 120 -and -not $child; $i++) {
    Start-Sleep -Milliseconds 500
    $child = Get-CimInstance Win32_Process -Filter "Name='shadow.exe'" | Where-Object { $_.CommandLine -match 'port 8095' } | Select-Object -First 1
  }
  if (-not $child) { Say "${name}: no shadow.exe child"; Stop-All; return }
  $cp = Get-Process -Id $child.ProcessId
  [Ws5]::SetProcessWorkingSetSizeEx($cp.Handle, [UIntPtr]([uint64]256MB), [UIntPtr]([uint64]30 * 1GB), 0x6) | Out-Null
  $up = $false
  for ($i = 0; $i -lt 100; $i++) {
    Start-Sleep 3
    $av = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
    if ($av -lt 4) { Say "${name}: KILLED, available RAM $av GiB"; Stop-All; return }
    try { if ((Invoke-WebRequest http://127.0.0.1:8095/health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $up = $true; break } } catch {}
  }
  Say "${name}: server up $up after $([int]((Get-Date) - $t0).TotalSeconds) s; VRAM used $((nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
  if ($up) {
    $t1 = Get-Date
    $cj = Start-Job -ScriptBlock { param($c) python $c 8095 2>&1 } -ArgumentList $client
    if (Wait-Job $cj -Timeout $waitSec) { Say "${name}: client finished in $([int]((Get-Date) - $t1).TotalSeconds) s: $((Receive-Job $cj 2>&1 | Out-String).Trim() -replace '\s+', ' ')" }
    else { Say "${name}: NO REPLY after $waitSec s; GPU util $((nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader))"; Stop-Job $cj }
  }
  # stop the engine alone: ncu writes its report when the target ends
  Stop-Process -Id $cp.Id -Force -ErrorAction SilentlyContinue
  $p.WaitForExit(120000) | Out-Null
  Stop-All
  $o = (Get-Content "$sp\ncu_diag2_$name.out" -ErrorAction SilentlyContinue | Select-Object -Last 3) -join ' / '
  Say "${name}: ncu stdout tail: $o"
  Say "${name}: report written: $(Test-Path "$sp\ncu_diag2_$name.ncu-rep")"
}

$short = "$sp\ncu_diag_client.py"
$q6 = @('-k', 'regex:matvec_cols_q6_k', '--section', 'SpeedOfLight')
$none = @{}
# F1: the failing launch (skip 400), STATUS 107's reordering switched off
Experiment 'F1_skip400_reorder_off' @() ($q6 + @('--launch-skip', '400', '--launch-count', '2')) $short 100 @{ SHADOW_SHARED_EARLY = '0'; SHADOW_FETCH_FIRST = '0' }
# F2: where does it start to fail: launch 20, then 100 (default settings)
Experiment 'F2_skip20' @() ($q6 + @('--launch-skip', '20', '--launch-count', '1')) $short 100 $none
Experiment 'F3_skip100' @() ($q6 + @('--launch-skip', '100', '--launch-count', '1')) $short 100 $none
# E1: the failing launch with a small VRAM footprint (a 0.5 GiB expert cache)
Experiment 'E1_small_skip400' @('--expert-cache-gib', '0.5') ($q6 + @('--launch-skip', '400', '--launch-count', '2')) $short 100 $none
Say 'all experiments done'
'done' | Out-File "$sp\ncu_diag2.done"
