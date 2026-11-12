# ELEVATED, round 3.  Rounds 1-2: Nsight Compute's --launch-skip makes every skipped matching launch cost ~2 s (it backs up the device memory each time), so a decode kernel late in a
# run is unreachable; profiling the very first launch works.  Fix under test: --profile-from-start off, and the engine opens the driver's profiler range itself around a batched
# pass (SHADOW_NCU_RANGE=from:n), so nothing before it is intercepted.  The engine is stopped alone at the end (ncu then writes its report).  No system setting is changed.
Set-Location J:\Shadow
$sp = 'C:\Users\a3k6g\AppData\Local\Temp\claude\c2eb5aa3e3d3334d\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$log = "$sp\ncu_diag3.log"
function Say($t) { "$([DateTime]::Now.ToString('HH:mm:ss')) $t" | Out-File -Append -Encoding utf8 $log }
Remove-Item "$sp\ncu_diag3.done", $log -ErrorAction SilentlyContinue
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe'
$m = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Ws6 {
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
  $nargs = @('-o', "$sp\ncu_diag3_$name", '-f') + $ncuExtra + @('J:\Shadow\target-dev\release\shadow.exe', 'serve', $m, '--port', '8095', '--ctx', '4096', '--ram-tier-gib', '4') + $serverExtra
  $t0 = Get-Date
  $p = Start-Process -FilePath $ncu -ArgumentList $nargs -PassThru -WindowStyle Hidden -RedirectStandardOutput "$sp\ncu_diag3_$name.out" -RedirectStandardError "$sp\ncu_diag3_$name.err"
  foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
  $child = $null
  for ($i = 0; $i -lt 120 -and -not $child; $i++) {
    Start-Sleep -Milliseconds 500
    $child = Get-CimInstance Win32_Process -Filter "Name='shadow.exe'" | Where-Object { $_.CommandLine -match 'port 8095' } | Select-Object -First 1
  }
  if (-not $child) { Say "${name}: no shadow.exe child"; Stop-All; return }
  $cp = Get-Process -Id $child.ProcessId
  [Ws6]::SetProcessWorkingSetSizeEx($cp.Handle, [UIntPtr]([uint64]256MB), [UIntPtr]([uint64]30 * 1GB), 0x6) | Out-Null
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
  $p.WaitForExit(180000) | Out-Null
  Stop-All
  $o = (Get-Content "$sp\ncu_diag3_$name.out" -ErrorAction SilentlyContinue | Select-Object -Last 3) -join ' / '
  Say "${name}: ncu stdout tail: $o"
  $err = (Get-Content "$sp\ncu_diag3_$name.err" -ErrorAction SilentlyContinue | Select-String 'ncu range' | Select-Object -First 3) -join ' / '
  Say "${name}: engine said: $err"
  Say "${name}: report written: $((Get-ChildItem "$sp\ncu_diag3_$name.ncu-rep*" -ErrorAction SilentlyContinue | Select-Object -First 1).Name)"
}

$full = 'J:\Shadow\bench\ncu_dec_client.py'
$kern = 'regex:matvec_cols_q6_k|matvec_cols_iq4_xs|matvec_cols_q8_0|matvec_cols_bf16|matvec_grouped_iq4_nl|matvec_grouped_iq2_s|matvec_grouped_iq3_xxs'
# G1: the engine's profiler range around one batched pass (#6), SpeedOfLight only
Experiment 'G1_range_sol' @() @('--profile-from-start', 'off', '-k', $kern, '--launch-count', '12', '--section', 'SpeedOfLight') $full 200 @{ SHADOW_NCU_RANGE = '6:1' }
# G2: the real profile: stall sections plus source counters, 40 kernels of that pass
Experiment 'G2_range_stalls' @() @('--profile-from-start', 'off', '-k', $kern, '--launch-count', '40', '--section', 'SpeedOfLight', '--section', 'WarpStateStats', '--section', 'SchedulerStats', '--section', 'Occupancy', '--section', 'LaunchStats', '--section', 'MemoryWorkloadAnalysis', '--section', 'InstructionStats', '--section', 'SourceCounters', '--import-source', 'yes') $full 600 @{ SHADOW_NCU_RANGE = '6:1' }
Say 'all experiments done'
'done' | Out-File "$sp\ncu_diag3.done"
