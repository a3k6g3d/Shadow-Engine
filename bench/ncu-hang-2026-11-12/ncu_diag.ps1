# ELEVATED diagnostics: why does Nsight Compute hang on the engine's first request?  Each experiment is time-boxed; the engine is RAM-capped (working set + free-RAM floor).
Set-Location J:\Shadow
$sp = 'C:\Users\a3k6g\AppData\Local\Temp\claude\c2eb5aa3e3d3334d\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$log = "$sp\ncu_diag.log"
function Say($t) { "$([DateTime]::Now.ToString('HH:mm:ss')) $t" | Out-File -Append -Encoding utf8 $log }
Remove-Item "$sp\ncu_diag.done", $log -ErrorAction SilentlyContinue
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe'
$m = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Ws4 {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
}
'@
$env:SHADOW_VERIFY_GRAPHS = '0'; $env:SHADOW_GRAPHS = '0'

function Stop-All { Get-Process ncu, shadow -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; Start-Sleep 4 }

function Thread-Dump($pid_, $label) {
  $p = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
  if (-not $p) { Say "${label}: process gone"; return }
  $a = @{}; $p.Threads | ForEach-Object { $a[$_.Id] = $_.TotalProcessorTime.TotalSeconds }
  Start-Sleep 3
  $p.Refresh()
  $rows = $p.Threads | ForEach-Object {
    $d = $_.TotalProcessorTime.TotalSeconds - $(if ($a.ContainsKey($_.Id)) { $a[$_.Id] } else { 0 })
    [pscustomobject]@{ Id = $_.Id; State = $_.ThreadState; Wait = $(if ($_.ThreadState -eq 'Wait') { $_.WaitReason } else { '' }); CpuIn3s = [math]::Round($d, 2); CpuTotal = [math]::Round($_.TotalProcessorTime.TotalSeconds, 1) }
  } | Sort-Object CpuIn3s -Descending | Select-Object -First 6
  Say "$label threads (top by CPU in 3 s): $(($rows | ForEach-Object { "$($_.Id):$($_.State)/$($_.Wait) +$($_.CpuIn3s)s tot $($_.CpuTotal)s" }) -join ' | ')"
}

function Experiment($name, $serverExtra, $ncuExtra) {
  Say "=== ${name}: server extra [$serverExtra], ncu extra [$ncuExtra]"
  Stop-All
  $nargs = @('-o', "$sp\ncu_diag_$name", '-f') + $ncuExtra + @('J:\Shadow\target-dev\release\shadow.exe', 'serve', $m, '--port', '8095', '--ctx', '8192', '--ram-tier-gib', '4') + $serverExtra
  $t0 = Get-Date
  $p = Start-Process -FilePath $ncu -ArgumentList $nargs -PassThru -WindowStyle Hidden -RedirectStandardOutput "$sp\ncu_diag_$name.out" -RedirectStandardError "$sp\ncu_diag_$name.err"
  $child = $null
  for ($i = 0; $i -lt 120 -and -not $child; $i++) {
    Start-Sleep -Milliseconds 500
    $child = Get-CimInstance Win32_Process -Filter "Name='shadow.exe'" | Where-Object { $_.CommandLine -match 'port 8095' } | Select-Object -First 1
  }
  if (-not $child) { Say "${name}: no shadow.exe child"; Stop-All; return }
  $cp = Get-Process -Id $child.ProcessId
  [Ws4]::SetProcessWorkingSetSizeEx($cp.Handle, [UIntPtr]([uint64]256MB), [UIntPtr]([uint64]30 * 1GB), 0x6) | Out-Null
  $up = $false
  for ($i = 0; $i -lt 100; $i++) {
    Start-Sleep 3
    $av = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
    if ($av -lt 4) { Say "${name}: KILLED, available RAM $av GiB"; Stop-All; return }
    try { if ((Invoke-WebRequest http://127.0.0.1:8095/health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $up = $true; break } } catch {}
  }
  Say "${name}: server up $up after $([int]((Get-Date) - $t0).TotalSeconds) s; VRAM used $((nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
  if ($up) {
    $cj = Start-Job -ScriptBlock { python 'C:\Users\a3k6g\AppData\Local\Temp\claude\c2eb5aa3e3d3334d\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad\ncu_diag_client.py' 8095 2>&1 }
    if (Wait-Job $cj -Timeout 150) { Say "${name}: client: $((Receive-Job $cj 2>&1 | Out-String).Trim())" }
    else {
      Say "${name}: NO REPLY after 150 s; GPU util $((nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader))"
      Thread-Dump $cp.Id "$name engine"
      Thread-Dump $p.Id "$name ncu"
      Stop-Job $cj
    }
  }
  Stop-All
  $o = (Get-Content "$sp\ncu_diag_$name.out" -ErrorAction SilentlyContinue | Select-Object -Last 4) -join ' / '
  Say "${name}: ncu stdout tail: $o"
  Say "${name}: report written: $(Test-Path "$sp\ncu_diag_$name.ncu-rep")"
}

$basic = @('--section', 'SpeedOfLight')
# A: one kernel, early, default VRAM (full)
Experiment 'A_q6k_skip0_count1' @() (@('-k', 'regex:matvec_cols_q6_k', '--launch-skip', '0', '--launch-count', '1') + $basic)
# B: the same, but with VRAM left free (a 2 GiB expert cache)
Experiment 'B_freevram' @('--expert-cache-gib', '2') (@('-k', 'regex:matvec_cols_q6_k', '--launch-skip', '0', '--launch-count', '1') + $basic)
# C: a late kernel (skip 400, the decode phase of the one request), default VRAM
Experiment 'C_skip400' @() (@('-k', 'regex:matvec_cols_q6_k', '--launch-skip', '400', '--launch-count', '2') + $basic)
# D: no profiling at all: ncu attached with an empty kernel filter that matches nothing
Experiment 'D_nomatch' @() (@('-k', 'regex:zzz_no_such_kernel', '--launch-count', '1') + $basic)
Say 'all experiments done'
'done' | Out-File "$sp\ncu_diag.done"
