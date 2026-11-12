# Nsight Compute on the decode pass's biggest kernels, for ELEVATED PowerShell (the driver keeps GPU counters admin-only).  STATUS 108.
#
#   powershell -File tools\ncu_decode.ps1 [-Out bench\ncu_decode_stalls] [-Pass 6] [-Kernels 40]
#
# Why it looks like this: `ncu --launch-skip N` makes every skipped matching launch cost ~2 s on this engine (it backs the device memory up each time), so a decode kernel late in a
# run is unreachable and the first request appears to hang.  Instead ncu starts with `--profile-from-start off` and the engine opens the driver's profiler range itself around
# batched pass number -Pass (`SHADOW_NCU_RANGE=Pass:1`, session.rs); nothing before it is intercepted.  Graphs are off so every kernel is a plain launch.
# The engine is capped (working set + free-RAM floor, as tools/run_capped.ps1) and stopped alone at the end, so ncu writes its report.
# Read the report without admin:  ncu --import <Out>.ncu-rep --page details   (or --page raw --csv --metrics ...)
param([string]$Out = 'bench\ncu_decode_stalls', [int]$Pass = 6, [int]$Kernels = 40, [int]$CapGiB = 30, [double]$FloorGiB = 4)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ncu = (Get-ChildItem 'C:\Program Files\NVIDIA Corporation\Nsight Compute*\target\windows-desktop-win7-x64\ncu.exe' | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$model = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$exe = Join-Path $root 'target-dev\release\shadow.exe'
if (Get-Process shadow, shadow-ui -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Shadow*' }) { 'a shadow engine is running - stop it first'; exit 1 }
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class WsNcu {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
}
'@
$env:SHADOW_VERIFY_GRAPHS = '0'; $env:SHADOW_GRAPHS = '0'; $env:SHADOW_NCU_RANGE = "${Pass}:1"
$kern = 'regex:matvec_cols_q6_k|matvec_cols_iq4_xs|matvec_cols_q8_0|matvec_cols_bf16|matvec_grouped_iq4_nl|matvec_grouped_iq2_s|matvec_grouped_iq3_xxs'
$nargs = @('-o', (Join-Path $root $Out), '-f', '--profile-from-start', 'off', '-k', $kern, '--launch-count', "$Kernels",
  '--section', 'SpeedOfLight', '--section', 'WarpStateStats', '--section', 'SchedulerStats', '--section', 'Occupancy', '--section', 'LaunchStats',
  '--section', 'MemoryWorkloadAnalysis', '--section', 'InstructionStats', '--section', 'SourceCounters', '--import-source', 'yes',
  $exe, 'serve', $model, '--port', '8095', '--ctx', '4096', '--ram-tier-gib', '4')
$p = Start-Process -FilePath $ncu -ArgumentList $nargs -PassThru -WindowStyle Hidden -RedirectStandardOutput "$Out.out.txt" -RedirectStandardError "$Out.err.txt"
$child = $null
for ($i = 0; $i -lt 120 -and -not $child; $i++) {
  Start-Sleep -Milliseconds 500
  $child = Get-CimInstance Win32_Process -Filter "Name='shadow.exe'" | Where-Object { $_.CommandLine -match 'port 8095' } | Select-Object -First 1
}
if (-not $child) { 'no engine appeared'; Stop-Process -Id $p.Id -Force; exit 1 }
$cp = Get-Process -Id $child.ProcessId
if (-not [WsNcu]::SetProcessWorkingSetSizeEx($cp.Handle, [UIntPtr]([uint64]256MB), [UIntPtr]([uint64]$CapGiB * 1GB), 0x6)) { Stop-Process -Id $cp.Id -Force; Stop-Process -Id $p.Id -Force; 'cap NOT set - killed'; exit 1 }
$up = $false
for ($i = 0; $i -lt 100 -and -not $up; $i++) {
  Start-Sleep 3
  if ((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024 -lt $FloorGiB) { 'KILLED: free RAM under the floor'; Stop-Process -Id $cp.Id -Force; Stop-Process -Id $p.Id -Force; exit 1 }
  try { $up = (Invoke-WebRequest http://127.0.0.1:8095/health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch {}
}
if ($up) {
  $job = Start-Job -ScriptBlock { python (Join-Path $using:root 'bench\ncu_dec_client.py') 2>&1 }
  if (-not (Wait-Job $job -Timeout 900)) { 'client still waiting after 15 min: stopping it'; Stop-Job $job }
  Receive-Job $job
}
Stop-Process -Id $cp.Id -Force -ErrorAction SilentlyContinue   # ncu writes its report when the target ends
$p.WaitForExit(180000) | Out-Null
"done - report: $(Join-Path $root $Out).ncu-rep (or .ncu-repz)"
