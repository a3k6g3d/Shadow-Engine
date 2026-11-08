# Nsight Compute on the prompt path's grouped MoE kernels as they are now (width classes 16/32/64, gate+up fused "gu" kernels, down projection), 2026-11-08.
# Needs an ELEVATED PowerShell (the driver keeps GPU performance counters admin-only).  ~4-6 minutes in all, two server runs, ~9.5k-token prompt each.
# Output: J:\Shadow\bench\ncu_moe_gu.ncu-rep and ncu_moe_down.ncu-rep (read them without admin: ncu --import FILE --page details).
# Memory: engine on the 4 GiB RAM tier; a watchdog in this script kills the engine and ncu if available RAM falls under 4 GiB (the repo's floor).
Set-Location J:\Shadow
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe'
$m = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
if (Get-Process shadow-* -ErrorAction SilentlyContinue) { 'a shadow-* engine is running - stop it first'; exit 1 }
Copy-Item J:\Shadow\target\release\shadow.exe J:\Shadow\target\release\shadow-ap.exe -Force   # a copy, so a rebuild never meets a locked file and the user's shadow.exe is never touched
function Avail { (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024 }
function Profile($kernel, $skip, $count, $out) {
  $p = Start-Process -FilePath $ncu -PassThru -WindowStyle Hidden -RedirectStandardOutput "bench\$out.txt" -RedirectStandardError "bench\$out.err" -ArgumentList @(
    '-o', "J:/Shadow/bench/$out", '-f', '-k', "regex:$kernel", '--launch-skip', "$skip", '--launch-count', "$count",
    '--section', 'SpeedOfLight', '--section', 'WarpStateStats', '--section', 'Occupancy', '--section', 'LaunchStats', '--section', 'MemoryWorkloadAnalysis', '--section', 'ComputeWorkloadAnalysis', '--section', 'SchedulerStats', '--section', 'InstructionStats', '--section', 'SourceCounters',
    'J:\Shadow\target\release\shadow-ap.exe', 'serve', $m, '--port', '8095', '--ctx', '24576', '--placement', 'J:/Shadow/profiles/routing-varied300.shrc', '--ram-tier-gib', '4')
  $ok = $false
  for ($i = 0; $i -lt 100; $i++) {
    Start-Sleep 3
    if ((Avail) -lt 4) { Get-Process shadow-*, ncu -ErrorAction SilentlyContinue | Stop-Process -Force; 'KILLED: available RAM under 4 GiB'; exit 1 }
    try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8095/health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {}
  }
  if (-not $ok) { 'server never became ready'; Get-Process shadow-*, ncu -ErrorAction SilentlyContinue | Stop-Process -Force; exit 1 }
  $job = Start-Job { python J:\Shadow\bench\ncu_client_p9k.py }
  while ($job.State -eq 'Running') {
    Start-Sleep 3
    if ((Avail) -lt 4) { Get-Process shadow-*, ncu -ErrorAction SilentlyContinue | Stop-Process -Force; 'KILLED: available RAM under 4 GiB'; exit 1 }
  }
  Receive-Job $job
  Start-Sleep 3
  Get-Process shadow-* -ErrorAction SilentlyContinue | Stop-Process -Force   # ncu writes its report when the target ends
  $p.WaitForExit(240000) | Out-Null
  "done - $out"
}
# gate+up (fused) kernels of every width class, then the down projections (mma_grouped, 16, 48, 64 - not the gu names)
Profile '.*mma_grouped.*_gu_.*' 60 12 'ncu_moe_gu'
Profile '.*mma_grouped(16|48|64)?_(q6_k|iq4_xs|iq4_nl|q2_0|iq2_s|iq3_xxs|iq3_s)' 60 12 'ncu_moe_down'
'all done'
