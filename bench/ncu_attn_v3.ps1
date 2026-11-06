# Profiles the sparse attention (`attention_group_sel_rows_v3`), the dense float attention and the 512-thread DeltaNet recurrence with Nsight Compute, one server run each (~3 minutes each).
# Needs an ELEVATED PowerShell: the driver keeps GPU performance counters admin-only.  Output: J:\Shadow\bench\ncu_attn_sel.ncu-rep and ncu_attn_dense.ncu-rep
# (read them without admin: ncu --import FILE --page details).  Uses a copy of the engine (target\release\shadow-ap.exe) on a 4 GiB RAM tier.
Set-Location J:\Shadow
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe'
$m = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
if (Get-Process shadow-* -ErrorAction SilentlyContinue) { 'a shadow-* engine is running - stop it first'; exit 1 }
function Profile($kernel, $skip, $out) {
  $p = Start-Process -FilePath $ncu -PassThru -WindowStyle Hidden -RedirectStandardOutput "bench\$out.txt" -RedirectStandardError "bench\$out.err" -ArgumentList @(
    '-o', "J:/Shadow/bench/$out", '-f', '-k', "regex:$kernel", '--launch-skip', "$skip", '--launch-count', '3',
    '--section', 'SpeedOfLight', '--section', 'WarpStateStats', '--section', 'Occupancy', '--section', 'LaunchStats', '--section', 'MemoryWorkloadAnalysis', '--section', 'ComputeWorkloadAnalysis', '--section', 'SchedulerStats', '--section', 'InstructionStats', '--section', 'SourceCounters',
    'J:\Shadow\target\release\shadow-ap.exe', 'serve', $m, '--port', '8095', '--ctx', '24576', '--placement', 'J:/Shadow/profiles/routing-varied300.shrc', '--ram-tier-gib', '4')
  $ok = $false
  for ($i = 0; $i -lt 100; $i++) { Start-Sleep 3; try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8095/health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {} }
  if (-not $ok) { 'server never became ready'; Get-Process shadow-*, ncu -ErrorAction SilentlyContinue | Stop-Process -Force; exit 1 }
  python J:\Shadow\bench\ncu_client_p9k.py
  Start-Sleep 3
  Get-Process shadow-* -ErrorAction SilentlyContinue | Stop-Process -Force   # ncu writes its report when the target ends
  $p.WaitForExit(240000) | Out-Null
  "done - $out"
}
Profile 'attention_group_sel_rows_v3' 200 'ncu_k_attn_v3'
Profile 'attention_group_dense_rows_f32' 100 'ncu_k_attn_dense_f32'
Profile 'delta_steps128_f4' 20 'ncu_k_delta_f4'
'all done'
