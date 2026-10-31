# Profiles two launches of the sparse attention kernel with Nsight Compute (needs an ELEVATED PowerShell: the driver
# keeps GPU performance counters admin-only).  Output: J:\Shadow\bench\ncu_attn.ncu-rep (read it without admin: ncu --import)
Set-Location J:\Shadow
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe'
$m = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$rep = 'J:/Shadow/bench/ncu_attn'
if (Get-Process shadow-* -ErrorAction SilentlyContinue) { 'a shadow engine is running - stop it first'; exit 1 }
$p = Start-Process -FilePath $ncu -PassThru -WindowStyle Hidden -RedirectStandardOutput 'bench\ncu_attn.txt' -RedirectStandardError 'bench\ncu_attn.err' -ArgumentList @(
  '-o', $rep, '-f', '-k', 'regex:attention_heads_sel_rows', '--launch-skip', '40', '--launch-count', '2',
  '--section', 'SpeedOfLight', '--section', 'WarpStateStats', '--section', 'Occupancy', '--section', 'LaunchStats', '--section', 'MemoryWorkloadAnalysis',
  'J:\Shadow\target\release\shadow-ap.exe', 'serve', $m, '--port', '8095', '--ctx', '8192', '--placement', 'J:/Shadow/profiles/routing-varied300.shrc', '--ram-tier-gib', '2')
$ok = $false
for ($i = 0; $i -lt 60; $i++) { Start-Sleep 3; try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8095/health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {} }
if (-not $ok) { 'server never became ready'; Get-Process shadow-*, ncu -ErrorAction SilentlyContinue | Stop-Process -Force; exit 1 }
python 'C:\Users\a3k6g\AppData\Local\Temp\claude\J--Shadow\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad\attcmp.py'
Start-Sleep 3
Get-Process shadow-* -ErrorAction SilentlyContinue | Stop-Process -Force   # ncu writes its report when the target ends
$p.WaitForExit(120000) | Out-Null
'done - report in J:\Shadow\bench\ncu_attn.ncu-rep'
