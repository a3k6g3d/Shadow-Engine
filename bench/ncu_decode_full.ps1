# Profiles the decode pass's biggest kernels with Nsight Compute (ELEVATED PowerShell: the driver keeps GPU counters admin-only).
# Output: J:\Shadow\bench\ncu_decode_full.ncu-rep (read it without admin: ncu --import ...)
Set-Location J:\Shadow
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe'
$m = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$rep = 'J:/Shadow/bench/ncu_decode_full'
if (Get-Process shadow-* -ErrorAction SilentlyContinue) { 'a shadow engine is running - stop it first'; exit 1 }
$kern = 'regex:matvec_cols_q6_k|matvec_cols_q5_k|matvec_cols_iq4_xs|matvec_cols_q8_0|matvec_grouped_iq4_nl|matvec_grouped_iq2_s|matvec_grouped_iq3_xxs'
$p = Start-Process -FilePath $ncu -PassThru -WindowStyle Hidden -RedirectStandardOutput 'bench\ncu_decode_full.txt' -RedirectStandardError 'bench\ncu_decode_full.err' -ArgumentList @(
  '-o', $rep, '-f', '-k', $kern, '--launch-skip', '1500', '--launch-count', '14', '--set', 'full',
  'J:\Shadow\target\release\shadow-ap.exe', 'serve', $m, '--port', '8095', '--ctx', '8192', '--placement', 'J:/Shadow/profiles/routing-varied300.shrc', '--ram-tier-gib', '2')
$ok = $false
for ($i = 0; $i -lt 60; $i++) { Start-Sleep 3; try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8095/health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {} }
if (-not $ok) { 'server never became ready'; Get-Process shadow-*, ncu -ErrorAction SilentlyContinue | Stop-Process -Force; exit 1 }
python 'J:\Shadow\bench\ncu_dec_client.py'
Start-Sleep 3
Get-Process shadow-* -ErrorAction SilentlyContinue | Stop-Process -Force   # ncu writes its report when the target ends
$p.WaitForExit(240000) | Out-Null
'done - report in J:\Shadow\bench\ncu_decode_full.ncu-rep'
