param([string]$Tag, [string]$Flags = '', [string]$Client = 'J:\Shadow\tools\nsys_decode_client.py', [string]$Model = 'J:/Models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-2bpw', [int]$CapGiB = 56, [double]$FloorGiB = 3.5)
# Decode under Nsight Systems: nsys starts shadowzero.exe; this script caps the shadowzero.exe child's working set and kills it below the free-RAM floor
# (the rule of tools/run_capped.ps1, applied to the child because run_capped would cap nsys itself).
Set-Location J:\Shadow
$sp = $env:TEMP
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.2\target-windows-x64\nsys.exe'
$model = $Model
$rep = "J:\Shadow\bench\nsys-$Tag"
Remove-Item "$rep.nsys-rep", "$rep.sqlite" -ErrorAction SilentlyContinue
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Ws2 {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
}
'@
$nargs = "profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o `"$rep`" -f true J:\Shadow\target-dev\release\shadowzero.exe serve $model --port 8095 --ctx 24576 $Flags"
$p = Start-Process -FilePath $nsys -ArgumentList $nargs -PassThru -WindowStyle Hidden -RedirectStandardOutput "$sp\nsys_$Tag.out" -RedirectStandardError "$sp\nsys_$Tag.err"
$child = $null
for ($i = 0; $i -lt 60 -and -not $child; $i++) {
  Start-Sleep -Milliseconds 500
  $child = Get-CimInstance Win32_Process -Filter "Name='shadowzero.exe'" | Where-Object { $_.CommandLine -match 'port 8095' } | Select-Object -First 1
}
if (-not $child) { 'no shadowzero.exe child appeared'; Stop-Process -Id $p.Id -Force; exit 1 }
$cp = Get-Process -Id $child.ProcessId
if (-not [Ws2]::SetProcessWorkingSetSizeEx($cp.Handle, [UIntPtr]([uint64]256MB), [UIntPtr]([uint64]$CapGiB * 1GB), 0x6)) {
  Stop-Process -Id $cp.Id -Force; Stop-Process -Id $p.Id -Force; 'cap NOT set - killed'; exit 1
}
"capped shadowzero.exe $($cp.Id) at $CapGiB GiB"
$watch = Start-Job -ArgumentList $cp.Id, $FloorGiB -ScriptBlock {
  param($id, $floor)
  while (Get-Process -Id $id -ErrorAction SilentlyContinue) {
    $av = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
    if ($av -lt $floor) { Stop-Process -Id $id -Force; "KILLED at $av GiB"; break }
    Start-Sleep -Milliseconds 1000
  }
}
$up = $false
for ($i = 0; $i -lt 150; $i++) { Start-Sleep 3; try { if ((Invoke-WebRequest http://127.0.0.1:8095/health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $up = $true; break } } catch {} }
if ($up) { python $Client 8095 2>&1 | Out-File "$sp\nsys_$Tag.client.txt" } else { 'server never came up' | Out-File "$sp\nsys_$Tag.client.txt" }
try { Invoke-WebRequest http://127.0.0.1:8095/shutdown -Method Post -UseBasicParsing -TimeoutSec 5 | Out-Null } catch {}
$p.WaitForExit(600000) | Out-Null
Receive-Job $watch
'done' | Out-File "$sp\nsys_$Tag.done"
