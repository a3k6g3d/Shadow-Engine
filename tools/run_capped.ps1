# Runs a program under a HARD working-set cap and a free-RAM watchdog (kill below -FloorGiB, default 4, available; 0 = no watchdog kill), the way every
# load of this model on this PC has to run (J:\Strata-work\RUN.md: an uncapped mmap load took available RAM from 55 to
# 0.73 GiB in 49 s).  Changes no system setting.  `shadow serve`'s default 36 GiB tier dips to ~4.1 GiB available: run it with -FloorGiB 3.
#
#   powershell -File tools/run_capped.ps1 -CapGiB 16 -Out log.txt -Exe target/release/shadow.exe -Arguments 'generate M "text" -n 6'
param([int]$CapGiB = 16, [double]$FloorGiB = 4, [string]$Out = 'run.out.txt', [Parameter(Mandatory = $true)] [string]$Exe, [string]$Arguments = '')
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Ws {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
}
'@
$t0 = Get-Date
$sp = @{ FilePath = $Exe; PassThru = $true; WindowStyle = 'Hidden'; RedirectStandardOutput = $Out; RedirectStandardError = "$Out.err" }
if ($Arguments) { $sp.ArgumentList = $Arguments } # Start-Process refuses an empty -ArgumentList
$p = Start-Process @sp
if (-not $p) { "could not start '$Exe' - nothing ran"; exit 1 } # else the watchdog below would loop on nothing forever
# QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE
if (-not [Ws]::SetProcessWorkingSetSizeEx($p.Handle, [UIntPtr]([uint64]256MB), [UIntPtr]([uint64]$CapGiB * 1GB), 0x6)) {
  Stop-Process -Id $p.Id -Force; "cap NOT set - killed, nothing ran uncapped"; exit 1
}
$min = 999; $peak = 0
while (-not $p.HasExited) {
  $av = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024
  if ($av -lt $min) { $min = $av }
  $p.Refresh(); if ($p.WorkingSet64 -gt $peak) { $peak = $p.WorkingSet64 }
  if ($FloorGiB -gt 0 -and $av -lt $FloorGiB) { Stop-Process -Id $p.Id -Force; "KILLED at {0:N2} GiB available" -f $av; break }
  Start-Sleep -Milliseconds 1000
}
"exit {0}, wall {1:N0} s, min available {2:N1} GiB, peak working set {3:N1} GiB (cap {4})" -f $p.ExitCode, ((Get-Date) - $t0).TotalSeconds, $min, ($peak / 1GB), $CapGiB
