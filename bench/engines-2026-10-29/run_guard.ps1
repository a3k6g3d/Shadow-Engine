# A program under a working-set cap and a tight RAM guard: available RAM polled every 200 ms, the whole process
# tree killed below $FloorGiB (for engines that may pin or copy the model).  Prints one closing line.
param([string]$Exe, [string]$Arguments, [string]$Out, [int]$CapGiB = 40, [double]$FloorGiB = 6)
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Ws2 {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool SetProcessWorkingSetSizeEx(IntPtr h, UIntPtr min, UIntPtr max, uint flags);
  [StructLayout(LayoutKind.Sequential)] public struct MS { public uint len, load; public ulong total, avail, tpf, apf, tv, av, aev; }
  [DllImport("kernel32.dll")] public static extern bool GlobalMemoryStatusEx(ref MS m);
  public static double AvailGiB() { MS m = new MS(); m.len = (uint)Marshal.SizeOf(typeof(MS)); GlobalMemoryStatusEx(ref m); return m.avail / 1073741824.0; }
}
'@
$t0 = Get-Date
$p = Start-Process -FilePath $Exe -ArgumentList $Arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError "$Out.err"
[void][Ws2]::SetProcessWorkingSetSizeEx($p.Handle, [UIntPtr]([uint64]256MB), [UIntPtr]([uint64]$CapGiB * 1GB), 0x6)
$min = 999
while (-not $p.HasExited) {
  $av = [Ws2]::AvailGiB()
  if ($av -lt $min) { $min = $av }
  if ($av -lt $FloorGiB) { taskkill /T /F /PID $p.Id | Out-Null; "KILLED at {0:N2} GiB available" -f $av; break }
  Start-Sleep -Milliseconds 200
}
"exit $($p.ExitCode), wall {0:N0} s, min available {1:N1} GiB (guard {2} GiB)" -f ((Get-Date) - $t0).TotalSeconds, $min, $FloorGiB
