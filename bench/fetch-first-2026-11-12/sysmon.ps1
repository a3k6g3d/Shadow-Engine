param([string]$Out)
"time,cpu_pct,avail_mb,disk_read_bps,disk_write_bps,pages_per_s" | Out-File -Encoding ascii $Out
$names = @('\Processor(_Total)\% Processor Time', '\Memory\Available MBytes', '\PhysicalDisk(_Total)\Disk Read Bytes/sec',
           '\PhysicalDisk(_Total)\Disk Write Bytes/sec', '\Memory\Pages/sec')
while ($true) {
  $c = Get-Counter -Counter $names -SampleInterval 1
  $v = $c.CounterSamples | ForEach-Object { $_.CookedValue }
  ('{0},{1:F1},{2:F0},{3:F0},{4:F0},{5:F1}' -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss.fff'), $v[0], $v[1], $v[2], $v[3], $v[4]) |
    Out-File -Encoding ascii -Append $Out
}
