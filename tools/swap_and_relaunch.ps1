# Waits for Shadow to close, swaps in the fresh release binaries, relaunches.
$log = 'J:\Shadow\target\swap.log'
"swap started $(Get-Date)" | Out-File $log
$deadline = (Get-Date).AddMinutes(10)
while ((Get-Date) -lt $deadline) {
    $alive = Get-Process shadow, shadow-ui -ErrorAction SilentlyContinue
    if (-not $alive) { break }
    Start-Sleep -Seconds 2
}
if (-not (Get-Process shadow, shadow-ui -ErrorAction SilentlyContinue)) {
    Copy-Item J:\Shadow\target\altbuild\release\shadow.exe     J:\Shadow\target\release\shadow.exe     -Force
    Copy-Item J:\Shadow\target\altbuild\release\shadow-ui.exe  J:\Shadow\target\release\shadow-ui.exe  -Force
    "exes swapped $(Get-Date)" | Out-File $log -Append
    Start-Process 'J:\Shadow\target\release\shadow-ui.exe' -WorkingDirectory 'J:\Shadow'
    "relaunched $(Get-Date)" | Out-File $log -Append
} else {
    "timed out waiting for Shadow to close - nothing copied" | Out-File $log -Append
}