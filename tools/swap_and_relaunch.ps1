# Waits for ShadowZero to close, swaps in the fresh release binaries, relaunches.
$log = 'J:\Shadow\target\swap.log'
"swap started $(Get-Date)" | Out-File $log
$deadline = (Get-Date).AddMinutes(10)
while ((Get-Date) -lt $deadline) {
    $alive = Get-Process shadow, shadowzero-ui -ErrorAction SilentlyContinue
    if (-not $alive) { break }
    Start-Sleep -Seconds 2
}
if (-not (Get-Process shadow, shadowzero-ui -ErrorAction SilentlyContinue)) {
    Copy-Item J:\Shadow\target\altbuild\release\shadowzero.exe     J:\Shadow\target\release\shadowzero.exe     -Force
    Copy-Item J:\Shadow\target\altbuild\release\shadowzero-ui.exe  J:\Shadow\target\release\shadowzero-ui.exe  -Force
    "exes swapped $(Get-Date)" | Out-File $log -Append
    Start-Process 'J:\Shadow\target\release\shadowzero-ui.exe' -WorkingDirectory 'J:\Shadow'
    "relaunched $(Get-Date)" | Out-File $log -Append
} else {
    "timed out waiting for ShadowZero to close - nothing copied" | Out-File $log -Append
}