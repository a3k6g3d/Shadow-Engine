# Creates (or refreshes) the Start menu entry "ShadowZero (server only)": shadowzero-ui.exe --headless = the server and a tray icon, no window and no chat page,
# for running another harness (Hermes, pi, ...) against http://127.0.0.1:8081/v1.  Quit from the tray icon stops the server.
#   powershell -File tools/make_headless_shortcut.ps1 [-Exe J:\Shadow\target\release\shadowzero-ui.exe]
param([string]$Exe = (Join-Path (Split-Path $PSScriptRoot -Parent) 'target\release\shadowzero-ui.exe'))
$dir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$lnk = Join-Path $dir 'ShadowZero (server only).lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = $Exe
$s.Arguments = '--headless'
$s.WorkingDirectory = Split-Path $Exe -Parent | Split-Path -Parent | Split-Path -Parent
$s.Description = 'ShadowZero server only (no window): for other harnesses, http://127.0.0.1:8081/v1. Quit from the tray icon.'
# same icon as the "ShadowZero" entry when it has one, else the exe's own
$main = Join-Path $dir 'ShadowZero.lnk'
$ico = if (Test-Path $main) { (New-Object -ComObject WScript.Shell).CreateShortcut($main).IconLocation } else { '' }
$s.IconLocation = if ($ico -and $ico -ne ',0') { $ico } else { "$Exe,0" }
$s.Save()
"$lnk -> $Exe --headless"
