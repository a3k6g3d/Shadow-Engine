# Resumes the interrupted EXL3 conversion of Huihui-Qwen3.5-35B-A3B-abliterated (2.55 bpw, 5060 Ti = CUDA device 1) from its work dir under the RAM cap
# (tools/run_capped.ps1: cap 36 GiB, floor 3.5 GiB).  The converter checkpoints after every layer; a stop loses at most the layer in progress.
#   powershell -File tools/resume_q35_convert.ps1 [-Log PATH]
param([string]$Log = 'F:\work\q35-exl3-2.55bpw_resume.log')
Set-Location I:\tools\exllamav3
$env:CUDA_VISIBLE_DEVICES = '1'
$env:PYTHONUNBUFFERED = '1'
Remove-Item $Log, "$Log.err" -ErrorAction SilentlyContinue
& 'J:\Shadow\tools\run_capped.ps1' -CapGiB 36 -FloorGiB 3.5 -Out $Log -Exe 'I:\tools\exl3-venv\Scripts\python.exe' -Arguments '-u convert.py -w F:\work\q35-exl3-2.55bpw -r'
"CONVERT RESUME ENDED $(Get-Date)" | Out-File 'F:\work\q35-exl3-2.55bpw_resume.done'
