Set-Location J:\Shadow
$sp = 'C:\Users\a3k6g\AppData\Local\Temp\claude\c2eb5aa3e3d3334d\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$model = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$L = "`$env:SHADOW_SHARED_EARLY='0'; & 'J:\Shadow\tools\run_capped.ps1' -CapGiB 42 -FloorGiB 3 -Out '{log}' -Exe 'target-dev/release/shadow.exe' -Arguments 'serve $model --port {port} --ctx 24576'"
$a = @('run','--model',$model,'--out','bench/shared-early-2026-11-12','--runs','3','--turns','36','--prompts-file','bench/dfdata/varied36.json',
  '--arm','early=--ctx 24576','--arm','late=-','--arm-exe','early=target-dev/release/shadow.exe','--arm-launch',"late=$L")
& target\release\shadow-bench.exe @a *> "$sp\sharedearly_bench.out"
python tools\bench_total.py bench/shared-early-2026-11-12 *>> "$sp\sharedearly_bench.out"
'done' | Out-File "$sp\sharedearly.done"
