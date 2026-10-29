param([string]$Out = 'bench\ik-try', [int]$Runs = 2, [string[]]$Variants)
Set-Location J:\Shadow
$sc = 'C:\Users\a3k6g\AppData\Local\Temp\claude\J--Shadow\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$M = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$a = @('run', '--model', $M, '--out', $Out, '--runs', "$Runs")
foreach ($v in $Variants) {
  $name, $flags = $v.Split('=', 2)
  $L = "& '$sc\run_guard.ps1' -Exe 'J:\Shadow\target-build-ik\bin\llama-server.exe' -Out '{log}' " +
       "-Arguments '-m $M --port {port} -c 8192 --jinja -t 8 --host 127.0.0.1 $flags'"
  $a += '--arm'; $a += "$name=-"; $a += '--arm-launch'; $a += "$name=$L"
}
& J:\Shadow\target-dev\bench-batch\shadow-bench.exe @a
