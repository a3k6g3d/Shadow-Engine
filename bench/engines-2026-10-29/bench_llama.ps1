param([string]$Out = 'bench\llama-try', [int]$Runs = 2,
      [string[]]$Variants = @('ncmoe43=J:\Shadow\target-llama\bin\llama-server.exe|-ngl 99 --n-cpu-moe 43'))
Set-Location J:\Shadow
$M = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$a = @('run', '--model', $M, '--out', $Out, '--runs', "$Runs")
foreach ($v in $Variants) {
  $name, $rest = $v.Split('=', 2)
  $exe, $flags = $rest.Split('|', 2)
  $L = "& 'J:\Shadow\tools\run_capped.ps1' -CapGiB 50 -Out '{log}' -Exe '$exe' " +
       "-Arguments '-m $M --port {port} -c 8192 --jinja -t 8 --host 127.0.0.1 $flags'"
  $a += '--arm'; $a += "$name=-"; $a += '--arm-launch'; $a += "$name=$L"
}
& J:\Shadow\target-dev\bench-batch\shadow-bench.exe @a
