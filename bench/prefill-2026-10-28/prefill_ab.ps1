# Interleaved prefill A/B on one binary: arms are "batch/scratch" pairs (SHADOW_BATCH, SHADOW_BATCH_SCRATCH).
param([int]$Runs = 1, [string]$Exe = 'target-dev\release\shadow.exe', [int]$N = 16, [string[]]$Arms = @('1/32', '8/32', '64/32'),
      [string]$Tag = 'pf')
Set-Location J:\Shadow
$sc = 'C:\Users\a3k6g\AppData\Local\Temp\claude\J--Shadow\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$M = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$args_ = "generate $M @$sc\prompt.txt --gpu --chat -n $N --ram-tier-gib 30 --placement profiles/routing-6prompts.shrc"
for ($i = 0; $i -lt $Runs; $i++) {
  foreach ($arm in $Arms) {
    $b, $s = $arm.Split('/')
    $env:SHADOW_BATCH = $b
    $env:SHADOW_BATCH_SCRATCH = $s
    $out = "$sc\${Tag}_b${b}_s${s}_$i.txt"
    $cap = & .\tools\run_capped.ps1 -CapGiB 44 -Out $out -Exe $Exe -Arguments $args_
    $err = Get-Content "$out.err"
    $p = ($err | Select-String 'prompt read in').Line
    $g = ($err | Select-String '^generated').Line
    $w = (($err | Select-String 'after the prompt').Line -split ';' | Select-String 'host waits').Line
    "run $i arm $arm | $cap | $p | $g |$w"
  }
}
Remove-Item Env:SHADOW_BATCH, Env:SHADOW_BATCH_SCRATCH
