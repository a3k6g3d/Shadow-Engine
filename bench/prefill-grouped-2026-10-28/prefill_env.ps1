# Interleaved prefill A/B on one binary: each arm is "name:VAR=V,VAR=V" (or "name:" for none).
param([int]$Runs = 8, [string]$Exe = 'target-dev\bench-batch\shadow-mtp.exe', [string[]]$Arms, [string]$Tag = 'pe', [int]$N = 16)
Set-Location J:\Shadow
$sc = 'C:\Users\a3k6g\AppData\Local\Temp\claude\J--Shadow\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$M = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$args_ = "generate $M @$sc\prompt.txt --gpu --chat -n $N --ram-tier-gib 30 --placement profiles/routing-6prompts.shrc"
for ($i = 0; $i -lt $Runs; $i++) {
  foreach ($arm in $Arms) {
    $name, $envs = $arm.Split(':', 2)
    $set = @()
    foreach ($kv in ($envs -split ',' | Where-Object { $_ })) { $k, $v = $kv.Split('='); Set-Item "Env:$k" $v; $set += $k }
    $out = "$sc\${Tag}_${name}_$i.txt"
    $cap = & .\tools\run_capped.ps1 -CapGiB 44 -Out $out -Exe $Exe -Arguments $args_
    foreach ($k in $set) { Remove-Item "Env:$k" }
    $err = Get-Content "$out.err"
    $p = ($err | Select-String 'prompt read in').Line
    $g = ($err | Select-String '^generated').Line -replace '\[.*\]', '[..]'
    $h = ((($err | Select-String '^generated').Line) -replace '^generated \d+ tokens ', '' -replace ' in .*', '').GetHashCode()
    "run $i arm $name | $cap | $p | $g | tokens#$h"
  }
}
