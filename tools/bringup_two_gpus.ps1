# One-command bring-up of a second graphics card (docs/PLAN_TWO_GPUS_2026-11-11.md).  Run from a normal PowerShell in the repo root once the card is in:
#
#   powershell -File tools\bringup_two_gpus.ps1 [-Out bench\two-gpus-bringup] [-Skip bench,tests,exact] [-Peer 1]
#
# Steps (each writes its log under -Out and prints one verdict line; a failed step does not stop the rest):
#   1 gpus     `shadow gpus`: the cards, host bandwidth alone and together, peer access, the host-staged hop (gate G1 of the plan: hop under ~150 us for 40 KiB)
#   2 tests    the pair tests with the second stage on the second card (SHADOW_PEER_GPU=1): every logit identical to one session, and the drafter's drafts
#   3 int8     tools of the native int8 kernels on the second card (crates/shadow-cuda/tests/int8_cols.rs, INT8_GPU=1: numerics and timing of the float, MMQ and mul_mat_vec_q paths)
#   4 exact    `shadow serve --exact` single card against `--gpus 0,1`: the 10-request conversation (tools/pair_e2e_test.py), replies identical or not
#   5 fast     the same conversation in default fast mode on the pair (drafter on the second card): runs, coherent, speculation lines
#   6 bench    decode of the 36 varied prompts, 2 runs each, interleaved: single card, the pair, the pair with --dflash off (shadow-bench)
# Power and temperature of every card are logged for the whole run (nvidia-smi, every 5 s).  Nothing here changes a system setting: power limits are the owner's call -
# on the 650 W supply (until the 1,000 W one) the script PRINTS the command to cap the cards, it does not run it.  Every model load goes through tools\run_capped.ps1.
param([string]$Out = 'bench\two-gpus-bringup', [string]$Skip = '', [int]$Peer = 1, [int]$CapGiB = 42)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$skip = $Skip.Split(',') | ForEach-Object { $_.Trim().ToLower() }
$model = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$dev = 'target-dev\release'
function Say($t) { $l = "$([DateTime]::Now.ToString('HH:mm:ss')) $t"; $l; $l | Out-File -Append -Encoding utf8 "$Out\bringup.log" }
function Stop-Mine { Get-NetTCPConnection -LocalPort 8095 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Start-Sleep 3 }

Say "== two-card bring-up, $((nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader) -join ' | ')"
$n = [int]((nvidia-smi --query-gpu=name --format=csv,noheader | Measure-Object -Line).Lines)
# -Peer 0 is a dry run on one card (both stages on it, small expert caches): it checks the script's plumbing, its numbers mean nothing
if ($n -lt 2 -and $Peer -ne 0) { Say "the driver lists $n card(s): nothing to bring up (is the second card seated, powered and its PCIe cables on?)"; exit 1 }
$pairExtra = if ($Peer -eq 0) { ' --expert-cache-gib 0.6 --peer-cache-gib 0.6 --ram-tier-gib 8' } else { '' }
Say "power: if the two cards plus the CPU trip the 650 W supply, cap them first, e.g.  nvidia-smi -i 0 -pl 200   and   nvidia-smi -i 1 -pl 140   (run by you, as administrator; not done here)"
if (Get-Process shadow, shadow-ui -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Shadow*' }) { Say 'a shadow engine is running (the launcher?): stop it first, this script starts its own servers on port 8095'; exit 1 }

# power / temperature log of every card for the whole run
$smi = Start-Process nvidia-smi -ArgumentList '--query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.sm,utilization.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current', '--format=csv', '-l', '5' -RedirectStandardOutput "$Out\nvidia-smi.csv" -PassThru -WindowStyle Hidden

function Run-Capped($log, $exe, $argstr) {
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_capped.ps1 -CapGiB 20 -FloorGiB 4 -Out $log -Exe $exe -Arguments $argstr | ForEach-Object { Say "   $_" }
}

function Serve($tag, $flags, $envs) {
  foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
  $argstr = '-NoProfile -ExecutionPolicy Bypass -File tools\run_capped.ps1 -CapGiB ' + $CapGiB + ' -FloorGiB 3.5 -Out "' + "$Out\serve_$tag.log" + '" -Exe ' + "$dev\shadow.exe" + ' -Arguments "serve ' + $model + ' --port 8095 --ctx 8192 ' + $flags + '"'
  Start-Process powershell -ArgumentList $argstr -WindowStyle Hidden
  foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
  for ($i = 0; $i -lt 150; $i++) { Start-Sleep 3; try { if ((Invoke-WebRequest http://127.0.0.1:8095/health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { return $true } } catch {} }
  return $false
}

Say 'building the dev binaries (target-dev)'
$env:CARGO_TARGET_DIR = 'target-dev'
cargo build --release -p shadow -p shadow-bench 2>&1 | Select-Object -Last 2 | ForEach-Object { Say "   $_" }
$twoStage = (cargo test --release -p shadow --test two_stage --no-run 2>&1 | Select-String 'Executable.*\((.*two_stage-[0-9a-f]+\.exe)\)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
$int8 = (cargo test --release -p shadow-cuda --test int8_cols --no-run 2>&1 | Select-String 'Executable.*\((.*int8_cols-[0-9a-f]+\.exe)\)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue

if ('gpus' -notin $skip) {
  Say '-- 1 gpus'
  $g = & "$dev\shadow.exe" gpus 2>&1 | ForEach-Object { "$_" }
  $g | Out-File -Encoding utf8 "$Out\1_gpus.txt"
  $g | Where-Object { $_.Trim() } | ForEach-Object { Say "   $_" }
}

if ('tests' -notin $skip -and $twoStage) {
  Say "-- 2 pair tests, second stage on card $Peer"
  $env:SHADOW_PEER_GPU = "$Peer"
  Run-Capped "$Out\2_two_stage.txt" (Join-Path $root $twoStage) '--test-threads 1'
  Remove-Item Env:SHADOW_PEER_GPU -ErrorAction SilentlyContinue
  Say "   result: $((Select-String -Path "$Out\2_two_stage.txt" -Pattern 'test result' | Select-Object -First 1).Line)"
}

if ('int8' -notin $skip -and $int8) {
  Say "-- 3 native int8 kernels on card $Peer"
  $env:INT8_GPU = "$Peer"; $env:SHADOW_MMQ_DLL = Join-Path $root 'native\mmq\shadow_mmq.dll'; $env:INT8_ROWS = '20000'
  Run-Capped "$Out\3_int8_cols.txt" (Join-Path $root $int8) '--nocapture'
  Remove-Item Env:INT8_GPU, Env:SHADOW_MMQ_DLL, Env:INT8_ROWS -ErrorAction SilentlyContinue
  Say "   result: $((Select-String -Path "$Out\3_int8_cols.txt" -Pattern 'test result' | Select-Object -First 1).Line)"
}

function Conversation($tag, $flags, $envs) {
  Stop-Mine
  if (-not (Serve $tag $flags $envs)) { Say "   ${tag}: server never came up (see $Out\serve_$tag.log.err)"; Stop-Mine; return $false }
  python tools\pair_e2e_test.py 8095 "$Out\conv_$tag.json" 2>&1 | Out-File -Encoding utf8 "$Out\conv_$tag.txt"
  try { Invoke-WebRequest http://127.0.0.1:8095/shutdown -Method Post -UseBasicParsing -TimeoutSec 5 | Out-Null } catch {}
  Start-Sleep 12; Stop-Mine
  return (Test-Path "$Out\conv_$tag.json")
}

if ('exact' -notin $skip) {
  Say '-- 4 exact: single card against the pair'
  $a = Conversation 'exact_single' '--exact --expert-cache-gib 1.0' @{}
  $b = Conversation 'exact_pair' "--exact --gpus 0,$Peer$pairExtra" @{}
  if ($a -and $b) {
    $same = python -c "import json; a=json.load(open(r'$Out\conv_exact_single.json')); b=json.load(open(r'$Out\conv_exact_pair.json')); print(sum(x['text']==y['text'] for x,y in zip(a,b)), 'of', len(a))"
    Say "   exact replies identical, single card vs the pair: $same"
  } else { Say '   (a conversation did not finish)' }
}

if ('fast' -notin $skip) {
  Say '-- 5 fast mode on the pair'
  if (Conversation 'fast_pair' "--gpus 0,$Peer$pairExtra" @{}) {
    $spec = Select-String -Path "$Out\serve_fast_pair.log.err" -Pattern 'speculation:' | Select-Object -Last 3 | ForEach-Object { $_.Line.Substring(0, [Math]::Min(150, $_.Line.Length)) }
    Say "   conversation finished; speculation lines: $($spec -join ' || ')"
    Say "   dflash on the pair: $((Select-String -Path "$Out\serve_fast_pair.log.err" -Pattern 'dflash:' | Select-Object -First 2 | ForEach-Object { $_.Line }) -join ' | ')"
  }
}

if ('bench' -notin $skip) {
  Say '-- 6 decode bench (36 prompts x 2 runs, interleaved): single card / pair / pair --dflash off'
  Stop-Mine
  $a = @('run', '--model', $model, '--out', "$Out\bench", '--runs', '2', '--turns', '36', '--prompts-file', 'bench/dfdata/varied36.json',
         '--arm', 'single=--ctx 24576', '--arm', "pair=--ctx 24576 --gpus 0,$Peer$pairExtra", '--arm', "pair_nodflash=--ctx 24576 --gpus 0,$Peer$pairExtra --dflash off",
         '--arm-exe', "single=$dev/shadow.exe", '--arm-exe', "pair=$dev/shadow.exe", '--arm-exe', "pair_nodflash=$dev/shadow.exe")
  & "$dev\shadow-bench.exe" @a *> "$Out\6_bench.txt"
  python tools\bench_total.py "$Out\bench" 2>&1 | Tee-Object -FilePath "$Out\6_bench_total.txt" | ForEach-Object { Say "   $_" }
}

Stop-Process -Id $smi.Id -Force -ErrorAction SilentlyContinue
Start-Sleep 1
$rows = Get-Content "$Out\nvidia-smi.csv" -ErrorAction SilentlyContinue | Select-Object -Skip 1 | Where-Object { $_.Trim() } | ForEach-Object {
  $f = $_ -split ','
  [pscustomobject]@{ Card = $f[1].Trim(); Watts = [double]($f[2] -replace '[^0-9.]', ''); Celsius = [double]($f[3] -replace '[^0-9.]', '') }
}
$rows | Group-Object Card | ForEach-Object {
  Say "card $($_.Name): peak power $(($_.Group | Measure-Object Watts -Maximum).Maximum) W, peak temperature $(($_.Group | Measure-Object Celsius -Maximum).Maximum) C"
}
Say "done - everything is in $Out (bringup.log first)"
