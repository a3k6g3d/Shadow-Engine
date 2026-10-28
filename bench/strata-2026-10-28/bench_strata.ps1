Set-Location J:\Shadow
$sc = 'C:\Users\a3k6g\AppData\Local\Temp\claude\J--Shadow\95988fd8-569b-47c7-99ea-9f100d75d035\scratchpad'
$M = 'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'
$F = '--ctx 8192 --ram-tier-gib 30 --placement J:\Shadow\profiles\routing-6prompts.shrc'
$L = "& '$sc\run_strata.ps1' -Port {port} -Out '{log}' -Config '$sc\strata\strata-bench.json'"
& J:\Shadow\target-dev\bench-batch\shadow-bench.exe run --model $M --out bench\strata-2026-10-28 --runs 8 `
    --arm "shadow=$F" --arm "strata=-" --arm-exe 'shadow=J:\Shadow\target-dev\bench-lru\shadow.exe' --arm-launch "strata=$L"
