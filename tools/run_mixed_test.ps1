$Bench = "target/release/shadow-bench.exe"
$Model = "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf"
$Placement = "J:/Shadow/profiles/routing-varied300.shrc"
$Out = "bench/direct_io_mixed_fixed"
& $Bench run --model $Model --out $Out --runs 2 --turns 6 --tokens 96 --port 8092 --prompts-file bench/dfdata/varied36.json --arm "armC_base=--ctx 8192 --ram-tier-gib 36 --ram-floor-gib 3 --placement $Placement" --arm "armD_direct=--ctx 8192 --ram-tier-gib 36 --ram-floor-gib 3 --direct-io --placement $Placement"
python tools/bench_prompt_and_decode.py $Out
