"""Comprehensive scoring for shadow-bench runs:
Calculates:
1. Prompt throughput: sum(prompt_n) / sum(prompt_ms) * 1000
2. Decode throughput: sum(predicted_n) / sum(predicted_ms) * 1000
3. Total combined throughput: sum(prompt_n + predicted_n) / sum(prompt_ms + predicted_ms) * 1000
over warm turns (turn 0 dropped) for each arm, per run and combined.

Usage:
    python tools/bench_prompt_and_decode.py bench/DIR [bench/DIR2 ...]
"""
import json
import sys
from collections import defaultdict

for d in sys.argv[1:]:
    runs = defaultdict(list)
    for line in open(f"{d}/results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("error"):
            continue
        t = [x for x in r["turns"] if x["turn"] > 0]
        prompt_n = sum(x.get("prompt_n", 0) for x in t)
        prompt_ms = sum(x.get("prompt_ms", 0.0) for x in t)
        dec_n = sum(x.get("predicted_n", 0) for x in t)
        dec_ms = sum(x.get("predicted_ms", 0.0) for x in t)
        early = sum(1 for x in t if x.get("finish") == "stop" and x.get("predicted_n", 0) < 0.5 * max(y.get("predicted_n", 0) for y in t))
        runs[r["arm"]].append({
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "dec_n": dec_n,
            "dec_ms": dec_ms,
            "early": early,
            "turns": len(t),
        })

    print(f"\n==================================================")
    print(f"Directory: {d}")
    print(f"==================================================")
    for arm, rs in runs.items():
        tot_prompt_n = sum(r["prompt_n"] for r in rs)
        tot_prompt_ms = sum(r["prompt_ms"] for r in rs)
        tot_dec_n = sum(r["dec_n"] for r in rs)
        tot_dec_ms = sum(r["dec_ms"] for r in rs)

        prompt_tps = (1000.0 * tot_prompt_n / tot_prompt_ms) if tot_prompt_ms > 0 else 0.0
        dec_tps = (1000.0 * tot_dec_n / tot_dec_ms) if tot_dec_ms > 0 else 0.0
        tot_tps = (1000.0 * (tot_prompt_n + tot_dec_n) / (tot_prompt_ms + tot_dec_ms)) if (tot_prompt_ms + tot_dec_ms) > 0 else 0.0

        per_prompt = ", ".join(f"{1000.0 * r['prompt_n'] / r['prompt_ms']:.1f}" if r['prompt_ms'] > 0 else "0.0" for r in rs)
        per_dec = ", ".join(f"{1000.0 * r['dec_n'] / r['dec_ms']:.1f}" if r['dec_ms'] > 0 else "0.0" for r in rs)
        per_tot = ", ".join(f"{1000.0 * (r['prompt_n'] + r['dec_n']) / (r['prompt_ms'] + r['dec_ms']):.1f}" if (r['prompt_ms'] + r['dec_ms']) > 0 else "0.0" for r in rs)

        print(f"Arm: {arm}")
        print(f"  Runs: {len(rs)}")
        print(f"  Prompt tok/s:   {prompt_tps:6.2f} tok/s ({per_prompt}) | {tot_prompt_n} total tokens")
        print(f"  Decode tok/s:   {dec_tps:6.2f} tok/s ({per_dec}) | {tot_dec_n} total tokens")
        print(f"  Combined tok/s: {tot_tps:6.2f} tok/s ({per_tot})")
        print(f"  Early stops:    {sum(r['early'] for r in rs) / len(rs):.1f} / {rs[0]['turns']} turns")
