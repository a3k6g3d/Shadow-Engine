"""Total decode throughput per arm from a shadow-bench directory: sum(predicted_n) / sum(predicted_ms) over the warm turns
(turn 0 dropped), per run and over all runs.  The bench's own headline is a median over turns, which compares different
turns when the arms generate different text (tail-skip, speculation); totals do not.

    python tools/bench_total.py bench/DIR [bench/DIR2 ...]
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
        n, ms = sum(x["predicted_n"] for x in t), sum(x["predicted_ms"] for x in t)
        # turns that stopped on their own before the token budget: a mode that truncates answers looks fast
        early = sum(1 for x in t if x.get("finish") == "stop" and x["predicted_n"] < 0.5 * max(y["predicted_n"] for y in t))
        runs[r["arm"]].append((n, ms, early, len(t)))
    print(d)
    for arm, rs in runs.items():
        tot = 1000 * sum(r[0] for r in rs) / sum(r[1] for r in rs)
        per = ", ".join(f"{1000 * r[0] / r[1]:.1f}" for r in rs)
        toks = sum(r[0] for r in rs) / len(rs)
        early = sum(r[2] for r in rs) / len(rs)
        print(f"  {arm:12s} {tot:6.2f} tok/s over {len(rs)} runs ({per}); {toks:.0f} tokens a run, {early:.1f} of {rs[0][3]} turns ended early")
