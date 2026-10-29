"""Autotune Shadow's serving flags with the paired bench (coordinate ascent; each round is one `shadow-bench run`).

For each knob in turn, one bench round measures the current best config against every other value of that knob
(arms interleaved by shadow-bench, so the machine's drift hits all arms alike).  A value is adopted when its median
decode tok/s beats the current best's by more than max(MIN_GAIN, 3 x the current best's own run-to-run floor) and its
paired median over the same turns is positive.  Passes repeat until a whole pass adopts nothing.  Everything is
logged to OUT/log.md and the best config to OUT/best.json (resumable: a finished round is not run again).

    python tools/autotune.py --exe target-dev/bench-batch/shadow-df.exe --out bench/autotune [--runs 3] [--passes 2]
                             [--knobs --spec,--tail-skip] [--space space.json] [--dry-run]

A knob is a flag and the values to try (None: the flag absent; True: the bare flag).  The default space is below;
`--space` replaces it with a JSON object {flag: [values...]}.  A new idea for the loop is one more entry there.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

MODEL = "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf"
BASE = {
    "--ctx": 8192,
    "--ram-tier-gib": 30,
    "--placement": "J:\\Shadow\\profiles\\routing-6prompts.shrc",
    "--tail-skip": 4,
    "--verify-resident": True,
    "--dflash": "J:\\Shadow\\models\\dflash2",
    "--spec": 3,
}
SPACE = {
    "--spec": [3, 4, 5, 6, 7],
    "--dflash-min-p": [None, 0.5, 0.7],
    "--tail-skip": [3, 4, 5],
    "--ram-tier-gib": [24, 30, 36],
    "--pin-margin": [None, 2, 4, 8],
    "--cpu-experts": [None, 2, 4],
    "--fuse": [None, 1, 2, 4, 7],
}
MIN_GAIN = 0.01


def flags(cfg):
    out = []
    for k, v in cfg.items():
        if v is None:
            continue
        out.append(k if v is True else f"{k} {v}")
    return " ".join(out)


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


def parse_summary(path):
    """arm -> (median tok/s, floor median %, paired median % or None)."""
    rows, paired = {}, {}
    sec = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("| arm"):
            sec += 1
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 6 or cells[0].startswith("---"):
            continue
        try:
            if sec == 1:
                rows[cells[0]] = (float(cells[2]), float(cells[6].rstrip("%")))
            elif sec == 2:
                paired[cells[0]] = float(cells[2].rstrip("%").lstrip("+"))
        except ValueError:
            pass
    return {a: (m, fl, paired.get(a)) for a, (m, fl) in rows.items() if not math.isnan(m)}


def round_(args, name, base, variants, log):
    d = os.path.join(args.out, name)
    arms = {"base": base}
    for i, v in enumerate(variants):
        arms[f"v{i}"] = v
    if not os.path.exists(os.path.join(d, "summary.md")):
        cmd = [args.bench, "run", "--model", MODEL, "--out", d, "--runs", str(args.runs)]
        for a, c in arms.items():
            cmd += ["--arm", f"{a}={flags(c)}", "--arm-exe", f"{a}={args.exe}"]
        log(f"- running `{name}`: {len(arms)} arms x {args.runs} runs")
        if args.dry_run:
            print(" ".join(cmd))
            return None
        subprocess.run(cmd, check=False)
    if not os.path.exists(os.path.join(d, "summary.md")):
        return None
    return parse_summary(os.path.join(d, "summary.md")), arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--bench", default="target-dev/bench-batch/shadow-bench.exe")
    ap.add_argument("--out", default="bench/autotune")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--knobs", default=None)
    ap.add_argument("--space", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    space = json.load(open(args.space)) if args.space else SPACE
    knobs = args.knobs.split(",") if args.knobs else list(space)
    best_path = os.path.join(args.out, "best.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else dict(BASE)
    logf = open(os.path.join(args.out, "log.md"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    for p in range(args.passes):
        adopted = False
        for knob in knobs:
            variants, seen = [], {key(best)}
            for val in space[knob]:
                c = dict(best)
                c[knob] = val
                if key(c) not in seen:
                    seen.add(key(c))
                    variants.append(c)
            if not variants:
                continue
            name = f"p{p}-{knob.strip('-')}-{abs(hash(key(best))) % 10**6}"
            res = round_(args, name, best, variants, log)
            if res is None:
                continue
            stats, arms = res
            if "base" not in stats:
                log(f"- {knob}: the base arm produced no result; skipped")
                continue
            bm, bfloor, _ = stats["base"]
            thresh = max(MIN_GAIN, 3 * bfloor / 100)
            log(f"### pass {p}, {knob}: base {bm:.2f} tok/s (floor {bfloor:.1f}%, needs +{100 * thresh:.1f}%)")
            pick = None
            for a, c in arms.items():
                if a == "base" or a not in stats:
                    continue
                m, fl, pm = stats[a]
                gain = m / bm - 1
                log(f"- `{knob} {c[knob]}`: {m:.2f} tok/s ({100 * gain:+.1f}%), paired median {pm}")
                if gain > thresh and (pm is None or pm > 0) and (pick is None or m > pick[1]):
                    pick = (c, m)
            if pick:
                best = pick[0]
                adopted = True
                json.dump(best, open(best_path, "w"), indent=1)
                log(f"**adopted {knob} = {best[knob]}** -> {pick[1]:.2f} tok/s; best flags now: `{flags(best)}`")
        if not adopted:
            break
    json.dump(best, open(best_path, "w"), indent=1)
    log(f"best flags: `{flags(best)}`")


main()
