"""How predictable is a token's routed experts from cheap signals?  (the draft-side route predictor question)

Reads routing logs (SHADOW_ROUTE_LOG=FILE shadow generate ... --gpu) and reports, per distance d ahead (a drafted position d
tokens after the last committed one) the recall of the actual top-k experts at each MoE layer within a predicted set:

  last      the experts of the last committed token at the same layer (|set| = k)
  last2     the union of the last 2 committed tokens' experts
  last4     the union of the last 4
  freq M    the M most frequent experts of that layer over the other logs (a static profile)
  last+freq the last committed token's experts plus the top (M - k) frequent ones, |set| = M

    python tools/route_predictability.py LOG.bin LOG.bin ... [--skip 20]
"""
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0])
from spec_probe import load  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 20
    if "--skip" in sys.argv:
        args = [a for a in args if a != sys.argv[sys.argv.index("--skip") + 1]]
    logs = [load(p) for p in args]
    k = logs[0][2]
    per = len(logs[0][1][0])
    for held in range(len(logs)):
        pass
    # frequency profile per layer from every log (the evaluation below excludes the log under test)
    def freq_of(skip_log):
        c = [Counter() for _ in range(per)]
        for i, (toks, routes, _) in enumerate(logs):
            if i == skip_log:
                continue
            for t in range(len(routes)):
                for li in range(per):
                    c[li].update(routes[t][li])
        return c

    dists = (1, 2, 3, 4)
    methods = ["last", "last2", "last4", "freq16", "freq32", "last+freq16", "last+freq32"]
    hit = {m: {d: 0 for d in dists} for m in methods}
    size = {m: 0 for m in methods}
    n = {d: 0 for d in dists}
    nsz = 0
    for li_log, (toks, routes, _) in enumerate(logs):
        freq = freq_of(li_log) if len(logs) > 1 else [Counter() for _ in range(per)]
        top = [[e for e, _ in f.most_common(64)] for f in freq]
        for t in range(skip + 4, len(routes) - max(dists)):
            for li in range(per):
                sets = {
                    "last": routes[t][li],
                    "last2": routes[t][li] | routes[t - 1][li],
                    "last4": routes[t][li] | routes[t - 1][li] | routes[t - 2][li] | routes[t - 3][li],
                    "freq16": set(top[li][:16]),
                    "freq32": set(top[li][:32]),
                    "last+freq16": routes[t][li] | set(top[li][: max(0, 16 - k)]),
                    "last+freq32": routes[t][li] | set(top[li][: max(0, 32 - k)]),
                }
                for m, s in sets.items():
                    size[m] += len(s)
                nsz += 1
                for d in dists:
                    actual = routes[t + d][li]
                    n[d] += len(actual)
                    for m, s in sets.items():
                        hit[m][d] += len(actual & s)
    print(f"{len(logs)} logs, k={k}, {per} MoE layers; recall of the actual top-{k} experts within the predicted set")
    print(f"{'predictor':14s} {'mean |set|':>10s} " + " ".join(f"d={d:<5d}" for d in dists))
    for m in methods:
        print(f"{m:14s} {size[m] / nsz:10.1f} " + " ".join(f"{hit[m][d] / n[d]:7.3f}" for d in dists))


main()
