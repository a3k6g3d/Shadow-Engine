"""What speculative decoding could buy Shadow, from routing logs (SHADOW_ROUTE_LOG=FILE shadow generate ... --gpu).

Two numbers decide it on an offloaded MoE:
  union(m)   distinct routed experts per layer when m consecutive tokens go through the model together, against
             10 per token alone: the expert bytes a verify pass has to move for m positions.
  accepted   how many tokens a drafter gets right per verify pass (greedy, so a draft token is kept only if it is
             exactly the token the model then chose).

    python tools/spec_probe.py LOG.bin [LOG.bin ...] [--prompt-len N ...]
"""
import struct
import sys
from collections import defaultdict


def load(path):
    b = open(path, "rb").read()
    (n,) = struct.unpack_from("<I", b, 0)
    toks = list(struct.unpack_from(f"<{n}I", b, 4))
    (k,) = struct.unpack_from("<I", b, 4 + 4 * n)
    vals = struct.unpack_from(f"<{(len(b) - 8 - 4 * n) // 2}H", b, 8 + 4 * n)
    recs = [vals[i:i + k + 1] for i in range(0, len(vals), k + 1)]
    layers = sorted({r[0] for r in recs})
    per = len(layers)
    assert len(recs) == per * n, (len(recs), per, n)
    # routes[t][li] = set of experts for token t at MoE layer li
    routes = [[set(recs[t * per + li][1:]) for li in range(per)] for t in range(n)]
    return toks, routes, k


def union_stats(routes, start, k, ms=(1, 2, 3, 4, 5, 6, 8)):
    out = {}
    per = len(routes[0])
    for m in ms:
        tot, cnt = 0, 0
        for t in range(start, len(routes) - m + 1):
            for li in range(per):
                u = set()
                for j in range(m):
                    u |= routes[t + j][li]
                tot += len(u)
                cnt += 1
        out[m] = tot / cnt / k if cnt else float("nan")  # in units of one token's experts
    return out


def lookup_draft(ctx, max_n=4, min_n=1, k=8):
    """Prompt/history lookup: the longest suffix of ctx (max_n down to min_n tokens) seen earlier; propose what followed
    its most recent earlier occurrence."""
    L = len(ctx)
    for n in range(max_n, min_n - 1, -1):
        if L < n + 1:
            continue
        suf = ctx[L - n:]
        for i in range(L - n - 1, -1, -1):
            if ctx[i:i + n] == suf:
                return ctx[i + n:i + n + k]
    return []


def simulate(toks, start, drafter, K):
    """Greedy speculative loop over the recorded continuation; returns (verify passes, generated tokens, positions
    verified)."""
    t = start
    passes = gen = positions = 0
    while t < len(toks):
        d = drafter(toks[:t])[:K]
        acc = 0
        while acc < len(d) and t + acc < len(toks) and d[acc] == toks[t + acc]:
            acc += 1
        step = min(acc + 1, len(toks) - t)  # the accepted drafts plus the model's own next token
        passes += 1
        gen += step
        positions += len(d) + 1
        t += step
    return passes, gen, positions


def main():
    args = sys.argv[1:]
    plens = []
    if "--prompt-len" in args:
        i = args.index("--prompt-len")
        plens = [int(x) for x in args[i + 1:]]
        args = args[:i]
    agg = defaultdict(list)
    for fi, path in enumerate(args):
        toks, routes, k = load(path)
        start = plens[fi] if fi < len(plens) else None
        if start is None:
            # the chat template's "<think>\n\n</think>\n\n" ends the prompt; fall back to half way
            start = len(toks) // 2
        u = union_stats(routes, start, k)
        line = " ".join(f"m{m}:{v:.2f}" for m, v in u.items())
        print(f"{path}: {len(toks)} tokens, {len(toks) - start} generated, {len(routes[0])} MoE layers")
        print(f"  experts per layer for m tokens together, in single-token units: {line}")
        for name, dr in [("lookup n<=4", lambda c: lookup_draft(c, 4, 1)), ("lookup n<=4, n>=2", lambda c: lookup_draft(c, 4, 2))]:
            for K in (2, 4, 8):
                p, g, pos = simulate(toks, start, dr, K)
                agg[(name, K)].append((p, g, pos))
                print(f"  {name:18s} K={K}: {g / p:.2f} tokens per verify pass, {pos / p:.2f} positions per pass")
        for m, v in u.items():
            agg[("union", m)].append(v)
    print("\nall prompts:")
    for key, v in agg.items():
        if key[0] == "union":
            continue
        p, g, pos = (sum(x[i] for x in v) for i in range(3))
        print(f"  {key[0]:18s} K={key[1]}: {g / p:.2f} tokens/pass, {pos / p:.2f} positions/pass")
    print("  union: " + " ".join(f"m{m}:{sum(agg[('union', m)]) / len(agg[('union', m)]):.2f}"
                                 for m in (1, 2, 3, 4, 5, 6, 8)))


if __name__ == "__main__":
    main()
