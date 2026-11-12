"""Decode only (`py tools/nsys_decode_rounds.py X.sqlite`, an export of `nsys profile --trace=cuda --cuda-graph-trace=node` over a served conversation with the DFlash drafter on): the intervals between consecutive DFlash draft rounds (a round = df_attention launches less than 2 ms apart), those under 300 ms
(a longer one holds a prompt pass or the gap between requests).  Per interval: kernel busy, H2D copy busy, both, neither; H2D bytes."""
import sqlite3, sys

db = sqlite3.connect(sys.argv[1])
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
ks = c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ms = c.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 order by start").fetchall()
df = [k for k in ks if names.get(k[2], '') == 'df_attention']
rounds = []
for s, e, _ in df:
    if rounds and s - rounds[-1][1] < 2e6:
        rounds[-1][1] = e
    else:
        rounds.append([s, e])
wins = [(rounds[i][0], rounds[i + 1][0]) for i in range(len(rounds) - 1) if rounds[i + 1][0] - rounds[i][0] < 300e6]


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


K = merge([(k[0], k[1]) for k in ks])
M = merge([(m[0], m[1]) for m in ms])


def clipped(iv, t0, t1):
    return [(max(s, t0), min(e, t1)) for s, e in iv if e > t0 and s < t1]


def inter(a, b):
    i = j = r = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            r += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return r


import bisect
kst = [x[0] for x in K]
mst = [x[0] for x in M]
mbs = [x[0] for x in ms]
tot = dict(w=0, k=0, m=0, both=0, bytes=0, draft=0)
for (t0, t1), r in zip(wins, rounds):
    def sl(iv, st):
        i = max(0, bisect.bisect_left(st, t0) - 1)
        j = bisect.bisect_right(st, t1)
        return clipped(iv[i:j], t0, t1)
    kk, mm = sl(K, kst), sl(M, mst)
    tot['w'] += t1 - t0
    tot['k'] += sum(e - s for s, e in kk)
    tot['m'] += sum(e - s for s, e in mm)
    tot['both'] += inter(kk, mm)
    i, j = bisect.bisect_left(mbs, t0), bisect.bisect_right(mbs, t1)
    tot['bytes'] += sum(x[2] for x in ms[i:j])
n = len(wins)
W = tot['w']
print('%d decode rounds (of %d draft rounds), %.2f s: %.1f ms a round' % (n, len(rounds), W / 1e9, W / n / 1e6))
print('per round: kernels %.1f ms, H2D copies %.1f ms (%.0f MB), both at once %.1f ms, neither %.1f ms' % (
    tot['k'] / n / 1e6, tot['m'] / n / 1e6, tot['bytes'] / n / 1e6, tot['both'] / n / 1e6, (W - tot['k'] - tot['m'] + tot['both']) / n / 1e6))
print('shares: kernel busy %.1f%%, copy busy %.1f%%, copy with no kernel %.1f%%, neither %.1f%%' % (
    100 * tot['k'] / W, 100 * tot['m'] / W, 100 * (tot['m'] - tot['both']) / W, 100 * (W - tot['k'] - tot['m'] + tot['both']) / W))
print('copy rate while copying: %.1f GB/s' % (tot['bytes'] / tot['m']))

# kernels by name and the gaps with no kernel, decode windows only
from collections import defaultdict
d, cnt = defaultdict(float), defaultdict(int)
starts = [k[0] for k in ks]
gaps = []
for (t0, t1) in wins:
    i, j = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1)
    for s, e, nm in ks[i:j]:
        d[names.get(nm, nm)] += min(e, t1) - s
        cnt[names.get(nm, nm)] += 1
    kk = merge(clipped([(k[0], k[1]) for k in ks[max(0, i - 1):j]], t0, t1))
    gaps += [kk[x + 1][0] - kk[x][1] for x in range(len(kk) - 1)]
tk = sum(d.values())
print('kernel time %.1f ms a round' % (tk / n / 1e6))
for nm, v in sorted(d.items(), key=lambda x: -x[1])[:14]:
    print('  %-34s %5.2f ms a round (%4.1f%%), %5.1f launches a round, %5.1f us each' % (nm, v / n / 1e6, 100 * v / tk, cnt[nm] / n, v / cnt[nm] / 1e3))
for lo, hi in [(0, 10e3), (10e3, 100e3), (100e3, 1e6), (1e6, 1e15)]:
    gg = [g for g in gaps if lo <= g < hi]
    print('gaps %.2f-%s ms: %.1f a round, %.2f ms a round' % (lo / 1e6, 'inf' if hi > 1e14 else '%.2f' % (hi / 1e6), len(gg) / n, sum(gg) / n / 1e6))

# the gaps of 0.1 ms or more: the CUDA API calls that overlap them (time inside the gap, by call), and the kernel the gap follows
api = c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
ast = [a[0] for a in api]
by_call, by_prev, nobody = defaultdict(float), defaultdict(float), 0.0
for (t0, t1) in wins:
    i, j = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1)
    seg = ks[i:j]
    for a, b in zip(seg, seg[1:]):
        g0, g1 = a[1], b[0]
        if g1 - g0 < 100e3:
            continue
        by_prev[names.get(a[2], a[2]) + ' -> ' + names.get(b[2], b[2])] += g1 - g0
        x = max(0, bisect.bisect_left(ast, g0 - 50e6))
        covered = 0
        while x < len(api) and api[x][0] < g1:
            s, e, nm = api[x]
            if e > g0:
                ov = min(e, g1) - max(s, g0)
                if ov > 20e3:
                    by_call[names.get(nm, nm)] += ov
                    covered += ov
            x += 1
        nobody += max(0, (g1 - g0) - covered)
print('gaps of 0.1 ms or more, by the API call running in them (ms a round; calls overlap):')
for nm, v in sorted(by_call.items(), key=lambda x: -x[1])[:10]:
    print('  %-40s %5.2f' % (nm, v / n / 1e6))
print('  (no API call over 20 us: %.2f ms a round: host work)' % (nobody / n / 1e6))
print('by the kernels around the gap:')
for nm, v in sorted(by_prev.items(), key=lambda x: -x[1])[:12]:
    print('  %-70s %5.2f' % (nm[:70], v / n / 1e6))
