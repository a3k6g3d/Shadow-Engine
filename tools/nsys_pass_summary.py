"""Summarizes an Nsight Systems export (SQLite) of a shadow prompt: how much of the prompt's window the GPU runs kernels, copies host to device, or does neither.

  nsys export --type sqlite -o X.sqlite -f true X.nsys-rep
  py tools/nsys_pass_summary.py X.sqlite

The window runs from the first to the last `select_blocks_rows` launch (the sparse attention of a long prompt)."""
import sqlite3, sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
ks = c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ms = c.execute("select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
sel = [k for k in ks if names.get(k[2], '') == 'select_blocks_rows']
t0, t1 = sel[0][0], sel[-1][1]


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def clip(iv):
    return [(max(s, t0), min(e, t1)) for s, e in iv if e > t0 and s < t1]


def tot(iv):
    return sum(e - s for s, e in iv)


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


K = merge(clip([(k[0], k[1]) for k in ks]))
M = merge(clip([(m[0], m[1]) for m in ms if m[3] == 1]))
W = t1 - t0
k, m, both = tot(K), tot(M), inter(K, M)
print('window %.2f s: kernel busy %.1f%%, H2D copy busy %.1f%%, both %.1f%%, neither %.1f%%, copy with no kernel %.1f%%' % (
    W / 1e9, 100 * k / W, 100 * m / W, 100 * both / W, 100 * (W - k - m + both) / W, 100 * (m - both) / W))
print('H2D bytes in the window: %.1f GB' % (sum(x[2] for x in ms if x[3] == 1 and x[0] >= t0 and x[1] <= t1) / 1e9))
d, n = defaultdict(float), defaultdict(int)
for s, e, nm in ks:
    if e > t0 and s < t1:
        d[names.get(nm, nm)] += min(e, t1) - max(s, t0)
        n[names.get(nm, nm)] += 1
tk = sum(d.values())
for nm, v in sorted(d.items(), key=lambda x: -x[1])[:16]:
    print('%-34s %7.0f ms %5.1f%% of kernel time, %6d launches, %.1f us each' % (nm, v / 1e6, 100 * v / tk, n[nm], v / n[nm] / 1e3))
gaps = [K[i + 1][0] - K[i][1] for i in range(len(K) - 1)]
for lo, hi in [(100e3, 1e6), (1e6, 10e6), (10e6, 50e6), (50e6, 1e15)]:
    gg = [g for g in gaps if lo <= g < hi]
    print('gaps with no kernel, %.1f-%s ms: %d, total %.0f ms' % (lo / 1e6, 'inf' if hi > 1e14 else '%.0f' % (hi / 1e6), len(gg), sum(gg) / 1e6))
