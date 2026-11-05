"""The steady-state view of an Nsight Systems export (SQLite) with several prompts in it: only the LAST prompt's window (a prompt's window starts at its first
`select_blocks_rows`; prompts are separated by more than 1 s without one).  Kernel / copy busy shares, the kernels by time, the gaps with no kernel, and the host's CUDA API time.

  py tools/nsys_last_run.py X.sqlite"""
import sqlite3, sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
ks = c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ms = c.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 order by start").fetchall()
sel = [k for k in ks if names.get(k[2], '') == 'select_blocks_rows']
starts = [0]
for i in range(1, len(sel)):
    if sel[i][0] - sel[i - 1][1] > 3e9:
        starts.append(i)
groups = [sel[a:b] for a, b in zip(starts, starts[1:] + [len(sel)])]
big = [g for g in groups if len(g) >= 100]   # a prompt's window; a few launches are a decode step or a short request
last = max(big[-2:] or groups, key=len)       # the biggest of the last two (a repeated prompt can reuse a cached prefix and be short)
t0, t1 = last[0][0], last[-1][1]


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def clip(iv):
    return [[max(s, t0), min(e, t1)] for s, e in iv if e > t0 and s < t1]


def total(iv):
    return sum(e - s for s, e in iv)


kw = ks
kv = merge(clip([(k[0], k[1]) for k in kw]))
cv = merge(clip([(m[0], m[1]) for m in ms]))
T = t1 - t0
kt, ct = total(kv), total(cv)
both = 0
j = 0
for s, e in kv:
    for s2, e2 in cv:
        both += max(0, min(e, e2) - max(s, s2))
print(f"last prompt window {T / 1e9:.2f} s: kernel busy {100 * kt / T:.1f}%, H2D copy busy {100 * ct / T:.1f}%, both {100 * both / T:.1f}%, neither {100 * (T - kt - ct + both) / T:.1f}%, "
      f"copy with no kernel {100 * (ct - both) / T:.1f}%; copied {sum(m[2] for m in ms if t0 <= m[0] <= t1) / 1e9:.1f} GB")
tm = defaultdict(float)
cn = defaultdict(int)
for k in kw:
    if t0 <= k[0] <= t1:
        n = names[k[2]]
        tm[n] += (k[1] - k[0]) / 1e6
        cn[n] += 1
tot = sum(tm.values())
for n, v in sorted(tm.items(), key=lambda x: -x[1])[:14]:
    print(f"{n:40s} {v:7.0f} ms {100 * v / tot:5.1f}%  {cn[n]:6d} launches {1000 * v / cn[n]:8.1f} us each")
gaps = defaultdict(lambda: [0, 0.0])
for a, b in zip(kv, kv[1:]):
    g = (b[0] - a[1]) / 1e6
    k = '<0.1' if g < 0.1 else '0.1-1' if g < 1 else '1-10' if g < 10 else '10-50' if g < 50 else '50+'
    gaps[k][0] += 1
    gaps[k][1] += g
for k in ['<0.1', '0.1-1', '1-10', '10-50', '50+']:
    print(f"gaps {k:6s} ms: {gaps[k][0]:5d} gaps, {gaps[k][1]:8.0f} ms")
api = c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?", (t0, t1)).fetchall()
at = defaultdict(float)
ac = defaultdict(int)
for s, e, n in api:
    at[names[n]] += (e - s) / 1e6
    ac[names[n]] += 1
for n, v in sorted(at.items(), key=lambda x: -x[1])[:6]:
    print(f"API {n:36s} {v:8.0f} ms {ac[n]:7d} calls")
