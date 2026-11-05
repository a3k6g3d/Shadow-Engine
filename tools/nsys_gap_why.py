"""For the biggest-prompt window of an Nsight Systems export: what the host was doing during each GPU gap of 5 ms or more.

  py tools/nsys_gap_why.py X.sqlite [first_gap_index] [count]

Prints, per gap: its length, the kernels around it, the CUDA API calls overlapping it that last more than 1 ms (name and length), and the copies running."""
import sqlite3, sys

db = sqlite3.connect(sys.argv[1])
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
ks = c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ms = c.execute("select start,end,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 order by start").fetchall()
sel = [k for k in ks if names.get(k[2], '') == 'select_blocks_rows']
starts = [0] + [i for i in range(1, len(sel)) if sel[i][0] - sel[i - 1][1] > 1e9]
groups = [sel[a:b] for a, b in zip(starts, starts[1:] + [len(sel)])]
last = [g for g in groups if len(g) >= 100][-1]
t0, t1 = last[0][0], last[-1][1]
ks = [k for k in ks if t0 <= k[0] <= t1]
api = c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where end>=? and start<=? and end-start>1000000 order by start", (t0, t1)).fetchall()
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
count = int(sys.argv[3]) if len(sys.argv) > 3 else 30
gaps = [(a[1], b[0], names[a[2]], names[b[2]]) for a, b in zip(ks, ks[1:]) if b[0] - a[1] > 5e6]
print(len(gaps), 'gaps of 5 ms or more; total %.0f ms' % (sum(g[1] - g[0] for g in gaps) / 1e6))
for g0, g1, na, nb in gaps[first:first + count]:
    calls = [(names[n], (min(e, g1) - max(s, g0)) / 1e6, (e - s) / 1e6) for s, e, n in api if e > g0 and s < g1]
    cp = sum(max(0, min(e, g1) - max(s, g0)) for s, e, _ in ms) / (g1 - g0)
    print("t=%7.0f gap %5.1f ms %-26s -> %-22s copy %.2f | %s" % ((g0 - t0) / 1e6, (g1 - g0) / 1e6, na[:26], nb[:22], cp,
          '; '.join("%s %.1f/%.1f ms" % x for x in calls) or '-'))
