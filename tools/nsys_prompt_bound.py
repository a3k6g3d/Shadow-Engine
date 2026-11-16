"""prompt_bound.py X.sqlite : long-prompt passes on one card. Splits the capture into bursts (gaps > 300 ms between kernels), keeps those > 2 s (the prompts), and per burst
reports wall time, kernel-busy, H2D-copy-busy, both, neither, bytes copied, and the kernel time by name (top)."""
import sqlite3, sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
ks = c.execute("select start,end,shortName,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ms = c.execute("select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
bursts = []
for k in ks:
    if bursts and k[0] - bursts[-1][1] < 300e6:
        bursts[-1][1] = max(bursts[-1][1], k[1])
    else:
        bursts.append([k[0], k[1]])
bursts = [b for b in bursts if b[1] - b[0] > 2e9]
print('bursts > 2 s:', [round((b[1] - b[0]) / 1e9, 2) for b in bursts])


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


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


for n, (t0, t1) in enumerate(bursts):
    W = t1 - t0
    kk = merge([(max(k[0], t0), min(k[1], t1)) for k in ks if k[1] > t0 and k[0] < t1])
    h2d = [(max(m[0], t0), min(m[1], t1)) for m in ms if m[3] == 1 and m[1] > t0 and m[0] < t1]
    mm = merge(h2d)
    kb, mb, both = sum(e - s for s, e in kk), sum(e - s for s, e in mm), inter(kk, mm)
    byts = sum(m[2] for m in ms if m[3] == 1 and t0 <= m[0] < t1)
    print('burst %d: %.2f s; kernels busy %.2f s (%.0f%%), H2D copies busy %.2f s (%.0f%%), both %.2f s, neither %.2f s; H2D %.1f GiB (%.1f GB/s while copying)' % (
        n, W / 1e9, kb / 1e9, 100 * kb / W, mb / 1e9, 100 * mb / W, both / 1e9, (W - kb - mb + both) / 1e9, byts / 2**30, byts / max(mb, 1)))
    t, cnt = defaultdict(float), defaultdict(int)
    for k in ks:
        if t0 <= k[0] < t1:
            nm = names.get(k[2], '?')
            t[nm] += k[1] - k[0]
            cnt[nm] += 1
    tot = sum(t.values())
    for nm, v in sorted(t.items(), key=lambda x: -x[1])[:14]:
        print('   %-46s %7.1f ms %5.1f%% %7d launches' % (nm[:46], v / 1e6, 100 * v / tot, cnt[nm]))
