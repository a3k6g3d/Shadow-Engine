"""pair_rounds.py X.sqlite : per DFlash round (df_attention launches < 2 ms apart; intervals < 300 ms between rounds) busy time per device, overlap between devices,
idle, copies by kind, and the segment timeline of a typical round."""
import sqlite3, sys, bisect
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
c = db.cursor()
names = {r[0]: r[1] for r in c.execute("select id,value from StringIds")}
ks = c.execute("select start,end,shortName,deviceId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ms = c.execute("select start,end,bytes,copyKind,srcKind,dstKind,deviceId,streamId from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
devs = sorted({k[3] for k in ks})
print('devices with kernels:', devs, ' kernels per device:', {d: sum(1 for k in ks if k[3] == d) for d in devs})
df = [k for k in ks if names.get(k[2], '') == 'df_attention']
print('df_attention on devices:', sorted({k[3] for k in df}))
rounds = []
for s, e, *_ in df:
    if rounds and s - rounds[-1][1] < 2e6:
        rounds[-1][1] = e
    else:
        rounds.append([s, e])
wins = [(rounds[i][0], rounds[i + 1][0]) for i in range(len(rounds) - 1) if rounds[i + 1][0] - rounds[i][0] < 300e6]
n = len(wins)
W = sum(b - a for a, b in wins)
print('%d decode rounds, %.2f s, %.1f ms a round' % (n, W / 1e9, W / n / 1e6))


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def clip(iv, t0, t1):
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


K = {d: merge([(k[0], k[1]) for k in ks if k[3] == d]) for d in devs}
allk = merge([(k[0], k[1]) for k in ks])
busy = defaultdict(float)
anyb = 0.0
both = 0.0
for t0, t1 in wins:
    ck = {d: clip(K[d], t0, t1) for d in devs}
    for d in devs:
        busy[d] += sum(e - s for s, e in ck[d])
    a = merge([x for d in devs for x in ck[d]])
    anyb += sum(e - s for s, e in a)
    if len(devs) > 1:
        both += inter(ck[devs[0]], ck[devs[1]])
for d in devs:
    print('device %d: kernels busy %.1f ms a round (%.0f%%)' % (d, busy[d] / n / 1e6, 100 * busy[d] / W))
print('some device busy %.1f ms (%.0f%%), neither %.1f ms a round' % (anyb / n / 1e6, 100 * anyb / W, (W - anyb) / n / 1e6))
if len(devs) > 1:
    print('both devices busy at once: %.2f ms a round' % (both / n / 1e6))

# copies in the windows by kind
ck = defaultdict(lambda: [0, 0.0, 0])
for m in ms:
    i = bisect.bisect_left([w[0] for w in wins], m[0]) - 1
    if i >= 0 and m[0] < wins[i][1]:
        key = (m[3], m[6])
        ck[key][0] += 1
        ck[key][1] += (m[1] - m[0])
        ck[key][2] += m[2]
print('copies a round by (kind 1=H2D 2=D2H 8=D2D 10=P2P, device): count, ms, MB')
for key, v in sorted(ck.items()):
    print('  ', key, '%.1f  %.2f ms  %.2f MB' % (v[0] / n, v[1] / n / 1e6, v[2] / n / 1e6))

# kernels by name per device
for d in devs:
    t, cnt = defaultdict(float), defaultdict(int)
    for k in ks:
        if k[3] != d:
            continue
        i = bisect.bisect_left([w[0] for w in wins], k[0]) - 1
        if i >= 0 and k[0] < wins[i][1]:
            nm = names.get(k[2], '?')
            t[nm] += k[1] - k[0]
            cnt[nm] += 1
    print('device %d top kernels a round (ms, launches):' % d)
    for nm, v in sorted(t.items(), key=lambda x: -x[1])[:12]:
        print('   %-44s %6.2f ms %6.1f' % (nm[:44], v / n / 1e6, cnt[nm] / n))

# one typical round: segments (runs of kernels per device with gaps under 30 us)
mid = wins[len(wins) // 2]
mid = sorted(wins, key=lambda w: abs((w[1] - w[0]) - W / n))[0]
print('typical round: %.1f ms' % ((mid[1] - mid[0]) / 1e6))
seg = []
for k in ks:
    if mid[0] <= k[0] < mid[1]:
        d = k[3]
        if seg and seg[-1][0] == d and k[0] - seg[-1][2] < 30e3:
            seg[-1][2] = max(seg[-1][2], k[1])
            seg[-1][3] += 1
        else:
            seg.append([d, k[0], k[1], 1])
for d, s, e, cnt in seg:
    if e - s > 150e3:
        print('  dev %d  +%7.2f ms  %6.2f ms  %4d kernels' % (d, (s - mid[0]) / 1e6, (e - s) / 1e6, cnt))
