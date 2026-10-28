"""Replays a routing log (SHADOW_ROUTE_LOG) through VRAM expert-cache policies, per layer, and prints each one's hit rate.

    python tools/cache_sim.py ROUTE_LOG CACHE_DUMP [--skip N]

CACHE_DUMP is what SHADOW_CACHE_DUMP writes when a session starts: per layer its slot count, how many of them are
evictable, and the pinned experts in placement order.  --skip N leaves the first N tokens (the prompt) out of the count
(they still warm the cache).  Every policy gets the same number of slots per layer as the session had; a token's own
experts are never evicted by its own misses (as `ExpertCache::victim`).
"""
import struct
import sys
from collections import OrderedDict


def load_log(path):
    b = open(path, 'rb').read()
    n = struct.unpack_from('<I', b, 0)[0]
    at = 4 + 4 * n
    k = struct.unpack_from('<I', b, at)[0]
    at += 4
    rec = (k + 1) * 2
    recs = [struct.unpack_from('<%dH' % (k + 1), b, at + i * rec) for i in range((len(b) - at) // rec)]
    layers = max(r[0] for r in recs) + 1
    toks = [recs[i:i + layers] for i in range(0, len(recs) // layers * layers, layers)]
    return toks, layers


def load_dump(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        out[int(p[0])] = (int(p[1]), int(p[2]), [int(x) for x in p[3].split(',')] if len(p) > 3 and p[3] else [])
    return out


class PinnedLru:
    """`pin` experts always resident, the rest of the slots LRU."""

    def __init__(self, slots, pin):
        self.pin = set(pin)
        self.cap = slots - len(self.pin)
        self.lru = OrderedDict()

    def access(self, experts):
        hits = 0
        for e in experts:
            if e in self.pin:
                hits += 1
            elif e in self.lru:
                hits += 1
                self.lru.move_to_end(e)
        for e in experts:
            if e in self.pin or e in self.lru:
                self.lru.move_to_end(e) if e in self.lru else None
                continue
            while len(self.lru) >= self.cap:
                victim = next(x for x in self.lru if x not in experts)
                del self.lru[victim]
            self.lru[e] = True
        return hits


class DecayLfu:
    """Every slot evictable; the victim is the resident expert with the lowest decayed use count (prior: placement)."""

    def __init__(self, slots, prior, decay):
        self.cap = slots
        self.decay = decay
        self.score = {e: (len(prior) - i) / len(prior) for i, e in enumerate(prior)}
        self.res = set(prior[:slots])

    def access(self, experts):
        hits = sum(e in self.res for e in experts)
        for e in list(self.score):
            self.score[e] *= self.decay
        for e in experts:
            self.score[e] = self.score.get(e, 0.0) + 1.0
        for e in experts:
            if e in self.res:
                continue
            if len(self.res) >= self.cap:
                victim = min((x for x in self.res if x not in experts), key=lambda x: self.score.get(x, 0.0))
                if self.score.get(victim, 0.0) >= self.score[e] and len(self.res) >= self.cap:
                    # admission: the newcomer replaces only a colder resident (it is still copied for this token)
                    continue
                self.res.discard(victim)
            self.res.add(e)
        return hits


def run(toks, layers, dump, make, skip):
    hits = total = 0
    sims = {l: make(*dump[l]) for l in range(layers) if l in dump}
    for t, tok in enumerate(toks):
        for l in range(layers):
            if l not in sims:
                continue
            ex = list(tok[l][1:])
            h = sims[l].access(ex)
            if t >= skip:
                hits += h
                total += len(ex)
    return hits / max(total, 1)


def main():
    toks, layers = load_log(sys.argv[1])
    dump = load_dump(sys.argv[2])
    skip = int(sys.argv[sys.argv.index('--skip') + 1]) if '--skip' in sys.argv else 0
    print(f"{len(toks)} tokens ({skip} not counted), {layers} layers, "
          f"{sum(d[0] for d in dump.values())} slots, {sum(d[1] for d in dump.values())} evictable")
    pol = {
        'pinned + LRU (today)': lambda s, ev, pin: PinnedLru(s, pin),
        'pinned + LRU, margin 24': lambda s, ev, pin: PinnedLru(s, pin[:max(0, s - 24)]),
        'pinned + LRU, margin 48': lambda s, ev, pin: PinnedLru(s, pin[:max(0, s - 48)]),
        'pure LRU': lambda s, ev, pin: PinnedLru(s, []),
    }
    for d in (0.99, 0.995, 0.999):
        pol[f'decayed LFU {d}'] = (lambda d: lambda s, ev, pin: DecayLfu(s, pin, d))(d)
    for name, make in pol.items():
        print(f"{name:28s} {100 * run(toks, layers, dump, make, skip):5.1f}% hits")


if __name__ == '__main__':
    main()
