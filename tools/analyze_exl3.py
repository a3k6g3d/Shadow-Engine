"""Reads the safetensors headers of an EXL3 repo by range requests and summarizes what ShadowZero's loader cares about: total size, the bit rate K of every routed expert
projection (per layer: uniform or mixed), the dense tensors' K, the n-gram table, MTP / vision presence.  usage: analyze_exl3.py REPO [REVISION]"""
import json, re, struct, sys, collections, urllib.request
from huggingface_hub import HfApi, get_token, hf_hub_url

repo = sys.argv[1]
rev = sys.argv[2] if len(sys.argv) > 2 else None
api = HfApi()
tok = get_token()
try:
    tree = [f for f in api.list_repo_tree(repo, revision=rev, recursive=True) if hasattr(f, 'size')]
except Exception as e:
    print(repo, 'ERROR', str(e)[:100])
    sys.exit(0)
sizes = {f.path: f.size for f in tree}
total = sum(sizes.values())
shards = sorted(p for p in sizes if p.endswith('.safetensors'))
ng = sum(s for p, s in sizes.items() if 'ngram' in p)


def rng(path, a, b):
    req = urllib.request.Request(hf_hub_url(repo, path, revision=rev), headers={'Range': f'bytes={a}-{b}', 'Authorization': f'Bearer {tok}'})
    return urllib.request.urlopen(req, timeout=120).read()


layer_k = collections.defaultdict(set)   # layer -> set of K of routed expert matrices
dense_k = collections.Counter()
other = collections.Counter()
for p in shards:
    if 'ngram' in p:
        continue
    n = struct.unpack('<Q', rng(p, 0, 7))[0]
    hdr = json.loads(rng(p, 8, 8 + n - 1))
    for name, t in hdr.items():
        if name == '__metadata__' or not name.endswith('.trellis'):
            continue
        k = t['shape'][-1] / 16
        m = re.search(r'layers\.(\d+)\.mlp\.experts\.(\d+)\.', name)
        if m:
            layer_k[int(m.group(1))].add(k)
        else:
            if 'mtp' in name:
                other['mtp K%g' % k] += 1
            elif 'visual' in name or 'vision' in name:
                other['vision K%g' % k] += 1
            else:
                dense_k[k] += 1
uni = collections.Counter()
mixed = 0
for l, ks in layer_k.items():
    if len(ks) == 1:
        uni[next(iter(ks))] += 1
    else:
        mixed += 1
exp_bytes = 0
print(f'{repo}{"@" + rev if rev else ""}: {total / 2**30:.1f} GiB total, n-gram table {ng / 2**30:.1f} GiB, weights {(total - ng) / 2**30:.1f} GiB')
print('   routed-expert layers (K uniform in the layer):', {('%g' % a): b for a, b in uni.items()}, 'mixed-K layers:', mixed, '| dense K:', {('%g' % a): b for a, b in dense_k.items()}, '| other:', dict(other))
