"""Fetches chosen tensors of an EXL3 repo branch by HTTP range requests into one small valid safetensors file.
usage: python fetch_exl3_subset.py REPO REV OUT.safetensors REGEX [REGEX...]   (tensor names matching any regex are fetched)"""
import json, re, struct, sys, urllib.request
from huggingface_hub import HfApi, get_token, hf_hub_url

repo, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
pats = [re.compile(p) for p in sys.argv[4:]]
tok = get_token()
api = HfApi()
md = api.get_safetensors_metadata(repo, revision=rev)
want = []   # (name, file, dtype, shape, start, end) absolute byte offsets in the file
for fn, m in md.files_metadata.items():
    url = hf_hub_url(repo, fn, revision=rev)
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-7', **({'Authorization': f'Bearer {tok}'} if tok else {})})
    hlen = struct.unpack('<Q', urllib.request.urlopen(req).read())[0]
    for name, t in m.tensors.items():
        if any(p.search(name) for p in pats):
            s, e = t.data_offsets
            want.append((name, fn, t.dtype, t.shape, 8 + hlen + s, 8 + hlen + e))
print(len(want), 'tensors,', sum(w[5] - w[4] for w in want) / 1e6, 'MB')
header, blobs, off = {}, [], 0
for name, fn, dt, shape, a, b in sorted(want):
    url = hf_hub_url(repo, fn, revision=rev)
    req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b - 1}', **({'Authorization': f'Bearer {tok}'} if tok else {})})
    data = urllib.request.urlopen(req).read()
    assert len(data) == b - a, (name, len(data), b - a)
    header[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [off, off + len(data)]}
    blobs.append(data)
    off += len(data)
h = json.dumps(header, separators=(',', ':')).encode()
h += b' ' * ((8 - len(h) % 8) % 8)
with open(out, 'wb') as f:
    f.write(struct.pack('<Q', len(h)))
    f.write(h)
    for d in blobs:
        f.write(d)
print('wrote', out, off, 'bytes of tensor data')
