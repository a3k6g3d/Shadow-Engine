"""Slices of the base checkpoint's fused expert tensors (gate_up_proj [512, 1280, 2560], down_proj [512, 2560, 640], BF16) for
chosen layers and experts, by range requests, written as a safetensors file with per-expert names (gate_proj / up_proj /
down_proj .weight).  usage: python fetch_base_experts.py OUT LAYERS(comma) EXPERTS(comma)"""
import json, struct, sys, urllib.request
from huggingface_hub import HfApi, get_token, hf_hub_url

out, layers, experts = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], [int(x) for x in sys.argv[3].split(',')]
repo = 'Qwen/Qwen3.8-Flash-Next'
tok = get_token()
hdr = {'Authorization': f'Bearer {tok}'} if tok else {}
md = HfApi().get_safetensors_metadata(repo)
loc = {}
for fn, m in md.files_metadata.items():
    for name, t in m.tensors.items():
        if name.startswith('model.language_model.layers.') and (name.endswith('mlp.experts.gate_up_proj') or name.endswith('mlp.experts.down_proj')):
            loc[name] = (fn, t.data_offsets, t.shape)
hlen = {}


def rng(fn, a, b):
    url = hf_hub_url(repo, fn)
    return urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={a}-{b - 1}', **hdr})).read()


def data_start(fn):
    if fn not in hlen:
        hlen[fn] = 8 + struct.unpack('<Q', rng(fn, 0, 8))[0]
    return hlen[fn]


header, blobs, off = {}, [], 0
for l in layers:
    for kind in ('gate_up_proj', 'down_proj'):
        name = f'model.language_model.layers.{l}.mlp.experts.{kind}'
        fn, (s, e), shape = loc[name]
        per = (e - s) // shape[0]
        for ex in experts:
            a = data_start(fn) + s + ex * per
            data = rng(fn, a, a + per)
            assert len(data) == per
            if kind == 'gate_up_proj':
                half = per // 2
                parts = [('gate_proj', data[:half], [shape[1] // 2, shape[2]]), ('up_proj', data[half:], [shape[1] // 2, shape[2]])]
            else:
                parts = [('down_proj', data, [shape[1], shape[2]])]
            for pn, d, sh in parts:
                key = f'model.language_model.layers.{l}.mlp.experts.{ex}.{pn}.weight'
                header[key] = {'dtype': 'BF16', 'shape': sh, 'data_offsets': [off, off + len(d)]}
                blobs.append(d)
                off += len(d)
h = json.dumps(header, separators=(',', ':')).encode()
h += b' ' * ((8 - len(h) % 8) % 8)
with open(out, 'wb') as f:
    f.write(struct.pack('<Q', len(h)))
    f.write(h)
    for d in blobs:
        f.write(d)
print('wrote', out, off, 'bytes,', len(header), 'tensors')
