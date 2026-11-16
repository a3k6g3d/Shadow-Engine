"""mix_checkpoint.py : a new EXL3 checkpoint from two quants of the same weights, tensor bytes copied (streamed, a few MB of RAM): everything from BASE except the routed experts
of the layers in LAYERS, which come from OTHER.  Same shard names and layout as BASE, a new index, the other files copied, the n-gram table hard-linked.

  python -I mix_checkpoint.py BASE OTHER OUT 0,1,2,...[,dense]   (dense: also every non-expert tensor of the layers except the PLE module, and lm_head)
"""
import json, os, re, shutil, struct, sys

base, other, out = sys.argv[1], sys.argv[2], sys.argv[3]
dense = 'dense' in sys.argv[4].split(',')
layers = {int(x) for x in sys.argv[4].split(',') if x != 'dense'}
os.makedirs(out, exist_ok=True)
EXP = re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.')


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def tensors_of(d, shards):
    t = {}
    for sh in shards:
        h, off = read_header(os.path.join(d, sh))
        for k, v in h.items():
            if k != '__metadata__':
                t[k] = (os.path.join(d, sh), off + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])
    return t


bidx = json.load(open(os.path.join(base, 'model.safetensors.index.json')))
oidx = json.load(open(os.path.join(other, 'model.safetensors.index.json')))
bshards = sorted(set(bidx['weight_map'].values()))
btens = tensors_of(base, bshards)
otens = tensors_of(other, sorted(set(oidx['weight_map'].values())))
LAY = re.compile(r'model\.language_model\.layers\.(\d+)\.(.*)')
def wanted(k):
    m = LAY.match(k)
    if m:
        rest = m.group(2)
        if rest.startswith('mlp.experts.'):
            return int(m.group(1)) in layers
        return dense and not rest.startswith('ple.')
    return dense and k.startswith('lm_head.')
swap = {k for k in btens if wanted(k)}
missing = [k for k in swap if k not in otens]
assert not missing, missing[:3]
print('tensors', len(btens), 'taken from OTHER', len(swap), flush=True)

new_map = {}
total = 0
for sh in bshards:
    h, off = read_header(os.path.join(base, sh))
    meta = h.get('__metadata__')
    keys = sorted((k for k in h if k != '__metadata__'), key=lambda k: h[k]['data_offsets'][0])
    hdr = {}
    if meta is not None:
        hdr['__metadata__'] = meta
    pos = 0
    plan = []
    for k in keys:
        src = otens[k] if k in swap else btens[k]
        _, _, n, dtype, shape = src
        hdr[k] = {'dtype': dtype, 'shape': shape, 'data_offsets': [pos, pos + n]}
        plan.append((k, src))
        pos += n
    hb = json.dumps(hdr, separators=(',', ':')).encode()
    hb += b' ' * ((8 - len(hb) % 8) % 8)
    with open(os.path.join(out, sh), 'wb') as fo:
        fo.write(struct.pack('<Q', len(hb)))
        fo.write(hb)
        for k, (path, start, n, _, _) in plan:
            with open(path, 'rb') as fi:
                fi.seek(start)
                left = n
                while left:
                    b = fi.read(min(left, 16 << 20))
                    assert b, 'short read ' + k
                    fo.write(b)
                    left -= len(b)
            new_map[k] = sh
    total += pos
    print(sh, round(pos / 2**30, 2), 'GiB', len(keys), 'tensors', flush=True)

json.dump({'metadata': {'total_size': total}, 'weight_map': new_map}, open(os.path.join(out, 'model.safetensors.index.json'), 'w'), indent=2)
for f in os.listdir(base):
    if f in bshards or f == 'model.safetensors.index.json' or f == 'README.md':
        continue
    src, dst = os.path.join(base, f), os.path.join(out, f)
    if os.path.isdir(src) or os.path.exists(dst):
        continue
    if f.endswith('.safetensors'):
        os.link(src, dst)   # the n-gram table: same volume, no copy
    else:
        shutil.copy2(src, dst)
open(os.path.join(out, 'MIX.txt'), 'w').write(f'Lygodactylus 2.05 bpw with the routed experts of layers {sorted(layers)} taken from the 3.05 bpw quant (same weights); made by mix_checkpoint.py (STATUS 165).\n')
print('done', flush=True)
