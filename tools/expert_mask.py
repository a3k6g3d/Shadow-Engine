"""Make a `SHADOWZERO_EXPERT_MASK` file (n_layer x n_expert bytes, 1 = the expert stays, 0 = pruned at run time) from a routing profile (`profiles/routing-*.shrc`, per-layer counts of how often
each expert was chosen on a recorded set of prompts): the K most-used experts of every layer stay.

    python tools/expert_mask.py PROFILE.shrc K OUT.bin           [--scores SCORES.npy]

`--scores` replaces the profile's counts by a (n_layer, n_expert) float array (e.g. the saliency of STATUS 168) for the ranking.  The same K in every layer: the config holds one expert count.
Needs numpy (the exllamav3 venv has it)."""
import struct
import sys

import numpy as np

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
prof, k, out = argv[0], int(argv[1]), argv[2]
raw = open(prof, "rb").read()
magic, ver, nl, ne = struct.unpack("<4sIII", raw[:16])
assert magic == b"SHRC", magic
counts = np.frombuffer(raw[24:24 + nl * ne * 8], dtype="<u8").reshape(nl, ne).astype(np.float64)
if "--scores" in sys.argv:
    counts = np.load(sys.argv[sys.argv.index("--scores") + 1]).astype(np.float64).reshape(nl, ne)
mask = np.zeros((nl, ne), dtype=np.uint8)
for l in range(nl):
    keep = np.argsort(-counts[l], kind="stable")[:k]
    mask[l, keep] = 1
open(out, "wb").write(mask.tobytes())
share = (counts * mask).sum(1) / counts.sum(1)
print(f"{k} of {ne} experts a layer kept; share of the profile's selections kept: mean {share.mean():.3f}, worst layer {share.min():.3f}")
