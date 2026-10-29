"""Reference check of the GPU DFlash 2 drafter (crates/shadow-cuda/src/session_dflash.rs) in CPU torch.

Run shadow with SHADOW_DF_DUMP=DIR (an existing, empty directory): it appends the features the drafter was given
(feats.bin) and every draft's input rows, final hidden, candidates, selector inputs and picks (drafts.bin).  This
script rebuilds the drafter from models/dflash2/model.safetensors, recomputes each draft from the same features, and
prints how far the GPU's block hidden and selector picks are from it.

    python tools/dflash2_ref.py DIR [models/dflash2/model.safetensors] [--max N]
"""
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

H, NH, NKV, HD, FF, LAYERS, BLOCK, RANK, TOPK, GROUP, EPS, THETA = 2560, 32, 8, 128, 7680, 5, 8, 256, 16, 16, 1e-6, 10000.0
WINDOW = 2048


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos):
    # x [rows, heads, 128], neox style over the full head
    half = HD // 2
    inv = THETA ** (-torch.arange(half, dtype=torch.float32) / half)
    ang = pos[:, None].float() * inv[None]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)


def conv(x, delta, base, side):
    # x [rows, H], delta [rows, 2, 2, groups] (side, tap, group)
    rows = x.shape[0]
    groups = H // GROUP
    d = delta[:, side].reshape(rows, 2, groups, 1)
    b = base[side].reshape(2, groups, GROUP)
    coef = b[None] + d  # [rows, 2, groups, GROUP]
    xb = x.reshape(rows, groups, GROUP)
    out = coef[:, 0] * xb
    prev = torch.cat([torch.zeros_like(xb[:1]), xb[:-1]], 0)
    prev[0] = 0
    out = out + coef[:, 1] * prev
    return out.reshape(rows, H)


def main():
    d = sys.argv[1]
    st = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "models/dflash2/model.safetensors"
    mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 3
    W = {}
    with safe_open(st, "pt") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k).float()
    nfeat = 5 * H
    feats = {}
    raw = open(f"{d}/feats.bin", "rb").read()
    at = 0
    while at < len(raw):
        p0, m = struct.unpack_from("<II", raw, at)
        at += 8
        arr = np.frombuffer(raw, dtype="<f4", count=m * nfeat, offset=at).reshape(m, nfeat)
        at += m * nfeat * 4
        for j in range(m):
            feats[p0 + j] = arr[j]
    dr = open(f"{d}/drafts.bin", "rb").read()
    at = 0
    done = 0
    sizes = [BLOCK * H, BLOCK * H, 7 * TOPK, 7 * TOPK, 7 * RANK, 7]
    while at < len(dr) and done < mx:
        n, nxt = struct.unpack_from("<II", dr, at)
        at += 8
        parts = []
        for k, sz in enumerate(sizes):
            dt = "<u4" if k in (2, 5) else "<f4"
            parts.append(np.frombuffer(dr, dtype=dt, count=sz, offset=at))
            at += sz * 4
        emb, hid, cand, unary, hp, out = parts
        if n < 24 and done == 0 and len(dr) > 10_000_000:
            pass
        ctx = torch.from_numpy(np.stack([feats[i] for i in range(n)]).copy())
        ch = rms(ctx @ W["fc.weight"].T, W["hidden_norm.weight"])
        pos_ctx = torch.arange(n)
        ks, vs = [], []
        for i in range(LAYERS):
            p = f"layers.{i}.self_attn"
            k = (ch @ W[f"{p}.k_proj.weight"].T).reshape(n, NKV, HD)
            k = rope(rms(k, W[f"{p}.k_norm.weight"]), pos_ctx)
            v = (ch @ W[f"{p}.v_proj.weight"].T).reshape(n, NKV, HD)
            ks.append(k)
            vs.append(v)
        x = torch.from_numpy(emb.reshape(BLOCK, H).copy())
        res = None
        pos_blk = torch.arange(n, n + BLOCK)
        for i in range(LAYERS):
            pre, p = f"layers.{i}", f"layers.{i}.self_attn"
            if res is None:
                res = x
                xn = rms(x, W[f"{pre}.input_layernorm.weight"])
            else:
                res = x + res
                xn = rms(res, W[f"{pre}.input_layernorm.weight"])
            kp = xn @ W[f"{pre}.attention_conv.kernel_projection.weight"].T
            kp = kp.reshape(BLOCK, 2, 2, H // GROUP)
            xc = conv(xn, kp, W[f"{pre}.attention_conv.base_kernel"], 0)
            q = (xc @ W[f"{p}.q_proj.weight"].T).reshape(BLOCK, NH, HD)
            k = (xc @ W[f"{p}.k_proj.weight"].T).reshape(BLOCK, NKV, HD)
            v = (xc @ W[f"{p}.v_proj.weight"].T).reshape(BLOCK, NKV, HD)
            q = rope(rms(q, W[f"{p}.q_norm.weight"]), pos_blk)
            k = rope(rms(k, W[f"{p}.k_norm.weight"]), pos_blk)
            K = torch.cat([ks[i], k], 0)
            V = torch.cat([vs[i], v], 0)
            kpos = torch.arange(n + BLOCK)
            ao = torch.zeros(BLOCK, NH, HD)
            for t in range(BLOCK):
                lo = max(0, n + t + 1 - WINDOW)
                sel = (kpos >= lo)
                Ks, Vs = K[sel], V[sel]
                for h in range(NH):
                    s = (Ks[:, h // 4] @ q[t, h]) * HD ** -0.5
                    ao[t, h] = torch.softmax(s, 0) @ Vs[:, h // 4]
            o = ao.reshape(BLOCK, NH * HD) @ W[f"{p}.o_proj.weight"].T
            o = conv(o, kp, W[f"{pre}.attention_conv.base_kernel"], 1)
            res = o + res
            xn2 = rms(res, W[f"{pre}.post_attention_layernorm.weight"])
            mp = (xn2 @ W[f"{pre}.mlp_conv.kernel_projection.weight"].T).reshape(BLOCK, 2, 2, H // GROUP)
            xm = conv(xn2, mp, W[f"{pre}.mlp_conv.base_kernel"], 0)
            g = xm @ W[f"{pre}.mlp.gate_proj.weight"].T
            u = xm @ W[f"{pre}.mlp.up_proj.weight"].T
            m_out = (torch.nn.functional.silu(g) * u) @ W[f"{pre}.mlp.down_proj.weight"].T
            x = conv(m_out, mp, W[f"{pre}.mlp_conv.base_kernel"], 1)
        res = x + res
        hid_ref = rms(res, W["norm.weight"])
        hid_gpu = torch.from_numpy(hid.reshape(BLOCK, H).copy())
        err = (hid_ref - hid_gpu).abs().max().item()
        rel = ((hid_ref - hid_gpu).norm() / hid_ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(hid_ref, hid_gpu, dim=-1)
        print(f"draft at n={n}: hidden max abs err {err:.4g}, relative {rel:.4g}, cosine per row {[round(c, 4) for c in cos.tolist()]}")
        # selector from the GPU's own candidates: hp and the walk
        hp_ref = hid_ref[1:] @ W["candidate_selector.hidden_projection.weight"].T
        print(f"  hidden_projection max err {(hp_ref - torch.from_numpy(hp.reshape(7, RANK).copy())).abs().max().item():.4g}")
        cand = cand.reshape(7, TOPK).astype(np.int64)
        unary = torch.from_numpy(unary.reshape(7, TOPK).copy())
        pred, succ = W["candidate_selector.predecessor_codebook"], W["candidate_selector.successor_codebook"]
        toks, prev = [], nxt
        for e in range(7):
            s = unary[e] + (succ[torch.from_numpy(cand[e])] @ (pred[prev] * hp_ref[e]))
            prev = int(cand[e][int(s.argmax())])
            toks.append(prev)
        print(f"  walk: reference {toks}, gpu {out.tolist()}")
        done += 1


main()
