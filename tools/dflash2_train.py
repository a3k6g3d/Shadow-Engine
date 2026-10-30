"""Fine-tune the DFlash 2 drafter (models/dflash2) on Shadow's own features.

Data: `feats.bin` files written by `shadow serve` with SHADOW_DF_DUMP=dir (see tools/dflash2_ref.py): per position the
five-layer feature row the main model produced and the token read.  Tables: token_embd.f32 / output.f32 from
`shadow dump MODEL token_embd.weight|output.weight FILE` (the main model's own embedding and head).

LoRA (fp32) on every linear of the drafter over frozen BF16 weights; the loss is what the drafter is used for:
full-vocabulary cross entropy of each mask slot's hidden state through the main model's head, plus a cross entropy over
the selector's 16 candidates scored like `df_walk` (unary logit + bilinear edge from the previous true token).  Eval:
the mean accepted prefix length of the greedy walk (the real thing: free-running, 7 slots) on held-out conversations.

    python tools/dflash2_train.py --data DIR [--out models/dflash2-ft] [--steps N] [--eval-only]
"""
import argparse
import glob
import math
import os
import shutil
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

H, NH, NKV, HD, LAYERS, BLOCK, RANK, TOPK, GROUP, EPS, THETA = 2560, 32, 8, 128, 5, 8, 256, 16, 16, 1e-6, 10000.0
NFEAT = 5 * H
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--tables", default=None, help="dir with token_embd.f32 and output.f32 (default: --data)")
ap.add_argument("--model", default="models/dflash2")
ap.add_argument("--out", default="models/dflash2-ft")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--anchors", type=int, default=16)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--maxlen", type=int, default=1100)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--eval-all", action="store_true", help="with --eval-only: every conversation, not the last tenth")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--max-seqs", type=int, default=0)
args = ap.parse_args()
dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
wdt = torch.float32 if dev == "cpu" else torch.bfloat16
torch.manual_seed(0)
np.random.seed(0)


# ------------------------------------------------------------------------------------------------------- data
def load_sequences(d):
    seqs = []
    for path in [p for one in d.split(",") for p in sorted(glob.glob(f"{one}/feats*.bin"))]:
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        at, n = 0, len(raw)
        toks, feats = None, None
        cur = None
        while at + 8 <= n:
            p0, m = struct.unpack_from("<II", raw, at)
            at += 8
            if at + 4 * m + 4 * m * NFEAT > n:
                break  # a record still being written
            t = np.frombuffer(raw, dtype="<u4", count=m, offset=at)
            at += 4 * m
            f = np.frombuffer(raw, dtype="<f4", count=m * NFEAT, offset=at).reshape(m, NFEAT)
            at += 4 * m * NFEAT
            if p0 == 0:
                if cur is not None:
                    seqs.append(cur)
                cur = {"tok": [], "feat": []}
            if cur is None or p0 > len(cur["tok"]):
                cur = None  # a gap: features for the start are missing
                continue
            del cur["tok"][p0:]
            del cur["feat"][p0:]
            cur["tok"].extend(t.tolist())
            cur["feat"].extend(f.astype(np.float16))  # one row each
        if cur is not None:
            seqs.append(cur)
    out = []
    for s in seqs:
        if len(s["tok"]) >= 48:
            tok = np.array(s["tok"], dtype=np.int64)
            out.append((tok, np.stack(s["feat"]), generated_mask(tok)))
    return out


IM_START, IM_END, THINK, END_THINK = 248045, 248046, 248068, 248069


def generated_mask(tok):
    """True where the main model generated the token (an assistant turn after its template header; a non-thinking turn's
    empty think block is part of the header).  Drafts only ever start inside these spans."""
    n = len(tok)
    m = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if tok[i] == IM_START and i + 2 < n and tok[i + 1] != 846:  # 846: "user"
            j = i + 3  # <|im_start|> assistant NEWLINE
            if j < n and tok[j] == THINK:
                k = j + 1
                while k < min(n, j + 4) and tok[k] != END_THINK:
                    k += 1
                if k < min(n, j + 4):
                    j = k + 2  # an empty think block closed in the header: the answer follows it
                else:
                    j = j + 2  # an open think block: the reasoning follows
            e = j
            while e < n and tok[e] != IM_END:
                e += 1
            m[j:min(e + 1, n)] = True
            i = e
        i += 1
    return m


def anchors_of(gen, L, k, rng=None):
    """k anchor positions (8 <= a < L - 8) inside generated spans."""
    ok = np.where(gen[:L])[0]
    ok = ok[(ok >= 8) & (ok < L - BLOCK)]
    if len(ok) == 0:
        return None
    return torch.from_numpy(ok[np.random.randint(len(ok), size=k)])


# ---------------------------------------------------------------------------------------------------- model
def rms(x, w):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float())


def rope(x, pos):
    # x [..., rows, heads, 128]; pos [..., rows]
    half = HD // 2
    inv = THETA ** (-torch.arange(half, dtype=torch.float32, device=x.device) / half)
    ang = pos[..., None].float() * inv
    c, s = ang.cos()[..., None, :], ang.sin()[..., None, :]
    a, b = x[..., :half].float(), x[..., half:].float()
    return torch.cat([a * c - b * s, b * c + a * s], -1)


class Drafter:
    def __init__(self, path, rank):
        self.W = {}
        with safe_open(path, "pt") as f:
            for k in f.keys():
                self.W[k] = f.get_tensor(k).to(dev, wdt)
        self.lora = {}
        self.scale = 2.0
        names = ["fc.weight", "candidate_selector.hidden_projection.weight"]
        for i in range(LAYERS):
            p = f"layers.{i}"
            names += [f"{p}.self_attn.{x}_proj.weight" for x in "qkvo"] + [f"{p}.mlp.{x}_proj.weight" for x in ("gate", "up", "down")]
            names += [f"{p}.attention_conv.kernel_projection.weight", f"{p}.mlp_conv.kernel_projection.weight"]
        for n in names:
            out_f, in_f = self.W[n].shape
            A = torch.empty(rank, in_f, device=dev)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            B = torch.zeros(out_f, rank, device=dev)
            self.lora[n] = (A.requires_grad_(), B.requires_grad_())

    def params(self):
        return [t for ab in self.lora.values() for t in ab]

    def lin(self, x, name):
        y = F.linear(x, self.W[name])
        if name in self.lora:
            A, B = self.lora[name]
            y = y + F.linear(F.linear(x, A.to(x.dtype)), B.to(x.dtype)) * self.scale
        return y

    def conv(self, x, delta, base, side):
        # x [..., 8, H]; delta [..., 8, 2, 2, G]; base [2, 2, H]
        G = H // GROUP
        d = delta[..., side, :, :].unsqueeze(-1)  # [..., 8, 2, G, 1]
        b = base.float().reshape(2, 2, G, GROUP)[side]  # [2, G, GROUP]
        coef = b + d  # [..., 8, 2, G, GROUP]
        xb = x.float().reshape(*x.shape[:-1], G, GROUP)
        prev = torch.cat([torch.zeros_like(xb[..., :1, :, :]), xb[..., :-1, :, :]], -3)
        out = coef[..., 0, :, :] * xb + coef[..., 1, :, :] * prev
        return out.reshape(*x.shape)

    def ctx_kv(self, feats):
        # feats [L, NFEAT] -> per layer (K [L, 8, 128] roped, V [L, 8, 128])
        L = feats.shape[0]
        ch = rms(self.lin(feats, "fc.weight"), self.W["hidden_norm.weight"])
        pos = torch.arange(L, device=feats.device)
        out = []
        for i in range(LAYERS):
            p = f"layers.{i}.self_attn"
            k = self.lin(ch.to(wdt), f"{p}.k_proj.weight").reshape(L, NKV, HD)
            k = rope(rms(k, self.W[f"{p}.k_norm.weight"]), pos)
            v = self.lin(ch.to(wdt), f"{p}.v_proj.weight").reshape(L, NKV, HD)
            out.append((k.to(wdt), v.to(wdt)))
        return out

    def block(self, emb, anchors, kv):
        # emb [N, 8, H]; anchors [N] (positions); kv = ctx_kv output for the sequence (L rows)
        N = emb.shape[0]
        L = kv[0][0].shape[0]
        bpos = anchors[:, None] + torch.arange(BLOCK, device=emb.device)[None]
        kpos = torch.arange(L, device=emb.device)
        mask = torch.cat([kpos[None, :] < anchors[:, None], torch.ones(N, BLOCK, dtype=torch.bool, device=emb.device)], 1)
        mask = mask[:, None, None, :]  # [N, 1, 1, L + 8]
        res = None
        x = emb
        G = H // GROUP
        for i in range(LAYERS):
            pre, p = f"layers.{i}", f"layers.{i}.self_attn"
            if res is None:
                res = x.float()
            else:
                res = x.float() + res
            xn = rms(res, self.W[f"{pre}.input_layernorm.weight"])
            kp = self.lin(xn.to(wdt), f"{pre}.attention_conv.kernel_projection.weight").float().reshape(N, BLOCK, 2, 2, G)
            xc = self.conv(xn, kp, self.W[f"{pre}.attention_conv.base_kernel"], 0).to(wdt)
            q = self.lin(xc, f"{p}.q_proj.weight").reshape(N, BLOCK, NH, HD)
            k = self.lin(xc, f"{p}.k_proj.weight").reshape(N, BLOCK, NKV, HD)
            v = self.lin(xc, f"{p}.v_proj.weight").reshape(N, BLOCK, NKV, HD)
            q = rope(rms(q, self.W[f"{p}.q_norm.weight"]), bpos).to(wdt)
            k = rope(rms(k, self.W[f"{p}.k_norm.weight"]), bpos).to(wdt)
            K = torch.cat([kv[i][0][None].expand(N, -1, -1, -1), k], 1)  # [N, L+8, 8, 128]
            V = torch.cat([kv[i][1][None].expand(N, -1, -1, -1), v], 1)
            K = K.repeat_interleave(NH // NKV, 2).permute(0, 2, 1, 3)
            V = V.repeat_interleave(NH // NKV, 2).permute(0, 2, 1, 3)
            ao = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), K, V, attn_mask=mask)
            ao = ao.permute(0, 2, 1, 3).reshape(N, BLOCK, NH * HD)
            o = self.lin(ao.to(wdt), f"{p}.o_proj.weight")
            o = self.conv(o, kp, self.W[f"{pre}.attention_conv.base_kernel"], 1)
            res = o + res
            xn2 = rms(res, self.W[f"{pre}.post_attention_layernorm.weight"])
            mp = self.lin(xn2.to(wdt), f"{pre}.mlp_conv.kernel_projection.weight").float().reshape(N, BLOCK, 2, 2, G)
            xm = self.conv(xn2, mp, self.W[f"{pre}.mlp_conv.base_kernel"], 0).to(wdt)
            g = self.lin(xm, f"{pre}.mlp.gate_proj.weight")
            u = self.lin(xm, f"{pre}.mlp.up_proj.weight")
            mo = self.lin(F.silu(g) * u, f"{pre}.mlp.down_proj.weight")
            x = self.conv(mo, mp, self.W[f"{pre}.mlp_conv.base_kernel"], 1)
        res = x.float() + res
        return rms(res, self.W["norm.weight"])  # [N, 8, H] float


def load_table(path, rows=248320):
    out = torch.empty(rows, H, dtype=wdt, device=dev)
    mm = np.memmap(path, dtype="<f4", mode="r", shape=(rows, H))
    for s in range(0, rows, 16384):
        out[s:s + 16384] = torch.from_numpy(np.asarray(mm[s:s + 16384])).to(dev, wdt)
    return out


def sample_block(model, emb_table, tok, feats, anchors):
    # anchors: LongTensor [N]
    N = anchors.shape[0]
    mask_id = 248077
    ids = torch.full((N, BLOCK), mask_id, dtype=torch.long, device=dev)
    ids[:, 0] = tok[anchors]
    return emb_table[ids]


def selector(model, hid, cand, unary, prev):
    # hid [N, 7, H]; cand [N, 7, 16]; unary [N, 7, 16] float; prev [N, 7] -> scores [N, 7, 16]
    hp = model.lin(hid.to(wdt), "candidate_selector.hidden_projection.weight").float()
    P = model.W["candidate_selector.predecessor_codebook"][prev].float()
    S = model.W["candidate_selector.successor_codebook"][cand].float()
    return unary + torch.einsum("ner,necr->nec", P * hp, S)


def forward_loss(model, emb_table, head, tok, feats, anchors):
    kv = model.ctx_kv(feats)
    emb = sample_block(model, emb_table, tok, feats, anchors)
    hid = model.block(emb, anchors, kv)[:, 1:]  # [N, 7, H]
    N = anchors.shape[0]
    tgt = tok[anchors[:, None] + torch.arange(1, BLOCK, device=dev)[None]]  # [N, 7]
    prev = tok[anchors[:, None] + torch.arange(0, BLOCK - 1, device=dev)[None]]
    logits = F.linear(hid.to(wdt), head).float()  # [N, 7, V]
    ce = F.cross_entropy(logits.reshape(N * 7, -1), tgt.reshape(-1))
    vals, cand = logits.topk(TOPK, -1)
    scores = selector(model, hid, cand, vals, prev)
    hit = cand == tgt[..., None]
    has = hit.any(-1)
    idx = hit.float().argmax(-1)
    sel = F.cross_entropy(scores[has], idx[has]) if has.any() else ce * 0
    return ce, sel, hid, logits


@torch.no_grad()
def evaluate(model, emb_table, head, data, n_per_seq=24, seqs=None):
    """Mean accepted prefix length of the free-running greedy walk, and teacher-forced slot accuracy."""
    model_acc, slot_hit, slot_n, total = 0.0, np.zeros(7), np.zeros(7), 0
    g = torch.Generator().manual_seed(1)
    for tok_np, feat_np, gen_np in data[:seqs] if seqs else data:
        L = min(len(tok_np), args.maxlen)
        tok = torch.from_numpy(tok_np[:L]).to(dev)
        feats = torch.from_numpy(feat_np[:L].astype(np.float32)).to(dev).to(wdt)
        if L < 40:
            continue
        anchors = anchors_of(gen_np, L, n_per_seq)
        if anchors is None:
            continue
        anchors = anchors.to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            kv = model.ctx_kv(feats)
            emb = sample_block(model, emb_table, tok, feats, anchors)
            hid = model.block(emb, anchors, kv)[:, 1:]
            logits = F.linear(hid.to(wdt), head).float()
            vals, cand = logits.topk(TOPK, -1)
        N = anchors.shape[0]
        tgt = tok[anchors[:, None] + torch.arange(1, BLOCK, device=dev)[None]]
        hp = model.lin(hid.to(wdt), "candidate_selector.hidden_projection.weight").float()
        P, S = model.W["candidate_selector.predecessor_codebook"], model.W["candidate_selector.successor_codebook"]
        prev = tok[anchors]
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        acc = torch.zeros(N, device=dev)
        for e in range(7):
            sc = vals[:, e] + torch.einsum("nr,ncr->nc", P[prev].float() * hp[:, e], S[cand[:, e]].float())
            pick = cand[:, e].gather(1, sc.argmax(-1, keepdim=True))[:, 0]
            ok = pick == tgt[:, e]
            alive = alive & ok
            acc += alive.float()
            prev = pick
            slot_hit[e] += ok.sum().item()
            slot_n[e] += N
        model_acc += acc.sum().item()
        total += N
    return model_acc / total, slot_hit / slot_n


def merge(model, path_in, path_out):
    out = {}
    with safe_open(path_in, "pt") as f:
        for k in f.keys():
            w = f.get_tensor(k)
            if k in model.lora:
                A, B = model.lora[k]
                w = (w.float() + (B @ A).detach().cpu().float() * model.scale).to(torch.bfloat16)
            out[k] = w.contiguous()
    save_file(out, path_out, metadata={"format": "pt"})


def main():
    t0 = time.time()
    data = load_sequences(args.data)
    if args.max_seqs:
        data = data[:args.max_seqs]
    tot = sum(len(t) for t, _, _ in data)
    print(f"{len(data)} conversations, {tot} tokens ({time.time() - t0:.0f} s)", flush=True)
    nval = max(4, len(data) // 10)
    train, val = data[:-nval], data[-nval:]
    if args.eval_all:
        val = data
    tdir = args.tables or args.data
    emb_table = load_table(f"{tdir}/token_embd.f32")
    head = load_table(f"{tdir}/output.f32")
    model = Drafter(f"{args.model}/model.safetensors", args.rank)
    for p in model.params():
        p.requires_grad_(not args.eval_only)
    base = evaluate(model, emb_table, head, val)
    print(f"baseline: accepted prefix {base[0]:.3f} of 7; slot hits {np.round(base[1], 3)}", flush=True)
    if args.eval_only:
        return
    opt = torch.optim.AdamW(model.params(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / args.steps)) * 0.95 + 0.05)
    best = (base[0], None)
    for step in range(args.steps):
        for g_ in opt.param_groups:
            g_["lr"] = args.lr * sched(step)
        lc = ls = 0.0
        for _ in range(args.accum):
            tok_np, feat_np, gen_np = train[np.random.randint(len(train))]
            L = min(len(tok_np), args.maxlen)
            tok = torch.from_numpy(tok_np[:L]).to(dev)
            feats = torch.from_numpy(feat_np[:L].astype(np.float32)).to(dev).to(wdt)
            anchors = anchors_of(gen_np, L, args.anchors)
            if anchors is None:
                continue
            anchors = anchors.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                ce, sel, _, _ = forward_loss(model, emb_table, head, tok, feats, anchors)
            (0.5 * ce + sel).div(args.accum).backward()
            lc += ce.item() / args.accum
            ls += sel.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.params(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"step {step}: ce {lc:.3f} selector {ls:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            r = evaluate(model, emb_table, head, val)
            print(f"  eval at {step + 1}: accepted prefix {r[0]:.3f} (baseline {base[0]:.3f}); slot hits {np.round(r[1], 3)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    merge(model, f"{args.model}/model.safetensors", f"{args.out}/model.safetensors")
    shutil.copy(f"{args.model}/config.json", f"{args.out}/config.json")
    print("merged weights written to", args.out, flush=True)


main()
