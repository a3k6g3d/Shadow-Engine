"""Reference fixture generator for RecursiveSI/flint (arch = "flint"): runs the published Flint reference
equations (sglang/flint_sglang/model.py) in PyTorch float32 on the GGUF's dequantized+scaled weights (via `shadow dump`)
for the first N layers, writing every layer's intermediate streams as a fixture (`eval/fixtures/reference_flint_gguf`).

Covers:
  - Layers 0, 1: no-RoPE full attention with learned sinks + dense SwiGLU MLP (intermediate_size = 6144)
  - Layers 2, 3: NeoX RoPE sliding-window attention with learned sinks + 256-expert sigmoid top-6 MoE + ungated shared expert

    .venv-ref/Scripts/python.exe tools/make_reference_fixture_flint.py [--layers 4]
"""
import argparse
import json
import pathlib
import subprocess
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is Paris. The capital of Germany is"
FLINT_DIR = pathlib.Path("I:/shadow-ref/flint")
GGUF = "I:/Flint-40B-A2.5B-abliterated-NVFP4.gguf"


class GgufDump:
    def __init__(self, shadow_exe: str, model: str):
        self.exe, self.model = shadow_exe, model
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="shadow-dump-flint-"))
        self.calls = 0

    def dump(self, name: str, rows=None, expert=None) -> torch.Tensor:
        out = self.tmp / "t.f32"
        cmd = [self.exe, "dump", self.model, name, str(out)]
        if rows is not None:
            cmd += ["--rows", ",".join(str(r) for r in rows)]
        if expert is not None:
            cmd += ["--expert", str(expert)]
        r, c = [int(x) for x in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()]
        self.calls += 1
        a = np.fromfile(out, dtype="<f4").reshape(r, c)
        return torch.from_numpy(a.copy())


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.double().pow(2).mean(dim=-1, keepdim=True)
    return (x.double() * torch.rsqrt(var + eps)).float() * w


def apply_neox_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    # x: [T, H, D]
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    freqs = torch.outer(positions.double(), inv_freq)  # [T, half]
    cs = torch.cos(freqs)[:, None, :]
    sn = torch.sin(freqs)[:, None, :]
    a = x[..., :half].double()
    b = x[..., half:].double()
    y0 = (a * cs - b * sn).float()
    y1 = (b * cs + a * sn).float()
    return torch.cat([y0, y1], dim=-1)


def attention_with_sinks(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: torch.Tensor,
                         sliding_window: int | None) -> torch.Tensor:
    # q: [T, nh, hd], k: [T, nkv, hd], v: [T, nkv, hd], sinks: [nh]
    t_len, nh, hd = q.shape
    nkv = k.shape[1]
    group = nh // nkv
    scale = hd ** -0.5
    out = torch.zeros_like(q)
    for t in range(t_len):
        start = max(0, t + 1 - sliding_window) if sliding_window is not None else 0
        for h in range(nh):
            g = h // group
            qh = q[t, h].double()
            kt = k[start:t + 1, g].double()
            vt = v[start:t + 1, g].double()
            scores = (kt @ qh) * scale  # [ntok]
            sink = sinks[h].double()
            mx = torch.maximum(scores.max(), sink)
            ew = torch.exp(scores - mx)
            z = ew.sum() + torch.exp(sink - mx)
            out[t, h] = ((ew[:, None] * vt).sum(dim=0) / z).float()
    return out.reshape(t_len, nh * hd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "eval" / "fixtures" / "reference_flint_gguf"))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--shadow", default=str(ROOT / "target" / "release" / "shadow.exe"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    t0 = time.time()
    torch.set_grad_enabled(False)

    cfg = json.loads((FLINT_DIR / "config.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(FLINT_DIR / "tokenizer.json"))
    ids = tok.encode(a.prompt, add_special_tokens=False).ids
    print(f"prompt {a.prompt!r}: {len(ids)} tokens {ids}")

    src = GgufDump(a.shadow, GGUF)
    eps = cfg["rms_norm_eps"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    theta = cfg["rope_theta"]
    top_k = cfg["num_experts_per_tok"]
    sw = cfg["sliding_window"]
    positions = torch.arange(len(ids), dtype=torch.long)

    cap = {}
    uniq = sorted(set(ids))
    emb_rows = src.dump("token_embd.weight", rows=uniq)
    emb_map = dict(zip(uniq, emb_rows))
    h = torch.stack([emb_map[i] for i in ids])  # [T, 2048]
    cap["embed"] = h.clone()

    for l in range(a.layers):
        p = f"blk.{l}"
        u = rms_norm(h, src.dump(f"{p}.attn_norm.weight").flatten(), eps)
        cap[f"layer{l}.attn_hc_u"] = u.clone()

        wq = src.dump(f"{p}.attn_q.weight")
        wk = src.dump(f"{p}.attn_k.weight")
        wv = src.dump(f"{p}.attn_v.weight")
        wo = src.dump(f"{p}.attn_output.weight")
        qn = src.dump(f"{p}.attn_q_norm.weight").flatten()
        kn = src.dump(f"{p}.attn_k_norm.weight").flatten()
        sinks = src.dump(f"{p}.attn_sinks.weight").flatten()

        q = rms_norm((u @ wq.T).reshape(len(ids), nh, hd), qn, eps)
        k = rms_norm((u @ wk.T).reshape(len(ids), nkv, hd), kn, eps)
        v = (u @ wv.T).reshape(len(ids), nkv, hd)
        if l not in cfg["no_rope_layer_indices"]:
            q = apply_neox_rope(q, positions, theta)
            k = apply_neox_rope(k, positions, theta)
        win = sw if cfg["layer_types"][l] == "sliding_attention" else None
        ao = attention_with_sinks(q, k, v, sinks, win)
        y_attn = ao @ wo.T
        cap[f"layer{l}.mixer"] = y_attn.clone()
        h = h + y_attn

        u_ffn = rms_norm(h, src.dump(f"{p}.ffn_norm.weight").flatten(), eps)
        cap[f"layer{l}.mlp_hc_u"] = u_ffn.clone()

        if l in cfg["mlp_only_layers"]:
            wg = src.dump(f"{p}.ffn_gate.weight")
            wu = src.dump(f"{p}.ffn_up.weight")
            wd = src.dump(f"{p}.ffn_down.weight")
            y_ffn = (F.silu(u_ffn @ wg.T) * (u_ffn @ wu.T)) @ wd.T
        else:
            wr = src.dump(f"{p}.ffn_gate_inp.weight")
            logits = u_ffn @ wr.T  # [T, 256]
            scores = torch.sigmoid(logits.double())
            vals, idx = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)
            weights = (vals / (vals.sum(dim=-1, keepdim=True) + 1e-20)).float()
            cap[f"layer{l}.route_idx"] = idx.float().clone()
            cap[f"layer{l}.route_w"] = weights.clone()

            need = sorted(set(idx.flatten().tolist()))
            exps = {}
            for e in need:
                exps[e] = (
                    src.dump(f"{p}.ffn_gate_exps.weight", expert=e),
                    src.dump(f"{p}.ffn_up_exps.weight", expert=e),
                    src.dump(f"{p}.ffn_down_exps.weight", expert=e),
                )
            y_ffn = torch.zeros_like(u_ffn)
            for t in range(len(ids)):
                ut = u_ffn[t]
                for k_i in range(top_k):
                    e = int(idx[t, k_i].item())
                    w = weights[t, k_i]
                    eg, eu, ed = exps[e]
                    eo = (F.silu(eg @ ut) * (eu @ ut)) @ ed.T
                    y_ffn[t] = y_ffn[t] + w * eo

            wgs = src.dump(f"{p}.ffn_gate_shexp.weight")
            wus = src.dump(f"{p}.ffn_up_shexp.weight")
            wds = src.dump(f"{p}.ffn_down_shexp.weight")
            y_sh = (F.silu(u_ffn @ wgs.T) * (u_ffn @ wus.T)) @ wds.T
            y_ffn = y_ffn + y_sh

        cap[f"layer{l}.moe"] = y_ffn.clone()
        h = h + y_ffn
        cap[f"layer{l}.out"] = h.clone()

    final_u = rms_norm(h, src.dump("output_norm.weight").flatten(), eps)
    cap["final_u"] = final_u.clone()

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prompt": a.prompt,
        "ids": ids,
        "positions": None,
        "layers": a.layers,
        "repo": "RecursiveSI/flint",
        "weights": "gguf",
        "torch": torch.__version__,
        "dtype": "float32",
        "note": "Flint reference equations (sglang/flint_sglang/model.py) run on Shadow's GGUF via `shadow dump`",
        "tensors": [],
    }
    blob = bytearray()
    for name, t in cap.items():
        arr = np.ascontiguousarray(t.numpy().astype("<f4"))
        manifest["tensors"].append({"name": name, "shape": list(arr.shape), "offset": len(blob)})
        blob += arr.tobytes()
    (out / "tensors.bin").write_bytes(bytes(blob))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {len(cap)} tensors, {len(blob) / 1e6:.2f} MB to {out}; {src.calls} dumps in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
