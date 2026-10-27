"""Stage 1's gate: the published reference implementation (Hugging Face transformers, models/qwen4_exp) run in float32
for the first N layers, with every layer's intermediate streams written out as a fixture.  Two weight sources:

  --weights checkpoint   the BF16 checkpoint's own weights (the true reference; Shadow, which runs on a quantized file,
                         is expected to agree in direction, not bit for bit)                -> eval/fixtures/reference
  --weights gguf         the SAME GGUF Shadow runs, dequantized by `shadow dump` (whose dequantizers are bit-exact
                         against ggml's) and mapped back into the reference's layout.  Now only floating-point order
                         differs, so Shadow must match to ~1e-5: this separates "Shadow's math is wrong" from
                         "quantization noise"                                            -> eval/fixtures/reference_gguf

Nothing big is downloaded in checkpoint mode.  tools/hf_range.py reads the dense weights of layers 0..N-1 whole and, from
each layer's fused expert tensors, only the experts the router actually picks for this prompt; the embedding rows are
the prompt's; the n-gram table rows are the ones the hash lands on.  Everything fetched is cached under I:/shadow-ref.

    .venv-ref/Scripts/python.exe tools/make_reference_fixture.py [--layers 4] [--weights checkpoint|gguf]

Fixture: manifest.json (prompt, ids, versions, where each tensor is) and tensors.bin (float32, little-endian).
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _refenv  # noqa: F401  (must precede transformers)
import numpy as np
import torch
import torch.nn as nn
import transformers
from hf_range import Checkpoint
from transformers import AutoTokenizer
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is Paris. The capital of Germany is"
HF_DIR = "I:/shadow-ref/hf"
REPO = "Qwen/Qwen3.8-Flash-Next"
GGUF = ("J:/Strata-work/models/swift15/IQ3_S/"
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf")
SHARD_ROWS = 2500012          # ngram_embedding.shard_k.weight rows (128 shards of [2500012, 160])
N_K, N_VPK = 16, 3            # key heads, value heads per key head (48 value heads in all)


# ------------------------------------------------------------------------------------------------ the two sources
class CheckpointSource:
    """The BF16 checkpoint, by range reads."""

    def __init__(self, ck):
        self.ck = ck

    def dense(self, model, layers):
        sd = {}
        pre = "model.language_model."
        for name in self.ck.index:
            if not name.startswith(pre):
                continue
            short = name[len(pre):]
            if short.startswith("layers."):
                if int(short.split(".")[1]) >= layers or "mlp.experts." in short or "ngram_embedding" in short:
                    continue
            elif not short.startswith("hyper_connection_mixer."):
                continue
            t = self.ck.tensor(name)
            sd[short] = t.float() if t.is_floating_point() else t
        # open question 5 of ARCH.md: are the PLE key and query norms really the same tensor?
        for l in range(layers):
            k, q = f"layers.{l}.ple.norm_key.weight", f"layers.{l}.ple.norm_query.weight"
            if k in sd:
                print(f"PLE norm_key == norm_query in the checkpoint: {bool(torch.equal(sd[k], sd[q]))}")
        return sd

    def embed(self, ids):
        t, idx = self.ck.rows("model.language_model.embed_tokens.weight", sorted(set(ids)))
        table = dict(zip(idx, t))
        return torch.stack([table[i].float() for i in ids])

    def experts(self, layer, need):
        pre = f"model.language_model.layers.{layer}.mlp.experts."
        gu, _ = self.ck.rows(pre + "gate_up_proj", need)
        dn, _ = self.ck.rows(pre + "down_proj", need)
        return {e: (gu[i].float(), dn[i].float()) for i, e in enumerate(need)}

    def ple_rows(self, layer, rows):
        out = []
        for r in rows:
            shard, off = divmod(r, SHARD_ROWS)
            t, _ = self.ck.rows(f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight", [off])
            out.append(t[0].float())
        return torch.stack(out)


def untile(x, hd):
    """The GGUF stores value heads tiled (index v*16 + k); the reference groups them (k*3 + v).  Rows of `x` are heads."""
    n = N_K * N_VPK
    rest = x.shape[1:]
    return x.reshape(N_VPK, N_K, hd, *rest).permute(1, 0, 2, *range(3, 3 + len(rest))).reshape(n * hd, *rest)


class GgufSource:
    """The GGUF Shadow runs, dequantized by `shadow dump` and mapped back into the reference's layout."""

    def __init__(self, shadow_exe, model):
        self.exe, self.model = str(shadow_exe), model
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="shadow-dump-"))
        self.calls = 0

    def dump(self, name, rows=None, expert=None):
        if rows is not None and len(rows) > 1000:       # Windows limits a command line to ~32 KB
            return torch.cat([self.dump(name, rows=rows[i:i + 1000]) for i in range(0, len(rows), 1000)])
        out = self.tmp / "t.f32"
        cmd = [self.exe, "dump", self.model, name, str(out)]
        if rows is not None:
            cmd += ["--rows", ",".join(str(r) for r in rows)]
        if expert is not None:
            cmd += ["--expert", str(expert)]
        r, c = [int(x) for x in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()]
        self.calls += 1
        return torch.from_numpy(np.fromfile(out, dtype="<f4").copy()).reshape(r, c)

    def get(self, key):
        """The reference's state-dict entry `key` (without the model.language_model. prefix), from the GGUF."""
        d = self.dump
        if key.startswith("hyper_connection_mixer."):
            p, rest = "output_hc", key[len("hyper_connection_mixer."):]
        else:
            _, l, rest = key.split(".", 2)
            p = None
        hc = {"hc_norm.weight": "norm", "input_mix_weight_down.weight": "down", "input_mix_weight_up.weight": "up",
              "block_inject_weight.weight": "inject"}
        if p is not None:
            kind = hc[rest]
            t = d(f"{p}_{kind}.weight")
            return t.reshape(-1) - 1 if kind == "norm" else t
        b = f"blk.{l}."
        for tag, pre in (("attn_hyper_connection.", "hc_attn_"), ("mlp_hyper_connection.", "hc_ffn_")):
            if rest.startswith(tag):
                kind = hc[rest[len(tag):]]
                t = d(f"{b}{pre}{kind}.weight")
                return t.reshape(-1) - 1 if kind == "norm" else t
        m1 = lambda name: d(b + name).reshape(-1) - 1      # zero-centred norms: the GGUF stores the +1
        if rest == "linear_attn.in_proj_qkv.weight":
            t = d(b + "attn_qkv.weight")
            return torch.cat([t[:4096], untile(t[4096:], 128)])
        if rest == "linear_attn.in_proj_z.weight":
            return untile(d(b + "attn_gate.weight"), 128)
        if rest == "linear_attn.in_proj_b.weight":
            return untile(d(b + "ssm_beta.weight"), 1)
        if rest == "linear_attn.in_proj_a.weight":
            return untile(d(b + "ssm_alpha.weight"), 1)
        if rest == "linear_attn.A_log":
            return torch.log(-untile(d(b + "ssm_a").reshape(-1), 1))
        if rest == "linear_attn.dt_bias":
            return untile(d(b + "ssm_dt.bias").reshape(-1), 1)
        if rest == "linear_attn.conv1d.weight":
            t = d(b + "ssm_conv1d.weight")
            return torch.cat([t[:4096], untile(t[4096:], 128)]).reshape(-1, 1, t.shape[1])
        if rest == "linear_attn.norm.weight":
            return d(b + "ssm_norm.weight").reshape(-1)
        if rest == "linear_attn.out_proj.weight":
            return untile(d(b + "ssm_out.weight").T.contiguous(), 128).T.contiguous()
        sa = {"self_attn.q_proj.weight": "attn_q", "self_attn.k_proj.weight": "attn_k", "self_attn.v_proj.weight": "attn_v",
              "self_attn.o_proj.weight": "attn_output", "mlp.gate.weight": "ffn_gate_inp",
              "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp", "mlp.shared_expert.up_proj.weight": "ffn_up_shexp",
              "mlp.shared_expert.down_proj.weight": "ffn_down_shexp", "ple.key_proj.weight": "ple_key",
              "ple.value_proj.weight": "ple_value"}
        if rest in sa:
            return d(f"{b}{sa[rest]}.weight")
        norms = {"self_attn.q_norm.weight": "attn_q_norm", "self_attn.k_norm.weight": "attn_k_norm",
                 "self_attn.indexer.q_layernorm.weight": "indexer.q_norm", "self_attn.indexer.k_layernorm.weight": "indexer.k_norm",
                 "ple.norm_key.weight": "ple_norm_key", "ple.norm_query.weight": "ple_norm_query",
                 "ple.norm_conv.weight": "ple_norm_conv"}
        if rest in norms:
            return m1(f"{norms[rest]}.weight")
        if rest == "self_attn.indexer.index_qk_proj.weight":
            return torch.cat([d(b + "indexer.q_proj.weight"), d(b + "indexer.k_proj.weight")])
        if rest == "mlp.shared_expert_gate.weight":
            return d(b + "ffn_gate_inp_shexp.weight").reshape(1, -1)
        if rest == "ple.conv1d.weight":
            t = d(b + "ple_conv1d.weight")
            return t.reshape(-1, 1, t.shape[1])
        raise KeyError(key)

    def dense(self, model, layers):
        sd = {}
        for key in model.state_dict().keys():
            if "mlp.experts." in key or "ngram_embedding" in key or key.startswith("embed_tokens") \
                    or "rotary_emb" in key or "ngram_heads" in key or key.endswith("layer_multipliers"):
                continue
            sd[key] = self.get(key)
        return sd

    def embed(self, ids):
        return self.dump("token_embd.weight", rows=ids)

    def experts(self, layer, need):
        out = {}
        for e in need:
            g = self.dump(f"blk.{layer}.ffn_gate_exps.weight", expert=e)
            u = self.dump(f"blk.{layer}.ffn_up_exps.weight", expert=e)
            out[e] = (torch.cat([g, u]), self.dump(f"blk.{layer}.ffn_down_exps.weight", expert=e))
        return out

    def ple_rows(self, layer, rows):
        return self.dump("per_layer_token_embd.weight", rows=rows)


# ------------------------------------------------------------------------------------------------ lazy stand-ins
class LazyExperts(nn.Module):
    """Stands in for a layer's fused experts: fetches the picked experts, then runs the library's own
    Qwen4ExpTextExperts on just those (indices remapped), so the expert math is the reference's, unchanged."""

    def __init__(self, src, layer, cfg):
        super().__init__()
        self.src, self.layer, self.cfg = src, layer, cfg
        self.have = {}
        self.picked = None

    GROUP = 48

    def forward(self, hidden_states, top_k_index, top_k_weights):
        need = sorted(set(top_k_index.flatten().tolist()))
        self.picked = (top_k_index.clone(), top_k_weights.clone())
        out = torch.zeros_like(hidden_states)
        for g in range(0, len(need), self.GROUP):
            group = need[g:g + self.GROUP]
            have = self.src.experts(self.layer, group)
            local = {e: i for i, e in enumerate(group)}
            small = type(self.cfg)(**{**self.cfg.to_dict(), "num_experts": len(group),
                                      "num_experts_per_tok": min(self.cfg.num_experts_per_tok, len(group))})
            ex = M.Qwen4ExpTextExperts(small)
            ex.gate_up_proj = nn.Parameter(torch.stack([have[e][0] for e in group]), requires_grad=False)
            ex.down_proj = nn.Parameter(torch.stack([have[e][1] for e in group]), requires_grad=False)
            idx = torch.tensor([[local.get(e, len(group)) for e in row] for row in top_k_index.tolist()],
                               dtype=top_k_index.dtype)
            out += ex(hidden_states, idx, top_k_weights)
            del have, ex
        return out


class RowStore(nn.Module):
    """Stands in for the 320M-row n-gram table."""

    def __init__(self, src, layer):
        super().__init__()
        self.src, self.layer = src, layer
        self.weight = torch.empty(0)       # the library asks which device the table lives on

    def forward(self, ids):
        rows = self.src.ple_rows(self.layer, ids.flatten().tolist())
        return rows.reshape(*ids.shape, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--weights", choices=["checkpoint", "gguf"], default="checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--prompt-file", nargs="+", help="tokenize these files, concatenated, instead of --prompt")
    ap.add_argument("--tokens", type=int, help="keep only the first N token ids of the prompt")
    ap.add_argument("--positions", help="comma-separated positions: write only those rows of every stream (long prompts)")
    ap.add_argument("--shadow", default=str(ROOT / "target" / "release" / "shadow.exe"))
    a = ap.parse_args()
    out = pathlib.Path(a.out or ROOT / "eval" / "fixtures" / ("reference" if a.weights == "checkpoint" else "reference_gguf"))
    t0 = time.time()
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(HF_DIR)
    if a.prompt_file:
        a.prompt = "\n\n".join(pathlib.Path(f).read_text(encoding="utf-8") for f in a.prompt_file)
    ids = tok(a.prompt, add_special_tokens=False)["input_ids"]
    if a.tokens:
        assert len(ids) >= a.tokens, f"the prompt has only {len(ids)} tokens"
        ids = ids[: a.tokens]
        a.prompt = tok.decode(ids)
    keep = [int(x) for x in a.positions.split(",")] if a.positions else None
    assert keep is None or all(0 <= k < len(ids) for k in keep), "a kept position is outside the prompt"
    print(f"prompt {a.prompt[:80]!r}...: {len(ids)} tokens {ids[:16]}...")
    ck = Checkpoint(REPO, cache="I:/shadow-ref")
    src = CheckpointSource(ck) if a.weights == "checkpoint" else GgufSource(a.shadow, GGUF)

    cfg = Qwen4ExpConfig.from_pretrained(HF_DIR)
    tc = cfg.text_config
    tc.num_hidden_layers = a.layers
    tc.layer_types = tc.layer_types[: a.layers]
    tc._attn_implementation = "eager"
    with torch.device("meta"):
        model = M.Qwen4ExpTextModel(tc)

    sd = src.dense(model, a.layers)
    if a.weights == "checkpoint":
        pass
    else:   # the n-gram multipliers are buffers the GGUF stores under other keys; the library computes the same ones
        pass
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    expected_missing = [m for m in missing if "experts" in m or "ngram_embedding" in m or m.startswith("embed_tokens")
                        or "rotary_emb" in m or "ngram_heads" in m or m.endswith("layer_multipliers")]
    left = [m for m in missing if m not in expected_missing]
    print(f"loaded {len(sd)} dense tensors; unexpected {unexpected}; unexplained missing {left}")
    assert not left and not unexpected

    model.rotary_emb = M.Qwen4ExpTextRotaryEmbedding(tc)
    for l, layer in enumerate(model.layers):
        layer.mlp.experts = LazyExperts(src, l, tc)
        if layer.ple is not None:
            ng = layer.ple.ple_embedding
            ng.layer_multipliers = M._build_layer_multipliers(ng.unigram_vocab_size, ng.ngram_size, ng.ple_layer_index, ng.seed)
            ng.ngram_heads_vocab_sizes = torch.tensor(ng.head_vocab_sizes, dtype=torch.long)
            ng.ngram_heads_offsets = torch.tensor(ng.head_offsets, dtype=torch.long)
            ng.ngram_embedding = RowStore(src, l)
            print("PLE layer multipliers (library):", ng.layer_multipliers.tolist())
    meta_left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.device.type == "meta"]
    assert all("embed_tokens" in n for n in meta_left), meta_left

    # ---- what to write down
    cap = {}

    def grab(name):
        def hook(_m, _i, o):
            o = o[0] if isinstance(o, tuple) else o
            o = o.detach().float().reshape(o.shape[1], -1)
            cap[name] = (o[keep] if keep is not None else o).clone()
        return hook

    for l, layer in enumerate(model.layers):
        layer.register_forward_hook(grab(f"layer{l}.out"))
        layer.mlp.register_forward_hook(grab(f"layer{l}.moe"))
        (layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn).register_forward_hook(
            grab(f"layer{l}.mixer"))
        if layer.ple is not None:
            layer.ple.register_forward_hook(grab(f"layer{l}.ple"))
        layer.attn_hyper_connection.register_forward_hook(grab(f"layer{l}.attn_hc_u"))
        layer.mlp_hyper_connection.register_forward_hook(grab(f"layer{l}.mlp_hc_u"))
    model.hyper_connection_mixer.register_forward_hook(grab("final_u"))

    def grab_selected(name, width):
        def hook(_m, _i, mask):
            # the indexer's additive mask [1, 1, T, KV]: 0 where the key is attended to
            sel = (mask == 0) if mask.dtype != torch.bool else mask
            rows = sel[0, 0]
            rows = rows[keep] if keep is not None else rows
            out = torch.full((rows.shape[0], width), -1.0)
            for i in range(rows.shape[0]):
                idx = torch.nonzero(rows[i]).flatten().float()
                out[i, : idx.numel()] = idx
            cap[name] = out
        return hook

    for l, layer in enumerate(model.layers):
        if layer.layer_type != "linear_attention":
            ix = layer.self_attn.indexer
            layer.self_attn.indexer.register_forward_hook(grab_selected(f"layer{l}.selected", ix.token_budget + ix.compress_ratio - 1))

    inputs_embeds = src.embed(ids)[None].float()
    cap["embed"] = (inputs_embeds[0][keep] if keep is not None else inputs_embeds[0]).clone()
    res = model(inputs_embeds=inputs_embeds, ple_input_ids=torch.tensor([ids]), use_cache=False)
    assert torch.equal((res.last_hidden_state[0][keep] if keep is not None else res.last_hidden_state[0]), cap["final_u"])
    for l, layer in enumerate(model.layers):
        idx, w = layer.mlp.experts.picked
        cap[f"layer{l}.route_idx"] = (idx[keep] if keep is not None else idx).float()
        cap[f"layer{l}.route_w"] = (w[keep] if keep is not None else w).float()

    # ---- write
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"prompt": a.prompt, "ids": ids, "positions": keep, "layers": a.layers, "repo": REPO, "weights": a.weights,
                "transformers": transformers.__version__, "torch": torch.__version__, "dtype": "float32",
                "note": ("reference run on the BF16 checkpoint widened to float32" if a.weights == "checkpoint" else
                         "reference run on Shadow's own GGUF, dequantized by `shadow dump` and mapped back"),
                "tensors": []}
    blob = bytearray()
    for name, t in cap.items():
        arr = np.ascontiguousarray(t.numpy().astype("<f4"))
        manifest["tensors"].append({"name": name, "shape": list(arr.shape), "offset": len(blob)})
        blob += arr.tobytes()
    (out / "tensors.bin").write_bytes(bytes(blob))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {len(cap)} tensors, {len(blob) / 1e6:.1f} MB to {out}; "
          f"{ck.fetched / 1e9:.2f} GB fetched, {getattr(src, 'calls', 0)} dumps, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
