"""Export the main model's token embedding and (dequantized) output head of an EXL3 checkpoint as float32 files, for tools/dflash2_train.py (which needs them where
`shadowzero dump MODEL token_embd.weight|output.weight FILE` gives them for a GGUF; `shadowzero dump` does not read EXL3 directories).  Run in the exllamav3 venv (CUDA torch):

    I:\\tools\\exl3-venv\\Scripts\\python.exe tools\\exl3_export_tables.py MODEL_DIR OUT_DIR [cuda_device]

Writes OUT_DIR/token_embd.f32 ([vocab, hidden] rows) and OUT_DIR/output.f32 ([vocab, hidden] rows: row v is the head's weight vector of token v).  The head is rebuilt by
exllamav3's own reconstruction (`LinearEXL3.get_weight_tensor`), i.e. what the checkpoint holds, rotations and scales applied."""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.environ.get("EXL3_DIR", r"I:\tools\exllamav3"))
from exllamav3.modules.quant.exl3 import LinearEXL3  # noqa: E402

model_dir, out_dir = sys.argv[1], sys.argv[2]
dev = f"cuda:{sys.argv[3]}" if len(sys.argv) > 3 else "cuda:0"
os.makedirs(out_dir, exist_ok=True)
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]


def get(key):
    with safe_open(os.path.join(model_dir, idx[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def write(path, t):
    # rows in slices, so the host never holds a second copy
    with open(path, "wb") as f:
        for i in range(0, t.shape[0], 8192):
            f.write(t[i:i + 8192].to(torch.float32).cpu().numpy().tobytes())


emb = get("model.language_model.embed_tokens.weight")
print("embedding", tuple(emb.shape), emb.dtype, flush=True)
write(os.path.join(out_dir, "token_embd.f32"), emb)

trellis = get("lm_head.trellis")
suh, svh, mul1 = get("lm_head.suh"), get("lm_head.svh"), get("lm_head.mul1")
in_f, out_f = trellis.shape[0] * 16, trellis.shape[1] * 16
print("head", in_f, "->", out_f, "trellis", tuple(trellis.shape), flush=True)
lin = LinearEXL3(None, in_f, out_f, suh=suh.to(dev), svh=svh.to(dev), trellis=trellis.to(dev), mul1=mul1.to(dev), key="lm_head")
w = lin.get_weight_tensor()  # [in, out]
print("weight", tuple(w.shape), w.dtype, "finite:", bool(torch.isfinite(w).all()), "rms %.5f" % w.float().pow(2).mean().sqrt().item(), flush=True)
write(os.path.join(out_dir, "output.f32"), w.t().contiguous())
print("done")
