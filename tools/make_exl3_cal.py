"""Pack chat conversations (JSONL lines {"messages": [{"role", "content"}, ...]}, e.g. tools/df_dump_to_chats.py output or the teacher answers of STATUS 164) into an exllamav3
calibration file for `convert.py -cd`: a safetensors with `input_ids` (rows x cols, right-padded) and `lengths`.  Conversations are written in the Qwen chat format
(<|im_start|>role\\n...<|im_end|>\\n; an assistant turn without a <think> block gets the empty one the non-thinking template adds) and packed greedily into rows of at most COLS tokens, so
each row is several whole conversations and none is cut mid-turn unless it alone is longer than a row.

    I:\\tools\\exl3-venv\\Scripts\\python.exe -I tools\\make_exl3_cal.py TOKENIZER.json IN.jsonl [IN2.jsonl ...] OUT.safetensors [--rows 250] [--cols 2048] [--seed 1]
"""
import json
import random
import sys

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

args = sys.argv[1:]


def opt(name, default):
    if name in args:
        i = args.index(name)
        v = int(args[i + 1])
        del args[i:i + 2]
        return v
    return default


rows, cols, seed = opt("--rows", 250), opt("--cols", 2048), opt("--seed", 1)
tokp, *ins, out = args
tok = Tokenizer.from_file(tokp)

convs = []
for p in ins:
    with open(p, encoding="utf-8") as f:
        for line in f:
            convs.append(json.loads(line)["messages"])
random.Random(seed).shuffle(convs)


def render(msgs):
    s = ""
    for m in msgs:
        body = m["content"]
        if m["role"] == "assistant" and not body.lstrip().startswith("<think>"):
            body = "<think>\n\n</think>\n\n" + body
        s += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
    return tok.encode(s, add_special_tokens=False).ids


packed, cur = [], []
for msgs in convs:
    ids = render(msgs)
    if len(ids) > cols:
        ids = ids[:cols]
    if len(cur) + len(ids) > cols:
        packed.append(cur)
        cur = []
        if len(packed) >= rows:
            break
    cur = cur + ids
if cur and len(packed) < rows:
    packed.append(cur)

lengths = torch.tensor([len(r) for r in packed], dtype=torch.int64)
t = torch.zeros(len(packed), cols, dtype=torch.int64)
for i, r in enumerate(packed):
    t[i, :len(r)] = torch.tensor(r, dtype=torch.int64)
save_file({"input_ids": t, "lengths": lengths}, out)
print(f"{len(packed)} rows x {cols}: {int(lengths.sum())} tokens (mean row {lengths.float().mean():.0f}), max id {int(t.max())}, from {len(convs)} conversations")
