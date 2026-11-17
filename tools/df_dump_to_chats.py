"""Recover the conversations a `SHADOWZERO_DF_DUMP` run recorded (feats.bin: per record p0, m, the m token ids, then m rows of features) as chat text, skipping the feature bytes by seeking:
one JSON line per conversation, {"messages": [{"role", "content"}, ...], "think": bool, "tokens": N}, the roles and contents cut at the chat template's markers.  The assistant's
content is kept as the model wrote it (a thinking turn starts with its <think> block; a non-thinking turn's empty think block is dropped).  Used to turn the 500k-token mixD18 run into
distillation data.

    I:\\tools\\exl3-venv\\Scripts\\python.exe tools\\df_dump_to_chats.py DIR TOKENIZER.json OUT.jsonl [maxlen]
"""
import json
import re
import struct
import sys

from tokenizers import Tokenizer

NFEAT = 5 * 2560
d, tokp, out = sys.argv[1], sys.argv[2], sys.argv[3]
maxlen = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
tok = Tokenizer.from_file(tokp)
seqs, cur = [], None
with open(f"{d}/feats.bin", "rb", buffering=1 << 20) as fh:
    size = fh.seek(0, 2)
    fh.seek(0)
    while True:
        head = fh.read(8)
        if len(head) < 8:
            break
        p0, m = struct.unpack("<II", head)
        tb = fh.read(4 * m)
        if len(tb) < 4 * m or fh.tell() + 4 * m * NFEAT > size:
            break
        fh.seek(4 * m * NFEAT, 1)
        ids = struct.unpack(f"<{m}I", tb)
        if p0 == 0:
            if cur is not None:
                seqs.append(cur)
            cur = []
        if cur is None or p0 > len(cur):
            cur = None
            continue
        del cur[p0:]
        cur.extend(ids)
    if cur is not None:
        seqs.append(cur)
print(len(seqs), "conversations", sum(map(len, seqs)), "tokens", flush=True)

pat = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|$)", re.S)
n = think_n = skipped = 0
with open(out, "w", encoding="utf-8") as f:
    for ids in seqs:
        text = tok.decode(list(ids[:maxlen]), skip_special_tokens=False)
        msgs = []
        think = False
        for role, body in pat.findall(text):
            body = body.strip("\n")
            if role == "assistant":
                if body.startswith("<think>\n\n</think>"):
                    body = body[len("<think>\n\n</think>"):].lstrip("\n")
                elif body.startswith("<think>"):
                    think = True
            if body:
                msgs.append({"role": role, "content": body})
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[-2]["role"] == "user":
            f.write(json.dumps({"messages": msgs, "think": think, "tokens": len(ids)}, ensure_ascii=False) + "\n")
            n += 1
            think_n += think
        else:
            skipped += 1
print(f"wrote {n} conversations ({think_n} with thinking), skipped {skipped}")
