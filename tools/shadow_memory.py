#!/usr/bin/env python3
"""tools/shadow_memory.py - semantic search over the memory folder the harness already writes, no server needed.

    python tools/shadow_memory.py add "the ABI symbol prefix stays strata_ on purpose"
    python tools/shadow_memory.py search "why is the C symbol prefix not shadow_"
    python tools/shadow_memory.py index memory/sessions.jsonl      # (re)embed what the harness logged
    python tools/shadow_memory.py stats

Memory here is a plain file first and an index second: `memory/memories.jsonl` is one JSON record per line, so grep
still finds it like any other folder, and this tool only adds the ability to ask in words instead of in letters.  It
reads the harness's own `sessions.jsonl` too, because that is where a long chat's earlier turns actually live - the
point of the whole thing is that a new session can remember the last one.

The vectors are not computed in Python: the embedder is `bin/llama-embedding.exe` (llama.cpp's own example, built
from the vendored source) running `models/bge-small-en-v1.5-q8_0.gguf`, called as a subprocess.  The harness stays
stdlib-only and no server is involved; one call embeds a whole batch, separated by `<#sep#>`, and prints OpenAI-style
JSON on stdout.  The model is small on purpose - 384 dimensions, ~0.5 s a batch on CPU - because the cost is paid on
every search, and a search that takes ten seconds does not get used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "bin", "llama-embedding.exe")
MODEL = os.path.join(ROOT, "models", "bge-small-en-v1.5-q8_0.gguf")
MEMORY = os.path.join(ROOT, "memory")
STORE = os.path.join(MEMORY, "memories.jsonl")

SEP = "<#sep#>"                                         # what the embedder splits one batch on
BATCH = 64                                              # texts per subprocess call; the model load is the slow part
DIM = 384                                               # bge-small's width, checked against the first reply
TOP = 5                                               # default hits
TIMEOUT = 300


def embed(texts: list[str]) -> list[list[float]]:
    """Ask the embedder for one vector per text, in order, in as few subprocess calls as fit."""
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        # the separator is the only thing the protocol reads literally, so a text containing it goes alone
        solo = [t for t in chunk if SEP in t]
        group = [t for t in chunk if SEP not in t]
        by_text = {}
        for batch in ([group] if group else []) + [[t] for t in solo]:
            p = subprocess.run([EXE, "-m", MODEL, "-p", SEP.join(batch), "--embd-separator", SEP,
                                "--embd-output-format", "json"],
                               capture_output=True, text=True, timeout=TIMEOUT)
            if p.returncode != 0 or not p.stdout.strip():
                sys.exit(f"embedder failed (rc {p.returncode}): {p.stderr.strip().splitlines()[-1:]}")
            data = json.loads(p.stdout)["data"]
            if len(data) != len(batch):
                sys.exit(f"embedder returned {len(data)} vectors for {len(batch)} texts")
            for item in sorted(data, key=lambda d: d["index"]):
                by_text[batch[item["index"]]] = item["embedding"]
        out += [by_text[t] for t in chunk]
    for v in out:
        if len(v) != DIM:
            sys.exit(f"a vector came back {len(v)} wide, not {DIM} - wrong model?")
    return out


def cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def read_records(path: str) -> list[dict]:
    """One JSON per line, the house format; a line that does not parse is skipped, not fatal."""
    recs = []
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")
    with open(path, encoding="utf-8-sig") as f:             # -sig: PowerShell writes a BOM, and it would eat line 1
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  (skipped {os.path.basename(path)}:{n} - not JSON)")
    return recs


def text_of(rec: dict) -> str:
    """The part of a harness log line worth embedding: what was asked and what came back, not the tool spam."""
    for k in ("text", "answer", "content", "task", "prompt", "note"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def load_store() -> list[dict]:
    return read_records(STORE)


def save(rec: dict) -> None:
    """Append, never rewrite: the file is the store, and grep should see the new line at once."""
    os.makedirs(MEMORY, exist_ok=True)
    with open(STORE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_add(a) -> int:
    text = " ".join(a.text).strip() if isinstance(a.text, list) else (a.text or "").strip()
    if not text:
        sys.exit("nothing to add")
    vec = embed([text])[0]
    save({"id": f"m{int(time.time() * 1000)}", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "source": "manual", "text": text, "vec": [round(x, 6) for x in vec]})
    print(f"added 1 ({len(text)} chars)")
    return 0


def cmd_index(a) -> int:
    """Embed a harness log (or any JSONL of {text}) into the store, so a past session becomes searchable."""
    recs = [(os.path.basename(p), r) for p in a.files for r in read_records(p)]
    recs = [(s, r) for s, r in recs if text_of(r)]
    if not recs:
        sys.exit("no embeddable records - a JSONL of {text: ...} lines, as the harness writes")
    texts = [text_of(r) for _, r in recs]
    vecs = embed(texts)
    for (src, r), vec in zip(recs, vecs):
        save({"id": f"m{src}@{r.get('id') or r.get('ts') or ''}", "ts": r.get("ts", ""),
              "source": src, "text": text_of(r), "vec": [round(x, 6) for x in vec]})
    print(f"indexed {len(recs)} records from {sorted({s for s, _ in recs})}")
    return 0


def cmd_search(a) -> int:
    store = load_store()
    if not store:
        sys.exit(f"nothing indexed yet - run `index` first ({STORE} is empty)")
    q = embed([" ".join(a.query) if isinstance(a.query, list) else a.query])[0]
    hits = sorted(((cos(q, r["vec"]), r) for r in store), key=lambda x: -x[0])[:a.top]
    for score, r in hits:
        print(f"{score:.4f}  {r['source']:<24} {r['ts']}\n        {r['text'][:200]}")
    return 0


def cmd_stats(a) -> int:
    store = load_store()
    by: dict[str, int] = {}
    for r in store:
        by[r.get("source", "?")] = by.get(r.get("source", "?"), 0) + 1
    print(f"{STORE}\n{len(store)} records, {DIM} dims, embedder: {os.path.basename(EXE)} "
          f"+ {os.path.basename(MODEL)}")
    for k in sorted(by):
        print(f"  {by[k]:>5}  {k}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="remember one thing")
    p.add_argument("text", nargs="+")
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("index", help="embed a harness JSONL into the store")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_index)
    p = sub.add_parser("search", help="ask in words")
    p.add_argument("query", nargs="+")
    p.add_argument("--top", type=int, default=TOP)
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("stats", help="what is in the store")
    p.set_defaults(fn=cmd_stats)
    a = ap.parse_args()
    for path in (EXE, MODEL):
        if not os.path.isfile(path):
            sys.exit(f"missing {path}")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
