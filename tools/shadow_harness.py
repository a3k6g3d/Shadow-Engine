#!/usr/bin/env python3
"""tools/pi_harness.py - a small command-line agent for Strata: six tools, one short prompt, no approval popups.

    python tools/pi_harness.py "find where the MCP result cap is set and tell me what it is"
    python tools/pi_harness.py                      # a chat that keeps its history
    python tools/pi_harness.py --skills tools/skills "add a probe for the sprite sheet"

Shaped like Pi (the small MIT harness this borrows the idea from) and not like the web app: the tools run in this
process, so there is no MCP server to start and no round-trip per call; the prompt is short and does not change from
one turn to the next, so the conversation cache keeps the whole prefix warm; several tool calls in one answer run at
the same time; and a long answer is cut at the end, never in the middle, because a search whose middle was dropped
costs a second search.

What it keeps from Strata, because they are the point of them: `mcp.blocked_paths` (a path on the list is never read,
listed or searched) and `mcp.protected_paths` (nothing is written under one).  There is no gate in front of anything
else - a command line cannot show a button - so bash runs, and an edit happens.

Skills are Hermes's idea, kept small: a folder of markdown procedures whose names and first lines the prompt lists,
and which the model reads when a task matches one.  Memory is the same idea: every run is appended to a JSONL under
`--memory`, and grep searches it like any other folder.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:                                                    # the server's own list, so the rules cannot drift apart
    from serve.mcp import BlockList                    # type: ignore
except Exception:                                       # a shallow install without serve/ on the path
    BlockList = None

MAX_RESULT = 60_000                                     # what one tool result may say (the server's MCP cap is 20,000)
MAX_FILE = 5_000_000                                    # the same file size cap the dev server uses
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode", "build"}
DEADLINE = 25.0
TIMEOUT = 120

SYSTEM = """You work in a command line on this machine and you have six tools: read_file, grep, glob, edit_file, \
write_file, bash. Search before you edit, and read a file before you change it.

Run several searches in one answer when they do not depend on each other; they run at the same time.

bash runs through cmd.exe on Windows: `dir` and not `ls`, `findstr` and not `grep`. Never put Python in a shell
command line - the shell eats the quotes - write the probe as a file and run the file.

Say what you ran and what it returned, in a few lines. Never state a number you did not read.
"""


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def api(cfg: dict, url: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


class Guard:
    """The two rules that survive without a popup: never read a blocked path, never write under a protected one."""

    def __init__(self, cfg: dict):
        self.block = BlockList(cfg.get("mcp", {}).get("blocked_paths", [])) if BlockList else None
        self.protected = [os.path.normcase(os.path.realpath(os.path.expandvars(os.path.expanduser(p))))
                          for p in cfg.get("mcp", {}).get("protected_paths", [])]

    def blocked(self, path: str) -> bool:
        if not self.block:
            return False
        try:                                            # BlockList.hit wants a resolved, case-folded path
            return self.block.hit(os.path.normcase(os.path.realpath(os.path.expandvars(os.path.expanduser(path)))))
        except (OSError, ValueError):
            return self.block.hit(None)                 # could not be checked counts as blocked

    def protected_hit(self, path: str) -> bool:
        real = os.path.normcase(os.path.realpath(os.path.expandvars(os.path.expanduser(path))))
        return any(real == p or real.startswith(p + os.sep) for p in self.protected)

    def skip(self, path: str) -> bool:
        return os.path.basename(path) in SKIP_DIRS or self.blocked(path)


def walk(top: str, g: Guard):
    try:
        with os.scandir(top) as it:
            entries = list(it)
    except OSError:
        return
    dirs, files = [], []
    for e in entries:
        try:
            if e.is_symlink():
                continue
            if e.is_dir(follow_symlinks=False):
                if not g.skip(e.path):
                    dirs.append(e.path)
            elif e.is_file(follow_symlinks=False) and not g.skip(e.path):
                files.append(e.path)
        except OSError:
            continue
    yield files
    for d in dirs:
        yield from walk(d, g)


def read_lines_of(path: str):
    try:
        if os.path.getsize(path) > MAX_FILE:
            return None
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", "replace").splitlines()


def tail_clip(s: str, n: int = MAX_RESULT) -> str:
    """Cut at the end, never in the middle: a dropped middle costs a second search."""
    return s if len(s) <= n else s[:n] + f"\n... cut at {n:,} characters; ask for a narrower path or a bigger --cap"


# ---- the six tools ----------------------------------------------------------------------------------------------

def t_read_file(g: Guard, path: str, start: int = 1, end: int = 400) -> str:
    if g.blocked(path):
        return "error: on the no-read list"
    if not os.path.isfile(path):
        return f"error: not a file: {path}"
    lines = read_lines_of(path)
    if lines is None:
        return "error: binary or over 5 MB"
    start, end = max(1, int(start)), min(len(lines), int(end or 400))
    out = "\n".join(f"{i:>5}|{lines[i - 1]}" for i in range(start, end + 1))
    if end < len(lines):
        out += f"\n(lines {start}-{end} of {len(lines)}; ask again with start={end + 1} for the rest)"
    return out


def t_glob(g: Guard, pattern: str, path: str = ".", max_results: int = 200) -> str:
    hits, roots = [], list(os.scandir(path)) if os.path.isdir(path) else []
    star = pattern.startswith("**/")
    name = pattern.split("/")[-1]
    for files in walk(path, g):
        for f in files:
            if (star and os.path.basename(f).lower().endswith(name.lstrip("*").lower())) or \
               (not star and fnmatch_match(os.path.relpath(f, path), pattern)):
                hits.append(f)
                if len(hits) >= max_results:
                    break
        if len(hits) >= max_results:
            break
    return "\n".join(hits[:max_results]) or "no matches"


def fnmatch_match(rel: str, pattern: str) -> bool:
    # fnmatch's `*` spans `/`, so `serve/**/*.py` matched nothing: its `**` needs at least one separator, and those
    # files sit right in serve/. Match one path segment at a time instead, where `**` may stand for no folders.
    pat, got = pattern.rstrip("/").lower().split("/"), rel.replace("\\", "/").lower().split("/")

    def m(pi: int, ri: int) -> bool:
        if pi == len(pat):
            return ri == len(got)
        if pat[pi] == "**":
            return any(m(pi + 1, k) for k in range(ri, len(got) + 1))
        return ri < len(got) and fnmatch.fnmatch(got[ri], pat[pi]) and m(pi + 1, ri + 1)

    return m(0, 0)


def grep_one(g: Guard, path: str, rx, ci: bool) -> list[str]:
    lines = read_lines_of(path)
    if lines is None:
        return []
    out = []
    for i, line in enumerate(lines, 1):
        if rx.search(line) if not ci else rx.search(line.lower()):
            out.append(f"{path}:{i}: {line[:300]}")
            if len(out) >= 40:
                out.append(f"{path}: ... more matches in this file")
                break
    return out


def t_grep(g: Guard, pattern: str, path: str = ".", glob: str = "", ignore_case: bool = False,
           max_results: int = 200) -> str:
    rx = re.compile(pattern if not ignore_case else pattern, re.I if ignore_case else 0)
    ext = glob.strip()
    t0 = time.time()
    files, n = [], 0
    todo = []
    if os.path.isfile(path):                            # one file named directly, not a folder
        return tail_clip("\n".join(grep_one(g, path, rx, ignore_case)) or "no matches")
    for batch in walk(path, g):
        todo.extend(f for f in batch if not ext or f.lower().endswith(ext.lstrip("*").lower()))
        if time.time() - t0 > DEADLINE:
            break
    with cf.ThreadPoolExecutor(16) as ex:                # the walk is I/O bound; this is what makes it fast
        for hits in ex.map(lambda f: grep_one(g, f, rx, ignore_case), todo):
            n += len(hits)
            files.extend(hits)
            if n >= max_results:
                break
    out = "\n".join(files[:max_results])
    if time.time() - t0 > DEADLINE:
        out += f"\n(stopped after {DEADLINE:.0f} s; searched a part of the tree)"
    return tail_clip(out or "no matches")


def t_edit_file(g: Guard, path: str, oldText: str, newText: str) -> str:
    if g.blocked(path):
        return "error: on the no-read list"
    if g.protected_hit(path):
        return "error: on the protected list, and this harness has no popup to ask with"
    lines = read_lines_of(path)
    if lines is None:
        return "error: binary, too large, or not there"
    text = "\n".join(lines)
    n = text.count(oldText)
    if n != 1:
        return f"error: that text appears {n} times, not once; read the file and copy it exactly"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace(oldText, newText))
    return f"edited {path}"


def t_write_file(g: Guard, path: str, content: str) -> str:
    if g.blocked(path):
        return "error: on the no-read list"
    if g.protected_hit(path):
        return "error: on the protected list, and this harness has no popup to ask with"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return f"wrote {len(content)} characters to {path}"


def t_bash(g: Guard, command: str, cwd: str = ".") -> str:
    try:
        p = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return f"error: the command ran over {TIMEOUT} s"
    return tail_clip(((p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")).strip() or "(no output)")


TOOLS = [
    {"name": "read_file", "description": "a file, or its first/last N lines",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "start": {"type": "integer"}, "end": {"type": "integer"}}}},
    {"name": "grep", "description": "search inside files under a folder (this is how you find code)",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                        "glob": {"type": "string"}, "ignore_case": {"type": "boolean"}}}},
    {"name": "glob", "description": "find files by name pattern, e.g. **/*.py",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}}},
    {"name": "edit_file", "description": "change text inside a file; oldText must match exactly, once",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "oldText": {"type": "string"},
                                                        "newText": {"type": "string"}}}},
    {"name": "write_file", "description": "create or replace a file",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "bash", "description": "run a command line",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "cwd": {"type": "string"}}}},
]
RUN = {"read_file": t_read_file, "grep": t_grep, "glob": t_glob, "edit_file": t_edit_file,
       "write_file": t_write_file, "bash": t_bash}


def skills_note(folder: str) -> str:
    """Names and first lines only - the body is read on purpose, so it costs nothing until it is used."""
    if not folder or not os.path.isdir(folder):
        return ""
    out = []
    for f in sorted(os.scandir(folder), key=lambda e: e.name):
        if not f.name.endswith(".md"):
            continue
        try:
            with open(f.path, encoding="utf-8", errors="replace") as fh:
                first = next((ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")), "")
            out.append(f"- {f.name}: {first[:120]}")
        except OSError:
            continue
    return ("\nSkills in " + folder + " - read the file before starting a task that matches one:\n" + "\n".join(out)) \
        if out else ""


def main() -> int:
    global MAX_RESULT
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", help="what to do; leave it out to chat")
    ap.add_argument("--config", default=r"J:\Strata-work\strata.json")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--skills", default="", help="a folder of .md procedures to list in the prompt")
    ap.add_argument("--memory", default="", help="a folder; every run is appended here as JSONL, and grep finds it")
    ap.add_argument("--cap", type=int, default=MAX_RESULT)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    MAX_RESULT = a.cap

    cfg = load_config(a.config)
    key = cfg.get("api_key") or os.environ.get("STRATA_API_KEY") or ""
    g = Guard(cfg)
    system = SYSTEM + (skills_note(a.skills) if a.skills else "")
    messages = [{"role": "system", "content": system}]
    task = " ".join(a.task).strip()
    if task:
        messages.append({"role": "user", "content": task})

    log = open(os.path.join(a.memory, "sessions.jsonl"), "a", encoding="utf-8") if a.memory and \
        os.path.isdir(a.memory) else None
    ex = cf.ThreadPoolExecutor(8)                       # several tool calls in one answer, at once
    while True:
        if not task:
            try:
                line = input("\n> ")
            except EOFError:
                break
            if not line.strip():
                break
            messages.append({"role": "user", "content": line})
        body = {"model": cfg.get("model_name", ""), "messages": messages, "tools": TOOLS,
                "max_tokens": a.max_tokens, "temperature": 0.2}
        t0 = time.time()
        r = api(cfg, a.url, body, key)
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        usage = r.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if not a.quiet:
            print(f"[{dt:5.1f} s  prompt {usage.get('prompt_tokens', 0):,}  cached {cached:,}  "
                  f"out {usage.get('completion_tokens', 0):,}]", flush=True)
        messages.append(msg)
        if not calls:
            if msg.get("content") and not a.quiet:
                print(msg["content"])
            if log:
                log.write(json.dumps({"task": task or messages[1]["content"], "answer": msg.get("content"),
                                      "turns": len(messages)}) + "\n")
            break
        results = [ex.submit(RUN[c["function"]["name"]], g, **json.loads(c["function"].get("arguments") or "{}"))
                   for c in calls]
        for c, fut in zip(calls, results):
            got = fut.result()
            if not a.quiet:
                print(f"  {c['function']['name']} {json.dumps(json.loads(c['function'].get('arguments') or '{}'))[:120]}"
                      f" -> {len(got)} chars", flush=True)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": got})
        task = task or ""
        if len([m for m in messages if m["role"] == "assistant"]) >= a.rounds:
            print(f"(stopped after {a.rounds} rounds)")
            break
    ex.shutdown(wait=False)
    if log:
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
