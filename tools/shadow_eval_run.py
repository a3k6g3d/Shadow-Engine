#!/usr/bin/env python3
"""tools/shadow_eval_run.py - run the answer arm: one fresh harness session per case, then grade what comes back.

    python tools/shadow_eval_run.py                 # all cases
    python tools/shadow_eval_run.py c04 c13         # just these

This is the script to hand to a session that is not the one being measured.  It is not a rule about Claude versus
me, it is a rule about the loop: the session that wrote the yardstick must not be the session answering off it,
because the answer arm is the half that can drift and the rank arm is the half that cannot.

Each case is answered by a NEW harness process - a fresh context, no memory of this conversation - so the answer
comes only from the notes `shadow_eval.py prompts` put in front of it.  One API call per case (--rounds 1, no
skills, no tools), which is the point: it is measuring whether the retrieved notes are enough, not whether the
model can fetch more.

Answers land in eval/answers/<id>.txt and grading is substring matching in shadow_eval.py, so no model judges this.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "tools", "shadow_harness.py")
PROMPTS = os.path.join(ROOT, "eval", "prompts")
ANSWERS = os.path.join(ROOT, "eval", "answers")
RUNLOG = os.path.join(ROOT, "eval", "answers")            # the harness appends sessions.jsonl here
CASES = os.path.join(ROOT, "eval", "cases.jsonl")
MAX_TOKENS = 220                                          # a one-or-two-sentence answer, not an essay


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def last_answer(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None


def main() -> int:
    wanted = sys.argv[1:]
    cases = [c for c in read_jsonl(CASES)] if not wanted else [c for c in read_jsonl(CASES) if c["id"] in wanted]
    if not cases:
        sys.exit("no matching cases")
    os.makedirs(ANSWERS, exist_ok=True)
    missing = [c["id"] for c in cases if not os.path.isfile(os.path.join(PROMPTS, c["id"] + ".txt"))]
    if missing:
        sys.exit(f"no prompt for {missing} - run `python tools/shadow_eval.py prompts` first")

    print(f"{len(cases)} cases, one fresh harness session each")
    for case in cases:
        prompt = open(os.path.join(PROMPTS, case["id"] + ".txt"), encoding="utf-8").read().strip()
        before = os.path.join(RUNLOG, "sessions.jsonl")
        seen = os.path.getsize(before) if os.path.isfile(before) else 0
        t0 = time.time()
        # the prompt is the task; --rounds 1 means one call and no tools, so the answer is the notes and nothing else
        p = subprocess.run([sys.executable, HARNESS, prompt, "--memory", RUNLOG, "--rounds", "1",
                            "--max-tokens", str(MAX_TOKENS), "--quiet"],
                           capture_output=True, text=True, cwd=ROOT, timeout=600)
        if p.returncode != 0:
            print(f"{case['id']:<6}  harness rc {p.returncode}: {(p.stderr or '').strip()[-120:]}")
            continue
        with open(before, encoding="utf-8-sig") as f:                       # only the line this run appended
            f.seek(seen)
            line = [l for l in f if l.strip()]
        rec = json.loads(line[-1]) if line else last_answer(before)
        answer = (rec or {}).get("answer") or ""
        if not answer.strip():
            print(f"{case['id']:<6}  empty answer ({time.time() - t0:.0f} s)")
            continue
        with open(os.path.join(ANSWERS, case["id"] + ".txt"), "w", encoding="utf-8") as f:
            f.write(answer.strip() + "\n")
        print(f"{case['id']:<6}  {time.time() - t0:>4.0f} s  {answer.strip()[:70]}")
    print(f"\nanswers in {ANSWERS}\ngrade with:  python tools/shadow_eval.py answers answers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
