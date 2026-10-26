#!/usr/bin/env python3
"""tools/shadow_eval.py - the frozen yardstick: does the memory actually answer the question it is asked.

    python tools/shadow_eval.py rank                 # retrieval only, no model, runs anywhere
    python tools/shadow_eval.py prompts              # write eval/prompts/<id>.txt for the answer arm
    python tools/shadow_eval.py answers <dir>        # grade the answers an external session wrote
    python tools/shadow_eval.py full                # rank + prompts, in one go

Two arms, deliberately split.  The retrieval arm touches no model at all - it is the same embedder subprocess the
memory tool uses, so it is deterministic and it can be run from inside the session that is being improved.  The
answer arm cannot: it needs a live model session, and the session doing the asking must not be the session being
judged, or the loop is grading its own homework.  So `prompts` writes one file per case, an outside session answers
them, and `answers` grades what comes back.

Grading is substring matching, not an LLM judge, on purpose: a judge inside the loop is slow, costs tokens, and gets
gameable the moment the loop learns what the judge likes.  `must_contain` lists the facts the answer has to state;
if they are there, the answer is right, and if they are not, it is not.

`eval/cases.jsonl` is frozen.  The loop may edit skills, prompts and index parameters; it may not edit this file,
because a yardstick the measured thing is allowed to move is not a yardstick.  Cases were written from questions
this project actually asked, including the one the memory got wrong on the first try (see README).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "eval", "cases.jsonl")
PROMPTS = os.path.join(ROOT, "eval", "prompts")
TOP = 5                                                 # what the memory tool returns by default

# the memory tool is a script, not a package, so load it by path and borrow its embed/cos rather than duplicating
_spec = importlib.util.spec_from_file_location("shadow_memory", os.path.join(ROOT, "tools", "shadow_memory.py"))
mem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mem)                            # type: ignore[union-attr]


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rank_case(case: dict, store: list[dict], qvec: list[float]) -> tuple[int, float]:
    """Where the gold record landed, 1-based, and what it scored."""
    hits = sorted(((mem.cos(qvec, r["vec"]), r) for r in store), key=lambda x: -x[0])
    for i, (score, r) in enumerate(hits, 1):
        if case["gold_sub"].lower() in r.get("text", "").lower():
            return i, score
    return 99, 0.0                                      # gold not in the top-k at all


def arm_rank(cases: list[dict], store: list[dict]) -> int:
    if not store:
        sys.exit("the store is empty - nothing to rank")
    qvecs = mem.embed([c["ask"] for c in cases])         # one embedder call for the whole set
    print(f"{'case':<6}{'rank':>5}{'score':>8}  question")
    rr = 0.0
    top1 = 0
    for case, qv in zip(cases, qvecs):
        rank, score = rank_case(case, store, qv)
        if rank != 99:
            rr += 1.0 / rank
        if rank == 1:
            top1 += 1
        flag = "" if rank == 1 else ("  <-- not first" if rank <= TOP else "  <-- outside top-%d" % TOP)
        print(f"{case['id']:<6}{rank:>5}{score:>8.4f}  {case['ask'][:58]}{flag}")
    n = len(cases)
    print(f"\n{n} cases, top-1 {top1}/{n}, MRR {rr / n:.3f}   (store: {len(store)} records)")
    print("this arm measures retrieval only; it says nothing about whether the answer was written correctly")
    return 0


def arm_prompts(cases: list[dict], store: list[dict]) -> int:
    os.makedirs(PROMPTS, exist_ok=True)
    qvecs = mem.embed([c["ask"] for c in cases])
    for case, qv in zip(cases, qvecs):
        hits = sorted(((mem.cos(qv, r["vec"]), r) for r in store), key=lambda x: -x[0])[:TOP]
        with open(os.path.join(PROMPTS, case["id"] + ".txt"), "w", encoding="utf-8") as f:
            f.write("Answer from the notes below alone. If they do not answer it, say so.\n\n"
                    "Question: " + case["ask"] + "\n\nNotes:\n")
            for score, r in hits:
                f.write(f"- ({score:.2f}) {r['text']}\n")
            f.write("\nAnswer in one or two sentences.\n")
    print(f"wrote {len(cases)} prompts to {PROMPTS}")
    print("each must be answered by a FRESH session - the one writing the prompt cannot grade its own answer")
    return 0


def arm_answers(cases: list[dict], answers_dir: str) -> int:
    d = os.path.join(ROOT, "eval", answers_dir) if not os.path.isabs(answers_dir) else answers_dir
    if not os.path.isdir(d):
        sys.exit(f"no such answers folder: {d}")
    ok = 0
    missing = 0
    for case in cases:
        p = os.path.join(d, case["id"] + ".txt")
        if not os.path.isfile(p):
            print(f"{case['id']:<6}  --  no answer file")
            missing += 1
            continue
        got = open(p, encoding="utf-8-sig").read().lower()
        need = [w for w in case["must_contain"] if w.lower() not in got]
        if need:
            print(f"{case['id']:<6}  no   missing {need}  <- {os.path.basename(p)}")
        else:
            print(f"{case['id']:<6}  yes  {case['ask'][:50]}")
            ok += 1
    n = len(cases) - missing
    print(f"\n{n} graded, {ok}/{n} correct, {missing} missing")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    cases = read_jsonl(CASES)
    store = mem.load_store()
    if cmd == "rank":
        return arm_rank(cases, store)
    if cmd == "prompts":
        return arm_prompts(cases, store)
    if cmd == "full":
        arm_rank(cases, store)
        print()
        return arm_prompts(cases, store)
    if cmd == "answers":
        if len(sys.argv) < 3:
            sys.exit("usage: shadow_eval.py answers <dir>")
        return arm_answers(cases, sys.argv[2])
    sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    sys.exit(main())
