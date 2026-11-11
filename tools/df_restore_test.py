import json, sys, time, uuid, urllib.request
port = int(sys.argv[1]); out = sys.argv[2]
P = json.load(open(r'J:\Shadow\bench\dfdata\varied36.json', encoding='utf-8'))
def chat(hist):
    body = json.dumps({"messages": hist, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(rq, timeout=600))
    return r["choices"][0]["message"]["content"], r.get("timings", {})
nonce = uuid.uuid4().hex[:8]
hist = []; rows = []
FORCE = 24   # at this turn the previous assistant reply is altered: the session's tokens no longer start the prompt -> reset + restore a snapshot
for k in range(34):
    text_p = P[k % len(P)] if k else f"(session {nonce}) " + P[0]
    if k == FORCE and hist and hist[-1]["role"] == "assistant":
        hist[-1]["content"] = hist[-1]["content"][:-3]
    hist.append({"role": "user", "content": text_p})
    text, tm = chat(hist); hist.append({"role": "assistant", "content": text})
    n, ms = tm.get("predicted_n", 0), tm.get("predicted_ms", 0)
    rows.append({"req": k, "pred_n": n, "pred_ms": round(ms), "tps": round(1000 * n / ms, 1) if ms else 0, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n")})
    print(rows[-1], flush=True)
json.dump(rows, open(out, "w"))
