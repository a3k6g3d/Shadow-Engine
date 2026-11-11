import json, sys, urllib.request
port = int(sys.argv[1]); out = sys.argv[2]
P = json.load(open(r'J:\Shadow\bench\dfdata\varied36.json', encoding='utf-8'))
res = []
hist = []
for k in range(3):
    hist.append({"role": "user", "content": P[k]})
    body = json.dumps({"messages": hist, "max_tokens": 96, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=600))
    t = r["choices"][0]["message"]["content"]; hist.append({"role": "assistant", "content": t})
    res.append({"turn": k, "text": t, "timings": r.get("timings")})
json.dump(res, open(out, "w"))
