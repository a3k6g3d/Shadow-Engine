import json, sys, time, urllib.request
port = int(sys.argv[1])
P = json.load(open(r'J:\Shadow\bench\dfdata\varied36.json', encoding='utf-8'))
for k in range(6):
    body = json.dumps({"messages": [{"role": "user", "content": P[k]}], "max_tokens": 160, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900))
    t = r.get("timings", {})
    print(k, t.get("prompt_n"), t.get("predicted_n"), t.get("predicted_per_second"), round(time.time() - t0, 2), flush=True)
