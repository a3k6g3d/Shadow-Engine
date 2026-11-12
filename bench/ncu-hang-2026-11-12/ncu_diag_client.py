import json, sys, time, urllib.request
port = int(sys.argv[1])
t0 = time.time()
body = json.dumps({"messages": [{"role": "user", "content": "Write one sentence about the sea."}], "max_tokens": 12, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=170))
print("reply", repr(r["choices"][0]["message"]["content"][:60]), "in", round(time.time() - t0, 1), "s", flush=True)
