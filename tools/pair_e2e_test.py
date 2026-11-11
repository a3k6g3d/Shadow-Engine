import json, sys, urllib.request
port = int(sys.argv[1]); out = sys.argv[2]
P = json.load(open(r'J:\Shadow\bench\dfdata\varied36.json', encoding='utf-8'))
PARA_A = ("The harbour town of Velmar sits at the mouth of a slow brown river, and every morning the fishing boats leave before the lamps in the market are put out. "
          "Its council keeps a ledger of every cargo, every toll and every dispute, and the clerks who write in it are famous for their small, upright handwriting. ")
PARA_B = ("In the mountain monastery of Karth, the monks measure the year by the thaw of a single glacier and record its retreat in knots tied on long cords. "
          "Travellers who climb the stair are given barley tea, a bed of felt, and one question to carry down the mountain again. ")
def chat(hist, n=40):
    body = json.dumps({"messages": hist, "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"]["content"], r.get("timings", {})
convs = {"A": [{"role": "user", "content": "E2E pair test A. " + PARA_A * 6 + "Summarize this in one short sentence."}],
         "B": [{"role": "user", "content": "E2E pair test B. " + PARA_B * 6 + "Summarize this in one short sentence."}]}
order = ["A", "A", "A", "A", "A", "B", "B", "A", "B", "A"]
nxt = {"A": 0, "B": 10}
res = []
for k, c in enumerate(order):
    h = convs[c]
    if k >= 1 and h[-1]["role"] == "assistant":
        h.append({"role": "user", "content": P[nxt[c] % len(P)]}); nxt[c] += 1
    text, tm = chat(h)
    h.append({"role": "assistant", "content": text})
    res.append({"req": k, "conv": c, "text": text, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "predicted_n": tm.get("predicted_n")})
    print(k, c, tm.get("prompt_n"), tm.get("cache_n"), tm.get("predicted_n"), repr(text[:60]), flush=True)
json.dump(res, open(out, "w"))
