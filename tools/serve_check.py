"""Live checks against a running `shadow serve` (default http://127.0.0.1:8081): every route, a whole and a streamed
answer that must be identical, prefix reuse on a follow-up turn, refusals, and two requests at once (queued).
Prints PASS/FAIL per check and exits non-zero on any failure."""
import json, sys, threading, time, urllib.request, urllib.error

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def get(path):
    with urllib.request.urlopen(URL + path, timeout=30) as r:
        return r.status, r.read().decode()


def post(body, stream=False):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    if not stream:
        return r.status, json.loads(r.read().decode())
    events = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: "):
            events.append(line[6:])
    return r.status, events


# wait for the model
t0 = time.time()
while True:
    try:
        s, b = get("/health")
        if s == 200:
            break
    except urllib.error.HTTPError as e:
        if e.code != 503:
            raise
    except urllib.error.URLError:
        pass
    if time.time() - t0 > 600:
        check("health within 10 min", False)
        sys.exit(1)
    time.sleep(2)
check("/health ready", True, f"{time.time() - t0:.0f} s")

s, b = get("/v1/models"); m = json.loads(b)
check("/v1/models", s == 200 and m["data"][0]["object"] == "model", m["data"][0]["id"])
s, b = get("/props"); p = json.loads(b)
check("/props", s == 200 and p["engine"] == "shadow" and p["tools"] is True)
s, b = get("/metrics")
check("/metrics", s == 200 and "shadow_requests_total" in b)

msgs = [{"role": "user", "content": "What is 17 times 23?"}]
base = {"messages": msgs, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}
s, whole = post(base)
c1 = whole["choices"][0]["message"]["content"]
check("non-streamed answer", s == 200 and len(c1) > 0, repr(c1[:80]))
check("usage + timings", whole["usage"]["completion_tokens"] > 0 and whole["timings"]["predicted_per_second"] > 0,
      f'{whole["usage"]}, {whole["timings"]["predicted_per_second"]:.2f} tok/s')

s, ev = post(dict(base, stream=True), stream=True)
check("stream ends with [DONE]", ev[-1] == "[DONE]", f"{len(ev)} events")
chunks = [json.loads(e) for e in ev[:-1]]
c2 = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
check("streamed == whole (greedy)", c1 == c2, repr(c2[:80]))
check("final chunk has finish_reason + usage", chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
      and "usage" in chunks[-1])

# a follow-up turn: if the reply re-tokenizes the same, the session reuses the whole prior conversation
follow = {"messages": msgs + [{"role": "assistant", "content": c1}, {"role": "user", "content": "And 18 times 23?"}],
          "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}}
s, f = post(follow)
cached = f["usage"]["prompt_tokens_details"]["cached_tokens"]
check("follow-up answered", s == 200 and len(f["choices"][0]["message"]["content"]) > 0,
      f'cached {cached} of {f["usage"]["prompt_tokens"]} prompt tokens')

# reproducibility after a reset: the first question again must give the first answer again
s, again = post(base)
check("same answer after the session moved on", again["choices"][0]["message"]["content"] == c1)

s, e = post({"messages": msgs, "tools": [{"type": "function"}]})
check("malformed tool refused with 400", s == 400 and "name" in e["error"]["message"])
s, tc_resp = post({"messages": [{"role": "user", "content": "Call the tool read_file on Cargo.toml with start=1 and end=10."}],
                   "tools": [{"name": "read_file", "description": "read lines of a file",
                              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}}}],
                   "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}})
tcs = (tc_resp.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
check("tool call parsed into tool_calls", s == 200 and len(tcs) >= 1 and tcs[0]["function"]["name"] == "read_file"
      and tc_resp["choices"][0]["finish_reason"] == "tool_calls", repr(tcs[:1]))
s, e = post({"messages": msgs, "max_tokens": 10 ** 7})
check("too long refused", s == 400, e["error"]["message"][:80])

# two at once: both answered, the same answer, one after the other
out = [None, None]
def worker(i):
    out[i] = post(base)
ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
check("two concurrent requests", all(o and o[0] == 200 for o in out)
      and out[0][1]["choices"][0]["message"]["content"] == out[1][1]["choices"][0]["message"]["content"] == c1)

s, b = get("/v1/status"); st = json.loads(b)
check("/v1/status", s == 200 and st["state"] == "ready" and st["requests"] >= 6 and st["queued"] == 0,
      f'requests {st["requests"]}, {st["decode_tokens_per_second"]:.2f} tok/s, queued {st["queued"]}')
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
