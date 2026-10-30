import json, sys, time, urllib.request
t0 = time.time()
prompts = json.load(open('J:/Shadow/bench/dfdata/varied36.json'))[:int(sys.argv[1])]
hist = []; rows = []
for i, p in enumerate(prompts):
    hist.append({'role': 'user', 'content': p})
    body = json.dumps({'messages': hist, 'max_tokens': 96, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8095/v1/chat/completions', body, {'Content-Type': 'application/json'}), timeout=900))
    c = r['choices'][0]; txt = c['message'].get('content') or ''
    n = r.get('timings', {}).get('predicted_n', r.get('usage', {}).get('completion_tokens'))
    rows.append((i, n, c.get('finish_reason')))
    hist.append({'role': 'assistant', 'content': txt})
short = [x for x in rows if x[2] == 'stop' and x[1] is not None and x[1] < 40]
print('turns', len(rows), 'stop<40 tokens:', len(short), 'tokens:', sum(x[1] or 0 for x in rows))
print('wall %.0f s, %.1f tok/s end-to-end' % (time.time() - t0, sum(x[1] or 0 for x in rows) / (time.time() - t0)))
print([(x[0], x[1], x[2][:1]) for x in rows])
