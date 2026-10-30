"""Same prompts, same history (the exact run's own replies); how far does each config's reply follow the exact reply?
   python quality_test.py save REF.json N      # run against an --exact server and save its replies
   python quality_test.py cmp REF.json N       # run against another server, history = the reference replies"""
import json, sys, time, urllib.request
mode, ref_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = json.load(open('J:/Shadow/bench/dfdata/varied36.json'))[:n]
ref = json.load(open(ref_path)) if mode == 'cmp' else []
hist, replies, match, toks = [], [], [], 0
t0 = time.time()
for i, p in enumerate(prompts):
    hist.append({'role': 'user', 'content': p})
    body = json.dumps({'messages': hist, 'max_tokens': 96, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8095/v1/chat/completions', body, {'Content-Type': 'application/json'}), timeout=900))
    txt = r['choices'][0]['message'].get('content') or ''
    toks += r.get('timings', {}).get('predicted_n', 0)
    replies.append(txt)
    if mode == 'cmp':
        a = ref[i]; k = 0
        while k < min(len(a), len(txt)) and a[k] == txt[k]: k += 1
        match.append(k / max(len(a), 1))
        hist.append({'role': 'assistant', 'content': a})
    else:
        hist.append({'role': 'assistant', 'content': txt})
if mode == 'save':
    json.dump(replies, open(ref_path, 'w')); print('saved', len(replies))
else:
    exact_turns = sum(1 for m in match if m >= 0.999)
    print('mean prefix match %.3f, %d of %d replies identical, %d tokens, %.1f tok/s end-to-end' % (sum(match) / len(match), exact_turns, len(match), toks, toks / (time.time() - t0)))
