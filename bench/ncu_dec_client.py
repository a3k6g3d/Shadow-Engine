import json, urllib.request
prompts = json.load(open('J:/Shadow/bench/dfdata/varied36.json'))[:4]
for p in prompts:
    req = {'messages': [{'role': 'user', 'content': p}], 'max_tokens': 64, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8095/v1/chat/completions', json.dumps(req).encode(), {'Content-Type': 'application/json'}), timeout=1800))
    print(r['timings']['predicted_n'], 'tokens')
