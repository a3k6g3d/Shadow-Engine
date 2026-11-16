import json, sys, urllib.request
port = sys.argv[1]
text = open('J:/Shadow/docs/STATUS.md', encoding='utf-8').read()
for i in range(3):
    doc = text[20000 + 7000 * i: 20000 + 7000 * i + 28000]
    body = {'messages': [{'role': 'user', 'content': doc + '\n\nQuestion: in one sentence, what is this text about?'}], 'max_tokens': 24, 'temperature': 0,
            'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=1800))
    t = r['timings']
    print(f"prompt {t['prompt_n']} tok at {t['prompt_per_second']:.1f} tok/s; decode {t['predicted_per_second']:.1f}", flush=True)
