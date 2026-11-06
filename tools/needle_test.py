"""Long-prompt retrieval check for attention changes: does a fast mode still find a sentence buried in a ~9k-token prompt?

  py tools/needle_test.py [port] [n_items] [chars] [multi]

Each item is the start of docs/STATUS.md (`chars` characters, default 28000 = ~9.5k tokens) with one sentence "Note: the secret code is WORD-NNNN." inserted at a fixed fraction of
the text, then the question.  With `multi`, four sentences "the access code of project NAME is WORD-NNNN" are spread through the text and the question names one project (the others
are distractors with the same shape).  Every prompt starts with a random id (no shared prefix, so nothing comes from the prompt cache).  Prints found/total and the misses.
The sparse attention and the DeltaNet recurrence only matter for long prompts, which the short suites (quality_test, task_check) do not reach."""
import json, random, sys, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
n_items = int(sys.argv[2]) if len(sys.argv) > 2 else 8
chars = int(sys.argv[3]) if len(sys.argv) > 3 else 28000
multi = len(sys.argv) > 4 and sys.argv[4] == 'multi'
NL = chr(10)
text = open('J:/Shadow/docs/STATUS.md', encoding='utf-8').read()[:chars]
rng = random.Random(20261106)
words = ['ZEBRA', 'MAPLE', 'COBALT', 'FALCON', 'ORCHID', 'TUNDRA', 'QUARTZ', 'LANTERN', 'MARBLE', 'SPARROW', 'VELVET', 'HARBOR']
found, failed = 0, []
for i in range(n_items):
    frac = 0.08 + 0.84 * i / max(1, n_items - 1)
    if multi:
        names = rng.sample(words, 4)
        codes = [f'{rng.choice(words)}-{rng.randint(1000, 9999)}' for _ in names]
        fracs = [0.15 + 0.7 * k / 3 + 0.04 * ((i + k) % 3) for k in range(4)]
        doc = text
        for k in sorted(range(4), key=lambda k: -fracs[k]):   # from the back, so the earlier positions stay valid
            pk = text.rfind(NL, 0, int(len(text) * fracs[k]))
            doc = doc[:pk] + NL + f'Note: the access code of project {names[k]} is {codes[k]}.' + NL + doc[pk:]
        ask = i % 4
        code = codes[ask]
        question = NL + NL + f'Question: what is the access code of project {names[ask]}? Reply with only the code.'
        frac = fracs[ask]
    else:
        code = f'{rng.choice(words)}-{rng.randint(1000, 9999)}'
        pos = text.rfind(NL, 0, int(len(text) * frac))
        doc = text[:pos] + NL + f'Note: the secret code is {code}.' + NL + text[pos:]
        question = NL + NL + 'Question: what is the secret code mentioned in the text above? Reply with only the code.'
    prompt = f'[{rng.randint(10**8, 10**9)}]' + NL + doc + question
    body = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 16, 'temperature': 0, 'stream': False, 'cache_prompt': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=1500))
    reply = r['choices'][0]['message']['content']
    ok = code in reply
    found += ok
    if not ok:
        failed.append((round(frac, 2), code, reply[:40]))
    print(f'  item {i} at {frac:.2f}: {"ok" if ok else "MISS"} ({code} / {reply[:30]!r}) {r["timings"]["prompt_n"]} tokens', flush=True)
print(f'found {found}/{n_items}', failed)
