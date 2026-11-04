"""A small auto-graded task suite for deciding whether an inexact decode mode (router bias, skipping) makes answers *worse*, not just different from `--exact`.

  py tools/task_check.py [port]      # against a running `shadow serve`; prints correct/total per kind and overall

48 short tasks from a fixed seed: arithmetic, word problems, string work, sorting, counting, facts.  The answer is graded by looking for the expected token in
the reply; the model is asked for only the final answer."""
import json, random, re, sys, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
rng = random.Random(20261104)
tasks = []  # (kind, prompt, expected regex)


def add(kind, prompt, expect):
    tasks.append((kind, prompt, expect))


for _ in range(12):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    add('arith', f'What is {a} * {b}?', rf'(?<![\d,.]){a * b:,}(?![\d])|(?<![\d,.]){a * b}(?![\d])')
for _ in range(8):
    a, b, c = rng.randint(3, 9), rng.randint(5, 20), rng.randint(2, 6)
    add('word', f'A shop sells pens in boxes of {a}. Maria buys {b} boxes and then gives away {c} pens. How many pens does she have left?', rf'(?<!\d){a * b - c}(?!\d)')
for _ in range(8):
    w = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(rng.randint(6, 9)))
    add('string', f'Reverse the string "{w}".', re.escape(w[::-1]))
for _ in range(8):
    xs = rng.sample(range(10, 99), 6)
    add('sort', f'Sort these numbers in ascending order: {", ".join(map(str, xs))}.', r'\D*'.join(map(str, sorted(xs))))
for _ in range(6):
    w = ''.join(rng.choice('aabcdeeiklmnooprstu') for _ in range(rng.randint(10, 14)))
    add('count', f'How many vowels (a, e, i, o, u) are in the string "{w}"?', rf'(?<!\d){sum(ch in "aeiou" for ch in w)}(?!\d)')
for q, a in [('What is the capital of France?', 'Paris'), ('What is the chemical symbol for gold?', 'Au'), ('How many days are in a leap year?', '366'),
             ('Which planet is closest to the Sun?', 'Mercury'), ('Who wrote the play Hamlet?', 'Shakespeare'), ('What is the square root of 144?', '12')]:
    add('fact', q, re.escape(a))

ok = {}
tot = {}
fails = []
for kind, prompt, expect in tasks:
    body = {'messages': [{'role': 'user', 'content': prompt + ' Answer with only the final answer.'}], 'max_tokens': 60, 'temperature': 0, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=600))
    txt = r['choices'][0]['message']['content'] or ''
    good = re.search(expect, txt, re.I) is not None
    tot[kind] = tot.get(kind, 0) + 1
    ok[kind] = ok.get(kind, 0) + good
    if not good:
        fails.append((prompt[:60], txt[:60]))
print('correct %d/%d: %s' % (sum(ok.values()), sum(tot.values()), ', '.join(f'{k} {ok.get(k, 0)}/{tot[k]}' for k in tot)))
for f in fails[:6]:
    print('  miss:', f)
