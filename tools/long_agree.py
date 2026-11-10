"""Long-prompt reply agreement: the short suites (quality_test, task_check) never reach the long-prompt MoE path (their chunks are too short for the device planner's stage pipeline), so this
does.  Eight prompts, each a window of docs/STATUS.md (about 5k or 9.5k tokens, at different offsets) and a question about it, temperature 0, 96 tokens, no prompt cache.

  py tools/long_agree.py run PORT OUT.json          # against a running `shadow serve`: save the replies
  py tools/long_agree.py cmp REF.json A.json [B.json ...]    # mean prefix match (and replies identical) of each file against the reference, and each pair of the others

Mean prefix match is the quality_test metric: how far a reply follows the reference's, as a fraction of the reference's length (harsh: one different token ends it)."""
import json, sys, urllib.request

NL = chr(10)


def prompts():
    text = open('J:/Shadow/docs/STATUS.md', encoding='utf-8').read()
    out = []
    for i, (chars, off) in enumerate([(15000, 40000), (15000, 90000), (15000, 140000), (15000, 190000), (28000, 20000), (28000, 70000), (28000, 120000), (28000, 170000)]):
        doc = text[off:off + chars]
        out.append(f'[long-agree-{i}]' + NL + doc + NL + NL + 'Question: in two sentences, what is the most important measured result described in the text above, with its numbers?')
    return out


def run(port, path):
    replies = []
    for p in prompts():
        body = {'messages': [{'role': 'user', 'content': p}], 'max_tokens': 96, 'temperature': 0, 'stream': False, 'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}
        r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=1800))
        replies.append(r['choices'][0]['message'].get('content') or '')
        print(len(replies), r['timings']['prompt_n'], 'tokens:', replies[-1][:90].replace(NL, ' '), flush=True)
    json.dump(replies, open(path, 'w'))


def prefix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k / max(len(a), 1)


def cmp(ref_path, paths):
    ref = json.load(open(ref_path))
    files = {p: json.load(open(p)) for p in paths}
    for p, rep in files.items():
        m = [prefix(ref[i], rep[i]) for i in range(len(ref))]
        print(f'{p} vs {ref_path}: mean prefix match {sum(m) / len(m):.3f}, {sum(1 for x in m if x >= 0.999)} of {len(m)} identical; per item', ' '.join(f'{x:.2f}' for x in m))
    ps = list(files)
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            m = [prefix(files[ps[i]][k], files[ps[j]][k]) for k in range(len(ref))]
            print(f'{ps[i]} vs {ps[j]}: mean prefix match {sum(m) / len(m):.3f}, {sum(1 for x in m if x >= 0.999)} of {len(m)} identical')


if sys.argv[1] == 'run':
    run(int(sys.argv[2]), sys.argv[3])
else:
    cmp(sys.argv[2], sys.argv[3:])
