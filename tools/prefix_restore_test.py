"""Regression test for the garbled reply of a chat that starts from the stored shared-prefix snapshot (docs/STATUS.md 2026-11-06 (53)).

  py tools/prefix_restore_test.py [port] [items]      # against a running `shadow serve` (fast mode, default flags)

Two short requests that share the system prompt and the tool list make the server keep a snapshot of that shared start; the next request then restores it and reads the rest of a
~7k-token conversation (tool results from docs/STATUS.md with a code word buried in one of them), and the reply to the question at its end must contain the code.  A server that
replays a CUDA graph over moved buffers answers with garbage (or a wrong or empty reply) here; a good one finds the code every time.  Each item uses a fresh random id
at its start, so the long request is never answered from a stored conversation."""
import json, random, sys, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
n_items = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NL = chr(10)
text = open('J:/Shadow/docs/STATUS.md', encoding='utf-8').read()
rng = random.Random(20261106)
words = ['ZEBRA', 'MAPLE', 'COBALT', 'FALCON', 'ORCHID', 'TUNDRA', 'QUARTZ', 'LANTERN']
tools = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/tools', timeout=30))
if isinstance(tools, dict):
    tools = tools.get('tools', tools.get('data', tools))
system = {'role': 'system', 'content': 'You are Shadow, an assistant on the user\'s Windows PC. Be direct and use the tools when asked.'}


def post(messages, max_tokens):
    body = {'messages': messages, 'stream': False, 'reasoning_effort': 'low', 'chat_template_kwargs': {'enable_thinking': False}, 'tools': tools, 'max_tokens': max_tokens, 'temperature': 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=1500))
    return r['choices'][0]['message'].get('content') or '', r.get('timings', {})


ok = 0
for i in range(n_items):
    for short in ('Hi', 'Hello, are you there?'):   # two requests sharing the system prompt and tools: the server snapshots that start
        post([system, {'role': 'user', 'content': f'{short} [{rng.randint(10**8, 10**9)}]'}], 4)
    code = f'{rng.choice(words)}-{rng.randint(1000, 9999)}'
    msgs = [system, {'role': 'user', 'content': f'[{rng.randint(10**8, 10**9)}] Read these pages and then answer my question at the end.'}]
    for k in range(3):   # three tool results of ~2,000 tokens each
        a = rng.randint(0, len(text) - 7000)
        page = text[a:a + 6500]
        if k == 1:
            cut = len(page) // 2
            page = page[:cut] + NL + f'Note: the vault code is {code}.' + NL + page[cut:]
        msgs.append({'role': 'assistant', 'content': f'<tool_call>{NL}<function=web_fetch>{NL}<parameter=url>{NL}https://example.org/p{k}{NL}</parameter>{NL}</function>{NL}</tool_call>'})
        msgs.append({'role': 'user', 'content': f'<tool_response>{NL}{page}{NL}</tool_response>'})
    msgs.append({'role': 'assistant', 'content': 'I have read the pages.'})
    msgs.append({'role': 'user', 'content': 'What is the vault code mentioned in the pages? Reply with only the code.'})
    reply, tm = post(msgs, 24)
    good = code in reply
    ok += good
    print(f'  item {i}: {"ok" if good else "MISS"} (expected {code}, got {reply[:60]!r}); read {tm.get("prompt_n")} tokens', flush=True)
print(f'found {ok}/{n_items}')
sys.exit(0 if ok == n_items else 1)
