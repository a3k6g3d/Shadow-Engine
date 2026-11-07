"""Replays a stored conversation snapshot (J:\\ShadowConvCache\\*.cnv) against a running `shadow serve`: the snapshot's tokens are turned back into text, split into chat messages,
and sent as a request, so a long agent conversation can be answered again under other settings (old defaults, `--exact`) and the replies compared.

  py tools/replay_snapshot.py FILE.cnv [port] [max_tokens] [gguf]      # prints the reply (reasoning and content) and its timings
  py tools/replay_snapshot.py --list                                    # the snapshots on disk, newest last

The text is rebuilt from the model's vocabulary (GPT-2 byte mapping), so it matches what the model was given up to tokenization details.  Temperature 0.
"""
import glob, json, os, re, struct, sys, time, urllib.request

DIR = r'J:/ShadowConvCache'
GGUF = r'J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'


def read_tokens(path):
    with open(path, 'rb') as fh:
        h = fh.read(24)
        assert h[:8] == b'SHDWCNV1', 'not a snapshot file'
        n = struct.unpack('<Q', h[16:24])[0]
        return list(struct.unpack('<%dI' % n, fh.read(4 * n)))


def vocab(gguf):
    f = open(gguf, 'rb')
    assert f.read(4) == b'GGUF'
    _ver, _nt, nkv = struct.unpack('<IQQ', f.read(20))

    def rs():
        n, = struct.unpack('<Q', f.read(8))
        return f.read(n).decode('utf-8', 'replace')

    def rv(t):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sizes:
            return f.read(sizes[t])
        if t == 8:
            return rs()
        if t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            return [rv(et) for _ in range(n)]
        raise Exception(t)

    for _ in range(nkv):
        k = rs()
        t, = struct.unpack('<I', f.read(4))
        v = rv(t)
        if k == 'tokenizer.ggml.tokens':
            return v


def detokenize(ids, toks):
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}
    keep = ('<think>', '</think>', '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>')
    out = bytearray()
    for i in ids:
        s = toks[i]
        if s.startswith('<|') or s in keep:
            out += s.encode()
        else:
            out += bytes(dec.get(ch, 63) for ch in s)
    return out.decode('utf-8', 'replace')


def to_messages(text):
    msgs = []
    for m in re.finditer(r'<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>\n?|$)', text, re.S):
        role, body = m.group(1), m.group(2)
        if role == 'assistant' and m.end() >= len(text.rstrip()) and body.strip() in ('<think>', ''):
            continue  # the open generation prompt at the end
        msgs.append({'role': role, 'content': body})
    return msgs


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        rows = []
        for p in glob.glob(DIR + '/*.cnv'):
            try:
                n = len(read_tokens(p))
            except Exception:
                continue
            rows.append((os.path.getmtime(p), n, p))
        for m, n, p in sorted(rows)[-15:]:
            print(f'{(time.time() - m) / 60:7.1f} min ago {n:6d} tokens {os.path.basename(p)}')
        sys.exit(0)
    path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8081
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    gguf = sys.argv[4] if len(sys.argv) > 4 else GGUF
    ids = read_tokens(path)
    text = detokenize(ids, vocab(gguf))
    msgs = to_messages(text)
    # REPLAY_APPEND_CHARS=N: a further tool result of N characters (docs/STATUS.md text) after the conversation's last message, as when a tool returns a big page: the prompt
    # then passes the context window's budget and the server compacts it (older tool outputs shortened, older turns folded into a summary)
    extra = int(os.environ.get('REPLAY_APPEND_CHARS', '0'))
    if extra:
        page = open('J:/Shadow/docs/STATUS.md', encoding='utf-8').read()[:extra]
        msgs.append({'role': 'assistant', 'reasoning_content': 'I need the next page of the list.', 'content': 'Let me read the next page.\n\n<tool_call>\n<function=web_fetch>\n<parameter=url>\nhttps://example.org/page\n</parameter>\n</function>\n</tool_call>'})
        msgs.append({'role': 'user', 'content': '<tool_response>\n<tool_response>\nhttps://example.org/page\n' + page + '\n</tool_response>\n</tool_response>'})
    print(f'{len(ids)} tokens -> {len(text)} chars -> {len(msgs)} messages ({", ".join(m["role"][0] for m in msgs[:3])} ... {msgs[-1]["role"]})', flush=True)
    if os.environ.get('REPLAY_TWICE') == '1' and extra:
        base_msgs = msgs[:-2]
        warm = {'messages': base_msgs, 'max_tokens': 8, 'temperature': 0, 'stream': False}
        r0 = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(warm).encode(), {'Content-Type': 'application/json'}), timeout=3000))
        print('warm-up request:', r0.get('timings', {}).get('prompt_n'), 'prompt tokens read', flush=True)
    body = {'messages': msgs, 'max_tokens': max_tokens, 'temperature': 0, 'stream': False}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3000))
    m = r['choices'][0]['message']
    tm = r.get('timings', {})
    print(f'prompt {tm.get("prompt_n")} tokens ({tm.get("prompt_per_second", 0):.0f} tok/s), reply {tm.get("predicted_n")} tokens ({tm.get("predicted_per_second", 0):.1f} tok/s), {time.time() - t0:.0f} s, finish {r["choices"][0].get("finish_reason")}')
    print('--- reasoning ---')
    print((m.get('reasoning_content') or '')[:1500])
    print('--- content ---')
    print((m.get('content') or '')[:1500])
    print('--- tool_calls ---')
    print(json.dumps(m.get('tool_calls'))[:600] if m.get('tool_calls') else None)
