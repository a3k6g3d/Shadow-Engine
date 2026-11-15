import json
from huggingface_hub import HfApi
api = HfApi()
seen = {}
queries = ["Qwen3.8-Flash-Next exl3", "Qwen3.8-Flash-Next-Uncensored exl3", "Qwen3.8-Flash-Next abliterated exl3", "Qwen3.8-Flash-Next heretic exl3",
           "Flash-Next uncensored exl3", "Flash-Next abliterated exl3", "Qwen3.8 Flash Next EXL3 2bpw", "Qwen3.8 Flash Next EXL3 2.5bpw", "Qwen3.8-Flash-Next exllamav3",
           "Huihui Qwen3.8-Flash-Next exl3", "Swift Qwen3.8-Flash-Next exl3", "Qwen3.8-Flash-Next-Uncensored", "Qwen3.8-Flash-Next-abliterated"]
for q in queries:
    try:
        for m in api.list_models(search=q, limit=100):
            seen[m.id] = m
    except Exception as e:
        print('query failed', q, e)
rows = []
for mid, m in seen.items():
    tags = set(m.tags or [])
    low = mid.lower()
    is_exl3 = 'exl3' in tags or 'exl3' in low or 'exllamav3' in tags
    unc = any(w in low for w in ('uncensor', 'abliter', 'heretic', 'huihui', 'unfilter', 'nsfw', 'dolphin', 'refusal')) or any(w in tags for w in ('abliterated', 'uncensored'))
    rows.append((mid, is_exl3, unc, m.downloads or 0, m.likes or 0, str(m.created_at)[:10], bool(getattr(m, 'gated', False))))
rows.sort(key=lambda r: (-r[1], -r[2], -r[3]))
for r in rows:
    if r[1]:
        print(('UNC ' if r[2] else '    '), r[0], 'dl', r[3], 'likes', r[4], r[5], 'GATED' if r[6] else '')
print('--- non-exl3 uncensored (for reference):', sum(1 for r in rows if not r[1] and r[2]))
