"""power_run.py PORT PHYSICAL_INDEX OUT.json : drives a running `shadow serve` through heavy phases while sampling the card's power every 200 ms."""
import json, subprocess, sys, time, urllib.request, datetime, statistics, csv, io
port, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
text = open('J:/Shadow/docs/STATUS.md', encoding='utf-8').read()

def chat(prompt, n):
    body = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': n, 'temperature': 0, 'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
    return r['timings']

samples_file = out + '.samples.csv'
open(samples_file, 'w').close()
smi = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,power.draw,clocks.sm,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits', '-lms', '200', '-f', samples_file])
marks = []
def phase(name, fn):
    t0 = datetime.datetime.now()
    r = fn()
    t1 = datetime.datetime.now()
    marks.append((name, t0, t1, r))
    print(name, r, flush=True)

chat('Hello', 4)
time.sleep(2)
phase('prompt 20k', lambda: [chat(text[20000 + 9000 * i: 20000 + 9000 * i + 60000] + '\n\nQuestion: one sentence, what is this about?', 8)['prompt_per_second'] for i in range(2)])
phase('prompt 9.5k', lambda: [chat(text[30000 + 9000 * i: 30000 + 9000 * i + 28000] + '\n\nQuestion: one sentence, what is this about?', 8)['prompt_per_second'] for i in range(3)])
phase('decode 700 tokens', lambda: [chat('Write a long, detailed story about a lighthouse keeper, at least 600 words.', 700)['predicted_per_second'] for i in range(3)])
time.sleep(2)
smi.terminate()
rows = []
time.sleep(1)
for line in open(samples_file, encoding='utf-8', errors='replace').read().splitlines():
    f = [x.strip() for x in line.split(',')]
    try:
        ts = datetime.datetime.strptime(f[0], '%Y/%m/%d %H:%M:%S.%f')
        rows.append((ts, int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
    except Exception:
        pass
res = {}
for name, t0, t1, r in marks:
    w = [x[2] for x in rows if x[1] == idx and t0 <= x[0] <= t1]
    if not w:
        print(name, 'no samples'); continue
    res[name] = {'speed': r, 'samples': len(w), 'max_w': max(w), 'p99_w': sorted(w)[int(0.99 * len(w))], 'p95_w': sorted(w)[int(0.95 * len(w))], 'mean_w': statistics.mean(w),
                 'max_temp': max(x[5] for x in rows if x[1] == idx and t0 <= x[0] <= t1), 'max_sm_mhz': max(x[3] for x in rows if x[1] == idx and t0 <= x[0] <= t1)}
    print(f"{name:20} max {res[name]['max_w']:6.1f} W  p99 {res[name]['p99_w']:6.1f}  p95 {res[name]['p95_w']:6.1f}  mean {res[name]['mean_w']:6.1f}  temp {res[name]['max_temp']:.0f} C  sm {res[name]['max_sm_mhz']:.0f} MHz  ({res[name]['samples']} samples)")
json.dump(res, open(out, 'w'), indent=1)
