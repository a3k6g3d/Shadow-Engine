"""Writes a bench copy of J:/Strata-work/strata.json: same engine flags, but nothing outward-facing or state-changing
(Discord off, no MCP tools, no conversation store, no learned-profile save, slots and log to scratch, loopback, no
API key), and matched to Shadow's bench (no vision, context 8192, greedy)."""
import json
import sys

out_dir = sys.argv[1]
c = json.load(open('J:/Strata-work/strata.json'))
a = c['args']
drop_with_value = {'--expert-profile-save'}
drop_alone = {'--vision'}
args = []
i = 0
while i < len(a):
    if a[i] in drop_with_value:
        i += 2
        continue
    if a[i] in drop_alone:
        i += 1
        continue
    args.append(a[i])
    i += 1
args[args.index('--max-context') + 1] = '8192'
c['args'] = args
for k in ('mcp_servers', 'mcp', 'vision', 'api_key'):
    c.pop(k, None)
c['discord'] = {'enabled': False}
c['conversation_store'] = {'enabled': False}
c['slot_save_path'] = out_dir + '/slots'
c['log'] = out_dir + '/strata-engine.log'
c['host'] = '127.0.0.1'
c['sampling'] = {'temperature': 0.0, 'top_p': 1.0, 'top_k': 1}
json.dump(c, open(out_dir + '/strata-bench.json', 'w'), indent=1)
print(' '.join(args))
