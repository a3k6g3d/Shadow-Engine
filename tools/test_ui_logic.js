// The chat page's own logic (pinned notes, recall of old messages, the numbered summary of folded turns) run in Node on a fake conversation: node tools/test_ui_logic.js
const fs = require('fs');
const path = require('path');
const ui = fs.readFileSync(path.join(__dirname, '..', 'crates', 'shadow-serve', 'src', 'ui.rs'), 'utf8');
function grab(name) {
  const i = ui.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('no ' + name);
  let j = ui.indexOf('{', i), d = 0, k = j;
  for (; k < ui.length; k++) { if (ui[k] === '{') d++; else if (ui[k] === '}') { d--; if (d === 0) break; } }
  return ui.slice(i, k + 1);
}
const names = ['runNotesTool', 'withNotes', 'recallMessages', 'buildContextMessages', 'parseToolCallsFromText', 'stripToolCalls', 'messageText', 'unwrapToolResponse'];
const src = names.map(grab).join('\n');
const factory = new Function('state', `
  const NOTES_MAX = 6000;
  let scratchpad = state.scratchpad;
  const history = state.history;
  let serverCtx = state.serverCtx;
  const renderNotes = () => {}, saveCurrent = () => {};
  const timingInfo = { textContent: '' }, document = { getElementById: () => ({}) };
  ${src}
  return { runNotesTool, withNotes, recallMessages, buildContextMessages, getNotes: () => scratchpad, setNotes: v => { scratchpad = v; } };
`);
let fails = 0;
const ok = (c, m) => { if (!c) { fails++; console.log('FAIL:', m); } else console.log('ok:  ', m); };

// ---- notes tool and injection
const st = { scratchpad: '', history: [], serverCtx: 25600 };
const t = factory(st);
ok(t.runNotesTool({}).output === 'Your notes are empty.', 'empty notes are reported');
ok(t.runNotesTool({ text: 'PLAN: edit ui.rs line 2415' }).output.startsWith('Notes saved'), 'text replaces the notes');
t.runNotesTool({ append: 'FACT: the table is built in dashRender' });
ok(t.getNotes() === 'PLAN: edit ui.rs line 2415\nFACT: the table is built in dashRender', 'append adds a line');
const big = t.runNotesTool({ text: 'x'.repeat(6001) });
ok(big.is_error && t.getNotes().startsWith('PLAN'), 'notes over the limit are refused and the old ones kept');
const toolMsg = { role: 'tool', content: '<tool_response>\nfile text\n</tool_response>' };
const w1 = t.withNotes([{ role: 'user', content: 'hi' }, toolMsg]);
ok(w1[1].content.endsWith('[Your notes so far - pinned, never shortened; change them with the notes tool]\nPLAN: edit ui.rs line 2415\nFACT: the table is built in dashRender\n</tool_response>') && w1[1].content.startsWith('<tool_response>\nfile text'), 'notes go inside the last tool response');
ok(w1[0].content === 'hi' && toolMsg.content === '<tool_response>\nfile text\n</tool_response>', 'earlier messages and the history message are not touched');
const w2 = t.withNotes([{ role: 'user', content: 'question' }]);
ok(w2[0].content.startsWith('question\n\n[Your notes so far'), 'notes are appended to a user message');
const w3 = t.withNotes([{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'x' } }, { type: 'text', text: 'look' }] }]);
ok(w3[0].content[1].text.startsWith('look\n\n[Your notes so far') && w3[0].content[0].type === 'image_url', 'notes go in the last text part of a message with a picture');
t.setNotes('');
ok(t.withNotes([{ role: 'user', content: 'hi' }])[0].content === 'hi', 'no notes, no change');

// ---- recall of messages
st.history.push({ role: 'user', content: 'Add a latency panel.\nIt needs p50 and p95.\nAnd a queue-wait column.' });
st.history.push({ role: 'assistant', content: 'Plan: 1) read ui.rs 2) add the card' });
st.history.push({ role: 'tool', content: '<tool_response>\n(output #7)\nline a\nline b NEEDLE here\nline c\n</tool_response>' });
const r1 = t.recallMessages({ message: 0, start: 2, end: 3 });
ok(r1.output.includes('message m0 (user), lines 2-3 of 3') && r1.output.includes('p50 and p95') && r1.output.includes('queue-wait') && !r1.output.includes('latency panel'), 'recall message by number and lines');
const r2 = t.recallMessages({ message: 'm2', query: 'needle' });
ok(r2.output.includes('line b NEEDLE here') && !r2.output.includes('line a') && !r2.output.includes('(output #7)'), 'recall message m2 by text, header dropped');
ok(t.recallMessages({ message: 9 }).is_error, 'an unknown message is an error');
const r3 = t.recallMessages({ query: 'queue-wait' });
ok(r3.output.startsWith('m0:3| And a queue-wait column.'), 'a text alone searches all messages');

// ---- the fold names its messages
const h = [];
for (let k = 1; k <= 14; k++) {
  h.push({ role: 'user', content: `Turn ${k}: ` + 'question words '.repeat(60) });
  h.push({ role: 'assistant', content: `Answer ${k}: ` + 'answer words '.repeat(60) + '\n<tool_call>\n<function=cmd>\n<parameter=command>\ndir\n</parameter>\n</function>\n</tool_call>' });
  h.push({ role: 'tool', content: '<tool_response>\n(output #' + k + ')\n' + 'listing '.repeat(120) + '\n</tool_response>' });
}
h.push({ role: 'user', content: 'What did we find?' });
const f = factory({ scratchpad: '', history: h, serverCtx: 8000 });
const out = f.buildContextMessages(h);
const first = out[0].content;
ok(/\[Earlier conversation summary/.test(first) && /recall message=N/.test(first), 'the summary says how to read a turn in full');
ok(/Turn 1 \(messages m0-m2\): User: "Turn 1:/.test(first), 'the first folded turn is m0-m2');
ok(/Turn 2 \(messages m3-m5\): User: "Turn 2:/.test(first), 'the second folded turn continues at m3');
const shortened = out.filter(m => m.role === 'tool' && /earlier tool output/.test(m.content));
ok(shortened.every(m => /recall (with id \d+|with message \d+)/.test(m.content)), 'a shortened output says how to read it again: ' + (shortened[0] ? shortened[0].content.slice(0, 140).replace(/\n/g, ' ') : 'none shortened'));
ok(out[out.length - 1].content === 'What did we find?', 'the last message is kept');
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
