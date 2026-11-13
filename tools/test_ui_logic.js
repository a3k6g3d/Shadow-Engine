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
const names = ['readOnlyStreak', 'treadmillNote', 'runNotesTool', 'withNotes', 'recallMessages', 'buildContextMessages', 'parseToolCallsFromText', 'stripToolCalls', 'messageText', 'unwrapToolResponse'];
function grabConst(name) {
  const i = ui.indexOf('const ' + name + ' =');
  if (i < 0) throw new Error('no ' + name);
  return ui.slice(i, ui.indexOf(';', i) + 1);
}
const src = ['READ_ONLY_TOOLS', 'PROGRESS_TOOLS'].map(grabConst).join('\n') + '\n' + names.map(grab).join('\n');
// the page's own defaults for the two shortening levers (STATUS 126), so the tests follow them
const lev = /let ctxPreviewChars = (\d+), ctxWorkShare = ([\d.]+);/.exec(ui);
if (!lev) throw new Error('no ctx levers');
const LEVERS = { ctxPreviewChars: +lev[1], ctxWorkShare: +lev[2] };
const factory = new Function('state', `
  const NOTES_MAX = 6000;
  let scratchpad = state.scratchpad;
  const history = state.history;
  let serverCtx = state.serverCtx;
  let ctxPreviewChars = state.ctxPreviewChars, ctxWorkShare = state.ctxWorkShare;
  const renderNotes = () => {}, saveCurrent = () => {};
  const timingInfo = { textContent: '' }, document = { getElementById: () => ({}) };
  ${src}
  return { readOnlyStreak, treadmillNote, runNotesTool, withNotes, recallMessages, buildContextMessages, getNotes: () => scratchpad, setNotes: v => { scratchpad = v; } };
`);
let fails = 0;
const ok = (c, m) => { if (!c) { fails++; console.log('FAIL:', m); } else console.log('ok:  ', m); };

// ---- notes tool and injection
const st = { scratchpad: '', history: [], serverCtx: 25600, ...LEVERS };
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
const f = factory({ scratchpad: '', history: h, serverCtx: 8000, ...LEVERS });
const out = f.buildContextMessages(h);
const first = out[0].content;
ok(/\[Earlier conversation summary/.test(first) && /recall message=N/.test(first), 'the summary says how to read a turn in full');
ok(/Turn 1 \(messages m0-m2\): User: "Turn 1:/.test(first), 'the first folded turn is m0-m2');
ok(/Turn 2 \(messages m3-m5\): User: "Turn 2:/.test(first), 'the second folded turn continues at m3');
const shortened = out.filter(m => m.role === 'tool' && /earlier tool output/.test(m.content));
ok(shortened.every(m => /recall (with id \d+|with message \d+)/.test(m.content)), 'a shortened output says how to read it again: ' + (shortened[0] ? shortened[0].content.slice(0, 140).replace(/\n/g, ' ') : 'none shortened'));
ok(out[out.length - 1].content === 'What did we find?', 'the last message is kept');

// ---- one long agent run (a single user message, 85 steps, big outputs): the oldest steps fold into one line each
const run = [{ role: 'user', content: 'Improve the dashboard: add p50/p95, acceptance rate, expert tiers.' }];
for (let k = 1; k <= 85; k++) {
  run.push({ role: 'assistant', content: `Step ${k}: reading the part I need.\n<tool_call>\n<function=read_file>\n<parameter=path>\ncrates/shadow-serve/src/ui.rs\n</parameter>\n<parameter=start>\n${k * 10}\n</parameter>\n<parameter=end>\n${k * 10 + 40}\n</parameter>\n</function>\n</tool_call>` });
  run.push({ role: 'tool', content: `<tool_response>\n(output #${k})\n` + `line of code number ${k} `.repeat(300) + '\n</tool_response>' });
}
const g = factory({ scratchpad: '', history: run, serverCtx: 25600, ...LEVERS });
const sz = m => (typeof m.content === 'string' ? m.content.length : 0);
const room = Math.max(6000, (25600 - 3800) * 3);
const folded = g.buildContextMessages(run);
const totalFolded = folded.reduce((n, m) => n + sz(m), 0);
ok(totalFolded < room * 0.8, `the run fits: ${totalFolded} of ${room} characters (it was ${run.reduce((n, m) => n + sz(m), 0)})`);
ok(/\[Earlier steps of this task, shortened/.test(folded[0].content) && /\[The task\]\nImprove the dashboard/.test(folded[0].content), 'the first message carries the steps summary and still the task');
ok(/- m\d+-m\d+: read_file\(crates.shadow-serve.src.ui\.rs \d+-\d+\) -> output #\d+ \| Step \d+: reading the part I need\./.test(folded[0].content), 'a step line names its messages, the call, the output number and the note');
ok(folded[1].role === 'assistant', 'the first message after the task is an assistant message (a step starts there)');
const tailSame = JSON.stringify(folded.slice(-8)) === JSON.stringify(run.slice(-8).map((m, i) => (m.role === 'tool' && i < 8 ? folded[folded.length - 8 + i] : m)));
ok(folded.slice(-2).every((m, i) => m.content === run[run.length - 2 + i].content), 'the last step is kept whole');
ok(folded.length < run.length * 0.7, `messages: ${folded.length} of ${run.length}`);
// message numbers continue where the fold stopped
const m1 = /- m(\d+)-m(\d+): read_file/.exec(folded[0].content.split('\n').slice(-12).join('\n'));
ok(m1 && Number(m1[2]) < run.length - 8, 'the newest folded line ends before the kept tail');
// a very long run (260 steps): the oldest lines merge into one, the newest 40 stay
const NL = String.fromCharCode(10);
const longRun = [{ role: 'user', content: 'Task.' }];
for (let k = 1; k <= 260; k++) {
  longRun.push({ role: 'assistant', content: ['Step ' + k + '.', '<tool_call>', '<function=grep>', '<parameter=pattern>', 'foo' + k, '</parameter>', '</function>', '</tool_call>'].join(NL) });
  longRun.push({ role: 'tool', content: ['<tool_response>', '(output #' + k + ')', 'match line '.repeat(200), '</tool_response>'].join(NL) });
}
const foldedLong = factory({ scratchpad: '', history: longRun, serverCtx: 25600, ...LEVERS }).buildContextMessages(longRun);
const lines = foldedLong[0].content.split(NL).filter(l => l.startsWith('- m'));
ok(lines.length <= 41 && /^- m1-m\d+: \d+ earlier steps \(grepx\d+\)/.test(lines[0]), 'the oldest steps are merged into one line: ' + lines[0]);
ok(foldedLong.reduce((n, m) => n + sz(m), 0) < room * 0.8, 'the 260-step run fits too');
// a short run is left alone
const small = run.slice(0, 7);
ok(JSON.stringify(g.buildContextMessages(small)) === JSON.stringify(small), 'a short run is not touched');

// ---- the working set: the newest tool outputs stay whole within a budget, older ones shrink
const NL2 = String.fromCharCode(10);
const ws = [{ role: 'user', content: 'Edit the code.' }];
for (let k = 1; k <= 10; k++) {
  ws.push({ role: 'assistant', content: ['Step ' + k + '.', '<tool_call>', '<function=read_file>', '<parameter=path>', 'a.rs', '</parameter>', '</function>', '</tool_call>'].join(NL2) });
  ws.push({ role: 'tool', content: ['<tool_response>', '(output #' + k + ')', 'x'.repeat(4200), '</tool_response>'].join(NL2) });
}
const wsOut = factory({ scratchpad: '', history: ws, serverCtx: 14000, ...LEVERS }).buildContextMessages(ws);
const wsTools = wsOut.filter(m => m.role === 'tool');
const wsWhole = wsTools.map(m => m.content.length > 4000);
ok(wsWhole.length === 10 && wsWhole[9] && wsWhole[8], 'the newest two tool outputs stay whole (a working set of more than one)');
ok(!wsWhole[8 - 1] || wsTools.slice(0, 7).some(m => m.content.startsWith('<tool_response>' + NL2 + '[earlier tool output')), 'older outputs shrink to previews');
ok(wsTools[0].content.includes('[earlier tool output #1 '), 'the oldest output is a preview that names its number');
const wsBudget = Math.max(6000, (14000 - 3800) * 3) * 0.3;
ok(wsTools.filter(m => m.content.length > 4000).length * 4300 <= wsBudget + 4300 * 1 + 1, 'the whole outputs fit the budget (the last step always stays)');

// ---- the read-only treadmill: a note after 8 and again from 14, reset by notes or an edit
const tm = factory({ scratchpad: '', history: [], serverCtx: 25600, ...LEVERS });
let streak = 0;
const notes = [];
for (let i = 1; i <= 20; i++) {
  streak = tm.readOnlyStreak(i % 2 ? 'read_file' : 'recall', streak);
  const n = tm.treadmillNote(streak);
  if (n) notes.push([streak, n.slice(0, 40)]);
}
ok(streak === 20 && JSON.stringify(notes.map(x => x[0])) === JSON.stringify([8, 14, 18]), 'notes at 8, then 14 and every 4 after: ' + JSON.stringify(notes.map(x => x[0])));
ok(/read-only calls in a row without saving notes or editing/.test(notes[0][1] + tm.treadmillNote(8)) && /Stop reading/.test(tm.treadmillNote(14)), 'the first note asks for notes and an edit, the second says stop reading');
ok(tm.readOnlyStreak('notes', 12) === 0 && tm.readOnlyStreak('replace_lines', 12) === 0 && tm.readOnlyStreak('edit_file', 12) === 0, 'notes and edits reset the streak');
ok(tm.readOnlyStreak('cmd', 5) === 5 && tm.readOnlyStreak('web_fetch', 5) === 5, 'other tools neither add nor reset');
ok(tm.treadmillNote(7) === '' && tm.treadmillNote(9) === '' && tm.treadmillNote(15) === '', 'no note in between');
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
