// The chat page's own logic (pinned notes, recall of old messages, the numbered summary of folded turns) run in Node on a fake conversation: node tools/test_ui_logic.js
const fs = require('fs');
const path = require('path');
const ui = fs.readFileSync(path.join(__dirname, '..', 'crates', 'shadowzero-serve', 'src', 'ui.rs'), 'utf8');
function grab(name) {
  let i = ui.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('no ' + name);
  if (i >= 6 && ui.slice(i - 6, i) === 'async ') i -= 6;
  let j = ui.indexOf('{', i), d = 0, k = j;
  for (; k < ui.length; k++) { if (ui[k] === '{') d++; else if (ui[k] === '}') { d--; if (d === 0) break; } }
  return ui.slice(i, k + 1);
}
const names = ['readOnlyStreak', 'treadmillNote', 'runNotesTool', 'withNotes', 'recallMessages', 'buildContextMessages', 'findToolCallSpans', 'parseToolCallsFromText', 'hasMalformedToolCall', 'stripToolCalls', 'messageText', 'unwrapToolResponse', 'latestDecodeSpeed', 'formatCtxUsage', 'classifyEmptyStop', 'unfinishedTurnNudge', 'parseSessionsList', 'shouldAutoRollover', 'buildRolloverPrompt', 'renderMediaCardsHtml', 'toolKind', 'toolStep', 'toolDetailHtml', 'toolCardHtml', 'toolGroupText', 'toolGroupHtml', 'executeLocalTool', 'executeLocalToolRemote', 'sortChatsNewestFirst', 'partitionProjectChats', 'archiveOlderKeepCount', 'resolveInitialRestore', 'filterRequestTools'];
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
  let collapseTools = !!state.collapseTools;
  const baseName = p => String(p === undefined || p === null ? '' : p).replace(/[\\/]+$/, '').split(/[\\/]/).pop() || String(p);
  const clipText = (s, n) => { s = String(s === undefined || s === null ? '' : s).replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n - 1) + '\u2026' : s; };
  let ctxPreviewChars = state.ctxPreviewChars, ctxWorkShare = state.ctxWorkShare;
  const renderNotes = () => {}, saveCurrent = () => {};
  const escapeHtml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const timingInfo = { textContent: '' }, document = { getElementById: () => ({}) };
  const fetch = state.fetch || (async () => ({ ok: true, json: async () => ({}) }));
  ${src}
  return { readOnlyStreak, treadmillNote, runNotesTool, withNotes, recallMessages, buildContextMessages, parseToolCallsFromText, stripToolCalls, latestDecodeSpeed, formatCtxUsage, classifyEmptyStop, unfinishedTurnNudge, parseSessionsList, shouldAutoRollover, buildRolloverPrompt, renderMediaCardsHtml, toolGroupHtml, executeLocalTool, executeLocalToolRemote, sortChatsNewestFirst, partitionProjectChats, archiveOlderKeepCount, resolveInitialRestore, filterRequestTools, getNotes: () => scratchpad, setNotes: v => { scratchpad = v; } };
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
  run.push({ role: 'assistant', content: `Step ${k}: reading the part I need.\n<tool_call>\n<function=read_file>\n<parameter=path>\ncrates/shadowzero-serve/src/ui.rs\n</parameter>\n<parameter=start>\n${k * 10}\n</parameter>\n<parameter=end>\n${k * 10 + 40}\n</parameter>\n</function>\n</tool_call>` });
  run.push({ role: 'tool', content: `<tool_response>\n(output #${k})\n` + `line of code number ${k} `.repeat(300) + '\n</tool_response>' });
}
const g = factory({ scratchpad: '', history: run, serverCtx: 25600, ...LEVERS });
const sz = m => (typeof m.content === 'string' ? m.content.length : 0);
const room = Math.max(6000, (25600 - 3800) * 3);
const folded = g.buildContextMessages(run);
const totalFolded = folded.reduce((n, m) => n + sz(m), 0);
ok(totalFolded < room * 0.8, `the run fits: ${totalFolded} of ${room} characters (it was ${run.reduce((n, m) => n + sz(m), 0)})`);
ok(/\[Earlier steps of this task, shortened/.test(folded[0].content) && /\[The task\]\nImprove the dashboard/.test(folded[0].content), 'the first message carries the steps summary and still the task');
ok(/- m\d+-m\d+: read_file\(crates.shadowzero-serve.src.ui\.rs \d+-\d+\) -> output #\d+ \| Step \d+: reading the part I need\./.test(folded[0].content), 'a step line names its messages, the call, the output number and the note');
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

// ---- header decode speed matches last reply and empty/malformed stops explain why
ok(Math.abs(tm.latestDecodeSpeed({ process: { tok_per_sec: 52.6 }, recent: [{ completion_tokens: 100, predicted_ms: 2000 }, { completion_tokens: 595, predicted_ms: 10000 }] }) - 59.5) < 1e-6, 'header speed uses last reply decode speed');
ok(Math.abs(tm.latestDecodeSpeed({ process: { tok_per_sec: 52.6 }, recent: [] }) - 52.6) < 1e-6, 'header speed falls back to session avg when recent is empty');
ok(tm.classifyEmptyStop('', 'thought', false).reason === 'model ended turn with an empty reply' && /after thinking/.test(tm.classifyEmptyStop('', 'thought', false).title), 'empty reply after thinking explains why it stopped');
ok(tm.classifyEmptyStop('<tool_call><function=edit_file>', '', false).reason === 'incomplete or unrecognized tool call', 'unparsed tool_call stripped to empty explains why it stopped');
ok(tm.classifyEmptyStop('', '', true).reason === 'cut off at the context limit', 'cut-off empty reply classifies as context limit');
ok(tm.unfinishedTurnNudge('The server side of F2 is complete and compiles. Now the client rewiring in ui.rs. Let me read the exact chat-store block:', '...', { toolsEnabled: true, finishActive: true, loopTurns: 5, cutOff: false, autoNudges: 0 })?.kind === 'dropped_tool_call', 'trailing colon mid-task triggers dropped_tool_call nudge');
ok(tm.unfinishedTurnNudge("No dashmap — std `Mutex<HashMap>` + oneshot is fine. Now the server-side gate module, mirroring the client's `commandIsSafe`/`permissionReason` exactly:", '...', { toolsEnabled: true, finishActive: false, loopTurns: 57, cutOff: false, autoNudges: 0 })?.kind === 'dropped_tool_call', 'trailing colon mid-task still triggers dropped_tool_call nudge when finishActive is false');
ok(tm.unfinishedTurnNudge('Now let me edit lib.rs.', 'Now let me edit lib.rs.', { toolsEnabled: true, finishActive: false, loopTurns: 4, cutOff: false, autoNudges: 0 })?.kind === 'dropped_tool_call', 'forward-looking sentence ending in period triggers dropped_tool_call nudge');
ok(tm.unfinishedTurnNudge("Good — the duplicate is gone. Now let me see how `handle` is invoked and where the peer address can come from (hyper's `ConnectionInfo`).\n#### 4 tool calls", '...', { toolsEnabled: true, finishActive: true, loopTurns: 12, cutOff: false, autoNudges: 0 })?.kind === 'dropped_tool_call', 'fake #### N tool calls heading after intent triggers dropped_tool_call nudge');
ok(tm.unfinishedTurnNudge('My notes are up to date. Now read the accept loop.', '...', { toolsEnabled: true, finishActive: true, loopTurns: 5, cutOff: false, autoNudges: 0 })?.kind === 'dropped_tool_call', 'imperative "Now read..." mid-task triggers dropped_tool_call nudge');
ok(tm.unfinishedTurnNudge('', '<tool_call><function=read_file>', { toolsEnabled: true, finishActive: true, loopTurns: 1, cutOff: false, autoNudges: 0 })?.kind === 'malformed_tool', 'unparsed tool_call triggers malformed_tool nudge');
ok(tm.unfinishedTurnNudge('', '', { toolsEnabled: true, finishActive: true, loopTurns: 3, cutOff: false, autoNudges: 0 })?.kind === 'empty_mid_task', 'empty reply mid-task triggers empty_mid_task nudge');
ok(tm.unfinishedTurnNudge('All done, everything passed.', 'All done, everything passed.', { toolsEnabled: true, finishActive: true, loopTurns: 3, cutOff: false, autoNudges: 0 }) === null, 'complete normal reply does not trigger nudge');
ok(tm.unfinishedTurnNudge('Let me read the file:', 'Let me read the file:', { toolsEnabled: true, finishActive: true, loopTurns: 3, cutOff: false, autoNudges: 3 }) === null, 'auto-nudge stops at cap 3');
ok(tm.parseSessionsList([{ id: 'c1', title: 'hi' }]).length === 1 && tm.parseSessionsList({ sessions: [{ id: 'c2' }] }).length === 1 && tm.parseSessionsList(null).length === 0, 'parseSessionsList accepts array or {sessions}');
ok(tm.shouldAutoRollover({ autoRollover: true, cutOff: true, contextError: false, lastTimings: null, serverCtx: 41984, historyLen: 6, rolloversDone: 0, loopTurns: 3, hasToolCalls: false, canRollover: true }).trigger === true, 'cutOff triggers auto-rollover');
ok(tm.shouldAutoRollover({ autoRollover: true, cutOff: false, contextError: false, lastTimings: { prompt_n: 2000, cache_n: 34000, predicted_n: 2500 }, serverCtx: 41984, historyLen: 10, rolloversDone: 0, loopTurns: 5, hasToolCalls: true, canRollover: true }).trigger === true, 'proactive 90% usage mid-tool-loop triggers auto-rollover');
ok(tm.shouldAutoRollover({ autoRollover: true, cutOff: false, contextError: false, lastTimings: { prompt_n: 2000, cache_n: 34000, predicted_n: 2500 }, serverCtx: 41984, historyLen: 10, rolloversDone: 0, loopTurns: 1, hasToolCalls: false, canRollover: true }).trigger === false, 'proactive 90% does not hijack a finished non-tool reply');
ok(tm.shouldAutoRollover({ autoRollover: true, cutOff: true, contextError: false, lastTimings: null, serverCtx: 41984, historyLen: 4, rolloversDone: 1, loopTurns: 1, hasToolCalls: false, canRollover: true }).trigger === false, 'prevents immediate rollover loop on step 1 after rollover');
const rp = tm.buildRolloverPrompt([
  { role: 'user', content: 'Build features F1 through F6' },
  { role: 'assistant', content: 'Writing gate\n<tool_call>\n<function=read_file>\n<parameter=path>lib.rs</parameter>\n</function>\n</tool_call>' },
  { role: 'tool', content: '<tool_response>\nfn main() {}\n</tool_response>' }
], 'F1-F4 done, F5 in progress', 'Build features', 'c123', 'reply hit the context window limit');
ok(rp.includes('Build features F1 through F6') && rp.includes('read_file(lib.rs)') && rp.includes('pinned Notes are preserved'), 'buildRolloverPrompt carries original task, recent tool step, and notes cue');
const cuIdle = tm.formatCtxUsage(null, 40960, true);
ok(cuIdle.label === 'Ctx: 41.0k' && cuIdle.level === 'ok', 'formatCtxUsage shows context limit cleanly when idle');
const cuWarn = tm.formatCtxUsage({ prompt_n: 2000, cache_n: 29000, predicted_n: 2000 }, 40960, false);
ok(cuWarn.level === 'warn' && cuWarn.label.includes('81%') && /Auto-rollover OFF/.test(cuWarn.title), 'formatCtxUsage flags >=75% as warn');
const cuHigh = tm.formatCtxUsage({ prompt_n: 1000, cache_n: 36000, predicted_n: 1000 }, 40960, true);
ok(cuHigh.level === 'high' && cuHigh.pct === 93 && /Auto-rollover ON/.test(cuHigh.title), 'formatCtxUsage flags >=90% as high');
ok(ui.includes('data-effort="high"') && ui.includes("['off', 'low', 'medium', 'high', 'xhigh']"), 'Reasoning Depth menu and command palette include High effort');
const imgHtml = tm.renderMediaCardsHtml('Generated HD photo (1920x1080) -> /v1/media/photo_1920x1080_a.png', null, false);
ok(/<article class="media-card"><img src="\/v1\/media\/photo_1920x1080_a\.png"/.test(imgHtml) && /width="1920" height="1080"/.test(imgHtml) && !/loading="lazy"/.test(imgHtml), 'live photo renders as crisp img card with natural aspect attrs');
const vidHtml = tm.renderMediaCardsHtml('Generated HD video -> /v1/media/clip_1280x720_b.mp4', null, true);
ok(/<video controls playsinline preload="metadata" loading="lazy" width="1280" height="720" src="\/v1\/media\/clip_1280x720_b\.mp4">/.test(vidHtml), 'saved video renders lazy-loaded video player with explicit dimensions');
ok(ui.includes("GPUs parked (media)"), 'Telemetry handles parked state distinctly');
const tgOpen = tm.toolGroupHtml([{ tc: { name: 'cmd', arguments: { command: 'dir' } }, state: 'done', out: 'ok' }]);
ok(/class="tool-group open"/.test(tgOpen) && /aria-expanded="true"/.test(tgOpen), 'toolGroupHtml defaults to open timeline with aria-expanded=true');
const tgClosed = factory({ scratchpad: '', history: [], serverCtx: 25600, collapseTools: true, ...LEVERS }).toolGroupHtml([{ tc: { name: 'cmd', arguments: { command: 'dir' } }, state: 'done', out: 'ok' }]);
ok(/class="tool-group"/.test(tgClosed) && /aria-expanded="false"/.test(tgClosed), 'toolGroupHtml collapses when collapseTools setting is true');
ok(ui.includes('id="set-collapse-tools"') && ui.includes('border: none;') && !ui.includes('.msg-row.assistant.working > .bubble, .tool-call-block.running { animation: throb'), 'Display toggle and borderless assistant canvas present');
const sm = /<script>([\s\S]*?)<\/script>/.exec(ui);
let jsOk = false;
try { new Function(sm[1]); jsOk = true; } catch (e) { console.log('JS syntax error:', e); }
ok(jsOk, 'main UI script compiles cleanly with zero JS syntax errors');
ok(ui.includes('executeLocalTool(tc.name, tc.arguments, stepSignal)'), 'sendMessage passes stepSignal into executeLocalTool');
ok(ui.includes('id="side-tab-projects"') && ui.includes('focusgroup="tablist nomemory"') && ui.includes('id="side-panel-projects" role="tabpanel" aria-labelledby="side-tab-projects" hidden="until-found"'), 'Sidebar Projects tablist and hidden=until-found panel are wired');
ok(ui.includes('dialog id="project-modal" closedby="any"') && ui.includes('dialog id="assign-project-modal" closedby="any"'), 'Native project creation and assignment dialogs are present');
const sampleProjChats = [
  { id: 'c1', title: 'Oldest task', ts: 100, project_id: 'p1', archived: false },
  { id: 'c2', title: 'Newest task', ts: 300, project_id: 'p1', archived: false },
  { id: 'c3', title: 'Middle task', ts: 200, project_id: 'p1', archived: false },
  { id: 'c4', title: 'Done archived', ts: 50, project_id: 'p1', archived: true },
  { id: 'c5', title: 'Other project', ts: 400, project_id: 'p2', archived: false }
];
const partP1 = tm.partitionProjectChats(sampleProjChats, 'p1', '');
ok(partP1.active.map(x => x.id).join(',') === 'c2,c3,c1', 'partitionProjectChats orders active project conversations newest to oldest');
ok(partP1.archived.map(x => x.id).join(',') === 'c4', 'partitionProjectChats separates archived conversations');
ok(tm.archiveOlderKeepCount(1) === 1 && tm.archiveOlderKeepCount(2) === 1 && tm.archiveOlderKeepCount(5) === 3, 'archiveOlderKeepCount keeps newest 1..3 and archives older chats');
const ktWrite = 'Strip `<tool_call>` and `</tool_call>` in UI.\n<tool_call>\n<function=write_file>\n<parameter=path>\nMainActivity.kt\n</parameter>\n<parameter=content>\nval re = Regex("<tool_call>[\\\\s\\\\S]*?</tool_call>")\n</parameter>\n</function>\n</tool_call>';
const ktCalls = tm.parseToolCallsFromText(ktWrite);
ok(ktCalls.length === 1 && ktCalls[0].name === 'write_file' && ktCalls[0].arguments.path === 'MainActivity.kt' && ktCalls[0].arguments.content.includes('</tool_call>'), 'parseToolCallsFromText preserves literal <tool_call>/</tool_call> inside written file content');
ok(tm.stripToolCalls(ktWrite) === 'Strip `<tool_call>` and `</tool_call>` in UI.', 'stripToolCalls strips real tool call while preserving prose tag mentions');
ok(tm.unfinishedTurnNudge('We strip `<tool_call>` tags cleanly.', 'We strip `<tool_call>` tags cleanly.', { toolsEnabled: true, finishActive: true, loopTurns: 3, cutOff: false, autoNudges: 0 }) === null, 'prose mentioning <tool_call> does not trigger false malformed_tool nudge');
const rExact = tm.resolveInitialRestore('c3', 'p1', sampleProjChats, [{ id: 'p1' }, { id: 'p2' }]);
ok(rExact.chatId === 'c3' && rExact.projectId === 'p1', 'resolveInitialRestore reopens the exact last conversation and its project');
const rProjFallback = tm.resolveInitialRestore('c_deleted', 'p1', sampleProjChats, [{ id: 'p1' }]);
ok(rProjFallback.chatId === 'c2' && rProjFallback.projectId === 'p1', 'resolveInitialRestore falls back to newest active chat in last project when saved chat is gone');
const rBlankInProj = tm.resolveInitialRestore('', 'p1', sampleProjChats, [{ id: 'p1' }]);
ok(rBlankInProj.chatId === null && rBlankInProj.projectId === 'p1', 'resolveInitialRestore preserves a blank new chat started inside a project');
const rFirstRun = tm.resolveInitialRestore(null, null, sampleProjChats, [{ id: 'p1' }, { id: 'p2' }]);
ok(rFirstRun.chatId === 'c5' && rFirstRun.projectId === 'p2', 'resolveInitialRestore opens newest active conversation on first run');
const sampleTools = [{ name: 'cmd' }, { name: 'finish' }, { name: 'pc_screenshot' }, { name: 'pc_mouse' }];
ok(tm.filterRequestTools(sampleTools, true, false).map(x => x.name).join(',') === 'cmd,finish', 'filterRequestTools hides pc_* tools when PC Control is OFF');
ok(tm.filterRequestTools(sampleTools, false, true).map(x => x.name).join(',') === 'cmd,pc_screenshot,pc_mouse', 'filterRequestTools exposes pc_* tools when PC Control is ON');
ok(ui.includes('id="pc-control-banner" role="status" aria-live="polite" hidden') && ui.includes('dialog id="pc-control-modal" closedby="any"'), 'PC Control takeover banner and confirmation dialog are present');
(async () => {
  const acPre = new AbortController();
  acPre.abort();
  let fetchCalled = false;
  const tfPre = factory({ ...st, fetch: async () => { fetchCalled = true; return { ok: true, json: async () => ({}) }; } });
  const rPre = await tfPre.executeLocalTool('cmd', { command: 'echo hi' }, acPre.signal);
  ok(rPre.aborted === true && rPre.is_error === true && !fetchCalled, 'executeLocalTool short-circuits when signal is already aborted');

  const acLive = new AbortController();
  let seenSignal = null;
  const tfLive = factory({
    ...st,
    fetch: (_url, opts) => new Promise((_resolve, reject) => {
      seenSignal = opts && opts.signal;
      seenSignal.addEventListener('abort', () => {
        const err = new Error('The user aborted a request.');
        err.name = 'AbortError';
        reject(err);
      });
    })
  });
  const pending = tfLive.executeLocalTool('cmd', { command: 'gradle' }, acLive.signal);
  acLive.abort();
  const rLive = await pending;
  ok(seenSignal === acLive.signal && rLive.aborted === true && rLive.is_error === true, 'executeLocalTool forwards AbortSignal to fetch and returns aborted=true on cancel');
  console.log(fails ? `${fails} FAILED` : 'all passed');
  process.exit(fails ? 1 : 0);
})();
