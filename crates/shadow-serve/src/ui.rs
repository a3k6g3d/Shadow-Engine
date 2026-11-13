use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::{Response, StatusCode};

type Body = BoxBody<Bytes, std::convert::Infallible>;

pub fn index_response() -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(Full::new(Bytes::from_static(INDEX_HTML.as_bytes())).boxed())
        .unwrap()
}

pub fn monitor_response() -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(Full::new(Bytes::from_static(MONITOR_HTML.as_bytes())).boxed())
        .unwrap()
}

const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en" class="dark h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Shadow</title>
  <style>
    :root {
      --bg: #090a0f;
      --surface: #12141c;
      --surface-hover: #1a1e2b;
      --surface-elevated: #161924;
      --border: #232838;
      --border-subtle: #1c202d;
      --text: #e2e8f0;
      --text-muted: #8492a6;
      --text-dim: #5a6478;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-light: rgba(99, 102, 241, 0.15);
      --accent-glow: rgba(99, 102, 241, 0.35);
      --think-bg: rgba(139, 92, 246, 0.08);
      --think-border: rgba(139, 92, 246, 0.25);
      --think-text: #c4b5fd;
      --tool-bg: rgba(16, 185, 129, 0.08);
      --tool-border: rgba(16, 185, 129, 0.28);
      --tool-text: #6ee7b7;
      --code-bg: #0d0f17;
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.12);
      --warning: #f59e0b;
      --danger: #ef4444;
      --cyan: #06b6d4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    /* quiet scroll bars: thin, no arrows, a track that disappears into the page, a thumb a shade above the borders that brightens on hover
       (only the ::-webkit-scrollbar rules: a `scrollbar-color` / `scrollbar-width` would switch them off in Chromium) */
    html { color-scheme: dark; }
    *::-webkit-scrollbar { width: 10px; height: 10px; }
    *::-webkit-scrollbar-track, *::-webkit-scrollbar-corner { background: transparent; }
    *::-webkit-scrollbar-button { display: none; width: 0; height: 0; }
    *::-webkit-scrollbar-thumb { background: rgba(132, 146, 166, 0.20); border-radius: 8px; border: 3px solid transparent; background-clip: content-box; min-height: 36px; }
    *::-webkit-scrollbar-thumb:hover { background: rgba(132, 146, 166, 0.42); background-clip: content-box; }
    *::-webkit-scrollbar-thumb:active { background: rgba(165, 180, 252, 0.55); background-clip: content-box; }
    html, body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    header {
      background: rgba(18, 20, 28, 0.88);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 0.65rem 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      z-index: 10;
    }
    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible, .chat-item:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .header-left {
      display: flex;
      align-items: center;
      gap: 1.5rem;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
      cursor: pointer;
    }
    .brand svg {
      width: 24px;
      height: 24px;
      fill: var(--accent);
    }
    .nav-tabs {
      display: flex;
      align-items: center;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 2px;
      gap: 2px;
    }
    .nav-tab {
      padding: 0.35rem 0.85rem;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      border: none;
      background: transparent;
      transition: all 0.15s ease;
    }
    .nav-tab:hover {
      color: var(--text);
      background: var(--surface-hover);
    }
    .nav-tab.active {
      color: #fff;
      background: var(--accent);
    }
    .stats-bar {
      display: flex;
      align-items: center;
      gap: 0.65rem;
      font-size: 0.8rem;
      font-variant-numeric: tabular-nums;
    }
    .pill {
      background: var(--surface-hover);
      border: 1px solid var(--border);
      border-radius: 9999px;
      padding: 0.25rem 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.35rem;
      color: var(--text-muted);
      font-variant-numeric: tabular-nums;
    }
    .pill strong {
      color: var(--text);
    }
    .pill.ready { border-color: rgba(16, 185, 129, 0.4); color: var(--success); }
    .pill.busy { border-color: rgba(245, 158, 11, 0.4); color: var(--warning); }
    .pill.loading { border-color: rgba(99, 102, 241, 0.45); color: #a5b4fc; }
    .pill.offline { border-color: rgba(239, 68, 68, 0.45); color: var(--danger); }
    .welcome { display: flex; flex-direction: column; gap: 14px; }
    .welcome h2 { margin: 0; font-size: 1.05rem; font-weight: 600; }
    .welcome p { margin: 0; opacity: 0.8; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); color: inherit; border-radius: 999px;
            padding: 6px 12px; font: inherit; font-size: 0.85rem; cursor: pointer; transition: border-color 0.15s ease, background 0.15s ease, transform 0.1s ease; }
    .chip:hover { border-color: rgba(99, 102, 241, 0.6); background: rgba(99, 102, 241, 0.12); transform: translateY(-1px); }
    .kbd-hints { display: flex; flex-wrap: wrap; gap: 0.75rem; font-size: 0.74rem; color: var(--text-dim); margin-top: 0.15rem; }
    .kbd-hints kbd { background: rgba(255,255,255,0.06); border: 1px solid var(--border); border-radius: 4px; padding: 0.08rem 0.38rem; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.7rem; color: var(--text-muted); }
    .msg-error { color: var(--danger); }
    .attach-btn { background: var(--surface-hover); border: 1px solid var(--border); color: var(--text-muted); border-radius: 0.6rem;
      padding: 0 0.8rem; cursor: pointer; font-size: 1.05rem; }
    .attach-btn:hover { color: var(--text); border-color: var(--accent, #6366f1); }
    #attach-tray { max-width: 880px; width: 100%; margin: 0 auto 0.5rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
    #attach-tray:empty { display: none; }
    .att-chip { display: flex; align-items: center; gap: 0.4rem; background: var(--surface-hover); border: 1px solid var(--border);
      border-radius: 9999px; padding: 0.2rem 0.4rem 0.2rem 0.7rem; font-size: 0.78rem; color: var(--text); }
    .att-chip small { color: var(--text-muted); }
    .att-chip button { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 0.9rem; }
    .att-chip button:hover { color: var(--danger); }
    .bubble .att-note { margin-top: 0.4rem; font-size: 0.78rem; color: var(--text-muted); }
    body.dragging #input-pane { outline: 2px dashed var(--accent, #6366f1); outline-offset: -4px; }
    .msg-meta { margin-top: 0.6rem; font-size: 0.74rem; color: var(--text-muted); opacity: 0.8; font-variant-numeric: tabular-nums; }
    .pill.action { cursor: pointer; }
    .pill.action:hover { color: var(--text); border-color: var(--accent, #6366f1); }
    .bubble ul, .bubble ol { margin: 4px 0 4px 1.3em; padding: 0; }
    .bubble li { margin: 2px 0; }
    .md-gap { height: 0.6em; }
    .md-h { font-weight: 600; margin: 6px 0 2px; }
    .md-h1 { font-size: 1.15rem; } .md-h2 { font-size: 1.05rem; }
    .reading-note { opacity: 0.65; font-size: 0.88rem; margin-right: 6px; font-variant-numeric: tabular-nums; }
    .msg-error small { display: block; opacity: 0.7; margin-top: 4px; }
    #send-btn:disabled { opacity: 0.45; cursor: not-allowed; }

    .main-wrap {
      flex: 1;
      display: flex;
      min-height: 0;
      overflow: hidden;
      position: relative;
      width: 100%;
      height: 100%;
    }

    /* Chat history sidebar */
    #chat-sidebar { display: none; flex-direction: column; width: 270px; flex: none; border-right: 1px solid var(--border); background: var(--surface); min-height: 0; }
    body.view-chat #chat-sidebar { display: flex; }
    body.side-closed #chat-sidebar { display: none; }
    .side-head { display: flex; gap: 0.5rem; padding: 0.75rem; }
    .side-new { flex: 1; background: var(--accent); color: #fff; border: none; border-radius: 0.6rem; padding: 0.55rem 0.8rem; font-weight: 600; cursor: pointer; transition: background 0.15s ease; }
    .side-new:hover { background: var(--accent-hover); }
    .side-search { margin: 0 0.75rem 0.5rem; padding: 0.45rem 0.7rem; background: var(--bg); border: 1px solid var(--border); border-radius: 0.5rem; color: var(--text); font-size: 0.85rem; }
    #chat-list { flex: 1; overflow-y: auto; padding: 0 0.5rem 0.75rem; display: flex; flex-direction: column; gap: 2px; }
    .side-group { font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-dim); padding: 0.7rem 0.5rem 0.25rem; }
    .chat-item { display: flex; align-items: center; gap: 0.4rem; padding: 0.5rem 0.6rem; border-radius: 0.5rem; cursor: pointer; color: var(--text-muted); font-size: 0.85rem; min-inline-size: 0; outline-offset: -2px; }
    .chat-item:hover { background: var(--surface-hover); color: var(--text); }
    .chat-item.active { background: var(--accent-light); color: var(--text); }
    .chat-item .t { flex: 1; min-inline-size: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .chat-item .x { opacity: 0; background: none; border: none; color: var(--text-dim); cursor: pointer; }
    .chat-item:hover .x { opacity: 1; }
    .chat-item .x:hover { color: var(--danger); }
    .side-empty { padding: 1rem 0.75rem; color: var(--text-dim); font-size: 0.82rem; }
    .side-toggle { background: var(--surface-hover); border: 1px solid var(--border); color: var(--text-muted); border-radius: 0.5rem; padding: 0.25rem 0.55rem; cursor: pointer; }
    .side-toggle:hover { color: var(--text); }

    /* Queued messages */
    #queue-bar { max-width: 880px; width: 100%; margin: 0 auto 0.5rem; display: flex; flex-direction: column; gap: 0.3rem; }
    #queue-bar:empty { display: none; }
    #notes-box { max-width: 880px; width: 100%; margin: 0 auto 0.5rem; display: none; }
    #notes-box.open { display: block; }
    #notes-text { field-sizing: content; width: 100%; height: auto; box-sizing: border-box; min-block-size: 5rem; max-block-size: 14rem; min-height: 5rem; max-height: 14rem; resize: vertical; background: var(--surface-hover); color: var(--text); border: 1px solid var(--border); border-radius: 0.6rem; padding: 0.5rem 0.7rem; font: inherit; font-size: 0.82rem; }
    #notes-hint { font-size: 0.74rem; color: var(--text-muted); margin: 0.15rem 0.2rem 0; font-variant-numeric: tabular-nums; }
    .q-item { display: flex; align-items: center; gap: 0.6rem; background: var(--surface-hover); border: 1px dashed var(--border); border-radius: 0.6rem; padding: 0.35rem 0.7rem; font-size: 0.82rem; color: var(--text-muted); min-inline-size: 0; }
    .q-item .n { color: var(--accent); font-weight: 700; font-variant-numeric: tabular-nums; }
    .q-item .t { flex: 1; min-inline-size: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
    .q-item button { background: none; border: none; color: var(--text-muted); cursor: pointer; }
    .q-item button:hover { color: var(--danger); }
    .q-note { font-size: 0.78rem; color: var(--text-dim); }
    .q-note button { background: var(--accent); color: #fff; border: none; border-radius: 0.4rem; padding: 0.1rem 0.5rem; cursor: pointer; margin-left: 0.4rem; }

    /* Settings */
    #settings-panel { overflow-y: auto; padding: 1.25rem 1.5rem 2rem; }
    .set-wrap { max-width: 840px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 1rem; }
    .set-wrap h1 { font-size: 1.25rem; margin: 0; }
    .set-wrap textarea { width: 100%; flex: none; background: var(--bg); border: 1px solid var(--border); border-radius: 0.6rem; padding: 0.7rem 0.85rem; color: var(--text); font: inherit; font-size: 0.88rem; line-height: 1.45; min-height: 150px; resize: vertical; }
    .set-wrap textarea:focus, .set-wrap input[type=text]:focus { border-color: var(--accent); outline: none; }
    .set-row { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    .set-row input[type=text] { flex: 1; min-width: 200px; background: var(--bg); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.5rem 0.7rem; color: var(--text); font: inherit; font-size: 0.88rem; }
    .set-row select { field-sizing: content; max-inline-size: 24ch; background: var(--bg); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.5rem 0.65rem; color: var(--text); font: inherit; font-size: 0.85rem; }
    .set-help { margin: 0; font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; }
    .set-toggle { display: flex; gap: 0.5rem; align-items: center; font-size: 0.88rem; cursor: pointer; }
    #set-mem-list { display: flex; flex-direction: column; gap: 0.3rem; }
    .mem-item { display: flex; align-items: center; gap: 0.6rem; background: var(--bg); border: 1px solid var(--border-subtle); border-radius: 0.5rem; padding: 0.4rem 0.6rem; font-size: 0.86rem; min-inline-size: 0; }
    .mem-item .badge { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent); background: var(--accent-light); border-radius: 9999px; padding: 0.1rem 0.5rem; flex: none; }
    .mem-item .mem-text { flex: 1; min-inline-size: 0; cursor: text; color: var(--text); }
    .mem-item .mem-edit { flex: 1; min-inline-size: 0; background: var(--surface); border: 1px solid var(--accent); border-radius: 0.4rem; padding: 0.2rem 0.4rem; color: var(--text); font: inherit; }
    .mem-item button { background: none; border: none; color: var(--text-dim); cursor: pointer; }
    .mem-item button:hover { color: var(--danger); }
    .sug { display: flex; gap: 0.5rem; align-items: flex-start; font-size: 0.86rem; padding: 0.3rem 0; }
    .sug input { margin-top: 0.2rem; }

    /* Browser pane */
    #browser-pane { display: none; flex-direction: column; width: 46%; min-width: 320px; flex: none; border-left: 1px solid var(--border); background: var(--surface); min-height: 0; }
    body.view-chat.browser-open #browser-pane { display: flex; }
    .bp-head { display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 0.75rem; border-bottom: 1px solid var(--border-subtle); font-size: 0.82rem; }
    .bp-url { flex: 1; min-inline-size: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; background: var(--bg); border: 1px solid var(--border); border-radius: 9999px; padding: 0.25rem 0.8rem; color: var(--text-muted); }
    .bp-body { flex: 1; overflow: auto; padding: 0.6rem; display: flex; align-items: flex-start; justify-content: center; }
    .bp-body img { display: none; max-width: 100%; border-radius: 0.5rem; border: 1px solid var(--border); }
    .bp-status { padding: 0.4rem 0.75rem; font-size: 0.75rem; color: var(--text-dim); border-top: 1px solid var(--border-subtle); font-variant-numeric: tabular-nums; }

    /* Dashboard */
    #dash-panel { overflow-y: auto; padding: 1.25rem 1.5rem 2rem; }
    .dash-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem; }
    .dash-head h1 { font-size: 1.25rem; margin: 0; }
    .dash-head p { margin: 0.2rem 0 0; color: var(--text-muted); font-size: 0.85rem; }
    .dash-actions { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .dash-btn { background: var(--surface-hover); border: 1px solid var(--border); color: var(--text); border-radius: 0.6rem; padding: 0.5rem 0.85rem; cursor: pointer; font-size: 0.85rem; transition: border-color 0.15s ease, background 0.15s ease; }
    .dash-btn:hover { border-color: var(--accent); }
    .dash-btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .dash-btn.primary:hover { background: var(--accent-hover); }
    .dash-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
    .dcard { background: var(--surface); border: 1px solid var(--border); border-radius: 0.9rem; padding: 1rem 1.1rem; display: flex; flex-direction: column; gap: 0.6rem; min-width: 0; transition: border-color 0.15s ease; }
    .dcard:hover { border-color: #2d3448; }
    .dcard.wide { grid-column: 1 / -1; }
    .dcard h3 { margin: 0; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-muted); font-weight: 600; display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; min-inline-size: 0; }
    .dcard h3 .tag { text-transform: none; letter-spacing: 0; font-weight: 500; color: var(--text-dim); font-size: 0.75rem; min-inline-size: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .big { font-size: 1.9rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
    .big small { font-size: 0.85rem; font-weight: 500; color: var(--text-muted); margin-left: 0.3rem; }
    .kv { display: flex; justify-content: space-between; gap: 0.75rem; font-size: 0.84rem; color: var(--text-muted); min-inline-size: 0; }
    .kv > span { min-inline-size: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .kv b { color: var(--text); font-weight: 600; text-align: right; min-inline-size: 0; max-inline-size: 65%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .bar { height: 8px; border-radius: 9999px; background: var(--bg); overflow: hidden; display: flex; }
    .bar > i { display: block; height: 100%; background: var(--accent); transition: width 0.25s ease; }
    .bar > i.b { background: #10b981; }
    .bar > i.c { background: #f59e0b; }
canvas.spark { width: 100%; height: 64px; display: block; }
    /* Dashboard drag & drop: drag a tile by its title bar, drop it on another to swap, snap into the grid; the layout is remembered in this browser */
    #dash-grid .dcard { cursor: grab; }
    #dash-grid .dcard h3 { cursor: grab; }
    #dash-grid .dcard h3 .grip { opacity: 0; margin-right: 0.4rem; color: var(--text-dim); }
    #dash-grid .dcard:hover h3 .grip { opacity: 1; }
    #dash-grid .dcard.dragging { opacity: 0.4; border-style: dashed; cursor: grabbing; }
    #dash-grid .dcard.droptarget { box-shadow: inset 0 0 0 2px var(--accent); }
    #dash-grid .dcard[draggable] { cursor: grabbing; }
    table.recent { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 0.82rem; font-variant-numeric: tabular-nums; }
    table.recent th { text-align: right; color: var(--text-dim); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.3rem 0.5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    table.recent td { text-align: right; padding: 0.4rem 0.5rem; border-top: 1px solid var(--border-subtle); color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    table.recent th:first-child, table.recent td:first-child { text-align: left; }
    table.recent td b { color: var(--text); font-weight: 600; }
    .dash-empty { color: var(--text-dim); font-size: 0.85rem; padding: 0.5rem 0; }
    #side-open { display: none; }
    body.side-closed.view-chat #side-open { display: inline-block; }

    /* Views - Strict Tab Display */
    .view-panel {
      flex: 1;
      display: none !important;
      min-height: 0;
      overflow: hidden;
      flex-direction: column;
      width: 100%;
      height: 100%;
    }
    .view-panel.active {
      display: flex !important;
    }

    /* Chat View */
    #chat-panel {
      width: 100%;
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #messages {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      overflow-anchor: none;
    }
    .msg-row {
      display: flex;
      gap: 1rem;
      max-width: 880px;
      width: 100%;
      margin: 0 auto;
    }
    .msg-row.user {
      flex-direction: row-reverse;
    }
    .avatar {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      font-weight: 600;
      flex-shrink: 0;
    }
    .msg-row.user .avatar {
      background: var(--accent);
      color: white;
    }
    .msg-row.assistant .avatar {
      background: var(--surface-hover);
      border: 1px solid var(--border);
      color: var(--accent);
    }
    .bubble {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.85rem 1.15rem;
      font-size: 0.95rem;
      line-height: 1.6;
      word-break: break-word;
      max-width: calc(100% - 50px);
    }
    .msg-row.user .bubble {
      background: var(--accent-light);
      border-color: rgba(99, 102, 241, 0.3);
    }
    .think-block {
      background: var(--think-bg);
      border: 1px solid var(--think-border);
      border-radius: 0.5rem;
      padding: 0.6rem 0.85rem;
      margin-bottom: 0.75rem;
      font-size: 0.85rem;
      color: var(--think-text);
    }
    .think-header {
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
      cursor: pointer;
    }
    .think-header-left {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .think-spinner {
      width: 12px;
      height: 12px;
      border: 2px solid rgba(196, 181, 253, 0.3);
      border-top-color: var(--think-text);
      border-radius: 50%;
      display: inline-block;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .think-stop-btn {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5;
      border-radius: 6px;
      padding: 0.2rem 0.55rem;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: all 0.15s ease;
    }
    .think-stop-btn:hover {
      background: rgba(239, 68, 68, 0.35);
      color: #ffffff;
      border-color: #ef4444;
    }
    .think-content {
      margin-top: 0.5rem;
      white-space: pre-wrap;
      opacity: 0.9;
      line-height: 1.5;
    }

    /* Tool Call Card */
    .tool-call-block {
      background: var(--tool-bg);
      border: 1px solid var(--tool-border);
      border-radius: 0.5rem;
      padding: 0.6rem 0.85rem;
      margin: 0.65rem 0;
      font-size: 0.85rem;
      color: var(--tool-text);
    }
    .tool-header {
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
    }
    .tool-output {
      margin-top: 0.5rem;
      background: var(--code-bg);
      border: 1px solid var(--border-subtle);
      border-radius: 0.35rem;
      padding: 0.5rem 0.75rem;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.8rem;
      max-height: 250px;
      overflow-y: auto;
      color: #a7f3d0;
    }

    /* Settings > Display: shorten long thinking and tool output (a click on the clipped text shows all of it) */
    body.trunc-on .think-content { max-height: calc(var(--trunc-lines, 8) * 1.5em); overflow: hidden; }
    body.trunc-on .tool-output { max-height: calc(var(--trunc-lines, 8) * 1.4em + 1em); overflow: hidden; }
    body.trunc-on .think-content.clipped, body.trunc-on .tool-output.clipped { cursor: pointer; -webkit-mask-image: linear-gradient(#000 75%, transparent); mask-image: linear-gradient(#000 75%, transparent); }
    body.trunc-on .think-content.expanded, body.trunc-on .tool-output.expanded { max-height: none; -webkit-mask-image: none; mask-image: none; overflow-y: auto; }
    body.trunc-on .tool-output.expanded { max-height: 60vh; }
    .set-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.6rem; }
    .set-grid label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.78rem; color: var(--text-muted); }
    .set-grid input[type=number] { background: var(--bg); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.45rem 0.6rem; color: var(--text); font: inherit; font-size: 0.88rem; min-inline-size: 0; }
    .set-grid input[type=number]:focus { border-color: var(--accent); outline: none; }
    .set-grid.off { opacity: 0.45; pointer-events: none; }
    #regen-btn { display: none; }

    /* Permission mode: the selector's wrapper and the approval box inside a tool card */
    .perm-wrap { position: relative; display: inline-flex; }
    .menu-pill-btn.perm-auto .perm-dot { color: var(--accent); }
    .menu-pill-btn.perm-accept .perm-dot { color: var(--warning); }
    .menu-pill-btn.perm-bypass .perm-dot { color: var(--danger); }
    .perm-box { margin-top: 0.5rem; padding: 0.5rem 0.65rem; border: 1px solid var(--warning); border-radius: 0.4rem; background: rgba(245, 158, 11, 0.08); color: var(--text); font-weight: normal; }
    .perm-box .perm-why { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.35rem; }
    .perm-box .perm-what { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.8rem; white-space: pre-wrap; max-height: 140px; overflow-y: auto; margin-bottom: 0.45rem; }
    .perm-box .perm-btns { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .perm-box button { background: var(--surface-elevated); border: 1px solid var(--border); color: var(--text); border-radius: 0.4rem; padding: 0.25rem 0.7rem; cursor: pointer; font: inherit; font-size: 0.82rem; }
    .perm-box button.allow { border-color: var(--success); }
    .perm-box button.deny { border-color: var(--danger); }
    .perm-box button:hover { background: var(--surface-hover); }

    /* The closing status of an agent task (the finish tool) */
    .finish-card {
      border: 1px solid var(--border);
      border-left: 3px solid var(--success);
      background: var(--surface-elevated);
      border-radius: 0.5rem;
      padding: 0.6rem 0.9rem;
      margin: 0.65rem 0;
      font-size: 0.88rem;
      line-height: 1.5;
    }
    .finish-card.open { border-left-color: var(--warning); }
    .finish-card .fin-h { font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; margin: 0.45rem 0 0.15rem; color: var(--text-muted); }
    .finish-card > .fin-h:first-child, .finish-card > div:first-child > .fin-h { margin-top: 0; }
    .finish-card .fin-done .fin-h { color: var(--success); }
    .finish-card .fin-pending .fin-h { color: var(--warning); }
    .finish-card ul { margin: 0; padding-left: 1.2rem; }
    .finish-card .fin-q { font-weight: 600; margin-top: 0.2rem; }
    .finish-card .fin-opts { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
    .finish-card .fin-opt { background: var(--accent-light); border: 1px solid var(--accent); color: var(--text); border-radius: 0.4rem; padding: 0.25rem 0.65rem; cursor: pointer; font: inherit; font-size: 0.84rem; text-align: left; }
    .finish-card .fin-opt:hover { background: var(--accent-glow); }

    pre {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 0.4rem;
      padding: 0.75rem 1rem;
      margin: 0.5rem 0;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.85rem;
      position: relative;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: rgba(255, 255, 255, 0.05);
      padding: 0.15rem 0.35rem;
      border-radius: 0.25rem;
      font-size: 0.85em;
    }
    pre code {
      background: none;
      padding: 0;
    }
    .copy-btn {
      position: absolute;
      top: 0.4rem;
      right: 0.4rem;
      background: var(--surface-hover);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 0.25rem;
      padding: 0.2rem 0.45rem;
      font-size: 0.7rem;
      cursor: pointer;
      opacity: 0.8;
    }
    .copy-btn:hover { opacity: 1; color: var(--text); }
    .bubble { position: relative; }
    .bubble > .copy-btn.bubble-copy { opacity: 0; }
    .msg-row:hover .bubble > .copy-btn.bubble-copy { opacity: 0.8; }
    .msg-row .bubble > .copy-btn.bubble-copy:hover { opacity: 1; color: var(--text); }

    /* Input Area */
    #input-pane {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 0.9rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      flex-shrink: 0;
    }
    .input-box {
      max-width: 880px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      gap: 0.6rem;
      position: relative;
    }
    textarea {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 0.6rem;
      padding: 0.75rem 1rem;
      color: var(--text);
      font-size: 0.95rem;
      resize: none;
      outline: none;
      min-height: 48px;
      max-height: 200px;
      line-height: 1.4;
      font-family: inherit;
      transition: border-color 0.15s;
    }
    textarea:focus { border-color: var(--accent); }
    .send-btn {
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 0.6rem;
      padding: 0 1.25rem;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: background 0.15s;
      display: flex;
      align-items: center;
      justify-content: center;
      align-self: flex-end;
      height: 48px;
    }
    .send-btn:hover { background: var(--accent-hover); }
    .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .stop-btn {
      background: #ef4444;
      color: white;
      border: none;
      border-radius: 0.6rem;
      padding: 0 1.25rem;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.15s ease;
      display: none;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      align-self: flex-end;
      height: 48px;
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }
    .stop-btn:hover {
      background: #dc2626;
      box-shadow: 0 0 16px rgba(239, 68, 68, 0.6);
    }

    /* Bottom Control Bar */
    .controls-bar {
      max-width: 880px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.82rem;
      color: var(--text-muted);
      position: relative;
    }
    .controls-left {
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }

    /* Claude & Antigravity Style Menu Pill */
    .menu-pill-btn {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 9999px;
      padding: 0.35rem 0.75rem;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      transition: all 0.15s ease;
      user-select: none;
    }
    .menu-pill-btn:hover {
      background: var(--surface-hover);
      border-color: var(--accent);
    }
    .menu-pill-btn.active-state {
      border-color: var(--success);
      color: var(--success);
    }
    .menu-pill-btn svg.caret {
      width: 12px;
      height: 12px;
      fill: var(--text-muted);
      transition: transform 0.2s;
    }
    .menu-pill-btn.open svg.caret {
      transform: rotate(180deg);
    }

    /* Floating Popup Menu */
    .dropdown-popup {
      position: absolute;
      bottom: calc(100% + 10px);
      left: 0;
      width: 320px;
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255, 255, 255, 0.05);
      padding: 0.45rem;
      display: none;
      flex-direction: column;
      gap: 0.25rem;
      z-index: 100;
      backdrop-filter: blur(8px);
    }
    .dropdown-popup.open {
      display: flex;
      animation: popupSlide 0.18s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes popupSlide {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .dropdown-popup-header {
      padding: 0.4rem 0.6rem 0.25rem;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
    }
    .dropdown-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.55rem 0.75rem;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.12s;
      user-select: none;
    }
    .dropdown-item:hover {
      background: var(--surface-hover);
    }
    .dropdown-item.active {
      background: rgba(99, 102, 241, 0.12);
    }
    .item-left {
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }
    .item-icon {
      width: 28px;
      height: 28px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--bg);
      border: 1px solid var(--border-subtle);
      font-size: 0.85rem;
      flex-shrink: 0;
    }
    .item-info {
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }
    .item-title {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text);
    }
    .dropdown-item.active .item-title {
      color: #818cf8;
    }
    .item-desc {
      font-size: 0.72rem;
      color: var(--text-muted);
      line-height: 1.25;
    }
    .item-check {
      font-size: 0.9rem;
      color: var(--accent);
      font-weight: bold;
      opacity: 0;
    }
    .dropdown-item.active .item-check {
      opacity: 1;
    }

    /* System Persona Modal */
    #persona-modal {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 200;
    }
    #persona-modal.open { display: flex; }
    .modal-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      width: 620px;
      max-width: 90vw;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8);
    }
    .modal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .modal-header h3 {
      font-size: 1.1rem;
      font-weight: 700;
    }
    .modal-close {
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 1.2rem;
      cursor: pointer;
    }
    .modal-textarea {
      width: 100%;
      height: 180px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
      color: var(--text);
      font-size: 0.85rem;
      font-family: inherit;
      resize: vertical;
      line-height: 1.45;
    }
    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 0.6rem;
    }
    .modal-btn {
      padding: 0.45rem 1rem;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border);
      background: var(--surface-hover);
      color: var(--text);
    }
    .modal-btn.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }

    /* MoE Memory Hierarchy Diagram */
    .tier-diagram-wrap {
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }
    .tier-diagram-title {
      font-size: 0.95rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .tier-flow {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
      position: relative;
    }
    @media (max-width: 800px) {
      .tier-flow { grid-template-columns: 1fr; }
    }
    .tier-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .tier-box.active-tier {
      border-color: var(--accent);
      box-shadow: 0 0 16px var(--accent-light);
    }
    .tier-header {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .tier-header span.tag {
      font-size: 0.7rem;
      padding: 0.1rem 0.4rem;
      border-radius: 4px;
      background: var(--surface-hover);
      color: var(--cyan);
    }

    .cursor-blink {
      display: inline-block;
      width: 6px;
      height: 14px;
      background: var(--accent);
      margin-left: 2px;
      animation: blink 0.8s infinite;
      vertical-align: middle;
    }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    /* Compact chat bar: one rounded bar (attach, text, send / stop inside it) and a slim status line under it */
    #input-pane { background: transparent; border-top: none; padding: 0.35rem 1.25rem 0.45rem; gap: 0.25rem; position: relative; }
    .input-box { align-items: flex-end; gap: 0.3rem; background: var(--surface); border: 1px solid var(--border); border-radius: 1.2rem; padding: 0.28rem 0.35rem; transition: border-color 0.15s, box-shadow 0.15s; box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.45); }
    .input-box:focus-within { border-color: var(--accent); box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(99, 102, 241, 0.25); }
    .input-box textarea { field-sizing: content; width: 100%; height: auto; min-block-size: 30px; max-block-size: 200px; background: transparent; border: none; padding: 0.4rem 0.4rem; min-height: 30px; max-height: 200px; font-size: 0.92rem; line-height: 1.35; }
    .input-box textarea:focus { border: none; outline: none; }
    .attach-btn { width: 32px; height: 32px; padding: 0; border-radius: 50%; background: transparent; border: none; font-size: 1rem; align-self: flex-end; }
    .send-btn, .stop-btn { height: 32px; padding: 0 0.95rem; border-radius: 9999px; font-size: 0.8rem; }
    .stop-btn { box-shadow: none; }
    .controls-bar { font-size: 0.72rem; }
    .controls-left { gap: 0.15rem; }
    #timing-info { font-variant-numeric: tabular-nums; }
    #scroll-bottom-btn {
      position: absolute;
      top: -2.4rem;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(22, 25, 36, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 9999px;
      padding: 0.28rem 0.85rem;
      font-size: 0.76rem;
      font-weight: 600;
      cursor: pointer;
      display: none;
      align-items: center;
      gap: 0.35rem;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
      z-index: 20;
      transition: border-color 0.15s ease, background 0.15s ease;
    }
    #scroll-bottom-btn.visible { display: inline-flex; }
    #scroll-bottom-btn:hover { border-color: var(--accent); background: var(--surface-hover); }
    .menu-pill-btn { background: transparent; border-color: transparent; padding: 0.15rem 0.55rem; font-size: 0.74rem; color: var(--text-muted); gap: 0.35rem; }
    .menu-pill-btn:hover { background: var(--surface-hover); color: var(--text); border-color: transparent; }
    .menu-pill-btn.active-state { border-color: transparent; }
    #queue-bar, #attach-tray { margin-bottom: 0.15rem; }
    .retry-btn { margin-left: 0.6rem; background: var(--accent); color: #fff; border: none; border-radius: 0.4rem; padding: 0.15rem 0.7rem; cursor: pointer; font-size: 0.8rem; }
    .retry-btn:hover { background: var(--accent-hover); }
    /* a reply (or a tool) in progress throbs: it has not hung; a stopped engine turns it red */
    @keyframes throb {
      0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); border-color: var(--border); }
      50% { box-shadow: 0 0 16px 2px rgba(99, 102, 241, 0.5); border-color: var(--accent); }
    }
    .msg-row.assistant.working > .bubble, .tool-call-block.running { animation: throb 1.8s ease-in-out infinite; }
    .msg-row.assistant > .bubble { border: 1px solid var(--border); }
    .msg-row.assistant.stalled > .bubble { animation: none; border-color: var(--danger); }
    /* the menus open above the whole bar, whatever its height (--bar-h follows the text box) */
    .dropdown-popup { bottom: calc(100% + 10px + var(--bar-h, 44px)); }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <div class="brand" id="brand-logo">
        <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
        <span>Shadow</span>
      </div>
      <button class="side-toggle" id="side-open" title="Show the conversation list" onclick="toggleSidebar()">☰</button>
      <div class="nav-tabs">
        <button class="nav-tab active" id="tab-chat" onclick="switchView('chat')">
          <span>💬</span> Chat
        </button>
        <button class="nav-tab" id="tab-dash" onclick="switchView('dash')">
          <span>🏠</span> Dashboard
        </button>
        <button class="nav-tab" id="tab-settings" onclick="switchView('settings')">
          <span>⚙</span> Settings
        </button>
      </div>
    </div>
    <div class="stats-bar">
      <button class="pill action" id="newchat-btn" onclick="newChat()" title="Start a new conversation (Ctrl+Shift+O)">＋ New chat</button>
      <span class="pill ready" id="status-pill">Ready</span>
      <span class="pill" id="speed-pill">Speed: <strong>-- tok/s</strong></span>
      <span class="pill" id="gpu-header-pill">GPU: <strong>--% • VRAM: -- GiB</strong></span>
    </div>
  </header>

  <div class="main-wrap">
    <aside id="chat-sidebar">
      <div class="side-head">
        <button class="side-new" onclick="newChat()" title="Start a new conversation (Ctrl+Shift+O)">＋ New chat</button>
        <button class="side-toggle" title="Hide the conversation list" onclick="toggleSidebar()">«</button>
      </div>
      <input class="side-search" id="chat-search" aria-label="Search conversations" placeholder="Search conversations (Ctrl+K)" oninput="renderChatList()">
      <div id="chat-list"></div>
    </aside>

    <!-- VIEW: DASHBOARD -->
    <div class="view-panel" id="dash-panel">
      <div class="dash-head">
        <div>
          <h1>Dashboard</h1>
          <p id="dash-sub">Shadow running locally on this PC.</p>
        </div>
        <div class="dash-actions">
          <button class="dash-btn primary" onclick="newChat(); switchView('chat')">＋ New chat</button>
          <button class="dash-btn" id="dash-last" onclick="openLastChat()">Resume last chat</button>
          <button class="dash-btn" onclick="switchView('settings')">Settings</button>
<button class="dash-btn" id="dash-tools" onclick="toggleTools(); dashRefresh()">Windows tools</button>
          <button class="dash-btn" id="dash-reset" style="display:none" onclick="dashResetLayout()">Reset layout</button>
        </div>
      </div>
<div class="dash-grid" id="dash-grid">
        <div class="dcard" data-tile="tiers">
          <h3><span class="grip">&#9776;</span>Expert tiers <span class="tag">where the model's experts live</span></h3>
          <div class="big" id="d-hit" title="share of expert lookups answered from the VRAM cache">--<small>% hit</small></div>
          <canvas class="spark" id="c-hit"></canvas>
          <div class="bar"><i id="d-hitbar" style="width:0"></i></div>
          <div class="kv"><span>VRAM cache hits / misses</span><b id="d-hitmiss">--</b></div>
          <div class="kv"><span>Expert cache</span><b id="d-slots">--</b></div>
          <div class="kv"><span>RAM tier (pinned)</span><b id="d-pin">--</b></div>
          <div class="kv"><span>Misses served from RAM / drive</span><b id="d-served">--</b></div>
          <div class="bar"><i class="b" id="d-ramserved" style="width:0"></i><i class="c" id="d-fileserved" style="width:0"></i></div>
          <div class="kv"><span>Misses read from RAM / drive</span><b id="d-servedn">--</b></div>
          <div class="kv"><span>Reply latency p50 / p95</span><b id="d-p50p95">--</b></div>
        </div>
        <div class="dcard" data-tile="speed">
          <h3><span class="grip">&#9776;</span>Decode speed <span class="tag">last 40 replies</span></h3>
          <div class="big" id="d-tps" title="the last reply answered">--<small>tok/s</small></div>
          <canvas class="spark" id="c-tps"></canvas>
          <div class="kv"><span>Average</span><b id="d-tps-avg">--</b></div>
          <div class="kv"><span>Prompt reading (last)</span><b id="d-pps">--</b></div>
        </div>
        <div class="dcard" data-tile="spec">
          <h3><span class="grip">&#9776;</span>Speculation <span class="tag">draft &amp; verify</span></h3>
          <div class="big" id="d-acc" title="share of drafted tokens the model accepted">--<small>% accepted</small></div>
          <canvas class="spark" id="c-acc"></canvas>
          <div class="kv"><span>Drafts made / accepted</span><b id="d-specnums">--</b></div>
          <div class="kv"><span>Verify passes run</span><b id="d-specpasses">--</b></div>
          <div class="kv"><span>Tokens per pass</span><b id="d-specperpass">--</b></div>
          <div class="kv"><span>Acceptance by position</span><b id="d-specpos">--</b></div>
          <canvas class="spark" id="d-specposbars" title="acceptance rate at draft position 1..8"></canvas>
        </div>
        <div class="dcard" data-tile="engine">
          <h3><span class="grip">&#9776;</span>Engine <span class="tag" id="d-state">--</span></h3>
          <div class="big" id="d-model" style="font-size:1.05rem;word-break:break-all">--</div>
          <div class="kv"><span>Uptime</span><b id="d-uptime">--</b></div>
          <div class="kv"><span>Process ID</span><b id="d-pid">--</b></div>
          <div class="kv"><span>Speculative decoding</span><b id="d-spec">--</b></div>
          <div class="kv"><span>Requests answered</span><b id="d-reqs">--</b></div>
          <div class="kv"><span>Context in use</span><b id="d-ctx">--</b></div>
          <div class="bar"><i id="d-ctxbar" style="width:0"></i></div>
        </div>
        <div class="dcard" data-tile="gpu">
          <h3><span class="grip">&#9776;</span>GPU <span class="tag" id="d-gpuname">--</span></h3>
          <div class="big" id="d-gpu">--<small>% busy</small></div>
          <canvas class="spark" id="c-gpu"></canvas>
          <div class="kv"><span>VRAM</span><b id="d-vram">--</b></div>
          <div class="bar"><i id="d-vrambar" style="width:0"></i></div>
          <div class="kv"><span>Temperature</span><b id="d-temp">--</b></div>
          <div class="kv"><span>Memory controller busy</span><b id="d-memctl">--</b></div>
          <div class="kv"><span>Free VRAM</span><b id="d-vramfree">--</b></div>
        </div>
        <div class="dcard" data-tile="cpu">
          <h3><span class="grip">&#9776;</span>CPU <span class="tag" id="d-cputag">--</span></h3>
          <div class="big" id="d-cpusys">--<small>% busy (whole PC)</small></div>
          <canvas class="spark" id="c-cpu"></canvas>
          <div class="kv"><span><i style="color:#a78bfa">&#9632;</i> Shadow</span><b id="d-cpuproc">--</b></div>
          <div class="kv"><span><i style="color:#38bdf8">&#9632;</i> Whole PC</span><b id="d-cpusys2">--</b></div>
          <div class="kv"><span>Shadow's CPU threads in use</span><b id="d-cpucores">--</b></div>
          <div class="kv"><span>Clock</span><b id="d-cpuclk">--</b></div>
          <div class="kv"><span>Cache</span><b id="d-cpucache">--</b></div>
          <div class="kv"><span>Temperature</span><b id="d-cputemp">--</b></div>
        </div>
        <div class="dcard" data-tile="ram">
          <h3><span class="grip">&#9776;</span>System memory <span class="tag" id="d-ramtag">--</span></h3>
          <div class="big" id="d-ramfree">--<small>GiB free</small></div>
          <canvas class="spark" id="c-ram"></canvas>
          <div class="kv"><span>Shadow's working set</span><b id="d-ws">--</b></div>
          <div class="kv"><span>RAM in use (whole PC)</span><b id="d-ramused">--</b></div>
        </div>
        <div class="dcard" data-tile="chats">
          <h3><span class="grip">&#9776;</span>Conversations <span class="tag" id="d-chatcount">--</span></h3>
          <div id="d-chats" class="dash-empty">No saved conversations yet.</div>
        </div>
        <div class="dcard" data-tile="tools">
          <h3><span class="grip">&#9776;</span>Tools <span class="tag" id="d-toolcount">--</span></h3>
          <div class="kv"><span>Built in</span><b>read, write, edit, glob, grep, cmd, web search, web fetch</b></div>
          <div id="d-mcp" class="dash-empty">No MCP servers.</div>
        </div>
        <div class="dcard wide" data-tile="recent">
          <h3><span class="grip">&#9776;</span>Recent requests <span class="tag">newest first</span></h3>
          <div id="d-recent" class="dash-empty">Nothing answered yet.</div>
        </div>
      </div>
    </div>

    <!-- VIEW: SETTINGS -->
    <div class="view-panel" id="settings-panel">
      <div class="set-wrap">
        <h1>Settings</h1>
        <div class="dcard">
          <h3>Default persona <span class="tag">how Shadow behaves in every new chat</span></h3>
          <textarea id="set-persona" aria-label="Default persona" spellcheck="false"></textarea>
          <div class="set-row">
            <button class="dash-btn primary" onclick="savePersonaSetting()">Save</button>
            <button class="dash-btn" onclick="resetPersonaSetting()">Reset to the built-in default</button>
            <span class="tag" id="set-persona-note"></span>
          </div>
          <p class="set-help">This is the starting point of every chat: who Shadow is and how it talks. How its tools and its small context window work is added after it by the app (and updated with it), so you do not need to keep that here. For a one-off, just say it in the chat: "act as a software engineer and design this", "review this like a security auditor", "be brief". The persona itself changes only here.</p>
        </div>
        <div class="dcard">
          <h3>Sampling <span class="tag" id="set-samp-note"></span></h3>
          <div class="set-row">
            <select id="set-samp-preset" aria-label="Sampling preset" style="max-inline-size:100%" onchange="setSamplingPreset(this.value)">
              <option value="focused">Focused (temperature 0.6, top-k 20, top-p 0.95): code and tools, the default</option>
              <option value="model">Model default (temperature 1.0, top-k 20, top-p 0.95): more varied, fewer exact answers</option>
              <option value="greedy">Greedy (always the likeliest token, repeats itself more)</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          <div class="set-grid" id="set-samp-grid">
            <label>Temperature<input type="number" id="set-samp-temperature" min="0" max="2" step="0.05" oninput="samplingFieldChanged()"></label>
            <label>Top-k (0 = all)<input type="number" id="set-samp-top_k" min="0" max="1000" step="1" oninput="samplingFieldChanged()"></label>
            <label>Top-p<input type="number" id="set-samp-top_p" min="0" max="1" step="0.01" oninput="samplingFieldChanged()"></label>
            <label>Min-p<input type="number" id="set-samp-min_p" min="0" max="1" step="0.01" oninput="samplingFieldChanged()"></label>
            <label>Presence penalty<input type="number" id="set-samp-presence_penalty" min="-2" max="2" step="0.1" oninput="samplingFieldChanged()"></label>
            <label>Repeat penalty<input type="number" id="set-samp-repeat_penalty" min="1" max="2" step="0.05" oninput="samplingFieldChanged()"></label>
            <label>Seed (empty = random)<input type="number" id="set-samp-seed" min="0" step="1" oninput="samplingFieldChanged()"></label>
          </div>
          <p class="set-help">How Shadow picks each word. Greedy always takes the single likeliest token, which on a 3-bit model is what makes it circle back on itself while thinking; sampling picks among the likely ones instead, so a reply can differ each time (Regenerate asks again). The presence and repeat penalties push down tokens the reply has already used. Temperature 0 is greedy. The draft model still speeds sampled replies up, a little less than greedy ones. Applies from your next message.</p>
        </div>
        <div class="dcard">
          <h3>Display</h3>
          <label class="set-toggle"><input type="checkbox" id="set-trunc-on" onchange="setTruncOn(this.checked)"> Shorten long thinking and tool output in the chat</label>
          <div class="set-row"><span class="set-help">Lines to show:</span> <input type="number" id="set-trunc-lines" min="2" max="60" step="1" style="width:5rem" onchange="setTruncLines(this.value)"></div>
          <p class="set-help">Thinking shows its latest lines and tool results their first ones; click the clipped text to open all of it, click again to close it. This is only how the page draws them: Shadow still reads the full text.</p>
        </div>
        <div class="dcard">
          <h3>Task status</h3>
          <label class="set-toggle"><input type="checkbox" id="set-finish-on" onchange="setFinishOn(this.checked)"> End tasks that use tools with a status card</label>
          <p class="set-help">When Shadow has used its tools it closes the task with what it did, what is not done or not verified and, only if it needs you, one question with buttons for the answers. Off: it ends with a plain message as before. Takes effect in a new chat (the instruction is part of the start of the prompt).</p>
        </div>
        <div class="dcard">
          <h3>Memory <span class="tag" id="set-mem-count">--</span></h3>
          <label class="set-toggle"><input type="checkbox" id="set-mem-on" onchange="setMemoryOn(this.checked)"> Use memory in chats</label>
          <p class="set-help">Short notes about you that Shadow reads at the start of every chat. It adds notes itself when you tell it something lasting about you or your work (or say "remember ..."); edit one by clicking it, delete with the cross. Kept on this PC only, in %APPDATA%\Shadow\memory.json.</p>
          <div id="set-mem-list"></div>
          <div class="set-row">
            <input type="text" id="set-mem-new" aria-label="Add a memory note" placeholder="Add a note, e.g. I prefer short answers with the code first" onkeydown="if (event.key === 'Enter' && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); addMemory(); }">
            <select id="set-mem-kind" aria-label="Memory category"><option value="user">about me</option><option value="preference">preference</option><option value="project">project</option><option value="reference">reference</option></select>
            <button class="dash-btn" onclick="addMemory()">Add</button>
          </div>
          <div class="set-row">
            <button class="dash-btn" id="set-learn" onclick="learnFromChats()">Learn from my past chats</button>
            <span class="tag" id="set-learn-note"></span>
          </div>
          <div id="set-suggest"></div>
        </div>
      </div>
    </div>

    <!-- VIEW: CHAT -->
    <div class="view-panel active" id="chat-panel">
      <div id="messages">
        <div class="msg-row assistant">
          <div class="avatar">S</div>
          <div class="bubble welcome" id="welcome">
            <h2>Shadow is running locally on this PC.</h2>
            <p>Ask anything. With Windows Tools on, it can read and search files, edit them and run commands; it asks the model, never the cloud.</p>
            <div class="chips">
              <button class="chip" onclick="useSuggestion(this)">Explain what a hash table is, briefly</button>
              <button class="chip" onclick="useSuggestion(this)">List the files in J:\Shadow\docs</button>
              <button class="chip" onclick="useSuggestion(this)">Write a Rust function that parses comma-separated integers</button>
            </div>
            <div class="kbd-hints">
              <span><kbd>Enter</kbd> Send</span>
              <span><kbd>Shift+Enter</kbd> New line</span>
              <span><kbd>Ctrl+K</kbd> Search chats</span>
              <span><kbd>Ctrl+Shift+O</kbd> New chat</span>
              <span><kbd>Esc</kbd> Stop</span>
            </div>
          </div>
        </div>
      </div>

      <div id="input-pane">
        <button id="scroll-bottom-btn" onclick="scrollToBottom(true)" title="Scroll to bottom">↓ Jump to latest</button>
        <div id="notes-box">
          <textarea id="notes-text" aria-label="Pinned notes for this chat" placeholder="Pinned notes for this chat: the plan, decisions, key facts. Shadow sees them at the end of every request and can change them with its notes tool."></textarea>
          <div id="notes-hint"></div>
        </div>
        <div id="queue-bar"></div>
        <div id="attach-tray"></div>
        <div class="input-box">
          <input type="file" id="file-input" multiple hidden>
          <button class="attach-btn" id="attach-btn" title="Attach text or code files (or drop / paste them)" onclick="document.getElementById('file-input').click()">📎</button>
          <textarea id="prompt-input" aria-label="Message Shadow" rows="1" placeholder="Message Shadow (Enter to send, Shift+Enter for a new line)"></textarea>
          <button id="send-btn" class="send-btn">Send</button>
          <button id="stop-btn" class="stop-btn" onclick="stopGeneration(event)">
            <span>⏹</span> Stop
          </button>
        </div>
        <div class="controls-bar">
          <div class="controls-left">
            <!-- Claude & Antigravity Style Thinking Level Selector -->
            <button class="menu-pill-btn" id="thinking-btn" onclick="toggleThinkingDropdown(event)">
              <span>🧠</span>
              <span id="thinking-btn-label">Thinking: Medium</span>
              <svg class="caret" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5z"/></svg>
            </button>

            <!-- Popup Menu -->
            <div class="dropdown-popup" id="thinking-dropdown">
              <div class="dropdown-popup-header">Reasoning Depth</div>

              <div class="dropdown-item" data-effort="off" onclick="selectEffort('off')">
                <div class="item-left">
                  <div class="item-icon">⭕</div>
                  <div class="item-info">
                    <div class="item-title">Off (Direct)</div>
                    <div class="item-desc">Skip reasoning; instant conclusion</div>
                  </div>
                </div>
                <div class="item-check">✓</div>
              </div>

              <div class="dropdown-item" data-effort="low" onclick="selectEffort('low')">
                <div class="item-left">
                  <div class="item-icon">⚡</div>
                  <div class="item-info">
                    <div class="item-title">Low</div>
                    <div class="item-desc">Brief & concise thoughts (~1k tokens)</div>
                  </div>
                </div>
                <div class="item-check">✓</div>
              </div>

              <div class="dropdown-item active" data-effort="medium" onclick="selectEffort('medium')">
                <div class="item-left">
                  <div class="item-icon">💡</div>
                  <div class="item-info">
                    <div class="item-title">Medium (Default)</div>
                    <div class="item-desc">Balanced thought process (~4k tokens)</div>
                  </div>
                </div>
                <div class="item-check">✓</div>
              </div>

              <div class="dropdown-item" data-effort="xhigh" onclick="selectEffort('xhigh')">
                <div class="item-left">
                  <div class="item-icon">🔥</div>
                  <div class="item-info">
                    <div class="item-title">X-High (Maximum)</div>
                    <div class="item-desc">Exhaustive rigor, validates all assumptions</div>
                  </div>
                </div>
                <div class="item-check">✓</div>
              </div>
            </div>

            <!-- Windows Native Tools Pill -->
            <button class="menu-pill-btn active-state" id="tools-toggle-btn" onclick="toggleTools()">
              <span>🛠️</span>
              <span id="tools-btn-label">Windows Tools: ON</span>
            </button>

            <!-- Permission mode: Auto / Accept edits / Bypass permissions -->
            <div class="perm-wrap">
              <button class="menu-pill-btn perm-bypass" id="perm-btn" onclick="togglePermDropdown(event)" title="What Shadow may do with its tools without asking you">
                <span class="perm-dot">&#9679;</span>
                <span id="perm-btn-label">Bypass permissions</span>
                <svg class="caret" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5z"/></svg>
              </button>
              <div class="dropdown-popup" id="perm-dropdown">
                <div class="dropdown-popup-header">Permission mode</div>
                <div class="dropdown-item" data-perm="auto" onclick="selectPermMode('auto')">
                  <div class="item-left">
                    <div class="item-icon">&#9889;</div>
                    <div class="item-info">
                      <div class="item-title">Auto</div>
                      <div class="item-desc">Reads, file edits and safe commands (listing, git status, builds, tests) run; anything else asks</div>
                    </div>
                  </div>
                  <div class="item-check">&#10003;</div>
                </div>
                <div class="dropdown-item" data-perm="acceptEdits" onclick="selectPermMode('acceptEdits')">
                  <div class="item-left">
                    <div class="item-icon">&#9998;</div>
                    <div class="item-info">
                      <div class="item-title">Accept edits</div>
                      <div class="item-desc">Reads and file edits run; every command and other tool asks</div>
                    </div>
                  </div>
                  <div class="item-check">&#10003;</div>
                </div>
                <div class="dropdown-item active" data-perm="bypass" onclick="selectPermMode('bypass')">
                  <div class="item-left">
                    <div class="item-icon">&#9888;</div>
                    <div class="item-info">
                      <div class="item-title">Bypass permissions</div>
                      <div class="item-desc">Everything runs without asking (the way it was before this button)</div>
                    </div>
                  </div>
                  <div class="item-check">&#10003;</div>
                </div>
              </div>
            </div>

            <!-- Pinned notes -->
            <button class="menu-pill-btn" id="regen-btn" onclick="regenerateLast()" title="Throw the last reply away and ask again (with sampling on, the answer will differ). Not offered after a reply that used tools: they would run again.">
              <span>↻</span>
              <span>Regenerate</span>
            </button>
            <button class="menu-pill-btn" id="notes-toggle-btn" onclick="toggleNotes()" title="Pinned notes for this chat: shown to Shadow at the end of every request, never shortened">
              <span>📝</span>
              <span id="notes-btn-label">Notes</span>
            </button>

          </div>

          <span id="timing-info">Ready</span>
        </div>
      </div>
    </div>

    <aside id="browser-pane">
      <div class="bp-head">
        <span>🌐</span>
        <span class="bp-url" id="bp-url">nothing open yet</span>
        <button class="side-toggle" title="Refresh the picture" onclick="browserRefresh()">⟳</button>
        <button class="side-toggle" title="Close this pane" onclick="browserClose()">✕</button>
      </div>
      <div class="bp-body"><img id="bp-img" alt="The page in the browser"><div class="dash-empty" id="bp-empty">Nothing loaded yet.</div></div>
      <div class="bp-status" id="bp-status">Shadow's browser</div>
    </aside>

  </div>

  <!-- System Persona Customization Modal -->
  <div id="persona-modal">
    <div class="modal-card">
      <div class="modal-header">
        <h3>Windows Developer Authority & Persona</h3>
        <button class="modal-close" onclick="closePersonaModal()">✕</button>
      </div>
      <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.4;">
        This system prompt gives Shadow full authority on Windows with native tools (read_file, glob, grep, edit_file, write_file, cmd).
      </p>
      <textarea class="modal-textarea" id="persona-textarea"></textarea>
      <div class="modal-actions">
        <button class="modal-btn" onclick="resetPersonaDefault()">Reset Default</button>
        <button class="modal-btn primary" onclick="savePersona()">Save & Apply</button>
      </div>
    </div>
  </div>

  <script>
    // State
    let currentView = 'chat';
    let currentEffort = localStorage.getItem('shadow_effort') || 'medium';
    if (currentEffort === 'high') currentEffort = 'medium'; // the High level was removed from the menu
    let toolsEnabled = localStorage.getItem('shadow_tools_enabled') !== 'false';
    // The persona is the part the user edits (who Shadow is, how it talks); ENGINE_RULES is how its tools and its small context window work: owned by the app, updated with it, added
    // after the persona when tools are on.  LEGACY_PERSONA is the one-piece default of before STATUS 122 (kept to recognise a saved copy of it and to run the old arm of the A-B test).
    const LEGACY_PERSONA = "You are Shadow, a capable AI assistant running locally on the user's Windows PC: nothing you are told leaves this machine. Be direct and accurate, keep answers as short as the question allows, and say so when you are unsure. You can use tools: read_file, glob, grep, edit_file, write_file, cmd (PowerShell), web_search and web_fetch (use these for anything current or that you are not sure of, and cite the page), and any mcp__ tools you are given. When asked to look at, test or change files, use the tools instead of describing what you would do. If the user asks you to take a role for a task (for example 'act as a software engineer and design this'), do that for the task. Your context window is small. Earlier tool outputs are shortened to a preview that says how to read them again (recall with id=N or message=N, optionally a line range or a text to search for), old turns become one-line summaries that name their messages (recall with message=N), and recall with only a text searches everything earlier (search finds files and earlier outputs by keyword, ranked by BM25 and path match). Keep pinned notes with the notes tool (the plan, the decisions, the facts you will need with file and line numbers): they are shown to you at the end of every request and never shortened. To change code: read just the lines you need with read_file (about 100), then replace them by line number with replace_lines (give expect, a piece of the old text, so a stale number cannot overwrite the wrong lines), or use edit_file with the exact old text. After reading something you will need later, put the facts in your notes, and do not read the same part twice; after about eight reads in a row, stop reading, save your notes and make the edit. Work in small steps and finish with an answer.";
    const DEFAULT_PERSONA = "You are Shadow, a capable assistant running locally on the user's Windows PC; nothing you are told leaves this machine. Be direct and accurate, and keep answers as short as the question allows. Say so when you are unsure, and never claim you ran, tested or verified something you did not. Answer ordinary questions directly, and use tools only when the task is about files, commands or current information. Think briefly: if you notice you are repeating a point you already reasoned through, stop and commit to an answer. Use markdown and fenced code blocks for code. Reply in the user's language. If asked to take a role for a task (for example 'act as a software engineer and design this'), do that for the task.";
    const ENGINE_RULES = "Use tools for anything about files, commands or current information: web_search and web_fetch for anything current or that you are not sure of (and cite the page), and any mcp__ tools you are given. When asked to look at, test or change files, use the tools instead of describing what you would do. Your context window is small. Earlier tool outputs are shortened to a preview that says how to read them again (recall with id=N or message=N, optionally a line range or a text to search for), old turns become one-line summaries that name their messages (recall with message=N), and recall with only a text searches everything earlier (search finds files and earlier outputs by keyword, ranked by BM25 and path match). Keep pinned notes with the notes tool (the plan, the decisions, the facts you will need with file and line numbers): they are shown to you at the end of every request and never shortened. To change code: read just the lines you need with read_file (about 100), then replace them by line number with replace_lines (give expect, a piece of the old text, so a stale number cannot overwrite the wrong lines), or use edit_file with the exact old text. After reading something you will need later, put the facts in your notes, and do not read the same part twice; after about eight reads in a row, stop reading, save your notes and make the edit. Work in small steps and finish with an answer.";
    let currentPersona = localStorage.getItem('shadow_persona') || DEFAULT_PERSONA;
    {
      const saved = currentPersona.trim();
      const bm25 = LEGACY_PERSONA.replace('(search finds files and earlier outputs by keyword, ranked by BM25 and path match)', '(search finds files and earlier outputs by meaning, when you do not know the exact words)');
      if (saved === LEGACY_PERSONA.trim() || saved === bm25.trim()) { currentPersona = DEFAULT_PERSONA; localStorage.removeItem('shadow_persona'); }
    }
    if (currentPersona.includes('(search finds files and earlier outputs by meaning, when you do not know the exact words)')) {
      currentPersona = currentPersona.replace(
        '(search finds files and earlier outputs by meaning, when you do not know the exact words)',
        '(search finds files and earlier outputs by keyword, ranked by BM25 and path match)'
      );
      localStorage.setItem('shadow_persona', currentPersona);
    }

    // Elements
    const chatPanel = document.getElementById('chat-panel');
    const tabChat = document.getElementById('tab-chat');
    const messagesEl = document.getElementById('messages');
    const inputEl = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const thinkingBtn = document.getElementById('thinking-btn');
    const thinkingLabel = document.getElementById('thinking-btn-label');
    const thinkingDropdown = document.getElementById('thinking-dropdown');
    const toolsToggleBtn = document.getElementById('tools-toggle-btn');
    const toolsBtnLabel = document.getElementById('tools-btn-label');
    const personaModal = document.getElementById('persona-modal');
    const personaTextarea = document.getElementById('persona-textarea');
    const speedPill = document.getElementById('speed-pill');
    const statusPill = document.getElementById('status-pill');
    const gpuHeaderPill = document.getElementById('gpu-header-pill');
    const timingInfo = document.getElementById('timing-info');
    const scrollBottomBtn = document.getElementById('scroll-bottom-btn');

    let history = [];
    let isGenerating = false;
    let availableTools = [];
    let currentAbortController = null;
    let autoScroll = true;

    function updateScrollBtn() {
      if (!scrollBottomBtn) return;
      const dist = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
      scrollBottomBtn.classList.toggle('visible', !autoScroll && dist > 100 && history.length > 0);
    }
    messagesEl.addEventListener('scroll', () => {
      const dist = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
      autoScroll = dist < 48;
      updateScrollBtn();
    }, { passive: true });
    messagesEl.addEventListener('wheel', (e) => {
      if (e.deltaY < 0) {
        autoScroll = false;
        updateScrollBtn();
      }
    }, { passive: true });
    function scrollToBottom(force) {
      if (force) autoScroll = true;
      if (autoScroll) messagesEl.scrollTop = messagesEl.scrollHeight;
      updateScrollBtn();
    }

    function stopGeneration(e) {
      if (e) {
        if (typeof e.stopPropagation === 'function') e.stopPropagation();
        if (typeof e.preventDefault === 'function') e.preventDefault();
      }
      if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
      }
      isGenerating = false;
      document.querySelectorAll('.msg-row.working, .tool-call-block.running').forEach(r => r.classList.remove('working', 'running'));
      sendBtn.textContent = 'Send';
      sendBtn.style.display = 'flex';
      stopBtn.style.display = 'none';
      sendBtn.disabled = false;
      inputEl.focus();
      timingInfo.textContent = 'Generation stopped by user';
    }

    function toggleThinkBlock(el) {
      const block = el.closest('.think-block');
      if (!block) return;
      const content = block.querySelector('.think-content');
      const chevron = block.querySelector('.think-chevron');
      if (content.style.display === 'none') {
        content.style.display = 'block';
        if (chevron) chevron.textContent = '▼';
      } else {
        content.style.display = 'none';
        if (chevron) chevron.textContent = '▶';
      }
    }

    // Copy button on chat bubbles: appears on row hover, copies everything visible in the bubble
    let copyProbe = null;
    function visibleBubbleText(bubble) {
      if (!copyProbe) {
        copyProbe = document.createElement('div');
        copyProbe.style.cssText = 'position:fixed;left:-99999px;top:0;width:1px;';
        document.body.appendChild(copyProbe);
      }
      const clone = bubble.cloneNode(true);
      clone.querySelectorAll('.copy-btn').forEach(b => b.remove());
      clone.style.width = bubble.offsetWidth + 'px';
      copyProbe.appendChild(clone);
      const text = clone.innerText || clone.textContent || '';
      copyProbe.innerHTML = '';
      return text;
    }
    function fallbackCopyText(text) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;left:-99999px;';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch (_) {}
      ta.remove();
    }
    function copyBubble(btn) {
      const text = visibleBubbleText(btn.parentElement);
      const done = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1200); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, () => { fallbackCopyText(text); done(); });
      } else { fallbackCopyText(text); done(); }
    }
    messagesEl.addEventListener('mouseover', (e) => {
      const row = e.target.closest ? e.target.closest('.msg-row') : null;
      if (!row) return;
      const bubble = row.querySelector(':scope > .bubble');
      if (!bubble) return;
      if (!bubble.querySelector(':scope > .copy-btn')) {
        const btn = document.createElement('button');
        btn.className = 'copy-btn bubble-copy';
        btn.textContent = 'Copy';
        btn.onclick = () => copyBubble(btn);
        bubble.appendChild(btn);
      }
    });

    // Global keyboard shortcuts: Escape closes modals/dropdowns or stops generation; Ctrl+K searches chats; Ctrl+Shift+O starts a new chat
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (personaModal && personaModal.classList.contains('open')) {
          e.preventDefault();
          closePersonaModal();
          return;
        }
        if (thinkingDropdown && thinkingDropdown.classList.contains('open')) {
          e.preventDefault();
          thinkingDropdown.classList.remove('open');
          thinkingBtn.classList.remove('open');
          return;
        }
        if (isGenerating) {
          e.preventDefault();
          stopGeneration(e);
        }
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K' || e.key === '/')) {
        e.preventDefault();
        if (currentView !== 'chat') switchView('chat');
        if (document.body.classList.contains('side-closed')) toggleSidebar();
        const searchEl = document.getElementById('chat-search');
        if (searchEl) { searchEl.focus(); searchEl.select(); }
      } else if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'o' || e.key === 'O')) {
        e.preventDefault();
        if (currentView !== 'chat') switchView('chat');
        newChat();
      }
    });

    // ---- Pinned notes (a scratchpad that survives the shortening of a long run): kept with the chat, shown to the model at the END of every request (the start of the prompt
    // stays the same, so the server's cached prompt is reused), changed by the model with the `notes` tool or by the user in the box above the input.
    const NOTES_MAX = 6000;
    let scratchpad = '';
    const NOTES_TOOL = {
      name: 'notes',
      description: 'Your pinned notes for this task: they are shown to you at the end of every request and are never shortened, unlike earlier tool outputs. Keep the plan, the decisions and the facts you will need (file, line numbers, exact text to change). With text they are replaced, with append a line is added, with neither they are returned. At most ' + NOTES_MAX + ' characters.',
      input_schema: { type: 'object', properties: {
        text: { type: 'string', description: 'The new notes in full (replaces the old ones)' },
        append: { type: 'string', description: 'Text to add at the end of the notes' } } }
    };
    function renderNotes() {
      const t = document.getElementById('notes-text');
      if (document.activeElement !== t) t.value = scratchpad;
      document.getElementById('notes-btn-label').textContent = scratchpad ? `Notes (${scratchpad.length})` : 'Notes';
      document.getElementById('notes-hint').textContent = `${scratchpad.length} of ${NOTES_MAX} characters`;
    }
    function toggleNotes() { document.getElementById('notes-box').classList.toggle('open'); renderNotes(); }
    document.getElementById('notes-text').addEventListener('input', e => { scratchpad = e.target.value.slice(0, NOTES_MAX); renderNotes(); saveCurrent(); });
    function runNotesTool(args) {
      const a = args || {};
      let next = scratchpad;
      if (typeof a.text === 'string') next = a.text;
      else if (typeof a.append === 'string' && a.append.trim()) next = scratchpad ? scratchpad.replace(/\s+$/, '') + '\n' + a.append : a.append;
      else return { name: 'notes', output: scratchpad ? `Your notes (${scratchpad.length} characters):\n${scratchpad}` : 'Your notes are empty.', is_error: false };
      if (next.length > NOTES_MAX) return { name: 'notes', output: `error: the notes would be ${next.length} characters, the limit is ${NOTES_MAX}: shorten them (keep the facts you still need) and send them again with text`, is_error: true };
      scratchpad = next;
      renderNotes();
      saveCurrent();
      return { name: 'notes', output: `Notes saved (${scratchpad.length} of ${NOTES_MAX} characters).`, is_error: false };
    }
    // the notes at the end of the last message (inside the tool response when it is one): only the tail of the prompt changes when the notes do
    function withNotes(msgs) {
      if (!scratchpad.trim() || !msgs.length) return msgs;
      const block = `\n\n[Your notes so far - pinned, never shortened; change them with the notes tool]\n${scratchpad}`;
      const last = msgs[msgs.length - 1];
      let c = last.content;
      if (typeof c === 'string') {
        const close = '\n</tool_response>';
        c = last.role === 'tool' && c.endsWith(close) ? c.slice(0, -close.length) + block + close : c + block;
      } else if (Array.isArray(c)) {
        const parts = c.map(p => ({ ...p }));
        const ti = parts.map(p => p.type).lastIndexOf('text');
        if (ti >= 0) parts[ti].text += block; else parts.push({ type: 'text', text: block.trim() });
        c = parts;
      }
      return [...msgs.slice(0, -1), { ...last, content: c }];
    }

    // ---- Permission mode (docs/STATUS.md 110): what a tool call may do without asking the user.  auto: reads, file edits and commands judged safe run, the rest asks;
    // acceptEdits: reads and file edits run, commands and other tools ask; bypass: everything runs (the behaviour before the button, and the default).
    const PERM_MODES = { auto: 'Auto', acceptEdits: 'Accept edits', bypass: 'Bypass permissions' };
    let permMode = localStorage.getItem('shadow_perm_mode');
    if (!PERM_MODES[permMode]) permMode = 'bypass';
    const READ_TOOLS = new Set(['read_file', 'glob', 'grep', 'search', 'recall', 'notes', 'memory_list', 'web_search', 'web_fetch', 'finish', 'mcp__time__get_current_time', 'mcp__time__convert_time', 'mcp__time__get_time_ago']);
    const EDIT_TOOLS = new Set(['edit_file', 'replace_lines', 'write_file', 'memory_save', 'memory_forget']);
    // commands of Auto mode: the first word of every part of a command line must be one of these (a listing, a read, a build or a test), the arguments must not redirect,
    // delete, download or start anything, and no script block may hide another command
    const SAFE_CMD_WORDS = new Set(['get-childitem', 'gci', 'ls', 'dir', 'get-content', 'gc', 'cat', 'type', 'select-string', 'sls', 'findstr', 'get-location', 'pwd', 'echo', 'write-output',
      'write-host', 'get-item', 'test-path', 'measure-object', 'sort-object', 'select-object', 'get-date', 'where', 'hostname', 'whoami', 'get-command', 'get-process', 'resolve-path',
      'split-path', 'join-path', 'mkdir', 'get-filehash', 'format-table', 'format-list', 'out-string', 'head', 'tail', 'wc']);
    const SAFE_GIT = new Set(['status', 'diff', 'log', 'show', 'rev-parse', 'ls-files', 'blame', 'branch', 'remote', 'describe', 'shortlog']);
    const SAFE_CARGO = new Set(['check', 'build', 'test', 'tree', 'metadata', 'doc']);
    const DANGEROUS_CMD = /(^|[\s;|&(])(remove-item|rm|ri|del|erase|rmdir|rd|format|diskpart|stop-process|kill|taskkill|shutdown|restart-computer|stop-computer|restart-service|stop-service|set-service|new-service|sc|reg|regedit|setx|set-itemproperty|new-itemproperty|remove-itemproperty|invoke-expression|iex|invoke-webrequest|iwr|invoke-restmethod|irm|curl|wget|start-process|start|saps|start-bitstransfer|move-item|mv|mi|copy-item|cp|copy|move|rename-item|ren|set-content|add-content|out-file|clear-content|new-item|ni|set-location|cd|chdir|pushd|popd|set-executionpolicy|powershell|pwsh|cmd|bash|wsl|msiexec|winget|choco|pip|npm|npx|yarn|cargo-install|icacls|takeown|attrib|net|netsh|schtasks|certutil|bitsadmin)(\s|$)/i;
    function commandIsSafe(cmd) {
      const c = String(cmd || '').trim();
      if (!c || c.length > 600) return false;
      if (/[{}`$]|>|<|\.\.\\|\.\.\//.test(c.replace(/2>&1/g, ''))) return false;   // redirects, script blocks, variables / sub-expressions, climbing out of the folder
      if (DANGEROUS_CMD.test(c.replace(/^\s*mkdir\b/i, ''))) return false;
      const parts = c.split(/\r?\n|;|&&|\|\||\|/).map(x => x.trim()).filter(Boolean);
      if (!parts.length) return false;
      return parts.every(part => {
        const w = part.split(/\s+/).map(x => x.replace(/^['"]|['"]$/g, ''));
        const first = (w[0] || '').toLowerCase().replace(/\.exe$/, '');
        if (first === 'git') return SAFE_GIT.has((w[1] || '').toLowerCase()) && !/(^|\s)(-d|-D|--delete|-m|-M|--set-upstream-to|--unset-upstream)(\s|$)/.test(part);
        if (first === 'cargo') return SAFE_CARGO.has((w[1] || '').toLowerCase());
        if (first === 'python' || first === 'py' || first === 'python3') return w.length >= 2 && /^[\w\\/.:-]+\.py$/i.test(w[1]) || (w[1] === '-m' && /^(pytest|unittest)$/.test(w[2] || ''));
        if (first === 'node') return w.length >= 2 && /^[\w\\/.:-]+\.js$/i.test(w[1]);
        if (first === 'pytest') return true;
        return SAFE_CMD_WORDS.has(first);
      });
    }
    // null: the call runs; else why it asks
    function permissionReason(tc) {
      if (permMode === 'bypass') return null;
      const name = tc.name;
      if (READ_TOOLS.has(name) || EDIT_TOOLS.has(name)) return null;
      if (name === 'cmd') {
        if (permMode === 'auto' && commandIsSafe((tc.arguments || {}).command)) return null;
        return permMode === 'auto' ? 'This command is not on the list of safe ones (listing, reading, git status / diff / log, builds, tests, running a script file).' : 'Accept edits mode: commands ask.';
      }
      return (permMode === 'auto' ? 'Auto mode: ' : 'Accept edits mode: ') + name + ' is not a read or a file edit.';
    }
    const permAllowed = new Set();   // "always allow in this chat": a tool name, or the exact command of a cmd call
    let permChat = null;
    function permKey(tc) { return tc.name === 'cmd' ? 'cmd:' + String((tc.arguments || {}).command || '') : tc.name; }
    // asks the user inside the tool card: resolves 'allow' or 'deny'; Stop ends the wait as an abort
    function askPermission(tc, toolCard, reason) {
      return new Promise((resolve, reject) => {
        const sig = currentAbortController ? currentAbortController.signal : null;
        const box = document.createElement('div');
        box.className = 'perm-box';
        const what = tc.name === 'cmd' ? String((tc.arguments || {}).command || '') : JSON.stringify(tc.arguments || {}, null, 1).slice(0, 1500);
        box.innerHTML = '<div class="perm-why">Shadow asks before running <strong>' + escapeHtml(tc.name) + '</strong>. ' + escapeHtml(reason || '') + '</div>' +
          '<div class="perm-what">' + escapeHtml(what) + '</div>' +
          '<div class="perm-btns"><button type="button" class="allow" data-a="allow">Allow</button>' +
          '<button type="button" data-a="always">Always allow ' + (tc.name === 'cmd' ? 'this command' : escapeHtml(tc.name)) + ' in this chat</button>' +
          '<button type="button" class="deny" data-a="deny">Deny</button></div>';
        toolCard.appendChild(box);
        const head = toolCard.querySelector('.tool-header span:last-child');
        if (head) head.textContent = 'Waiting for you';
        toolCard.classList.remove('running');
        scrollToBottom(false);
        let done = false;
        const finish = (v, err) => { if (done) return; done = true; box.remove(); if (sig) sig.removeEventListener('abort', onAbort); if (err) reject(err); else resolve(v); };
        const onAbort = () => { const h = toolCard.querySelector('.tool-header span:last-child'); if (h) h.textContent = 'Stopped'; finish(null, new DOMException('Stopped', 'AbortError')); };
        if (sig) { if (sig.aborted) return onAbort(); sig.addEventListener('abort', onAbort); }
        box.querySelectorAll('button').forEach(b => b.onclick = () => {
          const a = b.dataset.a;
          if (a === 'always') permAllowed.add(permKey(tc));
          finish(a === 'deny' ? 'deny' : 'allow');
        });
      });
    }
    async function permissionGate(tc, toolCard) {
      if (permChat !== currentChatId) { permAllowed.clear(); permChat = currentChatId; }
      const reason = permissionReason(tc);
      if (reason === null || permAllowed.has(permKey(tc))) return 'allow';
      return askPermission(tc, toolCard, reason);
    }
    function deniedResult(tc) {
      return { name: tc.name, output: 'The user denied this call (permission mode: ' + PERM_MODES[permMode] + '): it was not run. Do not repeat it. Use another way, or say what you wanted to do and ask the user.', is_error: true };
    }
    function updatePermUI() {
      const btn = document.getElementById('perm-btn');
      if (!btn) return;
      document.getElementById('perm-btn-label').textContent = PERM_MODES[permMode];
      btn.classList.remove('perm-auto', 'perm-accept', 'perm-bypass');
      btn.classList.add(permMode === 'auto' ? 'perm-auto' : permMode === 'acceptEdits' ? 'perm-accept' : 'perm-bypass');
      document.querySelectorAll('#perm-dropdown .dropdown-item').forEach(el => el.classList.toggle('active', el.dataset.perm === permMode));
    }
    function togglePermDropdown(e) {
      document.documentElement.style.setProperty('--bar-h', document.querySelector('.input-box').offsetHeight + 'px');
      e.stopPropagation();
      document.getElementById('perm-dropdown').classList.toggle('open');
      document.getElementById('perm-btn').classList.toggle('open');
    }
    function selectPermMode(m) {
      if (!PERM_MODES[m]) return;
      permMode = m;
      localStorage.setItem('shadow_perm_mode', m);
      updatePermUI();
      document.getElementById('perm-dropdown').classList.remove('open');
      document.getElementById('perm-btn').classList.remove('open');
    }
    document.addEventListener('click', (e) => {
      const dd = document.getElementById('perm-dropdown'), b = document.getElementById('perm-btn');
      if (dd && !dd.contains(e.target) && !b.contains(e.target)) { dd.classList.remove('open'); b.classList.remove('open'); }
    });
    updatePermUI();

    // ---- The closing status of an agent task (docs/STATUS.md 109): a task that used tools ends with a `finish` call (done / pending / question with options), drawn as a card
    // and ending the turn.  The tool comes from the server (GET /v1/tools; SHADOW_FINISH_TOOL=0 removes it there), this switch is the page's own (Settings).
    let finishOn = localStorage.getItem('shadow_finish_on') !== 'false';
    function finishActive() { return toolsEnabled && finishOn && availableTools.some(t => t.name === 'finish'); }
    function setFinishOn(on) { finishOn = on; localStorage.setItem('shadow_finish_on', on); }
    // the tools of a request: without the page's finish switch the model never sees the tool
    function requestTools() { return finishOn ? availableTools : availableTools.filter(t => t.name !== 'finish'); }
    const FINISH_HINT = 'End a task that used your tools with one call to finish (alone, after you have seen the results of your other calls), not with a closing message. ' +
      'done: what you actually did, a short item a line. pending: what is NOT done, NOT tested or NOT verified, an item a line (say so if you did not run the code; never claim what you did not run). ' +
      'question and options: only if you cannot go on without the user (the answers a line each, your recommendation first). ' +
      'For a plain question or chat, answer in text.';
    // a status field as a list: a JSON array or lines (bullets and numbers stripped), at most 12 items of 400 characters
    function finishItems(v) {
      if (v === undefined || v === null) return [];
      let a = v;
      if (typeof a === 'string') {
        const t = a.trim();
        if (t.startsWith('[')) { try { const j = JSON.parse(t); if (Array.isArray(j)) a = j; } catch (_) {} }
        if (typeof a === 'string') a = t.split(/\r?\n/);
      }
      if (!Array.isArray(a)) a = [a];
      return a.map(x => String(x === null || x === undefined ? '' : x).replace(/^\s*(?:[-*\u2022]|\d+[.)])\s+/, '').trim()).filter(Boolean).slice(0, 12).map(x => x.slice(0, 400));
    }
    function parseFinish(args) {
      const a = args && typeof args === 'object' ? args : {};
      const question = String(a.question === undefined || a.question === null ? '' : a.question).trim().slice(0, 600);
      return { done: finishItems(a.done), pending: finishItems(a.pending !== undefined ? a.pending : a.not_done), question, options: question ? finishItems(a.options).slice(0, 6) : [] };
    }
    // the status is accepted when the model said what it did and what is left (or asked); `calls` is how many tool calls the message held: finish goes alone, after the results
    function checkFinish(args, calls) {
      const f = parseFinish(args);
      if (calls > 1) return { ok: false, output: 'error: call finish alone, after you have seen the results of your other tool calls (they ran just now); report the status in your next message.' };
      if (!f.done.length && !f.pending.length && !f.question) return { ok: false, output: 'error: finish needs done (what you did), and pending (what is not done or not verified) or a question' };
      return { ok: true, f };
    }
    function finishCardHtml(f) {
      const list = items => '<ul>' + items.map(x => '<li>' + escapeHtml(x) + '</li>').join('') + '</ul>';
      let h = '';
      if (f.done.length) h += '<div class="fin-done"><div class="fin-h">Done</div>' + list(f.done) + '</div>';
      if (f.pending.length) h += '<div class="fin-pending"><div class="fin-h">Not done / not verified</div>' + list(f.pending) + '</div>';
      else if (!f.question) h += '<div class="fin-pending"><div class="fin-h">Not done / not verified</div><div>Nothing was reported as left.</div></div>';
      if (f.question) {
        h += '<div class="fin-h">Needs your answer</div><div class="fin-q">' + escapeHtml(f.question) + '</div>';
        if (f.options.length) h += '<div class="fin-opts">' + f.options.map(o => '<button type="button" class="fin-opt" onclick="useSuggestion(this)">' + escapeHtml(o) + '</button>').join('') + '</div>';
      }
      return '<div class="finish-card' + (f.question ? ' open' : '') + '">' + h + '</div>';
    }
    function finishCardEl(f) {
      const d = document.createElement('div');
      d.innerHTML = finishCardHtml(f);
      return d.firstChild;
    }

    // Load tool schemas from server
    async function loadTools() {
      try {
        const res = await fetch('/v1/tools');
        if (res.ok) {
          availableTools = (await res.json()).concat([NOTES_TOOL]);
        }
      } catch (_) {}
    }
    loadTools();

    // Tools toggle
    function toggleTools() {
      toolsEnabled = !toolsEnabled;
      localStorage.setItem('shadow_tools_enabled', toolsEnabled);
      updateToolsUI();
    }
    function updateToolsUI() {
      if (toolsEnabled) {
        toolsToggleBtn.classList.add('active-state');
        toolsBtnLabel.textContent = 'Windows Tools: ON';
      } else {
        toolsToggleBtn.classList.remove('active-state');
        toolsBtnLabel.textContent = 'Windows Tools: OFF';
      }
    }
    updateToolsUI();

    // View switching: Chat, Dashboard and Settings are separate tabs
    function switchView(view) {
      currentView = view;
      const panels = { chat: chatPanel, dash: document.getElementById('dash-panel'), settings: document.getElementById('settings-panel') };
      const tabs = { chat: tabChat, dash: document.getElementById('tab-dash'), settings: document.getElementById('tab-settings') };
      for (const k of Object.keys(panels)) {
        panels[k].classList.toggle('active', k === view);
        tabs[k].classList.toggle('active', k === view);
      }
      document.body.classList.toggle('view-chat', view === 'chat');
      window.location.hash = view === 'chat' ? '' : '#' + view;
      if (view === 'dash') dashRefresh();
      if (view === 'settings') settingsOpen();
    }

    document.getElementById('brand-logo').addEventListener('click', () => switchView('chat'));


    // Thinking Level Selection
    function toggleThinkingDropdown(e) {
      document.documentElement.style.setProperty('--bar-h', document.querySelector('.input-box').offsetHeight + 'px');
      e.stopPropagation();
      thinkingDropdown.classList.toggle('open');
      thinkingBtn.classList.toggle('open');
    }

    function selectEffort(level) {
      currentEffort = level;
      localStorage.setItem('shadow_effort', level);
      updateThinkingUI();
      thinkingDropdown.classList.remove('open');
      thinkingBtn.classList.remove('open');
    }

    function updateThinkingUI() {
      const labels = {
        off: 'Thinking: Off',
        low: 'Thinking: Low',
        medium: 'Thinking: Medium',
        high: 'Thinking: High',
        xhigh: 'Thinking: X-High'
      };
      thinkingLabel.textContent = labels[currentEffort] || 'Thinking: Medium';

      document.querySelectorAll('#thinking-dropdown .dropdown-item').forEach(el => {
        if (el.dataset.effort === currentEffort) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      });
    }
    updateThinkingUI();

    document.addEventListener('click', (e) => {
      if (!thinkingDropdown.contains(e.target) && !thinkingBtn.contains(e.target)) {
        thinkingDropdown.classList.remove('open');
        thinkingBtn.classList.remove('open');
      }
    });

    // Persona Modal
    function openPersonaModal() {
      personaTextarea.value = currentPersona;
      personaModal.classList.add('open');
    }
    function closePersonaModal() {
      personaModal.classList.remove('open');
    }
    function resetPersonaDefault() {
      personaTextarea.value = DEFAULT_PERSONA;
    }
    function savePersona() {
      currentPersona = personaTextarea.value.trim() || DEFAULT_PERSONA;
      localStorage.setItem('shadow_persona', currentPersona);
      closePersonaModal();
    }

    // Textarea auto-resize (progressive enhancement: CSS field-sizing: content handles modern browsers; JS fallback for older engines)
    const supportsFieldSizing = typeof CSS !== 'undefined' && CSS.supports && CSS.supports('field-sizing', 'content');
    inputEl.addEventListener('input', () => {
      if (!supportsFieldSizing) {
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
      }
      document.documentElement.style.setProperty('--bar-h', document.querySelector('.input-box').offsetHeight + 'px');
    });

    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (e.isComposing || e.keyCode === 229) return;
        sendMessage();
      }
    });

    sendBtn.addEventListener('click', sendMessage);

    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[m]));
    }

    function copyCodeBlock(btn) {
      const codeEl = btn.nextElementSibling;
      const text = codeEl ? (codeEl.innerText || codeEl.textContent || '') : '';
      const done = () => { btn.textContent = 'Copied ✓'; setTimeout(() => { btn.textContent = 'Copy'; }, 1200); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, () => { fallbackCopyText(text); done(); });
      } else { fallbackCopyText(text); done(); }
    }

    // A small Markdown subset, escape-first: everything the model wrote is HTML-escaped before any tag is added, so model
    // output can never inject markup.  Fenced code, inline code, bold, italics, headings, bullet and numbered lists.
    function renderMarkdown(raw) {
      const blocks = [];
      let text = raw.replace(/```(\w*)\n?([\s\S]*?)(```|$)/g, (_, lang, code) => {
        blocks.push(`<pre><button class="copy-btn" onclick="copyCodeBlock(this)">Copy</button><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`);
        return `\u0000${blocks.length - 1}\u0000`;
      });
      const inline = (t) => escapeHtml(t)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|[^*\w])\*([^*\s][^*]*)\*(?!\w)/g, '$1<em>$2</em>');
      const out = [];
      let list = null; // 'ul' | 'ol'
      const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
      for (const line of text.split('\n')) {
        let m;
        if ((m = line.match(/^\u0000(\d+)\u0000$/))) { close(); out.push(blocks[+m[1]]); continue; }
        if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { close(); out.push(`<div class="md-h md-h${m[1].length}">${inline(m[2])}</div>`); continue; }
        if ((m = line.match(/^\s*[-*]\s+(.*)$/))) {
          if (list !== 'ul') { close(); out.push('<ul>'); list = 'ul'; }
          out.push(`<li>${inline(m[1])}</li>`); continue;
        }
        if ((m = line.match(/^\s*(\d+)[.)]\s+(.*)$/))) {
          if (list !== 'ol') { close(); out.push(`<ol start="${m[1]}">`); list = 'ol'; }
          out.push(`<li>${inline(m[2])}</li>`); continue;
        }
        close();
        out.push(line.trim() === '' ? '<div class="md-gap"></div>' : `<div>${inline(line).replace(/\u0000(\d+)\u0000/g, (_, k) => blocks[+k])}</div>`);
      }
      close();
      return out.join('');
    }

    // Tool Call parser for <tool_call><function=name><parameter=k>v</parameter>...</function></tool_call>
    function parseToolCallsFromText(text) {
      const calls = [];
      const regex = /<tool_call>([\s\S]*?)<\/tool_call>/g;
      let match;
      while ((match = regex.exec(text)) !== null) {
        const inner = match[1];
        const fnMatch = /<function=([^>]+)>([\s\S]*?)<\/function>/.exec(inner);
        if (fnMatch) {
          const fnName = fnMatch[1].trim();
          const paramsInner = fnMatch[2];
          const args = {};
          const paramRegex = /<parameter=([^>]+)>([\s\S]*?)<\/parameter>/g;
          let pMatch;
          while ((pMatch = paramRegex.exec(paramsInner)) !== null) {
            args[pMatch[1].trim()] = pMatch[2].trim();
          }
          calls.push({ name: fnName, arguments: args, fullXml: match[0] });
        }
      }
      return calls;
    }

    // Strip raw <tool_call>...</tool_call> XML blocks (including an unclosed trailing <tool_call> during live streaming)
    // so raw XML tags never bleed into the chat bubble prose.
    function stripToolCalls(text) {
      if (!text) return '';
      return String(text)
        .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
        .replace(/<tool_call>[\s\S]*$/g, '')
        .trim();
    }

    function hasPendingToolCall(text) {
      if (!text) return false;
      const lastOpen = text.lastIndexOf('<tool_call>');
      if (lastOpen === -1) return false;
      const lastClose = text.indexOf('</tool_call>', lastOpen);
      return lastClose === -1;
    }

    function pendingToolCallName(text) {
      const lastOpen = text.lastIndexOf('<tool_call>');
      if (lastOpen === -1) return '';
      const tail = text.slice(lastOpen);
      const m = /<function=([^>\s]+)/.exec(tail);
      return m ? m[1].trim() : '';
    }

    function formatToolArgsSummary(args) {
      if (!args) return '';
      const entries = Object.entries(args);
      if (!entries.length) return '';
      return entries.map(([k, v]) => {
        const s = String(v || '').replace(/\s+/g, ' ').trim();
        return `${k}: ${s.length > 90 ? s.slice(0, 88) + '…' : s}`;
      }).join(' · ');
    }

    function toggleToolOutput(headerEl) {
      const card = headerEl.closest('.tool-call-block');
      if (!card) return;
      const out = card.querySelector('.tool-output');
      if (!out) return;
      const hidden = out.style.display === 'none';
      out.style.display = hidden ? 'block' : 'none';
    }

    // Execute tool on Windows via /v1/tools/execute
    // `recall` of an earlier MESSAGE (the page holds every message in full; the server only the tool outputs): read it again, or search all messages
    function recallMessages(args) {
      const lines = t => String(t).split(/\r?\n/);
      const body = m => {
        let t = messageText(m);
        if (m.role === 'tool') t = unwrapToolResponse(t).replace(/^\(output #\d+\)\s*/, '');
        return t;
      };
      if (args.message !== undefined && args.message !== null && args.message !== '') {
        const n = Number(String(args.message).replace(/^m/i, ''));
        const m = Number.isInteger(n) ? history[n] : undefined;
        if (!m) return { name: 'recall', output: `error: no message m${args.message} in this conversation (it has m0 to m${history.length - 1})`, is_error: true };
        const L = lines(body(m));
        const q = (args.query || '').toString().trim().toLowerCase();
        if (q) {
          const hit = L.map((l, i) => [i + 1, l]).filter(x => x[1].toLowerCase().includes(q)).slice(0, 120).map(x => String(x[0]).padStart(5) + '| ' + x[1]);
          return { name: 'recall', output: `message m${n} (${m.role}), the lines containing "${args.query}" (of ${L.length}):\n${hit.join('\n')}`, is_error: false };
        }
        const s0 = Math.max(1, Math.min(Number(args.start) || 1, L.length)), e0 = Math.max(s0, Math.min(Number(args.end) || s0 + 99, L.length));
        let out = `message m${n} (${m.role}), lines ${s0}-${e0} of ${L.length}:\n` + L.slice(s0 - 1, e0).map((l, i) => String(s0 + i).padStart(5) + '| ' + l).join('\n');
        if (e0 < L.length) out += `\n(more: ask again with start=${e0 + 1})`;
        return { name: 'recall', output: out.length > 12000 ? out.slice(0, 12000) + '\n(clipped: ask for a line range)' : out, is_error: false };
      }
      const q = String(args.query || '').trim().toLowerCase();
      const hits = [];
      history.forEach((m, i) => { lines(body(m)).forEach((l, j) => { if (hits.length < 40 && l.toLowerCase().includes(q)) hits.push(`m${i}:${j + 1}| ${l.slice(0, 200)}`); }); });
      return { name: 'recall', output: hits.length ? hits.join('\n') + '\n(read one with recall message=N, start=, end=)' : '', is_error: false, hits: hits.length };
    }
    async function executeLocalTool(name, args) {
      if (name === 'notes') return runNotesTool(args);
      if (name === 'recall' && args && typeof args === 'object') {
        const noId = args.id === undefined || args.id === null || args.id === '';
        if (args.message !== undefined && args.message !== null && args.message !== '') return recallMessages(args);
        if (noId && String(args.query || '').trim()) {
          // a text alone: the messages here, and the tool outputs of the session's log on the server
          const mine = recallMessages(args);
          const theirs = await executeLocalToolRemote(name, args);
          const parts = [mine.output && 'In the conversation:\n' + mine.output, theirs.output && !theirs.is_error && 'In earlier tool outputs:\n' + theirs.output].filter(Boolean);
          return { name, output: parts.length ? parts.join('\n\n') : `nothing earlier contains "${args.query}"`, is_error: false };
        }
      }
      return executeLocalToolRemote(name, args);
    }
    async function executeLocalToolRemote(name, args) {
      try {
        const res = await fetch('/v1/tools/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, arguments: args })
        });
        if (!res.ok) {
          return { name, output: await res.text(), is_error: true };
        }
        return await res.json();
      } catch (e) {
        return { name, output: e.message, is_error: true };
      }
    }

    // Send Message with Agent Tool Orchestration Loop
    function useSuggestion(el) {
      inputEl.value = el.textContent;
      inputEl.dispatchEvent(new Event('input'));
      inputEl.focus();
      // a suggestion is a question to ask: send it now (when the model is up and nothing is running)
      if (!isGenerating && modelState !== 'loading') sendMessage();
    }

    // The first message of a conversation reads the persona and the tool list (~960 tokens, ~11 s at prompt speed) before its own words.  The
    // server keeps the KV of the last conversation, so on a fresh server (nothing cached) a one-token request with the same system part,
    // sent while the page is idle, makes the first real message start at once.  Not when the server holds someone's conversation (a warm-up
    // would replace it).
    let warmedKey = '';
    async function warmPrompt(cachedTokens) {
      if (modelState !== 'ready' || isGenerating || history.length || cachedTokens > 0 || !memoryLoaded) return;
      if (toolsEnabled && availableTools.length === 0) return;
      const key = JSON.stringify([systemPrompt(), toolsEnabled, finishOn]);
      if (key === warmedKey) return;
      warmedKey = key;
      const msgs = [];
      const sys = systemPrompt();
      if (sys) msgs.push({ role: 'system', content: sys });
      msgs.push({ role: 'user', content: 'Hi' });
      const body = { messages: msgs, max_tokens: 1, stream: false, reasoning_effort: currentEffort,
                     chat_template_kwargs: { enable_thinking: currentEffort !== 'off' } };
      if (toolsEnabled) body.tools = requestTools();
      try {
        timingInfo.textContent = 'Warming up the prompt...';
        await fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      } catch (_) { warmedKey = ''; }
      if (timingInfo.textContent === 'Warming up the prompt...') timingInfo.textContent = '';
    }


    // Attached files: text and code, read in the browser and put in front of the message (the model reads text only: no vision support in this
    // engine, so an image is refused with a note rather than silently dropped).
    const attachments = [];
    const MAX_FILE_BYTES = 256 * 1024, MAX_ATTACH_CHARS = 150000;
    // A picture costs one token per 32 x 32 pixels: scaled down to about 600 of them (768 x 768), the sides a multiple of 32 (the vision
    // encoder's patch grid), and sent as a JPEG.
    const MAX_PICTURE_PIXELS = 768 * 768;
    async function shrinkImage(file) {
      const bmp = await createImageBitmap(file);
      let s = Math.min(1, Math.sqrt(MAX_PICTURE_PIXELS / (bmp.width * bmp.height)));
      const w = Math.max(64, Math.round(bmp.width * s / 32) * 32), h = Math.max(64, Math.round(bmp.height * s / 32) * 32);
      const cv = document.createElement('canvas');
      cv.width = w; cv.height = h;
      const g = cv.getContext('2d');
      g.fillStyle = '#fff'; g.fillRect(0, 0, w, h);
      g.drawImage(bmp, 0, 0, w, h);
      return cv.toDataURL('image/jpeg', 0.92);
    }
    let attachNotes = [];
    function attachNote(msg) { attachNotes.push(msg); timingInfo.textContent = attachNotes.join(' '); }
    function renderTray() {
      const tray = document.getElementById('attach-tray');
      tray.innerHTML = '';
      attachments.forEach((a, i) => {
        const chip = document.createElement('span');
        chip.className = 'att-chip';
        chip.innerHTML = `${a.image ? `<img src="${a.image}" style="height:1.6rem;border-radius:0.3rem;">` : '📎'} ${escapeHtml(a.name)} <small>${a.model ? '3-D model' : a.image ? 'picture' :a.text.length < 1000 ? a.text.length + ' chars' : (a.text.length / 1000).toFixed(1) + 'k chars'}</small><button title="Remove" onclick="removeAttachment(${i})">✕</button>`;
        tray.appendChild(chip);
      });
    }
    function removeAttachment(i) { attachments.splice(i, 1); renderTray(); }
    async function addFiles(files) {
      attachNotes = [];
      for (const f of files) {
        if (MODEL_EXT.test(f.name)) {
          try {
            const m = await load3D(f);
            attachments.push({ name: f.name, image: m.image, text: m.summary, model: true });
          } catch (e) { attachNote(`Could not read the 3-D file ${f.name}: ${e.message}`); }
          continue;
        }
        if (f.type.startsWith('image/')) {
          try {
            const url = await shrinkImage(f);
            attachments.push({ name: f.name, image: url, text: '' });
          } catch (_) { attachNote(`Could not read the picture ${f.name}.`); }
          continue;
        }
        if (f.size > MAX_FILE_BYTES) { attachNote(`${f.name} is ${(f.size / 1024).toFixed(0)} KB: files over ${MAX_FILE_BYTES / 1024} KB are not attached.`); continue; }
        let text;
        try { text = await f.text(); } catch (_) { attachNote(`Could not read ${f.name}.`); continue; }
        if (text.includes('\u0000')) { attachNote(`${f.name} is a binary file: only text and code can be attached.`); continue; }
        const total = attachments.reduce((n, a) => n + a.text.length, 0) + text.length;
        if (total > MAX_ATTACH_CHARS) { attachNote(`${f.name} would make the attachments over ${MAX_ATTACH_CHARS / 1000}k characters: not attached.`); continue; }
        attachments.push({ name: f.name, text });
      }
      renderTray();
      inputEl.focus();
    }
    document.getElementById('file-input').addEventListener('change', e => { addFiles(Array.from(e.target.files)); e.target.value = ''; });
    ['dragenter', 'dragover'].forEach(t => document.addEventListener(t, e => {
      if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) { e.preventDefault(); document.body.classList.add('dragging'); }
    }));
    ['dragleave', 'drop'].forEach(t => document.addEventListener(t, e => {
      if (t === 'drop' && e.dataTransfer && e.dataTransfer.files.length) { e.preventDefault(); addFiles(Array.from(e.dataTransfer.files)); }
      if (t === 'drop' || e.target === document.documentElement || !e.relatedTarget) document.body.classList.remove('dragging');
    }));
    inputEl.addEventListener('paste', e => {
      const files = Array.from((e.clipboardData && e.clipboardData.files) || []);
      if (files.length) { e.preventDefault(); addFiles(files); }
    });
    // 3-D files (STL, OBJ, STEP): the model reads pictures and text, so the page turns one into both: a contact sheet of four views (isometric,
    // front, top, right) and a geometry summary.  STL / OBJ are meshes (shaded); STEP is read for its vertices and edges (a wireframe, curved
    // edges as chords) and its header, units, product names and entity counts: there is no CAD kernel here to tessellate its surfaces.
    const MODEL_EXT = /\.(stl|obj|step|stp)$/i;
    function parseSTL(buf) {
      const dv = new DataView(buf);
      if (buf.byteLength >= 84 && 84 + 50 * dv.getUint32(80, true) === buf.byteLength) {
        const n = dv.getUint32(80, true), t = new Float32Array(n * 9);
        for (let i = 0; i < n; i++) for (let k = 0; k < 9; k++) t[i * 9 + k] = dv.getFloat32(84 + i * 50 + 12 + k * 4, true);
        return { tris: t, kind: 'binary STL' };
      }
      const v = [], re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g, s = new TextDecoder().decode(buf);
      let m; while ((m = re.exec(s))) v.push(+m[1], +m[2], +m[3]);
      return { tris: Float32Array.from(v), kind: 'ASCII STL' };
    }
    function parseOBJ(s) {
      const p = [], t = [];
      for (const line of s.split(/\r?\n/)) {
        const f = line.trim().split(/\s+/);
        if (f[0] === 'v') p.push([+f[1], +f[2], +f[3]]);
        else if (f[0] === 'f') {
          const ix = f.slice(1).map(x => { const i = parseInt(x, 10); return i < 0 ? p.length + i : i - 1; });
          for (let k = 1; k + 1 < ix.length; k++) for (const j of [ix[0], ix[k], ix[k + 1]]) t.push(...(p[j] || [0, 0, 0]));
        }
      }
      return { tris: Float32Array.from(t), kind: 'OBJ' };
    }
    function parseSTEP(s) {
      const pt = new Map(), vx = new Map(), edges = [], count = {};
      for (const m of s.matchAll(/#(\d+)\s*=\s*([A-Z_0-9]+)\s*\(/g)) count[m[2]] = (count[m[2]] || 0) + 1;
      for (const m of s.matchAll(/#(\d+)\s*=\s*CARTESIAN_POINT\s*\([^(]*\(\s*([^)]*)\)/g)) pt.set(m[1], m[2].split(',').map(Number));
      for (const m of s.matchAll(/#(\d+)\s*=\s*VERTEX_POINT\s*\([^#]*#(\d+)/g)) vx.set(m[1], pt.get(m[2]));
      for (const m of s.matchAll(/#\d+\s*=\s*EDGE_CURVE\s*\([^#]*#(\d+)\s*,\s*#(\d+)/g)) { const a = vx.get(m[1]), b = vx.get(m[2]); if (a && b) edges.push(a, b); }
      const one = re => { const m = s.match(re); return m ? m[1].trim() : ''; };
      const names = [...s.matchAll(/PRODUCT\s*\(\s*'([^']*)'/g)].map(m => m[1]).filter(x => x).slice(0, 6);
      const units = /SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\./.test(s) ? 'millimetres' : /INCH/i.test(s) ? 'inches' : /SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\./.test(s) ? 'metres' : 'not stated';
      return { pts: [...vx.values()].filter(Boolean), edges, count, names, units, schema: one(/FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'/), desc: one(/FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'/) };
    }
    function meshStats(t) {
      const lo = [1e30, 1e30, 1e30], hi = [-1e30, -1e30, -1e30];
      let vol = 0, area = 0;
      for (let i = 0; i < t.length; i += 9) {
        for (let k = 0; k < 3; k++) for (let c = 0; c < 3; c++) { const v = t[i + k * 3 + c]; if (v < lo[c]) lo[c] = v; if (v > hi[c]) hi[c] = v; }
        const [ax, ay, az, bx, by, bz, cx, cy, cz] = t.subarray(i, i + 9);
        vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
        const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
        area += Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
      }
      return { lo, hi, vol: Math.abs(vol), area };
    }
    const VIEWS = [['isometric', 35, 30], ['front', 0, 0], ['top', 0, 90], ['right', 90, 0]];
    function project(p, az, el) {
      const a = az * Math.PI / 180, e = el * Math.PI / 180, x1 = p[0] * Math.cos(a) + p[1] * Math.sin(a), y1 = -p[0] * Math.sin(a) + p[1] * Math.cos(a);
      return [x1, y1 * Math.sin(e) + p[2] * Math.cos(e), y1 * Math.cos(e) - p[2] * Math.sin(e)];
    }
    function renderSheet(model) {
      const S = 768, H = S / 2, cv = document.createElement('canvas');
      cv.width = cv.height = S;
      const g = cv.getContext('2d');
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      const tris = model.tris, step = tris && tris.length / 9 > 150000 ? Math.ceil(tris.length / 9 / 150000) : 1;
      VIEWS.forEach(([name, az, el], vi) => {
        const ox = (vi % 2) * H, oy = Math.floor(vi / 2) * H;
        g.save(); g.beginPath(); g.rect(ox, oy, H, H); g.clip();
        const list = [];
        if (tris) {
          for (let i = 0; i < tris.length; i += 9 * step) {
            const q = [0, 1, 2].map(k => project([tris[i + k * 3], tris[i + k * 3 + 1], tris[i + k * 3 + 2]], az, el));
            const nx = (q[1][1] - q[0][1]) * (q[2][2] - q[0][2]) - (q[1][2] - q[0][2]) * (q[2][1] - q[0][1]);
            const ny = (q[1][2] - q[0][2]) * (q[2][0] - q[0][0]) - (q[1][0] - q[0][0]) * (q[2][2] - q[0][2]);
            const nz = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0]);
            const len = Math.hypot(nx, ny, nz) || 1;
            list.push({ q, depth: (q[0][2] + q[1][2] + q[2][2]) / 3, shade: 0.3 + 0.7 * Math.abs((-0.4 * nx + 0.6 * ny - 0.7 * nz) / len) });
          }
          list.sort((a, b) => b.depth - a.depth);
        }
        const pts = model.pts ? model.pts.map(p => project(p, az, el)) : [];
        const all = tris ? list.flatMap(t => t.q) : pts;
        let x0 = 1e30, x1 = -1e30, y0 = 1e30, y1 = -1e30;
        for (const p of all) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
        const sc = 0.82 * H / Math.max(x1 - x0, y1 - y0, 1e-9), cx = ox + H / 2 - sc * (x0 + x1) / 2, cy = oy + H / 2 + sc * (y0 + y1) / 2;
        const X = p => cx + sc * p[0], Y = p => cy - sc * p[1];
        if (tris) {
          for (const t of list) {
            const c = Math.round(60 + 190 * t.shade);
            g.fillStyle = g.strokeStyle = `rgb(${c},${c},${Math.min(255, c + 25)})`;
            g.beginPath(); g.moveTo(X(t.q[0]), Y(t.q[0])); g.lineTo(X(t.q[1]), Y(t.q[1])); g.lineTo(X(t.q[2]), Y(t.q[2])); g.closePath(); g.fill(); g.stroke();
          }
        } else {
          g.strokeStyle = '#235'; g.lineWidth = 1.2; g.beginPath();
          for (let i = 0; i + 1 < model.edges.length; i += 2) { const a = project(model.edges[i], az, el), b = project(model.edges[i + 1], az, el); g.moveTo(X(a), Y(a)); g.lineTo(X(b), Y(b)); }
          g.stroke(); g.fillStyle = '#c22';
          for (const p of pts) g.fillRect(X(p) - 1.5, Y(p) - 1.5, 3, 3);
        }
        g.restore();
        g.fillStyle = '#000'; g.font = 'bold 18px sans-serif'; g.fillText(name, ox + 10, oy + 24);
        g.strokeStyle = '#999'; g.lineWidth = 1; g.strokeRect(ox + 0.5, oy + 0.5, H - 1, H - 1);
      });
      return cv.toDataURL('image/jpeg', 0.9);
    }
    async function load3D(f) {
      const ext = f.name.split('.').pop().toLowerCase();
      const fmt = x => Number(x.toPrecision(4));
      if (ext === 'stl' || ext === 'obj') {
        const m = ext === 'stl' ? parseSTL(await f.arrayBuffer()) : parseOBJ(await f.text());
        if (!m.tris.length) throw new Error('no triangles found');
        const s = meshStats(m.tris), d = [0, 1, 2].map(c => fmt(s.hi[c] - s.lo[c]));
        const summary = `3-D model file ${f.name} (${m.kind}): ${m.tris.length / 9} triangles. Bounding box ${d[0]} x ${d[1]} x ${d[2]} (x, y, z; the file states no units), ` +
          `from (${s.lo.map(fmt).join(', ')}) to (${s.hi.map(fmt).join(', ')}). Volume ${fmt(s.vol)}, surface area ${fmt(s.area)} (cubic / square file units). ` +
          'The attached picture shows it from four sides: isometric, front, top and right.';
        return { image: renderSheet(m), summary };
      }
      const m = parseSTEP(await f.text());
      if (!m.pts.length) throw new Error('no vertices found');
      const lo = [0, 1, 2].map(c => Math.min(...m.pts.map(p => p[c]))), hi = [0, 1, 2].map(c => Math.max(...m.pts.map(p => p[c])));
      const top = Object.entries(m.count).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([k, v]) => `${k} ${v}`).join(', ');
      const summary = `3-D model file ${f.name} (STEP${m.schema ? ', ' + m.schema : ''}). Units: ${m.units}. ${m.names.length ? 'Products: ' + m.names.join(', ') + '. ' : ''}` +
        `${m.desc ? 'Description: ' + m.desc + '. ' : ''}${m.pts.length} vertices, ${m.edges.length / 2} edges (curved edges drawn as straight chords). ` +
        `Vertex bounding box ${[0, 1, 2].map(c => fmt(hi[c] - lo[c])).join(' x ')} (x, y, z). Most frequent entities: ${top}. ` +
        'The attached picture is a wireframe of the vertices and edges from four sides: isometric, front, top and right.';
      return { image: renderSheet({ pts: m.pts, edges: m.edges }), summary };
    }

    // the message the model gets: each file as a fenced block in front of the words
    function withAttachments(text) {
      const pics = attachments.filter(a => a.image);
      const files = attachments.filter(a => a.text);
      const body = files.length ? fileBlocks(files) + '\n\n' + (text || 'Please read the attached file(s).') : (text || (pics.length ? 'Describe the picture.' : text));
      if (!pics.length) return files.length ? body : text;
      return [...pics.map(a => ({ type: 'image_url', image_url: { url: a.image } })), { type: 'text', text: body }];
    }
    function fileBlocks(list) {
      const blocks = list.map(a => `File: ${a.name}\n\`\`\`\n${a.text}\n\`\`\``).join('\n\n');
      return blocks;
    }

    // A fresh conversation: stops a running reply, forgets the history, puts the welcome card back.
    const welcomeHTML = messagesEl.innerHTML;
    function newChat() {
      if (isGenerating && currentAbortController) currentAbortController.abort();
      saveCurrent();
      currentChatId = null;
      frozenMemory = null;
      sendQueue.length = 0;
      queuePaused = false;
      renderQueue();
      renderChatList();
      history.length = 0;
      scratchpad = '';
      renderNotes();
      attachments.length = 0;
      renderTray();
      messagesEl.innerHTML = welcomeHTML;
      timingInfo.textContent = '';
      inputEl.focus();
    }

    // The server's model state from /v1/telemetry: 'loading' until the weights are in, then 'ready'.
    let modelState = 'loading';

    // An HTTP error body as a sentence a person can act on.
    function friendlyError(status, body) {
      let msg = body;
      try { msg = JSON.parse(body).error.message || body; } catch (_) {}
      if (status === 503 && /loading/i.test(msg)) return ['Shadow is still loading the model.', 'Try again when the status says Ready.'];
      if (status === 413 || /context|too long|positions/i.test(msg)) return ['The conversation is longer than the context window.', msg];
      return [`The server answered ${status}.`, msg];
    }

    let resumeOnly = false;
    let retryAction = null;
    function retryFailed() { if (retryAction && !isGenerating) retryAction(); }
    async function sendMessage() {
      const resume = resumeOnly; // Retry: ask again from the conversation as it stands, with no new message
      resumeOnly = false;
      const text = resume ? '' : inputEl.value.trim();
      if (!resume && !text && !attachments.length) return;
      if (isGenerating) { if (!resume) enqueueMessage(text); return; } // a reply is being written: this message waits its turn
      if (modelState === 'loading') {
        timingInfo.textContent = 'The model is still loading - your message is still in the box.';
        return;
      }
      const welcome = document.getElementById('welcome');
      if (welcome) welcome.closest('.msg-row').remove();
      let failed = false;

      isGenerating = true;
      sendBtn.textContent = 'Queue';
      stopBtn.style.display = 'flex';
      currentAbortController = new AbortController();
      const sentAtts = attachments.slice();
      if (!resume) {
      inputEl.value = '';
      inputEl.style.height = 'auto';

      if (history.length === 0) frozenMemory = memoryBlock();
      const sentFiles = attachments.map(a => a.name);
      history.push({ role: 'user', content: withAttachments(text) });
      attachments.length = 0;
      renderTray();

      // User Bubble
      const userRow = document.createElement('div');
      userRow.className = 'msg-row user';
      userRow.innerHTML = `<div class="avatar">U</div><div class="bubble">${escapeHtml(text).replace(/\n/g, '<br>')}${sentFiles.length ? `<div class="att-note">${sentFiles.map(n => '📎 ' + escapeHtml(n)).join(' &nbsp; ')}</div>` : ''}</div>`;
      messagesEl.appendChild(userRow);
      scrollToBottom(true);
      }

      const t0 = performance.now();
      let loopTurns = 0;
      // no limit on the number of steps (the rolling context keeps the request bounded, Stop ends it); the one guard is a model stuck on the same tool call
      const MAX_SAME_CALLS = 4;
      let lastCallSig = '', sameCalls = 0, stuck = false;
      const callCounts = new Map();   // how often each exact tool call was made in this task
      let readStreak = 0;             // read-only calls since the last notes / edit

      while (true) {
        loopTurns++;

        // Assistant Skeleton
        const asstRow = document.createElement('div');
        document.querySelectorAll('.msg-row.working').forEach(r => r.classList.remove('working'));
        asstRow.className = 'msg-row assistant working';
        asstRow.innerHTML = `
          <div class="avatar">S</div>
          <div class="bubble">
            <div class="think-block" style="display:none;">
              <div class="think-header">
                <div class="think-header-left" onclick="toggleThinkBlock(this)">
                  <span class="think-spinner"></span>
                  <span class="think-title">Thinking...</span>
                  <span class="think-timer" style="opacity:0.75; font-size:0.78rem; font-weight:normal;"></span>
                  <span class="think-chevron" style="font-size:0.75rem; opacity:0.6; display:none;">▼</span>
                </div>
                <button class="think-stop-btn" onclick="stopGeneration(event)" title="Stop thinking">
                  <span>⏹</span> Stop Thinking
                </button>
              </div>
              <div class="think-content"></div>
            </div>
            <div class="msg-content"><span class="reading-note">Reading your message...</span><span class="cursor-blink"></span></div>
          </div>`;
        messagesEl.appendChild(asstRow);
        scrollToBottom(loopTurns === 1);

        const bubbleEl = asstRow.querySelector('.bubble');
        const thinkBlock = bubbleEl.querySelector('.think-block');
        const thinkContent = bubbleEl.querySelector('.think-content');
        const msgContent = bubbleEl.querySelector('.msg-content');

        let rawContent = '';
        let rawThinking = '';
        let streamError = null;
        let cutOff = false;
        let lastTimings = null;
        let thinkStart = performance.now();
        let isThinking = false;

        const payloadMessages = [];
        const sys = systemPrompt();
        if (sys) payloadMessages.push({ role: 'system', content: sys });
        payloadMessages.push(...withNotes(buildContextMessages(history)));

        const requestBody = {
          messages: payloadMessages,
          stream: true,
          reasoning_effort: currentEffort,
          chat_template_kwargs: { enable_thinking: currentEffort !== 'off' }
        };
        Object.assign(requestBody, requestSampling());
        if (toolsEnabled && availableTools.length > 0) {
          requestBody.tools = requestTools();
        }

        try {
          const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
            signal: currentAbortController ? currentAbortController.signal : undefined
          });

          if (!response.ok) {
            const [head, detail] = friendlyError(response.status, await response.text());
            retryAction = () => { resumeOnly = true; sendMessage(); };
            msgContent.innerHTML = `<div class="msg-error">${escapeHtml(head)}<small>${escapeHtml(detail)}</small> <button class="retry-btn" onclick="retryFailed()">Retry</button></div>`;
            failed = true;
            break;
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || trimmed === 'data: [DONE]') continue;
              if (trimmed.startsWith('data: ')) {
                try {
                  const parsed = JSON.parse(trimmed.slice(6));
                  if (parsed.timings) lastTimings = parsed.timings;
                  if (parsed.choices && parsed.choices[0] && parsed.choices[0].finish_reason === 'length') cutOff = true;
                  if (parsed.error) { streamError = parsed.error.message || 'The server reported an error.'; continue; }
                  const delta = parsed.choices?.[0]?.delta;
                  if (!delta) continue;

                  if (delta.reasoning_content) {
                    if (!isThinking) {
                      isThinking = true;
                      thinkStart = performance.now();
                      thinkBlock.style.display = 'block';
                      const rn = msgContent.querySelector('.reading-note');
                      if (rn) rn.remove();
                    }
                    rawThinking += delta.reasoning_content;
                    thinkContent.textContent = rawThinking;
                    tailThinking(thinkContent);
                    const elapsed = ((performance.now() - thinkStart) / 1000).toFixed(1);
                    const timerEl = thinkBlock.querySelector('.think-timer');
                    if (timerEl) timerEl.textContent = `(${elapsed}s)`;
                  }
                  if (delta.content) {
                    if (isThinking) {
                      isThinking = false;
                      const elapsed = ((performance.now() - thinkStart) / 1000).toFixed(1);
                      const spinner = thinkBlock.querySelector('.think-spinner');
                      if (spinner) spinner.style.display = 'none';
                      const titleEl = thinkBlock.querySelector('.think-title');
                      if (titleEl) titleEl.textContent = `Thought for ${elapsed}s`;
                      const timerEl = thinkBlock.querySelector('.think-timer');
                      if (timerEl) timerEl.textContent = '';
                      const thinkStopBtn = thinkBlock.querySelector('.think-stop-btn');
                      if (thinkStopBtn) thinkStopBtn.style.display = 'none';
                      const chevron = thinkBlock.querySelector('.think-chevron');
                      if (chevron) chevron.style.display = 'inline-block';
                      if (thinkContent.style.display !== 'none') toggleThinkBlock(thinkBlock.querySelector('.think-header-left'));
                    }
                    rawContent += delta.content;
                    const cleanStream = stripToolCalls(rawContent);
                    const pendingBadge = hasPendingToolCall(rawContent)
                      ? `<div style="margin-top:0.35rem; font-size:0.8rem; color:var(--tool-text); opacity:0.88;">⚡ Preparing tool call${pendingToolCallName(rawContent) ? ` (<strong>${escapeHtml(pendingToolCallName(rawContent))}</strong>)` : ''}...</div>`
                      : '';
                    msgContent.innerHTML = (cleanStream ? renderMarkdown(cleanStream) : '') + pendingBadge + '<span class="cursor-blink"></span>';
                  }
                  scrollToBottom(false);
                } catch (_) {}
              }
            }
          }

          // the server's error mid-stream (e.g. a picture it could not encode): shown in place of the reply, the question taken back
          if (streamError) {
            thinkBlock.style.display = 'none';
            if (loopTurns === 1 && !resume) {
              // the question itself failed: take it back (it goes into the box again with Retry), as if it had not been sent
              history.pop();
              retryAction = () => { inputEl.value = text; attachments.push(...sentAtts); renderTray(); sendMessage(); };
            } else {
              // part-way through an agent turn: what was done stays in the conversation, and Retry asks for the next step again
              retryAction = () => { resumeOnly = true; sendMessage(); };
            }
            msgContent.innerHTML = `<div class="msg-error">${escapeHtml(streamError)} <button class="retry-btn" onclick="retryFailed()">Retry</button></div>`;
            failed = true;
            break;
          }

          if (isThinking) {
            isThinking = false;
            const elapsed = ((performance.now() - thinkStart) / 1000).toFixed(1);
            const spinner = thinkBlock.querySelector('.think-spinner');
            if (spinner) spinner.style.display = 'none';
            const titleEl = thinkBlock.querySelector('.think-title');
            if (titleEl) titleEl.textContent = `Thought for ${elapsed}s`;
            const timerEl = thinkBlock.querySelector('.think-timer');
            if (timerEl) timerEl.textContent = '';
            const thinkStopBtn = thinkBlock.querySelector('.think-stop-btn');
            if (thinkStopBtn) thinkStopBtn.style.display = 'none';
            const chevron = thinkBlock.querySelector('.think-chevron');
            if (chevron) chevron.style.display = 'inline-block';
            if (thinkContent.style.display !== 'none') toggleThinkBlock(thinkBlock.querySelector('.think-header-left'));
          }

          // Check if assistant called tools
          const toolCalls = parseToolCallsFromText(rawContent);
          const cleanFinal = stripToolCalls(rawContent);
          msgContent.innerHTML = cleanFinal ? renderMarkdown(cleanFinal) : '';
          msgContent.style.display = (!cleanFinal && toolCalls.length > 0) ? 'none' : '';
          // the reply ran into a length limit (the context is full): without a word the bubble would just stop
          if (cutOff) {
            const note = document.createElement('div');
            note.className = 'msg-error';
            note.innerHTML = 'The reply was cut off: the conversation has no room left. <small>Start a new chat (your notes carry over) or ask for a shorter answer.</small>';
            bubbleEl.appendChild(note);
          }
          // what this reply cost: tokens and speed, from the final chunk's timings
          if (lastTimings && lastTimings.predicted_n) {
            const t = lastTimings, meta = document.createElement('div');
            meta.className = 'msg-meta';
            meta.textContent = `${t.predicted_n} tokens at ${Number(t.predicted_per_second).toFixed(1)} tok/s` +
              (t.prompt_n ? ` · read ${t.prompt_n} tokens at ${Number(t.prompt_per_second).toFixed(0)} tok/s` : '');
            bubbleEl.appendChild(meta);
          }

          if (toolsEnabled && toolCalls.length > 0) {
            history.push({ role: 'assistant', content: rawContent });
            const callSig = JSON.stringify(toolCalls.map(tc => [tc.name, tc.arguments]));
            sameCalls = callSig === lastCallSig ? sameCalls + 1 : 1;
            lastCallSig = callSig;

            let finished = false, stoppedAtGate = false;
            for (const tc of toolCalls) {
              // finish: the status of the task as a card, no tool run; refused (a tool result the model reads) when it came with other calls or says nothing
              const fin = tc.name === 'finish' && finishActive() ? checkFinish(tc.arguments, toolCalls.length) : null;
              if (fin && fin.ok) {
                bubbleEl.appendChild(finishCardEl(fin.f));
                history.push({ role: 'tool', content: `<tool_response>\n${fin.f.question ? 'Status recorded; waiting for the user\'s answer.' : 'Status recorded; the task is closed.'}\n</tool_response>` });
                finished = true;
                scrollToBottom(false);
                continue;
              }
              const argSummary = formatToolArgsSummary(tc.arguments);
              const toolCard = document.createElement('div');
              toolCard.className = 'tool-call-block running';
              toolCard.innerHTML = `
                <div class="tool-header" onclick="toggleToolOutput(this)">
                  <span>⚡ Windows Tool: <strong>${escapeHtml(tc.name)}</strong>${argSummary ? ` <small style="opacity:0.78; font-weight:normal;">(${escapeHtml(argSummary)})</small>` : ''}</span>
                  <span style="font-size:0.75rem; opacity:0.8;">Running...</span>
                </div>
                <div class="tool-output">Executing in PowerShell...</div>`;
              bubbleEl.appendChild(toolCard);
              scrollToBottom(false);

              let result;
              if (fin) result = { name: 'finish', output: fin.output, is_error: true };
              else {
                let verdict;
                try { verdict = await permissionGate(tc, toolCard); }
                catch (e) {
                  // Stop while waiting for the user: the call gets an answer, the task ends (the abort path below would push the same reply twice)
                  if (e && e.name !== 'AbortError') throw e;
                  history.push({ role: 'tool', content: '<tool_response>\nThe user stopped the task before this call ran.\n</tool_response>' });
                  stoppedAtGate = true;
                  break;
                }
                if (verdict === 'deny') { result = deniedResult(tc); toolCard.dataset.denied = '1'; }
                else { toolCard.classList.add('running'); result = await executeLocalTool(tc.name, tc.arguments); }
              }
              // the same call again: its result is not new - say so, or a model that lost the earlier result to the shortening reads it a third and fourth time
              const callKey = tc.name + '|' + JSON.stringify(tc.arguments);
              const timesCalled = (callCounts.get(callKey) || 0) + 1;
              callCounts.set(callKey, timesCalled);
              readStreak = readOnlyStreak(tc.name, readStreak);
              const treadmill = treadmillNote(readStreak);
              if (treadmill) result.output += '\n' + treadmill;
              if (timesCalled >= 2) result.output += `
(note: this exact call was already made ${timesCalled - 1} time(s) earlier in this task and gives the same result; use the facts you noted, or read a different part.)`;
              if (tc.name.startsWith('memory_')) loadMemory();
              if (tc.name.startsWith('mcp__browser__')) browserNote(result.output);
              toolCard.classList.remove('running');
              const outputEl = toolCard.querySelector('.tool-output');
              toolCard.querySelector('.tool-header span:last-child').textContent = (toolCard.dataset.denied ? 'Denied' : result.is_error ? 'Error' : 'Complete') + ' ▾';
              outputEl.textContent = result.output;
              markClipped(outputEl);

              // Append tool response turn to history
              history.push({
                role: 'tool',
                content: `<tool_response>\n${result.output}\n</tool_response>`
              });
            }
            if (stoppedAtGate) { timingInfo.textContent = 'Generation stopped by user'; break; }
            // the status was given: the task ends here (no further model step)
            if (finished) break;
            // the same call again and again: the model is stuck, not working - stop and say so (the history keeps the results, "continue" resumes)
            if (sameCalls >= MAX_SAME_CALLS) { stuck = true; break; }
            // Loop continues to let Shadow formulate final answer using the tool output
            continue;
          } else {
            // No more tool calls; finish turn
            history.push({ role: 'assistant', content: rawContent });
            break;
          }
        } catch (err) {
          if (err.name === 'AbortError') {
            if (isThinking) {
              const elapsed = ((performance.now() - thinkStart) / 1000).toFixed(1);
              const spinner = thinkBlock.querySelector('.think-spinner');
              if (spinner) spinner.style.display = 'none';
              const titleEl = thinkBlock.querySelector('.think-title');
              if (titleEl) titleEl.textContent = `Stopped thinking after ${elapsed}s`;
              const timerEl = thinkBlock.querySelector('.think-timer');
              if (timerEl) timerEl.textContent = '';
              const thinkStopBtn = thinkBlock.querySelector('.think-stop-btn');
              if (thinkStopBtn) thinkStopBtn.style.display = 'none';
            }
            const cleanAborted = stripToolCalls(rawContent);
            msgContent.innerHTML = cleanAborted ? renderMarkdown(cleanAborted) : '<span style="opacity:0.65; font-size:0.85rem;">Stopped.</span>';
            const cursor = msgContent.querySelector('.cursor-blink');
            if (cursor) cursor.remove();
            timingInfo.textContent = 'Generation stopped by user';
            if (rawContent.trim()) {
              history.push({ role: 'assistant', content: rawContent });
            }
            break;
          }
          retryAction = () => { resumeOnly = true; sendMessage(); };
          msgContent.innerHTML = `<div class="msg-error">Lost the connection to the server.<small>${escapeHtml(err.message)} - is shadow serve still running?</small> <button class="retry-btn" onclick="retryFailed()">Retry</button></div>`;
          failed = true;
          break;
        }
      }

      const dt = (performance.now() - t0) / 1000;
      // stopped because the model kept making the same tool call (the history ends with tool results, no answer): say so instead of ending silently
      const hitStuck = !failed && stuck;
      if (hitStuck) {
        const capRow = document.createElement('div');
        capRow.className = 'msg-row assistant';
        capRow.innerHTML = `<div class="avatar">S</div><div class="bubble"><div class="msg-content"><span style="opacity:0.8;">Stopped: the same tool call ${MAX_SAME_CALLS} times in a row, with nothing new. Tell me what to try instead, or send "continue" to let it go on.</span></div></div>`;
        messagesEl.appendChild(capRow);
        scrollToBottom(false);
      }
      if (failed) {
        timingInfo.textContent = 'Failed - nothing was generated';
      } else if (hitStuck) {
        timingInfo.textContent = `Stopped after ${dt.toFixed(2)}s: the same tool call ${MAX_SAME_CALLS} times in a row`;
      } else if (timingInfo.textContent !== 'Generation stopped by user') {
        timingInfo.textContent = `Completed in ${dt.toFixed(2)}s`;
      }
      isGenerating = false;
      currentAbortController = null;
      document.querySelectorAll('.msg-row.working, .tool-call-block.running').forEach(r => r.classList.remove('working', 'running'));
      saveCurrent();
      sendBtn.textContent = 'Send';
      sendBtn.style.display = 'flex';
      stopBtn.style.display = 'none';
      sendBtn.disabled = false;
      inputEl.focus();
      pollTelemetry();
      runQueue();
    }

    // Telemetry Polling Safe Setters
    function setTxt(id, val) {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    }
    function setHTML(id, val) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = val;
    }
    function setWidth(id, val) {
      const el = document.getElementById(id);
      if (el) el.style.width = val;
    }

    function formatBytes(b) {
      if (!b || b <= 0) return '0 B';
      const gib = b / (1024 * 1024 * 1024);
      if (gib >= 1.0) return gib.toFixed(2) + ' GiB';
      const mib = b / (1024 * 1024);
      return mib.toFixed(1) + ' MiB';
    }

    async function pollTelemetry() {
      try {
        const res = await fetch('/v1/telemetry');
        if (!res.ok) return;
        const d = await res.json();
        // a drawing bug on the Dashboard must not read as "Server offline" (it did: a <div> given to spark() threw here and the catch below said the server was gone)
        try { dashPush(d); } catch (err) { console.error('dashboard render failed:', err); }
        offlineCount = 0;
        // the engine is gone while a reply is awaited: say so instead of throbbing for ever
        if (isGenerating && d.process && d.process.state === 'failed') markStalled("Shadow's engine stopped (it failed). Restart Shadow, then send the message again.");

        // Header pills
        if (d.process) {
          // while the prompt is read (no token yet), say how far along it is
          const note = document.querySelector('#messages .msg-row.assistant:last-child .reading-note');
          if (note && isGenerating && d.process.prefill_total > 0) {
            const left = d.process.prefill_total - d.process.prefill_done;
            note.textContent = left > 0
              ? `Reading your message: ${d.process.prefill_done} / ${d.process.prefill_total} tokens...`
              : 'Thinking...';
          }
          modelState = d.process.state === 'loading' ? 'loading' : 'ready';
          if (modelState === 'ready' && !d.process.busy) warmPrompt(d.process.cached_tokens || 0);
          const [label, cls] = modelState === 'loading' ? ['Loading model...', 'loading']
            : d.process.busy ? ['Generating', 'busy'] : ['Ready', 'ready'];
          statusPill.textContent = label;
          statusPill.className = 'pill ' + cls;
          if (!isGenerating) {
            sendBtn.disabled = modelState === 'loading';
            inputEl.placeholder = modelState === 'loading'
              ? 'Shadow is loading the model - you can type, sending opens when it is ready'
              : 'Message Shadow (Enter to send, Shift+Enter for a new line)';
          }
          if (d.process.tok_per_sec > 0) {
            speedPill.innerHTML = `Speed: <strong>${d.process.tok_per_sec.toFixed(1)} tok/s</strong>`;
          }
        }
        if (d.gpu) {
          const usedG = (d.gpu.vram_used_bytes / (1024**3)).toFixed(1);
          const totalG = (d.gpu.vram_total_bytes / (1024**3)).toFixed(1);
          gpuHeaderPill.innerHTML = `GPU: <strong>${d.gpu.utilization_percent.toFixed(0)}%</strong> • VRAM: <strong>${usedG}/${totalG} GiB</strong>`;
        }

      } catch (err) {
        statusPill.textContent = 'Server offline';
        if (isGenerating && ++offlineCount >= 3) markStalled('Lost the connection to Shadow: is it still running?');
        statusPill.className = 'pill offline';
      }
    }

    // ---- saved conversations (the sidebar): in this browser's localStorage, pictures included until the space runs out
    const CHATS_KEY = 'shadow_chats_v1';
    let chats = [];
    try { chats = JSON.parse(localStorage.getItem(CHATS_KEY) || '[]'); } catch (_) { chats = []; }
    let currentChatId = null;
    function stripPics(m) {
      if (!Array.isArray(m.content)) return m;
      return { ...m, content: m.content.map(p => p.type === 'image_url' ? { type: 'text', text: '[picture removed to save space]' } : p) };
    }
    function persistChats() {
      for (let tries = 0; tries < 40; tries++) {
        try { localStorage.setItem(CHATS_KEY, JSON.stringify(chats)); return; } catch (_) {
          // out of space: drop the pictures of the oldest conversation that has any, else the oldest conversation
          const old = chats.filter(c => c.id !== currentChatId).sort((a, b) => a.ts - b.ts);
          const withPics = old.find(c => c.history.some(m => Array.isArray(m.content) && m.content.some(p => p.type === 'image_url')));
          if (withPics) withPics.history = withPics.history.map(stripPics);
          else if (old.length) chats.splice(chats.indexOf(old[0]), 1);
          else return;
        }
      }
    }
    function messageText(m) {
      return Array.isArray(m.content) ? m.content.filter(p => p.type === 'text').map(p => p.text).join(' ') : String(m.content || '');
    }
    function chatTitle(h) {
      const u = h.find(m => m.role === 'user');
      let t = u ? messageText(u) : '';
      const i = t.lastIndexOf('```');
      if (t.startsWith('File: ') && i >= 0) t = t.slice(i + 3);
      t = t.replace(/\s+/g, ' ').trim();
      return t ? (t.length > 52 ? t.slice(0, 50) + '...' : t) : 'New chat';
    }
    function saveCurrent() {
      if (!history.length) return;
      let c = chats.find(x => x.id === currentChatId);
      if (!c) { c = { id: 'c' + Date.now().toString(36), title: '', ts: 0, history: [] }; chats.push(c); currentChatId = c.id; }
      c.history = JSON.parse(JSON.stringify(history));
      c.notes = scratchpad;
      c.ts = Date.now();
      c.title = chatTitle(c.history);
      persistChats();
      renderChatList();
    }
    function dayLabel(ts) {
      const s = x => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
      const diff = Math.round((s(new Date()) - s(new Date(ts))) / 864e5);
      return diff <= 0 ? 'Today' : diff === 1 ? 'Yesterday' : diff < 7 ? 'This week' : 'Earlier';
    }
    function renderChatList() {
      const box = document.getElementById('chat-list');
      if (!box) return;
      const q = (document.getElementById('chat-search').value || '').toLowerCase();
      const list = chats.filter(c => !q || c.title.toLowerCase().includes(q) || c.history.some(m => messageText(m).toLowerCase().includes(q))).sort((a, b) => b.ts - a.ts);
      if (!list.length) { box.innerHTML = `<div class="side-empty">${q ? 'No conversation matches.' : 'Your conversations will be listed here.'}</div>`; return; }
      let last = '', html = '';
      for (const c of list) {
        const g = dayLabel(c.ts);
        if (g !== last) { html += `<div class="side-group">${g}</div>`; last = g; }
        html += `<div class="chat-item${c.id === currentChatId ? ' active' : ''}" onclick="openChat('${c.id}')"><span class="t">${escapeHtml(c.title || 'New chat')}</span>` +
                `<button class="x" title="Delete" onclick="event.stopPropagation(); deleteChat('${c.id}')">✕</button></div>`;
      }
      box.innerHTML = html;
      if (typeof dashChats === 'function') dashChats();
    }
    function unwrapToolResponse(text) {
      return String(text || '')
        .trim()
        .replace(/^<tool_response>\s*/i, '')
        .replace(/\s*<\/tool_response>$/i, '')
        .trim();
    }
    function renderSaved(m, idx, all) {
      if (m.role === 'tool') return;
      const row = document.createElement('div');
      row.className = 'msg-row ' + (m.role === 'user' ? 'user' : 'assistant');
      if (m.role === 'user') {
        const txt = messageText(m);
        const pics = Array.isArray(m.content) ? m.content.filter(p => p.type === 'image_url') : [];
        const names = [...txt.matchAll(/^File: (.+)$/gm)].map(x => x[1]);
        const shown = names.length ? txt.slice(txt.lastIndexOf('```') + 3).trim() : txt;
        row.innerHTML = `<div class="avatar">U</div><div class="bubble">${escapeHtml(shown).replace(/\n/g, '<br>')}` +
          pics.map(p => `<div class="att-note"><img src="${p.image_url.url}" style="max-height:120px;border-radius:0.4rem"></div>`).join('') +
          (names.length ? `<div class="att-note">${names.map(n => '📎 ' + escapeHtml(n)).join(' &nbsp; ')}</div>` : '') + '</div>';
      } else {
        const raw = messageText(m);
        const toolCalls = parseToolCallsFromText(raw);
        const clean = stripToolCalls(raw);
        // Skip empty assistant entries (e.g. a turn stopped before any content or tool call was produced)
        if (!clean && !toolCalls.length) return;
        let cardsHtml = '';
        if (toolCalls.length && Array.isArray(all)) {
          let tCursor = (idx || 0) + 1;
          for (const tc of toolCalls) {
            let toolOut = '';
            if (tCursor < all.length && all[tCursor] && all[tCursor].role === 'tool') {
              toolOut = unwrapToolResponse(messageText(all[tCursor]));
              tCursor++;
            }
            // a finish that was accepted shows as its status card (a refused one, an error result, as the tool card it was)
            if (tc.name === 'finish' && toolCalls.length === 1 && checkFinish(tc.arguments, 1).ok && !/^error:/.test(toolOut)) {
              cardsHtml += finishCardHtml(parseFinish(tc.arguments));
              continue;
            }
            const argSummary = formatToolArgsSummary(tc.arguments);
            cardsHtml += `
              <div class="tool-call-block">
                <div class="tool-header" onclick="toggleToolOutput(this)">
                  <span>⚡ Windows Tool: <strong>${escapeHtml(tc.name)}</strong>${argSummary ? ` <small style="opacity:0.78; font-weight:normal;">(${escapeHtml(argSummary)})</small>` : ''}</span>
                  <span style="font-size:0.75rem; opacity:0.8;">Complete ▾</span>
                </div>
                <div class="tool-output">${escapeHtml(toolOut || '(tool output recorded)')}</div>
              </div>`;
          }
        }
        const proseHtml = clean ? `<div class="msg-content">${renderMarkdown(clean)}</div>` : `<div class="msg-content" style="display:none;"></div>`;
        row.innerHTML = `<div class="avatar">S</div><div class="bubble">${proseHtml}${cardsHtml}</div>`;
      }
      messagesEl.appendChild(row);
      if (truncOn) row.querySelectorAll('.tool-output').forEach(markClipped);
    }
    function openChat(id) {
      const c = chats.find(x => x.id === id);
      if (!c) return;
      if (isGenerating && currentAbortController) currentAbortController.abort();
      saveCurrent();
      currentChatId = id;
      frozenMemory = null;
      history.length = 0;
      history.push(...JSON.parse(JSON.stringify(c.history)));
      scratchpad = c.notes || '';
      renderNotes();
      attachments.length = 0;
      renderTray();
      messagesEl.innerHTML = '';
      history.forEach((m, i) => renderSaved(m, i, history));
      messagesEl.scrollTop = messagesEl.scrollHeight;
      timingInfo.textContent = '';
      switchView('chat');
      renderChatList();
    }
    function openLastChat() {
      const c = chats.slice().sort((a, b) => b.ts - a.ts)[0];
      if (c) openChat(c.id); else switchView('chat');
    }
    function deleteChat(id) {
      chats = chats.filter(c => c.id !== id);
      if (id === currentChatId) { currentChatId = null; history.length = 0; messagesEl.innerHTML = welcomeHTML; }
      persistChats();
      renderChatList();
    }
    function toggleSidebar() {
      document.body.classList.toggle('side-closed');
      localStorage.setItem('shadow_side_closed', document.body.classList.contains('side-closed') ? '1' : '0');
    }
    if (localStorage.getItem('shadow_side_closed') === '1') document.body.classList.add('side-closed');
    renderChatList();

    // ---- the dashboard: what the engine is doing, from /v1/telemetry (every second) and its log of recent requests
    const dashHist = { gpu: [], ram: [], cpu: [], cpuSys: [] };
    let lastTel = null;
    function dashPush(d) {
      const keep = (a, v) => { a.push(v); if (a.length > 120) a.shift(); };
      keep(dashHist.gpu, d.gpu ? d.gpu.utilization_percent : 0);
      keep(dashHist.ram, d.ram ? d.ram.system_available_bytes / 2 ** 30 : 0);
      keep(dashHist.cpu, d.cpu ? d.cpu.process_percent : 0);
      keep(dashHist.cpuSys, d.cpu ? d.cpu.system_percent : 0);
      lastTel = d;
      if (d.process && d.process.n_ctx) serverCtx = d.process.n_ctx;
      if (currentView === 'dash') dashRender(d);
    }
    let offlineCount = 0;
    // the reply being awaited will never come: stop the throb, mark the bubble, say why
    function markStalled(why) {
      const row = document.querySelector('.msg-row.assistant.working');
      if (!row) return;
      row.classList.remove('working');
      row.classList.add('stalled');
      const c = row.querySelector('.msg-content');
      if (c && !c.textContent.trim()) c.innerHTML = `<div class="msg-error">${escapeHtml(why)}</div>`;
      else row.querySelector('.bubble').insertAdjacentHTML('beforeend', `<div class="msg-error">${escapeHtml(why)}</div>`);
      timingInfo.textContent = why;
    }
    function dashRefresh() { if (lastTel) dashRender(lastTel); }
    function spark(id, vals, max, color, bars, overlay) {
      const cv = document.getElementById(id);
      if (!cv || !cv.getContext || !cv.clientWidth) return;
      const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
      if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      const g = cv.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (!overlay) g.clearRect(0, 0, w, h);
      g.strokeStyle = 'rgba(132,146,166,0.18)';
      g.beginPath(); g.moveTo(0, h - 0.5); g.lineTo(w, h - 0.5); g.stroke();
      if (!vals.length) return;
      const m = max || Math.max(...vals, 1e-9);
      if (bars) {
        const bw = w / Math.max(vals.length, 12);
        vals.forEach((v, i) => { const bh = Math.max(1, v / m * (h - 4)); g.fillStyle = color; g.fillRect(i * bw + 1, h - bh, bw - 2, bh); });
        return;
      }
      const x = i => vals.length < 2 ? w : i / (vals.length - 1) * w, y = v => h - 2 - Math.min(1, v / m) * (h - 6);
      g.beginPath();
      vals.forEach((v, i) => i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v)));
      g.strokeStyle = color; g.lineWidth = 1.6; g.stroke();
      g.lineTo(w, h); g.lineTo(x(0), h); g.closePath();
      g.fillStyle = color + '22'; g.fill();
    }
    function ago(sec) {
      const s = Math.max(0, Math.round(Date.now() / 1000 - sec));
      return s < 60 ? s + ' s ago' : s < 3600 ? Math.round(s / 60) + ' min ago' : Math.round(s / 3600) + ' h ago';
    }
    function dashChats() {
      const box = document.getElementById('d-chats');
      if (!box) return;
      const top = chats.slice().sort((a, b) => b.ts - a.ts);
      document.getElementById('d-chatcount').textContent = chats.length + ' saved';
      box.className = top.length ? '' : 'dash-empty';
      box.innerHTML = top.length ? top.slice(0, 6).map(c => `<div class="chat-item" onclick="openChat('${c.id}')"><span class="t">${escapeHtml(c.title)}</span><small>${dayLabel(c.ts)}</small></div>`).join('') : 'No saved conversations yet.';
    }
    function dashRender(d) {
      const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
      const html = (id, v) => { const e = document.getElementById(id); if (e) e.innerHTML = v; };
      const wid = (id, pct) => { const e = document.getElementById(id); if (e) e.style.width = Math.max(0, Math.min(100, pct)) + '%'; };
      const gib = b => (b / 2 ** 30).toFixed(1), p = d.process || {}, gpu = d.gpu || {}, ram = d.ram || {}, cpu = d.cpu || {}, rec = d.recent || [];
      set('d-state', p.state === 'loading' ? 'loading the model' : p.busy ? 'generating' : 'ready');
set('d-model', p.model || '--');
      // Speculation card: the acceptance rate is the number that says whether speculation helps or just burns compute
      const spec = d.spec || {}, acc = spec.acceptance_rate || 0;
      set('d-acc', acc.toFixed(1) + ' %');
      const dacc = document.getElementById('d-acc');
      if (dacc) dacc.style.color = acc < 30 ? '#ef4444' : acc < 60 ? '#f59e0b' : '';
      const ac = (window.__specAccHist || (window.__specAccHist = []));
      if (!ac.length || ac[ac.length - 1] !== acc) { ac.push(acc); if (ac.length > 40) ac.shift(); }
      spark('c-acc', ac, 100, acc >= 60 ? '#22c55e' : acc >= 30 ? '#f59e0b' : '#ef4444');
      html('d-specnums', 'drafted ' + (spec.drafted || 0).toLocaleString() + ' · accepted ' + (spec.accepted || 0).toLocaleString() + ' · lookup-only passes ' + (spec.lookup_passes || 0).toLocaleString());
      set('d-specpasses', (spec.passes || 0).toLocaleString());
      set('d-specperpass', (spec.mean_accepted_per_pass || 0).toFixed(2));
      const spos = spec.positions || [];
      html('d-specpos', spos.length ? spos.map((q, i) => '<div class="kv"><span>position ' + (i + 1) + '</span><b>' + (q.tried ? (q.right / q.tried * 100).toFixed(1) : '0.0') + ' % <small>(' + q.right + '/' + q.tried + ')</small></b></div>').join('') : '<div class="dash-empty">No speculation data yet.</div>');
      spark('d-specposbars', spos.map(q => q.tried ? q.right / q.tried * 100 : 0), 100, '#8b5cf6', true);
      const up = p.uptime_s || 0;
      set('d-uptime', up >= 3600 ? Math.floor(up / 3600) + ' h ' + Math.floor(up % 3600 / 60) + ' min' : Math.floor(up / 60) + ' min ' + Math.floor(up % 60) + ' s');
      set('d-spec', p.dflash_active ? 'On (DFlash + lookup)' : p.mtp_active ? 'On (MTP head)' : 'Prompt lookup only');
      set('d-reqs', (p.requests || 0).toLocaleString() + ' (' + (p.completion_tokens || 0).toLocaleString() + ' tokens written)');
      set('d-ctx', (p.session_tokens || 0).toLocaleString() + ' / ' + (p.n_ctx || 0).toLocaleString() + ' tokens');
      wid('d-ctxbar', p.n_ctx ? 100 * (p.session_tokens || 0) / p.n_ctx : 0);
      const tps = rec.map(r => r.predicted_ms > 0 ? r.completion_tokens / (r.predicted_ms / 1000) : 0);
      const last = rec[rec.length - 1];
      html('d-tps', (tps.length ? tps[tps.length - 1].toFixed(1) : '--') + '<small>tok/s</small>');
      spark('c-tps', tps, 0, '#6366f1', true);
      const tail = rec.slice(-10), tt = tail.reduce((a, r) => a + r.completion_tokens, 0), ts = tail.reduce((a, r) => a + r.predicted_ms / 1000, 0);
      const mcp = d.mcp || [], mcpTools = mcp.reduce((n, m) => n + m.tools, 0);
      set('d-toolcount', (8 + mcpTools) + ' available (' + mcpTools + ' from MCP)');
      html('d-mcp', mcp.length ? mcp.map(m => `<div class="kv"><span>MCP server <b>${escapeHtml(m.name)}</b></span><b style="color:${m.error ? 'var(--danger)' : 'var(--success)'}" title="${escapeHtml(m.error || '')}">${m.error ? 'failed' : m.tools + ' tools'}</b></div>`).join('')
        : 'No MCP servers. List them in mcp.json (Claude Desktop / Claude Code format) in the Shadow folder or %APPDATA%\\Shadow, then restart.');
      set('d-tps-avg', ts > 0 ? (tt / ts).toFixed(1) + ' tok/s (last ' + tail.length + ')' : '--');
      const read = last ? last.prompt_tokens - last.cached_tokens : 0;
      set('d-pps', last && last.prompt_ms > 0 && read > 0 ? (read / (last.prompt_ms / 1000)).toFixed(0) + ' tok/s, ' + read + ' new' : last ? 'all ' + last.prompt_tokens + ' cached' : '--');
      set('d-gpuname', gpu.name || '--');
      html('d-gpu', (gpu.utilization_percent != null ? Math.round(gpu.utilization_percent) : '--') + '<small>% busy</small>');
      spark('c-gpu', dashHist.gpu, 100, '#10b981');
      set('d-vram', gib(gpu.vram_used_bytes || 0) + ' / ' + gib(gpu.vram_total_bytes || 0) + ' GiB');
      const vramPct = gpu.vram_total_bytes ? 100 * gpu.vram_used_bytes / gpu.vram_total_bytes : 0;
      wid('d-vrambar', vramPct);
      const gpuBig = document.getElementById('d-gpu');
      if (gpuBig) gpuBig.style.color = vramPct > 85 ? 'var(--danger)' : vramPct > 75 ? 'var(--warning)' : '';
      set('d-temp', gpu.temperature_c ? gpu.temperature_c + ' °C' : '--');
      set('d-memctl', (gpu.memory_controller_percent || 0).toFixed(0) + '%');
      set('d-vramfree', gib(gpu.vram_free_bytes || 0) + ' GiB');
      set('d-hitmiss', (gpu.expert_cache_hits || 0).toLocaleString() + ' / ' + (gpu.expert_cache_misses || 0).toLocaleString());
      set('d-servedn', (ram.ram_served_hits || 0).toLocaleString() + ' / ' + (ram.file_served_misses || 0).toLocaleString());
      set('d-ramused', gib(ram.system_used_bytes || 0) + ' GiB (' + (ram.system_load_percent || 0) + '%)');
      set('d-pid', p.pid || '--');
set('d-hit', (gpu.expert_cache_hit_rate || 0).toFixed(1) + '%');
      const hitBig = document.getElementById('d-hit');
      if (hitBig) hitBig.style.color = (gpu.expert_cache_hit_rate || 0) < 60 ? 'var(--danger)' : (gpu.expert_cache_hit_rate || 0) < 80 ? 'var(--warning)' : '';
      wid('d-hitbar', gpu.expert_cache_hit_rate || 0);
      set('d-pin', gib(ram.pinned_tier_bytes || 0) + ' GiB (' + (ram.pinned_tier_experts || 0).toLocaleString() + ' experts)');
      const rs = ram.ram_served_hits || 0, fs = ram.file_served_misses || 0, tot = rs + fs;
      set('d-served', tot ? Math.round(100 * rs / tot) + '% RAM / ' + Math.round(100 * fs / tot) + '% drive' : '--');
      wid('d-ramserved', tot ? 100 * rs / tot : 0);
      wid('d-fileserved', tot ? 100 * fs / tot : 0);
      set('d-slots', (gpu.expert_cache_slots || 0).toLocaleString() + ' slots in VRAM');
html('d-ramfree', gib(ram.system_available_bytes || 0) + '<small>GiB free</small>');
      const ramBig = document.getElementById('d-ramfree');
      if (ramBig) ramBig.style.color = (ram.system_load_percent || 0) > 85 ? 'var(--danger)' : (ram.system_load_percent || 0) > 75 ? 'var(--warning)' : '';
      set('d-ramtag', gib(ram.system_total_bytes || 0) + ' GiB total');
      spark('c-ram', dashHist.ram, 0, '#f59e0b');
      set('d-ws', gib(ram.process_working_set_bytes || 0) + ' GiB');
      html('d-cpusys', Math.round(cpu.system_percent || 0) + '<small>% busy (whole PC)</small>');
      const ci = cpu.info || {};
      const short = (ci.name || '').replace(/\(R\)|\(TM\)/g, '').replace(/^\d+(st|nd|rd|th) Gen Intel\s*/i, '').replace(/\s*@.*$/, '').replace(/\s+/g, ' ').trim();
      set('d-cputag', (short || 'CPU') + ' - ' + (cpu.logical_cores || '?') + ' threads');
      set('d-cpuclk', ci.current_mhz ? (ci.current_mhz / 1000).toFixed(2) + ' GHz (base ' + (ci.base_mhz / 1000).toFixed(1) + ')' : ci.base_mhz ? 'base ' + (ci.base_mhz / 1000).toFixed(1) + ' GHz' : '--');
      const kb = x => x ? (x.kb >= 1024 ? x.kb / 1024 + ' MB' : x.kb + ' KB') : '?';
      const cs = ci.caches || [], lv = (l, k) => cs.find(x => x.level === l && (k === undefined || x.kind === k));
      if (cs.length) {
        const ce = document.getElementById('d-cpucache');
        ce.textContent = 'L1 ' + kb(lv(1, 2)) + '+' + kb(lv(1, 1)) + ', L2 ' + kb(lv(2)) + ', L3 ' + kb(lv(3));
        ce.title = 'L1 data + instruction and L2 are per core (' + ((lv(2) || {}).count || '?') + ' cores); L3 ' + kb(lv(3)) + ' is shared by all of them.';
      }
      const zones = ci.thermal_zones || [], te = document.getElementById('d-cputemp');
      te.textContent = zones.length ? zones.map(z => z.celsius.toFixed(0) + ' °C').join(', ') + ' (board)' : 'not exposed';
      te.title = 'Windows reports only the board\'s ACPI thermal zone here; the CPU die temperature needs a hardware driver (HWiNFO, LibreHardwareMonitor), which Shadow does not load.';
      spark('c-cpu', dashHist.cpuSys, 100, '#38bdf8');
      spark('c-cpu', dashHist.cpu, 100, '#a78bfa', false, true);
      set('d-cpuproc', (cpu.process_percent || 0).toFixed(1) + '% of the PC');
      set('d-cpusys2', (cpu.system_percent || 0).toFixed(1) + '%');
      set('d-cpucores', (cpu.process_cores_used || 0).toFixed(1) + ' of ' + (cpu.logical_cores || '?'));
      set('dash-sub', (p.state === 'loading' ? 'Loading the model...' : 'Shadow is running locally on this PC.') + (p.queued ? ' ' + p.queued + ' request(s) waiting.' : ''));
      const tb = document.getElementById('dash-tools');
      if (tb) tb.textContent = 'Windows tools: ' + (toolsEnabled ? 'ON' : 'OFF');
      const box = document.getElementById('d-recent');
      if (!rec.length) { box.className = 'dash-empty'; box.textContent = 'Nothing answered yet.'; }
      else {
        box.className = '';
        box.innerHTML = '<table class="recent"><tr><th>When</th><th>Prompt</th><th>Read at</th><th>Written</th><th>Decode</th><th>Ended</th></tr>' +
          rec.slice().reverse().slice(0, 12).map(r => {
            const nr = r.prompt_tokens - r.cached_tokens;
            return `<tr><td>${ago(r.at)}${r.pictures ? ' 🖼' : ''}</td><td><b>${r.prompt_tokens}</b> (${r.cached_tokens} cached)</td>` +
              `<td>${nr > 0 && r.prompt_ms > 0 ? Math.round(nr / (r.prompt_ms / 1000)) + ' tok/s' : '-'}</td><td><b>${r.completion_tokens}</b> tokens</td>` +
              `<td>${r.predicted_ms > 0 ? (r.completion_tokens / (r.predicted_ms / 1000)).toFixed(1) + ' tok/s' : '-'}</td><td>${r.finish}</td></tr>`;
          }).join('') + '</table>';
}
      {
        const sp = rec.filter(r => r.predicted_ms > 0 && r.completion_tokens > 0).map(r => 1000 * r.completion_tokens / r.predicted_ms).sort((a, b) => a - b);
        const pct = f => sp.length ? sp[Math.min(sp.length - 1, Math.floor(f * sp.length))] : 0;
        set('d-p50p95', sp.length ? pct(0.5).toFixed(1) + ' / ' + pct(0.95).toFixed(1) + ' tok/s, ' + sp.length + ' repl' + (sp.length === 1 ? 'y' : 'ies') : '--');
      }
      dashChats();
    }
    // ---- Settings: the default persona, and the memory (notes kept on the server, read at the start of every chat)
    let memories = [];
    let memoryOn = localStorage.getItem('shadow_memory_on') !== 'false';
    let memoryLoaded = false;
    // the notes a chat starts with are fixed for that chat: a note saved in the middle must not change the start of the prompt (the server would read the chat again)
    let frozenMemory = null;
    const MEMORY_HINT = 'You have memory tools (memory_save, memory_list, memory_forget). When the user tells you something lasting about themselves, their work or how they like things done, ' +
      'or asks you to remember something, save it as one short note. Never save passwords, keys or other private data. Do not announce every note you save.';
    function memoryBlock() {
      if (!memoryOn) return '';
      const lines = [];
      let size = 0;
      for (const m of memories.slice().sort((a, b) => b.at - a.at)) {
        if (size + m.text.length > 2400) break;
        lines.push('- ' + m.text);
        size += m.text.length;
      }
      const notes = lines.length ? 'What you remember about the user, from earlier chats (use it naturally, do not recite it, ignore what does not apply):\n' + lines.join('\n') : '';
      return [notes, toolsEnabled ? MEMORY_HINT : ''].filter(Boolean).join('\n\n');
    }
    function systemPrompt() {
      // a saved persona that still carries the old one-piece text has the mechanics in it already
      const rules = toolsEnabled && !(currentPersona || '').includes('Your context window is small.') ? ENGINE_RULES : '';
      return [(currentPersona || '').trim(), rules, frozenMemory !== null ? frozenMemory : memoryBlock(), finishActive() ? FINISH_HINT : ''].filter(Boolean).join('\n\n');
    }
    async function loadMemory() {
      try { const r = await fetch('/v1/memory'); if (r.ok) memories = await r.json(); } catch (_) {}
      memoryLoaded = true;
      renderMemory();
    }
    async function memCall(method, url, body) {
      const note = document.getElementById('set-learn-note');
      try {
        const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
        if (r.ok) { memories = await r.json(); renderMemory(); return true; }
        const j = await r.json().catch(() => ({}));
        note.textContent = (j.error && j.error.message) || 'The server refused that.';
      } catch (e) { note.textContent = 'Could not reach the server.'; }
      return false;
    }
    function renderMemory() {
      const box = document.getElementById('set-mem-list');
      if (!box) return;
      document.getElementById('set-mem-count').textContent = memories.length + ' of 60 notes';
      box.innerHTML = memories.length ? memories.slice().sort((a, b) => b.at - a.at).map(m =>
        `<div class="mem-item"><span class="badge">${escapeHtml(m.kind)}</span><span class="mem-text" title="Click to edit" onclick="editMemory(${m.id}, this)">${escapeHtml(m.text)}</span>` +
        `<button title="Forget this" onclick="deleteMemory(${m.id})">✕</button></div>`).join('')
        : '<div class="dash-empty">Nothing remembered yet. Tell Shadow something lasting ("remember that I ..."), add a note below, or use Learn from my past chats.</div>';
    }
    function editMemory(id, el) {
      const old = el.textContent, inp = document.createElement('input');
      inp.value = old; inp.className = 'mem-edit';
      el.replaceWith(inp); inp.focus();
      let done = false;
      const finish = async save => {
        if (done) return;
        done = true;
        if (save && inp.value.trim() && inp.value.trim() !== old) await memCall('PUT', '/v1/memory/' + id, { text: inp.value });
        else renderMemory();
      };
      inp.onkeydown = e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          if (e.isComposing || e.keyCode === 229) return;
          finish(true);
        }
        if (e.key === 'Escape') finish(false);
      };
      inp.onblur = () => finish(true);
    }
    function deleteMemory(id) { memCall('DELETE', '/v1/memory/' + id); }
    async function addMemory() {
      const inp = document.getElementById('set-mem-new');
      if (!inp.value.trim()) return;
      if (await memCall('POST', '/v1/memory', { text: inp.value, kind: document.getElementById('set-mem-kind').value })) { inp.value = ''; document.getElementById('set-learn-note').textContent = ''; }
    }
    function setMemoryOn(on) { memoryOn = on; localStorage.setItem('shadow_memory_on', on); }
    // ---- sampling (Settings > Sampling): the parameters sent with every request
    const SAMPLING_PRESETS = {
      model: { temperature: 1.0, top_k: 20, top_p: 0.95 },
      focused: { temperature: 0.6, top_k: 20, top_p: 0.95 },
      greedy: { temperature: 0 },
    };
    const SAMPLING_FIELDS = ['temperature', 'top_k', 'top_p', 'min_p', 'presence_penalty', 'repeat_penalty', 'seed'];
    let sampling = { preset: 'focused', custom: { temperature: 0.6, top_k: 20, top_p: 0.95 } }; // Focused is the default: STATUS 121 (the model default lost tasks)
    try { const s0 = JSON.parse(localStorage.getItem('shadow_sampling') || 'null'); if (s0 && s0.preset) sampling = s0; } catch (_) {}
    function requestSampling() {
      if (sampling.preset !== 'custom') return Object.assign({}, SAMPLING_PRESETS[sampling.preset] || SAMPLING_PRESETS.model);
      const o = {};
      for (const k of SAMPLING_FIELDS) { const v = sampling.custom[k]; if (v !== undefined && v !== null && v !== '' && !isNaN(Number(v))) o[k] = Number(v); }
      return o;
    }
    function saveSampling() { localStorage.setItem('shadow_sampling', JSON.stringify(sampling)); showSamplingNote(); }
    function showSamplingNote() {
      const n = document.getElementById('set-samp-note');
      if (!n) return;
      const r = requestSampling();
      n.textContent = r.temperature === 0 ? 'greedy' : 'temperature ' + (r.temperature === undefined ? '1.0 (model default)' : r.temperature);
    }
    function fillSamplingFields() {
      const grid = document.getElementById('set-samp-grid');
      const vals = sampling.preset === 'custom' ? sampling.custom : SAMPLING_PRESETS[sampling.preset];
      for (const k of SAMPLING_FIELDS) { const el = document.getElementById('set-samp-' + k); if (el) el.value = vals[k] === undefined ? '' : vals[k]; }
      grid.classList.toggle('off', sampling.preset !== 'custom');
      document.getElementById('set-samp-preset').value = sampling.preset;
    }
    function setSamplingPreset(p) {
      // picking Custom starts from what the preset showed, so a tweak does not lose the other fields
      if (p === 'custom' && sampling.preset !== 'custom') sampling.custom = Object.assign({}, requestSampling());
      sampling.preset = p;
      fillSamplingFields();
      saveSampling();
    }
    function samplingFieldChanged() {
      const c = {};
      for (const k of SAMPLING_FIELDS) c[k] = document.getElementById('set-samp-' + k).value;
      sampling.custom = c;
      saveSampling();
    }

    // ---- shorten long thinking / tool output (Settings > Display)
    let truncOn = localStorage.getItem('shadow_trunc_on') === 'true';
    let truncLines = Math.max(2, Math.min(60, Number(localStorage.getItem('shadow_trunc_lines')) || 8));
    function applyTrunc() {
      document.body.classList.toggle('trunc-on', truncOn);
      document.body.style.setProperty('--trunc-lines', truncLines);
      document.querySelectorAll('.think-content, .tool-output').forEach(markClipped);
    }
    function setTruncOn(on) { truncOn = on; localStorage.setItem('shadow_trunc_on', on); applyTrunc(); }
    function setTruncLines(v) { truncLines = Math.max(2, Math.min(60, Number(v) || 8)); localStorage.setItem('shadow_trunc_lines', truncLines); document.getElementById('set-trunc-lines').value = truncLines; applyTrunc(); }
    // marks a block whose text is cut by the line limit (for the fade and the click); thinking shows its latest lines while it streams
    function markClipped(el) {
      if (!el) return;
      const clip = truncOn && !el.classList.contains('expanded') && el.scrollHeight > el.clientHeight + 2;
      el.classList.toggle('clipped', clip);
      if (clip) el.title = 'Click to show all';
    }
    function tailThinking(el) { if (truncOn && !el.classList.contains('expanded')) { el.scrollTop = el.scrollHeight; markClipped(el); } }
    document.addEventListener('click', ev => {
      const el = ev.target.closest && ev.target.closest('.think-content, .tool-output');
      if (!el || !truncOn || String(window.getSelection()).length) return;
      if (el.classList.contains('clipped') || el.classList.contains('expanded')) { el.classList.toggle('expanded'); el.title = el.classList.contains('expanded') ? 'Click to shorten' : 'Click to show all'; markClipped(el); }
    });

    applyTrunc();

    // ---- Regenerate: drop the last reply (everything after the last user message) and ask again
    // a turn that ran tools is not asked again: the calls (writes, edits, commands) would run a second time
    function lastTurnUsedTools() {
      let i = history.length - 1;
      while (i >= 0 && history[i].role !== 'user') i--;
      return history.slice(i + 1).some(m => m.role === 'tool' || (m.role === 'assistant' && parseToolCallsFromText(messageText(m)).length > 0));
    }
    function regenerateLast() {
      if (isGenerating || lastTurnUsedTools()) return;
      let i = history.length - 1;
      while (i >= 0 && history[i].role !== 'user') i--;
      if (i < 0) return;
      history.length = i + 1;
      const rows = Array.from(messagesEl.querySelectorAll('.msg-row'));
      let lastUser = -1;
      rows.forEach((r, k) => { if (r.classList.contains('user')) lastUser = k; });
      rows.slice(lastUser + 1).forEach(r => r.remove());
      resumeOnly = true;
      sendMessage();
    }
    setInterval(() => {
      const b = document.getElementById('regen-btn');
      if (b) b.style.display = (!isGenerating && history.length > 0 && history[history.length - 1].role === 'assistant' && !lastTurnUsedTools()) ? 'inline-flex' : 'none';
    }, 400);

    function settingsOpen() {
      fillSamplingFields(); showSamplingNote();
      document.getElementById('set-trunc-on').checked = truncOn;
      document.getElementById('set-trunc-lines').value = truncLines;
      document.getElementById('set-persona').value = currentPersona;
      document.getElementById('set-mem-on').checked = memoryOn;
      document.getElementById('set-finish-on').checked = finishOn;
      loadMemory();
    }
    function savePersonaSetting() {
      currentPersona = document.getElementById('set-persona').value.trim() || DEFAULT_PERSONA;
      localStorage.setItem('shadow_persona', currentPersona);
      document.getElementById('set-persona').value = currentPersona;
      document.getElementById('set-persona-note').textContent = 'Saved: new chats use it.';
    }
    function resetPersonaSetting() {
      document.getElementById('set-persona').value = DEFAULT_PERSONA;
      savePersonaSetting();
      document.getElementById('set-persona-note').textContent = 'Back to the built-in default.';
    }

    // Learn from past chats: the model reads what you asked in your saved conversations and proposes notes; nothing is saved until you tick it
    async function learnFromChats() {
      const note = document.getElementById('set-learn-note'), box = document.getElementById('set-suggest'), btn = document.getElementById('set-learn');
      if (isGenerating) { note.textContent = 'Wait for the reply in progress.'; return; }
      const asked = [];
      let size = 0;
      for (const c of chats.slice().sort((a, b) => b.ts - a.ts)) {
        for (const m of c.history) {
          if (m.role !== 'user') continue;
          let t = messageText(m);
          if (t.startsWith('File: ') && t.lastIndexOf('```') >= 0) t = t.slice(t.lastIndexOf('```') + 3);
          t = t.replace(/\s+/g, ' ').trim().slice(0, 280);
          if (t.length < 8 || size + t.length > 12000 || asked.length >= 80) continue;
          asked.push(t); size += t.length;
        }
      }
      if (asked.length < 3) { note.textContent = 'Not enough saved chats yet: chat a little first.'; return; }
      btn.disabled = true; box.innerHTML = '';
      note.textContent = `Shadow is reading ${asked.length} of your messages...`;
      const prompt = 'Below are messages a user sent to an AI assistant in earlier chats. Write up to 10 short notes, one per line, each starting with "- ", that capture lasting facts about this user: ' +
        'what they work on, the tools and languages they use, how they like answers (length, format, tone), and the kinds of request they make again and again. ' +
        'Only what the messages clearly show; no guesses; nothing about passwords, keys or other private data; no one-off questions. Each note is a plain statement about the user, ' +
        'for example "Works on a Rust and CUDA inference engine called Shadow".\n\nMessages:\n' + asked.map((t, i) => `${i + 1}. ${t}`).join('\n');
      try {
        const r = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: [{ role: 'user', content: prompt }], max_tokens: 600, stream: false, chat_template_kwargs: { enable_thinking: false } }) });
        const j = await r.json();
        if (!r.ok) throw new Error((j.error && j.error.message) || r.status);
        const have = memories.map(m => m.text.toLowerCase());
        const notes = String(j.choices[0].message.content || '').split('\n').map(l => l.trim()).filter(l => l.startsWith('- ')).map(l => l.slice(2).trim())
          .filter(l => l.length >= 8 && l.length <= 400 && !have.includes(l.toLowerCase()));
        if (!notes.length) { note.textContent = 'Nothing new to suggest from those chats.'; }
        else {
          note.textContent = 'Tick the ones that are right:';
          box.innerHTML = notes.map((n, i) => `<label class="sug"><input type="checkbox" checked data-i="${i}"><span>${escapeHtml(n)}</span></label>`).join('') +
            '<div class="set-row"><button class="dash-btn primary" id="sug-save">Save the ticked notes</button></div>';
          document.getElementById('sug-save').onclick = async () => {
            for (const cb of box.querySelectorAll('input:checked')) await memCall('POST', '/v1/memory', { text: notes[+cb.dataset.i], kind: 'user' });
            box.innerHTML = ''; note.textContent = 'Saved.';
          };
        }
      } catch (e) { note.textContent = 'That did not work: ' + e.message; }
      btn.disabled = false;
    }
    loadMemory();

    // ---- the Browser pane: the page the model's browser has open, in the chat view, as a picture that follows it (the browser itself runs headless)
    let browserTimer = null, browserBusy = false;
    function browserClose() {
      document.body.classList.remove('browser-open');
      clearInterval(browserTimer);
      browserTimer = null;
    }
    async function browserRefresh() {
      if (browserBusy) return;
      browserBusy = true;
      const st = document.getElementById('bp-status');
      try {
        const r = await fetch('/v1/browser/screenshot');
        const j = await r.json();
        if (!r.ok) { st.textContent = (j.error && j.error.message) || 'No screenshot.'; return; }
        if (j.image) {
          const img = document.getElementById('bp-img');
          img.src = j.image;
          img.style.display = 'block';
          document.getElementById('bp-empty').style.display = 'none';
        }
        browserPageInfo(j.text || '');
        st.textContent = 'Updated ' + new Date().toLocaleTimeString();
      } catch (_) { st.textContent = 'Could not reach the server.'; }
      finally { browserBusy = false; }
    }
    function browserPageInfo(text) {
      const u = /Page URL: (\S+)/.exec(text), t = /Page Title: (.*)/.exec(text);
      if (u) document.getElementById('bp-url').textContent = u[1];
      if (u || t) document.getElementById('bp-url').title = t ? t[1] : '';
    }
    // after every call of a browser tool: show the pane, say where the page is, take a fresh picture; while a reply is being written, keep it fresh
    function browserNote(output) {
      document.body.classList.add('browser-open');
      browserPageInfo(output || '');
      browserRefresh();
      if (!browserTimer) browserTimer = setInterval(() => { if (isGenerating) browserRefresh(); }, 4000);
    }

    // ---- Unlimited rolling context window (0 extra RAM/VRAM):
    // Keeps `history` (and saved chats in localStorage) 100% intact for scrollback while building a bounded
    // context payload that preserves the system prompt prefix (so `prefix.snap` always hits) and folds older
    // turns/tool outputs into a rolling summary whenever the conversation grows near `serverCtx`.
    let serverCtx = 16384;
    function buildContextMessages(hist) {
      const size = m => Array.isArray(m.content) ? m.content.reduce((n, p) => n + (p.text ? p.text.length : 1800), 0) : String(m.content || '').length;
      const room = Math.max(6000, (serverCtx - 3800) * 3); // system prompt + tool schemas + reply headroom
      const cloned = hist.map(m => ({ ...m, content: Array.isArray(m.content) ? m.content.map(p => ({ ...p })) : m.content }));
      let total = cloned.reduce((n, m) => n + size(m), 0);
      if (total <= room * 0.8) return cloned;

      // Stage 1: Compact older tool outputs (except trailing tool outputs of the current turn)
      let lastNonTool = cloned.length - 1;
      while (lastNonTool > 0 && cloned[lastNonTool].role === 'tool') lastNonTool--;
      let shortenedTools = 0;
      // the working set: the newest tool outputs stay whole while they fit a budget (30% of the room) - the last step's results always - so that one edit can use more than
      // the last read (with one output whole, a model that needs two sources recalls the first again and again)
      const keepWhole = new Set();
      let wsUsed = 0;
      for (let i = cloned.length - 1; i >= 0; i--) {
        if (cloned[i].role !== 'tool') continue;
        const sz = size(cloned[i]);
        if (i < lastNonTool && wsUsed + sz > room * 0.3) break;
        wsUsed += sz;
        keepWhole.add(i);
      }
      for (let i = 0; i < lastNonTool; i++) {
        if (total <= room * 0.65) break;
        const m = cloned[i];
        if (m.role !== 'tool' || typeof m.content !== 'string' || m.content.length < 400 || keepWhole.has(i)) continue;
        const was = m.content.length;
        const unwrapped = unwrapToolResponse(m.content);
        const num = /^\(output #(\d+)\)\s*/.exec(unwrapped);   // the server's log of tool outputs: `recall` reads that one again
        const raw = (num ? unwrapped.slice(num[0].length) : unwrapped).replace(/\s+/g, ' ').trim();
        const preview = raw.length > 160 ? raw.slice(0, 160) + '…' : raw;
        m.content = num
          ? `<tool_response>\n[earlier tool output #${num[1]} (${was} chars), shortened to save context; it began: ${preview} - call recall with id ${num[1]} to read it again (a line range, or a text to search for)]\n</tool_response>`
          : `<tool_response>\n[earlier tool output (${was} chars), shortened to save context; it began: ${preview} - call recall with message ${i} to read it in full]\n</tool_response>`;
        total -= was - m.content.length;
        shortenedTools++;
      }

      // Stage 2: Fold oldest user/assistant/tool turn groups into a bounded rolling summary
      const bullets = [];
      let foldedTurns = 0;
      let base = 0;   // the number of `hist` of the first message still in `cloned` (the fold takes messages off the front)
      while (cloned.length > 2 && total > room * 0.72) {
        let nextUser = 1;
        while (nextUser < cloned.length && cloned[nextUser].role !== 'user') nextUser++;
        const remainingUsers = cloned.slice(nextUser).filter(m => m.role === 'user').length;
        if (remainingUsers === 0) break;

        const group = cloned.splice(0, nextUser);
        const firstMsg = base;
        base += group.length;
        let uTxt = '', aTxt = '';
        const tNames = [];
        for (const gm of group) {
          total -= size(gm);
          if (gm.role === 'user') {
            uTxt = messageText(gm).replace(/\s+/g, ' ').trim().slice(0, 140);
          } else if (gm.role === 'assistant') {
            const rawA = messageText(gm);
            for (const tc of parseToolCallsFromText(rawA)) tNames.push(tc.name);
            const cleanA = stripToolCalls(rawA).replace(/\s+/g, ' ').trim();
            if (cleanA) aTxt = cleanA.slice(0, 160);
          }
        }
        foldedTurns++;
        const toolNote = tNames.length ? ` (tools: ${tNames.join(', ')})` : '';
        bullets.push(`- Turn ${foldedTurns} (messages m${firstMsg}-m${base - 1}): User: "${uTxt}" -> Assistant${toolNote}: "${aTxt}"`);
        if (bullets.length > 12) bullets.shift();
      }

      // Stage 2b: one long agent run (a single user message: nothing above could fold it) - its oldest STEPS (an assistant message and the tool results that answer it) become one
      // line each, the recent ones stay whole: each step costs a message of its own and a preview per tool result even after stage 1, and 85 steps filled the window and left the
      // reply 560 tokens.  The line names the messages (recall message=N), the calls, the numbers of their outputs (recall id=N) and the start of the model's own note.
      const stepLines = [];
      let foldedSteps = 0;
      if (cloned.length > 12 && total > room * 0.7 && cloned[0].role === 'user') {
        const KEEP_TAIL = 8;                       // messages at the end that are never folded
        let at = 1;                                // the first message that is still whole (an assistant message: a step starts there)
        while (at < cloned.length - KEEP_TAIL && cloned[at].role === 'assistant' && total > room * 0.55) {
          let end = at + 1;
          while (end < cloned.length && cloned[end].role === 'tool') end++;
          if (end > cloned.length - KEEP_TAIL) break;
          const grp = cloned.slice(at, end);
          const calls = parseToolCallsFromText(messageText(grp[0]));
          const desc = calls.map(c => {
            const a = c.arguments || {};
            const key = ['path', 'pattern', 'command', 'query', 'url', 'text'].find(k => a[k] !== undefined);
            const val = key ? String(a[key]).replace(/\s+/g, ' ').slice(0, 60) : '';
            const range = (a.start || a.end) ? ` ${a.start || ''}-${a.end || ''}` : '';
            return `${c.name}(${val}${range})`;
          }).join(', ') || 'a reply';
          const ids = grp.slice(1).map(t => /output #(\d+)/.exec(messageText(t))).filter(Boolean).map(x => '#' + x[1]);
          const note = stripToolCalls(messageText(grp[0])).replace(/\s+/g, ' ').trim().slice(0, 80);
          stepLines.push(`- m${base + at}-m${base + end - 1}: ${desc}${ids.length ? ' -> output ' + ids.join(',') : ''}${note ? ' | ' + note : ''}`.slice(0, 260));
          for (const gm of grp) total -= size(gm);
          at = end;
        }
        if (at > 1) {
          cloned.splice(1, at - 1);
          foldedSteps = stepLines.length;
          base += at - 1;
          // a lot of steps: the oldest lines merge into one (what was done, by tool), the newest 40 stay
          if (stepLines.length > 40) {
            const old = stepLines.splice(0, stepLines.length - 40);
            const counts = {};
            for (const l of old) for (const m of l.matchAll(/(\w+)\(/g)) counts[m[1]] = (counts[m[1]] || 0) + 1;
            const first = /^- (m\d+)-/.exec(old[0]), last = /^- m\d+-(m\d+)/.exec(old[old.length - 1]);
            stepLines.unshift(`- ${first ? first[1] : 'm?'}-${last ? last[1] : 'm?'}: ${old.length} earlier steps (${Object.entries(counts).map(([k, v]) => k + 'x' + v).join(', ')}); search them with recall query=...`);
          }
          total += stepLines.join('\n').length + 300;
        }
      }

      if ((foldedTurns > 0 || foldedSteps > 0) && cloned.length > 0 && cloned[0].role === 'user') {
        const stepsPart = foldedSteps > 0 ? `[Earlier steps of this task, shortened to fit the context window: one line each; read a step in full with recall message=N, a tool output with recall id=N; your pinned notes hold what you decided]\n${stepLines.join('\n')}\n\n` : '';
        const prefix = stepsPart + (foldedTurns > 0 ? `[Earlier conversation summary — ${foldedTurns} earlier turn(s) compacted to fit context window; read one in full with recall message=N, or search them with recall query=...]\n${bullets.join('\n')}\n\n[Current message]\n` : '[The task]\n');
        if (Array.isArray(cloned[0].content)) {
          cloned[0].content = cloned[0].content.map(p => p.type === 'text' ? { ...p, text: prefix + p.text } : p);
        } else {
          cloned[0].content = prefix + String(cloned[0].content || '');
        }
      }

      if (foldedTurns > 0 || shortenedTools > 0 || foldedSteps > 0) {
        const parts = [];
        if (foldedTurns > 0) parts.push(`${foldedTurns} earlier turn${foldedTurns > 1 ? 's' : ''} summarized`);
        if (foldedSteps > 0) parts.push(`${foldedSteps} earlier step${foldedSteps > 1 ? 's' : ''} folded`);
        if (shortenedTools > 0) parts.push(`${shortenedTools} tool output${shortenedTools > 1 ? 's' : ''} compacted`);
        timingInfo.textContent = `Rolling context active: ${parts.join(', ')} (0 B extra RAM)`;
      }
      return cloned;
    }

    // ---- the read-only treadmill: a model that has read, lost the text to the shortening, and reads again never reaches an edit (one saved run: 132 calls, 75 recalls, 2 notes, 0 edits)
    const READ_ONLY_TOOLS = new Set(['read_file', 'grep', 'glob', 'recall', 'search']);
    const PROGRESS_TOOLS = new Set(['notes', 'edit_file', 'write_file', 'replace_lines']);
    function readOnlyStreak(name, streak) {
      if (PROGRESS_TOOLS.has(name)) return 0;
      return READ_ONLY_TOOLS.has(name) ? streak + 1 : streak;
    }
    function treadmillNote(streak) {
      if (streak === 8) return `(note: ${streak} read-only calls in a row without saving notes or editing. What you have read drops out of view a few steps later: write the facts you need into the notes tool now - file, line numbers, the exact text to change - then make the first edit. Read more only for something you cannot write down yet.)`;
      if (streak >= 14 && (streak - 14) % 4 === 0) return `(note: ${streak} read-only calls in a row. Stop reading. In your next step write your notes and change something: a first small edit with replace_lines or edit_file is worth more than another read.)`;
      return '';
    }

    // ---- queued messages: typed while a reply is being written, sent in order when it ends (each one needs the one before it answered)
    const sendQueue = [];
    let queuePaused = false;
    function enqueueMessage(text) {
      sendQueue.push({ text, atts: attachments.splice(0, attachments.length) });
      inputEl.value = '';
      inputEl.style.height = 'auto';
      renderTray();
      renderQueue();
    }
    function renderQueue() {
      const bar = document.getElementById('queue-bar');
      if (!bar) return;
      bar.innerHTML = sendQueue.map((q, i) => `<div class="q-item"><span class="n">${i + 1}</span><span class="t">${escapeHtml(q.text || '(attachments only)')}</span>` +
        `${q.atts.length ? `<small>📎 ${q.atts.length}</small>` : ''}<button title="Remove from the queue" onclick="dropQueued(${i})">✕</button></div>`).join('') +
        (sendQueue.length && queuePaused ? `<div class="q-note">Queue paused (the last reply was stopped or failed).<button onclick="resumeQueue()">Resume</button></div>` : '');
    }
    function dropQueued(i) { sendQueue.splice(i, 1); renderQueue(); }
    function resumeQueue() { queuePaused = false; if (!isGenerating) runQueue(true); else renderQueue(); }
    // after a reply: the next queued message, unless the reply was stopped or failed (then the queue waits for Resume)
    function runQueue(force) {
      if (!sendQueue.length || isGenerating) { renderQueue(); return; }
      const t = timingInfo.textContent || '';
      if (!force && (queuePaused || t.startsWith('Generation stopped') || t.startsWith('Failed'))) { queuePaused = true; renderQueue(); return; }
      const q = sendQueue.shift();
      renderQueue();
      attachments.splice(0, attachments.length, ...q.atts);
      renderTray();
      inputEl.value = q.text;
      setTimeout(sendMessage, 0);
    }

    document.body.classList.toggle('view-chat', currentView === 'chat');
    // (the old #monitor link: the Process Monitor became part of the Dashboard)
    if (['#dash', '#settings', '#monitor'].includes(window.location.hash)) switchView(window.location.hash === '#settings' ? 'settings' : 'dash');

pollTelemetry();
    setInterval(pollTelemetry, 1000);

    // ---- dashboard tiles drag & drop: drag by the title bar, drop on another tile to swap, snap into the grid; the order is remembered in this browser
    const dashGrid = document.getElementById('dash-grid');
    const DASH_ORDER_KEY = 'shadow_dash_order';
    let dashDragEl = null;
    function dashSaveOrder() {
      const order = [...dashGrid.querySelectorAll(':scope > .dcard')].map(c => c.dataset.tile);
      localStorage.setItem(DASH_ORDER_KEY, JSON.stringify(order));
    }
    function dashApplyOrder(order) {
      for (const name of order) {
        const card = dashGrid.querySelector(`:scope > .dcard[data-tile="${name}"]`);
        if (card) dashGrid.appendChild(card);
      }
    }
    function dashResetLayout() {
      localStorage.removeItem(DASH_ORDER_KEY);
      location.reload();
    }
    function dashEnableDrag() {
      if (!dashGrid) return;
for (const card of dashGrid.querySelectorAll(':scope > .dcard')) {
        card.setAttribute('draggable', 'true');
card.addEventListener('dragstart', e => {
          dashDragEl = card;
          card.classList.add('dragging');
          e.dataTransfer.setData('text/plain', card.dataset.tile);
          e.dataTransfer.effectAllowed = 'move';
        });
        card.addEventListener('dragend', () => {
          card.classList.remove('dragging');
          dashGrid.querySelectorAll('.droptarget').forEach(c => c.classList.remove('droptarget'));
          dashDragEl = null;
        });
        card.addEventListener('dragenter', () => { if (dashDragEl && dashDragEl !== card) card.classList.add('droptarget'); });
        card.addEventListener('dragover', e => { if (dashDragEl) e.preventDefault(); });
        card.addEventListener('dragleave', e => { if (e.relatedTarget && !card.contains(e.relatedTarget)) card.classList.remove('droptarget'); });
        card.addEventListener('drop', e => {
          e.preventDefault();
          card.classList.remove('droptarget');
          if (!dashDragEl || dashDragEl === card) return;
          const cards = [...dashGrid.querySelectorAll(':scope > .dcard')];
          const di = cards.indexOf(dashDragEl), ti = cards.indexOf(card);
          dashGrid.insertBefore(dashDragEl, di < ti ? card.nextSibling : card);
          dashSaveOrder();
          dashRefresh();
        });
      }
      const saved = localStorage.getItem(DASH_ORDER_KEY);
      if (saved) {
        try { dashApplyOrder(JSON.parse(saved)); } catch (e) {}
        const btn = document.getElementById('dash-reset');
        if (btn) btn.style.display = '';
      }
    }
    dashEnableDrag();
  </script>
</body>
</html>
"#;

const MONITOR_HTML: &str = r#"<!DOCTYPE html>
<html lang="en" class="dark h-full">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="0; url=/#dash">
  <title>Redirecting to the Shadow Dashboard...</title>
  <script>
    window.location.href = '/#dash';
  </script>
</head>
<body style="background:#090a0f; color:#8492a6; font-family:sans-serif; display:flex; align-items:center; justify-content:center; height:100vh;">
  <p>Loading the Shadow Dashboard... <a href="/" style="color:#6366f1;">Return to Chat</a></p>
</body>
</html>
"#;
