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
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0.65rem 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      z-index: 10;
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
            padding: 6px 12px; font: inherit; font-size: 0.85rem; cursor: pointer; }
    .chip:hover { border-color: rgba(99, 102, 241, 0.6); background: rgba(99, 102, 241, 0.12); }
    .msg-error { color: var(--danger); }
    .bubble ul, .bubble ol { margin: 4px 0 4px 1.3em; padding: 0; }
    .bubble li { margin: 2px 0; }
    .md-gap { height: 0.6em; }
    .md-h { font-weight: 600; margin: 6px 0 2px; }
    .md-h1 { font-size: 1.15rem; } .md-h2 { font-size: 1.05rem; }
    .reading-note { opacity: 0.65; font-size: 0.88rem; margin-right: 6px; }
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
      scroll-behavior: smooth;
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

    /* Process Monitor View - Strictly separate tab */
    #monitor-panel {
      width: 100%;
      height: 100%;
      min-height: 0;
      overflow-y: auto;
      padding: 1.5rem;
      gap: 1.25rem;
    }
    .monitor-grid {
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
    }
    .mon-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.15rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .mon-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .mon-card-title {
      font-size: 0.95rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text);
    }
    .mon-card-badge {
      font-size: 0.72rem;
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      background: var(--surface-hover);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }
    .mon-stat-big {
      display: flex;
      align-items: baseline;
      gap: 0.4rem;
    }
    .mon-big-val {
      font-size: 1.85rem;
      font-weight: 800;
      color: #fff;
      font-variant-numeric: tabular-nums;
    }
    .mon-big-unit {
      font-size: 0.9rem;
      color: var(--text-muted);
      font-weight: 500;
    }
    .progress-bar-wrap {
      width: 100%;
      height: 7px;
      background: var(--bg);
      border-radius: 9999px;
      overflow: hidden;
      border: 1px solid var(--border-subtle);
    }
    .progress-bar {
      height: 100%;
      background: var(--accent);
      border-radius: 9999px;
      transition: width 0.3s ease;
    }
    .progress-bar.cyan { background: var(--cyan); }
    .progress-bar.warning { background: var(--warning); }
    .progress-bar.success { background: var(--success); }

    .stat-rows {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      font-size: 0.8rem;
    }
    .stat-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--text-muted);
    }
    .stat-row strong {
      color: var(--text);
      font-variant-numeric: tabular-nums;
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
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <div class="brand" id="brand-logo">
        <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
        <span>Shadow</span>
      </div>
      <div class="nav-tabs">
        <button class="nav-tab active" id="tab-chat" onclick="switchView('chat')">
          <span>💬</span> Chat
        </button>
        <button class="nav-tab" id="tab-monitor" onclick="switchView('monitor')">
          <span>📊</span> Process Monitor
        </button>
      </div>
    </div>
    <div class="stats-bar">
      <span class="pill ready" id="status-pill">Ready</span>
      <span class="pill" id="speed-pill">Speed: <strong>-- tok/s</strong></span>
      <span class="pill" id="gpu-header-pill">GPU: <strong>--% • VRAM: -- GiB</strong></span>
    </div>
  </header>

  <div class="main-wrap">
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
          </div>
        </div>
      </div>

      <div id="input-pane">
        <div class="input-box">
          <textarea id="prompt-input" rows="1" placeholder="Message Shadow (Enter to send, Shift+Enter for a new line)"></textarea>
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

              <div class="dropdown-item" data-effort="high" onclick="selectEffort('high')">
                <div class="item-left">
                  <div class="item-icon">🔬</div>
                  <div class="item-info">
                    <div class="item-title">High</div>
                    <div class="item-desc">Deep problem solving & verification (~16k)</div>
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

            <!-- System Persona Selector Pill -->
            <button class="menu-pill-btn" id="persona-btn" onclick="openPersonaModal()">
              <span>⚙</span>
              <span id="persona-btn-label">Persona: Windows Dev</span>
            </button>
          </div>

          <span id="timing-info">Ready</span>
        </div>
      </div>
    </div>

    <!-- VIEW: PROCESS MONITOR (Separate Tab Only) -->
    <div class="view-panel" id="monitor-panel">
      <div class="monitor-grid">
        <!-- CPU Card -->
        <div class="mon-card">
          <div class="mon-card-header">
            <div class="mon-card-title"><span>⚡</span> CPU Usage</div>
            <span class="mon-card-badge" id="cpu-cores-badge">-- Cores</span>
          </div>
          <div class="mon-stat-big">
            <span class="mon-big-val" id="cpu-proc-val">0.0%</span>
            <span class="mon-big-unit">Process (<span id="cpu-proc-cores">0.0</span> cores • Sys: <span id="cpu-sys-val">0.0%</span>)</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar" id="cpu-bar" style="width: 0%;"></div>
          </div>
          <div class="stat-rows">
            <div class="stat-row"><span>Shadow Process CPU</span><strong id="cpu-proc-txt">0.0%</strong></div>
            <div class="stat-row"><span>Overall System CPU</span><strong id="cpu-sys-txt">0.0%</strong></div>
            <div class="stat-row"><span>Active CPU Cores</span><strong id="cpu-cores-used-txt">0.0 / --</strong></div>
            <div class="stat-row"><span>Logical Cores Total</span><strong id="cpu-cores-txt">--</strong></div>
          </div>
        </div>

        <!-- GPU & VRAM Card (Real NVML Compute Usage + VRAM) -->
        <div class="mon-card">
          <div class="mon-card-header">
            <div class="mon-card-title"><span>🎮</span> GPU Compute & VRAM</div>
            <span class="mon-card-badge" id="gpu-device-badge">NVIDIA GPU</span>
          </div>
          <div class="mon-stat-big">
            <span class="mon-big-val" id="gpu-util-val">0.0%</span>
            <span class="mon-big-unit">Compute (VRAM: <span id="vram-used-val">--</span> / <span id="vram-total-val">-- GiB</span>)</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar cyan" id="gpu-bar" style="width: 0%;"></div>
          </div>
          <div class="stat-rows">
            <div class="stat-row"><span>GPU Compute Engine</span><strong id="gpu-util-txt">0.0%</strong></div>
            <div class="stat-row"><span>VRAM Allocation</span><strong id="vram-used-total-txt">-- / -- GiB</strong></div>
            <div class="stat-row"><span>GPU Temperature</span><strong id="gpu-temp-txt">-- °C</strong></div>
            <div class="stat-row"><span>Memory Controller Activity</span><strong id="gpu-mem-txt">0.0%</strong></div>
            <div class="stat-row"><span>Free VRAM</span><strong id="vram-free-txt">-- GiB</strong></div>
            <div class="stat-row"><span>Expert VRAM Slots</span><strong id="vram-slots-txt">--</strong></div>
            <div class="stat-row"><span>VRAM Cache Hit Rate</span><strong id="vram-hit-rate-txt">--%</strong></div>
          </div>
        </div>

        <!-- RAM Card -->
        <div class="mon-card">
          <div class="mon-card-header">
            <div class="mon-card-title"><span>🧠</span> System Memory (RAM)</div>
            <span class="mon-card-badge" id="ram-load-badge">--% Load</span>
          </div>
          <div class="mon-stat-big">
            <span class="mon-big-val" id="ram-used-val">--</span>
            <span class="mon-big-unit">/ <span id="ram-total-val">-- GiB</span></span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar warning" id="ram-bar" style="width: 0%;"></div>
          </div>
          <div class="stat-rows">
            <div class="stat-row"><span>Shadow Working Set</span><strong id="ram-proc-ws-txt">-- GiB</strong></div>
            <div class="stat-row"><span>Pinned Tier (Hot Cache)</span><strong id="ram-pinned-txt">-- GiB</strong></div>
            <div class="stat-row"><span>Available System RAM</span><strong id="ram-avail-txt">-- GiB</strong></div>
          </div>
        </div>

        <!-- Engine & Process Card -->
        <div class="mon-card">
          <div class="mon-card-header">
            <div class="mon-card-title"><span>⚙</span> Shadow Engine</div>
            <span class="mon-card-badge" id="engine-pid-badge">PID: --</span>
          </div>
          <div class="mon-stat-big">
            <span class="mon-big-val" id="engine-speed-val">0.0</span>
            <span class="mon-big-unit">tok/s decode</span>
          </div>
          <div class="stat-rows">
            <div class="stat-row"><span>Active Model</span><strong id="engine-model-txt" style="max-width: 170px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">--</strong></div>
            <div class="stat-row"><span>Uptime</span><strong id="engine-uptime-txt">--</strong></div>
            <div class="stat-row"><span>Total Requests</span><strong id="engine-requests-txt">0</strong></div>
            <div class="stat-row"><span>Tokens Generated</span><strong id="engine-tokens-txt">0</strong></div>
            <div class="stat-row"><span>Speculative Decoding</span><strong id="engine-mtp-txt">--</strong></div>
            <div class="stat-row"><span>Queued Requests</span><strong id="engine-queue-txt">0</strong></div>
          </div>
        </div>
      </div>

      <!-- MoE Memory Hierarchy Diagram -->
      <div class="tier-diagram-wrap">
        <div class="tier-diagram-title"><span>⚡</span> MoE 3-Tier Memory Architecture</div>
        <div class="tier-flow">
          <div class="tier-box active-tier">
            <div class="tier-header">
              <span>Tier 1: VRAM Arena</span>
              <span class="tag">Zero-Copy GPU</span>
            </div>
            <div class="stat-rows">
              <div class="stat-row"><span>VRAM Slots:</span><strong id="tier1-slots">--</strong></div>
              <div class="stat-row"><span>Cache Hits:</span><strong id="tier1-hits" style="color:var(--success);">--</strong></div>
              <div class="stat-row"><span>Cache Misses:</span><strong id="tier1-misses">--</strong></div>
            </div>
          </div>

          <div class="tier-box">
            <div class="tier-header">
              <span>Tier 2: Pinned RAM Tier</span>
              <span class="tag">DMA Bus Speed</span>
            </div>
            <div class="stat-rows">
              <div class="stat-row"><span>Pinned Capacity:</span><strong id="tier2-cap">30.0 GiB</strong></div>
              <div class="stat-row"><span>Pinned Experts:</span><strong id="tier2-experts">--</strong></div>
              <div class="stat-row"><span>RAM Tier Hits:</span><strong id="tier2-hits" style="color:var(--cyan);">--</strong></div>
            </div>
          </div>

          <div class="tier-box">
            <div class="tier-header">
              <span>Tier 3: NVMe File Tier</span>
              <span class="tag">Cold Shards</span>
            </div>
            <div class="stat-rows">
              <div class="stat-row"><span>NVMe Reads:</span><strong id="tier3-reads">--</strong></div>
              <div class="stat-row"><span>Status:</span><strong style="color:var(--text-muted);">Asynchronous Mmap</strong></div>
            </div>
          </div>
        </div>
      </div>
    </div>
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
    let toolsEnabled = localStorage.getItem('shadow_tools_enabled') !== 'false';
    const DEFAULT_PERSONA = "You are Shadow, a sovereign, high-performance local AI assistant running natively on the user's Windows workstation. You have full permission and local authority from the user to analyze, inspect, refactor, and modify files and code within this project and previous sessions (such as J:\\Strata-work\\sessions and J:\\Strata-work\\conversations). You have access to native Windows tools: read_file, glob, grep, edit_file, write_file, and cmd (PowerShell). When asked to inspect past chats, run tests, or edit code, invoke the appropriate tools directly without hesitation.";
    let currentPersona = localStorage.getItem('shadow_persona') || DEFAULT_PERSONA;

    // Elements
    const chatPanel = document.getElementById('chat-panel');
    const monitorPanel = document.getElementById('monitor-panel');
    const tabChat = document.getElementById('tab-chat');
    const tabMonitor = document.getElementById('tab-monitor');
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

    let history = [];
    let isGenerating = false;
    let availableTools = [];
    let currentAbortController = null;

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

    // Global Escape key to instantly halt active generation
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && isGenerating) {
        e.preventDefault();
        stopGeneration(e);
      }
    });

    // Load tool schemas from server
    async function loadTools() {
      try {
        const res = await fetch('/v1/tools');
        if (res.ok) {
          availableTools = await res.json();
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

    // View Switching: Chat and Process Monitor are completely separate tabs
    function switchView(view) {
      currentView = view;
      if (view === 'chat') {
        chatPanel.classList.add('active');
        monitorPanel.classList.remove('active');
        tabChat.classList.add('active');
        tabMonitor.classList.remove('active');
        window.location.hash = '';
      } else {
        chatPanel.classList.remove('active');
        monitorPanel.classList.add('active');
        tabChat.classList.remove('active');
        tabMonitor.classList.add('active');
        window.location.hash = '#monitor';
        pollTelemetry();
      }
    }

    document.getElementById('brand-logo').addEventListener('click', () => switchView('chat'));

    if (window.location.hash === '#monitor') {
      switchView('monitor');
    }

    // Thinking Level Selection
    function toggleThinkingDropdown(e) {
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

    // Textarea auto-resize
    inputEl.addEventListener('input', () => {
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
    });

    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener('click', sendMessage);

    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[m]));
    }

    // A small Markdown subset, escape-first: everything the model wrote is HTML-escaped before any tag is added, so model
    // output can never inject markup.  Fenced code, inline code, bold, italics, headings, bullet and numbered lists.
    function renderMarkdown(raw) {
      const blocks = [];
      let text = raw.replace(/```(\w*)\n?([\s\S]*?)(```|$)/g, (_, lang, code) => {
        blocks.push(`<pre><button class="copy-btn" onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText)">Copy</button><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`);
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

    // Execute tool on Windows via /v1/tools/execute
    async function executeLocalTool(name, args) {
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

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text || isGenerating) return;
      if (modelState === 'loading') {
        timingInfo.textContent = 'The model is still loading - your message is still in the box.';
        return;
      }
      const welcome = document.getElementById('welcome');
      if (welcome) welcome.closest('.msg-row').remove();
      let failed = false;

      isGenerating = true;
      sendBtn.style.display = 'none';
      stopBtn.style.display = 'flex';
      currentAbortController = new AbortController();
      inputEl.value = '';
      inputEl.style.height = 'auto';

      history.push({ role: 'user', content: text });

      // User Bubble
      const userRow = document.createElement('div');
      userRow.className = 'msg-row user';
      userRow.innerHTML = `<div class="avatar">U</div><div class="bubble">${escapeHtml(text).replace(/\n/g, '<br>')}</div>`;
      messagesEl.appendChild(userRow);

      const t0 = performance.now();
      let loopTurns = 0;
      const MAX_AGENT_TURNS = 6;

      while (loopTurns < MAX_AGENT_TURNS) {
        loopTurns++;

        // Assistant Skeleton
        const asstRow = document.createElement('div');
        asstRow.className = 'msg-row assistant';
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
        messagesEl.scrollTop = messagesEl.scrollHeight;

        const bubbleEl = asstRow.querySelector('.bubble');
        const thinkBlock = bubbleEl.querySelector('.think-block');
        const thinkContent = bubbleEl.querySelector('.think-content');
        const msgContent = bubbleEl.querySelector('.msg-content');

        let rawContent = '';
        let rawThinking = '';
        let thinkStart = performance.now();
        let isThinking = false;

        const payloadMessages = [];
        if (currentPersona && currentPersona.trim()) {
          payloadMessages.push({ role: 'system', content: currentPersona.trim() });
        }
        payloadMessages.push(...history);

        const requestBody = {
          messages: payloadMessages,
          stream: true,
          reasoning_effort: currentEffort,
          chat_template_kwargs: { enable_thinking: currentEffort !== 'off' }
        };
        if (toolsEnabled && availableTools.length > 0) {
          requestBody.tools = availableTools;
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
            msgContent.innerHTML = `<div class="msg-error">${escapeHtml(head)}<small>${escapeHtml(detail)}</small></div>`;
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
                    }
                    rawContent += delta.content;
                    msgContent.innerHTML = renderMarkdown(rawContent) + '<span class="cursor-blink"></span>';
                  }
                  messagesEl.scrollTop = messagesEl.scrollHeight;
                } catch (_) {}
              }
            }
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
          }

          msgContent.innerHTML = renderMarkdown(rawContent);

          // Check if assistant called tools
          const toolCalls = parseToolCallsFromText(rawContent);
          if (toolsEnabled && toolCalls.length > 0) {
            history.push({ role: 'assistant', content: rawContent });

            for (const tc of toolCalls) {
              const toolCard = document.createElement('div');
              toolCard.className = 'tool-call-block';
              toolCard.innerHTML = `
                <div class="tool-header">
                  <span>⚡ Windows Tool: <strong>${escapeHtml(tc.name)}</strong></span>
                  <span style="font-size:0.75rem; opacity:0.8;">Running...</span>
                </div>
                <div class="tool-output">Executing in PowerShell...</div>`;
              bubbleEl.appendChild(toolCard);
              messagesEl.scrollTop = messagesEl.scrollHeight;

              const result = await executeLocalTool(tc.name, tc.arguments);
              const outputEl = toolCard.querySelector('.tool-output');
              toolCard.querySelector('.tool-header span:last-child').textContent = result.is_error ? 'Error' : 'Complete';
              outputEl.textContent = result.output;

              // Append tool response turn to history
              history.push({
                role: 'tool',
                content: `<tool_response>\n${result.output}\n</tool_response>`
              });
            }
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
            msgContent.innerHTML = renderMarkdown(rawContent);
            const cursor = msgContent.querySelector('.cursor-blink');
            if (cursor) cursor.remove();
            timingInfo.textContent = 'Generation stopped by user';
            history.push({ role: 'assistant', content: rawContent });
            break;
          }
          msgContent.innerHTML = `<div class="msg-error">Lost the connection to the server.<small>${escapeHtml(err.message)} - is shadow serve still running?</small></div>`;
          failed = true;
          break;
        }
      }

      const dt = (performance.now() - t0) / 1000;
      if (failed) {
        timingInfo.textContent = 'Failed - nothing was generated';
      } else if (timingInfo.textContent !== 'Generation stopped by user') {
        timingInfo.textContent = `Completed in ${dt.toFixed(2)}s`;
      }
      isGenerating = false;
      currentAbortController = null;
      sendBtn.style.display = 'flex';
      stopBtn.style.display = 'none';
      sendBtn.disabled = false;
      inputEl.focus();
      pollTelemetry();
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

        // Process Monitor View - CPU
        if (d.cpu) {
          setTxt('cpu-proc-val', d.cpu.process_percent.toFixed(1) + '%');
          setTxt('cpu-proc-cores', d.cpu.process_cores_used.toFixed(1));
          setTxt('cpu-sys-val', d.cpu.system_percent.toFixed(1) + '%');
          setTxt('cpu-proc-txt', d.cpu.process_percent.toFixed(1) + '%');
          setTxt('cpu-sys-txt', d.cpu.system_percent.toFixed(1) + '%');
          setTxt('cpu-cores-used-txt', `${d.cpu.process_cores_used.toFixed(1)} / ${d.cpu.logical_cores}`);
          setTxt('cpu-cores-txt', d.cpu.logical_cores);
          setTxt('cpu-cores-badge', d.cpu.logical_cores + ' Cores');
          setWidth('cpu-bar', Math.min(d.cpu.process_percent * 2, 100) + '%');
        }

        // GPU & VRAM
        if (d.gpu) {
          const usedG = (d.gpu.vram_used_bytes / (1024**3)).toFixed(2);
          const totalG = (d.gpu.vram_total_bytes / (1024**3)).toFixed(1);
          const freeG = (d.gpu.vram_free_bytes / (1024**3)).toFixed(2);
          setTxt('gpu-device-badge', d.gpu.name || 'NVIDIA GPU');
          setTxt('gpu-util-val', d.gpu.utilization_percent.toFixed(1) + '%');
          setTxt('gpu-util-txt', d.gpu.utilization_percent.toFixed(1) + '%');
          setTxt('gpu-mem-txt', d.gpu.memory_controller_percent.toFixed(1) + '%');
          setTxt('gpu-temp-txt', d.gpu.temperature_c > 0 ? `${d.gpu.temperature_c} °C` : 'N/A');
          setTxt('vram-used-val', usedG);
          setTxt('vram-total-val', totalG + ' GiB');
          setTxt('vram-used-total-txt', `${usedG} / ${totalG} GiB`);
          setTxt('vram-free-txt', freeG + ' GiB');
          setTxt('vram-slots-txt', d.gpu.expert_cache_slots + ' slots');
          setTxt('vram-hit-rate-txt', d.gpu.expert_cache_hit_rate.toFixed(1) + '%');
          setWidth('gpu-bar', Math.min(d.gpu.utilization_percent, 100) + '%');

          // Tier 1 stats
          setTxt('tier1-slots', d.gpu.expert_cache_slots);
          setTxt('tier1-hits', d.gpu.expert_cache_hits.toLocaleString());
          setTxt('tier1-misses', d.gpu.expert_cache_misses.toLocaleString());
        }

        // RAM
        if (d.ram) {
          const usedG = (d.ram.system_used_bytes / (1024**3)).toFixed(1);
          const totalG = (d.ram.system_total_bytes / (1024**3)).toFixed(1);
          const availG = (d.ram.system_available_bytes / (1024**3)).toFixed(1);
          setTxt('ram-used-val', usedG);
          setTxt('ram-total-val', totalG + ' GiB');
          setTxt('ram-avail-txt', availG + ' GiB');
          setTxt('ram-load-badge', d.ram.system_load_percent + '% Load');
          setTxt('ram-proc-ws-txt', formatBytes(d.ram.process_working_set_bytes));
          setTxt('ram-pinned-txt', formatBytes(d.ram.pinned_tier_bytes));
          const pct = d.ram.system_total_bytes > 0 ? (d.ram.system_used_bytes / d.ram.system_total_bytes) * 100 : 0;
          setWidth('ram-bar', Math.min(pct, 100) + '%');

          // Tier 2 stats
          setTxt('tier2-cap', formatBytes(d.ram.pinned_tier_bytes));
          setTxt('tier2-experts', d.ram.pinned_tier_experts);
          setTxt('tier2-hits', d.ram.ram_served_hits.toLocaleString());
          setTxt('tier3-reads', d.ram.file_served_misses.toLocaleString());
        }

        // Engine & Process
        if (d.process) {
          setTxt('engine-pid-badge', 'PID: ' + d.process.pid);
          setTxt('engine-speed-val', d.process.tok_per_sec.toFixed(1));
          if (d.process.model) {
            const parts = d.process.model.split(/[\\/]/);
            setTxt('engine-model-txt', parts[parts.length - 1]);
            const mEl = document.getElementById('engine-model-txt');
            if (mEl) mEl.title = d.process.model;
          }
          const up = d.process.uptime_s;
          const mins = Math.floor(up / 60);
          const secs = Math.floor(up % 60);
          setTxt('engine-uptime-txt', `${mins}m ${secs}s`);
          setTxt('engine-requests-txt', d.process.requests.toLocaleString());
          setTxt('engine-tokens-txt', d.process.completion_tokens.toLocaleString());
          setTxt('engine-mtp-txt', d.process.dflash_active ? 'On (DFlash drafter)' : d.process.mtp_active ? 'MTP head loaded (experimental)' : 'Off');
          setTxt('engine-queue-txt', d.process.queued);
        }
      } catch (err) {
        statusPill.textContent = 'Server offline';
        statusPill.className = 'pill offline';
      }
    }

    pollTelemetry();
    setInterval(pollTelemetry, 1000);
  </script>
</body>
</html>
"#;

const MONITOR_HTML: &str = r#"<!DOCTYPE html>
<html lang="en" class="dark h-full">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="0; url=/#monitor">
  <title>Redirecting to Shadow Monitor...</title>
  <script>
    window.location.href = '/#monitor';
  </script>
</head>
<body style="background:#090a0f; color:#8492a6; font-family:sans-serif; display:flex; align-items:center; justify-content:center; height:100vh;">
  <p>Loading Shadow Process Monitor... <a href="/" style="color:#6366f1;">Return to Chat</a></p>
</body>
</html>
"#;
