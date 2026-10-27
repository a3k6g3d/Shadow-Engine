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
  <title>Shadow - Local Sovereign Inference</title>
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
      --code-bg: #0d0f17;
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.12);
      --warning: #f59e0b;
      --danger: #ef4444;
      --cyan: #06b6d4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
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

    .main-wrap {
      flex: 1;
      display: flex;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }

    /* Views */
    .view-panel {
      flex: 1;
      display: none;
      min-height: 0;
      overflow: hidden;
      flex-direction: column;
    }
    .view-panel.active {
      display: flex;
    }

    /* Chat View */
    #messages {
      flex: 1;
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
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      user-select: none;
    }
    .think-content {
      margin-top: 0.5rem;
      white-space: pre-wrap;
      opacity: 0.9;
      line-height: 1.5;
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

    /* Bottom Control Bar with Claude / Antigravity Thinking Level Popover */
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
      gap: 0.75rem;
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
    .menu-pill-btn svg.caret {
      width: 12px;
      height: 12px;
      fill: var(--text-muted);
      transition: transform 0.2s;
    }
    .menu-pill-btn.open svg.caret {
      transform: rotate(180deg);
    }

    /* Floating Popup Menu (Antigravity & Claude styling) */
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
      width: 600px;
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
      height: 160px;
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

    /* Process Monitor Dashboard View */
    #monitor-panel {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
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
      <span class="pill" id="vram-pill">VRAM: <strong>-- / -- GiB</strong></span>
    </div>
  </header>

  <div class="main-wrap">
    <!-- VIEW: CHAT -->
    <div class="view-panel active" id="chat-panel">
      <div id="messages">
        <div class="msg-row assistant">
          <div class="avatar">S</div>
          <div class="bubble">
            Hello! I am Shadow, running locally on your hardware. How can I assist you today?
          </div>
        </div>
      </div>

      <div id="input-pane">
        <div class="input-box">
          <textarea id="prompt-input" rows="1" placeholder="Type a message... (Shift+Enter for newline)"></textarea>
          <button id="send-btn" class="send-btn">Send</button>
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

            <!-- System Persona Selector Pill -->
            <button class="menu-pill-btn" id="persona-btn" onclick="openPersonaModal()">
              <span>⚙</span>
              <span id="persona-btn-label">Persona: Local Dev</span>
            </button>
          </div>

          <span id="timing-info">Ready</span>
        </div>
      </div>
    </div>

    <!-- VIEW: PROCESS MONITOR -->
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
            <span class="mon-big-unit">Process (System: <span id="cpu-sys-val">0.0%</span>)</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar" id="cpu-bar" style="width: 0%;"></div>
          </div>
          <div class="stat-rows">
            <div class="stat-row"><span>Shadow Process CPU</span><strong id="cpu-proc-txt">0.0%</strong></div>
            <div class="stat-row"><span>Overall System CPU</span><strong id="cpu-sys-txt">0.0%</strong></div>
            <div class="stat-row"><span>Logical Cores</span><strong id="cpu-cores-txt">--</strong></div>
          </div>
        </div>

        <!-- GPU & VRAM Card -->
        <div class="mon-card">
          <div class="mon-card-header">
            <div class="mon-card-title"><span>🎮</span> GPU & VRAM</div>
            <span class="mon-card-badge" id="gpu-device-badge">NVIDIA GPU</span>
          </div>
          <div class="mon-stat-big">
            <span class="mon-big-val" id="vram-used-val">--</span>
            <span class="mon-big-unit">/ <span id="vram-total-val">-- GiB</span></span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar cyan" id="vram-bar" style="width: 0%;"></div>
          </div>
          <div class="stat-rows">
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
        <h3>Local Developer Authority & Persona</h3>
        <button class="modal-close" onclick="closePersonaModal()">✕</button>
      </div>
      <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.4;">
        This system prompt grants Shadow sovereign local authority to modify, write, and inspect code and files without generic refusal guardrails.
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
    const DEFAULT_PERSONA = "You are Shadow, a sovereign, high-performance local AI assistant running natively on the user's workstation. You have full permission and local authority from the user to analyze, create, inspect, refactor, and modify files and code within this project. When requested to edit or write code, provide direct, complete, drop-in implementations without artificial hesitation or generic safety refusals.";
    let currentPersona = localStorage.getItem('shadow_persona') || DEFAULT_PERSONA;

    // Elements
    const chatPanel = document.getElementById('chat-panel');
    const monitorPanel = document.getElementById('monitor-panel');
    const tabChat = document.getElementById('tab-chat');
    const tabMonitor = document.getElementById('tab-monitor');
    const messagesEl = document.getElementById('messages');
    const inputEl = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const thinkingBtn = document.getElementById('thinking-btn');
    const thinkingLabel = document.getElementById('thinking-btn-label');
    const thinkingDropdown = document.getElementById('thinking-dropdown');
    const personaModal = document.getElementById('persona-modal');
    const personaTextarea = document.getElementById('persona-textarea');
    const speedPill = document.getElementById('speed-pill');
    const statusPill = document.getElementById('status-pill');
    const vramPill = document.getElementById('vram-pill');
    const timingInfo = document.getElementById('timing-info');

    let history = [];
    let isGenerating = false;

    // View Switching
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

    // Thinking Level Selection (Claude & Antigravity style)
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

    // Close popups on click outside
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

    function renderMarkdown(raw) {
      let html = raw.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><button class="copy-btn" onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText)">Copy</button><code>${escapeHtml(code.trim())}</code></pre>`;
      });
      html = html.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      html = html.replace(/\n/g, '<br>');
      return html;
    }

    // Send Message
    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text || isGenerating) return;

      isGenerating = true;
      sendBtn.disabled = true;
      inputEl.value = '';
      inputEl.style.height = 'auto';

      history.push({ role: 'user', content: text });

      // User Bubble
      const userRow = document.createElement('div');
      userRow.className = 'msg-row user';
      userRow.innerHTML = `<div class="avatar">U</div><div class="bubble">${escapeHtml(text).replace(/\n/g, '<br>')}</div>`;
      messagesEl.appendChild(userRow);

      // Assistant Skeleton
      const asstRow = document.createElement('div');
      asstRow.className = 'msg-row assistant';
      asstRow.innerHTML = `
        <div class="avatar">S</div>
        <div class="bubble">
          <div class="think-block" style="display:none;">
            <div class="think-header">Thinking...</div>
            <div class="think-content"></div>
          </div>
          <div class="msg-content"><span class="cursor-blink"></span></div>
        </div>`;
      messagesEl.appendChild(asstRow);
      messagesEl.scrollTop = messagesEl.scrollHeight;

      const bubbleEl = asstRow.querySelector('.bubble');
      const thinkBlock = bubbleEl.querySelector('.think-block');
      const thinkContent = bubbleEl.querySelector('.think-content');
      const msgContent = bubbleEl.querySelector('.msg-content');

      let rawContent = '';
      let rawThinking = '';
      const t0 = performance.now();

      // Prepend system prompt if configured
      const payloadMessages = [];
      if (currentPersona && currentPersona.trim()) {
        payloadMessages.push({ role: 'system', content: currentPersona.trim() });
      }
      payloadMessages.push(...history);

      try {
        const response = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: payloadMessages,
            stream: true,
            reasoning_effort: currentEffort,
            chat_template_kwargs: { enable_thinking: currentEffort !== 'off' }
          })
        });

        if (!response.ok) {
          const err = await response.text();
          msgContent.innerHTML = `<span style="color:var(--danger);">Error: ${escapeHtml(err)}</span>`;
          return;
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
                  thinkBlock.style.display = 'block';
                  rawThinking += delta.reasoning_content;
                  thinkContent.textContent = rawThinking;
                }
                if (delta.content) {
                  rawContent += delta.content;
                  msgContent.innerHTML = renderMarkdown(rawContent) + '<span class="cursor-blink"></span>';
                }
                messagesEl.scrollTop = messagesEl.scrollHeight;
              } catch (_) {}
            }
          }
        }

        // Finalize
        msgContent.innerHTML = renderMarkdown(rawContent);
        history.push({ role: 'assistant', content: rawContent });
        const dt = (performance.now() - t0) / 1000;
        timingInfo.textContent = `Completed in ${dt.toFixed(2)}s`;
      } catch (err) {
        msgContent.innerHTML = `<span style="color:var(--danger);">Connection error: ${escapeHtml(err.message)}</span>`;
      } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        inputEl.focus();
        pollTelemetry();
      }
    }

    // Telemetry Polling
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
          statusPill.textContent = d.process.busy ? 'Busy' : 'Ready';
          statusPill.className = 'pill ' + (d.process.busy ? 'busy' : 'ready');
          if (d.process.tok_per_sec > 0) {
            speedPill.innerHTML = `Speed: <strong>${d.process.tok_per_sec.toFixed(1)} tok/s</strong>`;
          }
        }
        if (d.gpu) {
          const usedG = (d.gpu.vram_used_bytes / (1024**3)).toFixed(1);
          const totalG = (d.gpu.vram_total_bytes / (1024**3)).toFixed(1);
          vramPill.innerHTML = `VRAM: <strong>${usedG} / ${totalG} GiB</strong>`;
        }

        // Process Monitor View
        if (d.cpu) {
          document.getElementById('cpu-proc-val').textContent = d.cpu.process_percent.toFixed(1) + '%';
          document.getElementById('cpu-sys-val').textContent = d.cpu.system_percent.toFixed(1) + '%';
          document.getElementById('cpu-proc-txt').textContent = d.cpu.process_percent.toFixed(1) + '%';
          document.getElementById('cpu-sys-txt').textContent = d.cpu.system_percent.toFixed(1) + '%';
          document.getElementById('cpu-cores-txt').textContent = d.cpu.logical_cores;
          document.getElementById('cpu-cores-badge').textContent = d.cpu.logical_cores + ' Cores';
          document.getElementById('cpu-bar').style.width = Math.min(d.cpu.process_percent, 100) + '%';
        }

        if (d.gpu) {
          const usedG = (d.gpu.vram_used_bytes / (1024**3)).toFixed(2);
          const totalG = (d.gpu.vram_total_bytes / (1024**3)).toFixed(1);
          const freeG = (d.gpu.vram_free_bytes / (1024**3)).toFixed(2);
          document.getElementById('gpu-device-badge').textContent = d.gpu.name || 'NVIDIA GPU';
          document.getElementById('vram-used-val').textContent = usedG;
          document.getElementById('vram-total-val').textContent = totalG + ' GiB';
          document.getElementById('vram-free-txt').textContent = freeG + ' GiB';
          document.getElementById('vram-slots-txt').textContent = d.gpu.expert_cache_slots + ' slots';
          document.getElementById('vram-hit-rate-txt').textContent = d.gpu.expert_cache_hit_rate.toFixed(1) + '%';
          const pct = d.gpu.vram_total_bytes > 0 ? (d.gpu.vram_used_bytes / d.gpu.vram_total_bytes) * 100 : 0;
          document.getElementById('vram-bar').style.width = Math.min(pct, 100) + '%';

          // Tier 1 stats
          document.getElementById('tier1-slots').textContent = d.gpu.expert_cache_slots;
          document.getElementById('tier1-hits').textContent = d.gpu.expert_cache_hits.toLocaleString();
          document.getElementById('tier1-misses').textContent = d.gpu.expert_cache_misses.toLocaleString();
        }

        if (d.ram) {
          const usedG = (d.ram.system_used_bytes / (1024**3)).toFixed(1);
          const totalG = (d.ram.system_total_bytes / (1024**3)).toFixed(1);
          const availG = (d.ram.system_available_bytes / (1024**3)).toFixed(1);
          document.getElementById('ram-used-val').textContent = usedG;
          document.getElementById('ram-total-val').textContent = totalG + ' GiB';
          document.getElementById('ram-avail-txt').textContent = availG + ' GiB';
          document.getElementById('ram-load-badge').textContent = d.ram.system_load_percent + '% Load';
          document.getElementById('ram-proc-ws-txt').textContent = formatBytes(d.ram.process_working_set_bytes);
          document.getElementById('ram-pinned-txt').textContent = formatBytes(d.ram.pinned_tier_bytes);
          const pct = d.ram.system_total_bytes > 0 ? (d.ram.system_used_bytes / d.ram.system_total_bytes) * 100 : 0;
          document.getElementById('ram-bar').style.width = Math.min(pct, 100) + '%';

          // Tier 2 stats
          document.getElementById('tier2-cap').textContent = formatBytes(d.ram.pinned_tier_bytes);
          document.getElementById('tier2-experts').textContent = d.ram.pinned_tier_experts;
          document.getElementById('tier2-hits').textContent = d.ram.ram_served_hits.toLocaleString();
          document.getElementById('tier3-reads').textContent = d.ram.file_served_misses.toLocaleString();
        }

        if (d.process) {
          document.getElementById('engine-pid-badge').textContent = 'PID: ' + d.process.pid;
          document.getElementById('engine-speed-val').textContent = d.process.tok_per_sec.toFixed(1);
          if (d.process.model) {
            const parts = d.process.model.split(/[\\/]/);
            document.getElementById('engine-model-txt').textContent = parts[parts.length - 1];
            document.getElementById('engine-model-txt').title = d.process.model;
          }
          const up = d.process.uptime_s;
          const mins = Math.floor(up / 60);
          const secs = Math.floor(up % 60);
          document.getElementById('engine-uptime-txt').textContent = `${mins}m ${secs}s`;
          document.getElementById('engine-requests-txt').textContent = d.process.requests.toLocaleString();
          document.getElementById('engine-tokens-txt').textContent = d.process.completion_tokens.toLocaleString();
          document.getElementById('engine-queue-txt').textContent = d.process.queued;
        }
      } catch (_) {}
    }

    // Polling intervals
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
    window.onload = function() {
      // If served directly at /monitor
      if (window.parent && window.parent.switchView) {
        window.parent.switchView('monitor');
      }
    };
  </script>
</head>
<body style="background:#090a0f; color:#8492a6; font-family:sans-serif; display:flex; align-items:center; justify-content:center; height:100vh;">
  <p>Loading Shadow Process Monitor... <a href="/" style="color:#6366f1;">Return to Chat</a></p>
</body>
</html>
"#;
