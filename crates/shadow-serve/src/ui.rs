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
      --surface-hover: #1a1e2a;
      --border: #222736;
      --text: #e2e8f0;
      --text-muted: #8492a6;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-light: rgba(99, 102, 241, 0.15);
      --think-bg: rgba(139, 92, 246, 0.08);
      --think-border: rgba(139, 92, 246, 0.25);
      --think-text: #c4b5fd;
      --code-bg: #0d0f17;
      --success: #10b981;
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
      padding: 0.75rem 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .brand svg {
      width: 24px;
      height: 24px;
      fill: var(--accent);
    }
    .stats-bar {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.8rem;
    }
    .pill {
      background: var(--surface-hover);
      border: 1px solid var(--border);
      border-radius: 9999px;
      padding: 0.25rem 0.65rem;
      display: flex;
      align-items: center;
      gap: 0.35rem;
      color: var(--text-muted);
    }
    .pill strong {
      color: var(--text);
    }
    .pill.ready { border-color: rgba(16, 185, 129, 0.3); color: var(--success); }
    .main-wrap {
      flex: 1;
      display: flex;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }
    #chat-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
    }
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
      max-width: 860px;
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
    #input-pane {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 1rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      flex-shrink: 0;
    }
    .input-box {
      max-width: 860px;
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
    .controls-bar {
      max-width: 860px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .toggle-label {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
      user-select: none;
    }
    .toggle-label input { accent-color: var(--accent); }
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
    <div class="brand">
      <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
      <span>Shadow</span>
    </div>
    <div class="stats-bar">
      <span class="pill ready" id="status-pill">Ready</span>
      <span class="pill" id="speed-pill">Speed: <strong>-- tok/s</strong></span>
      <span class="pill" id="vram-pill">Model: <strong id="model-name">Loading...</strong></span>
    </div>
  </header>

  <div class="main-wrap">
    <div id="chat-container">
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
          <label class="toggle-label">
            <input type="checkbox" id="thinking-toggle" checked>
            <span>Enable Reasoning (<think>)</span>
          </label>
          <span id="timing-info">Ready</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    const messagesEl = document.getElementById('messages');
    const inputEl = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const thinkingToggle = document.getElementById('thinking-toggle');
    const speedPill = document.getElementById('speed-pill');
    const modelNameEl = document.getElementById('model-name');
    const timingInfo = document.getElementById('timing-info');

    let history = [];
    let isGenerating = false;

    // Auto-resize input
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

    // Refresh model status
    async function updateStatus() {
      try {
        const res = await fetch('/v1/status');
        if (res.ok) {
          const d = await res.json();
          if (d.model) {
            const parts = d.model.split(/[\\/]/);
            modelNameEl.textContent = parts[parts.length - 1];
          }
          if (d.decode_tokens_per_second > 0) {
            speedPill.innerHTML = `Speed: <strong>${d.decode_tokens_per_second.toFixed(1)} tok/s</strong>`;
          }
        }
      } catch (_) {}
    }
    updateStatus();
    setInterval(updateStatus, 3000);

    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[m]));
    }

    function renderMarkdown(raw) {
      // Code blocks
      let html = raw.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><button class="copy-btn" onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText)">Copy</button><code>${escapeHtml(code.trim())}</code></pre>`;
      });
      // Inline code
      html = html.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);
      // Bold
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      // Line breaks
      html = html.replace(/\n/g, '<br>');
      return html;
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text || isGenerating) return;

      isGenerating = true;
      sendBtn.disabled = true;
      inputEl.value = '';
      inputEl.style.height = 'auto';

      history.push({ role: 'user', content: text });

      // Append user bubble
      const userRow = document.createElement('div');
      userRow.className = 'msg-row user';
      userRow.innerHTML = `<div class="avatar">U</div><div class="bubble">${escapeHtml(text).replace(/\n/g, '<br>')}</div>`;
      messagesEl.appendChild(userRow);

      // Append assistant bubble skeleton
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

      try {
        const response = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: history,
            stream: true,
            chat_template_kwargs: { enable_thinking: thinkingToggle.checked }
          })
        });

        if (!response.ok) {
          const err = await response.text();
          msgContent.innerHTML = `<span style="color:#ef4444;">Error: ${escapeHtml(err)}</span>`;
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
        msgContent.innerHTML = `<span style="color:#ef4444;">Connection error: ${escapeHtml(err.message)}</span>`;
      } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        inputEl.focus();
        updateStatus();
      }
    }
  </script>
</body>
</html>
"#;
