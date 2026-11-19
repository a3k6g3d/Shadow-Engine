//! ShadowZero's HTTP server (docs/SHADOWZERO_PLAN.md, stage 4): an OpenAI-style `/v1/chat/completions` (whole or streamed as
//! server-sent events, with tool definitions and `<tool_call>` parsing), `/health`, `/v1/models`, `/v1/status`, `/props`
//! and `/metrics`, over hyper, in front of one engine thread (engine.rs).  Requests queue; one is answered at a time.
//!
//! Sampling: see sampler.rs.

pub mod auth;
pub mod convdisk;
pub mod cpuinfo;
pub mod engine;
pub mod events;
pub mod mcp;
pub mod memory;

pub mod orchestrator;
pub mod pc_control;
pub mod sessions;
pub mod sampler;
pub mod semsearch;
pub mod telemetry;
pub mod ui;
pub mod vision;
pub mod webtools;

use bytes::Bytes;
use engine::{Event, Status};
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{json, Value};
use shadowzero_model::chat::{parse_tool_calls, Effort, Message, Options, Role, ToolCall};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

type Body = BoxBody<Bytes, Infallible>;

struct App {
    engine: Mutex<std::sync::mpsc::Sender<engine::Cmd>>,
    status: Arc<Mutex<Status>>,
    started: std::time::Instant,
    /// Requests handed to the engine (the engine counts the ones it finished: the difference is the queue).
    sent: std::sync::atomic::AtomicU64,
    telemetry: telemetry::TelemetryService,
    /// Pushes telemetry ticks, engine-status changes and tool results to every
    /// open browser tab over `/v1/events` (see events.rs).
    events: events::EventHub,
    /// Remote pairing: the live pin, the paired phone's bearer-token hash (auth.rs).
    pairing: Arc<auth::Pairing>,
}

pub struct ServeConfig {
    pub engine: engine::Config,
    pub addr: std::net::SocketAddr,
}

/// Folder where generated photos and videos are stored and served from (`/v1/media/<file>`).
pub fn media_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SHADOWZERO_MEDIA_DIR") {
        let t = p.trim();
        if !t.is_empty() {
            return std::path::PathBuf::from(t);
        }
    }
    if let Some(root) = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
    {
        if root.is_dir() {
            return root.join("outputs");
        }
    }
    std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("outputs")
}

/// Runs the server until the process ends.
pub fn run(cfg: ServeConfig) -> Result<(), String> {
    if cfg.addr.port() != 0 {
        std::env::set_var(
            "SHADOWZERO_SERVE_URL",
            format!("http://127.0.0.1:{}", cfg.addr.port()),
        );
    }
    let mdir = media_dir();
    let _ = std::fs::create_dir_all(&mdir);
    std::env::set_var("SHADOWZERO_MEDIA_DIR", &mdir);
    mcp::start_background();
    let status = Arc::new(Mutex::new(Status {
        state: "loading",
        ..Status::default()
    }));
    let tx = engine::start(cfg.engine, status.clone());
    let app = Arc::new(App {
        engine: Mutex::new(tx),
        status,
        started: std::time::Instant::now(),
        sent: std::sync::atomic::AtomicU64::new(0),
        telemetry: telemetry::TelemetryService::new(),
        events: events::EventHub::new(),
        pairing: Arc::new(auth::Pairing::load()),
    });
    pc_control::bind_events(app.events.clone());
    let pump_app = app.clone();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(cfg.addr)
            .await
            .map_err(|e| format!("{}: {e}", cfg.addr))?;
        eprintln!("shadowzero serve: listening on http://{}", cfg.addr);
        // The push pump: every second it publishes a telemetry snapshot, and it
        // publishes an engine-status event whenever the state string changes (the
        // model loading -> ready transition, an error, a reload).  The UI subscribes
        // to this instead of polling, so the dashboard updates the moment things
        // happen rather than up to a second later.  It runs for the life of the
        // server; the broadcast channel drops events for tabs that fall behind.
        tokio::spawn(async move {
            let mut last_state = String::new();
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                ticker.tick().await;
                let s = pump_app.status.lock().unwrap().clone();
                let queued = pump_app
                    .sent
                    .load(std::sync::atomic::Ordering::SeqCst)
                    .saturating_sub(s.finished);
                let uptime = pump_app.started.elapsed().as_secs_f64();
                let snap = pump_app.telemetry.snapshot(&s, uptime, queued);
                pump_app.events.publish(json!({"kind": "telemetry", "data": snap}));
                if s.state != last_state {
                    last_state = s.state.to_string();
                    pump_app.events.publish(json!({"kind": "status", "state": s.state, "busy": s.busy, "error": s.error}));
                }
            }
        });
        loop {
            let (stream, peer) = listener.accept().await.map_err(|e| e.to_string())?;
            let app = app.clone();
            tokio::spawn(async move {
                let svc = hyper::service::service_fn(move |req| handle(req, app.clone(), peer));
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(hyper_util::rt::TokioIo::new(stream), svc)
                    .await;
            });
        }
    })
}

fn full(status: StatusCode, ctype: &str, body: impl Into<Bytes>) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", ctype)
        .body(Full::new(body.into()).boxed())
        .unwrap()
}

fn json_resp(status: StatusCode, v: &Value) -> Response<Body> {
    full(status, "application/json", v.to_string())
}

fn error(status: StatusCode, msg: &str) -> Response<Body> {
    json_resp(
        status,
        &json!({"error": {"message": msg, "type": "invalid_request_error", "code": status.as_u16()}}),
    )
}

/// One value out of a URL query string (`path=crates/foo.rs`), percent-decoded.
fn query_param(q: Option<&str>, key: &str) -> String {
    let Some(q) = q else { return String::new() };
    for pair in q.split('&') {
        let mut it = pair.splitn(2, '=');
        if it.next() == Some(key) {
            let raw = it.next().unwrap_or("");
            let mut out = String::with_capacity(raw.len());
            let mut chars = raw.bytes();
            while let Some(b) = chars.next() {
                if b == b'%' {
                    let hex: String = (&mut chars).take(2).map(|b| b as char).collect();
                    if let Ok(n) = u8::from_str_radix(&hex, 16) {
                        out.push(n as char);
                        continue;
                    }
                    out.push('%');
                    out.push_str(&hex);
                } else if b == b'+' {
                    out.push(' ');
                } else {
                    out.push(b as char);
                }
            }
            return out;
        }
    }
    String::new()
}

async fn handle(req: Request<Incoming>, app: Arc<App>, peer: std::net::SocketAddr) -> Result<Response<Body>, Infallible> {
    let path = req.uri().path().to_string();
    let local = auth::is_local(peer);
    let origin = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    // A browser on another machine sends OPTIONS before its POST/DELETE; it gets the
    // allowed verbs back and no route is looked up.  A native client sends no Origin
    // at all, so it never preflights and cors() below leaves its replies untouched.
    if req.method() == Method::OPTIONS {
        let mut resp = full(StatusCode::NO_CONTENT, "text/plain; charset=utf-8", "");
        auth::cors(origin.as_deref(), &mut resp);
        return Ok(resp);
    }
    // Everything under /v1 from a peer that is not this machine needs the paired
    // bearer token (auth.rs); the /pair endpoints themselves stay open so a phone
    // can fetch its token, and /pair/pin is answered on loopback only.  A server
    // bound to 127.0.0.1 therefore behaves exactly as it did before auth existed.
    if !local && path.starts_with("/v1") && !app.pairing.check(&auth::presented(&req)) {
        return Ok(error(
            StatusCode::UNAUTHORIZED,
            "not paired: ask the desktop for its pin (GET /pair/pin) and send it to POST /pair",
        ));
    }
    let mut resp = match (req.method(), path.as_str()) {
        (&Method::GET, "/") | (&Method::GET, "/index.html") => ui::index_response(),
        (&Method::GET, "/monitor") | (&Method::GET, "/monitor.html") => ui::monitor_response(),
        (&Method::GET, "/v1/telemetry") | (&Method::GET, "/telemetry") => {
            let s = app.status.lock().unwrap().clone();
            let queued = app
                .sent
                .load(std::sync::atomic::Ordering::SeqCst)
                .saturating_sub(s.finished);
            let uptime = app.started.elapsed().as_secs_f64();
            let mut snap = app.telemetry.snapshot(&s, uptime, queued);
            if let Some(obj) = snap.as_object_mut() {
                obj.insert("pc_control".into(), pc_control::status_json());
            }
            json_resp(StatusCode::OK, &snap)
        }
        (&Method::GET, "/health") => {
            let s = app.status.lock().unwrap().clone();
            match s.state {
                "ready" => json_resp(StatusCode::OK, &json!({"status": "ok"})),
                "loading" | "parked" => json_resp(
                    StatusCode::SERVICE_UNAVAILABLE,
                    &json!({"status": s.state}),
                ),
                _ => json_resp(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &json!({"status": "error", "error": s.error}),
                ),
            }
        }
        (&Method::GET, "/v1/models") => {
            let s = app.status.lock().unwrap().clone();
            json_resp(
                StatusCode::OK,
                &json!({"object": "list", "data": [
                {"id": s.model, "object": "model", "created": 0, "owned_by": "shadow"}]}),
            )
        }
        (&Method::GET, "/v1/status") => {
            let s = app.status.lock().unwrap().clone();
            json_resp(
                StatusCode::OK,
                &json!({
                    "state": s.state, "error": s.error, "model": s.model, "busy": s.busy,
                    "queued": app.sent.load(std::sync::atomic::Ordering::SeqCst).saturating_sub(s.finished),
                    "uptime_s": app.started.elapsed().as_secs_f64(), "load_s": s.load_secs,
                    "requests": s.requests, "prompt_tokens": s.prompt_tokens, "cached_tokens": s.cached_tokens,
                    "completion_tokens": s.completion_tokens,
                    "decode_tokens_per_second": if s.decode_secs > 0.0 { s.completion_tokens as f64 / s.decode_secs } else { 0.0 },
                    "session_tokens": s.session_tokens, "n_ctx": s.n_ctx, "tiers": s.report,
                    "available_ram_gib": shadowzero_cuda::pinned::available_ram() as f64 / (1u64 << 30) as f64,
                }),
            )
        }
        (&Method::GET, "/props") => {
            let s = app.status.lock().unwrap().clone();
            json_resp(
                StatusCode::OK,
                &json!({
                    "model_path": s.model, "n_ctx": s.n_ctx, "engine": "shadow", "version": env!("CARGO_PKG_VERSION"),
                    "sampling": "temperature, top_k, top_p, min_p, repeat_penalty / repetition_penalty, presence_penalty, frequency_penalty, seed; greedy when none is sent or temperature is 0",
                    "sampling_defaults": {"temperature": sampler::MODEL_TEMP, "top_k": sampler::MODEL_TOP_K, "top_p": sampler::MODEL_TOP_P},
                    "tools": true, "streaming": true, "default_generation_settings": {"n_ctx": s.n_ctx},
                }),
            )
        }
        (&Method::GET, "/metrics") => {
            let s = app.status.lock().unwrap().clone();
            let text = format!(
                "# TYPE shadow_requests_total counter\nshadow_requests_total {}\n\
                 # TYPE shadow_prompt_tokens_total counter\nshadow_prompt_tokens_total {}\n\
                 # TYPE shadow_cached_prompt_tokens_total counter\nshadow_cached_prompt_tokens_total {}\n\
                 # TYPE shadow_completion_tokens_total counter\nshadow_completion_tokens_total {}\n\
                 # TYPE shadow_prompt_seconds_total counter\nshadow_prompt_seconds_total {:.3}\n\
                 # TYPE shadow_decode_seconds_total counter\nshadow_decode_seconds_total {:.3}\n\
                 # TYPE shadow_busy gauge\nshadow_busy {}\n\
                 # TYPE shadow_available_ram_bytes gauge\nshadow_available_ram_bytes {}\n",
                s.requests, s.prompt_tokens, s.cached_tokens, s.completion_tokens, s.prompt_secs, s.decode_secs,
                s.busy as u8, shadowzero_cuda::pinned::available_ram());
            full(StatusCode::OK, "text/plain; version=0.0.4", text)
        }
        (&Method::GET, "/v1/tools") => json_resp(StatusCode::OK, &orchestrator::tool_definitions()),
        (&Method::GET, "/v1/memory") => json_resp(StatusCode::OK, &memory::list()),
        (&Method::GET, "/v1/browser/screenshot") => {
            // blocking: the browser server answers in a second or two
            match tokio::task::spawn_blocking(mcp::screenshot).await {
                Ok(Ok(o)) => json_resp(
                    StatusCode::OK,
                    &json!({"image": o.images.first(), "text": o.text}),
                ),
                Ok(Err(e)) => error(StatusCode::NOT_FOUND, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (&Method::POST, "/v1/memory") | (&Method::PUT, _) | (&Method::DELETE, _)
            if path.starts_with("/v1/memory") =>
        {
            memory_api(req, &path).await
        }
        (&Method::GET, "/v1/sessions") => {
            match tokio::task::spawn_blocking(sessions::list).await {
                Ok(rows) => json_resp(StatusCode::OK, &Value::Array(rows)),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (&Method::POST, "/v1/sessions") | (&Method::POST, "/v1/sessions/import")
        | (&Method::PUT, _) | (&Method::DELETE, _) if path.starts_with("/v1/sessions") => {
            sessions_api(req, &path).await
        }
        (&Method::GET, _) if path.starts_with("/v1/sessions/") => {
            let id = path.trim_start_matches("/v1/sessions/").to_string();
            match tokio::task::spawn_blocking(move || sessions::load(&id)).await {
                Ok(Ok(v)) => json_resp(StatusCode::OK, &v),
                Ok(Err(e)) => error(StatusCode::NOT_FOUND, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (&Method::GET, "/v1/projects") => {
            match tokio::task::spawn_blocking(sessions::list_projects).await {
                Ok(rows) => json_resp(StatusCode::OK, &Value::Array(rows)),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (&Method::POST, _) | (&Method::PUT, _) | (&Method::DELETE, _)
            if path == "/v1/projects" || path.starts_with("/v1/projects/") =>
        {
            projects_api(req, &path).await
        }
        (&Method::POST, "/v1/tools/execute") => tool_execute(req, app.clone()).await,
        (&Method::GET, "/v1/pc-control") => json_resp(StatusCode::OK, &pc_control::status_json()),
        (&Method::POST, "/v1/pc-control") => {
            if !local {
                error(StatusCode::FORBIDDEN, "PC Control can only be toggled from this PC")
            } else {
                let bytes = match req.into_body().collect().await {
                    Ok(b) => b.to_bytes(),
                    Err(e) => return Ok(error(StatusCode::BAD_REQUEST, &e.to_string())),
                };
                let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                let enable = body
                    .get("enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let st = pc_control::set_enabled(enable);
                json_resp(StatusCode::OK, &st)
            }
        }
        (&Method::POST, "/v1/engine/park") => engine_park(&app, local).await,
        (&Method::POST, "/v1/engine/resume") => engine_resume(&app, local).await,
        (m, _) if (*m == Method::GET || *m == Method::HEAD) && path.starts_with("/v1/media/") => {
            serve_media(&path, &req, *m == Method::HEAD).await
        }
        (&Method::GET, "/v1/events") => sse_events(app.clone()),
        (&Method::POST, "/v1/chat/completions") => chat(req, app).await,
        // for shadowzero-bench, which ends each run's server this way; only ever bound to what --host says
        (&Method::POST, "/shutdown") => {
            tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                std::process::exit(0);
            });
            json_resp(StatusCode::OK, &json!({"status": "shutting down"}))
        }
        // Remote pairing (auth.rs, docs/PLAN_APK_2026-11-13.md Phase 1): the pin is
        // shown on this machine only - the four digits never cross the LAN - while
        // POST /pair is open to any peer, since that is how a phone gets its token
        // in the first place.  /pair/status lets it ask whether a pin is waiting
        // before it shows the input field.
        (&Method::GET, "/pair/pin") => {
            if local {
                json_resp(StatusCode::OK, &app.pairing.pin())
            } else {
                error(StatusCode::FORBIDDEN, "the pin is shown on the desktop only")
            }
        }
        (&Method::GET, "/pair/status") => json_resp(StatusCode::OK, &app.pairing.status()),
        (&Method::POST, "/pair") => {
            let bytes = match req.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(e) => return Ok(error(StatusCode::BAD_REQUEST, &e.to_string())),
            };
            let typed = match serde_json::from_slice::<Value>(&bytes) {
                Ok(v) => v
                    .get("pin")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
                Err(e) => {
                    return Ok(error(StatusCode::BAD_REQUEST, &format!("bad JSON body: {e}")))
                }
            };
            match app.pairing.pair(&typed) {
                // The token shows once, here, and is what the phone keeps.
                Ok(token) => json_resp(StatusCode::OK, &json!({"token": token})),
                Err(e) => error(StatusCode::UNAUTHORIZED, &e.message()),
            }
        }
        // Forgets this phone: a remote caller's own token must match, so a stray
        // request cannot unpair everybody (auth::unpair); an empty token is only
        // honoured for this machine, where the pin dialog lives (auth::is_local).
        (&Method::DELETE, "/pair") => {
            let presented = auth::presented(&req);
            if !local && presented.trim().is_empty() {
                error(StatusCode::FORBIDDEN, "a remote device must forget itself with its own token")
            } else {
                match app.pairing.unpair(&presented) {
                    Ok(()) => json_resp(StatusCode::OK, &json!({"paired": false})),
                    Err(e) => error(StatusCode::FORBIDDEN, &e),
                }
            }
        }
        _ => error(
            StatusCode::NOT_FOUND,
            &format!("no route for {} {path}", req.method()),
        ),
    };
    // A browser across the LAN sent an Origin and needs it reflected; a native
    // client sends none, so its replies stay untouched.
    auth::cors(origin.as_deref(), &mut resp);
    Ok(resp)
}

/// What a picture is in the prompt text: the chat template's own marker; the engine widens the one pad token to the picture's rows.
pub const IMAGE_MARK: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// The data URL of an `image_url` content part (`{"image_url": {"url": ...}}` or `{"image_url": "..."}`).
fn image_url(p: &Value) -> Option<&str> {
    let u = p.get("image_url")?;
    u.get("url").and_then(Value::as_str).or_else(|| u.as_str())
}

/// Text that came from a message (a page a tool fetched, a file, what the user typed, the model's own earlier words) with every
/// special-token string in it (`<|image_pad|>`, `<|im_start|>`, ...) disarmed by a zero-width space after its `<`: the prompt is encoded
/// with special tokens on (the chat template's own markers must be), so a literal `<|image_pad|>` in a fetched web page would otherwise
/// become a picture slot ("0 pictures sent but the prompt has 2 picture slots") and `<|im_start|>` in a page could open a fake turn.
/// Only the shape `<|name|>` is touched, and once the engine has loaded the tokenizer (`set_special_tokens`) only the names that really are special tokens of the model: text
/// such as `<|role|>` in a document is not a token, and changing it would change the model's input (and every hash of a run on that text) for nothing.
pub fn neutralize_specials(s: &str) -> String {
    neutralize_with(s, SPECIALS.get())
}

static SPECIALS: std::sync::OnceLock<std::collections::HashSet<String>> =
    std::sync::OnceLock::new();

/// The special tokens' strings of the loaded model, from the engine's tokenizer; until it is called every `<|name|>` counts.
pub fn set_special_tokens<'a>(strings: impl Iterator<Item = &'a str>) {
    let _ = SPECIALS.set(strings.map(str::to_string).collect());
}

fn neutralize_with(s: &str, specials: Option<&std::collections::HashSet<String>>) -> String {
    if !s.contains("<|") {
        return s.to_string();
    }
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'<' && b.get(i + 1) == Some(&b'|') {
            let mut j = i + 2;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            if j > i + 2
                && b.get(j) == Some(&b'|')
                && b.get(j + 1) == Some(&b'>')
                && specials.is_none_or(|sp| sp.contains(&s[i..j + 2]))
            {
                out.push('<');
                out.push('\u{200b}');
                i += 1;
                continue;
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn disarm_xml_tags(s: &str, tags: &[&str]) -> String {
    if !s.contains('<') {
        return s.to_string();
    }
    let mut out = s.to_string();
    for tag in tags {
        if out.contains(tag) {
            let disarmed = format!("<\u{200b}{}", &tag[1..]);
            out = out.replace(tag, &disarmed);
        }
    }
    out
}

/// Strips one outer `<tool_response>...</tool_response>` wrapper (which the web UI adds) and
/// disarms inner Qwen XML control tokens (`</tool_response>`, `<tool_call>`, `</think>`, ...)
/// inside tool outputs / user messages so source code containing those literals cannot prematurely
/// close a `<tool_response>` block during tokenization.
fn disarm_control_tags(role: Role, text: String) -> String {
    match role {
        Role::Tool => {
            let mut body = text.as_str();
            if let Some(rest) = body
                .trim_start()
                .strip_prefix("<tool_response>")
                .and_then(|s| s.trim_end().strip_suffix("</tool_response>"))
            {
                let rest = rest
                    .strip_prefix("\r\n")
                    .or_else(|| rest.strip_prefix('\n'))
                    .unwrap_or(rest);
                body = rest
                    .strip_suffix("\r\n")
                    .or_else(|| rest.strip_suffix('\n'))
                    .unwrap_or(rest);
            }
            disarm_xml_tags(
                body,
                &[
                    "<tool_response>",
                    "</tool_response>",
                    "<tool_call>",
                    "</tool_call>",
                    "<think>",
                    "</think>",
                ],
            )
        }
        Role::User => disarm_xml_tags(
            &text,
            &["<tool_response>", "</tool_response>", "<think>", "</think>"],
        ),
        _ => text,
    }
}

/// A message's content: a string, or an array of parts: text ones are joined, a picture becomes `IMAGE_MARK` (its bytes come from
/// `parse_images`); anything else is refused.  Text is passed through `neutralize_specials`, so only a real picture part makes a slot.
fn content_text(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(neutralize_specials(s)),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => Ok(neutralize_specials(
                    p.get("text").and_then(Value::as_str).unwrap_or(""),
                )),
                Some("image_url") if image_url(p).is_some() => Ok(IMAGE_MARK.to_string()),
                other => Err(format!(
                    "content part type {other:?} is not supported (text and images only)"
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|v| v.concat()),
        _ => Err("message content must be a string or an array of text parts".into()),
    }
}

/// The pictures of a request, in prompt order: the bytes of each `data:image/...;base64,` URL in the messages' content arrays.
pub fn parse_images(body: &Value) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::new();
    for m in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for p in m
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if p.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            let url = image_url(p).ok_or("an image_url part needs a url")?;
            let b64 = url
                .strip_prefix("data:")
                .and_then(|r| r.split_once(";base64,"))
                .map(|(_, d)| d)
                .ok_or("pictures must be data:image/...;base64, URLs")?;
            out.push(base64_decode(b64)?);
        }
    }
    Ok(out)
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return Err("the picture's base64 has a bad character".into()),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Formats a `serde_json::Value` like Jinja's `tojson(ensure_ascii=False)` (`", "` and `": "`), ordering common
/// schema keys first so `BTreeMap` does not scramble tool definitions.
fn tojson_serde(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(tojson_serde).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(m) => {
            const ORDER: &[&str] = &[
                "type",
                "function",
                "name",
                "description",
                "parameters",
                "input_schema",
                "properties",
                "required",
                "path",
                "pattern",
                "glob",
                "ignore_case",
                "start",
                "end",
                "oldText",
                "newText",
                "content",
                "command",
                "cwd",
            ];
            let mut items = Vec::with_capacity(m.len());
            for &k in ORDER {
                if let Some(val) = m.get(k) {
                    items.push(format!(
                        "{}: {}",
                        serde_json::to_string(k).unwrap(),
                        tojson_serde(val)
                    ));
                }
            }
            for (k, val) in m {
                if !ORDER.contains(&k.as_str()) {
                    items.push(format!(
                        "{}: {}",
                        serde_json::to_string(k).unwrap(),
                        tojson_serde(val)
                    ));
                }
            }
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Converts a parsed `<parameter=K>V</parameter>` string into a JSON value using the tool's schema when available.
/// The OpenAI `tool_calls` array of parsed calls: arguments as a JSON object string with the parameters in the model's order and typed by the tool's schema.
fn tool_calls_json(calls: &[ToolCall], tools: Option<&Vec<Value>>) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, tc)| {
            let pairs: Vec<String> = tc
                .arguments
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), arg_to_json(tools, &tc.name, k, v)))
                .collect();
            json!({
                "id": format!("call_{i}"),
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": format!("{{{}}}", pairs.join(",")),
                }
            })
        })
        .collect()
}

/// A streamed chat answer for a request that carries `tools` (OpenAI streaming: a tool-using client always streams): text before a `<tool_call>` block goes out as
/// `content` deltas (a tail that could be the start of the marker is held back until it is decided); from the marker on, the text is kept, and at the end the parsed calls
/// go out as one `tool_calls` delta with finish_reason `tool_calls` (a block that does not parse goes out as text).  STATUS 152.
fn stream_with_tools(rx: tokio::sync::mpsc::UnboundedReceiver<Event>, id: String, model: String, created: u64, tools: Vec<Value>) -> Response<Body> {
    const MARK: &str = "<tool_call>";
    struct St {
        rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
        first: bool,
        held: String,
        acc: String,
        in_call: bool,
        done: bool,
    }
    let chunk = move |delta: Value, finish: Value| -> String {
        let c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        format!("data: {c}\n\n")
    };
    let st0 = St { rx, first: true, held: String::new(), acc: String::new(), in_call: false, done: false };
    let s = futures_util::stream::unfold(st0, move |mut st: St| {
        let chunk = chunk.clone();
        let tools = tools.clone();
        async move {
            if st.done {
                return None;
            }
            if st.first {
                st.first = false;
                let b = Bytes::from(chunk(json!({"role": "assistant", "content": ""}), Value::Null));
                return Some((Ok::<_, Infallible>(Frame::data(b)), st));
            }
            loop {
                match st.rx.recv().await {
                    Some(Event::Content(t)) => {
                        if st.in_call {
                            st.acc.push_str(&t);
                            continue;
                        }
                        st.held.push_str(&t);
                        if let Some(i) = st.held.find(MARK) {
                            let before = st.held[..i].to_string();
                            st.acc = st.held[i..].to_string();
                            st.held.clear();
                            st.in_call = true;
                            if !before.is_empty() {
                                return Some((Ok(Frame::data(Bytes::from(chunk(json!({"content": before}), Value::Null)))), st));
                            }
                            continue;
                        }
                        // the longest tail of the held text that is a prefix of the marker stays held (the marker is ASCII: the cut is on a character boundary)
                        let keep = (1..MARK.len()).rev().find(|&k| st.held.ends_with(&MARK[..k])).unwrap_or(0);
                        let cut = st.held.len() - keep;
                        if cut > 0 {
                            let out: String = st.held.drain(..cut).collect();
                            return Some((Ok(Frame::data(Bytes::from(chunk(json!({"content": out}), Value::Null)))), st));
                        }
                    }
                    Some(Event::Reasoning(t)) => {
                        return Some((Ok(Frame::data(Bytes::from(chunk(json!({"reasoning_content": t}), Value::Null)))), st));
                    }
                    Some(Event::Done(stat)) => {
                        let mut out = String::new();
                        let mut finish = json!(stat.finish_reason);
                        if st.in_call {
                            let (_, calls) = parse_tool_calls(&st.acc);
                            if calls.is_empty() {
                                out += &chunk(json!({"content": st.acc.clone()}), Value::Null);
                            } else {
                                let tcs: Vec<Value> = tool_calls_json(&calls, Some(&tools))
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, mut c)| {
                                        c["index"] = json!(i);
                                        c
                                    })
                                    .collect();
                                out += &chunk(json!({"tool_calls": tcs}), Value::Null);
                                finish = json!("tool_calls");
                            }
                        } else if !st.held.is_empty() {
                            out += &chunk(json!({"content": st.held.clone()}), Value::Null);
                        }
                        let (u, t) = usage(&stat);
                        let last = chunk(json!({}), finish);
                        let mut v: Value = serde_json::from_str(&last[6..last.len() - 2]).unwrap();
                        v["usage"] = u;
                        v["timings"] = t;
                        out += &format!("data: {v}\n\ndata: [DONE]\n\n");
                        st.done = true;
                        return Some((Ok(Frame::data(Bytes::from(out))), st));
                    }
                    Some(Event::Error(e)) => {
                        let v = json!({"error": {"message": e}});
                        st.done = true;
                        return Some((Ok(Frame::data(Bytes::from(format!("data: {v}\n\ndata: [DONE]\n\n")))), st));
                    }
                    None => return None,
                }
            }
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(StreamBody::new(s).boxed())
        .unwrap()
}

fn arg_to_json(tools: Option<&Vec<Value>>, fn_name: &str, key: &str, val: &str) -> Value {
    let prop_type = tools.and_then(|ts| {
        ts.iter().find_map(|t| {
            let f = t.get("function").unwrap_or(t);
            if f.get("name").and_then(Value::as_str) != Some(fn_name) {
                return None;
            }
            let schema = f.get("parameters").or_else(|| f.get("input_schema"))?;
            schema.get("properties")?.get(key)?.get("type")?.as_str()
        })
    });
    match prop_type {
        Some("string") => Value::String(val.to_string()),
        Some("integer") | Some("number") | Some("boolean") | Some("array") | Some("object") => {
            serde_json::from_str(val.trim()).unwrap_or_else(|_| Value::String(val.to_string()))
        }
        _ => match serde_json::from_str::<Value>(val.trim()) {
            Ok(v @ (Value::Number(_) | Value::Bool(_) | Value::Array(_) | Value::Object(_))) => v,
            _ => Value::String(val.to_string()),
        },
    }
}

/// Parses a top-level JSON object string into `(key, Value)` pairs in exact source key order (whereas `serde_json::Map`
/// uses `BTreeMap` and alphabetizes keys, which would reorder `<parameter=...>` tags on the next turn and break prefix
/// cache reuse).
fn parse_json_object_ordered(s: &str) -> Result<Vec<(String, Value)>, String> {
    let v: Value = serde_json::from_str(s)
        .map_err(|e| format!("tool_call arguments is not valid JSON: {e}"))?;
    if !v.is_object() {
        return Err("tool_call arguments must be a JSON object".into());
    }
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if bytes.get(i) != Some(&b'{') {
        return Ok(Vec::new());
    }
    i += 1;
    let mut out = Vec::new();
    loop {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        if bytes[i] != b'"' {
            break;
        }
        let key_start = i;
        i += 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                i += 1;
                break;
            }
            i += 1;
        }
        let key: String = serde_json::from_str(&s[key_start..i]).map_err(|e| e.to_string())?;
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b':') {
            i += 1;
        }
        let val_start = i;
        let mut depth = 0i32;
        let mut in_str = false;
        while i < bytes.len() {
            let b = bytes[i];
            if in_str {
                if b == b'\\' {
                    i += 2;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
            } else {
                match b {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    b',' if depth == 0 => break,
                    _ => {}
                }
            }
            i += 1;
        }
        let val: Value = serde_json::from_str(s[val_start..i].trim()).map_err(|e| e.to_string())?;
        out.push((key, val));
    }
    Ok(out)
}

/// The request body as the engine's request (without its channel).
pub fn parse_chat(body: &Value) -> Result<(Vec<Message>, Options, usize, bool), String> {
    let mut opts = Options::default();
    if let Some(tools_val) = body.get("tools") {
        let arr = tools_val.as_array().ok_or("\"tools\" must be an array")?;
        for t in arr {
            let fn_obj = t.get("function").unwrap_or(t);
            if fn_obj.get("name").and_then(Value::as_str).is_none() {
                return Err("each tool definition must have a \"name\"".into());
            }
            opts.tools.push(tojson_serde(t));
        }
    }
    let msgs = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("\"messages\" must be an array")?;
    let mut out = Vec::new();
    for m in msgs {
        let role = match m.get("role").and_then(Value::as_str) {
            Some("system") | Some("developer") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            other => return Err(format!("unknown role {other:?}")),
        };
        let content = disarm_control_tags(
            role,
            content_text(m.get("content").unwrap_or(&Value::Null))?,
        );
        let mut msg = Message::new(role, &content);
        msg.reasoning = m
            .get("reasoning_content")
            .and_then(Value::as_str)
            .map(neutralize_specials);
        if let Some(tcs) = m.get("tool_calls").and_then(Value::as_array) {
            for tc in tcs {
                let fn_obj = tc.get("function").unwrap_or(tc);
                let name = fn_obj
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("tool_call must have a function \"name\"")?
                    .to_string();
                let pairs: Vec<(String, Value)> = match fn_obj.get("arguments") {
                    Some(Value::String(s)) if !s.trim().is_empty() => parse_json_object_ordered(s)?,
                    Some(Value::Object(map)) => {
                        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    }
                    _ => Vec::new(),
                };
                let mut arguments = Vec::with_capacity(pairs.len());
                for (k, v) in pairs {
                    let vs = match v {
                        Value::String(s) => neutralize_specials(&s),
                        other => neutralize_specials(&tojson_serde(&other)),
                    };
                    arguments.push((k, vs));
                }
                msg.tool_calls.push(ToolCall { name, arguments });
            }
        }
        out.push(msg);
    }
    if let Some(k) = body.get("chat_template_kwargs") {
        if let Some(t) = k.get("enable_thinking").and_then(Value::as_bool) {
            opts.thinking = Some(t);
        }
    }
    if let Some(effort_str) = body.get("reasoning_effort").and_then(Value::as_str) {
        if effort_str == "off" || effort_str == "none" {
            opts.thinking = Some(false);
        } else {
            opts.effort = match effort_str {
                "low" => Effort::Low,
                "medium" => Effort::Medium,
                "high" => Effort::High,
                "xhigh" | "x-high" | "extra_high" => Effort::XHigh,
                _ => Effort::XHigh,
            };
        }
    }
    let max_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize; // 0: no limit given, the engine lets the reply run to the end of the context
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    Ok((out, opts, max_tokens, stream))
}

/// The sampling parameters of a request body (OpenAI names, plus llama.cpp's `top_k`, `min_p`, `repeat_penalty`).
fn parse_sampling(body: &Value) -> sampler::Sampling {
    let f = |k: &str| body.get(k).and_then(Value::as_f64).map(|v| v as f32);
    let seed = body
        .get("seed")
        .and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64)));
    sampler::Sampling {
        temperature: f("temperature"),
        top_k: body.get("top_k").and_then(Value::as_u64).map(|v| v as usize),
        top_p: f("top_p"),
        min_p: f("min_p"),
        repeat_penalty: f("repeat_penalty").or_else(|| f("repetition_penalty")),
        presence_penalty: f("presence_penalty"),
        frequency_penalty: f("frequency_penalty"),
        seed,
    }
}

fn usage(st: &engine::Stats) -> (Value, Value) {
    (
        json!({"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
            "total_tokens": st.prompt_tokens + st.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": st.cached_tokens}}),
        json!({"prompt_n": st.prompt_tokens - st.cached_tokens, "prompt_ms": st.prompt_ms,
            "prompt_per_second": (st.prompt_tokens - st.cached_tokens) as f64 / (st.prompt_ms / 1e3).max(1e-9),
            "predicted_n": st.completion_tokens, "predicted_ms": st.predicted_ms,
            "predicted_per_second": st.completion_tokens as f64 / (st.predicted_ms / 1e3).max(1e-9),
            "cache_n": st.cached_tokens}),
    )
}

/// POST /v1/memory {text, kind}: add a note; PUT /v1/memory/<id> {text}: reword it; DELETE /v1/memory/<id>: forget it.
async fn memory_api(req: Request<Incoming>, path: &str) -> Response<Body> {
    let method = req.method().clone();
    let id: Option<u64> = path
        .strip_prefix("/v1/memory/")
        .and_then(|s| s.parse().ok());
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let text = body.get("text").and_then(Value::as_str).unwrap_or("");
    let done = |r: Result<(), String>| match r {
        Ok(()) => json_resp(StatusCode::OK, &memory::list()),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    };
    match (method, id) {
        (Method::POST, None) => match memory::add(
            body.get("kind").and_then(Value::as_str).unwrap_or("user"),
            text,
        ) {
            Ok(_) => json_resp(StatusCode::OK, &memory::list()),
            Err(e) => error(StatusCode::BAD_REQUEST, &e),
        },
        (Method::PUT, Some(id)) => done(memory::update(id, text)),
        (Method::DELETE, Some(id)) => done(memory::forget(id)),
        _ => error(StatusCode::NOT_FOUND, "no such memory request"),
    }
}

/// POST /v1/sessions {history, notes, id?}: save a conversation, answers {id};
/// POST /v1/sessions/import [...]: move a browser's localStorage store over once;
/// PUT /v1/sessions/<id> {title}: rename; DELETE /v1/sessions/<id>: forget.
/// The files are small; still off the async threads, the way the screenshot route is.
async fn sessions_api(req: Request<Incoming>, path: &str) -> Response<Body> {
    let method = req.method().clone();
    let id: Option<String> = path
        .strip_prefix("/v1/sessions/")
        .filter(|s| !s.is_empty() && *s != "import")
        .map(|s| s.to_string());
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let import = path == "/v1/sessions/import";
    match (method, id) {
        (Method::POST, None) if import => {
            match tokio::task::spawn_blocking(move || sessions::import_all(&body)).await {
                Ok(Ok(n)) => json_resp(StatusCode::OK, &json!({"imported": n})),
                Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (Method::POST, None) => {
            match tokio::task::spawn_blocking(move || sessions::save(&body)).await {
                Ok(Ok(id)) => json_resp(StatusCode::OK, &json!({"id": id})),
                Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (Method::PUT, Some(id)) => {
            match tokio::task::spawn_blocking(move || sessions::update(&id, &body)).await {
                Ok(Ok(())) => json_resp(StatusCode::OK, &json!({"ok": true})),
                Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (Method::DELETE, Some(id)) => {
            match tokio::task::spawn_blocking(move || sessions::delete(&id)).await {
                Ok(Ok(())) => json_resp(StatusCode::OK, &json!({"ok": true})),
                Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        _ => error(StatusCode::NOT_FOUND, "no such session request"),
    }
}

/// POST /v1/projects {name, description?}: create a project/job;
/// PUT /v1/projects/<id> {name?, description?, archived?}: update;
/// DELETE /v1/projects/<id>: remove project (keeps conversations unassigned);
/// POST /v1/projects/<id>/archive-older {keep_newest?}: archive oldest conversations in project.
async fn projects_api(req: Request<Incoming>, path: &str) -> Response<Body> {
    let method = req.method().clone();
    let tail = path.strip_prefix("/v1/projects").unwrap_or("");
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    if method == Method::POST && (tail.is_empty() || tail == "/") {
        return match tokio::task::spawn_blocking(move || sessions::create_project(&body)).await {
            Ok(Ok(p)) => json_resp(StatusCode::OK, &p),
            Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
            Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };
    }
    let rest = tail.trim_start_matches('/');
    if method == Method::POST && rest.ends_with("/archive-older") {
        let id = rest.trim_end_matches("/archive-older").to_string();
        let keep = body
            .get("keep_newest")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(3);
        return match tokio::task::spawn_blocking(move || {
            sessions::archive_older_in_project(&id, keep)
        })
        .await
        {
            Ok(Ok(n)) => json_resp(StatusCode::OK, &json!({"archived": n})),
            Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
            Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };
    }
    let id = rest.to_string();
    if id.is_empty() || id.contains('/') {
        return error(StatusCode::NOT_FOUND, "no such project request");
    }
    match method {
        Method::PUT => {
            match tokio::task::spawn_blocking(move || sessions::update_project(&id, &body)).await {
                Ok(Ok(p)) => json_resp(StatusCode::OK, &p),
                Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        Method::DELETE => {
            match tokio::task::spawn_blocking(move || sessions::delete_project(&id)).await {
                Ok(Ok(())) => json_resp(StatusCode::OK, &json!({"ok": true})),
                Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        _ => error(StatusCode::NOT_FOUND, "no such project request"),
    }
}

async fn tool_execute(req: Request<Incoming>, app: Arc<App>) -> Response<Body> {
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let v: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                &format!("invalid tool invocation json: {e}"),
            )
        }
    };
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let arguments = v.get("arguments").cloned().unwrap_or(Value::Null);
    let inv = orchestrator::ToolInvocation { name, arguments };
    // a tool can take a while (a command, a web page, an MCP server): not on the server's async threads.
    // Dropping this handler (e.g. client aborts /v1/tools/execute on Stop / Send now) signals the tool to terminate cleanly.
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    struct CancelOnDrop(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let _cancel_guard = CancelOnDrop(Arc::clone(&cancel));
    let cancel_bg = Arc::clone(&cancel);
    let name = inv.name.clone();
    let res = tokio::task::spawn_blocking(move || {
        orchestrator::execute_tool_with_cancel(&inv, &cancel_bg)
    })
    .await
    .unwrap_or_else(|e| orchestrator::ToolResult {
        name,
        output: format!("the tool crashed: {e}"),
        is_error: true,
        images: vec![],
    });
    // Tell every open tab about the result the moment it lands: the UI's tool
    // cards, the Browser pane and the permission gate see it without waiting for
    // the HTTP response of the request that produced it (which still arrives too).
    app.events.publish(json!({"kind": "tool", "data": res.to_json()}));
    json_resp(StatusCode::OK, &res.to_json())
}

/// `/v1/events`: a long-lived `text/event-stream` carrying every event the hub
/// publishes (telemetry, engine-status, tool results).  A tab that falls behind
/// the 256-deep channel loses frames rather than blocking the publisher, and a
/// closed tab's receiver is simply dropped when the stream ends.
fn sse_events(app: Arc<App>) -> Response<Body> {
    let rx = app.events.subscribe();
    let s = futures_util::stream::unfold((rx, None::<Bytes>, false), |(mut rx, pending, done)| {
        async move {
            if let Some(b) = pending {
                return Some((Ok::<_, Infallible>(Frame::data(b)), (rx, None, done)));
            }
            if done {
                return None;
            }
            match rx.recv().await {
                Ok(v) => Some((
                    Ok(Frame::data(Bytes::from(format!("data: {v}\n\n")))),
                    (rx, None, false),
                )),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // The tab missed frames; keep going with the next one.
                    Some((
                        Ok(Frame::data(Bytes::from("data: {\"kind\":\"lagged\"}\n\n"))),
                        (rx, None, false),
                    ))
                }
                // The hub is gone (the server is shutting down): end the stream.
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    Some((Ok(Frame::data(Bytes::from("data: [DONE]\n\n"))), (rx, None, true)))
                }
            }
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(StreamBody::new(s).boxed())
        .unwrap()
}

async fn engine_lifecycle(app: &App, local: bool, park: bool) -> Response<Body> {
    if !local {
        return error(
            StatusCode::FORBIDDEN,
            "engine lifecycle endpoints are loopback-only",
        );
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let cmd = if park {
        engine::Cmd::Park(tx)
    } else {
        engine::Cmd::Resume(tx)
    };
    if app.engine.lock().unwrap().send(cmd).is_err() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "the engine has stopped");
    }
    let timeout = std::time::Duration::from_secs(if park { 120 } else { 600 });
    match tokio::task::spawn_blocking(move || rx.recv_timeout(timeout)).await {
        Ok(Ok(Ok(()))) => json_resp(
            StatusCode::OK,
            &json!({"status": if park { "parked" } else { "ready" }}),
        ),
        Ok(Ok(Err(e))) => error(StatusCode::INTERNAL_SERVER_ERROR, &e),
        Ok(Err(_)) => error(
            StatusCode::GATEWAY_TIMEOUT,
            if park {
                "timed out waiting for the engine to park"
            } else {
                "timed out waiting for the engine to reload"
            },
        ),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn engine_park(app: &Arc<App>, local: bool) -> Response<Body> {
    engine_lifecycle(app, local, true).await
}

async fn engine_resume(app: &Arc<App>, local: bool) -> Response<Body> {
    engine_lifecycle(app, local, false).await
}

fn is_safe_media_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.contains("..")
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

fn media_mime(name: &str) -> Option<&'static str> {
    let ext = name.rsplit('.').next()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "mp4" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        _ => None,
    }
}

fn parse_byte_range(hdr: &str, size: u64) -> Option<Result<(u64, u64), ()>> {
    let spec = hdr.strip_prefix("bytes=")?.split(',').next()?.trim();
    let (a, b) = spec.split_once('-')?;
    if a.is_empty() && b.is_empty() {
        return Some(Err(()));
    }
    if size == 0 {
        return Some(Err(()));
    }
    if a.is_empty() {
        let suffix: u64 = b.parse().ok()?;
        if suffix == 0 {
            return Some(Err(()));
        }
        let start = size.saturating_sub(suffix);
        return Some(Ok((start, size - 1)));
    }
    let start: u64 = a.parse().ok()?;
    let end = if b.is_empty() {
        size - 1
    } else {
        b.parse::<u64>().ok()?.min(size - 1)
    };
    if start >= size || start > end {
        return Some(Err(()));
    }
    Some(Ok((start, end)))
}

async fn serve_media(path: &str, req: &Request<Incoming>, head: bool) -> Response<Body> {
    let dir = media_dir();
    let path = path.to_string();
    let range_hdr = req
        .headers()
        .get("range")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    match tokio::task::spawn_blocking(move || read_media_response(&dir, &path, range_hdr.as_deref(), head)).await {
        Ok(resp) => resp,
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn read_media_response(
    dir: &std::path::Path,
    path: &str,
    range_hdr: Option<&str>,
    head: bool,
) -> Response<Body> {
    use std::io::{Read, Seek, SeekFrom};
    let name = path.trim_start_matches("/v1/media/");
    if !is_safe_media_name(name) {
        return error(StatusCode::BAD_REQUEST, "invalid media filename");
    }
    let Some(mime) = media_mime(name) else {
        return error(StatusCode::NOT_FOUND, "unsupported media type");
    };
    let file_path = dir.join(name);
    let Ok(mut f) = std::fs::File::open(&file_path) else {
        return error(StatusCode::NOT_FOUND, "media file not found");
    };
    let Ok(meta) = f.metadata() else {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "failed to stat media file");
    };
    let size = meta.len();
    if let Some(rh) = range_hdr {
        if let Some(parsed) = parse_byte_range(rh, size) {
            match parsed {
                Err(()) => {
                    return Response::builder()
                        .status(StatusCode::RANGE_NOT_SATISFIABLE)
                        .header("content-range", format!("bytes */{size}"))
                        .body(Full::new(Bytes::new()).boxed())
                        .unwrap();
                }
                Ok((start, end)) => {
                    let len = (end - start + 1) as usize;
                    let mut buf = Vec::new();
                    if !head {
                        if f.seek(SeekFrom::Start(start)).is_err() {
                            return error(StatusCode::INTERNAL_SERVER_ERROR, "seek failed");
                        }
                        buf.resize(len, 0);
                        if f.read_exact(&mut buf).is_err() {
                            return error(StatusCode::INTERNAL_SERVER_ERROR, "read failed");
                        }
                    }
                    return Response::builder()
                        .status(StatusCode::PARTIAL_CONTENT)
                        .header("content-type", mime)
                        .header("accept-ranges", "bytes")
                        .header("content-range", format!("bytes {start}-{end}/{size}"))
                        .header("content-length", len.to_string())
                        .header("cache-control", "public, max-age=3600")
                        .body(Full::new(Bytes::from(buf)).boxed())
                        .unwrap();
                }
            }
        }
    }
    let mut buf = Vec::new();
    if !head && f.read_to_end(&mut buf).is_err() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "read failed");
    }
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", mime)
        .header("accept-ranges", "bytes")
        .header("content-length", size.to_string())
        .header("cache-control", "public, max-age=3600")
        .body(Full::new(Bytes::from(buf)).boxed())
        .unwrap()
}

async fn chat(req: Request<Incoming>, app: Arc<App>) -> Response<Body> {
    let state = app.status.lock().unwrap().state;
    if state != "ready" && state != "parked" {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("the model is {state}"),
        );
    }
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("reading the body: {e}")),
    };
    let body: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                &format!("the body is not JSON: {e}"),
            )
        }
    };
    let (messages, opts, max_tokens, stream) = match parse_chat(&body) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e),
    };
    let images = match parse_images(&body) {
        Ok(i) => i,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e),
    };
    let cache_prompt = body
        .get("cache_prompt")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let sampling = parse_sampling(&body);
    let model = app.status.lock().unwrap().model.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    app.sent.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if app
        .engine
        .lock()
        .unwrap()
        .send(engine::Cmd::Chat(engine::Request {
            messages,
            opts,
            max_tokens,
            images,
            cache_prompt,
            sampling,
            tx,
        }))
        .is_err()
    {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "the engine has stopped");
    }
    let id = format!(
        "chatcmpl-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if !stream {
        let (mut content, mut reasoning) = (String::new(), String::new());
        let resp = loop {
            match rx.recv().await {
                Some(Event::Content(s)) => content += &s,
                Some(Event::Reasoning(s)) => reasoning += &s,
                Some(Event::Done(st)) => {
                    let (u, t) = usage(&st);
                    let (clean_content, calls) = parse_tool_calls(&content);
                    let mut msg = json!({"role": "assistant", "content": clean_content});
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = json!(reasoning);
                    }
                    let mut finish_reason = st.finish_reason;
                    if !calls.is_empty() {
                        let tools_arr = body.get("tools").and_then(Value::as_array);
                        msg["tool_calls"] = Value::Array(tool_calls_json(&calls, tools_arr));
                        finish_reason = "tool_calls";
                    }
                    break json_resp(
                        StatusCode::OK,
                        &json!({
                        "id": id, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
                        "usage": u, "timings": t}),
                    );
                }
                Some(Event::Error(e)) => break error(StatusCode::BAD_REQUEST, &e),
                None => {
                    break error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "the engine dropped the request",
                    )
                }
            }
        };
        return resp;
    }

    // a request with tools: the text streams until a `<tool_call>` block starts, the block is held back and goes out as `tool_calls` deltas at the end
    if let Some(tools) = body.get("tools").and_then(Value::as_array).filter(|t| !t.is_empty()) {
        return stream_with_tools(rx, id, model, created, tools.clone());
    }
    // streamed: one server-sent event per piece, a final chunk with finish_reason, usage and timings, then [DONE]
    let chunk = move |delta: Value, finish: Value| -> Bytes {
        let c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        Bytes::from(format!("data: {c}\n\n"))
    };
    let first = chunk(json!({"role": "assistant", "content": ""}), Value::Null);
    let s =
        futures_util::stream::unfold((rx, Some(first), false), move |(mut rx, pending, done)| {
            let chunk = chunk.clone();
            async move {
                if let Some(b) = pending {
                    return Some((Ok::<_, Infallible>(Frame::data(b)), (rx, None, false)));
                }
                if done {
                    return None;
                }
                let b = match rx.recv().await {
                    Some(Event::Content(s)) => chunk(json!({"content": s}), Value::Null),
                    Some(Event::Reasoning(s)) => {
                        chunk(json!({"reasoning_content": s}), Value::Null)
                    }
                    Some(Event::Done(st)) => {
                        let (u, t) = usage(&st);
                        let last = chunk(json!({}), json!(st.finish_reason));
                        let mut v: Value =
                            serde_json::from_slice(&last[6..last.len() - 2]).unwrap();
                        v["usage"] = u;
                        v["timings"] = t;
                        let b = Bytes::from(format!("data: {v}\n\ndata: [DONE]\n\n"));
                        return Some((Ok(Frame::data(b)), (rx, None, true)));
                    }
                    Some(Event::Error(e)) => {
                        let v = json!({"error": {"message": e}});
                        let b = Bytes::from(format!("data: {v}\n\ndata: [DONE]\n\n"));
                        return Some((Ok(Frame::data(b)), (rx, None, true)));
                    }
                    None => return None,
                };
                Some((Ok(Frame::data(b)), (rx, None, false)))
            }
        });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(StreamBody::new(s).boxed())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `data:` payloads of a streamed answer, in order (the `[DONE]` line left out).
    async fn streamed(events: Vec<Event>) -> Vec<Value> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        for e in events {
            tx.send(e).unwrap();
        }
        drop(tx);
        let tools = vec![json!({"type": "function", "function": {"name": "add", "parameters": {"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}})];
        let resp = stream_with_tools(rx, "id".into(), "m".into(), 1, tools);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn done(reason: &'static str) -> Event {
        Event::Done(engine::Stats { finish_reason: reason, ..Default::default() })
    }

    fn text_of(chunks: &[Value]) -> String {
        chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect()
    }

    #[tokio::test]
    async fn a_streamed_answer_with_tools_but_no_call_is_plain_text() {
        let c = streamed(vec![Event::Content("Hello ".into()), Event::Content("wor".into()), Event::Content("ld <".into()), done("stop")]).await;
        assert_eq!(text_of(&c), "Hello world <");
        assert_eq!(c.last().unwrap()["choices"][0]["finish_reason"], "stop");
        assert!(c.iter().all(|x| x["choices"][0]["delta"].get("tool_calls").is_none()));
    }

    #[tokio::test]
    async fn a_tool_call_split_across_pieces_goes_out_as_tool_calls_deltas() {
        let c = streamed(vec![
            Event::Content("Sure. <tool".into()),
            Event::Content("_call>
<function=add>
<parameter=a>
2
</parameter>
<parameter=b>
3
</parameter>
</function>
</tool_call>".into()),
            Event::Content("<tool_call>
<function=add>
<parameter=a>
10
</parameter>
<parameter=b>
20
</parameter>
</function>
</tool_call>".into()),
            done("stop"),
        ])
        .await;
        assert_eq!(text_of(&c), "Sure. ");
        let calls: Vec<&Value> = c.iter().filter_map(|x| x["choices"][0]["delta"].get("tool_calls")).collect();
        assert_eq!(calls.len(), 1, "one delta holds the calls");
        let tc = calls[0].as_array().unwrap();
        assert_eq!(tc.len(), 2);
        assert_eq!((tc[0]["index"].as_i64(), tc[1]["index"].as_i64()), (Some(0), Some(1)));
        assert_eq!(tc[0]["function"]["name"], "add");
        assert_eq!(serde_json::from_str::<Value>(tc[0]["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"a": 2, "b": 3}));
        assert_eq!(serde_json::from_str::<Value>(tc[1]["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"a": 10, "b": 20}));
        assert_eq!(c.last().unwrap()["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn a_tool_call_block_that_does_not_parse_is_text() {
        let c = streamed(vec![Event::Content("<tool_call>garbage".into()), done("stop")]).await;
        assert_eq!(text_of(&c), "<tool_call>garbage");
        assert_eq!(c.last().unwrap()["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn parses_a_chat_request() {
        let b = json!({"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": [
            {"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]}],
            "max_tokens": 7, "stream": true, "chat_template_kwargs": {"enable_thinking": false}});
        let (m, o, n, s) = parse_chat(&b).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].content, "hi there");
        assert_eq!((o.thinking, n, s), (Some(false), 7, true));
    }

    #[test]
    fn special_token_strings_in_message_text_are_disarmed() {
        assert_eq!(
            neutralize_specials("a <|image_pad|> b"),
            "a <\u{200b}|image_pad|> b"
        );
        assert_eq!(
            neutralize_specials("<|im_start|>user<|im_end|>"),
            "<\u{200b}|im_start|>user<\u{200b}|im_end|>"
        );
        // not the shape of a special token: untouched, and so is text without any
        for s in [
            "a <|> b",
            "x <| y |> z",
            "<||>",
            "<|a b|>",
            "plain text, no markers",
            "naïve <| ünï",
        ] {
            assert_eq!(neutralize_specials(s), s);
        }
        // with the model's special tokens known, only those are touched
        let sp: std::collections::HashSet<String> = ["<|im_start|>", "<|image_pad|>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            neutralize_with("a <|role|> b <|im_start|> c", Some(&sp)),
            "a <|role|> b <\u{200b}|im_start|> c"
        );
        assert_eq!(neutralize_with("a <|role|> b", Some(&sp)), "a <|role|> b");
        // literal </tool_response> inside a tool output (e.g. reading source code) is disarmed without double-wrapping
        let tool_wrapped = "<tool_response>\n247: const val TOOL_CLOSE = \"\\n</tool_response>\"\n</tool_response>";
        let disarmed = disarm_control_tags(Role::Tool, tool_wrapped.to_string());
        assert_eq!(
            disarmed,
            "247: const val TOOL_CLOSE = \"\\n<\u{200b}/tool_response>\""
        );
        // a fetched page that mentions the picture token does not make picture slots; a real picture still does
        let b = json!({"messages": [
            {"role": "tool", "content": "{\"image_token\":\"<|image_pad|>\",\"v\":\"<|vision_start|>\"}"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                {"type": "text", "text": "the literal <|vision_start|><|image_pad|><|vision_end|> is just text"}]}]});
        let (m, ..) = parse_chat(&b).unwrap();
        assert_eq!(m[0].content.matches("<|image_pad|>").count(), 0);
        assert_eq!(m[1].content.matches("<|image_pad|>").count(), 1);
        assert_eq!(m[1].content.matches(IMAGE_MARK).count(), 1);
        assert_eq!(parse_images(&b).unwrap().len(), 1);
    }

    #[test]
    fn refuses_what_it_cannot_render() {
        assert!(parse_chat(&json!({"messages": [], "tools": [{"type": "function"}]})).is_err());
        assert!(parse_chat(
            &json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
        )
        .is_err());
        // a picture becomes the template's marker, and its bytes come out of the data URL in order
        let b = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}, {"type": "text", "text": "what is this?"}]}]});
        let (m, ..) = parse_chat(&b).unwrap();
        assert_eq!(m[0].content, format!("{IMAGE_MARK}what is this?"));
        assert_eq!(parse_images(&b).unwrap(), vec![b"hello".to_vec()]);
        assert!(parse_images(&json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}]})).is_err());
        assert!(parse_chat(&json!({"messages": [{"role": "wizard", "content": "x"}]})).is_err());
        assert!(parse_chat(&json!({"nope": 1})).is_err());
    }

    #[test]
    fn parses_tools_and_tool_calls() {
        let b = json!({
            "messages": [
                {"role": "user", "content": "read foo.py"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"foo.py\",\"start\":1,\"end\":8}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_0", "content": "1|hi"}
            ],
            "tools": [{"name": "read_file", "description": "read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"}}}}]
        });
        let (m, o, _, _) = parse_chat(&b).unwrap();
        assert_eq!(o.tools.len(), 1);
        assert_eq!(m[1].tool_calls.len(), 1);
        assert_eq!(m[1].tool_calls[0].name, "read_file");
        assert_eq!(
            m[1].tool_calls[0].arguments,
            vec![
                ("path".into(), "foo.py".into()),
                ("start".into(), "1".into()),
                ("end".into(), "8".into())
            ]
        );
    }

    #[test]
    fn serves_index_html() {
        let resp = ui::index_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/html; charset=utf-8"
        );
    }

    #[test]
    fn media_name_and_byte_range_validation() {
        assert!(is_safe_media_name("photo_1.png"));
        assert!(is_safe_media_name("clip-2026.mp4"));
        assert!(!is_safe_media_name(""));
        assert!(!is_safe_media_name(".hidden.png"));
        assert!(!is_safe_media_name("../secret.png"));
        assert!(!is_safe_media_name("dir/file.png"));
        assert_eq!(parse_byte_range("bytes=0-99", 1000), Some(Ok((0, 99))));
        assert_eq!(parse_byte_range("bytes=500-", 1000), Some(Ok((500, 999))));
        assert_eq!(parse_byte_range("bytes=-200", 1000), Some(Ok((800, 999))));
        assert_eq!(parse_byte_range("bytes=900-5000", 1000), Some(Ok((900, 999))));
        assert_eq!(parse_byte_range("bytes=1000-1200", 1000), Some(Err(())));
        assert_eq!(parse_byte_range("bytes=200-100", 1000), Some(Err(())));
    }

    #[tokio::test]
    async fn serves_media_files_and_byte_ranges() {
        let dir = std::env::temp_dir().join(format!("sz_media_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("clip.mp4");
        std::fs::write(&file, b"0123456789abcdef").unwrap();

        let full = read_media_response(&dir, "/v1/media/clip.mp4", None, false);
        assert_eq!(full.status(), StatusCode::OK);
        assert_eq!(full.headers().get("content-type").unwrap(), "video/mp4");
        assert_eq!(full.headers().get("content-length").unwrap(), "16");
        let body = full.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"0123456789abcdef");

        let part = read_media_response(&dir, "/v1/media/clip.mp4", Some("bytes=4-9"), false);
        assert_eq!(part.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(part.headers().get("content-range").unwrap(), "bytes 4-9/16");
        let pbody = part.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&pbody[..], b"456789");

        let bad = read_media_response(&dir, "/v1/media/../secret.mp4", None, false);
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);

        let oob = read_media_response(&dir, "/v1/media/clip.mp4", Some("bytes=100-200"), false);
        assert_eq!(oob.status(), StatusCode::RANGE_NOT_SATISFIABLE);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
