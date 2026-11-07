//! Shadow's HTTP server (docs/SHADOW_PLAN.md, stage 4): an OpenAI-style `/v1/chat/completions` (whole or streamed as
//! server-sent events, with tool definitions and `<tool_call>` parsing), `/health`, `/v1/models`, `/v1/status`, `/props`
//! and `/metrics`, over hyper, in front of one engine thread (engine.rs).  Requests queue; one is answered at a time.
//!
//! Not yet: sampling (greedy only), images.

pub mod cpuinfo;
pub mod convdisk;
pub mod engine;
pub mod mcp;
pub mod memory;
pub mod orchestrator;
pub mod telemetry;
pub mod ui;
pub mod webtools;
pub mod vision;

use bytes::Bytes;
use engine::{Event, Status};
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{json, Value};
use shadow_model::chat::{parse_tool_calls, Effort, Message, Options, Role, ToolCall};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

type Body = BoxBody<Bytes, Infallible>;

struct App {
    engine: Mutex<std::sync::mpsc::Sender<engine::Request>>,
    status: Arc<Mutex<Status>>,
    started: std::time::Instant,
    /// Requests handed to the engine (the engine counts the ones it finished: the difference is the queue).
    sent: std::sync::atomic::AtomicU64,
    telemetry: telemetry::TelemetryService,
}

pub struct ServeConfig {
    pub engine: engine::Config,
    pub addr: std::net::SocketAddr,
}

/// Runs the server until the process ends.
pub fn run(cfg: ServeConfig) -> Result<(), String> {
    mcp::start_background();
    let status = Arc::new(Mutex::new(Status { state: "loading", ..Status::default() }));
    let tx = engine::start(cfg.engine, status.clone());
    let app = Arc::new(App {
        engine: Mutex::new(tx),
        status,
        started: std::time::Instant::now(),
        sent: std::sync::atomic::AtomicU64::new(0),
        telemetry: telemetry::TelemetryService::new(),
    });
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(cfg.addr).await.map_err(|e| format!("{}: {e}", cfg.addr))?;
        eprintln!("shadow serve: listening on http://{}", cfg.addr);
        loop {
            let (stream, _) = listener.accept().await.map_err(|e| e.to_string())?;
            let app = app.clone();
            tokio::spawn(async move {
                let svc = hyper::service::service_fn(move |req| handle(req, app.clone()));
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
    json_resp(status, &json!({"error": {"message": msg, "type": "invalid_request_error", "code": status.as_u16()}}))
}

async fn handle(req: Request<Incoming>, app: Arc<App>) -> Result<Response<Body>, Infallible> {
    let path = req.uri().path().to_string();
    Ok(match (req.method(), path.as_str()) {
        (&Method::GET, "/") | (&Method::GET, "/index.html") => ui::index_response(),
        (&Method::GET, "/monitor") | (&Method::GET, "/monitor.html") => ui::monitor_response(),
        (&Method::GET, "/v1/telemetry") | (&Method::GET, "/telemetry") => {
            let s = app.status.lock().unwrap().clone();
            let queued = app.sent.load(std::sync::atomic::Ordering::SeqCst).saturating_sub(s.finished);
            let uptime = app.started.elapsed().as_secs_f64();
            let snap = app.telemetry.snapshot(&s, uptime, queued);
            json_resp(StatusCode::OK, &snap)
        }
        (&Method::GET, "/health") => {
            let s = app.status.lock().unwrap().clone();
            match s.state {
                "ready" => json_resp(StatusCode::OK, &json!({"status": "ok"})),
                "loading" => json_resp(StatusCode::SERVICE_UNAVAILABLE, &json!({"status": "loading"})),
                _ => json_resp(StatusCode::INTERNAL_SERVER_ERROR, &json!({"status": "error", "error": s.error})),
            }
        }
        (&Method::GET, "/v1/models") => {
            let s = app.status.lock().unwrap().clone();
            json_resp(StatusCode::OK, &json!({"object": "list", "data": [
                {"id": s.model, "object": "model", "created": 0, "owned_by": "shadow"}]}))
        }
        (&Method::GET, "/v1/status") => {
            let s = app.status.lock().unwrap().clone();
            json_resp(StatusCode::OK, &json!({
                "state": s.state, "error": s.error, "model": s.model, "busy": s.busy,
                "queued": app.sent.load(std::sync::atomic::Ordering::SeqCst).saturating_sub(s.finished),
                "uptime_s": app.started.elapsed().as_secs_f64(), "load_s": s.load_secs,
                "requests": s.requests, "prompt_tokens": s.prompt_tokens, "cached_tokens": s.cached_tokens,
                "completion_tokens": s.completion_tokens,
                "decode_tokens_per_second": if s.decode_secs > 0.0 { s.completion_tokens as f64 / s.decode_secs } else { 0.0 },
                "session_tokens": s.session_tokens, "n_ctx": s.n_ctx, "tiers": s.report,
                "available_ram_gib": shadow_cuda::pinned::available_ram() as f64 / (1u64 << 30) as f64,
            }))
        }
        (&Method::GET, "/props") => {
            let s = app.status.lock().unwrap().clone();
            json_resp(StatusCode::OK, &json!({
                "model_path": s.model, "n_ctx": s.n_ctx, "engine": "shadow", "version": env!("CARGO_PKG_VERSION"),
                "sampling": "greedy only (temperature, top_p, seed are accepted and ignored)",
                "tools": true, "streaming": true, "default_generation_settings": {"n_ctx": s.n_ctx},
            }))
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
                s.busy as u8, shadow_cuda::pinned::available_ram());
            full(StatusCode::OK, "text/plain; version=0.0.4", text)
        }
        (&Method::GET, "/v1/tools") => json_resp(StatusCode::OK, &orchestrator::tool_definitions()),
        (&Method::GET, "/v1/memory") => json_resp(StatusCode::OK, &memory::list()),
        (&Method::GET, "/v1/browser/screenshot") => {
            // blocking: the browser server answers in a second or two
            match tokio::task::spawn_blocking(mcp::screenshot).await {
                Ok(Ok(o)) => json_resp(StatusCode::OK, &json!({"image": o.images.first(), "text": o.text})),
                Ok(Err(e)) => error(StatusCode::NOT_FOUND, &e),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
        (&Method::POST, "/v1/memory") | (&Method::PUT, _) | (&Method::DELETE, _) if path.starts_with("/v1/memory") => memory_api(req, &path).await,
        (&Method::POST, "/v1/tools/execute") => tool_execute(req).await,
        (&Method::POST, "/v1/chat/completions") => chat(req, app).await,
        // for shadow-bench, which ends each run's server this way; only ever bound to what --host says
        (&Method::POST, "/shutdown") => {
            tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                std::process::exit(0);
            });
            json_resp(StatusCode::OK, &json!({"status": "shutting down"}))
        }
        _ => error(StatusCode::NOT_FOUND, &format!("no route for {} {path}", req.method())),
    })
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

static SPECIALS: std::sync::OnceLock<std::collections::HashSet<String>> = std::sync::OnceLock::new();

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
            if j > i + 2 && b.get(j) == Some(&b'|') && b.get(j + 1) == Some(&b'>') && specials.is_none_or(|sp| sp.contains(&s[i..j + 2])) {
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

/// A message's content: a string, or an array of parts: text ones are joined, a picture becomes `IMAGE_MARK` (its bytes come from
/// `parse_images`); anything else is refused.  Text is passed through `neutralize_specials`, so only a real picture part makes a slot.
fn content_text(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(neutralize_specials(s)),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => parts.iter().map(|p| match p.get("type").and_then(Value::as_str) {
            Some("text") => Ok(neutralize_specials(p.get("text").and_then(Value::as_str).unwrap_or(""))),
            Some("image_url") if image_url(p).is_some() => Ok(IMAGE_MARK.to_string()),
            other => Err(format!("content part type {other:?} is not supported (text and images only)")),
        }).collect::<Result<Vec<_>, _>>().map(|v| v.concat()),
        _ => Err("message content must be a string or an array of text parts".into()),
    }
}

/// The pictures of a request, in prompt order: the bytes of each `data:image/...;base64,` URL in the messages' content arrays.
pub fn parse_images(body: &Value) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::new();
    for m in body.get("messages").and_then(Value::as_array).into_iter().flatten() {
        for p in m.get("content").and_then(Value::as_array).into_iter().flatten() {
            if p.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            let url = image_url(p).ok_or("an image_url part needs a url")?;
            let b64 = url.strip_prefix("data:").and_then(|r| r.split_once(";base64,")).map(|(_, d)| d)
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
        Value::Array(a) => format!("[{}]", a.iter().map(tojson_serde).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => {
            const ORDER: &[&str] = &[
                "type", "function", "name", "description", "parameters", "input_schema", "properties", "required",
                "path", "pattern", "glob", "ignore_case", "start", "end", "oldText", "newText", "content", "command", "cwd",
            ];
            let mut items = Vec::with_capacity(m.len());
            for &k in ORDER {
                if let Some(val) = m.get(k) {
                    items.push(format!("{}: {}", serde_json::to_string(k).unwrap(), tojson_serde(val)));
                }
            }
            for (k, val) in m {
                if !ORDER.contains(&k.as_str()) {
                    items.push(format!("{}: {}", serde_json::to_string(k).unwrap(), tojson_serde(val)));
                }
            }
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Converts a parsed `<parameter=K>V</parameter>` string into a JSON value using the tool's schema when available.
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
    let v: Value = serde_json::from_str(s).map_err(|e| format!("tool_call arguments is not valid JSON: {e}"))?;
    if !v.is_object() {
        return Err("tool_call arguments must be a JSON object".into());
    }
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }
    if bytes.get(i) != Some(&b'{') { return Ok(Vec::new()); }
    i += 1;
    let mut out = Vec::new();
    loop {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') { i += 1; }
        if i >= bytes.len() || bytes[i] == b'}' { break; }
        if bytes[i] != b'"' { break; }
        let key_start = i;
        i += 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' { i += 2; continue; }
            if bytes[i] == b'"' { i += 1; break; }
            i += 1;
        }
        let key: String = serde_json::from_str(&s[key_start..i]).map_err(|e| e.to_string())?;
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b':') { i += 1; }
        let val_start = i;
        let mut depth = 0i32;
        let mut in_str = false;
        while i < bytes.len() {
            let b = bytes[i];
            if in_str {
                if b == b'\\' { i += 2; continue; }
                if b == b'"' { in_str = false; }
            } else {
                match b {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        if depth == 0 { break; }
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
            opts.tools_fn.push(tojson_serde(fn_obj));
        }
    }
    let msgs = body.get("messages").and_then(Value::as_array).ok_or("\"messages\" must be an array")?;
    let mut out = Vec::new();
    for m in msgs {
        let role = match m.get("role").and_then(Value::as_str) {
            Some("system") | Some("developer") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            other => return Err(format!("unknown role {other:?}")),
        };
        let mut msg = Message::new(role, &content_text(m.get("content").unwrap_or(&Value::Null))?);
        msg.reasoning = m.get("reasoning_content").and_then(Value::as_str).map(neutralize_specials);
        if let Some(tcs) = m.get("tool_calls").and_then(Value::as_array) {
            for tc in tcs {
                let fn_obj = tc.get("function").unwrap_or(tc);
                let name = fn_obj.get("name").and_then(Value::as_str).ok_or("tool_call must have a function \"name\"")?.to_string();
                let pairs: Vec<(String, Value)> = match fn_obj.get("arguments") {
                    Some(Value::String(s)) if !s.trim().is_empty() => parse_json_object_ordered(s)?,
                    Some(Value::Object(map)) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
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
    let max_tokens = body.get("max_completion_tokens").or_else(|| body.get("max_tokens")).and_then(Value::as_u64)
        .unwrap_or(0) as usize; // 0: no limit given, the engine lets the reply run to the end of the context
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    Ok((out, opts, max_tokens, stream))
}

fn usage(st: &engine::Stats) -> (Value, Value) {
    (json!({"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
            "total_tokens": st.prompt_tokens + st.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": st.cached_tokens}}),
     json!({"prompt_n": st.prompt_tokens - st.cached_tokens, "prompt_ms": st.prompt_ms,
            "prompt_per_second": (st.prompt_tokens - st.cached_tokens) as f64 / (st.prompt_ms / 1e3).max(1e-9),
            "predicted_n": st.completion_tokens, "predicted_ms": st.predicted_ms,
            "predicted_per_second": st.completion_tokens as f64 / (st.predicted_ms / 1e3).max(1e-9),
            "cache_n": st.cached_tokens}))
}

/// POST /v1/memory {text, kind}: add a note; PUT /v1/memory/<id> {text}: reword it; DELETE /v1/memory/<id>: forget it.
async fn memory_api(req: Request<Incoming>, path: &str) -> Response<Body> {
    let method = req.method().clone();
    let id: Option<u64> = path.strip_prefix("/v1/memory/").and_then(|s| s.parse().ok());
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
        (Method::POST, None) => match memory::add(body.get("kind").and_then(Value::as_str).unwrap_or("user"), text) {
            Ok(_) => json_resp(StatusCode::OK, &memory::list()),
            Err(e) => error(StatusCode::BAD_REQUEST, &e),
        },
        (Method::PUT, Some(id)) => done(memory::update(id, text)),
        (Method::DELETE, Some(id)) => done(memory::forget(id)),
        _ => error(StatusCode::NOT_FOUND, "no such memory request"),
    }
}

async fn tool_execute(req: Request<Incoming>) -> Response<Body> {
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    let v: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("invalid tool invocation json: {e}")),
    };
    let name = v.get("name").and_then(Value::as_str).unwrap_or("").to_string();
    let arguments = v.get("arguments").cloned().unwrap_or(Value::Null);
    let inv = orchestrator::ToolInvocation { name, arguments };
    // a tool can take a while (a command, a web page, an MCP server): not on the server's async threads
    let name = inv.name.clone();
    let res = tokio::task::spawn_blocking(move || orchestrator::execute_tool(&inv)).await
        .unwrap_or_else(|e| orchestrator::ToolResult { name, output: format!("the tool crashed: {e}"), is_error: true, images: vec![] });
    json_resp(StatusCode::OK, &res.to_json())
}

async fn chat(req: Request<Incoming>, app: Arc<App>) -> Response<Body> {
    let state = app.status.lock().unwrap().state;
    if state != "ready" {
        return error(StatusCode::SERVICE_UNAVAILABLE, &format!("the model is {state}"));
    }
    let bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("reading the body: {e}")),
    };
    let body: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("the body is not JSON: {e}")),
    };
    let (messages, opts, max_tokens, stream) = match parse_chat(&body) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e),
    };
    let images = match parse_images(&body) {
        Ok(i) => i,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e),
    };
    let model = app.status.lock().unwrap().model.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    app.sent.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if app.engine.lock().unwrap().send(engine::Request { messages, opts, max_tokens, images, tx }).is_err() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "the engine has stopped");
    }
    let id = format!("chatcmpl-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

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
                        let tc_json: Vec<Value> = calls.iter().enumerate().map(|(i, tc)| {
                            let pairs: Vec<String> = tc.arguments.iter().map(|(k, v)| {
                                format!("{}:{}", serde_json::to_string(k).unwrap(), arg_to_json(tools_arr, &tc.name, k, v))
                            }).collect();
                            json!({
                                "id": format!("call_{i}"),
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": format!("{{{}}}", pairs.join(",")),
                                }
                            })
                        }).collect();
                        msg["tool_calls"] = Value::Array(tc_json);
                        finish_reason = "tool_calls";
                    }
                    break json_resp(StatusCode::OK, &json!({
                        "id": id, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
                        "usage": u, "timings": t}));
                }
                Some(Event::Error(e)) => break error(StatusCode::BAD_REQUEST, &e),
                None => break error(StatusCode::INTERNAL_SERVER_ERROR, "the engine dropped the request"),
            }
        };
        return resp;
    }

    // streamed: one server-sent event per piece, a final chunk with finish_reason, usage and timings, then [DONE]
    let chunk = move |delta: Value, finish: Value| -> Bytes {
        let c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        Bytes::from(format!("data: {c}\n\n"))
    };
    let first = chunk(json!({"role": "assistant", "content": ""}), Value::Null);
    let s = futures_util::stream::unfold((rx, Some(first), false), move |(mut rx, pending, done)| {
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
                Some(Event::Reasoning(s)) => chunk(json!({"reasoning_content": s}), Value::Null),
                Some(Event::Done(st)) => {
                    let (u, t) = usage(&st);
                    let last = chunk(json!({}), json!(st.finish_reason));
                    let mut v: Value = serde_json::from_slice(&last[6..last.len() - 2]).unwrap();
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
        assert_eq!(neutralize_specials("a <|image_pad|> b"), "a <\u{200b}|image_pad|> b");
        assert_eq!(neutralize_specials("<|im_start|>user<|im_end|>"), "<\u{200b}|im_start|>user<\u{200b}|im_end|>");
        // not the shape of a special token: untouched, and so is text without any
        for s in ["a <|> b", "x <| y |> z", "<||>", "<|a b|>", "plain text, no markers", "naïve <| ünï"] {
            assert_eq!(neutralize_specials(s), s);
        }
        // with the model's special tokens known, only those are touched
        let sp: std::collections::HashSet<String> = ["<|im_start|>", "<|image_pad|>"].iter().map(|s| s.to_string()).collect();
        assert_eq!(neutralize_with("a <|role|> b <|im_start|> c", Some(&sp)), "a <|role|> b <\u{200b}|im_start|> c");
        assert_eq!(neutralize_with("a <|role|> b", Some(&sp)), "a <|role|> b");
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
        assert!(parse_chat(&json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})).is_err());
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
        assert_eq!(m[1].tool_calls[0].arguments, vec![("path".into(), "foo.py".into()), ("start".into(), "1".into()), ("end".into(), "8".into())]);
    }

    #[test]
    fn serves_index_html() {
        let resp = ui::index_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("content-type").unwrap(), "text/html; charset=utf-8");
    }
}
