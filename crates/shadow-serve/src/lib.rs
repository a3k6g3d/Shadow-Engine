//! Shadow's HTTP server (docs/SHADOW_PLAN.md, stage 4): an OpenAI-style `/v1/chat/completions` (whole or streamed as
//! server-sent events), `/health`, `/v1/models`, `/v1/status`, `/props` and `/metrics`, over hyper, in front of one
//! engine thread (engine.rs).  Requests queue; one is answered at a time.
//!
//! Not yet: tool definitions and tool calls (the chat template refuses them, so a request with `tools` is answered
//! 400 rather than rendered wrongly), sampling (greedy only), images.

pub mod engine;

use bytes::Bytes;
use engine::{Event, Status};
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{json, Value};
use shadow_model::chat::{Effort, Message, Options, Role};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

type Body = BoxBody<Bytes, Infallible>;

struct App {
    engine: Mutex<std::sync::mpsc::Sender<engine::Request>>,
    status: Arc<Mutex<Status>>,
    started: std::time::Instant,
    /// Requests handed to the engine (the engine counts the ones it finished: the difference is the queue).
    sent: std::sync::atomic::AtomicU64,
}

pub struct ServeConfig {
    pub engine: engine::Config,
    pub addr: std::net::SocketAddr,
}

/// Runs the server until the process ends.
pub fn run(cfg: ServeConfig) -> Result<(), String> {
    let status = Arc::new(Mutex::new(Status { state: "loading", ..Status::default() }));
    let tx = engine::start(cfg.engine, status.clone());
    let app = Arc::new(App { engine: Mutex::new(tx), status, started: std::time::Instant::now(),
                             sent: std::sync::atomic::AtomicU64::new(0) });
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
                "tools": false, "streaming": true, "default_generation_settings": {"n_ctx": s.n_ctx},
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

/// A message's content: a string, or an array of parts of which the text ones are joined (anything else refused).
fn content_text(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => parts.iter().map(|p| match p.get("type").and_then(Value::as_str) {
            Some("text") => Ok(p.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
            other => Err(format!("content part type {other:?} is not supported (text only)")),
        }).collect::<Result<Vec<_>, _>>().map(|v| v.concat()),
        _ => Err("message content must be a string or an array of text parts".into()),
    }
}

/// The request body as the engine's request (without its channel).
pub fn parse_chat(body: &Value) -> Result<(Vec<Message>, Options, usize, bool), String> {
    if body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty()) {
        return Err("tools are not supported yet: Shadow's chat template does not render tool definitions".into());
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
        if m.get("tool_calls").and_then(Value::as_array).is_some_and(|t| !t.is_empty()) {
            return Err("assistant tool_calls are not supported yet".into());
        }
        let mut msg = Message::new(role, &content_text(m.get("content").unwrap_or(&Value::Null))?);
        msg.reasoning = m.get("reasoning_content").and_then(Value::as_str).map(str::to_string);
        out.push(msg);
    }
    let mut opts = Options::default();
    if let Some(k) = body.get("chat_template_kwargs") {
        if let Some(t) = k.get("enable_thinking").and_then(Value::as_bool) {
            opts.thinking = Some(t);
        }
    }
    opts.effort = match body.get("reasoning_effort").and_then(Value::as_str) {
        Some("low") => Effort::Low,
        Some("medium") => Effort::Medium,
        _ => Effort::XHigh,
    };
    let max_tokens = body.get("max_completion_tokens").or_else(|| body.get("max_tokens")).and_then(Value::as_u64)
        .unwrap_or(1024) as usize;
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
    let model = app.status.lock().unwrap().model.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    app.sent.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if app.engine.lock().unwrap().send(engine::Request { messages, opts, max_tokens, tx }).is_err() {
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
                    let mut msg = json!({"role": "assistant", "content": content});
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = json!(reasoning);
                    }
                    break json_resp(StatusCode::OK, &json!({
                        "id": id, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": msg, "finish_reason": st.finish_reason}],
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
    fn refuses_what_it_cannot_render() {
        assert!(parse_chat(&json!({"messages": [], "tools": [{"type": "function"}]})).is_err());
        assert!(parse_chat(&json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})).is_err());
        assert!(parse_chat(&json!({"messages": [{"role": "wizard", "content": "x"}]})).is_err());
        assert!(parse_chat(&json!({"nope": 1})).is_err());
    }
}
