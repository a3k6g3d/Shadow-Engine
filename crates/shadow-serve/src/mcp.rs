//! MCP (Model Context Protocol) client, stdio transport: the tools of external MCP servers become tools of the model next to Shadow's own.
//!
//! An MCP server is a program that speaks JSON-RPC 2.0 over its stdin and stdout, one JSON message a line: `initialize`, then
//! `tools/list` (each tool a name, a description and a JSON schema, the shape of Shadow's own tool definitions) and `tools/call`.
//! Servers are listed in `mcp.json` in the format Claude Desktop and Claude Code use, so an existing list can be copied:
//!
//! ```json
//! { "mcpServers": { "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "J:/work"], "env": {} } } }
//! ```
//!
//! A server entry may carry `"disabled": true` and `"allow": ["tool", ...]` (only those tools are offered).
//!
//! Read from `SHADOW_MCP_CONFIG`, else the first `mcp.json` in: the working directory, the exe's folder, the checkout Shadow was built from, `%APPDATA%\Shadow`.  A tool is offered to the
//! model as `mcp__<server>__<tool>`.  The servers start in the background when `shadow serve` does; one that fails is listed with its error.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex, Once};
use std::time::{Duration, Instant};

pub struct McpTool {
    /// What the model calls it: `mcp__server__tool`.
    pub name: String,
    /// The server's own name for it.
    pub raw: String,
    pub description: String,
    pub schema: Value,
}

struct Io {
    stdin: ChildStdin,
    rx: mpsc::Receiver<Value>,
    next: u64,
}

/// What a tool call returned: its text, and the pictures it carried (as `data:` URLs).
pub struct McpOut {
    pub text: String,
    pub images: Vec<String>,
}

pub struct Server {
    pub name: String,
    pub tools: Vec<McpTool>,
    /// Tools the server has that `"allow"` keeps from the model (Shadow itself may still call them: the browser pane's screenshots).
    pub hidden: Vec<String>,
    pub error: Option<String>,
    io: Option<Mutex<Io>>,
    child: Mutex<Option<Child>>,
}

static SERVERS: Mutex<Vec<Arc<Server>>> = Mutex::new(Vec::new());
static START: Once = Once::new();

/// A name safe inside a tool name: letters, digits, `_` and `-`.
fn clean(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

fn config_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SHADOW_MCP_CONFIG") {
        return Some(PathBuf::from(p));
    }
    // wherever Shadow was started from (the desktop app does not start in the checkout): the working directory, beside the exe, the checkout it was
    // built from, then the user's folder
    let mut places = vec![PathBuf::from("mcp.json")];
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
        places.push(d.join("mcp.json"));
    }
    places.push(PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../mcp.json")));
    if let Ok(a) = std::env::var("APPDATA") {
        places.push(PathBuf::from(a).join("Shadow").join("mcp.json"));
    }
    places.into_iter().find(|p| p.is_file())
}

/// The (name, config) pairs of a config file's `mcpServers` (a server with `"disabled": true` is skipped).
pub fn parse_config(text: &str) -> Result<Vec<(String, Value)>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("mcp.json is not JSON: {e}"))?;
    let servers = v.get("mcpServers").or_else(|| v.get("servers")).and_then(Value::as_object).ok_or("mcp.json has no \"mcpServers\" object")?;
    Ok(servers.iter().filter(|(_, c)| !c.get("disabled").and_then(Value::as_bool).unwrap_or(false)).map(|(n, c)| (n.clone(), c.clone())).collect())
}

/// A server's processes (npx starts node starts a browser) must not outlive Shadow: they go into a Windows job object that kills them when
/// Shadow's last handle to it closes, which is when Shadow ends, however it ends.
#[cfg(windows)]
fn tie_to_shadow(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    #[repr(C)]
    struct Limits {
        basic: [u8; 64],
        io: [u8; 48],
        rest: [usize; 4],
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attrs: *const u8, name: *const u16) -> isize;
        fn SetInformationJobObject(job: isize, class: i32, info: *const Limits, len: u32) -> i32;
        fn AssignProcessToJobObject(job: isize, process: isize) -> i32;
    }
    static JOB: std::sync::OnceLock<isize> = std::sync::OnceLock::new();
    let job = *JOB.get_or_init(|| {
        // SAFETY: a job object whose only limit is JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (flags at offset 16 of the basic limits); never closed here.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            let mut l = Limits { basic: [0; 64], io: [0; 48], rest: [0; 4] };
            l.basic[16..20].copy_from_slice(&0x2000u32.to_le_bytes());
            if job != 0 {
                SetInformationJobObject(job, 9, &l, std::mem::size_of::<Limits>() as u32);
            }
            job
        }
    });
    if job != 0 {
        // SAFETY: the child's process handle is valid while `child` lives.
        unsafe { AssignProcessToJobObject(job, child.as_raw_handle() as isize) };
    }
}

#[cfg(not(windows))]
fn tie_to_shadow(_child: &Child) {}

fn spawn(cfg: &Value) -> Result<Child, String> {
    let command = cfg.get("command").and_then(Value::as_str).ok_or("no \"command\"")?;
    let args: Vec<String> = cfg.get("args").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
    let build = |program: &str, argv: &[String]| {
        let mut c = Command::new(program);
        c.args(argv).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        if let Some(env) = cfg.get("env").and_then(Value::as_object) {
            for (k, v) in env {
                if let Some(v) = v.as_str() {
                    c.env(k, v);
                }
            }
        }
        if let Some(d) = cfg.get("cwd").and_then(Value::as_str) {
            c.current_dir(d);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        c
    };
    let started = match build(command, &args).spawn() {
        Ok(c) => Ok(c),
        // `npx` and friends are .cmd files on Windows: CreateProcess does not find them, cmd does
        Err(e) if cfg!(windows) && e.kind() == std::io::ErrorKind::NotFound => {
            let mut argv = vec!["/C".to_string(), command.to_string()];
            argv.extend(args);
            build("cmd.exe", &argv).spawn().map_err(|e| format!("starting {command}: {e}"))
        }
        Err(e) => Err(format!("starting {command}: {e}")),
    };
    if let Ok(c) = &started {
        tie_to_shadow(c);
    }
    started
}

/// One request and its answer (messages that are not the answer, notifications and the server's own requests, are skipped).
fn rpc(io: &mut Io, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
    let id = io.next;
    io.next += 1;
    let line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
    writeln!(io.stdin, "{line}").and_then(|_| io.stdin.flush()).map_err(|e| format!("the server is gone: {e}"))?;
    let end = Instant::now() + timeout;
    loop {
        let left = end.saturating_duration_since(Instant::now());
        let msg = io.rx.recv_timeout(left).map_err(|e| match e {
            mpsc::RecvTimeoutError::Timeout => format!("{method}: no answer in {} s", timeout.as_secs()),
            mpsc::RecvTimeoutError::Disconnected => "the server closed its output".to_string(),
        })?;
        if msg.get("id") == Some(&json!(id)) && msg.get("method").is_none() {
            return match msg.get("error") {
                Some(e) => Err(e.get("message").and_then(Value::as_str).unwrap_or("error").to_string()),
                None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
            };
        }
    }
}

fn start_server(name: &str, cfg: &Value) -> Server {
    let fail = |e: String| Server { name: name.to_string(), tools: vec![], hidden: vec![], error: Some(e), io: None, child: Mutex::new(None) };
    let mut child = match spawn(cfg) {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else { return fail("no pipes".into()) };
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                if tx.send(v).is_err() {
                    break;
                }
            }
        }
    });
    let mut io = Io { stdin, rx, next: 1 };
    let init = rpc(&mut io, "initialize", json!({"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "shadow", "version": env!("CARGO_PKG_VERSION")}}),
                   Duration::from_secs(60));
    if let Err(e) = init {
        let _ = child.kill();
        return fail(format!("initialize: {e}"));
    }
    let _ = writeln!(io.stdin, "{}", json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).and_then(|_| io.stdin.flush());
    let mut tools = vec![];
    let mut cursor: Option<String> = None;
    loop {
        let params = cursor.as_ref().map_or(json!({}), |c| json!({"cursor": c}));
        let r = match rpc(&mut io, "tools/list", params, Duration::from_secs(30)) {
            Ok(r) => r,
            Err(e) => {
                let _ = child.kill();
                return fail(format!("tools/list: {e}"));
            }
        };
        for t in r.get("tools").and_then(Value::as_array).into_iter().flatten() {
            let Some(raw) = t.get("name").and_then(Value::as_str) else { continue };
            tools.push(McpTool {
                name: format!("mcp__{}__{}", clean(name), clean(raw)),
                raw: raw.to_string(),
                description: t.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
                schema: t.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            });
        }
        cursor = r.get("nextCursor").and_then(Value::as_str).map(String::from);
        if cursor.is_none() {
            break;
        }
    }
    // `"allow": [...]` offers only those of the server's tools (a long tool list costs prompt tokens in every chat)
    let mut hidden = vec![];
    if let Some(allow) = cfg.get("allow").and_then(Value::as_array) {
        let allow: Vec<&str> = allow.iter().filter_map(Value::as_str).collect();
        hidden = tools.iter().filter(|t| !allow.contains(&t.raw.as_str())).map(|t| t.raw.clone()).collect();
        tools.retain(|t| allow.contains(&t.raw.as_str()));
    }
    // `"hints": {"tool": "text"}` is added to that tool's description (what the model should know that the server does not say)
    if let Some(hints) = cfg.get("hints").and_then(Value::as_object) {
        for t in tools.iter_mut() {
            if let Some(h) = hints.get(&t.raw).and_then(Value::as_str) {
                t.description = format!("{} {h}", t.description);
            }
        }
    }
    Server { name: name.to_string(), tools, hidden, error: None, io: Some(Mutex::new(io)), child: Mutex::new(Some(child)) }
}

/// Starts the configured servers (once), each on its own thread; `servers()` fills in as they come up.
pub fn start_background() {
    START.call_once(|| {
        let Some(path) = config_path() else { return };
        let list = match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| parse_config(&t)) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("shadow serve: {}: {e}", path.display());
                return;
            }
        };
        eprintln!("shadow serve: starting {} MCP server(s) from {}", list.len(), path.display());
        for (name, cfg) in list {
            std::thread::spawn(move || {
                let s = start_server(&name, &cfg);
                match &s.error {
                    None => eprintln!("shadow serve: MCP server {name}: {} tools", s.tools.len()),
                    Some(e) => eprintln!("shadow serve: MCP server {name} failed: {e}"),
                }
                SERVERS.lock().unwrap().push(Arc::new(s));
            });
        }
    });
}

pub fn servers() -> Vec<Arc<Server>> {
    SERVERS.lock().unwrap().clone()
}

/// The MCP tools as Shadow's tool definitions (`name`, `description`, `input_schema`).
pub fn tool_definitions() -> Vec<Value> {
    servers().iter().flat_map(|s| s.tools.iter().map(move |t| json!({
        "name": t.name,
        "description": format!("[MCP server {}] {}", s.name, t.description),
        "input_schema": t.schema,
    }))).collect()
}

/// Calls an `mcp__server__tool` with its arguments: its answer (text and pictures), or an error message.
pub fn call(name: &str, arguments: &Value) -> Result<McpOut, String> {
    let server = servers().into_iter().find(|s| s.tools.iter().any(|t| t.name == name)).ok_or_else(|| format!("unknown MCP tool '{name}'"))?;
    let raw = server.tools.iter().find(|t| t.name == name).unwrap().raw.clone();
    call_raw(&server, &raw, arguments)
}

fn call_raw(server: &Server, raw: &str, arguments: &Value) -> Result<McpOut, String> {
    let io = server.io.as_ref().ok_or("the server is not running")?;
    let args = if arguments.is_object() { arguments.clone() } else { json!({}) };
    let r = rpc(&mut io.lock().unwrap(), "tools/call", json!({"name": raw, "arguments": args}), Duration::from_secs(180))?;
    let (mut text, mut images) = (String::new(), vec![]);
    for part in r.get("content").and_then(Value::as_array).into_iter().flatten() {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(part.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("image") => {
                let (mime, data) = (part.get("mimeType").and_then(Value::as_str).unwrap_or("image/png"), part.get("data").and_then(Value::as_str).unwrap_or(""));
                images.push(format!("data:{mime};base64,{data}"));
                text.push_str("[picture shown to the user]");
            }
            Some(other) => text.push_str(&format!("[{other} content not shown]")),
            None => {}
        }
        text.push('\n');
    }
    let text = text.trim().to_string();
    if r.get("isError").and_then(Value::as_bool).unwrap_or(false) {
        Err(if text.is_empty() { "the tool reported an error".into() } else { text })
    } else {
        Ok(McpOut { text, images })
    }
}

/// A picture of the browser's current page (for the page's Browser pane), by the browser server's `browser_take_screenshot`, offered to the model or not.
pub fn screenshot() -> Result<McpOut, String> {
    let server = servers().into_iter().find(|s| s.tools.iter().any(|t| t.raw == "browser_take_screenshot") || s.hidden.iter().any(|t| t == "browser_take_screenshot"))
        .ok_or("no browser server (see mcp.json)")?;
    call_raw(&server, "browser_take_screenshot", &json!({"type": "jpeg"}))
}

/// For the dashboard: (server, tools, error).
pub fn status() -> Vec<(String, usize, Option<String>)> {
    servers().iter().map(|s| (s.name.clone(), s.tools.len(), s.error.clone())).collect()
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.lock().unwrap().take() {
            let _ = c.kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_is_read_in_claudes_format() {
        let l = parse_config(r#"{"mcpServers": {"a": {"command": "npx", "args": ["-y", "x"]}, "b": {"command": "y", "disabled": true}}}"#).unwrap();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].0, "a");
        assert!(parse_config("{}").is_err());
        assert!(parse_config("nope").is_err());
    }

    #[test]
    fn names_are_made_safe() {
        assert_eq!(clean("my server.v2"), "my_server_v2");
    }
}
