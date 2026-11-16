//! The OpenCode tab (STATUS 153): Shadow starts and stops a local `opencode serve` (sst/opencode, a coding-agent harness with its own web page) whose only provider is
//! this Shadow server, and the UI shows that page in a frame.  Shadow does not ship OpenCode: it looks for an install (`SHADOW_OPENCODE`, then
//! `%LOCALAPPDATA%\Shadow\opencode\node_modules\.bin\opencode.cmd`, then `opencode` on the PATH) and reports how to get one when there is none.
//!
//! The server listens on 127.0.0.1 only, on a free port; its config is rewritten at every start into `%LOCALAPPDATA%\Shadow\opencode\opencode.json` (the provider is
//! `http://127.0.0.1:<shadow port>/v1`, the context is the model's, no auto-update, no sharing) and handed over through `OPENCODE_CONFIG`.  Its working folder is
//! `SHADOW_OPENCODE_DIR`, else the folder Shadow runs in; the page lets the user open other projects.  The process tree lives in the job object the MCP servers use,
//! so it ends with Shadow however Shadow ends.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct Running {
    child: Child,
    port: u16,
}

static RUN: Mutex<Option<Running>> = Mutex::new(None);

fn home() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join("Shadow").join("opencode")
}

/// The OpenCode launcher this machine has, if any.
pub fn find() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SHADOW_OPENCODE").map(PathBuf::from).filter(|p| p.exists()) {
        return Some(p);
    }
    let local = home().join("node_modules").join(".bin").join(if cfg!(windows) { "opencode.cmd" } else { "opencode" });
    if local.exists() {
        return Some(local);
    }
    let finder = if cfg!(windows) { "where" } else { "which" };
    let out = Command::new(finder).arg("opencode").stderr(Stdio::null()).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    // `where` lists the npm shim without an extension first: only a .cmd / .exe starts from here
    text.lines().map(str::trim).map(PathBuf::from).find(|p| p.exists() && (!cfg!(windows) || matches!(p.extension().and_then(|e| e.to_str()), Some("cmd" | "exe"))))
}

/// True while the process we started is alive (a finished one is cleared).
fn running_port() -> Option<u16> {
    let mut g = RUN.lock().unwrap();
    let alive = match g.as_mut() {
        Some(r) => matches!(r.child.try_wait(), Ok(None)),
        None => return None,
    };
    if alive {
        g.as_ref().map(|r| r.port)
    } else {
        *g = None;
        None
    }
}

fn answers(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_millis(300)).is_ok()
}

/// What the UI shows: whether OpenCode is installed, whether it runs, and where.
pub fn status() -> Value {
    let found = find();
    let port = running_port();
    json!({
        "installed": found.is_some(),
        "path": found.map(|p| p.display().to_string()),
        "running": port.is_some(),
        "port": port,
        "install_hint": format!("npm install opencode-ai --prefix \"{}\"  (or set SHADOW_OPENCODE to an opencode launcher)", home().display()),
        "workdir": workdir().display().to_string(),
    })
}

fn workdir() -> PathBuf {
    std::env::var_os("SHADOW_OPENCODE_DIR").map(PathBuf::from).filter(|p| p.is_dir()).or_else(|| std::env::current_dir().ok()).unwrap_or_else(|| PathBuf::from("."))
}

/// The config OpenCode reads: Shadow as the one provider.
fn config(shadow_port: u16, model: &str, n_ctx: usize) -> Value {
    json!({
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": false,
        "share": "disabled",
        "provider": {
            "shadow": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Shadow (local)",
                "options": { "baseURL": format!("http://127.0.0.1:{shadow_port}/v1"), "apiKey": "none" },
                "models": { "local": { "name": model, "tool_call": true, "limit": { "context": n_ctx.max(4096), "output": 4096 } } }
            }
        },
        "model": "shadow/local"
    })
}

/// Starts OpenCode (nothing happens when it already runs); returns the status once its port answers.
pub fn start(shadow_port: u16, model: &str, n_ctx: usize) -> Result<Value, String> {
    if running_port().is_some() {
        return Ok(status());
    }
    let exe = find().ok_or_else(|| format!("OpenCode is not installed.  {}", status()["install_hint"].as_str().unwrap_or("")))?;
    let dir = home();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let cfg = dir.join("opencode.json");
    std::fs::write(&cfg, serde_json::to_vec_pretty(&config(shadow_port, model, n_ctx)).unwrap()).map_err(|e| format!("{}: {e}", cfg.display()))?;
    let port = {
        let l = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
        l.local_addr().map_err(|e| e.to_string())?.port()
    };
    let mut cmd = Command::new(&exe);
    cmd.args(["serve", "--port", &port.to_string(), "--hostname", "127.0.0.1"])
        .env("OPENCODE_CONFIG", &cfg)
        .current_dir(workdir())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let child = cmd.spawn().map_err(|e| format!("starting {}: {e}", exe.display()))?;
    crate::mcp::tie_to_shadow(&child);
    *RUN.lock().unwrap() = Some(Running { child, port });
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(30) {
        if running_port().is_none() {
            return Err("OpenCode ended right after it started".into());
        }
        if answers(port) {
            return Ok(status());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    stop();
    Err("OpenCode did not answer within 30 s".into())
}

/// Ends OpenCode and everything it started.
pub fn stop() {
    if let Some(mut r) = RUN.lock().unwrap().take() {
        #[cfg(windows)]
        {
            // the launcher is a .cmd that starts node that starts the agent: the whole tree
            let _ = Command::new("taskkill").args(["/T", "/F", "/PID", &r.child.id().to_string()]).stdout(Stdio::null()).stderr(Stdio::null()).status();
        }
        let _ = r.child.kill();
        let _ = r.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_points_opencode_at_shadow() {
        let c = config(8081, "Qwen", 24576);
        assert_eq!(c["provider"]["shadow"]["options"]["baseURL"], "http://127.0.0.1:8081/v1");
        assert_eq!(c["provider"]["shadow"]["models"]["local"]["limit"]["context"], 24576);
        assert_eq!(c["model"], "shadow/local");
        assert_eq!(c["share"], "disabled");
    }

    #[test]
    fn status_says_how_to_install_when_there_is_nothing() {
        let s = status();
        assert!(s["install_hint"].as_str().unwrap().contains("npm install opencode-ai"));
        assert_eq!(s["running"], false);
    }
}
