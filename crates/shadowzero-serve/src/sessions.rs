// Server-side conversations: one JSON file per chat under %LOCALAPPDATA%\Shadow\sessions,
// plus an index the sidebar reads. localStorage held these before; the server holds them
// now so ids are real and a different tab or machine sees the same list.
//
// Blocking file IO: every call is small and fast; the routes wrap them in spawn_blocking
// the same way the screenshot route does.

use serde_json::Value;
use std::path::{Path, PathBuf};

const KEEP_SESSIONS: usize = 200;

fn dir() -> PathBuf {
    std::env::var_os("LOCALAPPDATA").map_or_else(
        || std::env::temp_dir().join("shadowzero-sessions"),
        |a| PathBuf::from(a).join("ShadowZero").join("sessions"),
    )
}

fn index_path() -> PathBuf {
    dir().join("index.json")
}

fn session_path(id: &str) -> Option<PathBuf> {
    // ids come from the client's URL: only our own shape (c + base36) is allowed on disk
    if id.len() <= 24
        && id.starts_with('c')
        && id[1..].chars().all(|c| c.is_ascii_alphanumeric())
    {
        Some(dir().join(format!("{id}.json")))
    } else {
        None
    }
}

fn read_index() -> Vec<Value> {
    std::fs::read(index_path())
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default()
}

fn write_index(rows: &[Value]) -> Result<(), String> {
    let _ = std::fs::create_dir_all(dir());
    std::fs::write(index_path(), serde_json::to_vec_pretty(rows).unwrap_or_default())
        .map_err(|e| e.to_string())
}

/// Sidebar rows, newest first: {id, title, ts, n}. The full history lives in the file.
pub fn list() -> Vec<Value> {
    let mut rows = read_index();
    rows.sort_by(|a, b| {
        b.get("ts")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            .cmp(&a.get("ts").and_then(Value::as_i64).unwrap_or(0))
    });
    rows
}

/// One conversation, exactly the shape the client stored before: {id, title, ts, history, notes}.
pub fn load(id: &str) -> Result<Value, String> {
    let p = session_path(id).ok_or_else(|| "bad session id".to_string())?;
    std::fs::read(p)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .ok_or_else(|| format!("no session {id}"))
}

/// Save (create when the body has no id): writes the file, updates the index row, prunes old.
/// Returns the id, new or old.
pub fn save(body: &Value) -> Result<String, String> {
    let id = match body.get("id").and_then(Value::as_str) {
        Some(s) if session_path(s).is_some() => s.to_string(),
        None | Some(_) => {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64);
            format!("c{t:x}")
        }
    };
    let p = session_path(&id).ok_or_else(|| "bad session id".to_string())?;
    let mut obj = body.as_object().cloned().unwrap_or_default();
    obj.insert("id".into(), Value::String(id.clone()));
    let hist = obj
        .get("history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    obj.insert("ts".into(), serde_json::json!(now_ms()));
    obj.insert("n".into(), serde_json::json!(hist.len()));
    let title = chat_title(&hist);
    obj.insert("title".into(), Value::String(title.clone()));
    let _ = std::fs::create_dir_all(dir());
    std::fs::write(&p, serde_json::to_vec(&obj).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut rows: Vec<Value> = read_index()
        .into_iter()
        .filter(|r| r.get("id").and_then(Value::as_str) != Some(id.as_str()))
        .collect();
    rows.push(serde_json::json!({"id": id, "title": title, "ts": now_ms(), "n": hist.len()}));
    write_index(&rows)?;
    prune();
    Ok(id)
}

pub fn delete(id: &str) -> Result<(), String> {
    let p = session_path(id).ok_or_else(|| "bad session id".to_string())?;
    let _ = std::fs::remove_file(p);
    let rows: Vec<Value> = read_index()
        .into_iter()
        .filter(|r| r.get("id").and_then(Value::as_str) != Some(id))
        .collect();
    write_index(&rows)
}

/// Rename: only the index row and the file's title change.
pub fn rename(id: &str, title: &str) -> Result<(), String> {
    let p = session_path(id).ok_or_else(|| "bad session id".to_string())?;
    if !p.exists() {
        return Err(format!("no session {id}"));
    }
    let mut rows: Vec<Value> = read_index();
    for r in rows.iter_mut() {
        if r.get("id").and_then(Value::as_str) == Some(id) {
            r["title"] = Value::String(title.to_string());
        }
    }
    write_index(&rows)?;
    if let Some(mut obj) = std::fs::read(&p)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v.as_object().cloned())
    {
        obj.insert("title".into(), Value::String(title.to_string()));
        let _ = std::fs::write(&p, serde_json::to_vec(&obj).unwrap_or_default());
    }
    Ok(())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

/// Oldest past KEEP_SESSIONS go, files and index rows both.
fn prune() {
    let mut rows = read_index();
    rows.sort_by_key(|r| r.get("ts").and_then(Value::as_i64).unwrap_or(0));
    while rows.len() > KEEP_SESSIONS {
        let old = rows.remove(0);
        if let Some(id) = old.get("id").and_then(Value::as_str) {
            if let Some(p) = session_path(id) {
                let _ = std::fs::remove_file(p);
            }
        }
    }
    let _ = write_index(&rows);
}

/// Same rule as the client's chatTitle: first user message, code-fence tail, 50 chars.
fn chat_title(hist: &[Value]) -> String {
    let u = hist
        .iter()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"));
    let mut t = match u {
        Some(m) => match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(" "),
            _ => String::new(),
        },
        None => String::new(),
    };
    let i = t.rfind("```");
    if t.starts_with("File: ") {
        if let Some(i) = i {
            t = t[i + 3..].to_string();
        }
    }
    t.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        "New chat".to_string()
    } else if t.chars().count() > 52 {
        t.chars().take(50).collect::<String>() + "..."
    } else {
        t
    }
}

/// One-time move of the old localStorage store into the server (the client sends its
/// whole array once at startup; the client deletes its copy when this says ok).
pub fn import_all(arr: &Value) -> Result<usize, String> {
    let items = arr.as_array().ok_or_else(|| "expected an array".to_string())?;
    let mut n = 0;
    for it in items {
        save(it).map(|_| n += 1)?;
    }
    Ok(n)
}

#[allow(dead_code)]
fn dir_exists() -> bool {
    Path::new(&dir()).exists()
}
