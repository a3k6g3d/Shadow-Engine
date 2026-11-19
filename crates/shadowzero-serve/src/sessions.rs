// Server-side conversations: one JSON file per chat under %LOCALAPPDATA%\Shadow\sessions,
// plus an index the sidebar reads. localStorage held these before; the server holds them
// now so ids are real and a different tab or machine sees the same list.
//
// Blocking file IO: every call is small and fast; the routes wrap them in spawn_blocking
// the same way the screenshot route does.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const KEEP_SESSIONS: usize = 200;
static ID_SEQ: AtomicU64 = AtomicU64::new(1);

fn dir() -> PathBuf {
    if let Ok(d) = std::env::var("SHADOWZERO_SESSIONS_DIR") {
        if !d.trim().is_empty() {
            return PathBuf::from(d);
        }
    }
    std::env::var_os("LOCALAPPDATA").map_or_else(
        || std::env::temp_dir().join("shadowzero-sessions"),
        |a| PathBuf::from(a).join("ShadowZero").join("sessions"),
    )
}

fn index_path_in(base: &Path) -> PathBuf {
    base.join("index.json")
}

fn projects_path_in(base: &Path) -> PathBuf {
    base.join("projects.json")
}

fn session_path_in(base: &Path, id: &str) -> Option<PathBuf> {
    // ids come from the client's URL: only our own shape (c + base36/hex) is allowed on disk
    if id.len() <= 28
        && id.starts_with('c')
        && id[1..].chars().all(|c| c.is_ascii_alphanumeric())
    {
        Some(base.join(format!("{id}.json")))
    } else {
        None
    }
}

fn project_id_ok(id: &str) -> bool {
    id.len() <= 28
        && id.starts_with('p')
        && id[1..].chars().all(|c| c.is_ascii_alphanumeric())
}

fn read_json_array(path: &Path) -> Vec<Value> {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default()
}

fn write_json_array(base: &Path, path: &Path, rows: &[Value]) -> Result<(), String> {
    let _ = std::fs::create_dir_all(base);
    std::fs::write(path, serde_json::to_vec_pretty(rows).unwrap_or_default())
        .map_err(|e| e.to_string())
}

fn read_index_in(base: &Path) -> Vec<Value> {
    read_json_array(&index_path_in(base))
}

fn write_index_in(base: &Path, rows: &[Value]) -> Result<(), String> {
    write_json_array(base, &index_path_in(base), rows)
}

fn read_projects_in(base: &Path) -> Vec<Value> {
    read_json_array(&projects_path_in(base))
}

fn write_projects_in(base: &Path, rows: &[Value]) -> Result<(), String> {
    write_json_array(base, &projects_path_in(base), rows)
}

/// Sidebar rows, newest first: {id, title, ts, n, project_id?, archived?}. The full history lives in the file.
pub fn list() -> Vec<Value> {
    list_in(&dir())
}

fn list_in(base: &Path) -> Vec<Value> {
    let mut rows = read_index_in(base);
    rows.sort_by(|a, b| {
        b.get("ts")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            .cmp(&a.get("ts").and_then(Value::as_i64).unwrap_or(0))
    });
    rows
}

/// One conversation, including optional project_id & archived flags.
pub fn load(id: &str) -> Result<Value, String> {
    load_in(&dir(), id)
}

fn load_in(base: &Path, id: &str) -> Result<Value, String> {
    let p = session_path_in(base, id).ok_or_else(|| "bad session id".to_string())?;
    std::fs::read(p)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .ok_or_else(|| format!("no session {id}"))
}

fn touch_project_ts_in(base: &Path, project_id: &str, ts: i64) {
    let mut projs = read_projects_in(base);
    let mut changed = false;
    for pr in projs.iter_mut() {
        if pr.get("id").and_then(Value::as_str) == Some(project_id) {
            pr["updated_ts"] = json!(ts);
            changed = true;
            break;
        }
    }
    if changed {
        let _ = write_projects_in(base, &projs);
    }
}

/// Save (create when the body has no id): writes the file, updates the index row, prunes old.
/// Preserves existing `project_id` and `archived` flags unless explicitly overridden in `body`.
/// Returns the id, new or old.
pub fn save(body: &Value) -> Result<String, String> {
    save_in(&dir(), body)
}

fn save_in(base: &Path, body: &Value) -> Result<String, String> {
    let id = match body.get("id").and_then(Value::as_str) {
        Some(s) if session_path_in(base, s).is_some() => s.to_string(),
        None | Some(_) => {
            let t = now_ms() as u64;
            let seq = ID_SEQ.fetch_add(1, Ordering::Relaxed) % 4096;
            format!("c{t:x}{seq:x}")
        }
    };
    let p = session_path_in(base, &id).ok_or_else(|| "bad session id".to_string())?;
    let existing_rows = read_index_in(base);
    let prev = existing_rows
        .iter()
        .find(|r| r.get("id").and_then(Value::as_str) == Some(id.as_str()));

    let project_id: Option<String> = match body.get("project_id") {
        Some(Value::String(s)) if project_id_ok(s) => Some(s.clone()),
        Some(Value::Null) | Some(Value::String(_)) => None,
        _ => prev
            .and_then(|r| r.get("project_id"))
            .and_then(Value::as_str)
            .filter(|s| project_id_ok(s))
            .map(String::from),
    };
    let archived: bool = match body.get("archived").and_then(Value::as_bool) {
        Some(b) => b,
        None => prev
            .and_then(|r| r.get("archived"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };

    let ts = now_ms();
    let mut obj = body.as_object().cloned().unwrap_or_default();
    obj.insert("id".into(), Value::String(id.clone()));
    let hist = obj
        .get("history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    obj.insert("ts".into(), json!(ts));
    obj.insert("n".into(), json!(hist.len()));
    let title = chat_title(&hist);
    obj.insert("title".into(), Value::String(title.clone()));
    if let Some(ref pid) = project_id {
        obj.insert("project_id".into(), Value::String(pid.clone()));
    } else {
        obj.remove("project_id");
    }
    if archived {
        obj.insert("archived".into(), Value::Bool(true));
    } else {
        obj.remove("archived");
    }

    let _ = std::fs::create_dir_all(base);
    std::fs::write(&p, serde_json::to_vec(&obj).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let mut row = serde_json::Map::new();
    row.insert("id".into(), Value::String(id.clone()));
    row.insert("title".into(), Value::String(title));
    row.insert("ts".into(), json!(ts));
    row.insert("n".into(), json!(hist.len()));
    if let Some(ref pid) = project_id {
        row.insert("project_id".into(), Value::String(pid.clone()));
        touch_project_ts_in(base, pid, ts);
    }
    if archived {
        row.insert("archived".into(), Value::Bool(true));
    }

    let mut rows: Vec<Value> = existing_rows
        .into_iter()
        .filter(|r| r.get("id").and_then(Value::as_str) != Some(id.as_str()))
        .collect();
    rows.push(Value::Object(row));
    write_index_in(base, &rows)?;
    prune_in(base);
    Ok(id)
}

pub fn delete(id: &str) -> Result<(), String> {
    delete_in(&dir(), id)
}

fn delete_in(base: &Path, id: &str) -> Result<(), String> {
    let p = session_path_in(base, id).ok_or_else(|| "bad session id".to_string())?;
    let _ = std::fs::remove_file(p);
    let rows: Vec<Value> = read_index_in(base)
        .into_iter()
        .filter(|r| r.get("id").and_then(Value::as_str) != Some(id))
        .collect();
    write_index_in(base, &rows)
}

/// Update session metadata (`title`, `project_id`, `archived`) without changing its timestamp.
pub fn update(id: &str, patch: &Value) -> Result<(), String> {
    update_in(&dir(), id, patch)
}

fn update_in(base: &Path, id: &str, patch: &Value) -> Result<(), String> {
    let p = session_path_in(base, id).ok_or_else(|| "bad session id".to_string())?;
    if !p.exists() {
        return Err(format!("no session {id}"));
    }
    let Some(map) = patch.as_object() else {
        return Err("expected json object".into());
    };
    let title_opt = map
        .get("title")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    let project_patch: Option<Option<String>> = map.get("project_id").map(|v| match v {
        Value::String(s) if project_id_ok(s) => Some(s.clone()),
        _ => None,
    });
    let archived_opt = map.get("archived").and_then(Value::as_bool);

    let mut rows = read_index_in(base);
    for r in rows.iter_mut() {
        if r.get("id").and_then(Value::as_str) == Some(id) {
            if let Some(obj) = r.as_object_mut() {
                if let Some(ref t) = title_opt {
                    obj.insert("title".into(), Value::String(t.clone()));
                }
                if let Some(ref pp) = project_patch {
                    match pp {
                        Some(pid) => {
                            obj.insert("project_id".into(), Value::String(pid.clone()));
                        }
                        None => {
                            obj.remove("project_id");
                        }
                    }
                }
                if let Some(a) = archived_opt {
                    if a {
                        obj.insert("archived".into(), Value::Bool(true));
                    } else {
                        obj.remove("archived");
                    }
                }
            }
        }
    }
    write_index_in(base, &rows)?;
    if let Some(mut obj) = std::fs::read(&p)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v.as_object().cloned())
    {
        if let Some(ref t) = title_opt {
            obj.insert("title".into(), Value::String(t.clone()));
        }
        if let Some(ref pp) = project_patch {
            match pp {
                Some(pid) => {
                    obj.insert("project_id".into(), Value::String(pid.clone()));
                }
                None => {
                    obj.remove("project_id");
                }
            }
        }
        if let Some(a) = archived_opt {
            if a {
                obj.insert("archived".into(), Value::Bool(true));
            } else {
                obj.remove("archived");
            }
        }
        let _ = std::fs::write(&p, serde_json::to_vec(&obj).unwrap_or_default());
    }
    if let Some(Some(ref pid)) = project_patch {
        touch_project_ts_in(base, pid, now_ms());
    }
    Ok(())
}

/// Rename: only the index row and the file's title change.
pub fn rename(id: &str, title: &str) -> Result<(), String> {
    update(id, &json!({ "title": title }))
}

/// Projects / jobs stored alongside sessions (`projects.json`), newest activity first.
pub fn list_projects() -> Vec<Value> {
    list_projects_in(&dir())
}

fn list_projects_in(base: &Path) -> Vec<Value> {
    let mut rows = read_projects_in(base);
    rows.sort_by(|a, b| {
        let bt = b
            .get("updated_ts")
            .or_else(|| b.get("created_ts"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let at = a
            .get("updated_ts")
            .or_else(|| a.get("created_ts"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        bt.cmp(&at)
    });
    rows
}

pub fn create_project(body: &Value) -> Result<Value, String> {
    create_project_in(&dir(), body)
}

fn create_project_in(base: &Path, body: &Value) -> Result<Value, String> {
    let raw_name = body
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if raw_name.is_empty() {
        return Err("project name is required".into());
    }
    let name: String = raw_name.chars().take(80).collect();
    let description: String = body
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .chars()
        .take(300)
        .collect();
    let ts = now_ms();
    let seq = ID_SEQ.fetch_add(1, Ordering::Relaxed) % 4096;
    let id = format!("p{:x}{:x}", ts as u64, seq);
    let proj = json!({
        "id": id,
        "name": name,
        "description": description,
        "created_ts": ts,
        "updated_ts": ts,
        "archived": false
    });
    let mut rows = read_projects_in(base);
    rows.push(proj.clone());
    write_projects_in(base, &rows)?;
    Ok(proj)
}

pub fn update_project(id: &str, body: &Value) -> Result<Value, String> {
    update_project_in(&dir(), id, body)
}

fn update_project_in(base: &Path, id: &str, body: &Value) -> Result<Value, String> {
    if !project_id_ok(id) {
        return Err("bad project id".into());
    }
    let mut rows = read_projects_in(base);
    let Some(pr) = rows
        .iter_mut()
        .find(|r| r.get("id").and_then(Value::as_str) == Some(id))
    else {
        return Err(format!("no project {id}"));
    };
    if let Some(name) = body.get("name").and_then(Value::as_str) {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("project name cannot be empty".into());
        }
        pr["name"] = Value::String(trimmed.chars().take(80).collect());
    }
    if let Some(desc) = body.get("description").and_then(Value::as_str) {
        pr["description"] = Value::String(desc.trim().chars().take(300).collect());
    }
    if let Some(arch) = body.get("archived").and_then(Value::as_bool) {
        pr["archived"] = Value::Bool(arch);
    }
    pr["updated_ts"] = json!(now_ms());
    let updated = pr.clone();
    write_projects_in(base, &rows)?;
    Ok(updated)
}

pub fn delete_project(id: &str) -> Result<(), String> {
    delete_project_in(&dir(), id)
}

fn delete_project_in(base: &Path, id: &str) -> Result<(), String> {
    if !project_id_ok(id) {
        return Err("bad project id".into());
    }
    let projs: Vec<Value> = read_projects_in(base)
        .into_iter()
        .filter(|r| r.get("id").and_then(Value::as_str) != Some(id))
        .collect();
    write_projects_in(base, &projs)?;
    // Unassign conversations from the deleted project rather than deleting their history.
    let mut rows = read_index_in(base);
    let mut affected = Vec::new();
    for r in rows.iter_mut() {
        if r.get("project_id").and_then(Value::as_str) == Some(id) {
            if let Some(cid) = r.get("id").and_then(Value::as_str) {
                affected.push(cid.to_string());
            }
            if let Some(obj) = r.as_object_mut() {
                obj.remove("project_id");
            }
        }
    }
    if !affected.is_empty() {
        write_index_in(base, &rows)?;
        for cid in affected {
            if let Some(p) = session_path_in(base, &cid) {
                if let Some(mut obj) = std::fs::read(&p)
                    .ok()
                    .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                    .and_then(|v| v.as_object().cloned())
                {
                    obj.remove("project_id");
                    let _ = std::fs::write(&p, serde_json::to_vec(&obj).unwrap_or_default());
                }
            }
        }
    }
    Ok(())
}

/// Archives all non-archived conversations in `project_id` older than the `keep_newest` most recent ones.
pub fn archive_older_in_project(project_id: &str, keep_newest: usize) -> Result<usize, String> {
    archive_older_in_project_in(&dir(), project_id, keep_newest)
}

fn archive_older_in_project_in(
    base: &Path,
    project_id: &str,
    keep_newest: usize,
) -> Result<usize, String> {
    if !project_id_ok(project_id) {
        return Err("bad project id".into());
    }
    let mut active: Vec<(String, i64)> = read_index_in(base)
        .into_iter()
        .filter(|r| {
            r.get("project_id").and_then(Value::as_str) == Some(project_id)
                && !r.get("archived").and_then(Value::as_bool).unwrap_or(false)
        })
        .filter_map(|r| {
            Some((
                r.get("id").and_then(Value::as_str)?.to_string(),
                r.get("ts").and_then(Value::as_i64).unwrap_or(0),
            ))
        })
        .collect();
    // Newest to oldest
    active.sort_by(|a, b| b.1.cmp(&a.1));
    if active.len() <= keep_newest {
        return Ok(0);
    }
    let to_archive: Vec<String> = active.into_iter().skip(keep_newest).map(|(id, _)| id).collect();
    let mut rows = read_index_in(base);
    for r in rows.iter_mut() {
        if let Some(cid) = r.get("id").and_then(Value::as_str) {
            if to_archive.iter().any(|x| x == cid) {
                if let Some(obj) = r.as_object_mut() {
                    obj.insert("archived".into(), Value::Bool(true));
                }
            }
        }
    }
    write_index_in(base, &rows)?;
    for cid in &to_archive {
        if let Some(p) = session_path_in(base, cid) {
            if let Some(mut obj) = std::fs::read(&p)
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .and_then(|v| v.as_object().cloned())
            {
                obj.insert("archived".into(), Value::Bool(true));
                let _ = std::fs::write(&p, serde_json::to_vec(&obj).unwrap_or_default());
            }
        }
    }
    Ok(to_archive.len())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

/// Oldest unarchived conversations past KEEP_SESSIONS go; explicitly archived conversations are preserved.
fn prune_in(base: &Path) {
    let rows = read_index_in(base);
    let mut active: Vec<Value> = Vec::new();
    let mut archived: Vec<Value> = Vec::new();
    for r in rows {
        if r.get("archived").and_then(Value::as_bool).unwrap_or(false) {
            archived.push(r);
        } else {
            active.push(r);
        }
    }
    if active.len() <= KEEP_SESSIONS {
        return;
    }
    active.sort_by_key(|r| r.get("ts").and_then(Value::as_i64).unwrap_or(0));
    while active.len() > KEEP_SESSIONS {
        let old = active.remove(0);
        if let Some(id) = old.get("id").and_then(Value::as_str) {
            if let Some(p) = session_path_in(base, id) {
                let _ = std::fs::remove_file(p);
            }
        }
    }
    active.extend(archived);
    let _ = write_index_in(base, &active);
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
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_sessions_dir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "shadowzero-sessions-{}-{}-{}",
            tag,
            std::process::id(),
            ID_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn projects_crud_ordering_and_archive_older() {
        let base = temp_sessions_dir("proj-archive");
        let p1 = create_project_in(&base, &json!({"name": "Android APK", "description": "Mobile client"})).unwrap();
        let pid = p1["id"].as_str().unwrap().to_string();
        assert!(pid.starts_with('p'));

        let c1 = save_in(&base, &json!({"history": [{"role": "user", "content": "First chat"}], "project_id": pid})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let c2 = save_in(&base, &json!({"history": [{"role": "user", "content": "Second chat"}], "project_id": pid})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let c3 = save_in(&base, &json!({"history": [{"role": "user", "content": "Third chat"}], "project_id": pid})).unwrap();

        // Newest to oldest ordering
        let rows = list_in(&base);
        let ids: Vec<&str> = rows.iter().filter_map(|r| r["id"].as_str()).collect();
        assert_eq!(ids, vec![c3.as_str(), c2.as_str(), c1.as_str()]);

        // Re-saving without project_id preserves existing project_id
        save_in(&base, &json!({"id": c3, "history": [{"role": "user", "content": "Third chat updated"}]})).unwrap();
        let loaded = load_in(&base, &c3).unwrap();
        assert_eq!(loaded["project_id"].as_str(), Some(pid.as_str()));

        // Archive older than the newest 1: c1 and c2 archived, c3 stays active
        let archived_count = archive_older_in_project_in(&base, &pid, 1).unwrap();
        assert_eq!(archived_count, 2);
        let rows = list_in(&base);
        let is_arch = |id: &str| {
            rows.iter()
                .find(|r| r["id"].as_str() == Some(id))
                .and_then(|r| r.get("archived"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        };
        assert!(!is_arch(&c3));
        assert!(is_arch(&c2));
        assert!(is_arch(&c1));

        // Unarchive c2 without disturbing timestamp
        let ts_before = rows.iter().find(|r| r["id"].as_str() == Some(c2.as_str())).unwrap()["ts"].as_i64().unwrap();
        update_in(&base, &c2, &json!({"archived": false})).unwrap();
        let rows_after = list_in(&base);
        let c2_row = rows_after.iter().find(|r| r["id"].as_str() == Some(c2.as_str())).unwrap();
        assert!(!c2_row.get("archived").and_then(Value::as_bool).unwrap_or(false));
        assert_eq!(c2_row["ts"].as_i64().unwrap(), ts_before);

        // Deleting a project unassigns conversations rather than deleting them
        delete_project_in(&base, &pid).unwrap();
        assert!(list_projects_in(&base).is_empty());
        let loaded_c3 = load_in(&base, &c3).unwrap();
        assert!(loaded_c3.get("project_id").is_none());

        let _ = std::fs::remove_dir_all(&base);
    }
}
