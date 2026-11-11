//! What Shadow remembers about its user across conversations: short notes in one JSON file (`%APPDATA%\Shadow\memory.json`, or
//! `SHADOW_MEMORY_FILE`), local to this PC.  The model writes them with the tools `memory_save` / `memory_list` / `memory_forget`; the page
//! lists, edits and deletes them (Settings) and puts them in the start of every chat's prompt.  A note is a sentence or two; the whole set is
//! capped so the prompt stays small.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;

pub const MAX_NOTES: usize = 60;
pub const MAX_NOTE_CHARS: usize = 400;
const KINDS: &[&str] = &["user", "project", "preference", "reference"];

#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub id: u64,
    pub kind: String,
    pub text: String,
    /// Unix seconds.
    pub at: f64,
}

static NOTES: Mutex<Option<Vec<Note>>> = Mutex::new(None);

fn path() -> PathBuf {
    if let Ok(p) = std::env::var("SHADOW_MEMORY_FILE") {
        return PathBuf::from(p);
    }
    std::env::var("APPDATA").map_or_else(
        |_| PathBuf::from("memory.json"),
        |a| PathBuf::from(a).join("Shadow").join("memory.json"),
    )
}

fn to_json(notes: &[Note]) -> Value {
    json!(notes
        .iter()
        .map(|n| json!({"id": n.id, "kind": n.kind, "text": n.text, "at": n.at}))
        .collect::<Vec<_>>())
}

fn from_json(v: &Value) -> Vec<Note> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| {
            Some(Note {
                id: n.get("id")?.as_u64()?,
                kind: n.get("kind")?.as_str()?.to_string(),
                text: n.get("text")?.as_str()?.to_string(),
                at: n.get("at").and_then(Value::as_f64).unwrap_or(0.0),
            })
        })
        .collect()
}

fn with<R>(f: impl FnOnce(&mut Vec<Note>) -> R) -> R {
    let mut g = NOTES.lock().unwrap();
    let notes = g.get_or_insert_with(|| {
        std::fs::read_to_string(path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .map(|v| from_json(&v))
            .unwrap_or_default()
    });
    f(notes)
}

fn save(notes: &[Note]) {
    let p = path();
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    // written beside, then renamed: a crash does not leave half a file
    let tmp = p.with_extension("json.tmp");
    if std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(&to_json(notes)).unwrap_or_default(),
    )
    .is_ok()
    {
        let _ = std::fs::rename(&tmp, &p);
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

pub fn list() -> Value {
    with(|n| to_json(n))
}

/// Adds a note (a repeat of an existing one only refreshes it).  Errors say why a note was refused.
pub fn add(kind: &str, text: &str) -> Result<Note, String> {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return Err("the note is empty".into());
    }
    if text.chars().count() > MAX_NOTE_CHARS {
        return Err(format!(
            "a note is at most {MAX_NOTE_CHARS} characters: say it shorter"
        ));
    }
    let kind = if KINDS.contains(&kind) { kind } else { "user" }.to_string();
    with(|notes| {
        if let Some(n) = notes
            .iter_mut()
            .find(|n| n.text.eq_ignore_ascii_case(&text))
        {
            n.at = now();
            let out = n.clone();
            save(notes);
            return Ok(out);
        }
        if notes.len() >= MAX_NOTES {
            return Err(format!(
                "memory is full ({MAX_NOTES} notes): forget one first (memory_list shows them)"
            ));
        }
        let id = notes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let note = Note {
            id,
            kind,
            text,
            at: now(),
        };
        notes.push(note.clone());
        save(notes);
        Ok(note)
    })
}

pub fn update(id: u64, text: &str) -> Result<(), String> {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() || text.chars().count() > MAX_NOTE_CHARS {
        return Err(format!("a note is 1 to {MAX_NOTE_CHARS} characters"));
    }
    with(|notes| match notes.iter_mut().find(|n| n.id == id) {
        Some(n) => {
            n.text = text;
            n.at = now();
            save(notes);
            Ok(())
        }
        None => Err(format!("no note {id}")),
    })
}

pub fn forget(id: u64) -> Result<(), String> {
    with(|notes| {
        let before = notes.len();
        notes.retain(|n| n.id != id);
        if notes.len() == before {
            return Err(format!("no note {id}"));
        }
        save(notes);
        Ok(())
    })
}

/// The notes as text for the model (the tool `memory_list`).
pub fn as_text() -> String {
    with(|notes| {
        if notes.is_empty() {
            return "No notes yet.".to_string();
        }
        notes
            .iter()
            .map(|n| format!("[{}] ({}) {}", n.id, n.kind, n.text))
            .collect::<Vec<_>>()
            .join("\n")
    })
}

#[cfg(test)]
pub fn reset_for_test(p: &str) {
    std::env::set_var("SHADOW_MEMORY_FILE", p);
    *NOTES.lock().unwrap() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_are_kept_edited_and_forgotten() {
        let p =
            std::env::temp_dir().join(format!("shadow-memory-test-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&p);
        reset_for_test(p.to_str().unwrap());
        let a = add("preference", "Likes short answers").unwrap();
        let b = add("bogus", "Works on the Shadow engine in Rust").unwrap();
        assert_eq!((a.id, b.id, b.kind.as_str()), (1, 2, "user"));
        assert_eq!(
            add("user", "likes SHORT answers").unwrap().id,
            1,
            "a repeat is not a new note"
        );
        assert!(add("user", "   ").is_err());
        assert!(add("user", &"x".repeat(MAX_NOTE_CHARS + 1)).is_err());
        update(2, "Works on Shadow, a Rust + CUDA engine").unwrap();
        // a fresh start reads the file back
        reset_for_test(p.to_str().unwrap());
        assert_eq!(list().as_array().unwrap().len(), 2);
        assert!(as_text().contains("Rust + CUDA"));
        forget(1).unwrap();
        assert!(forget(1).is_err());
        assert_eq!(list().as_array().unwrap().len(), 1);
        let _ = std::fs::remove_file(&p);
    }
}
