//! Shadow's self-improvement yardstick, in Rust (the project is Rust-based; the yardstick should be too).
//!
//! Three things live here: the frozen case set (`eval/cases.jsonl`), the scorer that ranks retrieval against it, and
//! the runner that hands each case to a fresh session.  Two arms, deliberately split.  The retrieval arm touches no
//! model session at all — it is the same embedder subprocess the memory tool uses, so it is deterministic and it can
//! be run from inside the session being improved.  The answer arm cannot: it needs a live session, and the session
//! doing the asking must not be the session being judged, or the loop is grading its own homework.
//!
//! `eval/cases.jsonl` and `eval/baseline.jsonl` are frozen.  The loop may edit skills, prompts and index parameters;
//! it may not edit those two, because a yardstick the measured thing is allowed to move is not a yardstick.

pub mod json;
pub mod score;

use std::path::{Path, PathBuf};
use std::process::Command;

pub const SEP: &str = "<#sep#>"; // what the embedder splits one batch on
pub const BATCH: usize = 64; // texts per subprocess call; the model load is the slow part
pub const TIMEOUT_S: u64 = 300;

/// The repo root, found by walking up from this binary's location (crates/shadow-eval/../../) and falling back to
/// the current directory.  Hard-coded absolute paths would work on this machine and nowhere else.
pub fn repo_root() -> PathBuf {
    if let Ok(r) = std::env::var("SHADOW_ROOT") {
        return PathBuf::from(r);
    }
    let mut dir = std::env::current_exe().map(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default();
    while let Some(d) = dir {
        if d.join("eval").join("cases.jsonl").is_file() {
            return d;
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    std::env::current_dir().unwrap_or_default()
}

pub fn repo_file(rel: &str) -> Option<PathBuf> {
    let p = repo_root().join(rel);
    p.is_file().then_some(p)
}

/// One JSON per line, the house format.  A BOM is skipped (PowerShell writes one, and it would eat line 1).
/// A line that does not parse is reported and skipped, not fatal — same rule the Python tool follows, so both arms
/// read the same store the same way.
pub fn read_jsonl(path: &Path) -> Result<Vec<json::Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match json::parse(line) {
            Ok(v) => out.push(v),
            Err(e) => eprintln!("  (skipped {}:{} - {e})", path.file_name().unwrap_or_default().to_string_lossy(), n + 1),
        }
    }
    Ok(out)
}

/// Ask the embedder for one vector per text, in order, in as few subprocess calls as fit — the same protocol,
/// batching and separator rules as the memory tool, so a vector means the same thing in both.
pub fn embed(texts: &[String]) -> Result<Vec<Vec<f64>>, String> {
    let root = repo_root();
    let exe = root.join("bin").join("llama-embedding.exe");
    let model = root.join("models").join("bge-small-en-v1.5-q8_0.gguf");
    for p in [&exe, &model] {
        if !p.is_file() {
            return Err(format!("missing {}", p.display()));
        }
    }
    let mut out: Vec<Vec<f64>> = Vec::new();
    for chunk in texts.chunks(BATCH) {
        // the separator is the only thing the protocol reads literally, so a text containing it goes alone
        let solo: Vec<&String> = chunk.iter().filter(|t| t.contains(SEP)).collect();
        let group: Vec<&String> = chunk.iter().filter(|t| !t.contains(SEP)).collect();
        let mut by_text: Vec<(&String, Vec<f64>)> = Vec::new();
        let mut batches: Vec<Vec<&String>> = Vec::new();
        if !group.is_empty() {
            batches.push(group);
        }
        for t in &solo {
            batches.push(vec![t]);
        }
        for batch in batches {
            let joined = batch.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(SEP);
            let out_txt = run(&exe, &["-m", &model.to_string_lossy(), "-p", &joined, "--embd-separator", SEP, "--embd-output-format", "json"])?;
            let reply = json::parse(&out_txt).map_err(|e| format!("embedder reply is not JSON: {e}"))?;
            let data = reply.get("data").and_then(json::Value::as_arr).ok_or("embedder reply has no `data`")?;
            if data.len() != batch.len() {
                return Err(format!("embedder returned {} vectors for {} texts", data.len(), batch.len()));
            }
            // index-ordered, exactly as the Python tool sorted them before mapping back by text
            let mut ordered: Vec<&json::Value> = data.iter().collect();
            ordered.sort_by_key(|d| d.get("index").and_then(json::Value::as_f64).unwrap_or(0.0) as i64);
            for (i, item) in ordered.iter().enumerate() {
                let vec = item.get("embedding").and_then(json::Value::as_arr).ok_or("an item has no `embedding`")?;
                let v: Vec<f64> = vec.iter().filter_map(json::Value::as_f64).collect();
                by_text.push((batch[i], v));
            }
        }
        for t in chunk {
            let (_, v) = by_text.iter().find(|(k, _)| *k == t).ok_or("a text came back without a vector")?;
            out.push(v.clone());
        }
    }
    for v in &out {
        if v.len() != score::DIM {
            return Err(format!("a vector came back {} wide, not {} - wrong model?", v.len(), score::DIM));
        }
    }
    Ok(out)
}

fn run(exe: &Path, args: &[&str]) -> Result<String, String> {
    let started = std::time::Instant::now();
    let out = Command::new(exe)
        .args(args)
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    if started.elapsed().as_secs() > TIMEOUT_S {
        return Err("embedder overran its timeout".to_string());
    }
    if !out.status.success() {
        let tail = String::from_utf8_lossy(&out.stderr).lines().next_back().unwrap_or("").to_string();
        return Err(format!("embedder failed (rc {}): {tail}", out.status.code().unwrap_or(-1)));
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        return Err("embedder printed nothing".to_string());
    }
    Ok(s)
}
