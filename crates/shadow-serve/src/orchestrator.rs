//! Native Windows Agent Tool Orchestration layer for Shadow.
//! Executes local tools (read_file, glob, grep, edit_file, write_file, cmd) natively on Windows, and the web tools (web_search, web_fetch: `webtools.rs`).

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct ToolInvocation {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub name: String,
    pub output: String,
    pub is_error: bool,
    /// Pictures the tool returned (`data:` URLs): shown in the page's tool card and Browser pane; the model gets the text.
    pub images: Vec<String>,
}

impl ToolResult {
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "output": self.output,
            "is_error": self.is_error,
            "images": self.images,
        })
    }
}

const MAX_OUTPUT_CHARS: usize = 40_000;
const MAX_FILE_SIZE: u64 = 10_000_000;

fn tail_clip(s: String) -> String {
    clip_to(s, MAX_OUTPUT_CHARS)
}

/// A name against a pattern with `*` (any run of characters) and `?` (one character); both already lower-cased.
fn seg_match(pat: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pat.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0usize, 0usize, usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Folder segments against pattern segments, where a `**` segment stands for any number of folders (none included).
fn segs_match(pat: &[&str], path: &[&str]) -> bool {
    match pat.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => (0..=path.len()).any(|i| segs_match(rest, &path[i..])),
        Some((first, rest)) => path
            .split_first()
            .is_some_and(|(p, tail)| seg_match(first, p) && segs_match(rest, tail)),
    }
}

/// Whether a file matches the `glob` tool's pattern (already lower-cased).  With a `*` or `?` in it, a real glob: `*` within a name, `**` across folders, `?` one
/// character, a pattern with no `/` against the file name (`*.rs` finds every .rs file), one with a `/` against the path under the root or the whole path
/// (`crates/**/Cargo.toml`, `src/*.rs`).  Without, a substring of the file name, as the tool always did (`readme` finds README.md).
fn glob_file_matches(
    pattern_lc: &str,
    glob_mode: bool,
    root: &Path,
    path: &Path,
    fname: &str,
) -> bool {
    if pattern_lc.is_empty() {
        return true;
    }
    if !glob_mode {
        return fname.to_lowercase().contains(pattern_lc);
    }
    let pat = pattern_lc.replace('\\', "/");
    let pat = pat.trim_start_matches("./");
    if !pat.contains('/') {
        return seg_match(pat, &fname.to_lowercase());
    }
    let segs = |p: &Path| -> Vec<String> {
        p.to_string_lossy()
            .to_lowercase()
            .replace('\\', "/")
            .split('/')
            .filter(|s| !s.is_empty() && *s != ".")
            .map(str::to_string)
            .collect()
    };
    let pat_segs: Vec<&str> = pat.split('/').filter(|s| !s.is_empty()).collect();
    let rel = segs(path.strip_prefix(root).unwrap_or(path));
    let full = segs(path);
    fn as_refs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    segs_match(&pat_segs, &as_refs(&rel)) || segs_match(&pat_segs, &as_refs(&full))
}

/// At most `max` characters of `s` (cut on a character boundary), with a note when it was cut.
fn clip_to(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n... [Output truncated at {} characters]",
        &s[..end],
        max
    )
}

/// What one MCP tool may return into the conversation: a page snapshot can be huge, and the context window is not.
const MAX_MCP_OUTPUT_CHARS: usize = 12_000;

/// The model writes every parameter as text (`<parameter=a>17</parameter>`); a tool's schema says which are numbers, booleans, lists or objects.
/// Text that parses as the declared type becomes that type (the rest is left as it is, for the tool to refuse).
/// The session's log of tool outputs, on disk: every output of a native tool longer than `LOGGED_MIN` characters is written to `<cache>/s<start time>/<n>.txt` and starts with
/// `(output #n)`, so that when the conversation shortens it to a preview (the window is small) the model can read it again with `recall` instead of calling the tool again -
/// or not being able to (docs/STATUS.md 76).  A session is a server run; the newest `KEEP_SESSIONS` are kept.
mod toollog {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub const LOGGED_MIN: usize = 280;
    const KEEP_SESSIONS: usize = 6;
    static NEXT: AtomicU64 = AtomicU64::new(1);
    static DIR: OnceLock<PathBuf> = OnceLock::new();

    fn root() -> PathBuf {
        if let Some(d) = std::env::var_os("SHADOW_TOOL_LOG_DIR") {
            return PathBuf::from(d);
        }
        if cfg!(test) {
            return std::env::temp_dir()
                .join(format!("shadow-toollog-test-{}", std::process::id()));
        }
        std::env::var_os("LOCALAPPDATA").map_or_else(
            || std::env::temp_dir().join("shadow-toolcache"),
            |a| PathBuf::from(a).join("Shadow").join("toolcache"),
        )
    }

    fn dir() -> &'static PathBuf {
        DIR.get_or_init(|| {
            let r = root();
            let _ = std::fs::create_dir_all(&r);
            // older sessions go (only folders this module made: s<digits>)
            let mut old: Vec<(u64, PathBuf)> = std::fs::read_dir(&r)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    n.strip_prefix('s')
                        .and_then(|d| d.parse::<u64>().ok())
                        .filter(|_| e.path().is_dir())
                        .map(|t| (t, e.path()))
                })
                .collect();
            old.sort();
            while old.len() >= KEEP_SESSIONS {
                let _ = std::fs::remove_dir_all(old.remove(0).1);
            }
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let d = r.join(format!("s{t}"));
            let _ = std::fs::create_dir_all(&d);
            d
        })
    }

    /// Writes one output to the log; its number.
    pub fn save(text: &str) -> Option<u64> {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        std::fs::write(dir().join(format!("{id}.txt")), text)
            .ok()
            .map(|_| id)
    }

    pub fn load(id: u64) -> Option<String> {
        std::fs::read_to_string(dir().join(format!("{id}.txt"))).ok()
    }

    /// The numbers of the logged outputs, in order.
    pub fn ids() -> Vec<u64> {
        let mut ids: Vec<u64> = std::fs::read_dir(dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                e.file_name()
                    .to_string_lossy()
                    .strip_suffix(".txt")
                    .and_then(|n| n.parse().ok())
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Lines containing `query` (any case) in every logged output: (output number, line number, line), at most `max`.
    pub fn search(query: &str, max: usize) -> Vec<(u64, usize, String)> {
        let q = query.to_lowercase();
        let mut ids: Vec<u64> = std::fs::read_dir(dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                e.file_name()
                    .to_string_lossy()
                    .strip_suffix(".txt")
                    .and_then(|n| n.parse().ok())
            })
            .collect();
        ids.sort_unstable();
        let mut out = vec![];
        for id in ids {
            let Some(text) = load(id) else { continue };
            for (i, l) in text.lines().enumerate() {
                if l.to_lowercase().contains(&q) {
                    out.push((id, i + 1, l.chars().take(200).collect()));
                    if out.len() >= max {
                        return out;
                    }
                }
            }
        }
        out
    }
}

/// The properties of one of the native tools' schemas (empty for a tool that is not one of them).
fn tool_props(name: &str) -> serde_json::Map<String, Value> {
    let defs = tool_definitions();
    defs.as_array()
        .and_then(|a| {
            a.iter()
                .find(|d| d.get("name").and_then(Value::as_str) == Some(name))
        })
        .and_then(|d| d.pointer("/input_schema/properties"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// Other names a model gives a tool's arguments (it knows `start_line` / `end_line` / `file_path` from other tools): (tool, the name it used, ours).  Without these `read_file`
/// ignored `start_line` and always answered with lines 1-400, and the model read the top of the same file again and again (docs/STATUS.md 73).
const ARGUMENT_ALIASES: &[(&str, &str, &str)] = &[
    ("read_file", "start_line", "start"),
    ("read_file", "line_start", "start"),
    ("read_file", "startLine", "start"),
    ("read_file", "from_line", "start"),
    ("read_file", "end_line", "end"),
    ("read_file", "line_end", "end"),
    ("read_file", "endLine", "end"),
    ("read_file", "to_line", "end"),
    ("read_file", "file_path", "path"),
    ("read_file", "filepath", "path"),
    ("read_file", "file", "path"),
    ("edit_file", "file_path", "path"),
    ("edit_file", "filepath", "path"),
    ("edit_file", "file", "path"),
    ("write_file", "file_path", "path"),
    ("write_file", "filepath", "path"),
    ("write_file", "file", "path"),
    ("replace_lines", "start_line", "start"),
    ("replace_lines", "line_start", "start"),
    ("replace_lines", "end_line", "end"),
    ("replace_lines", "line_end", "end"),
    ("replace_lines", "file_path", "path"),
    ("replace_lines", "filepath", "path"),
    ("replace_lines", "file", "path"),
    ("replace_lines", "new_text", "newText"),
    ("replace_lines", "newtext", "newText"),
    ("replace_lines", "text", "newText"),
    ("replace_lines", "content", "newText"),
    ("replace_lines", "expected", "expect"),
    ("replace_lines", "contains", "expect"),
    ("recall", "output", "id"),
    ("recall", "output_id", "id"),
    ("recall", "number", "id"),
    ("recall", "n", "id"),
    ("recall", "text", "query"),
    ("recall", "search", "query"),
    ("recall", "start_line", "start"),
    ("recall", "end_line", "end"),
    ("search", "q", "query"),
    ("search", "text", "query"),
    ("search", "question", "query"),
    ("search", "limit", "max_results"),
    ("search", "k", "max_results"),
    ("search", "folder", "path"),
    ("search", "directory", "path"),
];

pub fn coerce_arguments(name: &str, arguments: &Value) -> Value {
    let props = tool_props(name);
    let Some(obj) = arguments.as_object() else {
        return arguments.clone();
    };
    let mut obj = obj.clone();
    for (tool, from, to) in ARGUMENT_ALIASES {
        if *tool == name && !obj.contains_key(*to) && props.contains_key(*to) {
            if let Some(v) = obj.remove(*from) {
                obj.insert((*to).to_string(), v);
            }
        }
    }
    let mut out = obj.clone();
    for (k, v) in &obj {
        let (Some(s), Some(ty)) = (
            v.as_str(),
            props
                .get(k)
                .and_then(|p| p.get("type"))
                .and_then(Value::as_str),
        ) else {
            continue;
        };
        let parsed = match ty {
            "integer" => s.trim().parse::<i64>().ok().map(Value::from),
            "number" => s.trim().parse::<f64>().ok().map(|f| {
                if f.fract() == 0.0 && f.abs() < 9e15 {
                    Value::from(f as i64)
                } else {
                    Value::from(f)
                }
            }),
            "boolean" => match s.trim().to_ascii_lowercase().as_str() {
                "true" => Some(Value::Bool(true)),
                "false" => Some(Value::Bool(false)),
                _ => None,
            },
            "array" | "object" => serde_json::from_str::<Value>(s.trim()).ok().filter(|p| {
                if ty == "array" {
                    p.is_array()
                } else {
                    p.is_object()
                }
            }),
            _ => None,
        };
        if let Some(p) = parsed {
            out.insert(k.clone(), p);
        }
    }
    Value::Object(out)
}

pub fn execute_tool(inv: &ToolInvocation) -> ToolResult {
    let mut r = execute_tool_inner(inv);
    // a long output goes to the session's log and says its number (`recall` reads it again once the conversation has shortened it)
    if !r.is_error
        && inv.name != "recall"
        && inv.name != "search"
        && r.output.chars().count() > toollog::LOGGED_MIN
    {
        if let Some(id) = toollog::save(&r.output) {
            r.output = format!("(output #{id})\n{}", r.output);
        }
    }
    // an argument the tool does not have is ignored: say so, or the model goes on calling with a name that does nothing
    let props = tool_props(&inv.name);
    if let (false, Some(obj)) = (
        props.is_empty(),
        coerce_arguments(&inv.name, &inv.arguments).as_object(),
    ) {
        let unknown: Vec<&str> = obj
            .keys()
            .map(String::as_str)
            .filter(|k| !props.contains_key(*k))
            .collect();
        if !unknown.is_empty() {
            let known: Vec<&str> = props.keys().map(String::as_str).collect();
            r.output.push_str(&format!(
                "
(note: ignored unknown argument(s) {}; this tool's arguments are: {})",
                unknown.join(", "),
                known.join(", ")
            ));
        }
    }
    r
}

/// `replace_lines`: lines `start`..=`end` (1-indexed, the numbers `read_file` shows) of a file become `newText` (empty: they are deleted; `start` one past the last line: it is
/// appended).  `expect`, when given, must occur in the old lines: a line number that went stale (an earlier edit moved the lines) is refused instead of overwriting the wrong
/// code.  The file's line endings are kept.  The answer shows the lines around the edit with their new numbers.
fn replace_lines(name: &str, args: &Value) -> ToolResult {
    let err = |m: String| ToolResult {
        name: name.to_string(),
        output: format!("error: {m}"),
        is_error: true,
        images: vec![],
    };
    let path_str = args.get("path").and_then(Value::as_str).unwrap_or("");
    let Some(start) = args
        .get("start")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
    else {
        return err("give start, the first line to replace (1-indexed)".into());
    };
    let end_arg = args.get("end").and_then(Value::as_u64).map(|v| v as usize);
    let Some(new_text) = args.get("newText").and_then(Value::as_str) else {
        return err("give newText, the replacement (an empty string deletes the lines)".into());
    };
    if path_str.is_empty() {
        return err("missing required argument 'path'".into());
    }
    let p = Path::new(path_str);
    let content = match std::fs::read_to_string(p) {
        Ok(c) => c,
        Err(e) => return err(format!("reading {path_str}: {e}")),
    };
    let crlf = content.contains("\r\n");
    let trailing_newline = content.ends_with('\n');
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let end = end_arg.unwrap_or(if start == total + 1 { total } else { start }); // one line by default; nothing (an append) when start is past the last line
    if start < 1 || start > total + 1 || end > total || end + 1 < start {
        return err(format!("lines {start}-{end} are outside {path_str}, which has {total} lines (start may be {} to append)", total + 1));
    }
    let old = &lines[start - 1..end];
    if let Some(expect) = args
        .get("expect")
        .and_then(Value::as_str)
        .filter(|e| !e.is_empty())
    {
        if !old.join("\n").contains(expect) {
            let shown: Vec<String> = old
                .iter()
                .take(20)
                .enumerate()
                .map(|(i, l)| format!("{:>5}| {l}", start + i))
                .collect();
            return err(format!("the text you expected is not in lines {start}-{end} of {path_str} (the file may have changed since you read it); those lines are now:\n{}", shown.join("\n")));
        }
    }
    let body = new_text
        .strip_suffix("\r\n")
        .or_else(|| new_text.strip_suffix('\n'))
        .unwrap_or(new_text);
    let new_lines: Vec<&str> = if new_text.is_empty() {
        vec![]
    } else {
        body.split('\n')
            .map(|l| l.strip_suffix('\r').unwrap_or(l))
            .collect()
    };
    let mut out: Vec<&str> = Vec::with_capacity(total + new_lines.len());
    out.extend_from_slice(&lines[..start - 1]);
    out.extend_from_slice(&new_lines);
    out.extend_from_slice(&lines[end..]);
    let nl = if crlf { "\r\n" } else { "\n" };
    let mut updated = out.join(nl);
    if trailing_newline && !out.is_empty() {
        updated.push_str(nl);
    }
    if let Err(e) = std::fs::write(p, updated) {
        return err(format!("writing {path_str}: {e}"));
    }
    let delta = new_lines.len() as i64 - old.len() as i64;
    let from = start.saturating_sub(2).max(1);
    let to = (start + new_lines.len() + 1).min(out.len()).min(from + 39);
    let shown: Vec<String> = (from..=to)
        .filter(|i| *i >= 1 && *i <= out.len())
        .map(|i| format!("{:>5}| {}", i, out[i - 1]))
        .collect();
    let moved = if delta == 0 {
        String::new()
    } else {
        format!("; the lines after line {end} moved by {delta:+} - read again before the next edit of this file")
    };
    ToolResult {
        name: name.to_string(),
        output: format!(
            "replaced lines {start}-{end} ({} -> {} lines) in {path_str}{moved}\n{}",
            old.len(),
            new_lines.len(),
            shown.join("\n")
        ),
        is_error: false,
        images: vec![],
    }
}

fn execute_tool_inner(inv: &ToolInvocation) -> ToolResult {
    let fixed = ToolInvocation {
        name: inv.name.clone(),
        arguments: coerce_arguments(&inv.name, &inv.arguments),
    };
    let inv = &fixed;
    match inv.name.as_str() {
        "memory_save" => {
            let text = inv
                .arguments
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("");
            let kind = inv
                .arguments
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("user");
            match crate::memory::add(kind, text) {
                Ok(n) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("Saved note {}: {}", n.id, n.text),
                    is_error: false,
                    images: vec![],
                },
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }
        "finish" if finish_tool_enabled() => finish_ack(&inv.arguments),
        "memory_list" => ToolResult {
            name: inv.name.clone(),
            output: crate::memory::as_text(),
            is_error: false,
            images: vec![],
        },
        "memory_forget" => {
            let id = inv.arguments.get("id").and_then(Value::as_u64).unwrap_or(0);
            match crate::memory::forget(id) {
                Ok(()) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("Forgot note {id}."),
                    is_error: false,
                    images: vec![],
                },
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }
        "web_search" => {
            let q = inv
                .arguments
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("");
            let n = inv
                .arguments
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(8) as usize;
            match crate::webtools::web_search(q, n) {
                Ok(s) => ToolResult {
                    name: inv.name.clone(),
                    output: tail_clip(s),
                    is_error: false,
                    images: vec![],
                },
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }
        "web_fetch" => {
            let url = inv
                .arguments
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or("");
            let start = inv
                .arguments
                .get("start")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let max = inv
                .arguments
                .get("max_chars")
                .and_then(Value::as_u64)
                .unwrap_or(12_000) as usize;
            match crate::webtools::web_fetch(url, start, max) {
                Ok(s) => ToolResult {
                    name: inv.name.clone(),
                    output: tail_clip(s),
                    is_error: false,
                    images: vec![],
                },
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }
        "read_file" => {
            let path_str = inv
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("");
            let start = inv
                .arguments
                .get("start")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;
            // 120 lines from `start` unless `end` is given (the old default was line 400, whatever `start` was: a read from line 2415 returned one line)
            let end = inv
                .arguments
                .get("end")
                .and_then(Value::as_u64)
                .map_or(start + 119, |e| e as usize);

            if path_str.is_empty() {
                return ToolResult {
                    name: inv.name.clone(),
                    output: "error: missing required argument 'path'".into(),
                    is_error: true,
                    images: vec![],
                };
            }
            let p = Path::new(path_str);
            if !p.exists() {
                return ToolResult {
                    name: inv.name.clone(),
                    output: format!("error: file does not exist: {path_str}"),
                    is_error: true,
                    images: vec![],
                };
            }
            if let Ok(meta) = p.metadata() {
                if meta.len() > MAX_FILE_SIZE {
                    return ToolResult {
                        name: inv.name.clone(),
                        output: "error: file exceeds maximum allowed size (10 MB)".into(),
                        is_error: true,
                        images: vec![],
                    };
                }
            }
            match std::fs::read_to_string(p) {
                Ok(content) => {
                    let lines: Vec<&str> = content.lines().collect();
                    let total = lines.len();
                    let s_idx = start.max(1).min(total.max(1));
                    let e_idx = end.min(total).max(s_idx);
                    let mut out = String::new();
                    for i in s_idx..=e_idx {
                        if i <= total {
                            out.push_str(&format!("{:>5}| {}\n", i, lines[i - 1]));
                        }
                    }
                    if e_idx < total {
                        out.push_str(&format!("(Showing lines {s_idx}-{e_idx} of {total}; ask again with start={} for more)\n", e_idx + 1));
                    }
                    ToolResult {
                        name: inv.name.clone(),
                        output: tail_clip(out),
                        is_error: false,
                        images: vec![],
                    }
                }
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error reading file: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }

        "recall" => {
            let id = inv.arguments.get("id").and_then(Value::as_u64);
            let query = inv
                .arguments
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let start = inv
                .arguments
                .get("start")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;
            let end = inv
                .arguments
                .get("end")
                .and_then(Value::as_u64)
                .map_or(start + 99, |e| e as usize); // 100 lines from `start` unless `end` is given
            let out = match (id, query.is_empty()) {
                (None, true) => return ToolResult { name: inv.name.clone(), output: "error: give the output's number (id), a text to search for (query), or both".into(), is_error: true, images: vec![] },
                (None, false) => {
                    let hits = toollog::search(&query, 60);
                    if hits.is_empty() {
                        format!("no line of an earlier output of this session contains \"{query}\"")
                    } else {
                        hits.iter().map(|(n, l, t)| format!("#{n}:{l}| {t}")).collect::<Vec<_>>().join("\n") + "\n(read one with recall id=N, start=, end=)"
                    }
                }
                (Some(n), _) => match toollog::load(n) {
                    None => return ToolResult { name: inv.name.clone(), output: format!("error: no output #{n} in this session's log (it holds the outputs since the server started)"), is_error: true, images: vec![] },
                    Some(text) => {
                        let body = text.strip_prefix(&format!("(output #{n})\n")).unwrap_or(&text);
                        let lines: Vec<&str> = body.lines().collect();
                        let total = lines.len();
                        if !query.is_empty() {
                            let q = query.to_lowercase();
                            let hits: Vec<String> = lines.iter().enumerate().filter(|(_, l)| l.to_lowercase().contains(&q)).take(120).map(|(i, l)| format!("{:>5}| {l}", i + 1)).collect();
                            format!("output #{n}, the lines containing \"{query}\" (of {total}):\n{}", hits.join("\n"))
                        } else {
                            let s_idx = start.max(1).min(total.max(1));
                            let e_idx = end.min(total).max(s_idx);
                            let mut o = format!("output #{n}, lines {s_idx}-{e_idx} of {total}:\n");
                            for i in s_idx..=e_idx.min(total) {
                                o.push_str(&format!("{:>5}| {}\n", i, lines[i - 1]));
                            }
                            if e_idx < total {
                                o.push_str(&format!("(more: ask again with start={})\n", e_idx + 1));
                            }
                            o
                        }
                    }
                },
            };
            ToolResult {
                name: inv.name.clone(),
                output: tail_clip(out),
                is_error: false,
                images: vec![],
            }
        }

        "search" => match crate::semsearch::tool(&inv.arguments, &toollog::ids(), &toollog::load) {
            Ok(s) => ToolResult {
                name: inv.name.clone(),
                output: tail_clip(s),
                is_error: false,
                images: vec![],
            },
            Err(e) => ToolResult {
                name: inv.name.clone(),
                output: format!("error: {e}"),
                is_error: true,
                images: vec![],
            },
        },

        "glob" => {
            let pattern = inv
                .arguments
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("*");
            let root = inv
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or(".");
            let max_results = inv
                .arguments
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(100) as usize;

            let mut matches = Vec::new();
            let mut stack = vec![PathBuf::from(root)];
            let glob_mode = pattern.contains('*') || pattern.contains('?');
            let search_term = pattern.to_lowercase();
            let root_path = PathBuf::from(root);

            while let Some(dir) = stack.pop() {
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if fname.starts_with('.')
                            || fname == "target"
                            || fname == "node_modules"
                            || fname == ".git"
                        {
                            continue;
                        }
                        if path.is_dir() {
                            stack.push(path);
                        } else if glob_file_matches(
                            &search_term,
                            glob_mode,
                            &root_path,
                            &path,
                            fname,
                        ) {
                            matches.push(path.to_string_lossy().to_string());
                            if matches.len() >= max_results {
                                break;
                            }
                        }
                    }
                }
                if matches.len() >= max_results {
                    break;
                }
            }
            if matches.is_empty() {
                ToolResult {
                    name: inv.name.clone(),
                    output: "no matching files found".into(),
                    is_error: false,
                    images: vec![],
                }
            } else {
                ToolResult {
                    name: inv.name.clone(),
                    output: tail_clip(matches.join("\n")),
                    is_error: false,
                    images: vec![],
                }
            }
        }

        "grep" => {
            let pattern = inv
                .arguments
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("");
            let root = inv
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or(".");
            let ignore_case = inv
                .arguments
                .get("ignore_case")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let max_results = inv
                .arguments
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(50) as usize;

            if pattern.is_empty() {
                return ToolResult {
                    name: inv.name.clone(),
                    output: "error: missing required argument 'pattern'".into(),
                    is_error: true,
                    images: vec![],
                };
            }

            let needle = if ignore_case {
                pattern.to_lowercase()
            } else {
                pattern.to_string()
            };
            // the schema says "substring or regex": a line matches when it contains the text or the text, read as a regex (when it is one), matches it
            let re = regex::RegexBuilder::new(pattern)
                .case_insensitive(ignore_case)
                .build()
                .ok();
            let mut matches = Vec::new();
            let mut stack = vec![PathBuf::from(root)];

            while let Some(dir) = stack.pop() {
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if fname.starts_with('.')
                            || fname == "target"
                            || fname == "node_modules"
                            || fname == ".git"
                        {
                            continue;
                        }
                        if path.is_dir() {
                            stack.push(path);
                        } else if let Ok(meta) = path.metadata() {
                            if meta.len() < MAX_FILE_SIZE {
                                if let Ok(content) = std::fs::read_to_string(&path) {
                                    for (line_idx, line) in content.lines().enumerate() {
                                        let matched =
                                            (if ignore_case {
                                                line.to_lowercase().contains(&needle)
                                            } else {
                                                line.contains(&needle)
                                            }) || re.as_ref().is_some_and(|r| r.is_match(line));
                                        if matched {
                                            matches.push(format!(
                                                "{}:{}: {}",
                                                path.display(),
                                                line_idx + 1,
                                                line.trim()
                                            ));
                                            if matches.len() >= max_results {
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if matches.len() >= max_results {
                            break;
                        }
                    }
                }
                if matches.len() >= max_results {
                    break;
                }
            }

            if matches.is_empty() {
                ToolResult {
                    name: inv.name.clone(),
                    output: format!("no occurrences found for '{pattern}'"),
                    is_error: false,
                    images: vec![],
                }
            } else {
                ToolResult {
                    name: inv.name.clone(),
                    output: tail_clip(matches.join("\n")),
                    is_error: false,
                    images: vec![],
                }
            }
        }

        "edit_file" => {
            let path_str = inv
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("");
            let old_text = inv
                .arguments
                .get("oldText")
                .or_else(|| inv.arguments.get("old_text"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let new_text = inv
                .arguments
                .get("newText")
                .or_else(|| inv.arguments.get("new_text"))
                .and_then(Value::as_str)
                .unwrap_or("");

            if path_str.is_empty() || old_text.is_empty() {
                return ToolResult {
                    name: inv.name.clone(),
                    output: "error: missing required arguments 'path' or 'oldText'".into(),
                    is_error: true,
                    images: vec![],
                };
            }
            let p = Path::new(path_str);
            match std::fs::read_to_string(p) {
                Ok(content) => {
                    let count = content.matches(old_text).count();
                    if count == 0 {
                        ToolResult {
                            name: inv.name.clone(),
                            output: format!("error: 'oldText' was not found in {path_str}"),
                            is_error: true,
                            images: vec![],
                        }
                    } else if count > 1 {
                        ToolResult { name: inv.name.clone(), output: format!("error: 'oldText' matches {count} times in {path_str}; provide more surrounding context to disambiguate"), is_error: true, images: vec![] }
                    } else {
                        let updated = content.replace(old_text, new_text);
                        if let Err(e) = std::fs::write(p, updated) {
                            ToolResult {
                                name: inv.name.clone(),
                                output: format!("error writing edits to {path_str}: {e}"),
                                is_error: true,
                                images: vec![],
                            }
                        } else {
                            ToolResult {
                                name: inv.name.clone(),
                                output: format!("successfully edited {path_str}"),
                                is_error: false,
                                images: vec![],
                            }
                        }
                    }
                }
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error reading {path_str}: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }

        "replace_lines" => replace_lines(&inv.name, &inv.arguments),

        "write_file" => {
            let path_str = inv
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("");
            let content = inv
                .arguments
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("");

            if path_str.is_empty() {
                return ToolResult {
                    name: inv.name.clone(),
                    output: "error: missing required argument 'path'".into(),
                    is_error: true,
                    images: vec![],
                };
            }
            let p = Path::new(path_str);
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(p, content) {
                Ok(_) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("successfully wrote {} bytes to {path_str}", content.len()),
                    is_error: false,
                    images: vec![],
                },
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error writing to {path_str}: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }

        "cmd" | "bash" | "powershell" => {
            let command = inv
                .arguments
                .get("command")
                .or_else(|| inv.arguments.get("cmd"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let cwd = inv
                .arguments
                .get("cwd")
                .and_then(Value::as_str)
                .unwrap_or(".");

            if command.is_empty() {
                return ToolResult {
                    name: inv.name.clone(),
                    output: "error: missing required argument 'command'".into(),
                    is_error: true,
                    images: vec![],
                };
            }

            // Execute natively via Windows PowerShell
            let mut cmd = Command::new("powershell");
            cmd.args([
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]);
            cmd.current_dir(cwd);

            match cmd.output() {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let mut combined = stdout.trim().to_string();
                    if !stderr.trim().is_empty() {
                        if !combined.is_empty() {
                            combined.push('\n');
                        }
                        combined.push_str("STDERR:\n");
                        combined.push_str(stderr.trim());
                    }
                    if combined.is_empty() {
                        combined = "(Command executed successfully with no output)".into();
                    }
                    ToolResult {
                        name: inv.name.clone(),
                        output: tail_clip(combined),
                        is_error: !output.status.success(),
                        images: vec![],
                    }
                }
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error launching PowerShell command on Windows: {e}"),
                    is_error: true,
                    images: vec![],
                },
            }
        }

        other if other.starts_with("mcp__") => match crate::mcp::call(other, &inv.arguments) {
            Ok(o) => ToolResult {
                name: inv.name.clone(),
                output: clip_to(o.text, MAX_MCP_OUTPUT_CHARS),
                is_error: false,
                images: o.images,
            },
            Err(e) => ToolResult {
                name: inv.name.clone(),
                output: format!("error: {e}"),
                is_error: true,
                images: vec![],
            },
        },

        other => ToolResult {
            name: other.to_string(),
            output: format!("unknown tool '{other}'"),
            is_error: true,
            images: vec![],
        },
    }
}

/// Whether the `finish` tool (the closing status of an agent task, docs/STATUS.md 109) is offered: on unless `SHADOW_FINISH_TOOL=0`.
pub fn finish_tool_enabled() -> bool {
    finish_enabled_from(std::env::var("SHADOW_FINISH_TOOL").ok().as_deref())
}

fn finish_enabled_from(v: Option<&str>) -> bool {
    !matches!(v.map(str::trim), Some("0") | Some("off") | Some("false"))
}

/// Shadow's own tools and those of the MCP servers that are up (`mcp.rs`).
pub fn tool_definitions() -> Value {
    let mut v = builtin_tool_definitions_with(finish_tool_enabled());
    v.as_array_mut()
        .expect("an array")
        .extend(crate::mcp::tool_definitions());
    v
}

/// The `finish` tool: the model ends a task that used tools by saying what is done, what is pending or unverified and, only when it cannot go on without the user, one
/// question with its options.  The chat page draws it as a status card and ends the turn; to an API client it is a plain tool whose execution is an acknowledgement.
/// Every field is a string (the page's tool-call parser hands over strings): one item a line, or a JSON array as text.
fn finish_tool_definition() -> Value {
    json!({
        "name": "finish",
        "description": "End a task that needed tools by reporting its status, instead of a closing message: what you did, what is NOT done or NOT verified, and - only if you cannot go on without the user - one question. Call it alone, after you have seen the results of your other tool calls. Do not use it to answer a plain question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "done": { "type": "string", "description": "What is finished, one short item per line (file names, what changed). Only what you actually did." },
                "pending": { "type": "string", "description": "What is NOT done, NOT tested or NOT verified (say so if you did not run the code or the tests), one item per line. Leave empty only when nothing is left." },
                "question": { "type": "string", "description": "Only when you need the user's input to go on: one question. Leave empty otherwise." },
                "options": { "type": "string", "description": "With a question: the answers you offer, one per line (the first is the one you recommend)." }
            },
            "required": ["done"]
        }
    })
}

fn builtin_tool_definitions_with(finish: bool) -> Value {
    let mut v = builtin_tool_definitions_core();
    if finish {
        v.as_array_mut().expect("an array").push(finish_tool_definition());
    }
    v
}

/// The acknowledgement of `finish`: short and not an error (the page ends the task and never reads it; an API client executing the call through `/v1/tools/execute` gets
/// it, and its own arguments are the status).
fn finish_ack(args: &Value) -> ToolResult {
    let text = |k: &str| {
        args.get(k)
            .map(|v| match v {
                Value::String(s) => s.trim().to_string(),
                Value::Array(a) => a
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            })
            .unwrap_or_default()
    };
    if text("done").is_empty() && text("pending").is_empty() && text("question").is_empty() {
        return ToolResult {
            name: "finish".into(),
            output: "error: finish needs done (what you did), and pending (what is not done or not verified) or a question".into(),
            is_error: true,
            images: vec![],
        };
    }
    ToolResult {
        name: "finish".into(),
        output: if text("question").is_empty() {
            "Status recorded; the task is closed.".into()
        } else {
            "Status recorded; waiting for the user's answer.".into()
        },
        is_error: false,
        images: vec![],
    }
}

fn builtin_tool_definitions_core() -> Value {
    json!([
        {
            "name": "memory_save",
            "description": "Remember something about the user for future conversations: one short note (a fact, a preference, a project, how they like answers). Not passwords, keys or private data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The note, one or two sentences" },
                    "kind": { "type": "string", "description": "user, project, preference or reference" }
                },
                "required": ["text"]
            }
        },
        {
            "name": "memory_list",
            "description": "List the notes remembered about the user (each with its id).",
            "input_schema": { "type": "object", "properties": {} }
        },
        {
            "name": "memory_forget",
            "description": "Forget a note by its id (see memory_list), when it is wrong, outdated or the user asks.",
            "input_schema": {
                "type": "object",
                "properties": { "id": { "type": "integer", "description": "The note's id" } },
                "required": ["id"]
            }
        },
        {
            "name": "web_search",
            "description": "Search the web (DuckDuckGo) and get titles, URLs and snippets. Use web_fetch to read a result.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to search for" },
                    "max_results": { "type": "integer", "description": "How many results (default 8, at most 15)" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "web_fetch",
            "description": "Fetch a web page (http or https) as readable text; long pages come in parts (use start to continue).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The page address, https://..." },
                    "start": { "type": "integer", "description": "Character to start from (default 0)" },
                    "max_chars": { "type": "integer", "description": "Characters to return (default 12000, at most 40000)" }
                },
                "required": ["url"]
            }
        },
        {
            "name": "read_file",
            "description": "Read file contents with line numbers on Windows: 120 lines from start unless end is given.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute or relative Windows file path" },
                    "start": { "type": "integer", "description": "Starting line number (1-indexed)" },
                    "end": { "type": "integer", "description": "Ending line number (inclusive)" }
                },
                "required": ["path"]
            }
        },
        {
            "name": "recall",
            "description": "Read again something earlier in this conversation that was shortened: a tool output by its number (id) or any message by its number (message), optionally a line range or only the lines containing a text; with only a text, search every earlier message and tool output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer", "description": "The output's number, as in '(output #12)' or 'earlier tool output #12'" },
                    "message": { "type": "integer", "description": "The number of an earlier message of the conversation, as in 'messages m3-m9' of a summary or 'recall with message 14'" },
                    "start": { "type": "integer", "description": "First line to show (1-indexed)" },
                    "end": { "type": "integer", "description": "Last line to show (inclusive; 100 lines from start by default)" },
                    "query": { "type": "string", "description": "Show only the lines containing this text (in one output with id, in all of them without)" }
                }
            }
        },
        {
            "name": "search",
            "description": "Search the project's files and this session's earlier tool outputs by KEYWORDS and by MEANING together (grep is for exact text and patterns): put in the distinctive words you expect (names, flags, identifiers - an identifier matches whole and by its parts: shadow_fused_moe, fused, moe) and, if you can, a few words of what you are after. Returns the best matching pieces with file, line and how they matched (keywords, meaning). Keyword matching works at once; the meaning part is filled in by a background index that is kept on disk, and the answer says when it is not (yet) available.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What you are looking for, in words" },
                    "max_results": { "type": "integer", "description": "How many pieces to return (default 6, at most 15)" },
                    "path": { "type": "string", "description": "Only under this folder or file (relative to the working folder)" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "glob",
            "description": "Find files matching pattern under a folder on Windows.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "A glob such as *.rs, **/*.toml or crates/**/Cargo.toml (without a / it is matched against file names at any depth; with one, against the path under 'path'), or plain text that is part of a file name" },
                    "path": { "type": "string", "description": "Root directory to search (defaults to current dir)" },
                    "max_results": { "type": "integer", "description": "Maximum number of results" }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "grep",
            "description": "Search inside files under a folder for a text pattern on Windows.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Substring or regex pattern to search for" },
                    "path": { "type": "string", "description": "Root directory (defaults to current dir)" },
                    "ignore_case": { "type": "boolean", "description": "Whether to search case-insensitively" },
                    "max_results": { "type": "integer", "description": "Maximum matches to return" }
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "edit_file",
            "description": "Replace exact text inside a file on Windows.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to modify" },
                    "oldText": { "type": "string", "description": "Exact text to replace" },
                    "newText": { "type": "string", "description": "Replacement text" }
                },
                "required": ["path", "oldText", "newText"]
            }
        },
        {
            "name": "replace_lines",
            "description": "Replace lines start..end (inclusive, 1-indexed, the numbers read_file shows) of a file with newText: no need to repeat the old text. An empty newText deletes the lines; start one past the last line appends. Give expect, a short piece of text that must be in the old lines, so a stale line number cannot overwrite the wrong code. The lines after the edit move: read the file again before the next edit of it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to modify" },
                    "start": { "type": "integer", "description": "First line to replace (1-indexed)" },
                    "end": { "type": "integer", "description": "Last line to replace (inclusive; default start)" },
                    "newText": { "type": "string", "description": "The replacement text (may be several lines; empty deletes)" },
                    "expect": { "type": "string", "description": "Text that must occur in the old lines (guards against a stale line number)" }
                },
                "required": ["path", "start", "newText"]
            }
        },
        {
            "name": "write_file",
            "description": "Write or overwrite a file on Windows.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path of the file to create/overwrite" },
                    "content": { "type": "string", "description": "File contents" }
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "cmd",
            "description": "Execute a command line natively in Windows PowerShell (e.g. cargo test, git status, dir).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "PowerShell command to run on Windows" },
                    "cwd": { "type": "string", "description": "Working directory" }
                },
                "required": ["command"]
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_is_offered_unless_switched_off() {
        for on in [None, Some(""), Some("1"), Some("on")] {
            assert!(finish_enabled_from(on), "{on:?}");
        }
        for off in [Some("0"), Some("off"), Some(" 0 "), Some("false")] {
            assert!(!finish_enabled_from(off), "{off:?}");
        }
        let names = |v: Value| -> Vec<String> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|d| d["name"].as_str().unwrap().to_string())
                .collect()
        };
        let with = names(builtin_tool_definitions_with(true));
        let without = names(builtin_tool_definitions_with(false));
        assert_eq!(with.last().map(String::as_str), Some("finish"));
        assert!(!without.contains(&"finish".to_string()));
        assert_eq!(with.len(), without.len() + 1);
        // every property of finish is a string: the page's tool-call parser hands over strings
        let def = finish_tool_definition();
        for (k, p) in def["input_schema"]["properties"].as_object().unwrap() {
            assert_eq!(p["type"], "string", "{k}");
        }
    }

    #[test]
    fn finish_acknowledges_a_status_and_refuses_an_empty_one() {
        let r = finish_ack(&json!({"done": "edited a.rs", "pending": "tests not run"}));
        assert!(!r.is_error && r.output.len() < toollog::LOGGED_MIN && !r.output.contains("(output #"));
        assert!(r.output.contains("closed"), "{}", r.output);
        let q = finish_ack(&json!({"done": "x", "question": "which one?", "options": "a
b"}));
        assert!(!q.is_error && q.output.contains("waiting"), "{}", q.output);
        // an array as the status of an API client works too
        assert!(!finish_ack(&json!({"done": ["a", "b"]})).is_error);
        // nothing said: the model is told what is missing and the task stays open
        for empty in [json!({}), json!({"done": "  "}), json!({"done": ""})] {
            let e = finish_ack(&empty);
            assert!(e.is_error && e.output.contains("needs done"), "{}", e.output);
        }
    }

    #[test]
    fn text_parameters_take_the_schema_types() {
        let a = coerce_arguments(
            "read_file",
            &json!({"path": "C:\\x.txt", "start": "5", "end": "20"}),
        );
        assert_eq!(a["start"], json!(5));
        assert_eq!(a["end"], json!(20));
        assert_eq!(a["path"], json!("C:\\x.txt"));
        assert_eq!(
            coerce_arguments("read_file", &json!({"start": "abc"}))["start"],
            json!("abc")
        );
        assert_eq!(
            coerce_arguments("nope", &json!({"a": "1"}))["a"],
            json!("1")
        );
    }

    #[test]
    fn replace_lines_edits_by_line_number_and_refuses_a_stale_one() {
        let dir = std::env::temp_dir().join(format!("shadow_replace_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("a.txt");
        let path = f.to_str().unwrap();
        let call = |args: Value| {
            execute_tool(&ToolInvocation {
                name: "replace_lines".into(),
                arguments: args,
            })
        };
        let text = |s: &str| std::fs::write(&f, s).unwrap();
        // two lines replace one: the answer shows the new numbers and says the rest moved
        text("one\ntwo\nthree\nfour\nfive\n");
        let r = call(
            json!({"path": path, "start": 3, "end": 3, "newText": "THREE-A\nTHREE-B", "expect": "three"}),
        );
        assert!(
            !r.is_error
                && r.output.contains("(1 -> 2 lines)")
                && r.output.contains("moved by +1")
                && r.output.contains("    3| THREE-A")
                && r.output.contains("    4| THREE-B"),
            "{}",
            r.output
        );
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "one\ntwo\nTHREE-A\nTHREE-B\nfour\nfive\n"
        );
        // a stale line number: the expected text is not there, the file is untouched and the current lines are shown
        let r =
            call(json!({"path": path, "start": 3, "end": 3, "newText": "X", "expect": "three"}));
        assert!(
            r.is_error && r.output.contains("not in lines 3-3") && r.output.contains("THREE-A"),
            "{}",
            r.output
        );
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "one\ntwo\nTHREE-A\nTHREE-B\nfour\nfive\n"
        );
        // a range, replaced by one line (the argument names a model might use are taken)
        let r = call(json!({"file_path": path, "start_line": 2, "end_line": 4, "new_text": "MID"}));
        assert!(
            !r.is_error && r.output.contains("(3 -> 1 lines)") && r.output.contains("moved by -2"),
            "{}",
            r.output
        );
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "one\nMID\nfour\nfive\n"
        );
        // delete (an empty newText) and append (one past the last line)
        assert!(!call(json!({"path": path, "start": 2, "end": 2, "newText": ""})).is_error);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "one\nfour\nfive\n");
        assert!(!call(json!({"path": path, "start": 4, "newText": "six\n"})).is_error);
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "one\nfour\nfive\nsix\n"
        );
        // out of range, and a missing newText (never read as "delete")
        assert!(call(json!({"path": path, "start": 9, "newText": "x"}))
            .output
            .contains("outside"));
        assert!(call(json!({"path": path, "start": 1, "end": 7, "newText": "x"})).is_error);
        let r = call(json!({"path": path, "start": 1}));
        assert!(r.is_error && r.output.contains("newText"), "{}", r.output);
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "one\nfour\nfive\nsix\n"
        );
        // the line endings stay as they were
        text("a\r\nb\r\nc\r\n");
        assert!(!call(json!({"path": path, "start": 2, "newText": "B1\nB2"})).is_error);
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "a\r\nB1\r\nB2\r\nc\r\n"
        );
        // no trailing newline stays none
        text("x\ny");
        assert!(!call(json!({"path": path, "start": 2, "newText": "Y"})).is_error);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "x\nY");
        let _ = std::fs::remove_file(&f);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn a_read_from_a_line_returns_120_lines_from_there_and_recall_100() {
        let dir = std::env::temp_dir().join(format!("shadow_defaults_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("long.txt");
        std::fs::write(
            &f,
            (1..=900)
                .map(|i| format!("line {i} of nine hundred with some padding words\n"))
                .collect::<String>(),
        )
        .unwrap();
        let r = execute_tool(&ToolInvocation {
            name: "read_file".into(),
            arguments: json!({"path": f.to_str().unwrap(), "start": 500}),
        });
        assert!(
            r.output.contains("  500| line 500 ")
                && r.output.contains("  619| line 619 ")
                && !r.output.contains("  620| line 620"),
            "{}",
            &r.output[..300.min(r.output.len())]
        );
        assert!(
            r.output.contains("ask again with start=620"),
            "{}",
            r.output.lines().last().unwrap_or("")
        );
        let first = execute_tool(&ToolInvocation {
            name: "read_file".into(),
            arguments: json!({"path": f.to_str().unwrap()}),
        });
        let id: u64 = first
            .output
            .strip_prefix("(output #")
            .and_then(|t| t.split(')').next())
            .and_then(|n| n.parse().ok())
            .unwrap();
        let rc = execute_tool(&ToolInvocation {
            name: "recall".into(),
            arguments: json!({"id": id, "start": 10}),
        });
        assert!(
            rc.output.contains("lines 10-109 of")
                && rc.output.contains("more: ask again with start=110"),
            "{}",
            rc.output.lines().next().unwrap_or("")
        );
        let _ = std::fs::remove_file(&f);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn long_outputs_are_logged_and_recall_reads_them_again() {
        let dir = std::env::temp_dir().join(format!("shadow_recall_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("big.txt");
        let text: String = (1..=120)
            .map(|i| format!("line number {i} of the file with some padding text\n"))
            .collect();
        std::fs::write(&f, &text).unwrap();
        let r = execute_tool(&ToolInvocation {
            name: "read_file".into(),
            arguments: json!({"path": f.to_str().unwrap(), "start": 1, "end": 120}),
        });
        let id: u64 = r
            .output
            .strip_prefix("(output #")
            .and_then(|t| t.split(')').next())
            .and_then(|n| n.parse().ok())
            .expect("a long output names its number");
        // a short output is not logged
        let short = execute_tool(&ToolInvocation {
            name: "cmd".into(),
            arguments: json!({"command": "echo hi"}),
        });
        assert!(!short.output.starts_with("(output #"), "{}", short.output);
        // recall by number and line range
        let a = execute_tool(&ToolInvocation {
            name: "recall".into(),
            arguments: json!({"id": id, "start_line": 40, "end_line": 42}),
        });
        assert!(
            a.output.contains("lines 40-42 of")
                && a.output.contains("line number 40 of")
                && a.output.contains("line number 42 of")
                && !a.output.contains("line number 43 "),
            "{}",
            a.output
        );
        assert!(!a.output.contains("ignored unknown"), "{}", a.output);
        // only the lines with a text, in one output and across all of them
        let b = execute_tool(&ToolInvocation {
            name: "recall".into(),
            arguments: json!({"id": id, "query": "NUMBER 77 "}),
        });
        assert!(
            b.output.contains("line number 77 of") && !b.output.contains("line number 78"),
            "{}",
            b.output
        );
        let c = execute_tool(&ToolInvocation {
            name: "recall".into(),
            arguments: json!({"query": "number 99 of"}),
        });
        assert!(c.output.contains(&format!("#{id}:99|")), "{}", c.output);
        // an output that is not there, and a call with nothing to go on
        let d = execute_tool(&ToolInvocation {
            name: "recall".into(),
            arguments: json!({"id": 999_999}),
        });
        assert!(
            d.is_error && d.output.contains("no output #999999"),
            "{}",
            d.output
        );
        assert!(
            execute_tool(&ToolInvocation {
                name: "recall".into(),
                arguments: json!({})
            })
            .is_error
        );
        let _ = std::fs::remove_file(&f);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn the_names_a_model_uses_for_line_ranges_are_taken_and_unknown_ones_are_reported() {
        // start_line / end_line / file_path are ours under other names; our own name wins when both are given
        let a = coerce_arguments(
            "read_file",
            &json!({"file_path": "a.rs", "start_line": 2415, "end_line": "2530"}),
        );
        assert_eq!(
            (a["path"].clone(), a["start"].clone(), a["end"].clone()),
            (json!("a.rs"), json!(2415), json!(2530))
        );
        assert!(a.get("start_line").is_none() && a.get("file_path").is_none());
        let b = coerce_arguments(
            "read_file",
            &json!({"path": "a.rs", "start": 3, "start_line": 9}),
        );
        assert_eq!(b["start"], json!(3));
        // a read of the 3rd..4th line of a real file starts at the 3rd
        let dir = std::env::temp_dir().join(format!("shadow_alias_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.txt");
        std::fs::write(
            &f,
            "one
two
three
four
five
",
        )
        .unwrap();
        let r = execute_tool(&ToolInvocation {
            name: "read_file".into(),
            arguments: json!({"file_path": f.to_str().unwrap(), "start_line": 3, "end_line": 4}),
        });
        assert!(
            r.output.contains("    3| three")
                && r.output.contains("    4| four")
                && !r.output.contains("| one"),
            "{}",
            r.output
        );
        assert!(!r.output.contains("ignored unknown"), "{}", r.output);
        // an argument that means nothing to the tool is reported in the result
        let r = execute_tool(&ToolInvocation {
            name: "read_file".into(),
            arguments: json!({"path": f.to_str().unwrap(), "lines": "3-4"}),
        });
        assert!(
            r.output.contains("ignored unknown argument(s) lines") && r.output.contains("path"),
            "{}",
            r.output
        );
        let _ = std::fs::remove_file(&f);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn glob_finds_files_by_real_patterns_and_grep_takes_regexes() {
        // a small tree: <tmp>/crates/a/src/engine.rs, <tmp>/crates/a/Cargo.toml, <tmp>/crates/b/src/lib.rs, <tmp>/Cargo.toml, <tmp>/README.md
        let root = std::env::temp_dir().join(format!("shadow-tools-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (rel, body) in [
            (
                "crates/a/src/engine.rs",
                "pub busy: bool,\nstruct Status {\nfn push(x: u32)",
            ),
            ("crates/a/Cargo.toml", "[package]"),
            ("crates/b/src/lib.rs", "let state = 1;\nfoo(bar)"),
            ("Cargo.toml", "[workspace]"),
            ("README.md", "hello"),
        ] {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, body).unwrap();
        }
        let r = root.to_string_lossy().to_string();
        let names = |pattern: &str| -> Vec<String> {
            let res = execute_tool(&ToolInvocation {
                name: "glob".into(),
                arguments: json!({"pattern": pattern, "path": r}),
            });
            let mut v: Vec<String> = res
                .output
                .lines()
                .filter(|l| !l.starts_with("no matching"))
                .map(|l| {
                    l.replace('\\', "/")
                        .rsplit(&format!("{}/", r.replace('\\', "/")))
                        .next()
                        .unwrap()
                        .to_string()
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            names("*.rs"),
            ["crates/a/src/engine.rs", "crates/b/src/lib.rs"]
        );
        assert_eq!(
            names("**/*.rs"),
            ["crates/a/src/engine.rs", "crates/b/src/lib.rs"]
        );
        assert_eq!(names("*.toml"), ["Cargo.toml", "crates/a/Cargo.toml"]);
        assert_eq!(names("crates/**/Cargo.toml"), ["crates/a/Cargo.toml"]);
        assert_eq!(
            names("crates/*/src/*.rs"),
            ["crates/a/src/engine.rs", "crates/b/src/lib.rs"]
        );
        assert_eq!(names("src/*.rs"), Vec::<String>::new()); // a pattern with a folder is anchored at the root
        assert_eq!(names("eng?ne.rs"), ["crates/a/src/engine.rs"]);
        assert_eq!(names("readme"), ["README.md"]); // no wildcard: a substring of the name, as before
        assert_eq!(names("*.nothing"), Vec::<String>::new());
        // grep: a literal, an alternation, a regex, and text that is a literal but not a valid regex
        let hits = |pattern: &str| -> usize {
            let res = execute_tool(&ToolInvocation {
                name: "grep".into(),
                arguments: json!({"pattern": pattern, "path": r}),
            });
            if res.output.starts_with("no occurrences") {
                0
            } else {
                res.output.lines().count()
            }
        };
        assert_eq!(hits("busy"), 1);
        assert_eq!(hits("busy|state"), 2); // the model writes regexes; this used to find nothing
        assert_eq!(hits("struct\\s+Status"), 1);
        assert_eq!(hits("struct Status {"), 1); // not a valid regex: taken literally
        assert_eq!(hits("fn push("), 1);
        assert_eq!(hits("foo(bar)"), 1); // a regex group, but the literal text is in the file
        assert_eq!(hits("not-in-any-file"), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn executes_read_and_cmd() {
        let inv_cmd = ToolInvocation {
            name: "cmd".into(),
            arguments: json!({"command": "Write-Output 'shadow-test-ok'"}),
        };
        let res_cmd = execute_tool(&inv_cmd);
        assert!(!res_cmd.is_error);
        assert!(res_cmd.output.contains("shadow-test-ok"));

        let inv_read = ToolInvocation {
            name: "read_file".into(),
            arguments: json!({"path": "Cargo.toml", "start": 1, "end": 5}),
        };
        let res_read = execute_tool(&inv_read);
        assert!(!res_read.is_error);
        assert!(res_read.output.contains("workspace"));
    }
}
