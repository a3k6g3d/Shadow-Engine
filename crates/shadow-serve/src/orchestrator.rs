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
}

impl ToolResult {
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "output": self.output,
            "is_error": self.is_error,
        })
    }
}

const MAX_OUTPUT_CHARS: usize = 40_000;
const MAX_FILE_SIZE: u64 = 10_000_000;

fn tail_clip(s: String) -> String {
    if s.len() <= MAX_OUTPUT_CHARS {
        s
    } else {
        format!("{}\n... [Output truncated at {} characters]", &s[..MAX_OUTPUT_CHARS], MAX_OUTPUT_CHARS)
    }
}

pub fn execute_tool(inv: &ToolInvocation) -> ToolResult {
    match inv.name.as_str() {
        "web_search" => {
            let q = inv.arguments.get("query").and_then(Value::as_str).unwrap_or("");
            let n = inv.arguments.get("max_results").and_then(Value::as_u64).unwrap_or(8) as usize;
            match crate::webtools::web_search(q, n) {
                Ok(s) => ToolResult { name: inv.name.clone(), output: tail_clip(s), is_error: false },
                Err(e) => ToolResult { name: inv.name.clone(), output: format!("error: {e}"), is_error: true },
            }
        }
        "web_fetch" => {
            let url = inv.arguments.get("url").and_then(Value::as_str).unwrap_or("");
            let start = inv.arguments.get("start").and_then(Value::as_u64).unwrap_or(0) as usize;
            let max = inv.arguments.get("max_chars").and_then(Value::as_u64).unwrap_or(12_000) as usize;
            match crate::webtools::web_fetch(url, start, max) {
                Ok(s) => ToolResult { name: inv.name.clone(), output: tail_clip(s), is_error: false },
                Err(e) => ToolResult { name: inv.name.clone(), output: format!("error: {e}"), is_error: true },
            }
        }
        "read_file" => {
            let path_str = inv.arguments.get("path").and_then(Value::as_str).unwrap_or("");
            let start = inv.arguments.get("start").and_then(Value::as_u64).unwrap_or(1) as usize;
            let end = inv.arguments.get("end").and_then(Value::as_u64).unwrap_or(400) as usize;

            if path_str.is_empty() {
                return ToolResult { name: inv.name.clone(), output: "error: missing required argument 'path'".into(), is_error: true };
            }
            let p = Path::new(path_str);
            if !p.exists() {
                return ToolResult { name: inv.name.clone(), output: format!("error: file does not exist: {path_str}"), is_error: true };
            }
            if let Ok(meta) = p.metadata() {
                if meta.len() > MAX_FILE_SIZE {
                    return ToolResult { name: inv.name.clone(), output: "error: file exceeds maximum allowed size (10 MB)".into(), is_error: true };
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
                    ToolResult { name: inv.name.clone(), output: tail_clip(out), is_error: false }
                }
                Err(e) => ToolResult { name: inv.name.clone(), output: format!("error reading file: {e}"), is_error: true },
            }
        }

        "glob" => {
            let pattern = inv.arguments.get("pattern").and_then(Value::as_str).unwrap_or("*");
            let root = inv.arguments.get("path").and_then(Value::as_str).unwrap_or(".");
            let max_results = inv.arguments.get("max_results").and_then(Value::as_u64).unwrap_or(100) as usize;

            let mut matches = Vec::new();
            let mut stack = vec![PathBuf::from(root)];
            let search_term = pattern.replace("**", "").replace("*", "").to_lowercase();

            while let Some(dir) = stack.pop() {
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if fname.starts_with('.') || fname == "target" || fname == "node_modules" || fname == ".git" {
                            continue;
                        }
                        if path.is_dir() {
                            stack.push(path);
                        } else if search_term.is_empty() || fname.to_lowercase().contains(&search_term) {
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
                ToolResult { name: inv.name.clone(), output: "no matching files found".into(), is_error: false }
            } else {
                ToolResult { name: inv.name.clone(), output: tail_clip(matches.join("\n")), is_error: false }
            }
        }

        "grep" => {
            let pattern = inv.arguments.get("pattern").and_then(Value::as_str).unwrap_or("");
            let root = inv.arguments.get("path").and_then(Value::as_str).unwrap_or(".");
            let ignore_case = inv.arguments.get("ignore_case").and_then(Value::as_bool).unwrap_or(true);
            let max_results = inv.arguments.get("max_results").and_then(Value::as_u64).unwrap_or(50) as usize;

            if pattern.is_empty() {
                return ToolResult { name: inv.name.clone(), output: "error: missing required argument 'pattern'".into(), is_error: true };
            }

            let needle = if ignore_case { pattern.to_lowercase() } else { pattern.to_string() };
            let mut matches = Vec::new();
            let mut stack = vec![PathBuf::from(root)];

            while let Some(dir) = stack.pop() {
                if let Ok(entries) = std::fs::read_dir(&dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if fname.starts_with('.') || fname == "target" || fname == "node_modules" || fname == ".git" {
                            continue;
                        }
                        if path.is_dir() {
                            stack.push(path);
                        } else if let Ok(meta) = path.metadata() {
                            if meta.len() < MAX_FILE_SIZE {
                                if let Ok(content) = std::fs::read_to_string(&path) {
                                    for (line_idx, line) in content.lines().enumerate() {
                                        let matched = if ignore_case {
                                            line.to_lowercase().contains(&needle)
                                        } else {
                                            line.contains(&needle)
                                        };
                                        if matched {
                                            matches.push(format!("{}:{}: {}", path.display(), line_idx + 1, line.trim()));
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
                ToolResult { name: inv.name.clone(), output: format!("no occurrences found for '{pattern}'"), is_error: false }
            } else {
                ToolResult { name: inv.name.clone(), output: tail_clip(matches.join("\n")), is_error: false }
            }
        }

        "edit_file" => {
            let path_str = inv.arguments.get("path").and_then(Value::as_str).unwrap_or("");
            let old_text = inv.arguments.get("oldText").or_else(|| inv.arguments.get("old_text")).and_then(Value::as_str).unwrap_or("");
            let new_text = inv.arguments.get("newText").or_else(|| inv.arguments.get("new_text")).and_then(Value::as_str).unwrap_or("");

            if path_str.is_empty() || old_text.is_empty() {
                return ToolResult { name: inv.name.clone(), output: "error: missing required arguments 'path' or 'oldText'".into(), is_error: true };
            }
            let p = Path::new(path_str);
            match std::fs::read_to_string(p) {
                Ok(content) => {
                    let count = content.matches(old_text).count();
                    if count == 0 {
                        ToolResult { name: inv.name.clone(), output: format!("error: 'oldText' was not found in {path_str}"), is_error: true }
                    } else if count > 1 {
                        ToolResult { name: inv.name.clone(), output: format!("error: 'oldText' matches {count} times in {path_str}; provide more surrounding context to disambiguate"), is_error: true }
                    } else {
                        let updated = content.replace(old_text, new_text);
                        if let Err(e) = std::fs::write(p, updated) {
                            ToolResult { name: inv.name.clone(), output: format!("error writing edits to {path_str}: {e}"), is_error: true }
                        } else {
                            ToolResult { name: inv.name.clone(), output: format!("successfully edited {path_str}"), is_error: false }
                        }
                    }
                }
                Err(e) => ToolResult { name: inv.name.clone(), output: format!("error reading {path_str}: {e}"), is_error: true },
            }
        }

        "write_file" => {
            let path_str = inv.arguments.get("path").and_then(Value::as_str).unwrap_or("");
            let content = inv.arguments.get("content").and_then(Value::as_str).unwrap_or("");

            if path_str.is_empty() {
                return ToolResult { name: inv.name.clone(), output: "error: missing required argument 'path'".into(), is_error: true };
            }
            let p = Path::new(path_str);
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(p, content) {
                Ok(_) => ToolResult { name: inv.name.clone(), output: format!("successfully wrote {} bytes to {path_str}", content.len()), is_error: false },
                Err(e) => ToolResult { name: inv.name.clone(), output: format!("error writing to {path_str}: {e}"), is_error: true },
            }
        }

        "cmd" | "bash" | "powershell" => {
            let command = inv.arguments.get("command").or_else(|| inv.arguments.get("cmd")).and_then(Value::as_str).unwrap_or("");
            let cwd = inv.arguments.get("cwd").and_then(Value::as_str).unwrap_or(".");

            if command.is_empty() {
                return ToolResult { name: inv.name.clone(), output: "error: missing required argument 'command'".into(), is_error: true };
            }

            // Execute natively via Windows PowerShell
            let mut cmd = Command::new("powershell");
            cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]);
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
                    }
                }
                Err(e) => ToolResult {
                    name: inv.name.clone(),
                    output: format!("error launching PowerShell command on Windows: {e}"),
                    is_error: true,
                },
            }
        }

        other => ToolResult {
            name: other.to_string(),
            output: format!("unknown tool '{other}'"),
            is_error: true,
        },
    }
}

pub fn tool_definitions() -> Value {
    json!([
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
            "description": "Read file contents with line numbers on Windows.",
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
            "name": "glob",
            "description": "Find files matching pattern under a folder on Windows.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "File search pattern e.g. *.rs or *.json" },
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
