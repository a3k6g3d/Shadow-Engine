//! The model's chat template, hard-coded: the parts of the GGUF's `tokenizer.chat_template` (8,952 bytes of Jinja)
//! that a text or tool-calling conversation uses, transcribed branch by branch, so no Jinja engine is needed.  What is
//! not transcribed is refused by name rather than rendered wrongly: image or video content.
//!
//! The template's behaviour worth knowing, because it is not ChatML as usually written:
//! - With thinking on (the default) and the default effort `xhigh`, a system block is ALWAYS emitted, carrying the
//!   effort instruction, even when the conversation has no system message. `medium` emits none; `low` its own text.
//! - Every assistant turn is rendered with a `<think>` block (empty if there is no reasoning), because
//!   `preserve_thinking` defaults to on.
//! - The generation prompt opens `<think>\n` (thinking on) or `<think>\n\n</think>\n\n` (thinking off).
//! - Contents are trimmed of surrounding whitespace.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// Ordered `(parameter_name, rendered_value)` pairs: raw string for string values, `tojson` for non-strings.
    pub arguments: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// An assistant message's reasoning, rendered inside `<think>`.
    pub reasoning: Option<String>,
    /// Assistant tool calls (`<tool_call>...</tool_call>`).
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn new(role: Role, content: &str) -> Message {
        Message { role, content: content.to_string(), reasoning: None, tool_calls: Vec::new() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    XHigh,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct Options {
    /// None = the template's `enable_thinking is undefined`, which behaves as on.
    pub thinking: Option<bool>,
    pub effort: Effort,
    pub add_generation_prompt: bool,
    /// None = undefined, which behaves as on.
    pub preserve_thinking: Option<bool>,
    /// Tool definitions already formatted as `tool | tojson`.
    pub tools: Vec<String>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            thinking: None,
            effort: Effort::XHigh,
            add_generation_prompt: true,
            preserve_thinking: None,
            tools: Vec::new(),
        }
    }
}

const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, \
consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const HIGH: &str = "Reasoning effort is set to high. Think thoroughly through the problem, explore alternative solutions, \
and verify your reasoning before concluding.";
const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion \
without unnecessary elaboration.";
const QWEN_TOOLS_SUFFIX: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>";

/// Python's `str.strip()` / Jinja's `trim`: Unicode whitespace from both ends.
fn trim(s: &str) -> &str {
    s.trim()
}

pub fn render(messages: &[Message], o: &Options) -> Result<String, String> {
    if messages.is_empty() {
        return Err("No messages provided.".into());
    }
    let thinking_on = o.thinking != Some(false);
    let instructions = if thinking_on {
        match o.effort {
            Effort::XHigh => XHIGH,
            Effort::High => HIGH,
            Effort::Low => LOW,
            Effort::Medium => "",
        }
    } else {
        ""
    };
    let mut out = String::new();
    if !o.tools.is_empty() {
        out += "<|im_start|>system\n";
        if !instructions.is_empty() {
            out += instructions;
            out += "\n\n";
        }
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for tool in &o.tools {
            out += "\n";
            out += tool;
        }
        out += QWEN_TOOLS_SUFFIX;
        if messages[0].role == Role::System {
            let content = trim(&messages[0].content);
            if !content.is_empty() {
                out += "\n\n";
                out += content;
            }
        }
        out += "<|im_end|>\n";
    } else if messages[0].role == Role::System {
        let content = trim(&messages[0].content);
        if !content.is_empty() {
            out += "<|im_start|>system\n";
            if !instructions.is_empty() {
                out += instructions;
                out += "\n\n";
            }
            out += content;
            out += "<|im_end|>\n";
        } else if !instructions.is_empty() {
            out += &format!("<|im_start|>system\n{instructions}<|im_end|>\n");
        }
    } else if !instructions.is_empty() {
        out += &format!("<|im_start|>system\n{instructions}<|im_end|>\n");
    }

    // The last real user query: walking back, the first user message that is not a wrapped tool response.
    let mut last_query = None;
    for (i, m) in messages.iter().enumerate().rev() {
        if m.role == Role::User {
            let c = trim(&m.content);
            if !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")) {
                last_query = Some(i);
                break;
            }
        }
    }
    let last_query = last_query.ok_or("No user query found in messages.")?;

    for (i, m) in messages.iter().enumerate() {
        let content = trim(&m.content);
        match m.role {
            Role::System => {
                if i != 0 {
                    return Err("System message must be at the beginning.".into());
                }
            }
            Role::User => out += &format!("<|im_start|>user\n{content}<|im_end|>\n"),
            Role::Assistant => {
                let reasoning = trim(m.reasoning.as_deref().unwrap_or(""));
                if o.preserve_thinking != Some(false) || i > last_query {
                    out += &format!("<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}");
                } else {
                    out += &format!("<|im_start|>assistant\n{content}");
                }
                for (tc_idx, tc) in m.tool_calls.iter().enumerate() {
                    if tc_idx == 0 {
                        if !content.is_empty() {
                            out += &format!("\n\n<tool_call>\n<function={}>\n", tc.name);
                        } else {
                            out += &format!("<tool_call>\n<function={}>\n", tc.name);
                        }
                    } else {
                        out += &format!("\n<tool_call>\n<function={}>\n", tc.name);
                    }
                    for (k, v) in &tc.arguments {
                        out += &format!("<parameter={k}>\n{v}\n</parameter>\n");
                    }
                    out += "</function>\n</tool_call>";
                }
                out += "<|im_end|>\n";
            }
            Role::Tool => {
                let prev_is_tool = i > 0 && messages[i - 1].role == Role::Tool;
                if i > 0 && !prev_is_tool {
                    out += "<|im_start|>user";
                }
                out += &format!("\n<tool_response>\n{content}\n</tool_response>");
                let last = i + 1 == messages.len();
                if last || messages[i + 1].role != Role::Tool {
                    out += "<|im_end|>\n";
                }
            }
        }
    }
    if o.add_generation_prompt {
        out += "<|im_start|>assistant\n";
        out += if o.thinking == Some(false) { "<think>\n\n</think>\n\n" } else { "<think>\n" };
    }
    Ok(out)
}

/// The chat template for an architecture.  Only the Qwen3.5-family template is implemented.
pub fn render_for_arch(_arch: &str, messages: &[Message], o: &Options) -> Result<String, String> {
    render(messages, o)
}

/// Finds the next genuine `<tool_call>\s*<function=NAME>` opener at or after `from`,
/// returning `(open_start, fn_name, body_start)`. Prose mentions like `` `<tool_call>` `` are skipped.
fn find_tool_call_opener(text: &str, from: usize) -> Option<(usize, String, usize)> {
    const OPEN: &str = "<tool_call>";
    const FN: &str = "<function=";
    let mut pos = from;
    while let Some(rel) = text[pos..].find(OPEN) {
        let open_start = pos + rel;
        let after_open = open_start + OPEN.len();
        let ws_len = text[after_open..]
            .bytes()
            .take_while(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
            .count();
        let fn_start = after_open + ws_len;
        if text[fn_start..].starts_with(FN) {
            let name_start = fn_start + FN.len();
            if let Some(gt_rel) = text[name_start..].find('>') {
                let name = text[name_start..name_start + gt_rel].trim();
                if !name.is_empty()
                    && !name.contains('<')
                    && !name.chars().any(char::is_whitespace)
                {
                    return Some((open_start, name.to_string(), name_start + gt_rel + 1));
                }
            }
        }
        pos = after_open;
    }
    None
}

/// Finds the matching `</function>\s*</tool_call>` close for a tool call body starting at `body_start`,
/// returning `(body_end, close_end)`. Skips literal `</tool_call>` or `</function>` inside code strings.
fn find_tool_call_close(text: &str, body_start: usize) -> Option<(usize, usize)> {
    const END_FN: &str = "</function>";
    const END_TC: &str = "</tool_call>";
    let mut pos = body_start;
    let mut fallback: Option<(usize, usize)> = None;
    while let Some(rel) = text[pos..].find(END_FN) {
        let end_fn = pos + rel;
        let after_fn = end_fn + END_FN.len();
        let ws_len = text[after_fn..]
            .bytes()
            .take_while(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
            .count();
        let tc_start = after_fn + ws_len;
        if text[tc_start..].starts_with(END_TC) {
            let close_end = tc_start + END_TC.len();
            let before = text[body_start..end_fn].trim_end();
            if before.is_empty() || before.ends_with("</parameter>") {
                return Some((end_fn, close_end));
            }
            if fallback.is_none() {
                fallback = Some((end_fn, close_end));
            }
        }
        pos = after_fn;
    }
    fallback
}

fn parse_tool_params(inner: &str) -> Vec<(String, String)> {
    const P_OPEN: &str = "<parameter=";
    const P_CLOSE: &str = "</parameter>";
    let mut arguments = Vec::new();
    let mut pos = 0;
    while let Some(rel) = inner[pos..].find(P_OPEN) {
        let after_p = pos + rel + P_OPEN.len();
        let Some(pgt) = inner[after_p..].find('>') else { break };
        let key = inner[after_p..after_p + pgt].trim().to_string();
        let val_start = after_p + pgt + 1;
        // Find the `</parameter>` followed by whitespace + next `<parameter=` or end of function body.
        let mut search = val_start;
        let mut matched_end: Option<usize> = None;
        while let Some(pe_rel) = inner[search..].find(P_CLOSE) {
            let pe = search + pe_rel;
            let after_close = inner[pe + P_CLOSE.len()..].trim_start();
            if after_close.is_empty() || after_close.starts_with(P_OPEN) {
                matched_end = Some(pe);
                break;
            }
            search = pe + P_CLOSE.len();
        }
        let Some(pe) = matched_end.or_else(|| inner[val_start..].find(P_CLOSE).map(|r| val_start + r)) else {
            break;
        };
        let mut val = &inner[val_start..pe];
        val = val.strip_prefix("\r\n").or_else(|| val.strip_prefix('\n')).unwrap_or(val);
        val = val.strip_suffix("\r\n").or_else(|| val.strip_suffix('\n')).unwrap_or(val);
        arguments.push((key, val.to_string()));
        pos = pe + P_CLOSE.len();
    }
    arguments
}

/// Extracts `<tool_call>...</tool_call>` blocks from an assistant message's output, returning `(remaining_content, tool_calls)`.
/// Understands Qwen's `<function=NAME>\n<parameter=KEY>\nVAL\n</parameter>\n</function>` form,
/// ignoring prose mentions of `<tool_call>` and literal `</tool_call>` tags inside file contents.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut first_pos: Option<usize> = None;
    let mut cursor = 0;
    while let Some((open_start, name, body_start)) = find_tool_call_opener(text, cursor) {
        let Some((body_end, close_end)) = find_tool_call_close(text, body_start) else {
            break;
        };
        if first_pos.is_none() {
            first_pos = Some(open_start);
        }
        let arguments = parse_tool_params(&text[body_start..body_end]);
        calls.push(ToolCall { name, arguments });
        cursor = close_end;
    }
    match first_pos {
        Some(fp) if !calls.is_empty() => (text[..fp].trim_end().to_string(), calls),
        _ => (text.to_string(), Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_adds_the_xhigh_system_block() {
        let s = render(&[Message::new(Role::User, "  hi  ")], &Options::default()).unwrap();
        assert_eq!(s, format!("<|im_start|>system\n{XHIGH}<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn thinking_off_with_a_system_prompt() {
        let o = Options { thinking: Some(false), ..Options::default() };
        let s = render(&[Message::new(Role::System, "Be brief."), Message::new(Role::User, "2+2?")], &o).unwrap();
        assert_eq!(s, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n\
                       <|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[test]
    fn medium_effort_and_history_with_reasoning() {
        let o = Options { effort: Effort::Medium, add_generation_prompt: false, ..Options::default() };
        let mut a = Message::new(Role::Assistant, "4");
        a.reasoning = Some("2 and 2".into());
        let s = render(&[Message::new(Role::User, "2+2?"), a], &o).unwrap();
        assert_eq!(s, "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n2 and 2\n</think>\n\n4<|im_end|>\n");
    }

    #[test]
    fn consecutive_tool_results_share_one_user_turn() {
        let o = Options { effort: Effort::Medium, add_generation_prompt: false, ..Options::default() };
        let m = [Message::new(Role::User, "go"), Message::new(Role::Assistant, "ok"), Message::new(Role::Tool, "r1"),
                 Message::new(Role::Tool, "r2")];
        let s = render(&m, &o).unwrap();
        assert!(s.ends_with("<|im_start|>user\n<tool_response>\nr1\n</tool_response>\n<tool_response>\nr2\n</tool_response><|im_end|>\n"), "{s}");
    }

    #[test]
    fn refusals() {
        assert!(render(&[], &Options::default()).is_err());
        assert!(render(&[Message::new(Role::Assistant, "x")], &Options::default()).is_err());
        let m = [Message::new(Role::User, "a"), Message::new(Role::System, "late")];
        assert!(render(&m, &Options::default()).is_err());
    }

    #[test]
    fn parse_tool_calls_handles_nested_tags_and_prose_mentions() {
        let txt = "Strip `<tool_call>` and `</tool_call>` blocks in MainActivity.kt.\n\
<tool_call>\n\
<function=write_file>\n\
<parameter=path>\nMainActivity.kt\n</parameter>\n\
<parameter=content>\nval re = Regex(\"<tool_call>[\\\\s\\\\S]*?</tool_call>\")\n</parameter>\n\
</function>\n\
</tool_call>";
        let (clean, calls) = parse_tool_calls(txt);
        assert_eq!(clean, "Strip `<tool_call>` and `</tool_call>` blocks in MainActivity.kt.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments[0], ("path".into(), "MainActivity.kt".into()));
        assert_eq!(
            calls[0].arguments[1],
            ("content".into(), "val re = Regex(\"<tool_call>[\\\\s\\\\S]*?</tool_call>\")".into())
        );
    }
}
