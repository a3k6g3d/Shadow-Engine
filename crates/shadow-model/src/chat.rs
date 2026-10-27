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
    /// Tool definitions already formatted as `tool | tojson` (for Qwen, or Flint when `tools_fn` is empty).
    pub tools: Vec<String>,
    /// Tool definitions unwrapped as `(tool.function if defined else tool) | tojson` for Flint.
    pub tools_fn: Vec<String>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            thinking: None,
            effort: Effort::XHigh,
            add_generation_prompt: true,
            preserve_thinking: None,
            tools: Vec::new(),
            tools_fn: Vec::new(),
        }
    }
}

const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, \
consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
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

fn flint_text(t: &str) -> String {
    t.replace('\0', "")
        .replace("<think>", "")
        .replace("</think>", "")
        .replace("<|begin_of_text|>", "&lt;|begin_of_text|&gt;")
        .replace("<|end_of_text|>", "&lt;|end_of_text|&gt;")
        .replace("<|system|>", "&lt;|system|&gt;")
        .replace("<|developer|>", "&lt;|developer|&gt;")
        .replace("<|user|>", "&lt;|user|&gt;")
        .replace("<|assistant|>", "&lt;|assistant|&gt;")
        .replace("<|observation|>", "&lt;|observation|&gt;")
}

/// Flint's chat template (`RecursiveSI/flint` `chat_template.jinja`).
pub fn render_flint(messages: &[Message], o: &Options) -> Result<String, String> {
    if messages.is_empty() {
        return Err("No messages provided.".into());
    }
    let tools_list = if !o.tools_fn.is_empty() { &o.tools_fn } else { &o.tools };
    let tool_preamble = if !tools_list.is_empty() {
        let mut p = String::from("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
            You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for t in tools_list {
            p += t;
        }
        p += "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n\
            <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n";
        p
    } else {
        String::new()
    };
    let mut tools_pending = true;
    let mut out = String::new();
    for m in messages {
        if tools_pending && m.role != Role::System {
            out += &tool_preamble;
            tools_pending = false;
        }
        match m.role {
            Role::System => {
                out += "<|system|>";
                out += &flint_text(&m.content);
            }
            Role::User => {
                out += "<|user|>";
                out += &flint_text(&m.content);
            }
            Role::Assistant => {
                out += "<|assistant|>";
                if let Some(r) = &m.reasoning {
                    if !r.is_empty() {
                        out += "<think>";
                        out += r;
                        out += "</think>";
                    }
                }
                out += &m.content;
                for tc in &m.tool_calls {
                    out += "<tool_call>";
                    out += &tc.name;
                    for (k, v) in &tc.arguments {
                        out += &format!("<arg_key>{k}</arg_key><arg_value>{v}</arg_value>");
                    }
                    out += "</tool_call>";
                }
            }
            Role::Tool => {
                out += "<|observation|><tool_response>";
                out += &flint_text(&m.content);
                out += "</tool_response>";
            }
        }
    }
    if tools_pending {
        out += &tool_preamble;
    }
    if o.add_generation_prompt {
        out += "<|assistant|>";
    }
    Ok(out)
}

pub fn render_for_arch(arch: &str, messages: &[Message], o: &Options) -> Result<String, String> {
    match arch {
        "flint" => render_flint(messages, o),
        _ => render(messages, o),
    }
}

/// Extracts `<tool_call>...</tool_call>` blocks from an assistant message's output, returning `(remaining_content, tool_calls)`.
/// Handles both Qwen's `<function=NAME>\n<parameter=KEY>\nVAL\n</parameter>\n</function>` and Flint's
/// `NAME<arg_key>KEY</arg_key><arg_value>VAL</arg_value>`.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let Some(first_pos) = text.find("<tool_call>") else {
        return (text.to_string(), Vec::new());
    };
    let mut calls = Vec::new();
    let mut rest = &text[first_pos..];
    while let Some(s) = rest.find("<tool_call>") {
        let after_open = &rest[s + "<tool_call>".len()..];
        let Some(e) = after_open.find("</tool_call>") else { break };
        let body = after_open[..e].trim();
        if let Some(fn_body) = body.strip_prefix("<function=") {
            if let Some(gt) = fn_body.find('>') {
                let name = fn_body[..gt].trim().to_string();
                let mut inner = &fn_body[gt + 1..];
                if let Some(end_fn) = inner.rfind("</function>") {
                    inner = &inner[..end_fn];
                }
                let mut arguments = Vec::new();
                while let Some(ps) = inner.find("<parameter=") {
                    let after_p = &inner[ps + "<parameter=".len()..];
                    let Some(pgt) = after_p.find('>') else { break };
                    let key = after_p[..pgt].trim().to_string();
                    let val_rest = &after_p[pgt + 1..];
                    let Some(pe) = val_rest.find("</parameter>") else { break };
                    let mut val = &val_rest[..pe];
                    val = val.strip_prefix("\r\n").or_else(|| val.strip_prefix('\n')).unwrap_or(val);
                    val = val.strip_suffix("\r\n").or_else(|| val.strip_suffix('\n')).unwrap_or(val);
                    arguments.push((key, val.to_string()));
                    inner = &val_rest[pe + "</parameter>".len()..];
                }
                if !name.is_empty() {
                    calls.push(ToolCall { name, arguments });
                }
            }
        } else {
            let name_end = body.find("<arg_key>").unwrap_or(body.len());
            let name = body[..name_end].trim().to_string();
            let mut inner = &body[name_end..];
            let mut arguments = Vec::new();
            while let Some(ks) = inner.find("<arg_key>") {
                let after_k = &inner[ks + "<arg_key>".len()..];
                let Some(ke) = after_k.find("</arg_key>") else { break };
                let key = after_k[..ke].trim().to_string();
                let after_ke = &after_k[ke + "</arg_key>".len()..];
                let Some(vs) = after_ke.find("<arg_value>") else { break };
                let after_v = &after_ke[vs + "<arg_value>".len()..];
                let Some(ve) = after_v.find("</arg_value>") else { break };
                let val = after_v[..ve].to_string();
                arguments.push((key, val));
                inner = &after_v[ve + "</arg_value>".len()..];
            }
            if !name.is_empty() {
                calls.push(ToolCall { name, arguments });
            }
        }
        rest = &after_open[e + "</tool_call>".len()..];
    }
    if calls.is_empty() {
        (text.to_string(), Vec::new())
    } else {
        (text[..first_pos].trim_end().to_string(), calls)
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
}
