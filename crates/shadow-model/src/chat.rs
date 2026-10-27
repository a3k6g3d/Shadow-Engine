//! The model's chat template, hard-coded: the parts of the GGUF's `tokenizer.chat_template` (8,952 bytes of Jinja)
//! that a text conversation uses, transcribed branch by branch, so no Jinja engine is needed.  What is not transcribed
//! is refused by name rather than rendered wrongly: tool definitions (`tools`), tool calls inside an assistant
//! message, and image or video content.
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

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// An assistant message's reasoning, rendered inside `<think>`.
    pub reasoning: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: &str) -> Message {
        Message { role, content: content.to_string(), reasoning: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    XHigh,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// None = the template's `enable_thinking is undefined`, which behaves as on.
    pub thinking: Option<bool>,
    pub effort: Effort,
    pub add_generation_prompt: bool,
    /// None = undefined, which behaves as on.
    pub preserve_thinking: Option<bool>,
}

impl Default for Options {
    fn default() -> Options {
        Options { thinking: None, effort: Effort::XHigh, add_generation_prompt: true, preserve_thinking: None }
    }
}

const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, \
consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion \
without unnecessary elaboration.";

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
    if messages[0].role == Role::System {
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
