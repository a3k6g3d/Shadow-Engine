//! Shadow's tokenizer and chat template against the checkpoint's own (eval/fixtures/tokenizer.json and chat.json,
//! written by tools/make_tokenizer_fixture.py from the Hugging Face tokenizer.json and chat_template.jinja).  This is
//! the outside reference stage 0 asked for: exact ids, exact rendered text.  Needs the model file for its vocabulary;
//! without it every test says SKIPPED and passes.

use shadow_eval::json::{parse, Value as J};
use shadow_model::chat::{render, render_flint, Effort, Message, Options, Role};
use shadow_model::gguf::Gguf;
use shadow_model::tokenizer::Tokenizer;

const MODEL: &str =
    "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";
const FLINT_MODEL: &str = "I:/Flint-40B-A2.5B-abliterated-NVFP4.gguf";

fn fixture(name: &str) -> J {
    let p = format!("{}/../../eval/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    parse(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn tokenizer_at(p: &str) -> Option<Tokenizer> {
    if !std::path::Path::new(p).is_file() {
        eprintln!("SKIPPED, not passed: no model at {p}");
        return None;
    }
    Some(Tokenizer::from_gguf(&Gguf::open(std::path::Path::new(p)).unwrap()).unwrap())
}

fn check_tokenizer_fixture(model_path: &str, fixture_name: &str) {
    let Some(t) = tokenizer_at(model_path) else { return };
    let f = fixture(fixture_name);
    let cases = f.get("cases").unwrap().as_arr().unwrap();
    let mut bad = Vec::new();
    for c in cases {
        let text = c.get("text").unwrap().as_str().unwrap();
        let want: Vec<u32> = c.get("ids").unwrap().as_arr().unwrap().iter().map(|v| v.as_f64().unwrap() as u32).collect();
        let got = t.encode(text, true);
        if got != want {
            bad.push(format!("{text:?}\n   want {want:?}\n   got  {got:?}"));
        }
    }
    assert!(bad.is_empty(), "{} of {} texts differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
    eprintln!("{} texts tokenized exactly like {fixture_name}", cases.len());
}

#[test]
fn ids_match_the_checkpoints_tokenizer() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    let fix = if p.to_ascii_lowercase().contains("flint") { "tokenizer_flint.json" } else { "tokenizer.json" };
    check_tokenizer_fixture(&p, fix);
}

#[test]
fn flint_ids_match_the_checkpoints_tokenizer() {
    check_tokenizer_fixture(FLINT_MODEL, "tokenizer_flint.json");
}

fn check_chat_fixture(fixture_name: &str, f_render: fn(&[Message], &Options) -> Result<String, String>) {
    let f = fixture(fixture_name);
    let cases = f.get("cases").unwrap().as_arr().unwrap();
    let mut bad = Vec::new();
    for c in cases {
        let name = c.get("name").unwrap().as_str().unwrap();
        let msgs: Vec<Message> = c.get("messages").unwrap().as_arr().unwrap().iter().map(|m| {
            let role = match m.get("role").unwrap().as_str().unwrap() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => Role::Tool,
            };
            let mut msg = Message::new(role, m.get("content").unwrap().as_str().unwrap());
            msg.reasoning = m.get("reasoning_content").and_then(J::as_str).map(str::to_string);
            msg
        }).collect();
        let opts = c.get("options").unwrap();
        let mut o = Options::default();
        if let Some(J::Bool(b)) = opts.get("enable_thinking") { o.thinking = Some(*b); }
        if let Some(J::Bool(b)) = opts.get("preserve_thinking") { o.preserve_thinking = Some(*b); }
        o.effort = match opts.get("reasoning_effort").and_then(J::as_str) {
            Some("medium") => Effort::Medium,
            Some("low") => Effort::Low,
            _ => Effort::XHigh,
        };
        o.add_generation_prompt = !matches!(c.get("name").unwrap().as_str(), Some("no generation prompt"));
        let want = c.get("text").unwrap().as_str().unwrap();
        let got = f_render(&msgs, &o).unwrap();
        if got != want {
            bad.push(format!("{name}:\n--- want\n{want:?}\n--- got\n{got:?}"));
        }
    }
    assert!(bad.is_empty(), "{} of {} conversations differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
    eprintln!("{} conversations rendered exactly like {fixture_name}", cases.len());
}

#[test]
fn chat_template_matches_the_checkpoints() {
    check_chat_fixture("chat.json", render);
    check_chat_fixture("chat_flint.json", render_flint);
}
