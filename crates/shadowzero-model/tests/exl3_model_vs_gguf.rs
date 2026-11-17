//! An EXL3 checkpoint opened as a model (`exl3_model::open`) against the GGUF of the same model family: the same config, the
//! same tokenizer (every token, type and merge, the chat template), the same tensor names with the same dims.  Needs the
//! checkpoint directory (`EXL3_MODEL`, default the uncensored 3 bpw repo on J:) and a GGUF (`SHADOWZERO_MODEL`); skips otherwise.

use shadowzero_model::exl3_model;
use shadowzero_model::forward::Config;
use shadowzero_model::gguf::{Gguf, Value};
use std::path::Path;

const REPO: &str = "J:/Models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw";
const GGUF: &str = "J:/Models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";

fn setup() -> Option<(exl3_model::Exl3Model, Gguf)> {
    let repo = std::env::var("EXL3_MODEL").unwrap_or_else(|_| REPO.into());
    let model = std::env::var("SHADOWZERO_MODEL").unwrap_or_else(|_| GGUF.into());
    if !Path::new(&repo).join("config.json").is_file() || !Path::new(&model).is_file() {
        eprintln!("SKIPPED, not passed: no checkpoint at {repo} or no GGUF at {model}");
        return None;
    }
    let t = std::time::Instant::now();
    let m = exl3_model::open(Path::new(&repo)).unwrap();
    eprintln!("opened {repo}: {} tensors, {} expert matrices in {:.2} s", m.gguf.tensors.len(), m.extra.experts.len(), t.elapsed().as_secs_f64());
    Some((m, Gguf::open(Path::new(&model)).unwrap()))
}

#[test]
fn the_config_is_the_ggufs() {
    let Some((m, g)) = setup() else { return };
    let a = Config::from_gguf(&m.gguf).unwrap();
    let b = Config::from_gguf(&g).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn the_tokenizer_keys_are_the_ggufs() {
    let Some((m, g)) = setup() else { return };
    for k in ["tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.chat_template"] {
        assert_eq!(m.gguf.get(k), g.get(k), "{k}");
    }
    for k in ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"] {
        let (a, b) = (m.gguf.get(k).and_then(Value::as_array).unwrap(), g.get(k).and_then(Value::as_array).unwrap());
        assert_eq!(a.len(), b.len(), "{k}: length");
        let ints = |v: &Value| match v { Value::I32(x) => *x as i64, Value::U32(x) => *x as i64, _ => i64::MIN };
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let same = if k.ends_with("token_type") { ints(x) == ints(y) } else { x == y };
            assert!(same, "{k}[{i}]: {x:?} against {y:?}");
        }
    }
    // the same tokenizer behaviour
    let ta = shadowzero_model::tokenizer::Tokenizer::from_gguf(&m.gguf).unwrap();
    let tb = shadowzero_model::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    for text in ["The capital of France is", "Hello, world!\n\n  indented <|im_start|>user\nWhat is 2+2?<|im_end|>\n<think>", "naïve café 日本語 🙂 \t tabs"] {
        assert_eq!(ta.encode(text, true), tb.encode(text, true), "{text:?}");
    }
}

#[test]
fn every_tensor_name_and_shape_is_the_ggufs() {
    let Some((m, g)) = setup() else { return };
    let mut missing = vec![];
    let mut wrong = vec![];
    for t in &g.tensors {
        match m.gguf.tensor(&t.name) {
            None => missing.push(t.name.clone()),
            Some(o) if o.dims != t.dims => wrong.push(format!("{}: {:?} against the GGUF's {:?}", t.name, o.dims, t.dims)),
            _ => {}
        }
    }
    let extra: Vec<&str> = m.gguf.tensors.iter().filter(|t| g.tensor(&t.name).is_none()).map(|t| t.name.as_str()).collect();
    assert!(missing.is_empty(), "tensors of the GGUF the checkpoint does not provide: {missing:?}");
    assert!(wrong.is_empty(), "dims differ: {wrong:?}");
    assert!(extra.is_empty(), "tensors only the checkpoint view has: {extra:?}");
}
