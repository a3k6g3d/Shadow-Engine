//! Checks against the real model file, when it is on this machine (SHADOW_MODEL, else the path the plan names).
//! Without it every test here says "SKIPPED, not passed" and returns.
//!
//! There is no outside oracle for the tokenizer yet (docs/SHADOW_PLAN.md, "The oracle"), so these are the checks that
//! need none: every text round-trips byte for byte, the vocabulary encodes to itself, the special tokens are the ids
//! the file says, and the header numbers are the ones stage 0's gate names.

use shadow_model::gguf::Gguf;
use shadow_model::tokenizer::Tokenizer;
use std::path::PathBuf;

const DEFAULT: &str =
    "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";

fn model() -> Option<Gguf> {
    let p = std::env::var("SHADOW_MODEL").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT));
    if !p.is_file() {
        eprintln!("SKIPPED, not passed: no model at {} (set SHADOW_MODEL)", p.display());
        return None;
    }
    Some(Gguf::open(&p).expect("the model opens"))
}

#[test]
fn stage0_gate_every_tensor_accounted_for() {
    let Some(g) = model() else { return };
    assert_eq!(g.tensors.len(), 1224);
    let gib = g.tensor_bytes() as f64 / (1u64 << 30) as f64;
    assert!((gib - 77.80).abs() < 0.005, "{gib:.3} GiB");
    let file_total: u64 = g.shards.iter().map(|s| s.file_len).sum();
    let headers: u64 = g.shards.iter().map(|s| s.data_start).sum();
    assert_eq!(g.tensor_bytes() + headers, file_total, "tensors + headers must be the whole of both files");
}

#[test]
fn the_vocabulary_round_trips() {
    let Some(g) = model() else { return };
    let t = Tokenizer::from_gguf(&g).unwrap();
    assert_eq!(t.tokens.len(), 248_320);
    assert_eq!(t.id("<|im_start|>"), Some(248_045));
    assert_eq!(t.id("<|im_end|>"), Some(248_046));
    assert_eq!(t.eos, Some(248_046));
    assert_eq!(t.encode("<|im_start|>user\nhi<|im_end|>", true)[0], 248_045);
    assert!(!t.encode("<|im_start|>", false).contains(&248_045), "without `special` it is plain text");

    // Every ordinary token, decoded to its bytes and encoded again, gives one token back.  Not always itself: BPE may
    // reach the same bytes through other merges, so the check is "one token, same bytes".
    let (mut ok, mut multi, mut checked) = (0, 0, 0);
    for id in 0..t.tokens.len() as u32 {
        if t.token_type[id as usize] != 1 {
            continue;
        }
        let bytes = t.decode_bytes(&[id]);
        let Ok(text) = String::from_utf8(bytes.clone()) else { continue };
        checked += 1;
        let back = t.encode(&text, false);
        assert_eq!(t.decode_bytes(&back), bytes, "token {id} does not round-trip");
        if back.len() == 1 { ok += 1 } else { multi += 1 }
    }
    eprintln!("{checked} whole-UTF-8 tokens: {ok} encode to one token, {multi} to several (the pre-tokenizer cuts them)");
    assert!(ok as f64 / checked as f64 > 0.95, "{ok} of {checked}");
}

#[test]
fn real_text_round_trips() {
    let Some(g) = model() else { return };
    let t = Tokenizer::from_gguf(&g).unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let mut texts: Vec<String> = ["docs/SHADOW_PLAN.md", "docs/BENCH_NOISE.md", "README.md", "kernels/prelude.cuh",
                                  "crates/shadow-model/src/tokenizer.rs"]
        .iter()
        .filter_map(|p| std::fs::read_to_string(root.join(p)).ok())
        .collect();
    texts.push("日本語のテキスト、中文文本，한국어 텍스트, émoji 🙂👍🏽, Ünïcödé, tabs\tand\r\nCRLF, x²+y³".into());
    let mut tokens = 0;
    let t0 = std::time::Instant::now();
    for s in &texts {
        let ids = t.encode(s, false);
        tokens += ids.len();
        assert_eq!(t.decode(&ids), *s);
    }
    let bytes: usize = texts.iter().map(String::len).sum();
    eprintln!("{bytes} bytes -> {tokens} tokens ({:.2} bytes/token) in {:?}", bytes as f64 / tokens as f64, t0.elapsed());
}
