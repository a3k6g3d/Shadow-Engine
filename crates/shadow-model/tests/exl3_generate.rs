//! The CPU reference forward pass over an EXL3 checkpoint, greedy, for `EXL3_TOKENS` tokens (default 40): the first end to end
//! check of the whole chain (checkpoint view, tensor map, EXL3 matvec, n-gram table, tokenizer).  ARCH.md records that a
//! wrong value-head order shows only after ~30 tokens ("Paris", then nonsense), hence the length.  Slow (seconds a token);
//! `#[ignore]`d: `cargo test --release -p shadow-model --test exl3_generate -- --ignored --nocapture`.
//! `EXL3_MODEL` names the checkpoint directory (default the uncensored 3 bpw repo), `EXL3_PROMPT` the prompt.

use shadow_model::forward::{CpuBackend, Model, Session, Stepper};
use shadow_model::tokenizer::Tokenizer;
use std::path::Path;

const REPO: &str = "J:/Models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw";

#[test]
#[ignore]
fn the_cpu_reference_forward_reads_an_exl3_checkpoint_and_speaks_sense() {
    let repo = std::env::var("EXL3_MODEL").unwrap_or_else(|_| REPO.into());
    if !Path::new(&repo).join("config.json").is_file() {
        return eprintln!("SKIPPED, not passed: no checkpoint at {repo}");
    }
    let n: usize = std::env::var("EXL3_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let prompt = std::env::var("EXL3_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let t0 = std::time::Instant::now();
    let m = Model::load_exl3(Path::new(&repo), false).unwrap();
    eprintln!("loaded in {:.1} s, dense tensors {:.2} GiB in RAM", t0.elapsed().as_secs_f64(), m.dense_bytes() as f64 / (1u64 << 30) as f64);
    let tok = Tokenizer::from_gguf(&m.g).unwrap();
    let ids = tok.encode(&prompt, true);
    eprintln!("prompt {prompt:?} = {ids:?}");
    let mut s = Session::new(&m, &CpuBackend);
    let mut out: Vec<u32> = vec![];
    let t1 = std::time::Instant::now();
    let mut logits = s.prefill(&ids);
    eprintln!("prefill of {} tokens: {:.1} s", ids.len(), t1.elapsed().as_secs_f64());
    for i in 0..n {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        out.push(next);
        eprintln!("token {i:2}: {next:6} {:?}   (so far: {:?})", tok.decode(&[next]), tok.decode(&out));
        if tok.is_eog(next) {
            break;
        }
        let t = std::time::Instant::now();
        logits = s.step(next);
        eprintln!("   step {:.1} s", t.elapsed().as_secs_f64());
    }
    eprintln!("GENERATED: {}{}", prompt, tok.decode(&out));
}
