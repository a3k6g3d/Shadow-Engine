//! The GPU backend against the CPU reference, over a whole forward pass on the real model: the same prompt through
//! both, and at every position the same top token and logits within 1e-3 of the largest logit.  Needs the model
//! (SHADOW_MODEL, else the plan's path) and a CUDA device; without either it says SKIPPED and passes.  ~25 s.

use shadow_cuda::backend::GpuBackend;
use shadow_cuda::Context;
use shadow_model::forward::{CpuBackend, Model, Session};
use shadow_model::gguf::Gguf;
use shadow_model::tokenizer::Tokenizer;

const MODEL: &str =
    "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
fn gpu_and_cpu_backends_agree_over_a_forward_pass() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let ids = Tokenizer::from_gguf(&g).unwrap().encode("The capital of France is", false);
    let model = Model::load(g).unwrap();
    let gpu = GpuBackend::new(&ctx, 8 << 30).unwrap();
    let (mut a, mut b) = (Session::new(&model, &CpuBackend), Session::new(&model, &gpu));
    for (i, &id) in ids.iter().enumerate() {
        let (la, lb) = (a.step(id), b.step(id));
        let scale = la.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = la.iter().zip(&lb).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!("position {i}: top {} / {}, max |logit diff| {worst:.2e} (largest logit {scale:.2})", argmax(&la),
                  argmax(&lb));
        assert_eq!(argmax(&la), argmax(&lb), "position {i}: the backends pick different tokens");
        assert!(worst <= 1e-3 * scale, "position {i}: logits differ by {worst} (scale {scale})");
    }
}
