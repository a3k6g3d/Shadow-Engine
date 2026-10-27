//! Stage 1's gate, as a test: Shadow's forward pass against the published reference implementation run on the BF16
//! checkpoint (eval/fixtures/reference) and on Shadow's own dequantized GGUF (eval/fixtures/reference_gguf).  Needs the
//! model file; without it each test says SKIPPED and passes.  ~10 s on the CPU.

use shadow::reference::{compare, Fixture, Report};
use shadow_model::forward::{CpuBackend, Model};
use shadow_model::gguf::Gguf;
use std::path::Path;

const MODEL: &str =
    "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";

fn run(fixture: &str) -> Option<Report> {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        eprintln!("SKIPPED, not passed: no model at {p}");
        return None;
    }
    let fx = Fixture::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures").join(fixture)).unwrap();
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    Some(compare(&model, &CpuBackend, &fx))
}

fn row<'a>(r: &'a Report, name: &str) -> &'a shadow::reference::Row {
    r.rows.iter().find(|x| x.name == name).unwrap_or_else(|| panic!("no stream {name}"))
}

/// The true reference: BF16 weights.  Quantization noise (the 2-bit experts above all) keeps the streams from matching,
/// but their direction must stay close at every layer - a wrong operation shows as a cliff, not a slope.
#[test]
fn agrees_with_the_bf16_reference_in_direction() {
    let Some(r) = run("reference") else { return };
    for l in 0..4 {
        let out = row(&r, &format!("layer{l}.out"));
        assert!(out.cos_min() > 0.99, "layer{l}.out: cosine {:.4} with the reference", out.cos_min());
        for part in ["attn_hc_u", "mixer", "mlp_hc_u"] {
            let x = row(&r, &format!("layer{l}.{part}"));
            assert!(x.cos_min() > 0.97, "layer{l}.{part}: cosine {:.4}", x.cos_min());
        }
        let moe = row(&r, &format!("layer{l}.moe"));
        assert!(moe.cos_min() > 0.90, "layer{l}.moe: cosine {:.4}", moe.cos_min());
    }
    assert!(row(&r, "embed").cos_min() > 0.99);
    assert!(row(&r, "layer1.ple").cos_min() > 0.99);
    assert!(row(&r, "final_u").cos_min() > 0.97);
    // routing is a hard decision on a noisy input: most of the ten experts must still agree
    for (name, shared) in &r.routes {
        let mean = shared.iter().sum::<usize>() as f64 / shared.len() as f64;
        assert!(mean >= 9.0, "{name}: only {mean:.2} of 10 routed experts agree on average");
    }
}
