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
    run_on(fixture, &CpuBackend)
}

fn run_on(fixture: &str, be: &dyn shadow_model::forward::Backend) -> Option<Report> {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        eprintln!("SKIPPED, not passed: no model at {p}");
        return None;
    }
    let fx = Fixture::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures").join(fixture)).unwrap();
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    Some(compare(&model, be, &fx))
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

fn assert_exact(r: &Report, what: &str, rel_max: f64) {
    for row in &r.rows {
        assert!(row.cos_min() > 0.99999 && row.rel_max() < rel_max,
                "{what}: {} cosine {:.7}, relative error {:.2e}", row.name, row.cos_min(), row.rel_max());
    }
    for (name, shared) in &r.routes {
        assert!(shared.iter().all(|&n| n == 10), "{what}: {name} routes differently: {shared:?}");
    }
}

/// The decisive one: the reference run on Shadow's OWN dequantized weights (eval/fixtures/reference_gguf).  There is no
/// quantization difference left, so every stream of all four layers - hyper-connections, DeltaNet, gated attention,
/// the layer-1 PLE, the MoE, the head's input - must match to floating-point order, and the routing exactly.
#[test]
fn matches_the_reference_on_shadows_own_weights() {
    let Some(r) = run("reference_gguf") else { return };
    assert_exact(&r, "cpu", 1e-3);
}

/// Same, with the matrix-vector products on the GPU (f32 sums in a different order): a little looser.
#[test]
fn the_gpu_backend_matches_it_too() {
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let gpu = shadow_cuda::backend::GpuBackend::new(&ctx, 8 << 30).unwrap();
    let Some(r) = run_on("reference_gguf", &gpu) else { return };
    assert_exact(&r, "gpu", 5e-3);
}

/// The sparse indexer (positions past 2,051, where more than 512 blocks are complete): a 2,600-token prompt through
/// layers 0-3 on Shadow's own weights, against the reference at the positions kept in
/// eval/fixtures/reference_gguf_long (written by `make_reference_fixture.py --positions ...`; see docs/HANDOFF.md).
/// Layer 3 is the first attention layer.  Strict: the attended positions must be the reference's, and every stream must
/// match.  Shadow runs all 2,600 tokens, so this takes minutes on the CPU; run it with `-- --ignored`.
#[test]
#[ignore = "minutes: 2,600 tokens through 4 layers"]
fn the_sparse_indexer_matches_the_reference() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures/reference_gguf_long");
    if !dir.join("manifest.json").is_file() {
        return eprintln!("SKIPPED, not passed: no long fixture (docs/HANDOFF.md says how to make it)");
    }
    let Some(r) = run("reference_gguf_long") else { return };
    shadow::reference::print(&r);
    let sel = r.selections.iter().find(|(n, _)| n == "layer3").expect("layer 3 selection");
    assert!(sel.1.iter().all(|&d| d == 0), "layer 3 attends to different positions than the reference: {:?}", sel.1);
    assert_exact(&r, "cpu, long prompt", 1e-3);
}

/// The whole forward pass on the GPU (shadow_cuda::session::GpuSession): every stream of layers 0-3 against the
/// reference on Shadow's own weights, and the routing exactly.
#[test]
fn the_gpu_session_matches_the_reference() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let fx = Fixture::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures/reference_gguf")).unwrap();
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    let r = shadow::reference::compare_stepper(&mut s, &fx);
    assert_exact(&r, "gpu session", 5e-3);
}

/// The GPU session's sparse indexer on the 2,600-token prompt (about a minute).
#[test]
#[ignore = "about a minute: 2,600 tokens through 4 layers on the GPU"]
fn the_gpu_sessions_sparse_indexer_matches_the_reference() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures/reference_gguf_long");
    if !Path::new(&p).is_file() || !dir.join("manifest.json").is_file() {
        return eprintln!("SKIPPED, not passed: no model or no long fixture");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let fx = Fixture::load(&dir).unwrap();
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    let r = shadow::reference::compare_stepper(&mut s, &fx);
    let sel = r.selections.iter().find(|(n, _)| n == "layer3").expect("layer 3 selection");
    assert!(sel.1.iter().all(|&d| d == 0), "layer 3 attends to different positions than the reference: {:?}", sel.1);
    assert_exact(&r, "gpu session, long prompt", 5e-3);
}
