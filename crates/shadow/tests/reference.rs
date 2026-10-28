//! Stage 1's gate, as a test: Shadow's forward pass against the published reference implementation run on the BF16
//! checkpoint (eval/fixtures/reference) and on Shadow's own dequantized GGUF (eval/fixtures/reference_gguf).  Needs the
//! model file; without it each test says SKIPPED and passes.  ~10 s on the CPU.

use shadow::reference::{compare, Fixture, Report};
use shadow_model::forward::{CpuBackend, Model};
use shadow_model::gguf::Gguf;
use std::path::Path;

const MODEL: &str =
    "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";
const FLINT_MODEL: &str = "I:/Flint-40B-A2.5B-abliterated-NVFP4.gguf";

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
    for (name, k, shared) in &r.routes {
        let mean = shared.iter().sum::<usize>() as f64 / shared.len() as f64;
        assert!(mean >= (*k as f64 - 1.0), "{name}: only {mean:.2} of {k} routed experts agree on average");
    }
}

fn assert_exact(r: &Report, what: &str, rel_max: f64) {
    for row in &r.rows {
        assert!(row.cos_min() > 0.99999 && row.rel_max() < rel_max,
                "{what}: {} cosine {:.7}, relative error {:.2e}", row.name, row.cos_min(), row.rel_max());
    }
    for (name, k, shared) in &r.routes {
        assert!(shared.iter().all(|&n| n == *k), "{what}: {name} routes differently: {shared:?}");
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

/// `GpuSession::reset` puts the session back exactly: the same session, compared, reset, compared again, must match
/// the reference both times (a state that survived the reset would move the second run's streams).  This is what the
/// server relies on to reuse one session across requests.
#[test]
fn a_reset_gpu_session_matches_the_reference_again() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let fx = Fixture::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures/reference_gguf")).unwrap();
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    let first = shadow::reference::compare_stepper(&mut s, &fx);
    assert_exact(&first, "gpu session, first run", 5e-3);
    // something unrelated in between, through all 48 layers, so the reset has state to clear everywhere
    s.stop_after = None;
    for t in [9707u32, 11, 1879, 553] {
        s.step(t);
    }
    s.reset();
    assert!(s.tokens().is_empty());
    let again = shadow::reference::compare_stepper(&mut s, &fx);
    assert_exact(&again, "gpu session, after reset", 5e-3);
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

#[test]
fn flint_matches_the_reference_on_shadows_own_weights() {
    let p = std::env::var("SHADOW_FLINT_MODEL").unwrap_or_else(|_| FLINT_MODEL.into());
    if !Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no Flint model at {p}");
    }
    let fx = Fixture::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures/reference_flint_gguf")).unwrap();
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let r = compare(&model, &CpuBackend, &fx);
    assert_exact(&r, "flint cpu", 1e-3);
}

#[test]
fn flint_gpu_session_matches_the_reference() {
    let p = std::env::var("SHADOW_FLINT_MODEL").unwrap_or_else(|_| FLINT_MODEL.into());
    if !Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no Flint model at {p}");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let fx = Fixture::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures/reference_flint_gguf")).unwrap();
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    let r = shadow::reference::compare_stepper(&mut s, &fx);
    assert_exact(&r, "flint gpu session", 5e-3);
}

/// `GpuSession::snapshot` / `restore`, the rollback a speculative pass needs and the prefix a new chat reuses: after
/// a prefix, a snapshot; then other tokens; then the restore and one more token must give exactly the logits a session
/// that only ever saw the prefix gives.  Without the KV (rollback: the prefix positions untouched) and with it (after a
/// reset and a different prompt overwrote them).
#[test]
fn a_restored_snapshot_continues_exactly() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let prefix: Vec<u32> = vec![760, 6722, 314, 9338, 369, 11751, 13, 576, 6722, 314, 9294, 369];
    let (next, other) = (13u32, [1059u32, 2101, 77, 4096, 512, 9]);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

    let mut fresh = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    for &t in &prefix {
        fresh.step_prefill(t);
    }
    let want = bits(&fresh.step(next));
    drop(fresh);

    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    for &t in &prefix {
        s.step_prefill(t);
    }
    let light = s.snapshot(false).unwrap();
    let full = s.snapshot(true).unwrap();
    for &t in &other {
        s.step_prefill(t);
    }
    s.restore(&light).unwrap();
    assert_eq!(s.tokens(), &prefix[..]);
    assert!(bits(&s.step(next)) == want, "rolled back without KV: the logits moved");

    s.reset();
    for &t in other.iter().chain(&other) {
        s.step_prefill(t);
    }
    assert!(s.restore(&light).is_err(), "a KV-less snapshot must refuse a session that moved off its tokens");
    s.restore(&full).unwrap();
    assert!(bits(&s.step(next)) == want, "restored with KV after a different prompt: the logits moved");

    // the same through host memory (what the server keeps a shared prompt prefix in)
    s.reset();
    for &t in &prefix {
        s.step_prefill(t);
    }
    let host = s.snapshot_to(true, true).unwrap();
    s.reset();
    for &t in &other {
        s.step_prefill(t);
    }
    s.restore(&host).unwrap();
    assert!(bits(&s.step(next)) == want, "restored from host memory: the logits moved");
}

/// `GpuSession::step_batch` (up to 8 tokens through the layers together) against `step` one token at a time: every
/// token's logits bit-identical, over chunks of 8 and a short last one, and the state after it the same (one more
/// `step` on each).  A RAM-tier-free session whose cache misses go through the staging buffer.
#[test]
fn a_batched_step_matches_one_token_at_a_time() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let prompt: Vec<u32> = [760u32, 6722, 314, 9338, 369, 11751, 13, 576, 6722, 314, 9294, 369, 13, 1059, 2101, 77,
                            4096, 512, 9, 760, 6722, 314, 9338, 369, 11751, 13, 576, 6722, 314]
        .to_vec();
    let next = 9294u32;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

    let mut one = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    let want: Vec<Vec<u32>> = prompt.iter().map(|&t| bits(&one.step(t))).collect();
    let want_next = bits(&one.step(next));
    drop(one);

    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    let got: Vec<Vec<u32>> = s.step_batch(&prompt).iter().map(|v| bits(v)).collect();
    assert_eq!(got.len(), prompt.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g == w, "token {i}: the batched logits differ from one token at a time");
    }
    assert!(bits(&s.step(next)) == want_next, "the state after the batched step differs");
    // and a prompt read without logits, then one step
    s.reset();
    s.prefill_batch(&prompt);
    assert!(bits(&s.step(next)) == want_next, "prefill_batch then step differs");
}

/// The same past the sparse indexer's threshold: the long fixture's 2,600 ids read in batches, then its last 8 tokens
/// through one `step_batch`, against one token at a time.
#[test]
#[ignore = "a few minutes: 2,600 tokens through 48 layers twice"]
fn a_batched_step_matches_one_token_at_a_time_past_the_indexer() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/fixtures/reference_gguf_long");
    if !Path::new(&p).is_file() || !dir.is_dir() {
        return eprintln!("SKIPPED, not passed: no model or no long fixture");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let ids = Fixture::load(&dir).unwrap().ids;
    let (head, tail) = ids.split_at(ids.len() - 8);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

    let mut one = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    for &t in head {
        one.step_prefill(t);
    }
    let want: Vec<Vec<u32>> = tail.iter().map(|&t| bits(&one.step(t))).collect();
    drop(one);

    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    s.prefill_batch(head);
    let got: Vec<Vec<u32>> = s.step_batch(tail).iter().map(|v| bits(v)).collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g == w, "position {}: the batched logits differ from one token at a time", head.len() + i);
    }
}

/// A speculative verify pass rolled back to a prefix (`step_batch_verify` + `rollback`, which redoes only the
/// recurrences from what the pass kept) continues exactly as reading that prefix one token at a time.
#[test]
fn a_rolled_back_verify_pass_continues_exactly() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let Ok(ctx) = shadow_cuda::Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    let model = Model::load(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let prompt: Vec<u32> = vec![760, 6722, 314, 9338, 369, 11751, 13, 576, 6722, 314, 9294, 369];
    let (pass, next) = ([13u32, 1059, 2101, 77], 4096u32);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();

    let mut one = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    for &t in &prompt {
        one.step_prefill(t);
    }
    let first = bits(&one.step(pass[0]));
    let second = bits(&one.step(pass[1]));
    let want = bits(&one.step(next));
    drop(one);

    let mut s = shadow_cuda::session::GpuSession::new(&ctx, &model, 4096).unwrap();
    s.prefill_batch(&prompt);
    let snap = s.snapshot(false).unwrap();
    let outs = s.step_batch_verify(&pass);
    assert!(bits(&outs[0]) == first && bits(&outs[1]) == second, "the verify pass differs from one token at a time");
    s.rollback(&snap, &pass[..2]).unwrap();
    assert_eq!(s.tokens(), &[&prompt[..], &pass[..2]].concat()[..]);
    assert!(bits(&s.step(next)) == want, "after the rollback the logits moved");
}
