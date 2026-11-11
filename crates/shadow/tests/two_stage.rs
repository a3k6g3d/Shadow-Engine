//! Phase 1 of docs/PLAN_TWO_GPUS_2026-11-11.md as a test: the model's layers split between two sessions on ONE card (each owning a range of layers,
//! the second taking the first's residual streams through host memory) give bit-for-bit the logits of one session that owns them all.
//! The sessions are the exact path (no drafter, no tail skip, no speculation, no VRAM expert cache).  Needs a card and the model file; without
//! them it says SKIPPED and passes.  Run it under tools/run_capped.ps1 (it loads the model's dense weights and reads experts from the files).

use shadow_cuda::driver::Context;
use shadow_cuda::session::{GpuSession, Tiers};
use shadow_model::forward::Model;
use shadow_model::gguf::Gguf;
use std::path::Path;

const MODEL: &str =
    "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";
const TOKENS: [u32; 6] = [1000, 2345, 56, 78901, 4, 12345];
const CAP: usize = 64;

fn tiers(stage: Option<(usize, usize)>) -> Tiers {
    Tiers { layers: stage, vram_cache: Some(0), ..Tiers::default() }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn two_stages_on_one_card_equal_one_session() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        eprintln!("SKIPPED, not passed: no model at {p}");
        return;
    }
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no card ({e})");
            return;
        }
    };
    let model = Model::load_lean(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let n_layer = model.cfg.n_layer;
    // the reference: one session owning every layer
    let want: Vec<Vec<f32>> = {
        let mut s = GpuSession::with_tiers(&ctx, &model, CAP, tiers(None)).unwrap();
        TOKENS.iter().map(|&t| s.step(t)).collect()
    };
    for split in [1usize, 24, n_layer - 1] {
        let mut a = GpuSession::with_tiers(&ctx, &model, CAP, tiers(Some((0, split)))).unwrap();
        let mut b = GpuSession::with_tiers(&ctx, &model, CAP, tiers(Some((split, n_layer)))).unwrap();
        assert_eq!((a.layers(), b.layers()), ((0, split), (split, n_layer)));
        for (i, &t) in TOKENS.iter().enumerate() {
            let streams = a.step_stage(t, None);
            let got = b.step_stage(t, Some(&streams));
            let (gb, wb) = (bits(&got), bits(&want[i]));
            if gb != wb {
                let first = gb.iter().zip(&wb).position(|(x, y)| x != y).unwrap();
                let worst = got.iter().zip(&want[i]).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                panic!("split at layer {split}, token {i}: logits differ from index {first} ({} of {} differ, worst {worst:e})",
                       gb.iter().zip(&wb).filter(|(x, y)| x != y).count(), gb.len());
            }
        }
        eprintln!("split at layer {split}: {} tokens, every logit identical", TOKENS.len());
    }
}
