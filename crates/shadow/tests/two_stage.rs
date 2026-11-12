//! Phase 1 of docs/PLAN_TWO_GPUS_2026-11-11.md as a test: the model's layers split between two sessions on ONE card (each owning a range of layers,
//! the second taking the first's residual streams through host memory) give bit-for-bit the logits of one session that owns them all.
//! The sessions are the exact path (no drafter, no tail skip, no speculation, no VRAM expert cache).  Needs a card and the model file; without
//! them it says SKIPPED and passes.  Run it under tools/run_capped.ps1 (it loads the model's dense weights and reads experts from the files).

use shadow_cuda::driver::Context;
use shadow_cuda::session::{GpuSession, Snapshot, StageOut, Tiers};
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

/// A VRAM expert cache of 384 MiB: enough for the batched path (a session without one steps a token at a time), small enough for two sessions.
fn tiers_cached(stage: Option<(usize, usize)>) -> Tiers {
    Tiers { layers: stage, vram_cache: Some(384 << 20), ..Tiers::default() }
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

/// The batched step: 5 tokens in one pass, then 3 more (the second pass starts at position 5), through one session and through the pair.
#[test]
fn batches_through_two_stages_equal_one_session() {
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
    let passes: [&[u32]; 2] = [&TOKENS[..5], &[777, 31337, 99]];
    let want: Vec<Vec<Vec<f32>>> = {
        let mut s = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(None)).unwrap();
        passes.iter().map(|t| s.step_batch(t)).collect()
    };
    for split in [1usize, 24, n_layer - 1] {
        let mut a = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(Some((0, split)))).unwrap();
        let mut b = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(Some((split, n_layer)))).unwrap();
        for (pi, toks) in passes.iter().enumerate() {
            let StageOut::Streams(streams) = a.step_batch_stage(toks, None) else { panic!("stage A is not the last") };
            let StageOut::Logits(got) = b.step_batch_stage(toks, Some(&streams)) else { panic!("stage B is the last") };
            assert_eq!(got.len(), want[pi].len());
            for (i, (g, w)) in got.iter().zip(&want[pi]).enumerate() {
                let (gb, wb) = (bits(g), bits(w));
                if gb != wb {
                    let first = gb.iter().zip(&wb).position(|(x, y)| x != y).unwrap();
                    panic!("split at layer {split}, pass {pi}, row {i}: logits differ from index {first} ({} of {} differ)",
                           gb.iter().zip(&wb).filter(|(x, y)| x != y).count(), gb.len());
                }
            }
        }
        eprintln!("batched, split at layer {split}: 2 passes ({} tokens), every logit identical", passes.iter().map(|p| p.len()).sum::<usize>());
    }
}

/// One token / a few tokens through the pair (stage A then stage B, the streams through a host `Vec`).
fn pair_step(a: &mut GpuSession, b: &mut GpuSession, t: u32) -> Vec<f32> {
    let streams = a.step_stage(t, None);
    b.step_stage(t, Some(&streams))
}

fn pair_batch(a: &mut GpuSession, b: &mut GpuSession, toks: &[u32]) -> Vec<Vec<f32>> {
    let StageOut::Streams(streams) = a.step_batch_stage(toks, None) else { panic!("stage A is not the last") };
    let StageOut::Logits(l) = b.step_batch_stage(toks, Some(&streams)) else { panic!("stage B is the last") };
    l
}

fn pair_verify(a: &mut GpuSession, b: &mut GpuSession, toks: &[u32]) -> Vec<Vec<f32>> {
    let StageOut::Streams(streams) = a.step_batch_verify_stage(toks, None) else { panic!("stage A is not the last") };
    let StageOut::Logits(l) = b.step_batch_verify_stage(toks, Some(&streams)) else { panic!("stage B is the last") };
    l
}

/// `rollback` on the pair: each stage replays what its verify pass recorded; when it did not (the kept tokens are not a prefix of the pass), the kept
/// tokens are read again through both stages.
fn pair_rollback<'c, 'm>(a: &mut GpuSession<'c, 'm>, b: &mut GpuSession<'c, 'm>, snaps: &(Snapshot<'c>, Snapshot<'c>), kept: &[u32]) -> bool {
    let (ra, rb) = (a.rollback_replay(&snaps.0, kept).unwrap(), b.rollback_replay(&snaps.1, kept).unwrap());
    assert_eq!(ra, rb, "both stages agree whether the pass recorded the kept tokens");
    if !ra {
        if kept.len() == 1 {
            pair_step(a, b, kept[0]);
        } else {
            pair_batch(a, b, kept);
        }
    }
    ra
}

fn same(got: &[f32], want: &[f32], what: &str) {
    let (gb, wb) = (bits(got), bits(want));
    if gb != wb {
        let first = gb.iter().zip(&wb).position(|(x, y)| x != y).unwrap();
        panic!("{what}: logits differ from index {first} ({} of {} differ)", gb.iter().zip(&wb).filter(|(x, y)| x != y).count(), gb.len());
    }
}

/// Speculation's own steps across the pair: a verify pass, `rollback` replaying the recorded recurrent states (the kept tokens a prefix of the pass), then one
/// that has to read its kept tokens again (they are not), each followed by a normal step: every logit identical to one session.
#[test]
fn verify_and_rollback_through_two_stages_equal_one_session() {
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
    const V: [u32; 4] = [11, 22, 33, 44];
    const V2: [u32; 3] = [66, 77, 88];
    // one session: the reference rows, in order: verify, after the replaying rollback, verify 2, after the re-reading rollback
    let want: Vec<Vec<Vec<f32>>> = {
        let mut s = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(None)).unwrap();
        s.step_batch(&TOKENS[..2]);
        let snap = s.snapshot(false).unwrap();
        let v = s.step_batch_verify(&V);
        s.rollback(&snap, &V[..2]).unwrap();
        let after1 = s.step(555);
        let snap2 = s.snapshot(true).unwrap(); // with the KV cache, as the server's conversation snapshots are
        let v2 = s.step_batch_verify(&V2);
        s.rollback(&snap2, &[66, 99]).unwrap();
        let after2 = s.step(123);
        vec![v, vec![after1], v2, vec![after2]]
    };
    for split in [1usize, 24, n_layer - 1] {
        let mut a = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(Some((0, split)))).unwrap();
        let mut b = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(Some((split, n_layer)))).unwrap();
        pair_batch(&mut a, &mut b, &TOKENS[..2]);
        let snaps = (a.snapshot(false).unwrap(), b.snapshot(false).unwrap());
        let v = pair_verify(&mut a, &mut b, &V);
        for (i, (g, w)) in v.iter().zip(&want[0]).enumerate() {
            same(g, w, &format!("split {split}, verify row {i}"));
        }
        assert!(pair_rollback(&mut a, &mut b, &snaps, &V[..2]), "the first rollback replays");
        same(&pair_step(&mut a, &mut b, 555), &want[1][0], &format!("split {split}, after the replaying rollback"));
        let snaps2 = (a.snapshot(true).unwrap(), b.snapshot(true).unwrap());
        let v2 = pair_verify(&mut a, &mut b, &V2);
        for (i, (g, w)) in v2.iter().zip(&want[2]).enumerate() {
            same(g, w, &format!("split {split}, verify 2 row {i}"));
        }
        assert!(!pair_rollback(&mut a, &mut b, &snaps2, &[66, 99]), "the second rollback reads its tokens again");
        same(&pair_step(&mut a, &mut b, 123), &want[3][0], &format!("split {split}, after the re-reading rollback"));
        eprintln!("verify + rollback (replayed, re-read), split at layer {split}: every logit identical");
    }
}

/// The same calls as the engine makes, on whatever `s` is: one session, or a first stage with a second attached (`attach_peer`).
fn script<'c, 'm>(s: &mut GpuSession<'c, 'm>) -> Vec<Vec<f32>> {
    const V: [u32; 4] = [11, 22, 33, 44];
    const V2: [u32; 3] = [66, 77, 88];
    let mut out = vec![];
    out.extend(s.step_batch(&TOKENS[..2]));
    let snap = s.snapshot(false).unwrap();
    out.extend(s.step_batch_verify(&V));
    s.rollback(&snap, &V[..2]).unwrap(); // replayed
    out.push(s.step(555));
    let snap_kv = s.snapshot(true).unwrap(); // after 5 tokens, with the KV cache
    out.extend(s.step_batch_verify(&V2));
    s.rollback(&snap_kv, &[66, 99]).unwrap(); // read again
    out.push(s.step(123));
    s.restore_conv(&snap_kv).unwrap(); // back to a conversation snapshot, as the server does
    out.extend(s.step_batch_last(&[5, 6, 7]));
    s.reset();
    out.extend(s.step_batch(&TOKENS[..3])); // a new conversation
    out
}

/// A first stage with the second attached is one session to the caller: the whole script (batches, a verify pass, both rollbacks, snapshots with and without
/// the KV, a restore, a reset and a new conversation) gives every logit identical to one session.
#[test]
fn a_pair_driven_through_the_first_stage_equals_one_session() {
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
    let want = {
        let mut s = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(None)).unwrap();
        script(&mut s)
    };
    for split in [1usize, 24, n_layer - 1] {
        let mut a = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(Some((0, split)))).unwrap();
        let b = GpuSession::with_tiers(&ctx, &model, CAP, tiers_cached(Some((split, n_layer)))).unwrap();
        a.attach_peer(b).unwrap();
        let got = script(&mut a);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            same(g, w, &format!("split {split}, output {i}"));
        }
        eprintln!("the pair as one session, split at layer {split}: {} outputs, every logit identical", got.len());
    }
}

// ---------------------------------------------------------------------------------------------------------------------- the drafter across the stages

fn drafter_dir() -> Option<std::path::PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/dflash2");
    d.join("model.safetensors").is_file().then_some(d)
}

fn tiers_drafter(stage: Option<(usize, usize)>, dir: &Path, first_of_pair: bool) -> Tiers {
    let mut t = Tiers { layers: stage, vram_cache: Some(384 << 20), dflash_q8: true, ..Tiers::default() };
    if first_of_pair {
        t.df_taps = true;
    } else {
        t.dflash_dir = Some(dir.to_path_buf());
    }
    t
}

/// What the drafter drafts through a session's life: after a batched prompt, single steps, a verify pass, a rollback (replayed), and a conversation snapshot's
/// restore (`restore_conv`, which keeps the drafter's K/V: STATUS 98) followed by more tokens.  The same script runs on a session and on a pair.
fn draft_script<'c, 'm>(s: &mut GpuSession<'c, 'm>) -> Vec<Vec<u32>> {
    let prompt: Vec<u32> = (0..20u32).map(|i| 1000 + i * 37).collect();
    let mut out = vec![];
    s.step_batch(&prompt);
    out.push(s.dflash_draft(555, 3, 0.0));
    s.step(555);
    out.push(s.dflash_draft(777, 3, 0.0));
    s.step(777);
    let snap_kv = s.snapshot(true).unwrap(); // 22 tokens, with the KV
    let snap = s.snapshot(false).unwrap();
    s.step_batch_verify(&[5, 11, 22, 33]);
    out.push(s.dflash_draft(44, 3, 0.0));
    s.rollback(&snap, &[5, 11]).unwrap(); // replayed
    out.push(s.dflash_draft(22, 3, 0.0));
    s.step(22);
    s.step(33);
    out.push(s.dflash_draft(66, 7, 0.0));
    // a new conversation from the snapshot of 22 tokens: the drafter keeps the K/V of the positions the two share
    s.restore_conv(&snap_kv).unwrap();
    out.push(s.dflash_draft(88, 3, 0.0));
    s.step_batch(&[88, 99, 111]);
    out.push(s.dflash_draft(222, 3, 0.0));
    s.step(222);
    out.push(s.dflash_draft(333, 7, 0.0));
    out
}

/// The DFlash drafter on the second stage, fed the features of the taps below the split by the first: every draft identical to the single session's.  Splits: 1
/// (no tap in the first stage), 20 (two), 40 (four), 47 (all five: the second stage has none).
#[test]
fn the_drafter_across_two_stages_drafts_what_one_session_drafts() {
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !Path::new(&p).is_file() {
        eprintln!("SKIPPED, not passed: no model at {p}");
        return;
    }
    let Some(dir) = drafter_dir() else {
        eprintln!("SKIPPED, not passed: no models/dflash2");
        return;
    };
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no card ({e})");
            return;
        }
    };
    let model = Model::load_lean(Gguf::open(Path::new(&p)).unwrap()).unwrap();
    let n_layer = model.cfg.n_layer;
    let want = {
        let mut t = tiers_drafter(None, &dir, false);
        t.layers = None;
        let mut s = GpuSession::with_tiers(&ctx, &model, CAP, t).unwrap();
        draft_script(&mut s)
    };
    let drafted = want.iter().filter(|d| !d.is_empty()).count();
    assert!(drafted >= want.len() - 1, "the reference session drafted in only {drafted} of {} rounds: {want:?}", want.len());
    for split in [1usize, 20, 40, n_layer - 1] {
        let mut a = GpuSession::with_tiers(&ctx, &model, CAP, tiers_drafter(Some((0, split)), &dir, true)).unwrap();
        let b = GpuSession::with_tiers(&ctx, &model, CAP, tiers_drafter(Some((split, n_layer)), &dir, false)).unwrap();
        a.attach_peer(b).unwrap();
        assert!(a.drafter().is_some());
        let got = draft_script(&mut a);
        assert_eq!(got, want, "split at layer {split}: the pair drafts other tokens than one session");
        eprintln!("the drafter across two stages, split at layer {split}: {} rounds, every draft identical ({drafted} drafted)", got.len());
    }
}

// ---------------------------------------------------------------------------------------------------------------------- a long prompt (the lent cache, big batches)

/// A prompt of 1,300 tokens read the way the engine reads one (`hint_prompt`, `lend_cache`, chunks of `batch_rows()`, the last with logits), then a few steps, on one
/// session and on the pair: the hop moves big passes (more than one chunk of the pinned buffer when the rows allow it).  Every logit identical.
#[test]
fn a_long_prompt_through_the_pair_equals_one_session() {
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
    const LONG: usize = 1300;
    let prompt: Vec<u32> = (0..LONG as u32).map(|i| 1000 + (i * 7919) % 9000).collect();
    let run = |s: &mut GpuSession, tag: &str| -> Vec<Vec<f32>> {
        let mut out = vec![];
        s.hint_prompt(prompt.len());
        let lent = s.lend_cache(false);
        let chunk = s.batch_rows();
        eprintln!("{tag}: lent {lent}, batch_rows {chunk}");
        let mut i = 0;
        while i < prompt.len() {
            let end = (i + chunk).min(prompt.len());
            if end == prompt.len() {
                out.push(s.step_batch_last(&prompt[i..end]).pop().unwrap());
            } else {
                s.prefill_batch(&prompt[i..end]);
            }
            i = end;
        }
        if lent {
            s.restore_cache();
        }
        out.push(s.step(4242));
        out.push(s.step(17));
        out
    };
    let big = |stage: Option<(usize, usize)>| Tiers { layers: stage, vram_cache: Some(1536 << 20), ..Tiers::default() };
    let want = {
        let mut s = GpuSession::with_tiers(&ctx, &model, LONG + 8, big(None)).unwrap();
        run(&mut s, "one session")
    };
    let mut a = GpuSession::with_tiers(&ctx, &model, LONG + 8, big(Some((0, 24)))).unwrap();
    let b = GpuSession::with_tiers(&ctx, &model, LONG + 8, big(Some((24, n_layer)))).unwrap();
    a.attach_peer(b).unwrap();
    let got = run(&mut a, "pair");
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        same(g, w, &format!("long prompt, output {i}"));
    }
    eprintln!("a {LONG}-token prompt through the pair: {} outputs, every logit identical", got.len());
}
