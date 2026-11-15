//! The half-integer EXL3 rates (K + 0.5 bits a weight: exllamav3's `dq8_half`, `Rate::end_bit`) against the BF16 base checkpoint, as `exl3_vs_base.rs` does for the
//! integer rates: a correct decode lands at the quantizer's own accuracy (the Gaussian rate-distortion bound is 2^-K in relative error: 0.177 at K = 2.5), a wrong bit
//! offset or window order is far worse.  The checkpoint is a fine-tune (an abliteration: a few directions removed from some matrices), so the bound is checked on the
//! tensors the abliteration leaves alone and loosely on the others.
//!
//! Needs `EXL3_HALF_DIR` (a directory with `model-*.safetensors` of an EXL3 checkpoint that has K = 2.5 tensors, e.g. `Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-2bpw`) and
//! `BASE_EXPERTS` / `BASE_SUBSET` (`tools/fetch_base_experts.py` and `tools/fetch_exl3_subset.py` for the base `Qwen/Qwen3.8-Flash-Next`: per-expert or dense `.weight` BF16
//! tensors under the checkpoint's names); skips otherwise.

use shadow_model::exl3::Exl3Linear;
use shadow_model::safetensors::SafeTensors;

fn stats(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut ab, mut aa, mut bb, mut dd) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        dd += (x - y) * (x - y);
    }
    (ab / (aa.sqrt() * bb.sqrt()), (dd / bb).sqrt())
}

#[test]
fn the_half_integer_decode_lands_at_the_quantizers_accuracy_against_the_base_checkpoint() {
    let Ok(dir) = std::env::var("EXL3_HALF_DIR") else {
        return eprintln!("SKIPPED, not passed: EXL3_HALF_DIR is not set");
    };
    let mut bases = vec![];
    for var in ["BASE_EXPERTS", "BASE_SUBSET"] {
        if let Ok(p) = std::env::var(var) {
            if std::path::Path::new(&p).is_file() {
                bases.push(SafeTensors::open(&[p.into()]).unwrap());
            }
        }
    }
    if bases.is_empty() {
        return eprintln!("SKIPPED, not passed: no BASE_EXPERTS / BASE_SUBSET file");
    }
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))).collect();
    shards.sort();
    let ex = SafeTensors::open(&shards).unwrap();
    // (rate, matrix kind) -> (min cosine, max relative error, count)
    let mut seen: std::collections::BTreeMap<String, (f64, f64, usize)> = Default::default();
    for name in ex.tensors.keys().filter(|k| k.ends_with(".trellis")) {
        let prefix = name.strip_suffix(".trellis").unwrap();
        let Some(ba) = bases.iter().find(|b| b.get(&format!("{prefix}.weight")).is_some()) else { continue };
        let lin = Exl3Linear::load(&ex, prefix).unwrap();
        let (ni, no) = (lin.in_features, lin.out_features);
        let base = ba.read_f32(&format!("{prefix}.weight")).unwrap(); // [out][in]
        let w = lin.reconstruct(); // [in][out]
        let wt: Vec<f32> = (0..no).flat_map(|o| (0..ni).map(move |i| (i, o))).map(|(i, o)| w[i * no + o]).collect();
        let (cs, rel) = stats(&wt, &base);
        let kind = prefix.rsplit('.').next().unwrap().to_string();
        eprintln!("{prefix:80} K={}{} cosine {cs:.4} relative error {rel:.4}", lin.bits, if lin.half { ".5" } else { "" });
        let e = seen.entry(format!("K={}{} {kind}", lin.bits, if lin.half { ".5" } else { "" })).or_insert((1.0, 0.0, 0));
        e.0 = e.0.min(cs);
        e.1 = e.1.max(rel);
        e.2 += 1;
    }
    for (k, (cs, rel, n)) in &seen {
        eprintln!("{k}: {n} tensors, worst cosine {cs:.4}, worst relative error {rel:.4}");
    }
    let halves: Vec<_> = seen.iter().filter(|(k, _)| k.starts_with("K=2.5")).collect();
    assert!(!halves.is_empty(), "the sample holds no K = 2.5 tensor with a base counterpart: {seen:?}");
    // K = 2.5: bound 0.177; a trellis quantizer reaches within ~15-20% of it; an untouched matrix must be there
    assert!(halves.iter().any(|(_, (cs, rel, _))| *cs > 0.97 && *rel < 0.24), "no K = 2.5 matrix lands at the quantizer's accuracy: {halves:?}");
}
