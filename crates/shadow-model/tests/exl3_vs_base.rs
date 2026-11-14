//! The EXL3 decoder against the checkpoint the EXL3 branch was quantized FROM (`Qwen/Qwen3.8-Flash-Next`, BF16): an
//! independent reference for the decode (the GGUF comparisons of `exl3_vs_gguf.rs` are against a fine-tune quantized
//! another way, so a small error in a codebook constant or in f16 rounding would hide in their spread).  A trellis
//! quantizer minimises the error to these very weights, so a correct decode lands at the quantizer's own accuracy:
//! relative error about 0.1 at K = 4 and 0.3 at K = 2 are the scale to expect; anything off by a constant, a sign or a
//! rotation is far worse.  The base checkpoint stores experts fused (`experts.gate_up_proj` `[512, 1280, 2560]`, gate
//! first, and `experts.down_proj`); the sample file holds per-expert slices under per-expert names (a range-request tool).
//!
//! Needs `EXL3_SUBSET` (`tools/fetch_exl3_subset.py` from `turboderp/Qwen3.8-Flash-Next-exl3` branch `2.05bpw_h4_ng4`:
//! layers 0-2 experts 0-1 and shared experts, layer-0 linear attention) and `BASE_SUBSET` (the same names with
//! `.weight`, from `Qwen/Qwen3.8-Flash-Next`); skips otherwise.

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
fn the_decode_lands_at_the_quantizers_accuracy_against_the_base_checkpoint() {
    let (Ok(e), Ok(b)) = (std::env::var("EXL3_SUBSET"), std::env::var("BASE_SUBSET")) else {
        return eprintln!("SKIPPED, not passed: EXL3_SUBSET and BASE_SUBSET are not set");
    };
    if !std::path::Path::new(&e).is_file() || !std::path::Path::new(&b).is_file() {
        return eprintln!("SKIPPED, not passed: no file at {e} or {b}");
    }
    let ex = SafeTensors::open(&[e.into()]).unwrap();
    let ba = SafeTensors::open(&[b.into()]).unwrap();
    let mut worst: [(f64, f64); 9] = [(1.0, 0.0); 9]; // per K: (min cosine, max relative error)
    let mut n = 0;
    for name in ex.tensors.keys().filter(|k| k.ends_with(".trellis")) {
        let prefix = name.strip_suffix(".trellis").unwrap();
        let Some(bt) = ba.get(&format!("{prefix}.weight")) else { continue };
        let lin = Exl3Linear::load(&ex, prefix).unwrap();
        let (ni, no) = (lin.in_features, lin.out_features);
        assert_eq!(bt.shape, vec![no, ni], "{prefix}: base shape");
        let base = ba.read_f32(&format!("{prefix}.weight")).unwrap(); // [out][in]
        let w = lin.reconstruct(); // [in][out]
        let wt: Vec<f32> = (0..no).flat_map(|o| (0..ni).map(move |i| (i, o))).map(|(i, o)| w[i * no + o]).collect();
        let (cs, rel) = stats(&wt, &base);
        eprintln!("{prefix:75} K={} cosine {cs:.4} relative error {rel:.4}", lin.bits);
        let k = lin.bits;
        worst[k].0 = worst[k].0.min(cs);
        worst[k].1 = worst[k].1.max(rel);
        n += 1;
    }
    assert!(n >= 10, "only {n} tensors had a base counterpart");
    for (k, (cs, rel)) in worst.iter().enumerate() {
        if *cs < 1.0 {
            eprintln!("K={k}: worst cosine {cs:.4}, worst relative error {rel:.4}");
        }
    }
    // the rate-distortion bound for a Gaussian is 2^-K in relative error (0.0625 at K=4, 0.25 at K=2); a trellis quantizer
    // reaches within a fifth or so of it
    assert!(worst[4].0 < 1.0 && worst[2].0 < 1.0, "the sample must hold K=2 and K=4 tensors: {worst:?}");
    assert!(worst[4].0 > 0.995 && worst[4].1 < 0.1, "K=4 against the base: {:?}", worst[4]);
    assert!(worst[2].0 > 0.95 && worst[2].1 < 0.32, "K=2 against the base: {:?}", worst[2]);
}
