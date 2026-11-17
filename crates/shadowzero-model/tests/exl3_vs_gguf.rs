//! The EXL3 decoder against a GGUF of the same model: the same weights quantized two unrelated ways, so a correct decode
//! of the trellis, the tile layout and both Hadamards makes the two agree (cosine well above 0.9) and any mistake in
//! them (a permuted tile, a missing rotation, a wrong sign vector) makes them agree no better than chance (cosine ~0).
//!
//! Needs `EXL3_SUBSET` (a safetensors file holding the tensors read below; `tools/fetch_exl3_subset.py` writes one from
//! a repo branch by range requests) and a GGUF of the model (`SHADOWZERO_MODEL`, default the IQ3_S file); skips otherwise.

use shadowzero_model::exl3::Exl3Linear;
use shadowzero_model::gguf::Gguf;
use shadowzero_model::quant::dequantize;
use shadowzero_model::safetensors::SafeTensors;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += *x as f64 * *x as f64;
        bb += *y as f64 * *y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// Expert `e` of a stacked GGUF expert tensor, dequantized: `[out][in]`.
fn gguf_expert(g: &Gguf, name: &str, e: usize) -> Vec<f32> {
    let t = g.tensor(name).unwrap_or_else(|| panic!("no tensor {name}"));
    let (inn, out, ne) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
    let per = (t.nbytes() as usize) / ne;
    let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
    f.seek(SeekFrom::Start(t.offset + (e * per) as u64)).unwrap();
    let mut b = vec![0u8; per];
    f.read_exact(&mut b).unwrap();
    let mut y = vec![0f32; inn * out];
    dequantize(t.ty, &b, &mut y).unwrap();
    y
}

#[test]
fn exl3_experts_agree_with_the_gguf_of_the_same_layer() {
    let Ok(subset) = std::env::var("EXL3_SUBSET") else { return eprintln!("SKIPPED, not passed: EXL3_SUBSET is not set") };
    let model = std::env::var("SHADOWZERO_MODEL").unwrap_or_else(|_| {
        "J:/Models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf".into()
    });
    if !Path::new(&model).is_file() || !Path::new(&subset).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {model} or no subset at {subset}");
    }
    let st = SafeTensors::open(&[subset.into()]).unwrap();
    let g = Gguf::open(Path::new(&model)).unwrap();
    for layer in 0..3 {
        for (proj, gname) in [("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")] {
            for e in 0..2 {
                let lin = Exl3Linear::load(&st, &format!("model.language_model.layers.{layer}.mlp.experts.{e}.{proj}")).unwrap();
                let w = lin.reconstruct(); // [in][out]
                let gg = gguf_expert(&g, &format!("blk.{layer}.{gname}.weight"), e); // [out][in]
                let (ni, no) = (lin.in_features, lin.out_features);
                assert_eq!(gg.len(), ni * no, "shape of {gname}");
                // cosine per output column, and over the whole matrix
                let mut cols = vec![];
                for o in 0..no.min(64) {
                    let a: Vec<f32> = (0..ni).map(|i| w[i * no + o]).collect();
                    cols.push(cosine(&a, &gg[o * ni..(o + 1) * ni]));
                }
                let mean: f64 = cols.iter().sum::<f64>() / cols.len() as f64;
                let wt: Vec<f32> = (0..no).flat_map(|o| (0..ni).map(move |i| (i, o))).map(|(i, o)| w[i * no + o]).collect();
                let all = cosine(&wt, &gg);
                eprintln!("layer {layer} expert {e} {proj} (K={}, cb {}, {}x{}, GGUF {}): cosine whole {all:.4}, mean of 64 columns {mean:.4}, min {:.4}",
                          lin.bits, lin.cb, ni, no, g.tensor(&format!("blk.{layer}.{gname}.weight")).unwrap().ty.name(), cols.iter().cloned().fold(1.0, f64::min));
                // the GGUF side is itself a 2-3 bit approximation of a (fine-tuned) model: Q2_0's four levels are the coarsest of them
                let floor = if g.tensor(&format!("blk.{layer}.{gname}.weight")).unwrap().ty.name() == "Q2_0" { 0.6 } else { 0.85 };
                assert!(all > floor, "layer {layer} expert {e} {proj}: cosine {all} - the decode does not match the GGUF");
            }
        }
    }
}

/// The 4-bit dense tensors: shared experts (`[in][out]` against GGUF `ffn_*_shexp` `[out][in]`) and the linear-attention
/// projections of layer 0, whose GGUF row order may differ (reported, not asserted).
#[test]
fn exl3_dense_tensors_agree_with_the_gguf_of_the_same_layer() {
    let Ok(subset) = std::env::var("EXL3_SUBSET") else { return eprintln!("SKIPPED, not passed: EXL3_SUBSET is not set") };
    let model = std::env::var("SHADOWZERO_MODEL").unwrap_or_else(|_| {
        "J:/Models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf".into()
    });
    if !Path::new(&model).is_file() || !Path::new(&subset).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {model} or no subset at {subset}");
    }
    let st = SafeTensors::open(&[subset.into()]).unwrap();
    let g = Gguf::open(Path::new(&model)).unwrap();
    let mut cases: Vec<(String, String, bool)> = vec![];
    for layer in 0..3 {
        for (proj, gname) in [("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")] {
            cases.push((format!("model.language_model.layers.{layer}.mlp.shared_expert.{proj}"), format!("blk.{layer}.{gname}.weight"), true));
        }
    }
    for (hf, gg) in [("in_proj_qkv", "attn_qkv"), ("in_proj_z", "attn_gate"), ("out_proj", "ssm_out")] {
        cases.push((format!("model.language_model.layers.0.linear_attn.{hf}"), format!("blk.0.{gg}.weight"), false));
    }
    for (hf, gname, must) in cases {
        let lin = Exl3Linear::load(&st, &hf).unwrap();
        let w = lin.reconstruct();
        let Some(t) = g.tensor(&gname) else { eprintln!("{hf}: GGUF has no {gname}"); continue };
        let (ni, no) = (lin.in_features, lin.out_features);
        if t.dims[0] as usize != ni || t.dims[1] as usize != no {
            eprintln!("{hf} {ni}x{no}: GGUF {gname} is {:?}, skipped", t.dims);
            continue;
        }
        let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
        f.seek(SeekFrom::Start(t.offset)).unwrap();
        let mut b = vec![0u8; t.nbytes() as usize];
        f.read_exact(&mut b).unwrap();
        let mut y = vec![0f32; ni * no];
        dequantize(t.ty, &b, &mut y).unwrap();
        let wt: Vec<f32> = (0..no).flat_map(|o| (0..ni).map(move |i| (i, o))).map(|(i, o)| w[i * no + o]).collect();
        let all = cosine(&wt, &y);
        eprintln!("{hf} (K={}, {ni}x{no}, GGUF {} {gname}): cosine {all:.4}", lin.bits, t.ty.name());
        if must {
            assert!(all > 0.85, "{hf}: cosine {all}");
        }
    }
}

/// A recorded decode of the two tensors of `eval/fixtures/exl3_sample.safetensors` (real ones, EXL3 2.05 bpw branch):
/// needs no model.  The numbers were taken from the decoder after it matched the GGUF (the tests above), so this
/// guards the decode against changes; it is not an independent reference.
#[test]
fn the_fixture_decodes_to_the_recorded_values() {
    let st = SafeTensors::open(&[concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/fixtures/exl3_sample.safetensors").into()]).unwrap();
    for (prefix, bits, sumsq, first) in [
        ("model.language_model.layers.0.mlp.experts.0.gate_proj", 2usize, 287.702278f64, [-0.042833105f32, -0.0065446747, 0.029719023, -0.030945573]),
        ("model.language_model.layers.0.mlp.shared_expert.gate_proj", 4usize, 161.198363f64, [-0.0071623405f32, 0.00092289475, -0.0021615676, -0.0026322217]),
    ] {
        let lin = Exl3Linear::load(&st, prefix).unwrap();
        assert_eq!((lin.bits, lin.cb, lin.in_features, lin.out_features), (bits, 2, 2560, 640));
        let w = lin.reconstruct();
        let ss: f64 = w.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        assert!((ss - sumsq).abs() < 1e-3 * sumsq, "{prefix}: sum of squares {ss}, recorded {sumsq}");
        for (g, r) in w[..4].iter().zip(first) {
            assert!((g - r).abs() < 1e-6, "{prefix}: first values {:?}, recorded {first:?}", &w[..4]);
        }
    }
}
