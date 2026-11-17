//! The EXL3 -> GGUF tensor map (`exl3_map.rs`) against a GGUF of the same model: every dense tensor of a GatedDeltaNet
//! layer (0), an attention layer (3), the PLE (layer 1) and the head mixer, built from the checkpoint and compared with
//! the GGUF's by cosine.  A wrong permutation, a missing +1 or a wrong name shows up as a cosine near 0 (or near 1 only by
//! accident for a norm).  The two files quantize the same weights differently (and the GGUF is of a fine-tune), so the
//! bar is agreement, not identity.
//!
//! Needs `EXL3_LAYERS` (a safetensors file with those tensors: `python tools/fetch_exl3_subset.py REPO BRANCH OUT
//! '^model\.language_model\.layers\.0\.(?!mlp\.experts\.)' '^model\.language_model\.layers\.3\.(?!mlp\.experts\.)'
//! '^model\.language_model\.layers\.1\.ple\.' '^model\.language_model\.hyper_connection_mixer\.'`) and a GGUF
//! (`SHADOWZERO_MODEL`); skips otherwise.

use shadowzero_model::exl3_map::{materialize, source_of, tiled_from_grouped, MapCfg, Source, Tensor, Transform};
use shadowzero_model::gguf::Gguf;
use shadowzero_model::quant::dequantize;
use shadowzero_model::safetensors::SafeTensors;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const GGUF: &str = "J:/Models/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += *x as f64 * *x as f64;
        bb += *y as f64 * *y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn gguf_f32(g: &Gguf, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    let t = g.tensor(name)?;
    let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
    f.seek(SeekFrom::Start(t.offset)).unwrap();
    let mut b = vec![0u8; t.nbytes() as usize];
    f.read_exact(&mut b).unwrap();
    let n: usize = t.dims.iter().map(|&d| d as usize).product();
    let mut y = vec![0f32; n];
    dequantize(t.ty, &b, &mut y).unwrap();
    Some((y, t.dims.iter().map(|&d| d as usize).collect()))
}

/// A materialized tensor in the GGUF's memory order ([out][in] for a matrix).
fn as_gguf_order(t: &Tensor) -> (Vec<f32>, Vec<usize>) {
    match t {
        Tensor::F32 { data, dims } => (data.clone(), dims.clone()),
        Tensor::Bf16 { data, dims } => (data.iter().map(|w| shadowzero_model::half::bf16_to_f32(*w)).collect(), dims.clone()),
        Tensor::Exl3(l) => {
            let w = l.reconstruct(); // [in][out]
            let (ni, no) = (l.in_features, l.out_features);
            let mut y = vec![0f32; ni * no];
            for i in 0..ni {
                for o in 0..no {
                    y[o * ni + i] = w[i * no + o];
                }
            }
            (y, vec![ni, no])
        }
    }
}

fn setup() -> Option<(SafeTensors, Gguf)> {
    let Ok(layers) = std::env::var("EXL3_LAYERS") else { eprintln!("SKIPPED, not passed: EXL3_LAYERS is not set"); return None };
    let model = std::env::var("SHADOWZERO_MODEL").unwrap_or_else(|_| GGUF.into());
    if !Path::new(&model).is_file() || !Path::new(&layers).is_file() {
        eprintln!("SKIPPED, not passed: no model at {model} or no tensors at {layers}");
        return None;
    }
    Some((SafeTensors::open(&[layers.into()]).unwrap(), Gguf::open(Path::new(&model)).unwrap()))
}

#[test]
fn every_dense_tensor_of_the_sample_layers_matches_the_gguf() {
    let Some((st, g)) = setup() else { return };
    let c = MapCfg::default();
    let mut worst_lin = 1f64;
    let mut worst_plain = 1f64;
    let mut n = 0;
    for t in &g.tensors {
        let name = &t.name;
        let sample = name.starts_with("blk.0.") || name.starts_with("blk.3.") || name.starts_with("blk.1.ple") || name.starts_with("output_hc");
        if !sample || name.contains("_exps") {
            continue;
        }
        let src = source_of(name, &c).unwrap_or_else(|| panic!("{name}: no source in the map"));
        let built = match &src {
            // taken as stored: a file range in the engine, read here and widened
            Source::Plain { name: hf, transform: Transform::Raw, .. } => {
                let shape = &st.get(hf).unwrap().shape;
                let mut dims: Vec<usize> = shape.iter().rev().copied().filter(|&d| d != 1).collect();
                if dims.is_empty() {
                    dims.push(shape.iter().product());
                }
                Tensor::F32 { data: st.read_f32(hf).unwrap(), dims }
            }
            _ => materialize(&st, &src, &c).unwrap_or_else(|e| panic!("{name}: {e}")),
        };
        let (a, adims) = as_gguf_order(&built);
        let (b, bdims) = gguf_f32(&g, name).unwrap();
        assert_eq!(adims, bdims, "{name}: dims");
        let cs = cosine(&a, &b);
        let plain = matches!(src, Source::Plain { .. });
        let kind = match &src { Source::Linear { .. } => format!("exl3 K={}", match &built { Tensor::Exl3(l) => l.bits, _ => 0 }), _ => "plain".into() };
        let (ma, mb) = (a.iter().map(|v| *v as f64).sum::<f64>() / a.len() as f64, b.iter().map(|v| *v as f64).sum::<f64>() / b.len() as f64);
        eprintln!("{name:40} {kind:10} {:?} cosine {cs:.4}  mean {ma:+.4} / {mb:+.4}  GGUF {}", adims, t.ty.name());
        // Q2_0 (four levels) is the coarsest GGUF type: agreement with it is capped around 0.75 for any correct decode
        let q2 = t.ty.name() == "Q2_0";
        if q2 { assert!(cs > 0.6, "{name}: cosine {cs} against Q2_0") } else if plain { worst_plain = worst_plain.min(cs) } else { worst_lin = worst_lin.min(cs) }
        n += 1;
    }
    eprintln!("{n} tensors; worst cosine: plain {worst_plain:.4}, exl3 {worst_lin:.4}");
    assert!(worst_plain > 0.95, "a plain tensor disagrees: worst cosine {worst_plain}");
    assert!(worst_lin > 0.85, "an EXL3 tensor disagrees: worst cosine {worst_lin}");
}

/// The value-head order, found from the data rather than assumed: for each GGUF (tiled) head of `attn_gate`, the
/// checkpoint head whose 128 x 2560 block agrees best.
#[test]
fn the_value_head_permutation_is_what_the_data_says() {
    let Some((st, g)) = setup() else { return };
    let lin = shadowzero_model::exl3::Exl3Linear::load(&st, "model.language_model.layers.0.linear_attn.in_proj_z").unwrap();
    let w = lin.reconstruct(); // [in][out], out = 48 heads x 128
    let (ni, no) = (lin.in_features, lin.out_features);
    let (gg, _) = gguf_f32(&g, "blk.0.attn_gate.weight").unwrap(); // [out][in]
    let head_hf = |h: usize| -> Vec<f32> { (0..128).flat_map(|c| (0..ni).map(move |i| (i, c))).map(|(i, c)| w[i * no + h * 128 + c]).collect() };
    let heads_hf: Vec<Vec<f32>> = (0..48).map(head_hf).collect();
    let mut found = vec![0usize; 48];
    for t in 0..48 {
        let mut block = Vec::with_capacity(128 * ni);
        for c in 0..128 {
            block.extend_from_slice(&gg[(t * 128 + c) * ni..(t * 128 + c + 1) * ni]);
        }
        // the HF block is [c][i] too (built that way above)
        let (best, cs) = heads_hf.iter().enumerate().map(|(h, b)| (h, cosine(b, &block))).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        found[t] = best;
        eprintln!("GGUF value head {t:2} = checkpoint head {best:2} (cosine {cs:.3})");
    }
    assert_eq!(found, tiled_from_grouped(16, 48), "the tiled-from-grouped formula does not match the data");
}
