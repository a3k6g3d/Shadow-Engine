//! The tensors of an EXL3 checkpoint of Qwen3.8-Flash-Next (Hugging Face names and layouts, `safetensors.rs`), mapped onto
//! the GGUF names and layouts the forward pass reads (docs/ARCH.md).  Three things differ between the two layouts:
//!
//! 1. **Value heads are tiled in the GGUF, grouped in the checkpoint** (ARCH.md, "the value heads are stored tiled").  The
//!    checkpoint's value head `k * 3 + v` belongs to key head `k`; llama.cpp's converter reorders every per-value-head
//!    tensor so that head `v * 16 + k` does, and `delta_net` pairs value head `h` with key head `h % 16`.  The reordering
//!    moves whole 128-wide heads, and 128 is the EXL3 Hadamard block, so for a trellis it is a permutation of 8-tile
//!    groups and of 128-chunks of `suh` / `svh`: exact, no decode (`Exl3Linear::permute_out_blocks`).  Affected: the v
//!    part of `attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_a`, `ssm_dt.bias`, the v channels of `ssm_conv1d`,
//!    and the input side of `ssm_out`.
//! 2. **Norm weights carry +1 in the GGUF** (zero-centred RMSNorm, the converter folds the 1 in), except the DeltaNet's
//!    `ssm_norm`; and `ssm_a` is `-exp(A_log)`.
//! 3. **The indexer's q and k projections are one tensor in the checkpoint** (`index_qk_proj`, 512 + 128 outputs) and
//!    two in the GGUF; 512 is a multiple of 128, so the split is a slice of tile columns (`slice_out_blocks`).
//!
//! `source_of` gives, for a GGUF tensor name, where it comes from and how; `materialize` builds it.  Checked against
//! the GGUF of the same model tensor by tensor in `tests/exl3_map_vs_gguf.rs`.

use crate::exl3::Exl3Linear;
use crate::safetensors::{Error, SafeTensors};

/// Shapes the map needs; the defaults are Qwen3.8-Flash-Next's.
#[derive(Clone, Debug)]
pub struct MapCfg {
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub ssm_head_dim: usize,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub ple_layer: usize,
    pub attn_interval: usize,
}

impl Default for MapCfg {
    fn default() -> Self {
        MapCfg { ssm_k_heads: 16, ssm_v_heads: 48, ssm_head_dim: 128, idx_heads: 4, idx_dim: 128, ple_layer: 1, attn_interval: 4 }
    }
}

/// For tiled value-head position `t` (the GGUF order), the grouped position (the checkpoint order) it holds.
pub fn tiled_from_grouped(kh: usize, vh: usize) -> Vec<usize> {
    let per = vh / kh;
    (0..vh).map(|t| (t % kh) * per + t / kh).collect()
}

/// A permutation of equal-sized units: unit `first + i` of the result is unit `first + map[i]` of the source.
#[derive(Clone, Debug)]
pub struct UnitPerm {
    pub first: usize,
    pub map: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// Taken as stored (a file range): `embed_tokens` (BF16), `ple.conv1d` (F16).
    Raw,
    /// To f32, values unchanged (the router, the hyper-connection norms' companions...).
    Identity,
    /// To BF16, values unchanged (f16 -> bf16 is exact for values that came from BF16): the matrices the GGUF keeps as BF16.
    ToBf16,
    /// The GGUF stores `w + 1` (zero-centred RMSNorm weights).
    PlusOne,
    /// The GGUF stores `-exp(w)` (`ssm_a` from `A_log`).
    NegExp,
}

#[derive(Clone, Debug)]
pub enum Source {
    /// An EXL3 linear layer `{prefix}.{trellis,suh,svh,mul1}`; permutations are of 128-blocks (8 tiles) on the output
    /// or input side; `out_blocks` keeps a range of output blocks only.
    Linear { prefix: String, perm_out: Option<UnitPerm>, perm_in: Option<UnitPerm>, out_blocks: Option<(usize, usize)> },
    /// A plain (f16 / bf16 / f32) tensor, read as f32 in its stored (row-major) order, which is the GGUF's memory order
    /// too; `unit` elements per permuted unit.
    Plain { name: String, transform: Transform, perm: Option<(usize, UnitPerm)> },
    /// A routed-expert matrix: one EXL3 linear per expert, `{prefix}` with `{e}` for the expert index.
    Expert { prefix: String },
}

/// Where GGUF tensor `name` comes from in the checkpoint, or None for a name the map does not know.
pub fn source_of(name: &str, c: &MapCfg) -> Option<Source> {
    let lin = |p: String| Source::Linear { prefix: p, perm_out: None, perm_in: None, out_blocks: None };
    let plain = |n: String, t: Transform| Source::Plain { name: n, transform: t, perm: None };
    let vperm = || UnitPerm { first: 0, map: tiled_from_grouped(c.ssm_k_heads, c.ssm_v_heads) };
    let lm = "model.language_model";
    if let Some(rest) = name.strip_prefix("blk.") {
        let (l, t) = rest.split_once('.')?;
        let l: usize = l.parse().ok()?;
        let p = format!("{lm}.layers.{l}");
        let attn = l % c.attn_interval == c.attn_interval - 1;
        let d = c.ssm_head_dim;
        let kv_blocks = 2 * c.ssm_k_heads * d / 128; // q and k blocks before the v part of qkv
        return Some(match t {
            // ---- GatedDeltaNet
            "attn_qkv.weight" if !attn => Source::Linear {
                prefix: format!("{p}.linear_attn.in_proj_qkv"), perm_in: None, out_blocks: None,
                perm_out: Some(UnitPerm { first: kv_blocks, map: tiled_from_grouped(c.ssm_k_heads, c.ssm_v_heads) }),
            },
            "attn_gate.weight" if !attn => Source::Linear { prefix: format!("{p}.linear_attn.in_proj_z"), perm_out: Some(vperm()), perm_in: None, out_blocks: None },
            "ssm_beta.weight" => Source::Plain { name: format!("{p}.linear_attn.in_proj_b.weight"), transform: Transform::Identity, perm: Some((2560usize.max(1) * 0 + hidden_of(&p), vperm())) },
            "ssm_alpha.weight" => Source::Plain { name: format!("{p}.linear_attn.in_proj_a.weight"), transform: Transform::Identity, perm: Some((hidden_of(&p), vperm())) },
            "ssm_conv1d.weight" => Source::Plain {
                name: format!("{p}.linear_attn.conv1d.weight"), transform: Transform::Identity,
                // channels in 128-blocks of (taps) elements each; the v channels start after q and k
                perm: Some((0, UnitPerm { first: kv_blocks, map: tiled_from_grouped(c.ssm_k_heads, c.ssm_v_heads) })),
            },
            "ssm_a" => Source::Plain { name: format!("{p}.linear_attn.A_log"), transform: Transform::NegExp, perm: Some((1, vperm())) },
            "ssm_dt.bias" => Source::Plain { name: format!("{p}.linear_attn.dt_bias"), transform: Transform::Identity, perm: Some((1, vperm())) },
            "ssm_norm.weight" => plain(format!("{p}.linear_attn.norm.weight"), Transform::Identity),
            "ssm_out.weight" => Source::Linear { prefix: format!("{p}.linear_attn.out_proj"), perm_in: Some(vperm()), perm_out: None, out_blocks: None },
            // ---- gated attention with the indexer
            "attn_q.weight" if attn => lin(format!("{p}.self_attn.q_proj")),
            "attn_k.weight" if attn => lin(format!("{p}.self_attn.k_proj")),
            "attn_v.weight" if attn => lin(format!("{p}.self_attn.v_proj")),
            "attn_output.weight" if attn => lin(format!("{p}.self_attn.o_proj")),
            "attn_q_norm.weight" => plain(format!("{p}.self_attn.q_norm.weight"), Transform::PlusOne),
            "attn_k_norm.weight" => plain(format!("{p}.self_attn.k_norm.weight"), Transform::PlusOne),
            "indexer.q_proj.weight" => Source::Linear {
                prefix: format!("{p}.self_attn.indexer.index_qk_proj"), perm_out: None, perm_in: None,
                out_blocks: Some((0, c.idx_heads * c.idx_dim / 128)),
            },
            "indexer.k_proj.weight" => Source::Linear {
                prefix: format!("{p}.self_attn.indexer.index_qk_proj"), perm_out: None, perm_in: None,
                out_blocks: Some((c.idx_heads * c.idx_dim / 128, (c.idx_heads + 1) * c.idx_dim / 128)),
            },
            "indexer.q_norm.weight" => plain(format!("{p}.self_attn.indexer.q_layernorm.weight"), Transform::PlusOne),
            "indexer.k_norm.weight" => plain(format!("{p}.self_attn.indexer.k_layernorm.weight"), Transform::PlusOne),
            // ---- MoE
            "ffn_gate_inp.weight" => plain(format!("{p}.mlp.gate.weight"), Transform::Identity),
            "ffn_gate_inp_shexp.weight" => plain(format!("{p}.mlp.shared_expert_gate.weight"), Transform::Identity),
            "ffn_gate_shexp.weight" => lin(format!("{p}.mlp.shared_expert.gate_proj")),
            "ffn_up_shexp.weight" => lin(format!("{p}.mlp.shared_expert.up_proj")),
            "ffn_down_shexp.weight" => lin(format!("{p}.mlp.shared_expert.down_proj")),
            "ffn_gate_exps.weight" => Source::Expert { prefix: format!("{p}.mlp.experts.{{e}}.gate_proj") },
            "ffn_up_exps.weight" => Source::Expert { prefix: format!("{p}.mlp.experts.{{e}}.up_proj") },
            "ffn_down_exps.weight" => Source::Expert { prefix: format!("{p}.mlp.experts.{{e}}.down_proj") },
            // ---- hyper connections
            "hc_attn_norm.weight" => plain(format!("{p}.attn_hyper_connection.hc_norm.weight"), Transform::PlusOne),
            "hc_attn_down.weight" => plain(format!("{p}.attn_hyper_connection.input_mix_weight_down.weight"), Transform::ToBf16),
            "hc_attn_up.weight" => plain(format!("{p}.attn_hyper_connection.input_mix_weight_up.weight"), Transform::ToBf16),
            "hc_attn_inject.weight" => plain(format!("{p}.attn_hyper_connection.block_inject_weight.weight"), Transform::ToBf16),
            "hc_ffn_norm.weight" => plain(format!("{p}.mlp_hyper_connection.hc_norm.weight"), Transform::PlusOne),
            "hc_ffn_down.weight" => plain(format!("{p}.mlp_hyper_connection.input_mix_weight_down.weight"), Transform::ToBf16),
            "hc_ffn_up.weight" => plain(format!("{p}.mlp_hyper_connection.input_mix_weight_up.weight"), Transform::ToBf16),
            "hc_ffn_inject.weight" => plain(format!("{p}.mlp_hyper_connection.block_inject_weight.weight"), Transform::ToBf16),
            // ---- PLE (one layer)
            "ple_key.weight" if l == c.ple_layer => plain(format!("{p}.ple.key_proj.weight"), Transform::ToBf16),
            "ple_value.weight" if l == c.ple_layer => plain(format!("{p}.ple.value_proj.weight"), Transform::ToBf16),
            "ple_norm_key.weight" if l == c.ple_layer => plain(format!("{p}.ple.norm_key.weight"), Transform::PlusOne),
            "ple_norm_query.weight" if l == c.ple_layer => plain(format!("{p}.ple.norm_query.weight"), Transform::PlusOne),
            "ple_norm_conv.weight" if l == c.ple_layer => plain(format!("{p}.ple.norm_conv.weight"), Transform::PlusOne),
            "ple_conv1d.weight" if l == c.ple_layer => plain(format!("{p}.ple.conv1d.weight"), Transform::Raw),
            _ => return None,
        });
    }
    Some(match name {
        "token_embd.weight" => plain(format!("{lm}.embed_tokens.weight"), Transform::Raw),
        "output.weight" => lin("lm_head".into()),
        "output_hc_norm.weight" => plain(format!("{lm}.hyper_connection_mixer.hc_norm.weight"), Transform::PlusOne),
        "output_hc_down.weight" => plain(format!("{lm}.hyper_connection_mixer.input_mix_weight_down.weight"), Transform::ToBf16),
        "output_hc_up.weight" => plain(format!("{lm}.hyper_connection_mixer.input_mix_weight_up.weight"), Transform::ToBf16),
        _ => return None,
    })
}

/// The hidden size (the unit of a per-head row of `ssm_beta` / `ssm_alpha`): the model's; a per-model value would come
/// from its config, this family has one.
fn hidden_of(_layer_prefix: &str) -> usize {
    2560
}

/// A tensor built from the checkpoint in the GGUF's layout.
#[derive(Clone, Debug)]
pub enum Tensor {
    Exl3(Exl3Linear),
    /// BF16 words in the GGUF's memory order.
    Bf16 { data: Vec<u16>, dims: Vec<usize> },
    /// f32 in the GGUF's memory order (`dims` in ggml order: dims[0] is the row length).
    F32 { data: Vec<f32>, dims: Vec<usize> },
}

/// Reorders `unit`-element units of `data`: unit `first + i` of the result is unit `first + map[i]` of the input.  `unit`
/// 0 means "derive from the length": `data.len() / total_units` is not known here, so callers pass it (conv taps).
pub fn permute_units(data: &mut [f32], unit: usize, p: &UnitPerm) {
    let src = data.to_vec();
    for (i, &m) in p.map.iter().enumerate() {
        let (d, s) = ((p.first + i) * unit, (p.first + m) * unit);
        data[d..d + unit].copy_from_slice(&src[s..s + unit]);
    }
}

/// Builds GGUF tensor `name` (a dense one: not an `Expert`) from the checkpoint.
pub fn materialize(st: &SafeTensors, src: &Source, cfg: &MapCfg) -> Result<Tensor, Error> {
    match src {
        Source::Linear { prefix, perm_out, perm_in, out_blocks } => {
            let mut lin = Exl3Linear::load(st, prefix)?;
            if let Some((a, b)) = out_blocks {
                lin = lin.slice_out_blocks(*a, *b - *a);
            }
            if let Some(p) = perm_out {
                lin.permute_out_blocks(p.first, &p.map);
            }
            if let Some(p) = perm_in {
                lin.permute_in_blocks(p.first, &p.map);
            }
            Ok(Tensor::Exl3(lin))
        }
        Source::Plain { name, transform, perm } => {
            let t = st.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?.clone();
            let mut data = st.read_f32(name)?;
            match transform {
                Transform::Raw => return Err(Error(format!("{name}: a raw tensor is read from its file range, not built"))),
                Transform::Identity | Transform::ToBf16 => {}
                Transform::PlusOne => data.iter_mut().for_each(|v| *v += 1.0),
                Transform::NegExp => data.iter_mut().for_each(|v| *v = -((*v as f64).exp()) as f32),
            }
            if let Some((unit, p)) = perm {
                // a conv weight [channels, 1, taps]: a unit is 128 channels of `taps` values
                let unit = if *unit == 0 { 128 * t.shape.last().copied().unwrap_or(1) } else { *unit };
                let _ = cfg;
                permute_units(&mut data, unit, p);
            }
            // ggml dims: the stored row-major shape reversed, singleton dims dropped
            let mut dims: Vec<usize> = t.shape.iter().rev().copied().filter(|&d| d != 1).collect();
            if dims.is_empty() {
                dims.push(data.len());
            }
            if *transform == Transform::ToBf16 {
                return Ok(Tensor::Bf16 { data: data.iter().map(|v| crate::half::f32_to_bf16(*v)).collect(), dims });
            }
            Ok(Tensor::F32 { data, dims })
        }
        Source::Expert { prefix } => Err(Error(format!("{prefix}: an expert tensor is loaded per expert, not materialized"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_order_pairs_value_head_h_with_key_head_h_mod_16() {
        let m = tiled_from_grouped(16, 48);
        assert_eq!(m.len(), 48);
        let mut seen = vec![false; 48];
        for (t, &g) in m.iter().enumerate() {
            assert_eq!(g / 3, t % 16, "tiled {t} holds grouped {g}, whose key head is {}", g / 3);
            assert!(!std::mem::replace(&mut seen[g], true));
        }
        assert_eq!(&m[..4], &[0, 3, 6, 9]);
        assert_eq!(m[16], 1);
    }

    #[test]
    fn every_forward_pass_name_has_a_source() {
        let c = MapCfg::default();
        for n in ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_conv1d", "ssm_norm", "ssm_out", "ffn_gate_inp", "ffn_gate_shexp", "ffn_up_shexp",
                  "ffn_down_shexp", "ffn_gate_inp_shexp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "hc_attn_norm", "hc_attn_down", "hc_attn_up",
                  "hc_attn_inject", "hc_ffn_norm", "hc_ffn_down", "hc_ffn_up", "hc_ffn_inject"] {
            assert!(source_of(&format!("blk.0.{n}.weight"), &c).is_some(), "{n}");
        }
        for n in ["ssm_a", "ssm_dt.bias"] {
            assert!(source_of(&format!("blk.0.{n}"), &c).is_some(), "{n}");
        }
        for n in ["attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm", "indexer.q_proj", "indexer.k_proj", "indexer.q_norm", "indexer.k_norm"] {
            assert!(source_of(&format!("blk.3.{n}.weight"), &c).is_some(), "{n}");
        }
        for n in ["ple_key", "ple_value", "ple_norm_key", "ple_norm_query", "ple_norm_conv", "ple_conv1d"] {
            assert!(source_of(&format!("blk.1.{n}.weight"), &c).is_some(), "{n}");
        }
        for n in ["token_embd.weight", "output.weight", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight"] {
            assert!(source_of(n, &c).is_some(), "{n}");
        }
        assert!(source_of("blk.0.attn_q.weight", &c).is_none(), "a J layer has no attn_q");
        assert!(source_of("blk.3.attn_qkv.weight", &c).is_none(), "a K layer has no attn_qkv");
    }
}
