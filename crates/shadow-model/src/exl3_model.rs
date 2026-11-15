//! An EXL3 checkpoint (a Hugging Face repo directory: `config.json`, `tokenizer.json`, `model-*.safetensors`,
//! `ngram_embedding.safetensors`) presented to the engine as a GGUF: the same tensor names, dims and key-values the
//! GGUF of this model has (`Gguf::from_parts`), so `Config::from_gguf`, `Tokenizer::from_gguf` and everything that finds a
//! tensor by name work unchanged.  What a GGUF has and a checkpoint does not is made on demand:
//!
//! * **file-backed** tensors are a range of a safetensors file: an EXL3 projection is one range (the repos store `suh`,
//!   `svh`, the marker and the trellis back to back, checked at open), `embed_tokens` is BF16 as stored;
//! * **computed** tensors are built from the checkpoint when read (`Exl3Extra::bytes`, the rules of `exl3_map.rs`): norms
//!   with the +1 folded in, `-exp(A_log)`, the value-head reorder, the indexer's q/k split, f16 matrices as BF16;
//! * **experts** are `n_expert` separate projections per matrix in the checkpoint; `Exl3Extra::experts` holds for each
//!   `(matrix, expert)` the file and offset of its blob, replacing the GGUF's `offset + e * slice` (experts are not evenly
//!   strided in the files);
//! * the **n-gram table** (`per_layer_token_embd` in the GGUF) is 128 shards of trellis rings in its own format (`NgramTable`).
//!
//! The matrix types are `EXL3_K` (`GgmlType::exl3`); the bytes of one are a blob (`exl3::blob_len`).

use crate::exl3::{blob_len, bits_from_last_dim, Exl3Linear};
use crate::exl3_map::{materialize, source_of, MapCfg, Source, Tensor, Transform};
use crate::gguf::{GgmlType, Gguf, Shard, TensorInfo, Value};
use crate::json::Json;
use crate::safetensors::SafeTensors;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The hashed n-gram memory of the PLE: `shards` files of `rows_per_shard` rows of `1 + 160 K / 16` words.
#[derive(Debug)]
pub struct NgramTable {
    /// (file index in `Exl3Extra::st.files`, byte offset of the shard's data) for each shard.
    pub shards: Vec<(usize, u64)>,
    pub rows_per_shard: usize,
    pub bits: usize,
    pub dim: usize,
    /// `[head][dim]` bias added to a row of that hash head.
    pub head_bias: Vec<f32>,
    pub head_offsets: Vec<i64>,
    pub head_vocab: Vec<i64>,
}

impl NgramTable {
    pub fn words_per_row(&self) -> usize {
        1 + self.dim * self.bits / 16
    }

    /// (file, byte offset, byte length) of row `row`.
    pub fn row_range(&self, row: i64) -> (usize, u64, usize) {
        let (sh, local) = (row as usize / self.rows_per_shard, row as usize % self.rows_per_shard);
        let wb = self.words_per_row() * 2;
        (self.shards[sh].0, self.shards[sh].1 + (local * wb) as u64, wb)
    }

    /// The decoded row from its bytes.
    pub fn decode(&self, row: i64, bytes: &[u8]) -> Vec<f32> {
        let words: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let head = (0..self.head_offsets.len()).find(|&h| row >= self.head_offsets[h] && row < self.head_offsets[h] + self.head_vocab[h]).unwrap_or(0);
        let mut out = vec![0f32; self.dim];
        crate::exl3::ngram_row(&words, self.bits, &self.head_bias[head * self.dim..(head + 1) * self.dim], &mut out);
        out
    }
}

/// What the virtual GGUF cannot say by itself.
#[derive(Debug)]
pub struct Exl3Extra {
    pub dir: PathBuf,
    pub st: SafeTensors,
    pub map: MapCfg,
    /// GGUF tensor name -> how to build its bytes, for the tensors that are not a file range.
    pub computed: HashMap<String, Source>,
    /// GGUF expert matrix name -> per expert (file index, byte offset) of its blob.
    pub experts: HashMap<String, Vec<(usize, u64)>>,
    pub ngram: NgramTable,
}

pub struct Exl3Model {
    pub gguf: Gguf,
    pub extra: Exl3Extra,
}

fn reversed_dims(shape: &[usize], numel: usize) -> Vec<u64> {
    let mut d: Vec<u64> = shape.iter().rev().copied().filter(|&x| x != 1).map(|x| x as u64).collect();
    if d.is_empty() {
        d.push(numel as u64);
    }
    d
}

/// The four tensors of a projection, if they sit back to back in one file in the order `suh`, `svh`, marker, `trellis`:
/// (file, offset of `suh`, length through the trellis).
fn contiguous(st: &SafeTensors, prefix: &str) -> Option<(usize, u64, u64)> {
    let g = |s: &str| st.get(&format!("{prefix}.{s}"));
    let (suh, svh, tr) = (g("suh")?, g("svh")?, g("trellis")?);
    let mk = g("mul1").or_else(|| g("mcg"))?;
    let same = suh.file == svh.file && svh.file == mk.file && mk.file == tr.file;
    (same && suh.end == svh.start && svh.end == mk.start && mk.end == tr.start).then_some((suh.file, suh.start, tr.end - suh.start))
}

/// (in, out, K, half, codebook) of a projection.
fn linear_shape(st: &SafeTensors, prefix: &str) -> Result<(usize, usize, usize, bool, u8), String> {
    let t = st.get(&format!("{prefix}.trellis")).ok_or_else(|| format!("no tensor {prefix}.trellis"))?;
    if t.shape.len() != 3 {
        return Err(format!("{prefix}.trellis: shape {:?}", t.shape));
    }
    let (bits, half) = bits_from_last_dim(t.shape[2]).ok_or_else(|| format!("{prefix}: {} words a tile is not a bit rate", t.shape[2]))?;
    let cb = if st.get(&format!("{prefix}.mcg")).is_some() { 1 } else if st.get(&format!("{prefix}.mul1")).is_some() { 2 } else { 0 };
    if half && cb != 2 {
        return Err(format!("{prefix}: a half-integer bit rate needs the mul1 codebook"));
    }
    Ok((t.shape[0] * 16, t.shape[1] * 16, bits, half, cb))
}

fn i64s(st: &SafeTensors, name: &str) -> Result<Vec<i64>, String> {
    let b = st.read(name).map_err(|e| e.to_string())?;
    Ok(b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
}

fn kv_u(v: u64) -> Value {
    Value::U32(v as u32)
}

/// Opens a checkpoint directory.
pub fn open(dir: &Path) -> Result<Exl3Model, String> {
    let cfg_text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?;
    let root = Json::parse(&cfg_text).map_err(|e| format!("config.json: {e}"))?;
    let tc = root.get("text_config").ok_or("config.json has no text_config")?;
    let num = |k: &str| tc.get(k).and_then(Json::as_u64).ok_or(format!("config.json text_config.{k}"));

    // model shards in name order, then the n-gram file; `mtp_*` patches and anything else are not part of the model here
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors")))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("{}: no model-*.safetensors", dir.display()));
    }
    let ngram_path = dir.join("ngram_embedding.safetensors");
    files.push(ngram_path.clone());
    let st = SafeTensors::open(&files).map_err(|e| e.to_string())?;
    let ngram_file = st.files.len() - 1;

    let n_layer = num("num_hidden_layers")? as usize;
    let hidden = num("hidden_size")? as usize;
    let n_expert = num("num_experts")? as usize;
    let interval = num("full_attention_interval")? as usize;
    let layer_types: Vec<String> = tc.get("layer_types").and_then(Json::as_arr).ok_or("layer_types")?.iter()
        .map(|j| j.as_str().unwrap_or("").to_string()).collect();
    let attn = |l: usize| layer_types.get(l).map(|t| t == "full_attention").unwrap_or(l % interval == interval - 1);
    for l in 0..n_layer {
        if attn(l) != (l % interval == interval - 1) {
            return Err(format!("layer {l}: layer_types disagrees with full_attention_interval {interval}"));
        }
    }
    let ple_layer = tc.get("ple_layer_ids").and_then(Json::as_arr).and_then(|a| a.first()).and_then(Json::as_u64).ok_or("ple_layer_ids")? as usize - 1;
    let map = MapCfg {
        ssm_k_heads: num("linear_num_key_heads")? as usize,
        ssm_v_heads: num("linear_num_value_heads")? as usize,
        ssm_head_dim: num("linear_value_head_dim")? as usize,
        idx_heads: num("indexer_n_heads")? as usize,
        idx_dim: num("indexer_head_dim")? as usize,
        ple_layer,
        attn_interval: interval,
    };

    // ---- the n-gram table
    let ng = |s: &str| format!("model.language_model.layers.{ple_layer}.ple.ple_embedding.ngram_embedding.{s}");
    let head_offsets = i64s(&st, &ng("head_offsets"))?;
    let head_vocab = i64s(&st, &ng("head_vocab_sizes"))?;
    let multipliers = i64s(&st, &ng("layer_multipliers"))?;
    let bias_t = st.get(&ng("head_bias")).ok_or("no head_bias")?;
    let (heads, dim) = (bias_t.shape[0], bias_t.shape[1]);
    let head_bias = st.read_f32(&ng("head_bias")).map_err(|e| e.to_string())?;
    let mut shards = Vec::new();
    let mut rows_per_shard = 0;
    let mut words = 0;
    for i in 0.. {
        let Some(t) = st.get(&ng(&format!("shard_{i}.trellis"))) else { break };
        assert_eq!(t.file, ngram_file);
        rows_per_shard = t.shape[0];
        words = t.shape[1];
        shards.push((t.file, t.start));
    }
    if shards.is_empty() {
        return Err("the n-gram file has no shard_N.trellis".into());
    }
    let nbits = (words - 1) * 16 / dim;
    if 1 + dim * nbits / 16 != words {
        return Err(format!("n-gram rows of {words} words do not hold {dim} values at an integer bit rate"));
    }
    let ngram = NgramTable { shards, rows_per_shard, bits: nbits, dim, head_bias, head_offsets: head_offsets.clone(), head_vocab: head_vocab.clone() };
    let _ = heads;

    // ---- tensors
    let mut tensors: Vec<TensorInfo> = Vec::new();
    let mut computed: HashMap<String, Source> = HashMap::new();
    let mut experts: HashMap<String, Vec<(usize, u64)>> = HashMap::new();
    let mut names: Vec<String> = vec!["token_embd.weight".into(), "output.weight".into(), "output_hc_norm.weight".into(), "output_hc_down.weight".into(), "output_hc_up.weight".into()];
    for l in 0..n_layer {
        let mixer: &[&str] = if attn(l) {
            &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
              "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"]
        } else {
            &["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"]
        };
        let common = ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                      "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                      "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                      "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"];
        for t in mixer.iter().chain(common.iter()) {
            names.push(format!("blk.{l}.{t}"));
        }
        if l == ple_layer {
            for t in ["ple_key", "ple_value", "ple_norm_key", "ple_norm_query", "ple_norm_conv", "ple_conv1d"] {
                names.push(format!("blk.{l}.{t}.weight"));
            }
        }
    }
    let new_info = |name: &str, dims: Vec<u64>, ty: GgmlType, shard: usize, offset: u64, nbytes: u64| TensorInfo {
        name: name.to_string(), dims, ty, shard, offset, nbytes_layout: Some(nbytes), nbytes_span: nbytes,
    };
    for name in &names {
        let src = source_of(name, &map).ok_or_else(|| format!("{name}: no source in the EXL3 map"))?;
        match &src {
            Source::Linear { prefix, perm_out, perm_in, out_blocks } => {
                let (inf, mut outf, bits, half, cb) = linear_shape(&st, prefix)?;
                if let Some((a, b)) = out_blocks {
                    outf = (b - a) * 128;
                }
                let ty = GgmlType::from_exl3_rate(bits, half, cb);
                let len = blob_len(inf, outf, (bits, half)) as u64;
                let direct = perm_out.is_none() && perm_in.is_none() && out_blocks.is_none();
                match contiguous(&st, prefix).filter(|_| direct) {
                    Some((file, off, l)) if l == len => tensors.push(new_info(name, vec![inf as u64, outf as u64], ty, file, off, len)),
                    _ => {
                        tensors.push(new_info(name, vec![inf as u64, outf as u64], ty, 0, 0, len));
                        computed.insert(name.clone(), src);
                    }
                }
            }
            Source::Plain { name: hf, transform, perm } => {
                let t = st.get(hf).ok_or_else(|| format!("{name}: no tensor {hf}"))?;
                let dims = reversed_dims(&t.shape, t.numel());
                let n = t.numel() as u64;
                match transform {
                    Transform::Raw => {
                        let ty = match t.dtype.as_str() { "BF16" => GgmlType::from_id(30), "F16" => GgmlType::from_id(1), "F32" => GgmlType::from_id(0), d => return Err(format!("{hf}: raw {d}")) };
                        tensors.push(new_info(name, dims, ty, t.file, t.start, t.end - t.start));
                    }
                    Transform::ToBf16 => {
                        tensors.push(new_info(name, dims, GgmlType::from_id(30), 0, 0, 2 * n));
                        computed.insert(name.clone(), src);
                    }
                    _ => {
                        let _ = perm;
                        tensors.push(new_info(name, dims, GgmlType::from_id(0), 0, 0, 4 * n));
                        computed.insert(name.clone(), src);
                    }
                }
            }
            Source::Expert { prefix } => {
                let p0 = prefix.replace("{e}", "0");
                let (inf, outf, bits, half, cb) = linear_shape(&st, &p0)?;
                let len = blob_len(inf, outf, (bits, half));
                let mut table = Vec::with_capacity(n_expert);
                for e in 0..n_expert {
                    let pe = prefix.replace("{e}", &e.to_string());
                    let (i2, o2, b2, h2, c2) = linear_shape(&st, &pe)?;
                    if (i2, o2, b2, h2, c2) != (inf, outf, bits, half, cb) {
                        return Err(format!("{pe}: {i2} x {o2} K={b2} (half {h2}) codebook {c2}, but expert 0 of the layer is {inf} x {outf} K={bits} (half {half}) codebook {cb}"));
                    }
                    let (file, off, l) = contiguous(&st, &pe).ok_or_else(|| format!("{pe}: its tensors are not one range of a file"))?;
                    if l as usize != len {
                        return Err(format!("{pe}: {l} bytes, expected {len}"));
                    }
                    table.push((file, off));
                }
                tensors.push(new_info(name, vec![inf as u64, outf as u64, n_expert as u64], GgmlType::from_exl3_rate(bits, half, cb), 0, 0, (len * n_expert) as u64));
                experts.insert(name.clone(), table);
            }
        }
    }
    tensors.push(new_info("per_layer_token_embd.weight", vec![dim as u64, (ngram.rows_per_shard * ngram.shards.len()) as u64], GgmlType::Unknown(0xE3),
                          ngram_file, 0, (ngram.rows_per_shard * ngram.shards.len() * ngram.words_per_row() * 2) as u64));

    // ---- key-values
    let mut kv: BTreeMap<String, Value> = BTreeMap::new();
    let mut put = |k: &str, v: Value| {
        kv.insert(k.to_string(), v);
    };
    put("general.architecture", Value::Str("qwen4exp".into()));
    put("general.name", Value::Str(dir.file_name().and_then(|n| n.to_str()).unwrap_or("exl3").to_string()));
    let head_dim = num("head_dim")?;
    let vh = num("linear_num_value_heads")?;
    let a = |k: &str| format!("qwen4exp.{k}");
    put(&a("block_count"), kv_u(n_layer as u64));
    put(&a("context_length"), kv_u(num("max_position_embeddings")?));
    put(&a("embedding_length"), kv_u(hidden as u64));
    put(&a("attention.head_count"), kv_u(num("num_attention_heads")?));
    put(&a("attention.head_count_kv"), kv_u(num("num_key_value_heads")?));
    put(&a("attention.layer_norm_rms_epsilon"), Value::F32(tc.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32));
    put(&a("expert_count"), kv_u(n_expert as u64));
    put(&a("expert_used_count"), kv_u(num("num_experts_per_tok")?));
    put(&a("attention.key_length"), kv_u(head_dim));
    put(&a("attention.value_length"), kv_u(head_dim));
    put(&a("expert_feed_forward_length"), kv_u(num("moe_intermediate_size")?));
    put(&a("expert_shared_feed_forward_length"), kv_u(num("shared_expert_intermediate_size")?));
    put(&a("ssm.conv_kernel"), kv_u(num("linear_conv_kernel_dim")?));
    put(&a("ssm.state_size"), kv_u(num("linear_key_head_dim")?));
    put(&a("ssm.group_count"), kv_u(num("linear_num_key_heads")?));
    put(&a("ssm.time_step_rank"), kv_u(vh));
    put(&a("ssm.inner_size"), kv_u(vh * num("linear_value_head_dim")?));
    put(&a("attention.recurrent_layers"), Value::Array((0..n_layer).map(|l| Value::Bool(!attn(l))).collect()));
    put(&a("full_attention_interval"), kv_u(interval as u64));
    put(&a("hyper_connection.count"), kv_u(num("hc_count")?));
    put(&a("hyper_connection.low_rank"), kv_u(num("hc_lowrank")?));
    put(&a("attention.indexer.head_count"), kv_u(num("indexer_n_heads")?));
    put(&a("attention.indexer.key_length"), kv_u(num("indexer_head_dim")?));
    put(&a("attention.indexer.top_k"), kv_u(num("indexer_budget")?));
    let ratio = num("indexer_compress_ratio")?;
    put(&a("attention.compress_ratios"), Value::Array((0..n_layer).map(|l| Value::U32(if attn(l) { ratio as u32 } else { 0 })).collect()));
    put(&a("ple.layers"), Value::Array(vec![Value::U32(ple_layer as u32)]));
    put(&a("ple.ngram_size"), kv_u(num("ngram_size")?));
    put(&a("ple.heads_per_ngram"), kv_u(num("heads_per_ngram")?));
    put(&a("ple.conv_kernel"), kv_u(num("ple_conv_kernel_size")?));
    put(&a("ple.eos_token_id"), kv_u(num("eos_token_id")?));
    put(&a("embedding_length_per_layer_input"), kv_u(dim as u64));
    put(&a("ple.layer_multipliers"), Value::Array(multipliers.iter().map(|&v| Value::I64(v)).collect()));
    put(&a("ple.head_offsets"), Value::Array(head_offsets.iter().map(|&v| Value::I64(v)).collect()));
    put(&a("ple.head_vocab_sizes"), Value::Array(head_vocab.iter().map(|&v| Value::I64(v)).collect()));
    let rp = tc.get("rope_parameters").ok_or("rope_parameters")?;
    put(&a("rope.freq_base"), Value::F64(rp.get("rope_theta").and_then(Json::as_f64).ok_or("rope_theta")?));
    let rot = (head_dim as f64 * tc.get("partial_rotary_factor").and_then(Json::as_f64).unwrap_or(1.0)).round() as u32;
    put(&a("rope.dimension_count"), Value::U32(rot));
    if let Ok(g) = std::fs::read_to_string(dir.join("generation_config.json")).map_err(|e| e.to_string()).and_then(|t| Json::parse(&t)) {
        if let Some(v) = g.get("top_k").and_then(Json::as_u64) { put("general.sampling.top_k", Value::I32(v as i32)) }
        if let Some(v) = g.get("top_p").and_then(Json::as_f64) { put("general.sampling.top_p", Value::F32(v as f32)) }
        if let Some(v) = g.get("temperature").and_then(Json::as_f64) { put("general.sampling.temp", Value::F32(v as f32)) }
    }
    drop(put);
    tokenizer_kv(dir, tc, &mut kv)?;

    let shards: Vec<Shard> = st.files.iter().map(|p| {
        let len = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        Shard { path: p.clone(), version: 3, file_len: len, data_start: 0, alignment: 1 }
    }).collect();
    let gguf = Gguf::from_parts(shards, kv, tensors);
    Ok(Exl3Model { gguf, extra: Exl3Extra { dir: dir.to_path_buf(), st, map, computed, experts, ngram } })
}

/// The `tokenizer.ggml.*` and chat template keys, from `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`, in
/// the shape llama.cpp's converter writes them (the vocabulary padded to the model's `vocab_size` with `[PADn]` tokens of type
/// 5; added tokens control (3) when special, else user-defined (4); the FIM and repo tokens of Qwen control).
fn tokenizer_kv(dir: &Path, tc: &Json, kv: &mut BTreeMap<String, Value>) -> Result<(), String> {
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).map_err(|e| format!("{f}: {e}"));
    let tj = Json::parse(&read("tokenizer.json")?).map_err(|e| format!("tokenizer.json: {e}"))?;
    let tcfg = Json::parse(&read("tokenizer_config.json")?).map_err(|e| format!("tokenizer_config.json: {e}"))?;
    let model = tj.get("model").ok_or("tokenizer.json has no model")?;
    let vocab = model.get("vocab").and_then(Json::as_obj).ok_or("tokenizer.json model.vocab")?;
    let size = tc.get("vocab_size").and_then(Json::as_u64).ok_or("vocab_size")? as usize;
    let mut tokens: Vec<Option<String>> = vec![None; size];
    let mut types = vec![5i32; size];
    for (t, id) in vocab {
        let id = id.as_u64().ok_or("vocab id")? as usize;
        if id >= size {
            return Err(format!("token id {id} beyond vocab_size {size}"));
        }
        tokens[id] = Some(t.clone());
        types[id] = 1;
    }
    const CONTROL_BY_NAME: [&str; 6] = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"];
    for a in tj.get("added_tokens").and_then(Json::as_arr).ok_or("added_tokens")? {
        let id = a.get("id").and_then(Json::as_u64).ok_or("added id")? as usize;
        let content = a.get("content").and_then(Json::as_str).ok_or("added content")?;
        let special = a.get("special").and_then(Json::as_bool).unwrap_or(false);
        tokens[id] = Some(content.to_string());
        types[id] = if special || CONTROL_BY_NAME.contains(&content) { 3 } else { 4 };
    }
    let tokens: Vec<Value> = tokens.into_iter().enumerate().map(|(i, t)| Value::Str(t.unwrap_or_else(|| format!("[PAD{i}]")))).collect();
    let merges: Vec<Value> = model.get("merges").and_then(Json::as_arr).ok_or("model.merges")?.iter().map(|m| match m {
        Json::Str(s) => Ok(Value::Str(s.clone())),
        Json::Arr(p) if p.len() == 2 => Ok(Value::Str(format!("{} {}", p[0].as_str().unwrap_or(""), p[1].as_str().unwrap_or("")))),
        _ => Err("a merge that is neither a string nor a pair"),
    }).collect::<Result<_, _>>()?;
    let id_of = |t: &str| -> Option<u32> { tokens.iter().position(|v| v.as_str() == Some(t)).map(|i| i as u32) };
    kv.insert("tokenizer.ggml.model".into(), Value::Str("gpt2".into()));
    kv.insert("tokenizer.ggml.pre".into(), Value::Str("qwen35".into()));
    let bos = tc.get("bos_token_id").and_then(Json::as_u64).map(|v| v as u32);
    let eos = tcfg.get("eos_token").and_then(Json::as_str).and_then(id_of).or(tc.get("eos_token_id").and_then(Json::as_u64).map(|v| v as u32));
    let pad = tcfg.get("pad_token").and_then(Json::as_str).and_then(id_of);
    if let Some(v) = bos { kv.insert("tokenizer.ggml.bos_token_id".into(), Value::U32(v)); }
    if let Some(v) = eos { kv.insert("tokenizer.ggml.eos_token_id".into(), Value::U32(v)); }
    if let Some(v) = pad { kv.insert("tokenizer.ggml.padding_token_id".into(), Value::U32(v)); }
    kv.insert("tokenizer.ggml.tokens".into(), Value::Array(tokens));
    kv.insert("tokenizer.ggml.token_type".into(), Value::Array(types.into_iter().map(Value::I32).collect()));
    kv.insert("tokenizer.ggml.merges".into(), Value::Array(merges));
    if let Ok(t) = read("chat_template.jinja") {
        kv.insert("tokenizer.chat_template".into(), Value::Str(t));
    } else if let Some(t) = tcfg.get("chat_template").and_then(Json::as_str) {
        kv.insert("tokenizer.chat_template".into(), Value::Str(t.to_string()));
    }
    Ok(())
}

impl Exl3Extra {
    /// The bytes of a computed tensor, in the layout of its `TensorInfo`.
    pub fn bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        let src = self.computed.get(name).ok_or_else(|| format!("{name}: not a computed tensor"))?;
        match materialize(&self.st, src, &self.map).map_err(|e| format!("{name}: {e}"))? {
            Tensor::Exl3(l) => Ok(l.to_blob()),
            Tensor::F32 { data, .. } => Ok(data.iter().flat_map(|v| v.to_le_bytes()).collect()),
            Tensor::Bf16 { data, .. } => Ok(data.iter().flat_map(|v| v.to_le_bytes()).collect()),
        }
    }
}

/// An `Exl3Linear` straight from its prefix in the checkpoint (tests).
pub fn load_linear(st: &SafeTensors, prefix: &str) -> Result<Exl3Linear, String> {
    Exl3Linear::load(st, prefix).map_err(|e| e.to_string())
}
