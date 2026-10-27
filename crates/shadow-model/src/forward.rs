//! Stage 1: the CPU reference forward pass, written from docs/ARCH.md, one token at a time.
//!
//! It is the yardstick, not the engine: f64 accumulation in every dot product, every weight dequantized on the fly
//! from its stored bytes, no shortcuts.  Dense tensors (attention, DeltaNet, hyper-connections, router, shared expert,
//! embedding, head: ~4.1 GiB as stored) are read into RAM once; the routed experts (46.8 GiB) and the n-gram table
//! (26.8 GiB) are read from the file per use, by positional reads, so the process never holds more than ~5 GiB.
//!
//! Limits, said here so nobody trips on them: text only (positions are plain 1-D, which is what the multimodal RoPE
//! reduces to for text); at most 2,048 positions, because past that the sparse indexer starts to select keys and it is
//! not implemented yet (below 2,048 it selects every key, so attention here is exact).

use crate::gguf::{block_index, Gguf, TensorInfo, Value};
use crate::quant::dequantize;
use std::collections::HashMap;
use std::fs::File;

pub const MAX_POSITIONS: usize = 2048;

/// The shapes, from the file's key-values (ARCH.md, "Shapes").
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub hidden: usize,
    pub streams: usize,
    pub hc_rank: usize,
    pub vocab: usize,
    pub eps: f32,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub rope_theta: f64,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_ff: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub ssm_head_dim: usize,
    pub conv_k: usize,
    pub attn_interval: usize,
    pub ple_layer: usize,
    pub ple_ngram: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_head_dim: usize,
    pub ple_conv_k: usize,
    pub ple_eos: u32,
    pub ple_multipliers: Vec<i64>,
    pub ple_vocab: Vec<i64>,
    pub ple_offsets: Vec<i64>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Config, String> {
        let u = |k: &str| g.arch_key(k).and_then(Value::as_u64).map(|v| v as usize).ok_or(format!("missing {k}"));
        let ints = |k: &str| -> Result<Vec<i64>, String> {
            g.arch_key(k).and_then(Value::as_array).ok_or(format!("missing {k}"))?.iter()
                .map(|v| match *v {
                    Value::I64(x) => Ok(x),
                    Value::U64(x) => Ok(x as i64),
                    Value::I32(x) => Ok(x as i64),
                    Value::U32(x) => Ok(x as i64),
                    _ => Err(format!("{k}: not integers")),
                })
                .collect()
        };
        let hidden = u("embedding_length")?;
        let ple_layers = ints("ple.layers")?;
        if ple_layers.len() != 1 {
            return Err(format!("ple.layers {ple_layers:?}: one PLE layer is implemented"));
        }
        let ssm_v_heads = u("ssm.time_step_rank")?;
        let ssm_inner = u("ssm.inner_size")?;
        Ok(Config {
            n_layer: u("block_count")?,
            hidden,
            streams: u("hyper_connection.count")?,
            hc_rank: u("hyper_connection.low_rank")?,
            vocab: g.tensor("output.weight").map(|t| t.dims[1] as usize).ok_or("no output.weight")?,
            eps: g.arch_key("attention.layer_norm_rms_epsilon").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            n_head: u("attention.head_count")?,
            n_kv_head: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rope_dims: u("rope.dimension_count")?,
            rope_theta: g.arch_key("rope.freq_base").and_then(Value::as_f64).unwrap_or(1e7),
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            expert_ff: u("expert_feed_forward_length")?,
            ssm_k_heads: u("ssm.group_count")?,
            ssm_v_heads,
            ssm_head_dim: ssm_inner / ssm_v_heads,
            conv_k: u("ssm.conv_kernel")?,
            attn_interval: u("full_attention_interval")?,
            ple_layer: ple_layers[0] as usize,
            ple_ngram: u("ple.ngram_size")?,
            ple_heads_per_ngram: u("ple.heads_per_ngram")?,
            ple_head_dim: u("embedding_length_per_layer_input")?,
            ple_conv_k: u("ple.conv_kernel")?,
            ple_eos: u("ple.eos_token_id")? as u32,
            ple_multipliers: ints("ple.layer_multipliers")?,
            ple_vocab: ints("ple.head_vocab_sizes")?,
            ple_offsets: ints("ple.head_offsets")?,
        })
    }

    pub fn is_attention(&self, l: usize) -> bool {
        l % self.attn_interval == self.attn_interval - 1
    }
}

/// Positional reads, safe from several threads at once.
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = f.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, offset)
    }
}

/// A matrix as stored: ggml `[in, out]`, so each of the `out` rows is `in` weights, contiguous.
struct Mat<'a> {
    t: &'a TensorInfo,
    bytes: &'a [u8],
}

impl Mat<'_> {
    fn rows(&self) -> usize {
        self.bytes.len() / self.row_bytes()
    }
    fn cols(&self) -> usize {
        self.t.dims[0] as usize
    }
    fn row_bytes(&self) -> usize {
        let (be, bb) = self.t.ty.block().unwrap();
        (self.t.dims[0] / be * bb) as usize
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// y = W x, rows split over threads; each row dequantized into scratch and dotted in f64.
fn matvec(m: &Mat, x: &[f32]) -> Vec<f32> {
    let (rows, cols, rb) = (m.rows(), m.cols(), m.row_bytes());
    assert_eq!(x.len(), cols, "{}: input {} for {} columns", m.t.name, x.len(), cols);
    let mut y = vec![0f32; rows];
    let per = rows.div_ceil(threads()).max(16);
    std::thread::scope(|s| {
        for (ci, chunk) in y.chunks_mut(per).enumerate() {
            s.spawn(move || {
                let mut w = vec![0f32; cols];
                for (k, out) in chunk.iter_mut().enumerate() {
                    let r = ci * per + k;
                    dequantize(m.t.ty, &m.bytes[r * rb..(r + 1) * rb], &mut w).unwrap();
                    *out = w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32;
                }
            });
        }
    });
    y
}

/// Where the matrix-vector products run.  Everything else in the forward pass is small and stays on the CPU.
pub trait Backend {
    /// y = W x for a matrix as stored (ggml `[cols, rows]`, whole rows contiguous).  `key` names a tensor whose bytes
    /// never change, so a backend may keep them (a GPU keeps the dense weights resident); `None` is a one-off, such as
    /// one expert's slice.
    fn matvec(&self, t: &TensorInfo, bytes: &[u8], key: Option<&str>, x: &[f32]) -> Vec<f32>;
}

/// The reference: every row dequantized and dotted in f64, rows split over the CPU's threads.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn matvec(&self, t: &TensorInfo, bytes: &[u8], _key: Option<&str>, x: &[f32]) -> Vec<f32> {
        matvec(&Mat { t, bytes }, x)
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps as f64).sqrt();
    x.iter().zip(w).map(|(&v, &g)| (v as f64 * r) as f32 * g).collect()
}

/// RMSNorm applied to each `group`-wide slice on its own (the hyper-connection and PLE norms).
fn rmsnorm_groups(x: &[f32], w: &[f32], group: usize, eps: f32) -> Vec<f32> {
    x.chunks(group).zip(w.chunks(group)).flat_map(|(xs, ws)| rmsnorm(xs, ws, eps)).collect()
}

fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { ((x as f64).exp().ln_1p()) as f32 }
}

struct DeltaState {
    s: Vec<f32>,    // v_heads x k_dim x v_dim
    conv: Vec<f32>, // (conv_k - 1) x channels, oldest first
}

struct AttnCache {
    k: Vec<f32>, // positions x kv_heads x head_dim
    v: Vec<f32>,
}

pub struct Model {
    pub g: Gguf,
    pub cfg: Config,
    files: Vec<File>,
    dense: HashMap<String, Vec<u8>>,
}

impl Model {
    /// Reads every tensor that is not a routed expert or the n-gram table into RAM (~4.1 GiB as stored).
    pub fn load(g: Gguf) -> Result<Model, String> {
        let cfg = Config::from_gguf(&g)?;
        let files = g.shards.iter().map(|s| File::open(&s.path).map_err(|e| format!("{}: {e}", s.path.display())))
            .collect::<Result<Vec<_>, _>>()?;
        let mut dense = HashMap::new();
        for t in &g.tensors {
            if t.name.contains("_exps") || t.name == "per_layer_token_embd.weight" {
                continue;
            }
            let mut b = vec![0u8; t.nbytes() as usize];
            read_at(&files[t.shard], &mut b, t.offset).map_err(|e| format!("{}: {e}", t.name))?;
            dense.insert(t.name.clone(), b);
        }
        Ok(Model { g, cfg, files, dense })
    }

    pub fn dense_bytes(&self) -> usize {
        self.dense.values().map(Vec::len).sum()
    }

    fn mat(&self, name: &str) -> Mat<'_> {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("no tensor {name}"));
        Mat { t, bytes: &self.dense[name] }
    }

    /// A small tensor, whole, as f32.
    fn vec(&self, name: &str) -> Vec<f32> {
        let m = self.mat(name);
        let n = m.bytes.len() / m.row_bytes() * m.cols();
        let mut y = vec![0f32; n];
        dequantize(m.t.ty, m.bytes, &mut y).unwrap();
        y
    }

    /// One expert's slice of a `[in, out, n_expert]` tensor, read from the file.
    fn expert(&self, name: &str, e: usize) -> (TensorInfo, Vec<u8>) {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("no tensor {name}"));
        let per = (t.nbytes() / t.dims[2]) as usize;
        let mut b = vec![0u8; per];
        read_at(&self.files[t.shard], &mut b, t.offset + (e * per) as u64).unwrap();
        let mut one = t.clone();
        one.dims = vec![t.dims[0], t.dims[1]];
        (one, b)
    }

    /// One row of the n-gram table (ple_head_dim weights).
    fn ple_row(&self, row: i64) -> Vec<f32> {
        let t = self.g.tensor("per_layer_token_embd.weight").unwrap();
        let (be, bb) = t.ty.block().unwrap();
        let rb = (t.dims[0] / be * bb) as usize;
        let mut b = vec![0u8; rb];
        read_at(&self.files[t.shard], &mut b, t.offset + row as u64 * rb as u64).unwrap();
        let mut y = vec![0f32; t.dims[0] as usize];
        dequantize(t.ty, &b, &mut y).unwrap();
        y
    }
}

/// The running state of one sequence: every layer's recurrent state or KV cache, and the token history.
pub struct Session<'m> {
    m: &'m Model,
    be: &'m dyn Backend,
    delta: Vec<Option<DeltaState>>,
    attn: Vec<Option<AttnCache>>,
    ple_conv: Vec<f32>, // the last (conv_k - 1) * ngram inputs, oldest first
    tokens: Vec<u32>,
}

impl<'m> Session<'m> {
    pub fn new(m: &'m Model, be: &'m dyn Backend) -> Session<'m> {
        let c = &m.cfg;
        let ch = c.ssm_k_heads * c.ssm_head_dim * 2 + c.ssm_v_heads * c.ssm_head_dim;
        let delta = (0..c.n_layer).map(|l| (!c.is_attention(l)).then(|| DeltaState {
            s: vec![0f32; c.ssm_v_heads * c.ssm_head_dim * c.ssm_head_dim],
            conv: vec![0f32; (c.conv_k - 1) * ch],
        })).collect();
        let attn = (0..c.n_layer).map(|l| c.is_attention(l).then(|| AttnCache { k: vec![], v: vec![] })).collect();
        let ple_conv = vec![0f32; (c.ple_conv_k - 1) * c.ple_ngram * c.streams * c.hidden];
        Session { m, be, delta, attn, ple_conv, tokens: vec![] }
    }

    /// y = W x for a dense tensor, through the backend.
    fn mv(&self, name: &str, x: &[f32]) -> Vec<f32> {
        let mt = self.m.mat(name);
        self.be.matvec(mt.t, mt.bytes, Some(name), x)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Feeds one token and returns the logits for the next.
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        assert!(self.tokens.len() < MAX_POSITIONS, "the indexer is not implemented: at most {MAX_POSITIONS} positions");
        self.tokens.push(token);
        let m = self.m;
        let c = &m.cfg;
        let (h, cs) = (c.hidden, c.streams);

        let emb = {
            let e = m.mat("token_embd.weight");
            let rb = e.row_bytes();
            let mut x = vec![0f32; h];
            dequantize(e.t.ty, &e.bytes[token as usize * rb..(token as usize + 1) * rb], &mut x).unwrap();
            x
        };
        let mut r: Vec<f32> = (0..cs).flat_map(|_| emb.iter().copied()).collect();

        for l in 0..c.n_layer {
            if l == c.ple_layer {
                let add = self.ple(&r, l);
                r.iter_mut().zip(&add).for_each(|(a, b)| *a += b);
            }
            let (u, g) = self.hc_read(&r, &format!("blk.{l}.hc_attn"), true);
            let y = if c.is_attention(l) { self.attention(l, &u) } else { self.delta_net(l, &u) };
            hc_write(&mut r, &g, &y, h);
            let (u, g) = self.hc_read(&r, &format!("blk.{l}.hc_ffn"), true);
            let y = self.moe(l, &u);
            hc_write(&mut r, &g, &y, h);
        }
        let (u, _) = self.hc_read(&r, "output_hc", false);
        self.mv("output.weight", &u)
    }

    /// A hyper-connection read: the sublayer's input and, with `inject`, the per-stream gains.
    fn hc_read(&self, r: &[f32], p: &str, inject: bool) -> (Vec<f32>, Vec<f32>) {
        let m = self.m;
        let c = &m.cfg;
        let (h, cs) = (c.hidden, c.streams);
        let rn = rmsnorm_groups(r, &m.vec(&format!("{p}_norm.weight")), h, c.eps);
        let down: Vec<f32> = self.mv(&format!("{p}_down.weight"), &rn).iter().map(|&v| silu(v / cs as f32)).collect();
        let mix: Vec<f32> = self.mv(&format!("{p}_up.weight"), &down).iter().map(|&v| sigmoid(v)).collect();
        let mut u = vec![0f32; h];
        for s in 0..cs {
            for i in 0..h {
                u[i] += mix[s * h + i] * rn[s * h + i];
            }
        }
        u.iter_mut().for_each(|v| *v /= cs as f32);
        let g = if inject {
            self.mv(&format!("{p}_inject.weight"), &rn).iter().map(|&v| 2.0 * sigmoid(v / cs as f32)).collect()
        } else {
            vec![]
        };
        (u, g)
    }

    fn delta_net(&mut self, l: usize, u: &[f32]) -> Vec<f32> {
        let m = self.m;
        let c = &m.cfg;
        let p = format!("blk.{l}");
        let (kh, vh, d) = (c.ssm_k_heads, c.ssm_v_heads, c.ssm_head_dim);
        let (kd, vd) = (kh * d, vh * d);
        let ch = 2 * kd + vd;
        let qkv = self.mv(&format!("{p}.attn_qkv.weight"), u);
        let z = self.mv(&format!("{p}.attn_gate.weight"), u);
        let b = self.mv(&format!("{p}.ssm_beta.weight"), u);
        let a = self.mv(&format!("{p}.ssm_alpha.weight"), u);
        let conv_w = m.vec(&format!("{p}.ssm_conv1d.weight")); // ch x conv_k, tap k multiplies x[t - (conv_k-1) + k]
        let st = self.delta[l].as_mut().unwrap();
        let k = c.conv_k;
        let mut x = vec![0f32; ch];
        for i in 0..ch {
            let mut acc = conv_w[i * k + (k - 1)] as f64 * qkv[i] as f64;
            for t in 0..k - 1 {
                acc += conv_w[i * k + t] as f64 * st.conv[t * ch + i] as f64;
            }
            x[i] = silu(acc as f32);
        }
        st.conv.copy_within(ch.., 0);
        st.conv[(k - 2) * ch..].copy_from_slice(&qkv);

        let l2 = |v: &mut [f32]| {
            let n = 1.0 / (v.iter().map(|&a| a as f64 * a as f64).sum::<f64>() + 1e-6).sqrt();
            v.iter_mut().for_each(|a| *a = (*a as f64 * n) as f32);
        };
        let (q, rest) = x.split_at_mut(kd);
        let (kk, v) = rest.split_at_mut(kd);
        q.chunks_mut(d).for_each(|hq| {
            l2(hq);
            hq.iter_mut().for_each(|a| *a /= (d as f32).sqrt());
        });
        kk.chunks_mut(d).for_each(l2);
        let a_log = m.vec(&format!("{p}.ssm_a"));
        let dt = m.vec(&format!("{p}.ssm_dt.bias"));
        let norm_w = m.vec(&format!("{p}.ssm_norm.weight"));
        let mut o = vec![0f32; vd];
        for hd in 0..vh {
            let beta = sigmoid(b[hd]);
            let decay = ((a_log[hd] as f64) * softplus(a[hd] + dt[hd]) as f64).exp() as f32;
            // The value heads are stored TILED, not grouped (ARCH.md): the converter reorders them so value head
            // v*kh + k belongs to key head k, for ggml's broadcast.  So the key head is hd % kh, not hd / (vh/kh).
            let kx = hd % kh;
            let (qh, kh_) = (&q[kx * d..(kx + 1) * d], &kk[kx * d..(kx + 1) * d]);
            let vh_ = &v[hd * d..(hd + 1) * d];
            let s = &mut st.s[hd * d * d..(hd + 1) * d * d]; // s[i * d + j]: key index i, value index j
            s.iter_mut().for_each(|e| *e *= decay);
            let mut delta = vec![0f32; d];
            for j in 0..d {
                let mut mem = 0f64;
                for i in 0..d {
                    mem += s[i * d + j] as f64 * kh_[i] as f64;
                }
                delta[j] = ((vh_[j] as f64 - mem) * beta as f64) as f32;
            }
            for i in 0..d {
                for j in 0..d {
                    s[i * d + j] += kh_[i] * delta[j];
                }
            }
            let oh = &mut o[hd * d..(hd + 1) * d];
            for j in 0..d {
                let mut acc = 0f64;
                for i in 0..d {
                    acc += s[i * d + j] as f64 * qh[i] as f64;
                }
                oh[j] = acc as f32;
            }
            let normed = rmsnorm(oh, &norm_w, c.eps);
            for j in 0..d {
                oh[j] = normed[j] * sigmoid(z[hd * d + j]);
            }
        }
        self.mv(&format!("{p}.ssm_out.weight"), &o)
    }

    fn attention(&mut self, l: usize, u: &[f32]) -> Vec<f32> {
        let m = self.m;
        let c = &m.cfg;
        let p = format!("blk.{l}");
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let qg = self.mv(&format!("{p}.attn_q.weight"), u);
        let k = self.mv(&format!("{p}.attn_k.weight"), u);
        let v = self.mv(&format!("{p}.attn_v.weight"), u);
        let (qn, kn) = (m.vec(&format!("{p}.attn_q_norm.weight")), m.vec(&format!("{p}.attn_k_norm.weight")));
        let pos = (self.tokens.len() - 1) as f64;
        let rope = |x: &mut [f32]| {
            let half = c.rope_dims / 2;
            for i in 0..half {
                let f = pos / c.rope_theta.powf(2.0 * i as f64 / c.rope_dims as f64);
                let (cs, sn) = (f.cos(), f.sin());
                let (a, b) = (x[i] as f64, x[i + half] as f64);
                x[i] = (a * cs - b * sn) as f32;
                x[i + half] = (b * cs + a * sn) as f32;
            }
        };
        let mut q = vec![0f32; nh * hd];
        let mut gate = vec![0f32; nh * hd];
        for h in 0..nh {
            let mut qh = rmsnorm(&qg[h * 2 * hd..h * 2 * hd + hd], &qn, c.eps);
            rope(&mut qh);
            q[h * hd..(h + 1) * hd].copy_from_slice(&qh);
            gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[h * 2 * hd + hd..(h + 1) * 2 * hd]);
        }
        let cache = self.attn[l].as_mut().unwrap();
        for h in 0..nkv {
            let mut kh = rmsnorm(&k[h * hd..(h + 1) * hd], &kn, c.eps);
            rope(&mut kh);
            cache.k.extend(kh);
        }
        cache.v.extend_from_slice(&v);
        let n = cache.k.len() / (nkv * hd);
        let mut o = vec![0f32; nh * hd];
        let scale = 1.0 / (hd as f64).sqrt();
        for h in 0..nh {
            let g = h / (nh / nkv);
            let qh = &q[h * hd..(h + 1) * hd];
            let scores: Vec<f64> = (0..n)
                .map(|t| {
                    let kt = &cache.k[(t * nkv + g) * hd..(t * nkv + g + 1) * hd];
                    qh.iter().zip(kt).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale
                })
                .collect();
            let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
            let ex: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = ex.iter().sum();
            for j in 0..hd {
                let acc: f64 = ex.iter().enumerate().map(|(t, &e)| e * cache.v[(t * nkv + g) * hd + j] as f64).sum();
                o[h * hd + j] = (acc / z) as f32 * sigmoid(gate[h * hd + j]);
            }
        }
        self.mv(&format!("{p}.attn_output.weight"), &o)
    }

    fn moe(&self, l: usize, u: &[f32]) -> Vec<f32> {
        let m = self.m;
        let c = &m.cfg;
        let p = format!("blk.{l}");
        let logits = self.mv(&format!("{p}.ffn_gate_inp.weight"), u);
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f64> = logits.iter().map(|&v| ((v - mx) as f64).exp()).collect();
        let z: f64 = ex.iter().sum();
        let mut idx: Vec<usize> = (0..c.n_expert).collect();
        idx.sort_by(|&a, &b| ex[b].partial_cmp(&ex[a]).unwrap().then(a.cmp(&b)));
        let top = &idx[..c.n_expert_used];
        let tz: f64 = top.iter().map(|&e| ex[e] / z).sum();
        let mut y = vec![0f64; c.hidden];
        let swiglu = |gate: Vec<f32>, up: Vec<f32>| -> Vec<f32> { gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect() };
        for &e in top {
            let w = (ex[e] / z) / tz;
            let (tg, bg) = m.expert(&format!("{p}.ffn_gate_exps.weight"), e);
            let (tu, bu) = m.expert(&format!("{p}.ffn_up_exps.weight"), e);
            let (td, bd) = m.expert(&format!("{p}.ffn_down_exps.weight"), e);
            let act = swiglu(self.be.matvec(&tg, &bg, None, u), self.be.matvec(&tu, &bu, None, u));
            let out = self.be.matvec(&td, &bd, None, &act);
            y.iter_mut().zip(&out).for_each(|(a, &b)| *a += w * b as f64);
        }
        let act = swiglu(self.mv(&format!("{p}.ffn_gate_shexp.weight"), u),
                         self.mv(&format!("{p}.ffn_up_shexp.weight"), u));
        let sh = self.mv(&format!("{p}.ffn_down_shexp.weight"), &act);
        let sg = sigmoid(self.mv(&format!("{p}.ffn_gate_inp_shexp.weight"), u)[0]);
        y.iter().zip(&sh).map(|(&a, &b)| (a + (sg * b) as f64) as f32).collect()
    }

    /// The hashed n-gram memory (ARCH.md, "PLE"), on the full residual stream.
    fn ple(&mut self, r: &[f32], l: usize) -> Vec<f32> {
        let m = self.m;
        let c = &m.cfg;
        let p = format!("blk.{l}");
        let (h, cs) = (c.hidden, c.streams);
        let n = self.tokens.len();
        let last_eos = self.tokens.iter().rposition(|&t| t == c.ple_eos);
        let hist = |s: usize| -> i64 {
            match n.checked_sub(1 + s) {
                Some(i) if last_eos.is_none_or(|e| i >= e) => self.tokens[i] as i64,
                _ => c.ple_eos as i64,
            }
        };
        let mut e = Vec::with_capacity(h);
        for order in 2..=c.ple_ngram {
            let mut mixed = hist(0).wrapping_mul(c.ple_multipliers[0]);
            for s in 1..order {
                mixed ^= hist(s).wrapping_mul(c.ple_multipliers[s]);
            }
            for j in 0..c.ple_heads_per_ngram {
                let i = (order - 2) * c.ple_heads_per_ngram + j;
                let row = mixed.rem_euclid(c.ple_vocab[i]) + c.ple_offsets[i];
                e.extend(m.ple_row(row));
            }
        }
        assert_eq!(e.len(), h, "PLE embedding width");
        let key = rmsnorm_groups(&self.mv(&format!("{p}.ple_key.weight"), &e),
                                 &m.vec(&format!("{p}.ple_norm_key.weight")), h, c.eps);
        let val = self.mv(&format!("{p}.ple_value.weight"), &e);
        let qry = rmsnorm_groups(r, &m.vec(&format!("{p}.ple_norm_query.weight")), h, c.eps);
        let mut gv = vec![0f32; cs * h];
        for s in 0..cs {
            let dot: f64 = (0..h).map(|i| key[s * h + i] as f64 * qry[s * h + i] as f64).sum::<f64>() / (h as f64).sqrt();
            let g = dot.abs().max(1e-6).sqrt() * dot.signum();
            let sg = sigmoid(g as f32);
            for i in 0..h {
                gv[s * h + i] = sg * val[i];
            }
        }
        let gvn = rmsnorm_groups(&gv, &m.vec(&format!("{p}.ple_norm_conv.weight")), h, c.eps);
        // depthwise causal conv, kernel k, dilation = ngram: tap t multiplies the input (k-1-t)*dil steps back
        let (k, dil, ch) = (c.ple_conv_k, c.ple_ngram, cs * h);
        let w = m.vec(&format!("{p}.ple_conv1d.weight"));
        let hist_len = (k - 1) * dil;
        let mut out = gv.clone();
        for i in 0..ch {
            let mut acc = w[i * k + (k - 1)] as f64 * gvn[i] as f64;
            for t in 0..k - 1 {
                let back = (k - 1 - t) * dil; // 1..=hist_len steps back
                acc += w[i * k + t] as f64 * self.ple_conv[(hist_len - back) * ch + i] as f64;
            }
            out[i] += silu(acc as f32);
        }
        self.ple_conv.copy_within(ch.., 0);
        self.ple_conv[(hist_len - 1) * ch..].copy_from_slice(&gvn);
        out
    }
}

fn hc_write(r: &mut [f32], g: &[f32], y: &[f32], h: usize) {
    for (s, &gs) in g.iter().enumerate() {
        for i in 0..h {
            r[s * h + i] += gs * y[i];
        }
    }
}

/// The tensors a layer is expected to have, for a load-time check that the file is the shape ARCH.md describes.
pub fn check_layout(g: &Gguf, c: &Config) -> Result<(), String> {
    for t in &g.tensors {
        if let Some(l) = block_index(&t.name) {
            if l >= c.n_layer {
                return Err(format!("{}: layer {l} of {}", t.name, c.n_layer));
            }
        }
    }
    for l in 0..c.n_layer {
        let need: &[&str] = if c.is_attention(l) {
            &["attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm"]
        } else {
            &["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_conv1d", "ssm_out", "ssm_norm"]
        };
        for n in need {
            let name = format!("blk.{l}.{n}.weight");
            if g.tensor(&name).is_none() {
                return Err(format!("missing {name}"));
            }
        }
    }
    Ok(())
}
