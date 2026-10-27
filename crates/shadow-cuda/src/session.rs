//! The forward pass on the GPU (docs/SHADOW_PLAN.md, stage 2): a device-resident mirror of the CPU reference
//! `shadow_model::forward::Session`, step for step and with the same trace names, so the reference comparison can hold
//! it to the same fixtures.
//!
//! On the device: every dense weight as stored (matvec.cu decodes it), the small tensors as f32, the residual streams,
//! every DeltaNet state, every KV cache with the indexer's keys, and the n-gram conv state.  Still on the host, because
//! each is a small decision or a read the next stage replaces: the token's embedding row and its n-gram rows (read and
//! dequantized on the CPU, then uploaded: 10 KiB each), the router's top-10 (512 logits come down), the indexer's
//! top-512 blocks past 2,051 positions (its scores come down), and the routed experts' first use.
//!
//! The routed experts live in a VRAM cache (stage 3's first tier, docs/SHADOW_PLAN.md): one arena of equal slots, each
//! holding one expert's gate, up and down slices; a miss reads the slices from the file and uploads them into the
//! least recently used slot.  The experts the current token routes to are never the victims of its own misses.

use crate::driver::{Context, DeviceBuffer};
use crate::kernels::{file, Arg, Program, QUANT_KERNELS};
use shadow_model::forward::{ple_rows, route, Model, Stepper};
use shadow_model::gguf::TensorInfo;
use shadow_model::quant::dequantize;
use std::collections::{BTreeMap, HashMap};

struct Attn<'c> {
    k: DeviceBuffer<'c>,
    v: DeviceBuffer<'c>,
    raw: DeviceBuffer<'c>,
    blocks: DeviceBuffer<'c>,
    nblocks: usize,
}

/// Routed experts resident in VRAM: `slots` equal slots of `slot_bytes`, each one expert's [gate | up | down] slices.
struct ExpertCache<'c> {
    arena: Option<DeviceBuffer<'c>>,
    slot_bytes: usize,
    slots: usize,
    map: HashMap<(usize, usize), usize>,
    owner: Vec<Option<(usize, usize)>>,
    stamp: Vec<u64>,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl ExpertCache<'_> {
    /// The slot of (layer, expert) if resident, marked as used now.
    fn touch(&mut self, key: (usize, usize)) -> Option<usize> {
        let s = *self.map.get(&key)?;
        self.stamp[s] = self.clock;
        Some(s)
    }

    /// A slot for a new expert: a free one, else the least recently used one not used by the current token.
    fn victim(&mut self) -> usize {
        let s = (0..self.slots)
            .filter(|&i| self.owner[i].is_none() || self.stamp[i] < self.clock)
            .min_by_key(|&i| (self.owner[i].is_some(), self.stamp[i]))
            .expect("the cache has fewer slots than one token routes to");
        if let Some(old) = self.owner[s].take() {
            self.map.remove(&old);
        }
        s
    }
}

/// Scratch vectors, one per intermediate the forward pass names.
struct Bufs<'c> {
    r: DeviceBuffer<'c>,
    rn: DeviceBuffer<'c>,
    down: DeviceBuffer<'c>,
    up: DeviceBuffer<'c>,
    u: DeviceBuffer<'c>,
    gl: DeviceBuffer<'c>,
    g: DeviceBuffer<'c>,
    y: DeviceBuffer<'c>,
    e: DeviceBuffer<'c>,
    // DeltaNet
    qkv: DeviceBuffer<'c>,
    x: DeviceBuffer<'c>,
    z: DeviceBuffer<'c>,
    beta: DeviceBuffer<'c>,
    alpha: DeviceBuffer<'c>,
    o: DeviceBuffer<'c>,
    // attention
    qg: DeviceBuffer<'c>,
    q: DeviceBuffer<'c>,
    gate: DeviceBuffer<'c>,
    kk: DeviceBuffer<'c>,
    vv: DeviceBuffer<'c>,
    iq: DeviceBuffer<'c>,
    iqn: DeviceBuffer<'c>,
    ik: DeviceBuffer<'c>,
    ao: DeviceBuffer<'c>,
    scores: DeviceBuffer<'c>,
    toks: DeviceBuffer<'c>,
    scratch: DeviceBuffer<'c>,
    // MoE
    rl: DeviceBuffer<'c>,
    my: DeviceBuffer<'c>,
    eg: DeviceBuffer<'c>,
    eu: DeviceBuffer<'c>,
    act: DeviceBuffer<'c>,
    eo: DeviceBuffer<'c>,
    sgl: DeviceBuffer<'c>,
    wg: DeviceBuffer<'c>,
    wu: DeviceBuffer<'c>,
    wd: DeviceBuffer<'c>,
    // PLE
    pe: DeviceBuffer<'c>,
    key: DeviceBuffer<'c>,
    keyn: DeviceBuffer<'c>,
    val: DeviceBuffer<'c>,
    qry: DeviceBuffer<'c>,
    gv: DeviceBuffer<'c>,
    gvn: DeviceBuffer<'c>,
    // head
    logits: DeviceBuffer<'c>,
}

pub struct GpuSession<'c, 'm> {
    ctx: &'c Context,
    m: &'m Model,
    mvp: Program<'c>,
    fw: Program<'c>,
    w: HashMap<String, DeviceBuffer<'c>>,
    v: HashMap<String, DeviceBuffer<'c>>,
    b: Bufs<'c>,
    delta: Vec<Option<(DeviceBuffer<'c>, DeviceBuffer<'c>)>>,
    attn: Vec<Option<Attn<'c>>>,
    ple_state: DeviceBuffer<'c>,
    tokens: Vec<u32>,
    cap: usize,
    pub trace: Option<BTreeMap<String, Vec<f32>>>,
    pub stop_after: Option<usize>,
    /// Bytes of weights uploaded to the device at the start, and expert slices uploaded per use since.
    pub resident_bytes: usize,
    pub expert_uploads: usize,
    cache: ExpertCache<'c>,
}

/// VRAM left free for everything else when the expert cache sizes itself (the desktop, the driver, other programs).
pub const VRAM_RESERVE: usize = 1536 << 20;

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn is_small(name: &str) -> bool {
    name.contains("norm") || name.ends_with(".ssm_a") || name.ends_with(".ssm_dt.bias")
        || name.ends_with("conv1d.weight")
}

impl<'c, 'm> GpuSession<'c, 'm> {
    /// Uploads the dense weights and allocates every state for sequences of up to `cap` positions; the expert cache
    /// takes the VRAM left over, less `VRAM_RESERVE`.
    pub fn new(ctx: &'c Context, m: &'m Model, cap: usize) -> Result<GpuSession<'c, 'm>, String> {
        Self::with_cache(ctx, m, cap, None)
    }

    /// As `new`, with the expert cache's size in bytes (`Some(0)`: no cache, every expert uploaded per use).
    pub fn with_cache(ctx: &'c Context, m: &'m Model, cap: usize, cache_bytes: Option<usize>)
                      -> Result<GpuSession<'c, 'm>, String> {
        let c = &m.cfg;
        let e = |x: crate::driver::CudaError| x.to_string();
        let mvp = Program::load(ctx, file("matvec.cu").ok_or("matvec.cu")?)?;
        let fw = Program::load(ctx, file("forward.cu").ok_or("forward.cu")?)?;
        let mut w = HashMap::new();
        let mut v = HashMap::new();
        let mut resident_bytes = 0;
        let mut names: Vec<&str> = m.dense_names().collect();
        names.sort_unstable();
        for name in names {
            if name == "token_embd.weight" {
                continue; // the embedding row is dequantized on the host
            }
            let (_, bytes) = m.dense(name);
            let mut buf = ctx.alloc(bytes.len()).map_err(e)?;
            buf.upload(bytes).map_err(e)?;
            resident_bytes += bytes.len();
            w.insert(name.to_string(), buf);
            if is_small(name) {
                let f = f32s(&m.vec(name));
                let mut sb = ctx.alloc(f.len()).map_err(e)?;
                sb.upload(&f).map_err(e)?;
                v.insert(name.to_string(), sb);
            }
        }
        let (h, cs) = (c.hidden, c.streams);
        let ch = cs * h;
        let kd = c.ssm_k_heads * c.ssm_head_dim;
        let vd = c.ssm_v_heads * c.ssm_head_dim;
        let qd = c.n_head * c.head_dim;
        let kvd = c.n_kv_head * c.head_dim;
        let id = c.idx_dim;
        let ratio = c.compress.iter().copied().filter(|&r| r > 0).max().unwrap_or(1);
        let max_expert = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .flat_map(|r| (0..c.n_layer).map(move |l| format!("blk.{l}.{r}.weight")))
            .filter_map(|n| m.g.tensor(&n).map(|t| (t.nbytes() / t.dims[2]) as usize))
            .max()
            .unwrap_or(0);
        let a = |n: usize| ctx.alloc(n.max(1) * 4).map_err(e);
        let b = Bufs {
            r: a(ch)?, rn: a(ch)?, down: a(c.hc_rank)?, up: a(ch)?, u: a(h)?, gl: a(cs)?, g: a(cs)?, y: a(h)?, e: a(h)?,
            qkv: a(2 * kd + vd)?, x: a(2 * kd + vd)?, z: a(vd)?, beta: a(c.ssm_v_heads)?, alpha: a(c.ssm_v_heads)?,
            o: a(vd)?,
            qg: a(2 * qd)?, q: a(qd)?, gate: a(qd)?, kk: a(kvd)?, vv: a(kvd)?, iq: a(c.idx_heads * id)?,
            iqn: a(c.idx_heads * id)?, ik: a(id)?, ao: a(qd)?, scores: a(cap / ratio + 1)?,
            toks: a(c.idx_budget + ratio)?, scratch: ctx.alloc(c.n_head * cap * 8).map_err(e)?,
            rl: a(c.n_expert)?, my: a(h)?, eg: a(c.expert_ff)?, eu: a(c.expert_ff)?, act: a(c.expert_ff)?, eo: a(h)?,
            sgl: a(1)?, wg: ctx.alloc(max_expert).map_err(e)?, wu: ctx.alloc(max_expert).map_err(e)?,
            wd: ctx.alloc(max_expert).map_err(e)?,
            pe: a(h)?, key: a(ch)?, keyn: a(ch)?, val: a(h)?, qry: a(ch)?, gv: a(ch)?, gvn: a(ch)?,
            logits: a(c.vocab)?,
        };
        let mut delta = Vec::new();
        let mut attn = Vec::new();
        for l in 0..c.n_layer {
            if c.is_attention(l) {
                delta.push(None);
                let blocks = cap / c.compress[l].max(1) + 1;
                let mut at = Attn { k: a(cap * kvd)?, v: a(cap * kvd)?, raw: a(cap * id)?, blocks: a(blocks * id)?, nblocks: 0 };
                at.k.zero().map_err(e)?;
                at.v.zero().map_err(e)?;
                attn.push(Some(at));
            } else {
                let mut s = a(c.ssm_v_heads * c.ssm_head_dim * c.ssm_head_dim)?;
                let mut conv = a((c.conv_k - 1) * (2 * kd + vd))?;
                s.zero().map_err(e)?;
                conv.zero().map_err(e)?;
                delta.push(Some((s, conv)));
                attn.push(None);
            }
        }
        let mut ple_state = a((c.ple_conv_k - 1) * c.ple_ngram * ch)?;
        ple_state.zero().map_err(e)?;

        // the expert cache: slots as big as the largest expert (gate + up + down) of any layer
        let per = |name: String| m.g.tensor(&name).map(|t| (t.nbytes() / t.dims[2]) as usize).unwrap_or(0);
        let slot_bytes = (0..c.n_layer)
            .map(|l| per(format!("blk.{l}.ffn_gate_exps.weight")) + per(format!("blk.{l}.ffn_up_exps.weight"))
                + per(format!("blk.{l}.ffn_down_exps.weight")))
            .max()
            .unwrap_or(0)
            .next_multiple_of(256);
        let budget = match cache_bytes {
            Some(b) => b,
            None => ctx.mem_info().map_err(e)?.0.saturating_sub(VRAM_RESERVE),
        };
        let mut slots = if slot_bytes == 0 { 0 } else { budget / slot_bytes };
        if slots > 0 && slots < c.n_expert_used {
            slots = 0; // too small to hold one token's experts: no cache rather than a thrashing one
        }
        let mut arena = None;
        while slots > 0 {
            match ctx.alloc(slots * slot_bytes) {
                Ok(buf) => {
                    arena = Some(buf);
                    break;
                }
                Err(_) => slots = slots * 7 / 8, // the free figure is a moment's; settle for what the driver gives
            }
        }
        if slots < c.n_expert_used {
            arena = None;
            slots = 0;
        }
        let cache = ExpertCache {
            arena, slot_bytes, slots, map: HashMap::new(), owner: vec![None; slots], stamp: vec![0; slots], clock: 0,
            hits: 0, misses: 0,
        };
        Ok(GpuSession {
            ctx, m, mvp, fw, w, v, b, delta, attn, ple_state, tokens: vec![], cap, trace: None, stop_after: None,
            resident_bytes, expert_uploads: 0, cache,
        })
    }

    /// (slots, bytes per slot, hits, misses) of the expert cache.
    pub fn cache_stats(&self) -> (usize, usize, u64, u64) {
        (self.cache.slots, self.cache.slot_bytes, self.cache.hits, self.cache.misses)
    }

    /// One expert's slice of a fused `[in, out, n_expert]` tensor: its info as a 2-D matrix, without reading it.
    fn expert_info(&self, name: &str) -> TensorInfo {
        let mut t = self.m.g.tensor(name).unwrap_or_else(|| panic!("no tensor {name}")).clone();
        t.dims = vec![t.dims[0], t.dims[1]];
        t
    }

    // -------------------------------------------------------------------------------------------- launch helpers
    fn k(&self, name: &str, grid: u32, block: u32, args: &[Arg]) {
        // SAFETY: every pointer passed is a buffer of this session sized for the dimensions passed beside it (the
        // registry checks the argument types and count).
        unsafe { self.fw.launch(name, (grid, 1, 1), (block, 1, 1), 0, None, args) }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
    }

    fn k1(&self, name: &str, n: usize, args: &[Arg]) {
        self.k(name, (n as u32).div_ceil(256), 256, args);
    }

    fn matvec(&self, t: &TensorInfo, wptr: u64, rows: usize, x: u64, y: u64) {
        let ty = t.ty.name();
        let &(_, _, _, _, entry) = QUANT_KERNELS.iter().find(|q| q.0 == ty)
            .unwrap_or_else(|| panic!("{}: no matvec kernel for {ty}", t.name));
        // SAFETY: `rows` whole rows of t.dims[0] weights at wptr; x holds t.dims[0] floats, y `rows`.
        unsafe {
            self.mvp.launch(entry, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(wptr), Arg::Ptr(x), Arg::Ptr(y), Arg::U32(rows as u32), Arg::U32(t.dims[0] as u32)])
        }
        .unwrap_or_else(|err| panic!("{}: {err}", t.name));
    }

    /// y = W x for a resident dense tensor.
    fn mv(&self, name: &str, x: u64, y: u64) {
        let (t, bytes) = self.m.dense(name);
        let (qk, qb) = t.ty.block().unwrap();
        let rows = bytes.len() / (t.dims[0] / qk * qb) as usize;
        self.matvec(t, self.w.get(name).unwrap_or_else(|| panic!("{name} is not resident")).ptr, rows, x, y);
    }

    fn sv(&self, name: &str) -> u64 {
        self.v.get(name).unwrap_or_else(|| panic!("{name} is not a small resident tensor")).ptr
    }

    fn rec(&mut self, name: String, buf: u64, n: usize) {
        if self.trace.is_none() {
            return;
        }
        let mut out = vec![0u8; n * 4];
        self.download(buf, &mut out);
        self.trace.as_mut().unwrap().entry(name).or_default().extend(floats(&out));
    }

    fn rec_host(&mut self, name: String, v: &[f32]) {
        if let Some(t) = self.trace.as_mut() {
            t.entry(name).or_default().extend_from_slice(v);
        }
    }

    /// Copies `out.len()` bytes from device address `ptr` (any offset into a buffer).
    fn download(&self, ptr: u64, out: &mut [u8]) {
        let api = self.ctx.api;
        self.ctx.bind().unwrap();
        // SAFETY: the address range lies inside a buffer of this session (the callers pass its sizes).
        let r = unsafe { (api.cuMemcpyDtoH)(out.as_mut_ptr() as *mut std::ffi::c_void, ptr, out.len()) };
        assert_eq!(r, 0, "cuMemcpyDtoH failed ({r})");
    }

    fn upload_to(&self, ptr: u64, data: &[u8]) {
        let api = self.ctx.api;
        self.ctx.bind().unwrap();
        // SAFETY: as download.
        let r = unsafe { (api.cuMemcpyHtoD)(ptr, data.as_ptr() as *const std::ffi::c_void, data.len()) };
        assert_eq!(r, 0, "cuMemcpyHtoD failed ({r})");
    }

    // -------------------------------------------------------------------------------------------- the forward pass
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        assert!(self.tokens.len() < self.cap, "this session was made for {} positions", self.cap);
        self.tokens.push(token);
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden, c.streams);

        let emb = {
            let (t, bytes) = self.m.dense("token_embd.weight");
            let (qk, qb) = t.ty.block().unwrap();
            let rb = (t.dims[0] / qk * qb) as usize;
            let mut x = vec![0f32; h];
            dequantize(t.ty, &bytes[token as usize * rb..(token as usize + 1) * rb], &mut x).unwrap();
            x
        };
        self.upload_to(self.b.e.ptr, &f32s(&emb));
        self.k1("broadcast", h * cs, &[Arg::Ptr(self.b.e.ptr), Arg::Ptr(self.b.r.ptr), Arg::U32(h as u32), Arg::U32(cs as u32)]);
        self.rec_host("embed".into(), &emb);

        for l in 0..c.n_layer {
            if self.stop_after == Some(l) {
                self.hc_read("output_hc", false);
                self.rec("final_u".into(), self.b.u.ptr, h);
                return vec![];
            }
            if l == c.ple_layer {
                self.ple(l);
                self.k1("add", h * cs, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.b.gv.ptr), Arg::U32((h * cs) as u32)]);
                self.rec(format!("layer{l}.ple"), self.b.gv.ptr, h * cs);
            }
            self.hc_read(&format!("blk.{l}.hc_attn"), true);
            self.rec(format!("layer{l}.attn_hc_u"), self.b.u.ptr, h);
            if c.is_attention(l) {
                self.attention(l);
            } else {
                self.delta_net(l);
            }
            self.rec(format!("layer{l}.mixer"), self.b.y.ptr, h);
            self.hc_write();
            self.hc_read(&format!("blk.{l}.hc_ffn"), true);
            self.rec(format!("layer{l}.mlp_hc_u"), self.b.u.ptr, h);
            self.moe(l);
            self.rec(format!("layer{l}.moe"), self.b.y.ptr, h);
            self.hc_write();
            self.rec(format!("layer{l}.out"), self.b.r.ptr, h * cs);
        }
        self.hc_read("output_hc", false);
        self.rec("final_u".into(), self.b.u.ptr, h);
        self.mv("output.weight", self.b.u.ptr, self.b.logits.ptr);
        let mut out = vec![0u8; c.vocab * 4];
        self.download(self.b.logits.ptr, &mut out);
        floats(&out)
    }

    /// The hyper-connection read: u (and g, with `inject`) from the streams r.
    fn hc_read(&self, p: &str, inject: bool) {
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden as u32, c.streams as u32);
        let b = &self.b;
        self.k("rmsnorm_groups", cs, 256, &[Arg::Ptr(b.r.ptr), Arg::Ptr(self.sv(&format!("{p}_norm.weight"))),
                                           Arg::Ptr(b.rn.ptr), Arg::U32(h), Arg::F32(c.eps)]);
        self.mv(&format!("{p}_down.weight"), b.rn.ptr, b.down.ptr);
        self.k1("silu_div", c.hc_rank, &[Arg::Ptr(b.down.ptr), Arg::U32(c.hc_rank as u32), Arg::F32(cs as f32)]);
        self.mv(&format!("{p}_up.weight"), b.down.ptr, b.up.ptr);
        self.k1("hc_mix", h as usize, &[Arg::Ptr(b.rn.ptr), Arg::Ptr(b.up.ptr), Arg::Ptr(b.u.ptr), Arg::U32(h), Arg::U32(cs)]);
        if inject {
            self.mv(&format!("{p}_inject.weight"), b.rn.ptr, b.gl.ptr);
            self.k1("sigmoid_div", cs as usize, &[Arg::Ptr(b.gl.ptr), Arg::Ptr(b.g.ptr), Arg::U32(cs), Arg::F32(cs as f32),
                                                  Arg::F32(2.0)]);
        }
    }

    fn hc_write(&self) {
        let c = &self.m.cfg;
        let b = &self.b;
        self.k1("hc_write", c.hidden * c.streams, &[Arg::Ptr(b.r.ptr), Arg::Ptr(b.g.ptr), Arg::Ptr(b.y.ptr),
                                                   Arg::U32(c.hidden as u32), Arg::U32(c.streams as u32)]);
    }

    fn delta_net(&self, l: usize) {
        let c = &self.m.cfg;
        let b = &self.b;
        let p = format!("blk.{l}");
        let (kh, vh, d) = (c.ssm_k_heads, c.ssm_v_heads, c.ssm_head_dim);
        let (kd, vd) = (kh * d, vh * d);
        let ch = 2 * kd + vd;
        self.mv(&format!("{p}.attn_qkv.weight"), b.u.ptr, b.qkv.ptr);
        self.mv(&format!("{p}.attn_gate.weight"), b.u.ptr, b.z.ptr);
        self.mv(&format!("{p}.ssm_beta.weight"), b.u.ptr, b.beta.ptr);
        self.mv(&format!("{p}.ssm_alpha.weight"), b.u.ptr, b.alpha.ptr);
        let (s, conv) = self.delta[l].as_ref().unwrap();
        self.k1("conv_silu_step", ch, &[Arg::Ptr(b.qkv.ptr), Arg::Ptr(conv.ptr), Arg::Ptr(self.sv(&format!("{p}.ssm_conv1d.weight"))),
                                       Arg::Ptr(b.x.ptr), Arg::U32(ch as u32), Arg::U32(c.conv_k as u32)]);
        self.k("l2norm_heads", kh as u32, 128, &[Arg::Ptr(b.x.ptr), Arg::U32(d as u32), Arg::F32((d as f32).sqrt())]);
        self.k("l2norm_heads", kh as u32, 128, &[Arg::Ptr(b.x.ptr + (kd * 4) as u64), Arg::U32(d as u32), Arg::F32(1.0)]);
        self.k("delta_step", vh as u32, d as u32, &[
            Arg::Ptr(b.x.ptr), Arg::Ptr(b.x.ptr + (kd * 4) as u64), Arg::Ptr(b.x.ptr + (2 * kd * 4) as u64),
            Arg::Ptr(b.beta.ptr), Arg::Ptr(b.alpha.ptr), Arg::Ptr(self.sv(&format!("{p}.ssm_a"))),
            Arg::Ptr(self.sv(&format!("{p}.ssm_dt.bias"))), Arg::Ptr(s.ptr), Arg::Ptr(b.o.ptr), Arg::U32(kh as u32),
            Arg::U32(d as u32)]);
        self.k("gated_rmsnorm_heads", vh as u32, 128, &[Arg::Ptr(b.o.ptr), Arg::Ptr(self.sv(&format!("{p}.ssm_norm.weight"))),
                                                       Arg::Ptr(b.z.ptr), Arg::U32(d as u32), Arg::F32(c.eps)]);
        let _ = vd;
        self.mv(&format!("{p}.ssm_out.weight"), b.o.ptr, b.y.ptr);
    }

    fn attention(&mut self, l: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let (id, ratio) = (c.idx_dim, c.compress[l].max(1));
        let pos = self.tokens.len() - 1;
        let (rd, th, eps) = (c.rope_dims as u32, c.rope_theta as f32, c.eps);
        {
            let b = &self.b;
            self.mv(&format!("{p}.attn_q.weight"), b.u.ptr, b.qg.ptr);
            self.mv(&format!("{p}.attn_k.weight"), b.u.ptr, b.kk.ptr);
            self.mv(&format!("{p}.attn_v.weight"), b.u.ptr, b.vv.ptr);
            self.mv(&format!("{p}.indexer.q_proj.weight"), b.u.ptr, b.iq.ptr);
            self.mv(&format!("{p}.indexer.k_proj.weight"), b.u.ptr, b.ik.ptr);
            self.k("head_norm_rope", nh as u32, 128, &[
                Arg::Ptr(b.qg.ptr), Arg::U32((2 * hd) as u32), Arg::Ptr(b.q.ptr), Arg::U32(hd as u32),
                Arg::Ptr(self.sv(&format!("{p}.attn_q_norm.weight"))), Arg::U32(hd as u32), Arg::F32(eps), Arg::U32(pos as u32),
                Arg::U32(rd), Arg::F32(th)]);
            self.k1("split_gate", nh * hd, &[Arg::Ptr(b.qg.ptr), Arg::Ptr(b.gate.ptr), Arg::U32(nh as u32), Arg::U32(hd as u32)]);
            let at = self.attn[l].as_ref().unwrap();
            let slot = (pos * nkv * hd * 4) as u64;
            self.k("head_norm_rope", nkv as u32, 128, &[
                Arg::Ptr(b.kk.ptr), Arg::U32(hd as u32), Arg::Ptr(at.k.ptr + slot), Arg::U32(hd as u32),
                Arg::Ptr(self.sv(&format!("{p}.attn_k_norm.weight"))), Arg::U32(hd as u32), Arg::F32(eps), Arg::U32(pos as u32),
                Arg::U32(rd), Arg::F32(th)]);
            self.k1("copy", nkv * hd, &[Arg::Ptr(at.v.ptr + slot), Arg::Ptr(b.vv.ptr), Arg::U32((nkv * hd) as u32)]);
            self.k1("copy", id, &[Arg::Ptr(at.raw.ptr + (pos * id * 4) as u64), Arg::Ptr(b.ik.ptr), Arg::U32(id as u32)]);
        }
        // complete blocks get their pooled keys once
        let n = pos + 1;
        let nb = n / ratio;
        let done = self.attn[l].as_ref().unwrap().nblocks;
        for blk in done..nb {
            let at = self.attn[l].as_ref().unwrap();
            self.k("pool_block", 1, 128, &[
                Arg::Ptr(at.raw.ptr), Arg::Ptr(at.blocks.ptr), Arg::Ptr(self.sv(&format!("{p}.indexer.k_norm.weight"))),
                Arg::U32(blk as u32), Arg::U32(ratio as u32), Arg::U32(id as u32), Arg::F32(eps), Arg::U32(rd), Arg::F32(th)]);
        }
        self.attn[l].as_mut().unwrap().nblocks = nb;
        let keep = c.idx_budget / ratio;
        let selected: Option<Vec<u32>> = (nb > keep).then(|| {
            let b = &self.b;
            let at = self.attn[l].as_ref().unwrap();
            self.k("head_norm_rope", c.idx_heads as u32, 128, &[
                Arg::Ptr(b.iq.ptr), Arg::U32(id as u32), Arg::Ptr(b.iqn.ptr), Arg::U32(id as u32),
                Arg::Ptr(self.sv(&format!("{p}.indexer.q_norm.weight"))), Arg::U32(id as u32), Arg::F32(eps), Arg::U32(pos as u32),
                Arg::U32(rd), Arg::F32(th)]);
            self.k1("index_scores", nb, &[Arg::Ptr(b.iqn.ptr), Arg::Ptr(at.blocks.ptr), Arg::Ptr(b.scores.ptr), Arg::U32(nb as u32),
                                          Arg::U32(c.idx_heads as u32), Arg::U32(id as u32)]);
            let mut sb = vec![0u8; nb * 4];
            self.download(b.scores.ptr, &mut sb);
            let mut scored: Vec<(f32, usize)> = floats(&sb).into_iter().zip(0..nb).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            let mut toks: Vec<u32> = scored[..keep].iter().flat_map(|&(_, blk)| (blk * ratio) as u32..((blk + 1) * ratio) as u32).collect();
            toks.extend((nb * ratio) as u32..n as u32);
            toks.sort_unstable();
            toks
        });
        let (tptr, ntok) = match &selected {
            Some(t) => {
                self.upload_to(self.b.toks.ptr, &t.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                (self.b.toks.ptr, t.len())
            }
            None => (0, n),
        };
        {
            let b = &self.b;
            let at = self.attn[l].as_ref().unwrap();
            self.k("attention_heads", nh as u32, 256, &[
                Arg::Ptr(b.q.ptr), Arg::Ptr(at.k.ptr), Arg::Ptr(at.v.ptr), Arg::Ptr(tptr), Arg::U32(ntok as u32),
                Arg::Ptr(b.gate.ptr), Arg::Ptr(b.ao.ptr), Arg::Ptr(b.scratch.ptr), Arg::U32(self.cap as u32),
                Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32((nh / nkv) as u32)]);
            self.mv(&format!("{p}.attn_output.weight"), b.ao.ptr, b.y.ptr);
        }
        if self.trace.is_some() {
            let mut row = vec![-1f32; c.idx_budget + ratio - 1];
            let all: Vec<u32> = selected.unwrap_or_else(|| (0..n as u32).collect());
            for (r, &t) in row.iter_mut().zip(&all) {
                *r = t as f32;
            }
            self.rec_host(format!("layer{l}.selected"), &row);
        }
    }

    fn moe(&mut self, l: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let h = c.hidden;
        self.mv(&format!("{p}.ffn_gate_inp.weight"), self.b.u.ptr, self.b.rl.ptr);
        let mut lb = vec![0u8; c.n_expert * 4];
        self.download(self.b.rl.ptr, &mut lb);
        let top = route(&floats(&lb), c.n_expert_used);
        self.rec_host(format!("layer{l}.route_idx"), &top.iter().map(|&(e, _)| e as f32).collect::<Vec<_>>());
        self.rec_host(format!("layer{l}.route_w"), &top.iter().map(|&(_, w)| w as f32).collect::<Vec<_>>());
        self.b.my.zero().unwrap();
        let names = [format!("{p}.ffn_gate_exps.weight"), format!("{p}.ffn_up_exps.weight"), format!("{p}.ffn_down_exps.weight")];
        let infos: Vec<TensorInfo> = names.iter().map(|n| self.expert_info(n)).collect();
        let sizes: Vec<usize> = names.iter().map(|n| {
            let t = self.m.g.tensor(n).unwrap();
            (t.nbytes() / t.dims[2]) as usize
        }).collect();
        // where each expert's three slices are on the device this time
        let mut at: Vec<[u64; 3]> = Vec::with_capacity(top.len());
        if self.cache.slots > 0 {
            self.cache.clock += 1;
            for &(e, _) in &top {
                self.cache.touch((l, e)); // protect this token's resident experts from its own misses
            }
            for &(e, _) in &top {
                let slot = match self.cache.touch((l, e)) {
                    Some(s) => {
                        self.cache.hits += 1;
                        s
                    }
                    None => {
                        self.cache.misses += 1;
                        let s = self.cache.victim();
                        let base = self.cache.arena.as_ref().unwrap().ptr + (s * self.cache.slot_bytes) as u64;
                        let mut off = 0u64;
                        for (name, &size) in names.iter().zip(&sizes) {
                            let (_, bytes) = self.m.expert(name, e);
                            self.upload_to(base + off, &bytes);
                            off += size as u64;
                        }
                        self.expert_uploads += 3;
                        self.cache.owner[s] = Some((l, e));
                        self.cache.map.insert((l, e), s);
                        self.cache.stamp[s] = self.cache.clock;
                        s
                    }
                };
                let base = self.cache.arena.as_ref().unwrap().ptr + (slot * self.cache.slot_bytes) as u64;
                at.push([base, base + sizes[0] as u64, base + (sizes[0] + sizes[1]) as u64]);
            }
        }
        let b = &self.b;
        for (i, &(e, wgt)) in top.iter().enumerate() {
            let [pg, pu, pd] = if self.cache.slots > 0 {
                at[i]
            } else {
                // no cache: through the scratch buffers, every time
                for (name, buf) in names.iter().zip([&b.wg, &b.wu, &b.wd]) {
                    let (_, bytes) = self.m.expert(name, e);
                    self.upload_to(buf.ptr, &bytes);
                }
                self.expert_uploads += 3;
                [b.wg.ptr, b.wu.ptr, b.wd.ptr]
            };
            let (tg, tu, td) = (&infos[0], &infos[1], &infos[2]);
            self.matvec(tg, pg, tg.dims[1] as usize, b.u.ptr, b.eg.ptr);
            self.matvec(tu, pu, tu.dims[1] as usize, b.u.ptr, b.eu.ptr);
            self.k1("swiglu", c.expert_ff, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(b.eu.ptr), Arg::Ptr(b.act.ptr), Arg::U32(c.expert_ff as u32)]);
            self.matvec(td, pd, td.dims[1] as usize, b.act.ptr, b.eo.ptr);
            self.k1("axpy", h, &[Arg::Ptr(b.my.ptr), Arg::Ptr(b.eo.ptr), Arg::U32(h as u32), Arg::F32(wgt as f32)]);
        }
        self.mv(&format!("{p}.ffn_gate_shexp.weight"), b.u.ptr, b.eg.ptr);
        self.mv(&format!("{p}.ffn_up_shexp.weight"), b.u.ptr, b.eu.ptr);
        self.k1("swiglu", c.expert_ff, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(b.eu.ptr), Arg::Ptr(b.act.ptr), Arg::U32(c.expert_ff as u32)]);
        self.mv(&format!("{p}.ffn_down_shexp.weight"), b.act.ptr, b.eo.ptr);
        self.mv(&format!("{p}.ffn_gate_inp_shexp.weight"), b.u.ptr, b.sgl.ptr);
        self.k1("axpy_sigmoid", h, &[Arg::Ptr(b.my.ptr), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.sgl.ptr), Arg::U32(h as u32)]);
        self.k1("copy", h, &[Arg::Ptr(b.y.ptr), Arg::Ptr(b.my.ptr), Arg::U32(h as u32)]);
    }

    /// The n-gram memory at layer l, into gv (the caller adds it to r).
    fn ple(&self, l: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (h, cs) = (c.hidden, c.streams);
        let ch = h * cs;
        let e: Vec<f32> = ple_rows(c, &self.tokens).into_iter().flat_map(|row| self.m.ple_row(row)).collect();
        let b = &self.b;
        self.upload_to(b.pe.ptr, &f32s(&e));
        self.mv(&format!("{p}.ple_key.weight"), b.pe.ptr, b.key.ptr);
        self.k("rmsnorm_groups", cs as u32, 256, &[Arg::Ptr(b.key.ptr), Arg::Ptr(self.sv(&format!("{p}.ple_norm_key.weight"))),
                                                  Arg::Ptr(b.keyn.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
        self.mv(&format!("{p}.ple_value.weight"), b.pe.ptr, b.val.ptr);
        self.k("rmsnorm_groups", cs as u32, 256, &[Arg::Ptr(b.r.ptr), Arg::Ptr(self.sv(&format!("{p}.ple_norm_query.weight"))),
                                                  Arg::Ptr(b.qry.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
        self.k("ple_gate", cs as u32, 256, &[Arg::Ptr(b.keyn.ptr), Arg::Ptr(b.qry.ptr), Arg::Ptr(b.val.ptr), Arg::Ptr(b.gv.ptr),
                                            Arg::U32(h as u32)]);
        self.k("rmsnorm_groups", cs as u32, 256, &[Arg::Ptr(b.gv.ptr), Arg::Ptr(self.sv(&format!("{p}.ple_norm_conv.weight"))),
                                                  Arg::Ptr(b.gvn.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
        self.k1("ple_conv_step", ch, &[Arg::Ptr(b.gvn.ptr), Arg::Ptr(self.ple_state.ptr),
                                      Arg::Ptr(self.sv(&format!("{p}.ple_conv1d.weight"))), Arg::Ptr(b.gv.ptr), Arg::U32(ch as u32),
                                      Arg::U32(c.ple_conv_k as u32), Arg::U32(c.ple_ngram as u32)]);
    }
}

impl Stepper for GpuSession<'_, '_> {
    fn step(&mut self, token: u32) -> Vec<f32> {
        GpuSession::step(self, token)
    }
    fn start_trace(&mut self) {
        self.trace = Some(BTreeMap::new());
    }
    fn take_trace(&mut self) -> Option<BTreeMap<String, Vec<f32>>> {
        self.trace.take()
    }
    fn set_stop_after(&mut self, n: Option<usize>) {
        self.stop_after = n;
    }
    fn report(&self) -> String {
        let (slots, bytes, hits, misses) = self.cache_stats();
        format!("expert cache: {slots} slots x {:.2} MiB = {:.2} GiB; {hits} hits, {misses} misses ({:.1}% hits); {} slice uploads",
                bytes as f64 / (1u64 << 20) as f64, (slots * bytes) as f64 / (1u64 << 30) as f64,
                100.0 * hits as f64 / (hits + misses).max(1) as f64, self.expert_uploads)
    }
}
