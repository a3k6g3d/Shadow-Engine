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

use crate::driver::{Context, DeviceBuffer, Event, Graph, Stream};
use crate::kernels::{file, Arg, Program, FAST_KERNELS, PAIR2_KERNELS, PAIR_KERNELS, QUANT_KERNELS};
use crate::pinned::PinnedRegion;
use shadow_model::forward::{ple_rows, route, route_biased, route_sigmoid, Model, Stepper};
use shadow_model::gguf::TensorInfo;
use shadow_model::placement::{uniform_order, RoutingCounts};
use shadow_model::quant::dequantize;
use std::collections::{BTreeMap, HashMap};

#[path = "session_mtp.rs"]
mod mtp_layer;
#[path = "session_dflash.rs"]
mod dflash_layer;
#[path = "session_pipeline.rs"]
mod pipeline_layer;

struct Attn<'c> {
    k: DeviceBuffer<'c>,
    v: DeviceBuffer<'c>,
    raw: DeviceBuffer<'c>,
    blocks: DeviceBuffer<'c>,
    nblocks: usize,
}

/// The part of the expert cache a long prompt borrows: `n` staging slots from `ptr`, the cache slots from `s0` on (their
/// entries `saved`, for the restoration) and every layer's evictable range as it was.
struct Lent {
    ptr: u64,
    n: usize,
    saved: Vec<(usize, usize, usize)>,
    evict_saved: Vec<std::ops::Range<usize>>,
}

/// A long prompt's layer-ahead pipeline (`moe_staged`): two halves of the lent cache hold the experts of the layer being computed and of
/// the next one (slot = expert number), the next layer's are copied in while this one runs, and the file-only ones (in neither the VRAM
/// cache nor the RAM tier) are read from the file on the read pool one layer further ahead, into two pinned buffers by layer parity.
struct StagePipe<'c> {
    /// device address of half 0; half p is at base + p * n_expert * slot_bytes
    base: u64,
    /// the copy of the layer in `ready[p]` into half p is queued (and done when `events[p]` is)
    events: [Event<'c>; 2],
    /// recorded on the compute stream after layer l's combine (half l & 1 is read by its experts until then): the copy two layers on waits for it
    done: [Event<'c>; 2],
    ready: [usize; 2],
    /// three read-ahead buffers, layer l's in `l % 3`: the one the next layer's read fills was last used two layers back, whose copy is long done, so the host
    /// need not wait for the copy it just queued (with two, the host blocked on every layer's copy before it could queue the next layer's compute)
    bufs: [PinnedRegion<'c>; 3],
    /// recorded on the copy stream after the copy out of buffer b
    buf_done: [Event<'c>; 3],
    /// the layer whose file-only experts buffer p holds, and expert -> slot in it
    layer: [usize; 3],
    at: [HashMap<usize, usize>; 3],
    slots: usize,
    pending: bool,
}

/// Routed experts resident in VRAM: either partitioned per layer at each layer's exact expert size (0 wasted padding
/// across mixed-quant layers) or shared `slots` of `slot_bytes` when the budget is too small to give every layer its
/// minimum slots.
struct ExpertCache<'c> {
    arena: Option<DeviceBuffer<'c>>,
    slot_offset: Vec<u64>,
    layer_range: Vec<std::ops::Range<usize>>,
    evict_range: Vec<std::ops::Range<usize>>,
    slot_bytes: usize,
    arena_bytes: usize,
    slots: usize,
    map: HashMap<(usize, usize), usize>,
    owner: Vec<Option<(usize, usize)>>,
    stamp: Vec<u64>,
    clock: u64,
    hits: u64,
    misses: u64,
    /// `--admit-after N`: an expert missed fewer than N times before is read through scratch, not given a cache slot
    admit_after: usize,
    seen: HashMap<(usize, usize), u32>,
    scratch_reads: u64,
}

impl ExpertCache<'_> {
    /// The slot of (layer, expert) if resident, marked as used now.
    fn touch(&mut self, key: (usize, usize)) -> Option<usize> {
        let s = *self.map.get(&key)?;
        self.stamp[s] = self.clock;
        Some(s)
    }

    /// A slot in layer `l` for a new expert: a free one, else the least recently used evictable slot not used by the
    /// current token.
    /// Whether layer l has a slot `victim` may take now.
    fn has_victim(&self, l: usize) -> bool {
        self.evict_range[l].clone().any(|i| self.owner[i].is_none() || self.stamp[i] < self.clock)
    }

    fn victim(&mut self, l: usize) -> usize {
        let r = self.evict_range[l].clone();
        let s = r
            .filter(|&i| self.owner[i].is_none() || self.stamp[i] < self.clock)
            .min_by_key(|&i| (self.owner[i].is_some(), self.stamp[i]))
            .expect("the cache has fewer slots than one token routes to");
        if let Some(old) = self.owner[s].take() {
            self.map.remove(&old);
        }
        s
    }
}

/// Where a step's time goes, with `SHADOW_PROFILE=1`: each `mark` synchronizes the device and charges the wall time
/// since the previous mark to a phase.  The synchronizations themselves cost time, so a profiled run is slower than an
/// unprofiled one; the split is what it is for.
struct Prof {
    last: std::time::Instant,
    phases: Vec<(&'static str, f64)>,
}

/// Scratch vectors, one per intermediate the forward pass names.
struct Bufs<'c> {
    r: DeviceBuffer<'c>,
    rn: DeviceBuffer<'c>,
    down: DeviceBuffer<'c>,
    up: DeviceBuffer<'c>,
    u: DeviceBuffer<'c>,
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
    iq: DeviceBuffer<'c>,
    iqn: DeviceBuffer<'c>,
    ao: DeviceBuffer<'c>,
    scores: DeviceBuffer<'c>,
    toks: DeviceBuffer<'c>,
    scratch: DeviceBuffer<'c>,
    // MoE
    rl: DeviceBuffer<'c>,
    eg: DeviceBuffer<'c>,
    eu: DeviceBuffer<'c>,
    act: DeviceBuffer<'c>,
    eo: DeviceBuffer<'c>,
    sgl: DeviceBuffer<'c>,
    wg: DeviceBuffer<'c>,
    wu: DeviceBuffer<'c>,
    wd: DeviceBuffer<'c>,
    /// Per layer: the routed experts' gate, up and down addresses (u64 each) and their weights (f32), for the batched
    /// matvecs and axpy_many.
    tab: DeviceBuffer<'c>,
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

/// The most tokens `step_batch` takes through the layers together.  The projections take them COLS_MAX at a time
/// (matvec_cols' limit); the MoE fetches the union of all of them, which grows slowly: on an 1,271-token prompt a
/// token needs 5.95 experts of its 10 a layer in groups of 8, 3.60 in groups of 32, 2.51 in groups of 64.
pub const MAX_BATCH: usize = 256;
/// Tokens a long prompt reads together in the lent cache (`big_batch_on`): per-token buffers carved from it, not kept.
pub const BIG_BATCH: usize = 2048;
/// Tokens a pass can take at most (`lend_kv` picks 2,048, 1,024 or 512 by the lent KV tail). 4,096 was tried (half the copying a token) but the drafter's feature buffers have to hold a whole pass: +150 MB of VRAM, -2 points of the 24k window's prompt speed, and the tail is only that big in a window band where the staging pipeline cannot start (STATUS.md 2026-11-03)
pub const BIG_BATCH_MAX: usize = 16384;
/// Rows a pass takes when the lent room allows (`SHADOW_BIG_ROWS` sets it): 4,096 read 9.5k and 15k prompts 14% and 23% faster than 2,048, 8,192 (run in sub-chunks, `sub_rows_cap`) 5-8% faster again
/// (docs/STATUS.md 2026-11-04), at some cost to the first reply's decode speed (the cache is emptier)
fn big_rows_cap() -> usize {
    std::env::var("SHADOW_BIG_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16384).min(BIG_BATCH_MAX)
}
/// Rows of the layer's temporary buffers when a pass is bigger (`SHADOW_SUB_ROWS`): the pass is run layer by layer in sub-chunks of this many rows, so only the residual
/// streams (4 x hidden floats a row) are kept for the whole pass and a pass of 8,192 rows takes less VRAM than 4,096 rows used to.
fn sub_rows_cap() -> usize {
    std::env::var("SHADOW_SUB_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048).max(256)
}
/// Bytes the big batch's buffers take for `total` rows a pass: the temporaries for the sub-chunk and the residual streams of every row.
fn big_batch_bytes(c: &shadow_model::forward::Config, total: usize) -> usize {
    let sub = sub_rows_cap().min(total);
    bb_row_floats(c) * 4 * sub + if total > sub { c.streams * c.hidden * 4 * total.next_multiple_of(sub) } else { 0 }
}
/// Unread prompt tokens from which the layer-ahead pipeline and the big batch are used.
pub const LONG_PROMPT: usize = 1024;
const COLS_MAX: usize = 8;
/// Tokens a batched attention launch takes (its score scratch is per token).
const ATT_ROWS: usize = 8;
/// Tokens of a long prompt whose indexer selections are made together (`attention_batch`): one scores download for all of them.
/// Rows per group of the sparse attention's selection and attention launches (`SHADOW_SEL_ROWS`): one group is ~5 launches, so the launch count of a long prompt scales with
/// Tokens a group of the tensor-core grouped expert kernels holds (`SHADOW_MMA_GROUP`, 32 or 64): a weight tile is dequantized once for a whole group, so 64 halves the
/// dequantization of an expert with more than 32 tokens (an average expert has ~40 in a 2,048-row sub-chunk).
fn mma_group() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new()
    ;
    *N.get_or_init(|| std::env::var("SHADOW_MMA_GROUP").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n == 32 || n == 64).unwrap_or(32))
}

/// 1 / this (the host's launches stall once per Windows timer tick, docs/STATUS.md 2026-11-05 (29)).
fn sel_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("SHADOW_SEL_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(128).clamp(1, 256))
}

/// The most tokens a speculative verify pass (`step_batch_verify`) keeps the recurrences' inputs of.
pub const MAX_VERIFY: usize = 8;

/// What `step_batch_verify` keeps so `rollback` can redo only the recurrences for the tokens it keeps: per DeltaNet
/// layer each token's conv input and beta / alpha, each token's n-gram conv input, and (first position, tokens).
struct Replay<'c> {
    qkv: DeviceBuffer<'c>,
    ba: DeviceBuffer<'c>,
    ple: DeviceBuffer<'c>,
    pass: Option<(usize, Vec<u32>)>,
    on: std::cell::Cell<bool>,
}

/// Expert slots outside the cache for a batched step's misses beyond the layer's free cache slots (a layer keeps only
/// a dozen evictable slots next to its pinned hot experts, and a prompt misses most of its experts); `SHADOW_BATCH_SCRATCH`
/// overrides it.
const BATCH_SCRATCH_SLOTS: usize = 32;

/// `step_batch`'s per-token intermediates, MAX_BATCH rows each, row j of a buffer at ptr + j * its stride: what goes
/// into or comes out of a projection all m tokens share (the rest stay in `Bufs`, used one token at a time).
struct BatchBufs<'c> {
    /// the allocation `qkv`..`ao` are views into when the two layer kinds share their buffers (`alias_sets`): it must live as long as they do
    _region: Option<DeviceBuffer<'c>>,
    r: DeviceBuffer<'c>,
    rn: DeviceBuffer<'c>,
    u: DeviceBuffer<'c>,
    g: DeviceBuffer<'c>,
    y: DeviceBuffer<'c>,
    down: DeviceBuffer<'c>,
    up: DeviceBuffer<'c>,
    qkv: DeviceBuffer<'c>,
    /// the conv's output, normalized: DeltaNet's q, k, v per token
    x: DeviceBuffer<'c>,
    z: DeviceBuffer<'c>,
    beta: DeviceBuffer<'c>,
    alpha: DeviceBuffer<'c>,
    o: DeviceBuffer<'c>,
    qg: DeviceBuffer<'c>,
    q: DeviceBuffer<'c>,
    gate: DeviceBuffer<'c>,
    kk: DeviceBuffer<'c>,
    iq: DeviceBuffer<'c>,
    ao: DeviceBuffer<'c>,
    /// attention_heads_rows' scores for ATT_ROWS tokens (heads x cap doubles each)
    att: DeviceBuffer<'c>,
    rl: DeviceBuffer<'c>,
    /// sel_rows() tokens' indexer queries (normalized, rotated), block scores and selected positions (`attention_batch`)
    sel_iqn: DeviceBuffer<'c>,
    sel_scores: DeviceBuffer<'c>,
    sel_toks: DeviceBuffer<'c>,
    sgl: DeviceBuffer<'c>,
    sg: DeviceBuffer<'c>,
    su: DeviceBuffer<'c>,
    sa: DeviceBuffer<'c>,
    so: DeviceBuffer<'c>,
    logits: DeviceBuffer<'c>,
    /// the grouped routed experts' tables (`moe_batch`), and per (token, routing rank): gate+up outputs,
    /// activations, outputs
    gtab: DeviceBuffer<'c>,
    eg: DeviceBuffer<'c>,
    act: DeviceBuffer<'c>,
    eo: DeviceBuffer<'c>,
}

/// The buffers of a `BatchBufs` that do not grow with the batch (`make_bb`'s `fixed`).
struct FixedBufs<'c> {
    att: DeviceBuffer<'c>,
    sel_iqn: DeviceBuffer<'c>,
    sel_scores: DeviceBuffer<'c>,
    sel_toks: DeviceBuffer<'c>,
    logits: DeviceBuffer<'c>,
}

/// Floats a row of `BatchBufs` takes, all the buffers that grow with the batch.
fn bb_row_floats(c: &shadow_model::forward::Config) -> usize {
    let (h, cs, k) = (c.hidden, c.streams, c.n_expert_used);
    let ch = cs * h;
    let kd = c.ssm_k_heads * c.ssm_head_dim;
    let vd = c.ssm_v_heads * c.ssm_head_dim;
    let (qd, kvd) = (c.n_head * c.head_dim, c.n_kv_head * c.head_dim);
    if std::env::var("SHADOW_ROW_INVENTORY").is_ok_and(|v| v == "1") {
        let parts = [("r,rn,up (3 x ch)", 3 * ch), ("u + y (2 x h)", 2 * h), ("g", cs), ("down (hc_rank)", c.hc_rank), ("qkv + x (2 x (2kd+vd))", 2 * (2 * kd + vd)), ("z + o (2 x vd)", 2 * vd),
                     ("beta + alpha", 2 * c.ssm_v_heads), ("qg + q + gate + ao", 2 * qd + qd + qd + qd), ("kk (kvd)", kvd), ("iq (idx_heads*idx_dim)", c.idx_heads * c.idx_dim),
                     ("rl + sgl (n_expert + 1)", c.n_expert + 1), ("sg + su + sa (3 x shared_ff)", 3 * c.shared_ff), ("so (h)", h), ("gtab (64 k)", 64 * k),
                     ("eg (2 k ff)", 2 * k * c.expert_ff), ("act (k ff)", k * c.expert_ff), ("eo (k h)", k * h)];
        let total: usize = parts.iter().map(|p| p.1).sum();
        eprintln!("ROWS: hidden {h} streams {cs} k {k} expert_ff {} shared_ff {} qd {qd} kvd {kvd} kd {kd} vd {vd}: {total} floats = {:.0} KB a row", c.expert_ff, c.shared_ff, total as f64 * 4.0 / 1024.0);
        for (n, f) in parts {
            eprintln!("ROWS   {n:<34} {f:>7} floats {:>5.1}%", 100.0 * f as f64 / total as f64);
        }
    }
    let (delta_set, attn_set) = (2 * (2 * kd + vd) + 2 * vd, 2 * qd + qd + qd + qd);
    // a layer is DeltaNet or attention, never both: the two sets of per-row buffers share one region (`alias_rows`), and `up`, which lives only inside a hyper-connection
    // read, shares `eg`'s, which lives only inside the experts
    let sets = if alias_sets() { delta_set.max(attn_set) } else { delta_set + attn_set };
    let up = if alias_up() { 0 } else { ch };
    2 * ch + up + 2 * h + cs + c.hc_rank + sets + 2 * c.ssm_v_heads + kvd
        + c.idx_heads * c.idx_dim + c.n_expert + 1 + 3 * c.shared_ff + h + 64 * k + 2 * k * c.expert_ff + k * c.expert_ff + k * h
}

/// Whether the per-row buffers of the two layer kinds (and `up` / `eg`) share memory: on by default (`SHADOW_ALIAS_ROWS=0` off, `sets` / `up` one of the two): +9-12% on 9-15k prompts, `--exact` byte-identical, but the fast modes agree less with `--exact` (0.219 against 0.339, docs/STATUS.md 2026-11-04); a pipelined session never.
static ALIAS_ROWS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(3);
fn alias_sets() -> bool {
    ALIAS_ROWS.load(std::sync::atomic::Ordering::Relaxed) & 1 != 0
}
fn alias_up() -> bool {
    ALIAS_ROWS.load(std::sync::atomic::Ordering::Relaxed) & 2 != 0
}

/// `BatchBufs` for `rows` tokens, each buffer from `take(floats)`, with the buffers that do not grow with the batch given.
fn make_bb<'c>(c: &shadow_model::forward::Config, rows: usize,
               take: &mut dyn FnMut(usize) -> Result<DeviceBuffer<'c>, String>, fixed: FixedBufs<'c>) -> Result<BatchBufs<'c>, String> {
    let (h, cs, k) = (c.hidden, c.streams, c.n_expert_used);
    let ch = cs * h;
    let kd = c.ssm_k_heads * c.ssm_head_dim;
    let vd = c.ssm_v_heads * c.ssm_head_dim;
    let (qd, kvd) = (c.n_head * c.head_dim, c.n_kv_head * c.head_dim);
    let id = c.idx_dim;
    let mut ab = |n: usize| take(rows * n);
    let (delta_set, attn_set) = (2 * (2 * kd + vd) + 2 * vd, 2 * qd + qd + qd + qd);
    let (alias, alias_up) = (alias_sets(), alias_up());
    // the two sets from one region when aliased: each buffer is its own rows x n array, laid end to end from the region's start
    // each carved buffer starts on a 256-byte boundary, as the separate allocations do (the tensor-core and vector loads need it; a row count that is no multiple of 64 breaks it otherwise), so the region has up to 4 x 63 floats of padding
    let region = if alias { Some(ab(delta_set.max(attn_set) + 256usize.div_ceil(rows))?) } else { None };
    let carve = |off: &mut usize, n: usize, region: &Option<DeviceBuffer<'c>>, ab: &mut dyn FnMut(usize) -> Result<DeviceBuffer<'c>, String>| -> Result<DeviceBuffer<'c>, String> {
        match region {
            // SAFETY: inside the region, which the BatchBufs keeps for as long as these views live; the two sets are never in use at once (`alias_rows`)
            Some(r) => { let b = unsafe { r.subview(*off * 4, rows * n * 4) }; *off += (rows * n).next_multiple_of(64); Ok(b) }
            None => ab(n),
        }
    };
    let mut off = 0usize;
    let qkv = carve(&mut off, 2 * kd + vd, &region, &mut ab)?;
    let x = carve(&mut off, 2 * kd + vd, &region, &mut ab)?;
    let z = carve(&mut off, vd, &region, &mut ab)?;
    let o = carve(&mut off, vd, &region, &mut ab)?;
    let mut off = 0usize;
    let qg = carve(&mut off, 2 * qd, &region, &mut ab)?;
    let q = carve(&mut off, qd, &region, &mut ab)?;
    let gate = carve(&mut off, qd, &region, &mut ab)?;
    let ao = carve(&mut off, qd, &region, &mut ab)?;
    let eg = ab(2 * k * c.expert_ff)?;
    // SAFETY: `up` is read and written only between a hyper-connection's down projection and its mix, `eg` only inside the experts: never at once (`alias_rows`)
    let up = if alias_up { unsafe { eg.subview(0, rows * ch * 4) } } else { ab(ch)? };
    Ok(BatchBufs {
        _region: region,
        r: ab(ch)?, rn: ab(ch)?, u: ab(h)?, g: ab(cs)?, y: ab(h)?, down: ab(c.hc_rank)?, up,
        qkv, x, z, beta: ab(c.ssm_v_heads)?, alpha: ab(c.ssm_v_heads)?, o,
        qg, q, gate, att: fixed.att, kk: ab(kvd)?, iq: ab(c.idx_heads * id)?, ao,
        rl: ab(c.n_expert)?, sgl: ab(1)?,
        sel_iqn: fixed.sel_iqn, sel_scores: fixed.sel_scores, sel_toks: fixed.sel_toks,
        sg: ab(c.shared_ff)?, su: ab(c.shared_ff)?, sa: ab(c.shared_ff)?, so: ab(h)?, logits: fixed.logits,
        gtab: ab(64 * k)?, eg, act: ab(k * c.expert_ff)?, eo: ab(k * h)?,
    })
}

pub struct GpuSession<'c, 'm> {
    ctx: &'c Context,
    m: &'m Model,
    mvp: Program<'c>,
    fw: Program<'c>,
    w: HashMap<String, DeviceBuffer<'c>>,
    v: HashMap<String, DeviceBuffer<'c>>,
    b: Bufs<'c>,
    bb: BatchBufs<'c>,
    replay: Replay<'c>,
    /// `delta_chunk_prep`'s output for up to BIG_BATCH tokens: 8,448 floats a (64-token chunk, value head)
    dscr: DeviceBuffer<'c>,
    /// with `--tensor-prompt` (`SHADOW_PREFILL_DEVPLAN=0` off): a long prompt's MoE is routed and planned on the device (`dev_plan`); these are its scratch buffers:
    /// each token's chosen experts, the users of each expert, each expert's first group, and the host's per-layer expert tables (copied flags, addresses)
    /// the residual streams of every row of a pass bigger than `rows` (see `sub_rows_cap`), one buffer of `rows` rows a part (carved wherever the lent ranges have room:
    /// a part is a sub-chunk's storage; the first sub-chunk, the remainder, sits at the end of part 0); `bb.r` is then a view of the sub-chunk being run
    rparts: Vec<DeviceBuffer<'c>>,
    /// `prefill` wants the last row's logits only: `batch` skips the output projection of the rows before it (0.2 s of a 15k prompt)
    logits_last: bool,
    /// whether the sub-chunk being run is the pass's last (the layer's copy pipeline advances once a layer, after it)
    sub_last: bool,
    sub_first: bool,
    devplan: bool,
    dp_tidx: DeviceBuffer<'c>,
    dp_cnt: DeviceBuffer<'c>,
    dp_gstart: DeviceBuffer<'c>,
    dp_copied: DeviceBuffer<'c>,
    dp_eaddr: DeviceBuffer<'c>,
    /// `SHADOW_KV_LEND=1`: the KV window's buffers are virtual-memory regions, and a long prompt borrows the unfilled tail of them (`lend_kv`) for its batch buffers
    kv_regions: Vec<crate::kvvmm::VmmRegion<'c>>,
    kv_extra: Option<DeviceBuffer<'c>>,
    /// the address range `kv_extra` lives in: fixed, so the CUDA graphs captured over the big batch's buffers stay valid from one prompt to the next
    kv_extra_reg: Option<crate::kvvmm::VmmRegion<'c>>,
    prompt_hint: usize,
    /// rows of the batch buffers `lend_kv` made room for (BIG_BATCH or BIG_BATCH_MAX)
    kv_rows: usize,
    /// `kv_extra` also holds the two staging halves (the cache arena is too small to lend them: windows past ~40k)
    kv_halves: bool,
    /// `SHADOW_ROUTER_BIAS=x`: x is added to the router logits of the experts the VRAM cache holds when the top-k is chosen (not to their weights)
    router_bias: f32,
    delta: Vec<Option<(DeviceBuffer<'c>, DeviceBuffer<'c>)>>,
    attn: Vec<Option<Attn<'c>>>,
    ple_state: DeviceBuffer<'c>,
    tokens: Vec<u32>,
    /// Pictures: (first token index, rows, the rows as n x hidden floats); the embedding of a token inside one is its row (`set_pictures`).
    pictures: Vec<(usize, usize, usize, std::sync::Arc<Vec<f32>>)>,
    /// The rotary position table (token index -> t, h, w; forward.cu `g_rope_tab`), made with the first picture; `rope_on`: the kernels read it.
    rope_tab: Option<DeviceBuffer<'c>>,
    rope_on: bool,
    cap: usize,
    pub trace: Option<BTreeMap<String, Vec<f32>>>,
    pub stop_after: Option<usize>,
    /// Bytes of weights uploaded to the device at the start, and expert slices uploaded per use since.
    pub resident_bytes: usize,
    pub expert_uploads: usize,
    cache: ExpertCache<'c>,
    /// Expert slots of `cache.slot_bytes` for `moe_batch`'s overflow misses (not cached), and how many.
    batch_scratch: Option<(DeviceBuffer<'c>, usize)>,
    ram: RamTier<'c>,
    copy_stream: Stream<'c>,
    /// a second copy stream for the staged halves of a long prompt (`SHADOW_COPY_STREAMS=2`: the experts of a layer are split between the two)
    copy_stream2: Stream<'c>,
    copy_ev2: Event<'c>,
    copy_event: Event<'c>,
    staging: Option<PinnedRegion<'c>>,
    /// This layer's drive reads, put off until all its misses are known so `read_drive` can issue them together.
    pending: Vec<PendingRead>,
    /// How often each (layer, expert) was routed to by this session (`--record-routing` saves it).
    pub routing: RoutingCounts,
    prof: Option<Prof>,
    /// `SHADOW_ROUTE_LOG=FILE`: every MoE layer's routed experts, in order, written with the tokens when the session
    /// drops (tools/spec_probe.py reads it).
    route_log: Option<(String, Vec<u16>)>,
    /// The hybrid MoE's CPU side (`Tiers::cpu_experts`), and how many experts it has computed.
    cpu: Option<(shadow_model::cpu_expert::ExpertPool, usize)>,
    pub cpu_served: u64,
    /// the verify pass's CPU experts: how many joins, the host's seconds waiting in them, and the seconds from each
    /// job's submit to its result (the job is out of the host's way until the join)
    cpu_join: (u64, f64, f64),
    /// seconds of host time choosing and submitting the CPU experts / uploading their outputs
    cpu_host: (f64, f64),
    tensor_prompt: bool,
    /// the tail of the VRAM expert cache, lent to a long prompt as staging (`lend_cache` / `restore_cache`)
    lent: Option<Lent>,
    sp: Option<StagePipe<'c>>,
    /// rows of `bb` now (`MAX_BATCH`, or `BIG_BATCH` while a long prompt is read), and the small buffers while the big ones are in
    rows: usize,
    small_bb: Option<BatchBufs<'c>>,
    /// `--pipeline`: a verify pass as two token groups (session_pipeline.rs); `pipe_active` while one runs, `row_shift` the
    /// first row of the group whose view of the batch buffers is current, `ticket_stage` the staging mark of each group's
    /// queued router read
    pipeline: bool,
    pipe_active: bool,
    pipelined_passes: u64,
    /// the batched MoE's host latency from "router logits received" to the first expert kernel queued: layers, seconds
    plan_lat: (u64, f64),
    row_shift: usize,
    mapped_event2: Event<'c>,
    group_event: Event<'c>,
    ticket_stage: [usize; 2],
    prefetch: Option<Prefetch<'c>>,
    /// Mapped pinned memory the router logits are copied into by a kernel, and the event the host waits on for them:
    /// a device-to-host memcpy would queue in the copy engine behind `copy_stream`'s uploads.
    mapped: PinnedRegion<'c>,
    mapped_event: Event<'c>,
    /// Small uploads (embeddings, expert tables, n-gram rows) go through here: the host writes them into this mapped
    /// region and `copy_words` moves them on the default stream, so they never queue in the copy engine behind
    /// `copy_stream`.  `stage_at` is reset whenever the host has seen the default stream drain.
    stage: PinnedRegion<'c>,
    stage_ptr: *mut u8,
    stage_at: std::cell::Cell<usize>,
    /// Seconds the host spent waiting: for router logits, for `copy_stream`, and on the file (unsynchronized timing).
    pub waits: [f64; 3],
    /// `SHADOW_PREGATE_PROBE=1`: layer l+1's router applied to layer l's MoE input, against l+1's real choice.
    pregate: Option<Pregate<'c>>,
    /// `SHADOW_MTP_PROBE=1` with `--mtp`: every step drafts first and counts (drafts, hits) against its own argmax.
    mtp_probe: Option<(u64, u64)>,
    pub mtp: Option<crate::mtp::MtpModel<'c>>,
    /// The DFlash 2 drafter (`--dflash DIR`).
    pub dflash: Option<crate::dflash::DflashModel<'c>>,
    /// The most tokens `step_batch` takes through the layers together: `MAX_BATCH`, or `SHADOW_BATCH=n` (1: one token
    /// at a time, for comparisons).
    pub batch_max: usize,
    /// CUDA graphs of the launches between two router reads that do not depend on the position (a DeltaNet layer's
    /// hc glue, mixer and router), keyed by (layer, whether its first norm was fused into the previous write); captured
    /// on first use on `gstream`, which orders with the default stream.  `SHADOW_GRAPHS=0`: none.
    graphs: HashMap<(usize, bool), Graph<'c>>,
    /// The same for the batched step: (layer, fused norm, tokens, keeping the rollback's inputs).
    batch_graphs: HashMap<(usize, bool, usize, bool, usize, u64), Graph<'c>>,
    /// `SHADOW_FILE_VIEWS=1`: a long prompt's file-only experts are copied to the GPU straight from the OS file cache (read-only views of the model files, registered
    /// for a layer at a time) instead of being read into pinned buffers by the CPU first
    fv_on: bool,
    fv: Option<crate::fileview::FileMaps>,
    fv_regs: [crate::fileview::RangeSet; 2],
    graphs_on: bool,
    /// `matvec_cols_multi`'s job tables, by (layer, group, rollback inputs): built before a graph is captured.
    multi: HashMap<(usize, u8, bool), DeviceBuffer<'c>>,
    /// `SHADOW_FUSE` bits: 1 the merged projections, 2 the hyper-connection up+mix, 4 the delta+norm kernel (0: none).
    fuse: u8,
    /// `SHADOW_LTIME=1`: CUDA events between a batched pass's layer segments, summed by kind in the report.
    ltime: Option<std::cell::RefCell<Vec<(&'static str, crate::driver::Event<'c>)>>>,
    timing_no_sync: bool,
    /// Host seconds and calls in the n-gram rows' reads (`ple_at`).
    ple_host: std::cell::Cell<f64>,
    ple_calls: std::cell::Cell<u64>,
    stale_logits: HashMap<(usize, usize), Vec<Vec<f32>>>,
    /// `SHADOW_ZERO_COPY=1`: the single-token path's RAM-tier misses are read in place over PCIe, not copied.
    zero_copy: bool,
    /// BF16 / F32 matrices in the 8-at-a-time sum order the quantized ones use (vector loads; one token and a batch
    /// still agree bit for bit); `SHADOW_REF_ORDER=1`: the reference kernels' order.
    fast_order: bool,
    /// Verify passes: count, tokens, seconds, of them waiting for the GPU (router reads) and the file, host seconds
    /// in kernel launches, launches.
    verify_stats: [f64; 7],
    /// `Tiers::tail_skip` (0: off), and how many routed experts it has skipped.
    tail_skip: usize,
    pub tail_skipped: u64,
    /// `--accept-moe`: AcceptMoE's verifier expert sets; `accept_beta` its commitment exponent; `commit` per draft position
    /// (committed, tried) from the verify passes so far (`note_verify`).
    accept_moe: bool,
    accept_beta: f64,
    accept_scale: f64,
    accept_budget: f64,
    commit: Vec<(u32, u32)>,
    verify_resident: bool,
    verify_rank: usize,
    /// `routed_overlapped` instead of waiting on the host for a layer's copies (`SHADOW_OVERLAP=0`: off).
    overlap: bool,
    gstream: Stream<'c>,
    capturing: std::cell::Cell<bool>,
}

/// What the pre-gating probe has seen: the prediction waiting for each layer, and per-layer counts of (real experts,
/// predicted in the top 10, in the top 16, real misses, misses predicted in the top 16).
pub struct Pregate<'c> {
    buf: DeviceBuffer<'c>,
    pred: Vec<Option<Vec<usize>>>,
    pub counts: Vec<[u64; 5]>,
}

/// The pre-gated prefetch (`Tiers::pregate`): the next layer's router logits, what was copied up ahead and not yet
/// routed to, and whether the default stream still has to wait for `copy_event` before the next layer's experts run.
struct Prefetch<'c> {
    k: usize,
    buf: DeviceBuffer<'c>,
    ahead: std::collections::HashSet<(usize, usize)>,
    wait: bool,
    issued: u64,
    used: u64,
}

/// The mapped staging region for small uploads (`GpuSession::upload_to`).
const STAGE_BYTES: usize = 4 << 20;

/// A session's state at a token boundary (`GpuSession::snapshot`).
pub struct Snapshot<'c> {
    tokens: Vec<u32>,
    delta: Vec<Option<(Saved<'c>, Saved<'c>)>>,
    ple: Saved<'c>,
    kv: Vec<Option<SnapKv<'c>>>,
    nblocks: Vec<usize>,
    /// The draft layer's pending residual (and with the KV its K/V cells), when the session has one.
    mtp: Option<(Saved<'c>, bool, Option<(Saved<'c>, Saved<'c>)>)>,
}

struct SnapKv<'c> {
    k: Saved<'c>,
    v: Saved<'c>,
    raw: Saved<'c>,
    blocks: Saved<'c>,
    nblocks: usize,
}

/// The first bytes of a device buffer, kept on the device (fast, costs VRAM) or in host memory (costs a PCIe trip).
enum Saved<'c> {
    Dev(DeviceBuffer<'c>, usize),
    Host(Vec<u8>),
}

impl<'c> Saved<'c> {
    fn take(ctx: &'c Context, b: &DeviceBuffer<'c>, bytes: usize, host: bool) -> Result<Saved<'c>, String> {
        let e = |x: crate::driver::CudaError| x.to_string();
        if host {
            let mut v = vec![0u8; bytes];
            if bytes > 0 {
                b.download(&mut v).map_err(e)?;
            }
            Ok(Saved::Host(v))
        } else {
            let d = ctx.alloc(bytes.max(4)).map_err(e)?;
            d.copy_prefix_from(b, bytes).map_err(e)?;
            Ok(Saved::Dev(d, bytes))
        }
    }

    fn put(&self, b: &DeviceBuffer<'c>) -> Result<(), String> {
        let e = |x: crate::driver::CudaError| x.to_string();
        match self {
            Saved::Dev(d, n) => b.copy_prefix_from(d, *n).map_err(e),
            Saved::Host(v) if v.is_empty() => Ok(()),
            Saved::Host(v) => b.upload(v).map_err(e),
        }
    }

    fn len(&self) -> usize {
        match self {
            Saved::Dev(_, n) => *n,
            Saved::Host(v) => v.len(),
        }
    }
}

impl Snapshot<'_> {
    /// The tokens the snapshot was taken after.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The snapshot as bytes (a host snapshot only: `snapshot_to(_, true)`), for a file; `from_bytes` is the other way.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        fn put_u64(o: &mut Vec<u8>, v: usize) {
            o.extend_from_slice(&(v as u64).to_le_bytes());
        }
        fn put(o: &mut Vec<u8>, s: &Saved) -> Result<(), String> {
            let Saved::Host(v) = s else { return Err("a snapshot kept on the device cannot be written to a file".into()) };
            put_u64(o, v.len());
            o.extend_from_slice(v);
            Ok(())
        }
        let mut o = Vec::with_capacity(self.host_bytes() + (1 << 16));
        put_u64(&mut o, self.tokens.len());
        for t in &self.tokens {
            o.extend_from_slice(&t.to_le_bytes());
        }
        put_u64(&mut o, self.delta.len());
        for d in &self.delta {
            match d {
                Some((a, b)) => { o.push(1); put(&mut o, a)?; put(&mut o, b)?; }
                None => o.push(0),
            }
        }
        put(&mut o, &self.ple)?;
        put_u64(&mut o, self.kv.len());
        for k in &self.kv {
            match k {
                Some(k) => { o.push(1); put(&mut o, &k.k)?; put(&mut o, &k.v)?; put(&mut o, &k.raw)?; put(&mut o, &k.blocks)?; put_u64(&mut o, k.nblocks); }
                None => o.push(0),
            }
        }
        put_u64(&mut o, self.nblocks.len());
        for &n in &self.nblocks {
            put_u64(&mut o, n);
        }
        match &self.mtp {
            Some((a, flag, kv)) => {
                o.push(1);
                put(&mut o, a)?;
                o.push(*flag as u8);
                match kv {
                    Some((b, c)) => { o.push(1); put(&mut o, b)?; put(&mut o, c)?; }
                    None => o.push(0),
                }
            }
            None => o.push(0),
        }
        Ok(o)
    }

    pub fn from_bytes<'c>(b: &[u8]) -> Result<Snapshot<'c>, String> {
        struct R<'a> { b: &'a [u8], at: usize }
        impl R<'_> {
            fn take(&mut self, n: usize) -> Result<&[u8], String> {
                if self.at + n > self.b.len() {
                    return Err("the snapshot file is cut short".into());
                }
                let s = &self.b[self.at..self.at + n];
                self.at += n;
                Ok(s)
            }
            fn u64(&mut self) -> Result<usize, String> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()) as usize)
            }
            fn u8(&mut self) -> Result<u8, String> {
                Ok(self.take(1)?[0])
            }
            fn saved<'c>(&mut self) -> Result<Saved<'c>, String> {
                let n = self.u64()?;
                Ok(Saved::Host(self.take(n)?.to_vec()))
            }
        }
        let mut r = R { b, at: 0 };
        let nt = r.u64()?;
        let tokens: Vec<u32> = r.take(nt * 4)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let nd = r.u64()?;
        let mut delta = Vec::with_capacity(nd);
        for _ in 0..nd {
            delta.push(if r.u8()? == 1 { Some((r.saved()?, r.saved()?)) } else { None });
        }
        let ple = r.saved()?;
        let nk = r.u64()?;
        let mut kv = Vec::with_capacity(nk);
        for _ in 0..nk {
            kv.push(if r.u8()? == 1 {
                let (k, v, raw, blocks) = (r.saved()?, r.saved()?, r.saved()?, r.saved()?);
                Some(SnapKv { k, v, raw, blocks, nblocks: r.u64()? })
            } else {
                None
            });
        }
        let nn = r.u64()?;
        let mut nblocks = Vec::with_capacity(nn);
        for _ in 0..nn {
            nblocks.push(r.u64()?);
        }
        let mtp = if r.u8()? == 1 {
            let a = r.saved()?;
            let flag = r.u8()? == 1;
            let kv = if r.u8()? == 1 { Some((r.saved()?, r.saved()?)) } else { None };
            Some((a, flag, kv))
        } else {
            None
        };
        Ok(Snapshot { tokens, delta, ple, kv, nblocks, mtp })
    }

    /// Bytes this snapshot holds in host memory (what a pool of them costs in RAM).
    pub fn host_bytes(&self) -> usize {
        let h = |s: &Saved| if let Saved::Host(v) = s { v.len() } else { 0 };
        let mut n = h(&self.ple);
        for d in self.delta.iter().flatten() {
            n += h(&d.0) + h(&d.1);
        }
        for k in self.kv.iter().flatten() {
            n += h(&k.k) + h(&k.v) + h(&k.raw) + h(&k.blocks);
        }
        if let Some((a, _, kv)) = &self.mtp {
            n += h(a);
            if let Some((b, c)) = kv {
                n += h(b) + h(c);
            }
        }
        n
    }
}

/// An expert to read from the file into pinned memory and then copy to `dst`: into `RamTier`'s region at `off` when
/// `ring`, else into the staging buffer at `off`.
struct PendingRead {
    e: usize,
    ring: bool,
    off: usize,
    dst: [u64; 3],
}

/// Sizes and placement of the expert tiers below VRAM's dense weights.
#[derive(Default, Clone)]
pub struct Tiers {
    /// The VRAM expert cache in bytes; `None`: the free VRAM less `VRAM_RESERVE`.
    pub vram_cache: Option<usize>,
    /// The pinned RAM tier in bytes (0: none).  Refused if it would leave less than the RAM floor (`pinned::ram_floor`, `--ram-floor-gib`) available.
    pub ram_bytes: usize,
    /// Size the RAM tier as `pinned::auto_tier_bytes()` says, measured when the session fills it - after the dense
    /// weights are loaded and uploaded, so the figure is what is really left (`ram_bytes` is then ignored).
    pub ram_auto: bool,
    /// Which experts the RAM tier holds, best first; `None`: `placement::uniform_order`.
    pub order: Option<Vec<(usize, usize)>>,
    /// Path to tuned MTP head directory (e.g. models/mtp-basalt-ft1).
    pub mtp_dir: Option<std::path::PathBuf>,
    /// `--dflash DIR`: the DFlash 2 drafter (models/dflash2).
    pub dflash_dir: Option<std::path::PathBuf>,
    /// `--dflash-min-p P`: a DFlash chain stops after a draft the selector gives less than P (default 0: all).
    pub dflash_min_p: f32,
    /// The drafter's matrices as Q8_0 (half the VRAM, ~220 more expert slots); `--dflash-bf16` keeps them BF16.
    pub dflash_q8: bool,
    /// Hybrid MoE: up to this many of a layer's misses that the RAM tier holds are computed on the CPU instead of
    /// copied to the GPU (0: off), on `cpu_threads` workers.
    pub cpu_experts: usize,
    pub cpu_threads: usize,
    /// Pre-gated prefetch: layer l+1's router applied to layer l's MoE input; its top `pregate` experts that the RAM
    /// tier holds and VRAM does not are copied up while layer l computes (0: off).
    pub pregate: usize,
    /// `--pin-margin N`: keep the placement's hottest experts pinned in VRAM (and out of the RAM tier), N evictable
    /// slots a layer; `None` (the default): every slot is LRU.  On real routing, LRU over a layer's slots hits more:
    /// 70.7% vs 56.6% with N = 12 (docs/STATUS.md, tools/cache_sim.py).
    pub pin_margin: Option<usize>,
    /// `--admit-after N`: a miss gets a VRAM slot only from its (N+1)th miss on; the first N are read through scratch
    /// slots (kept out of the cache, so a one-off expert does not push a useful one out).  0: every miss is admitted.
    pub admit_after: usize,
    /// `--tail-skip R` (inexact, off by default; Strata's STRATA_ROUTE_TAIL_SKIP): a routed expert that is not in VRAM
    /// and that every token of a step or verify pass routes only at rank >= R (0-based, of the top 10) is neither
    /// copied nor computed - its contribution is zero and the others keep their weights.  Changes answers.
    pub tail_skip: usize,
    /// `--verify-resident` (inexact; AcceptMoE, arXiv 2608.02989): in a speculative verify pass, the drafted positions
    /// (all but the first) use only the routed experts already in VRAM, so the pass copies about what one step does.
    pub verify_resident: bool,
    /// `--verify-rank R` (inexact, implies `--verify-resident`): the drafted positions keep their top R routed experts even
    /// when they are not in VRAM (copied as usual) and drop only the non-resident ones below that; 0 = `--verify-resident`.
    pub verify_rank: usize,
    /// `--tensor-prompt` (or `SHADOW_TENSOR_PROMPT=1`): a prompt's Q6_K and IQ4_XS projections (16 or more inputs at once) run on the tensor cores with
    /// FP16 operands and FP32 accumulation (relative error ~3e-4 against the reference sums): inexact, so default mode only.
    pub tensor_prompt: bool,
    /// `--pipeline` (or `SHADOW_PIPELINE=1`): a verify pass as two token groups, one group's expert phase running under the
    /// other group's mixer (session_pipeline.rs).  Same arithmetic per token.
    pub pipeline: bool,
    /// `--accept-moe` (inexact; AcceptMoE, arXiv 2608.02989, docs/SPECULATION.md): a verify pass routes its drafted positions
    /// through one block-wide eligible expert set (the anchor's top-k plus the experts the committed positions demand, sized
    /// by the demand's effective rank, non-resident members pruned lowest demand first); each token still takes k experts,
    /// re-picked among the eligible ones.  `--accept-beta B` (default 1.0; the paper uses 0.5) sharpens the commitment weights.
    pub accept_moe: bool,
    pub accept_beta: f64,
    /// `--accept-scale S` (default 1): the set holds S x the effective rank of the demand beyond the anchor; `--accept-budget F`
    /// (default 1): the pruning may displace F x the assignments forming the set already displaced.
    pub accept_scale: f64,
    pub accept_budget: f64,
    /// `--fuse N`: the kernel fusions to use (`SHADOW_FUSE`'s bits; default none: measured 1.1% slower at 7, a tie at 0).
    pub fuse: Option<u8>,
}

impl Tiers {
    /// From command-line flags: `--expert-cache-gib G`, `--ram-tier-gib G|auto`, `--placement FILE` (SHRC), `--mtp DIR|off`.
    pub fn from_args(args: &[String]) -> Result<Tiers, String> {
        const GIB: f64 = (1u64 << 30) as f64;
        let flag = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1));
        let gib = |f: &str| -> Result<Option<usize>, String> {
            flag(f).map(|v| v.parse::<f64>().map(|g| (g * GIB) as usize).map_err(|_| format!("{f}: not a number: {v}")))
                .transpose()
        };
        if let Some(v) = flag("--vram-reserve-mib") {
            let mib: usize = v.parse().map_err(|_| format!("--vram-reserve-mib: not a number: {v}"))?;
            VRAM_RESERVE_SET.store(mib << 20, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(g) = gib("--ram-floor-gib")? {
            crate::pinned::set_ram_floor(g as u64);
            if (g as u64) < crate::pinned::RAM_FLOOR {
                eprintln!("RAM floor: {:.1} GiB (default {:.0}): a pinned tier cannot be paged out, and this PC has frozen hard at 0.4 GiB free",
                          g as f64 / GIB, crate::pinned::RAM_FLOOR as f64 / GIB);
            }
        }
        let order = match flag("--placement") {
            Some(p) => {
                let rc = RoutingCounts::load(std::path::Path::new(p)).map_err(|e| format!("--placement: {e}"))?;
                eprintln!("placement: {p}, {} tokens of routing", rc.tokens);
                Some(rc.ranked())
            }
            None => None,
        };
        let ram_auto = flag("--ram-tier-gib").map(String::as_str) == Some("auto");
        let ram_bytes = if ram_auto { 0 } else { gib("--ram-tier-gib")?.unwrap_or(0) };
        if (ram_auto || ram_bytes > 0) && order.is_none() {
            eprintln!("RAM tier: no --placement, so an arbitrary (uniform) choice of experts");
        }
        let mtp_dir = match flag("--mtp").map(String::as_str) {
            Some("off") => None,
            Some(p) => Some(std::path::PathBuf::from(p)),
            None => None,
        };
        let num = |f: &str, d: usize| -> Result<usize, String> {
            flag(f).map_or(Ok(d), |v| v.parse().map_err(|_| format!("{f}: not a number: {v}")))
        };
        let dflash_dir = flag("--dflash").filter(|p| p.as_str() != "off").map(std::path::PathBuf::from);
        Ok(Tiers { vram_cache: gib("--expert-cache-gib")?, ram_bytes, ram_auto, order, mtp_dir, dflash_dir,
                   dflash_q8: !args.iter().any(|a| a == "--dflash-bf16"),
                   dflash_min_p: flag("--dflash-min-p").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                   cpu_experts: num("--cpu-experts", 0)?, cpu_threads: num("--cpu-threads", 8)?,
                   pregate: num("--pregate", 0)?,
                   tail_skip: num("--tail-skip", 0)?,
                   verify_resident: args.iter().any(|a| a == "--verify-resident" || a == "--verify-rank"),
                   accept_moe: args.iter().any(|a| a == "--accept-moe"),
                   accept_beta: flag("--accept-beta").and_then(|v| v.parse().ok()).unwrap_or(1.0),
                   accept_scale: flag("--accept-scale").and_then(|v| v.parse().ok()).unwrap_or(1.0),
                   accept_budget: flag("--accept-budget").and_then(|v| v.parse().ok()).unwrap_or(1.0),
                   pipeline: args.iter().any(|a| a == "--pipeline") || std::env::var("SHADOW_PIPELINE").is_ok_and(|v| v == "1"),
                   tensor_prompt: args.iter().any(|a| a == "--tensor-prompt") || std::env::var("SHADOW_TENSOR_PROMPT").is_ok_and(|v| v == "1"),
                   verify_rank: flag("--verify-rank").and_then(|v| v.parse().ok()).unwrap_or(0),
                   fuse: flag("--fuse").and_then(|v| v.parse().ok()),
                   admit_after: num("--admit-after", 0)?,
                   pin_margin: flag("--pin-margin").map(|v| v.parse().map_err(|_| format!("--pin-margin: not a number: {v}")))
                       .transpose()? })
    }
}

/// Routed experts in pinned RAM (stage 3's second tier): each expert's [gate | up | down] slices packed together at
/// their exact layer size, filled at the start from the placement order.  The cold tail slots form an exact-size
/// Clock-replaced promotion ring so any out-of-profile expert read from the file is retained in pinned RAM.
struct RamTier<'c> {
    region: Option<PinnedRegion<'c>>,
    at: HashMap<(usize, usize), usize>,
    ring_by_off: HashMap<usize, usize>,
    ring_offs: Vec<usize>,
    ring_sizes: Vec<usize>,
    ring_owner: Vec<Option<(usize, usize)>>,
    ring_ref: Vec<bool>,
    ring_hand: usize,
    /// Offsets in use since the copies were last waited for (a copy from them in flight, or a file read into them
    /// pending): the Clock hand must not hand them out again, even when it wraps.  Cleared by `moe` per layer.
    claimed: std::collections::HashSet<usize>,
    /// Experts the kernels read in place (`SHADOW_ZERO_COPY=1`).
    zc_reads: u64,
    /// Misses served from this tier, and from the file.
    served: u64,
    drive: u64,
    fill_secs: f64,
}

impl<'c> RamTier<'c> {
    fn fill(ctx: &'c Context, m: &Model, bytes: usize, order: Option<Vec<(usize, usize)>>,
            vram_pinned: &[(usize, usize, u64)], arena_ptr: u64) -> Result<RamTier<'c>, String> {
        let mut t = RamTier {
            region: None, at: HashMap::new(), ring_by_off: HashMap::new(), ring_offs: Vec::new(),
            ring_sizes: Vec::new(), ring_owner: Vec::new(), ring_ref: Vec::new(), ring_hand: 0,
            claimed: Default::default(), zc_reads: 0, served: 0, drive: 0, fill_secs: 0.0,
        };
        if bytes == 0 {
            return Ok(t);
        }
        let c = &m.cfg;
        let order = order.unwrap_or_else(|| uniform_order(c.n_layer, c.n_expert));
        let names = |l: usize| ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(|r| format!("blk.{l}.{r}.weight"));
        let size = |l: usize| names(l).iter().map(|n| m.expert_bytes(n)).sum::<usize>().next_multiple_of(256);
        let pinned_set: std::collections::HashSet<(usize, usize)> = vram_pinned.iter().map(|&(l, e, _)| (l, e)).collect();
        let mut chosen = Vec::new();
        let mut total = 0usize;
        for &(l, e) in &order {
            if !c.is_moe(l) || pinned_set.contains(&(l, e)) {
                continue;
            }
            let s = size(l);
            if total + s > bytes {
                break;
            }
            t.at.insert((l, e), total);
            chosen.push((l, e, total, s));
            total += s;
        }
        if chosen.is_empty() {
            return Ok(t);
        }
        let ring_count = (chosen.len() / 8).min(1536);
        let ring_start_idx = chosen.len().saturating_sub(ring_count);
        for &(l, e, off, s) in &chosen[ring_start_idx..] {
            let idx = t.ring_offs.len();
            t.ring_by_off.insert(off, idx);
            t.ring_offs.push(off);
            t.ring_sizes.push(s);
            t.ring_owner.push(Some((l, e)));
            t.ring_ref.push(false);
        }
        let t0 = std::time::Instant::now();
        let mut region = PinnedRegion::new(ctx, total)?;
        let e_err = |x: crate::driver::CudaError| x.to_string();
        // First pass: if VRAM has exclusively pinned hot experts, stream them sequentially through `region` into VRAM.
        if !vram_pinned.is_empty() && arena_ptr != 0 {
            let mut v_chosen = Vec::with_capacity(vram_pinned.len());
            let mut v_off = 0usize;
            for &(l, ep, slot_off) in vram_pinned {
                let s = size(l);
                if v_off + s <= total {
                    v_chosen.push((l, ep, v_off, slot_off));
                    v_off += s;
                }
            }
            let mut v_reads = Vec::with_capacity(v_chosen.len() * 3);
            for &(l, ep, r_off, _) in &v_chosen {
                let mut o = r_off;
                for (part, n) in names(l).iter().enumerate() {
                    let nb = m.expert_bytes(n);
                    v_reads.push((l, part, ep, o, nb));
                    o += nb;
                }
            }
            v_reads.sort_unstable_by_key(|&(l, part, ep, _, _)| (l, part, ep));
            for (l, part, ep, o, nb) in v_reads {
                m.expert_into(&names(l)[part], ep, &mut region.as_mut_slice()[o..o + nb]);
            }
            ctx.bind().map_err(e_err)?;
            let src = region.as_slice();
            for (l, _ep, r_off, slot_off) in v_chosen {
                let tot: usize = names(l).iter().map(|n| m.expert_bytes(n)).sum();
                let r = unsafe {
                    (ctx.api.cuMemcpyHtoDAsync)(arena_ptr + slot_off, src[r_off..r_off + tot].as_ptr() as *const std::ffi::c_void,
                                                tot, std::ptr::null_mut())
                };
                assert_eq!(r, 0, "cuMemcpyHtoDAsync vram_pinned failed ({r})");
            }
            ctx.synchronize().map_err(e_err)?;
        }
        // Sort the reads by (layer, tensor, expert) so file I/O is strictly sequential regardless of placement order.
        let mut reads: Vec<(usize, usize, usize, usize, usize)> = Vec::with_capacity(chosen.len() * 3);
        for (l, e, off, _) in chosen {
            let mut o = off;
            for (part, n) in names(l).iter().enumerate() {
                let n_bytes = m.expert_bytes(n);
                reads.push((l, part, e, o, n_bytes));
                o += n_bytes;
            }
        }
        reads.sort_unstable_by_key(|&(l, part, e, _, _)| (l, part, e));
        let mut since_check = 0usize;
        for (l, part, e, o, n_bytes) in reads {
            let n = &names(l)[part];
            m.expert_into(n, e, &mut region.as_mut_slice()[o..o + n_bytes]);
            since_check += n_bytes;
            if since_check >= 256 << 20 {
                region.check_floor()?; // the file cache and everything else move while we fill
                since_check = 0;
            }
        }
        region.check_floor()?;
        t.fill_secs = t0.elapsed().as_secs_f64();
        t.region = Some(region);
        Ok(t)
    }

    /// Picks a Clock victim slot in the tail promotion ring whose byte size matches `req_size`, evicting its old owner.
    fn promote_slot(&mut self, key: (usize, usize), req_size: usize) -> Option<usize> {
        let len = self.ring_offs.len();
        if len == 0 {
            return None;
        }
        for _ in 0..(2 * len) {
            let h = self.ring_hand;
            self.ring_hand = (h + 1) % len;
            if self.ring_sizes[h] != req_size || self.claimed.contains(&self.ring_offs[h]) {
                continue;
            }
            if self.ring_ref[h] {
                self.ring_ref[h] = false;
                continue;
            }
            if let Some(old) = self.ring_owner[h].take() {
                self.at.remove(&old);
            }
            let off = self.ring_offs[h];
            self.claimed.insert(off);
            self.ring_owner[h] = Some(key);
            self.ring_ref[h] = true;
            self.at.insert(key, off);
            return Some(off);
        }
        None
    }
}

/// VRAM left free for everything else when the expert cache sizes itself (the desktop, the driver, other programs).
// 96 MiB: the display runs off the integrated GPU here, so the RTX card's VRAM is Shadow's alone (384 MiB left 288 MiB, ~147 expert slots, unused:
// +4.5% decode, 52.26 against 50.01 tok/s, 3 runs; 16 MiB tied with 96).  `--vram-reserve-mib` raises it for a card that also drives the display.
pub const VRAM_RESERVE: usize = 96 << 20;
/// `--vram-reserve-mib`: the VRAM left unused by the expert cache, in bytes (`usize::MAX`: `VRAM_RESERVE`).
static VRAM_RESERVE_SET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
fn vram_reserve() -> usize {
    match VRAM_RESERVE_SET.load(std::sync::atomic::Ordering::Relaxed) {
        usize::MAX => VRAM_RESERVE,
        b => b,
    }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Rows a block of a matvec_cols kernel covers (8 warps): the reference-order kernels take one row a warp below
/// COLS_REF_SPLIT (2048) rows (matvec.cu), the others COLS_ROWS (4).
fn cols_rows_per_block(entry: &str, rows: usize) -> u32 {
    let reference = entry.starts_with("matvec_cols_ref") || entry.starts_with("hc_down_inject_cols");
    if reference && rows < 2048 { 8 } else if reference { 32 } else { COLS_PLAIN_ROWS }
}

/// Rows a block of the plain `matvec_cols_*` kernels (and `matvec_cols_multi`) covers: 8 warps x `COLS_PLAIN` (matvec.cu) rows (tests/matvec.rs assumes 8).
const COLS_PLAIN_ROWS: u32 = 8 * 1;

/// `moe_batch`'s grouped routed experts for tokens start..end, as one upload: per phase (0: experts resident before
/// this group's copies, 1: the ones being copied) the groups - an expert and up to 8 of its tokens - as matrix
/// addresses [gate.., up.., down..] and 17-u32 records [gate.., up.., down..] (count, input rows, output rows), then
/// each token's routing weights.  Pair (token j, rank i) is row j * k + i of eg's gate half, of eg's up half (`up`
/// rows on), of act and of eo; gate and up read token j's row of u, down reads the pair's row of act.
struct GroupedPlan {
    bytes: Vec<u8>,
    /// per phase: (groups, byte offset of its matrix addresses, byte offset of its records)
    phase: [(usize, u64, u64); 2],
    /// byte offset of the routing weights (k f32 per token, from token 0)
    weights: u64,
    /// byte offset of each token's expert count (u32, from token 0)
    counts: u64,
    /// tokens a group holds (8: `matvec_grouped_*`; 32: `mma_grouped_*`)
    chunk_n: usize,
}

#[allow(clippy::too_many_arguments)]
/// `route` (or `route_biased`) for every token's logits, spread over the CPU's threads when there are many (a long prompt's pass: 4,000+ tokens x 512 f64 exponentials
/// was ~27 ms of one thread a layer with the GPU idle); each token is routed by the same code on one thread, so the result is bit for bit the sequential one.
fn route_all(logits: &[Vec<f32>], k: usize, bias: Option<(&HashMap<(usize, usize), usize>, usize, f32)>) -> Vec<Vec<(usize, f64)>> {
    let one = |lg: &Vec<f32>| match bias {
        Some((map, l, b)) => route_biased(lg, &|e| if map.contains_key(&(l, e)) { b } else { 0.0 }, k),
        None => route(lg, k),
    };
    if logits.len() < 256 {
        return logits.iter().map(one).collect();
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let chunk = logits.len().div_ceil(threads);
    let mut out: Vec<Vec<(usize, f64)>> = vec![vec![]; logits.len()];
    std::thread::scope(|s| {
        for (lc, oc) in logits.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let one = &one;
            s.spawn(move || {
                for (lg, o) in lc.iter().zip(oc.iter_mut()) {
                    *o = one(lg);
                }
            });
        }
    });
    out
}

/// `SHADOW_HEAPCHECK=1` (debugging a heap-corruption crash): validates the process heap and reports the first `what` after which it is corrupt.
#[cfg(windows)]
fn heap_check(what: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static BAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static VERBOSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("SHADOW_HEAPCHECK").is_ok_and(|v| v == "1" || v == "2")) || BAD.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if *VERBOSE.get_or_init(|| std::env::var("SHADOW_HEAPCHECK").is_ok_and(|v| v == "2")) {
        eprintln!("HEAPCHECK ok before: {what}");
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetProcessHeap() -> *mut std::ffi::c_void;
        fn HeapValidate(heap: *mut std::ffi::c_void, flags: u32, mem: *const std::ffi::c_void) -> i32;
    }
    // SAFETY: plain Win32 calls on the process heap.
    if unsafe { HeapValidate(GetProcessHeap(), 0, std::ptr::null()) } == 0 {
        BAD.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("HEAPCHECK: the process heap is CORRUPT after: {what}");
    }
}
#[cfg(not(windows))]
fn heap_check(_what: &str) {}

fn grouped_plan(tops: &[Vec<(usize, f64)>], start: usize, end: usize, set: &[usize], k: usize, up: usize,
                addr: impl Fn(usize) -> u64, copied: impl Fn(usize) -> bool, part: [u64; 3], chunk_n: usize) -> GroupedPlan {
    // (expert, [(token, rank)]) per phase, at most `chunk_n` (8, or 32 for the tensor-core kernels) tokens a group
    let mut groups: [Vec<(usize, Vec<(usize, usize)>)>; 2] = [vec![], vec![]];
    // each expert's (token, rank) pairs in token order, from one walk over the tokens (it was a search of every token for every expert)
    let mut by_expert: Vec<Vec<(usize, usize)>> = vec![vec![]; set.iter().copied().max().map_or(0, |e| e + 1)];
    for j in start..end {
        for (i, &(x, _)) in tops[j].iter().enumerate() {
            if let Some(u) = by_expert.get_mut(x) {
                // the first rank an expert has in a token's list (a list names an expert once)
                if u.last().is_none_or(|l| l.0 != j) {
                    u.push((j, i));
                }
            }
        }
    }
    for &e in set {
        let users = &by_expert[e];
        for chunk in users.chunks(chunk_n) {
            groups[copied(e) as usize].push((e, chunk.to_vec()));
        }
    }
    let mut addrs: Vec<u8> = vec![];
    let mut recs: Vec<u32> = vec![];
    let mut spans = [(0usize, 0usize, 0usize); 2];
    for (ph, gs) in groups.iter().enumerate() {
        spans[ph] = (gs.len(), addrs.len(), recs.len());
        for (pi, p) in part.iter().enumerate() {
            for (e, users) in gs {
                addrs.extend_from_slice(&(addr(*e) + p).to_le_bytes());
                let mut r = vec![0u32; 1 + 2 * chunk_n];
                r[0] = users.len() as u32;
                for (u, &(j, i)) in users.iter().enumerate() {
                    let pair = (j * k + i) as u32;
                    (r[1 + u], r[1 + chunk_n + u]) = match pi {
                        0 => (j as u32, pair),
                        1 => (j as u32, up as u32 + pair),
                        _ => (pair, pair),
                    };
                }
                recs.extend_from_slice(&r);
            }
        }
    }
    let rec_at = addrs.len();
    let mut bytes = addrs;
    bytes.extend(recs.iter().flat_map(|r| r.to_le_bytes()));
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let weights = bytes.len() as u64;
    for top in tops.iter().take(end) {
        for i in 0..k {
            let w = top.get(i).map_or(0.0, |&(_, w)| w as f32);
            bytes.extend_from_slice(&w.to_le_bytes());
        }
    }
    let counts = bytes.len() as u64;
    for top in tops.iter().take(end) {
        bytes.extend_from_slice(&(top.len() as u32).to_le_bytes());
    }
    let phase = spans.map(|(n, a, r)| (n, a as u64, (rec_at + 4 * r) as u64));
    GroupedPlan { bytes, phase, weights, counts, chunk_n }
}

/// Row j of a [rows][stride] f32 buffer.
fn row(b: &DeviceBuffer, stride: usize, j: usize) -> u64 {
    b.ptr + (j * stride * 4) as u64
}

fn is_small(name: &str) -> bool {
    name.contains("norm") || name.ends_with(".ssm_a") || name.ends_with(".ssm_dt.bias")
        || name.ends_with("conv1d.weight") || name.ends_with(".attn_sinks.weight")
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
        Self::with_tiers(ctx, m, cap, Tiers { vram_cache: cache_bytes, ..Tiers::default() })
    }

    /// As `new`, with every tier's size and the RAM tier's placement.
    pub fn with_tiers(ctx: &'c Context, m: &'m Model, cap: usize, tiers: Tiers)
                      -> Result<GpuSession<'c, 'm>, String> {
        let cache_bytes = tiers.vram_cache;
        let c = &m.cfg;
        let e = |x: crate::driver::CudaError| x.to_string();
        let vmap = std::env::var("SHADOW_VRAM_MAP").is_ok();
        let vm = |s: &str| {
            if vmap {
                eprintln!("vram map: {s}: {} MiB free", ctx.mem_info().map_or(0, |x| x.0 >> 20));
            }
        };
        vm("start");
        let mvp = Program::load(ctx, file("matvec.cu").ok_or("matvec.cu")?)?;
        let fw = Program::load(ctx, file("forward.cu").ok_or("forward.cu")?)?;
        let mut w = HashMap::new();
        let mut v = HashMap::new();
        let mut resident_bytes = 0;
        let mut names: Vec<&str> = m.dense_names().collect();
        names.sort_unstable();
        for name in names {
            if name == "token_embd.weight" || name.ends_with(".scale") {
                continue; // the embedding row is dequantized on the host; .scale scalars/vectors stay on the host
            }
            let (_, bytes) = m.dense(name);
            let buf = ctx.alloc(bytes.len()).map_err(e)?;
            buf.upload(bytes).map_err(e)?;
            resident_bytes += bytes.len();
            w.insert(name.to_string(), buf);
            if is_small(name) {
                let f = f32s(&m.vec(name));
                let sb = ctx.alloc(f.len()).map_err(e)?;
                sb.upload(&f).map_err(e)?;
                v.insert(name.to_string(), sb);
            }
        }
        vm("dense weights on the GPU");
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
        let k = c.n_expert_used;
        let max_ff = ((k + 2) * c.expert_ff).max(c.dense_ff).max(c.shared_ff);
        let scratch_exp = if cache_bytes == Some(0) { (k * max_expert).max(4) } else { 4 };
        let mut b = Bufs {
            r: a(ch)?, rn: a(ch)?, down: a(c.hc_rank)?, up: a(ch)?, u: a(h)?, g: a(cs)?, y: a(h)?, e: a(h)?,
            qkv: a(2 * kd + vd)?, x: a(2 * kd + vd)?, z: a(vd)?, beta: a(c.ssm_v_heads)?, alpha: a(c.ssm_v_heads)?,
            o: a(vd)?,
            qg: a(2 * qd)?, q: a(qd)?, gate: a(qd)?, kk: a(kvd)?, iq: a(c.idx_heads * id)?,
            iqn: a(c.idx_heads * id)?, ao: a(qd)?, scores: a(cap / ratio + 1)?,
            toks: a(c.idx_budget + ratio)?, scratch: ctx.alloc(c.n_head * cap * 8).map_err(e)?,
            // the routed experts' intermediates, one row per routed expert (eg holds 2*max_ff when gate+up batch together)
            rl: a(c.n_expert)?, eg: a(2 * max_ff)?, eu: a(max_ff)?, act: a(max_ff)?,
            eo: a((k + 1).max(1) * h)?, sgl: a(1)?, wg: ctx.alloc(scratch_exp).map_err(e)?, wu: ctx.alloc(scratch_exp).map_err(e)?,
            wd: ctx.alloc(scratch_exp).map_err(e)?, tab: ctx.alloc((k + 2).max(1) * (3 * 8 + 4 * 4)).map_err(e)?,
            pe: a(h)?, key: a(ch)?, keyn: a(ch)?, val: a(h)?, qry: a(ch)?, gv: a(ch)?, gvn: a(ch)?,
            logits: a(c.vocab)?,
        };
        let fixed = FixedBufs {
            att: ctx.alloc(ATT_ROWS * c.n_head * cap * 8).map_err(e)?,
            sel_iqn: a(sel_rows() * c.idx_heads * id)?,
            sel_scores: a(sel_rows() * (cap / c.compress.iter().copied().filter(|&r| r > 0).min().unwrap_or(1) + 1))?,
            sel_toks: a(sel_rows() * (c.idx_budget + ratio) + sel_rows())?,
            logits: a(COLS_MAX * c.vocab)?,
        };
        ALIAS_ROWS.store(if tiers.pipeline { 0 } else { match std::env::var("SHADOW_ALIAS_ROWS").as_deref() { Ok("sets") => 1, Ok("up") => 2, Ok("0") | Ok("off") => 0, _ => 3 } }, std::sync::atomic::Ordering::Relaxed);
        let bb = make_bb(c, MAX_BATCH, &mut |n| a(n), fixed)?;
        let replay = Replay {
            qkv: a(c.n_layer * MAX_VERIFY * (2 * kd + vd))?, ba: a(c.n_layer * MAX_VERIFY * 2 * c.ssm_v_heads)?,
            ple: a(MAX_VERIFY * ch)?, pass: None, on: std::cell::Cell::new(false),
        };
        let dscr = a(BIG_BATCH.div_ceil(64) * c.ssm_v_heads * 8448)?;
        let (dp_tidx, dp_cnt, dp_gstart, dp_copied, dp_eaddr) = (a(BIG_BATCH_MAX * 16)?, a(1024)?, a(1024)?, a(256)?, a(6144)?);
        let kv_lend = !std::env::var("SHADOW_KV_LEND").is_ok_and(|v| v == "0");
        let mut kv_regions: Vec<crate::kvvmm::VmmRegion<'c>> = Vec::new();
        let kv_extra_reg: Option<crate::kvvmm::VmmRegion<'c>> = None; // made at the first lend, when the slot size is known
        let mut delta = Vec::new();
        let mut attn = Vec::new();
        for l in 0..c.n_layer {
            if c.is_attention(l) {
                delta.push(None);
                let blocks = if id > 0 { cap / c.compress[l].max(1) + 1 } else { 0 };
                let mut at = if kv_lend {
                    let mk = |per_pos: usize, regs: &mut Vec<crate::kvvmm::VmmRegion<'c>>| -> Result<DeviceBuffer<'c>, String> {
                        let r = crate::kvvmm::VmmRegion::new(ctx, per_pos, cap)?;
                        // SAFETY: a view of the region's address range; the session owns the region for as long as the view lives.
                        let v = unsafe { DeviceBuffer::view(ctx, r.base, cap * per_pos) };
                        regs.push(r);
                        Ok(v)
                    };
                    let (k, v, raw) = (mk(kvd * 4, &mut kv_regions)?, mk(kvd * 4, &mut kv_regions)?, mk(id * 4, &mut kv_regions)?);
                    Attn { k, v, raw, blocks: a(blocks * id)?, nblocks: 0 }
                } else {
                    let mut at = Attn { k: a(cap * kvd)?, v: a(cap * kvd)?, raw: a(cap * id)?, blocks: a(blocks * id)?, nblocks: 0 };
                    at.k.zero().map_err(e)?;
                    at.v.zero().map_err(e)?;
                    at
                };
                attn.push(Some(at));
            } else {
                let mut s = a(c.ssm_v_heads * c.ssm_head_dim * c.ssm_head_dim)?;
                let mut conv = a((c.conv_k.max(1) - 1) * (2 * kd + vd))?;
                s.zero().map_err(e)?;
                conv.zero().map_err(e)?;
                delta.push(Some((s, conv)));
                attn.push(None);
            }
        }
        let mut ple_state = a((c.ple_conv_k.max(1) - 1) * c.ple_ngram * ch)?;
        ple_state.zero().map_err(e)?;

        let mtp = match tiers.mtp_dir.as_ref() {
            Some(p) => match crate::mtp::MtpModel::load(ctx, p, cap, c.vocab) {
                Ok(m) => {
                    eprintln!("mtp: loaded tuned draft head from {} (~824 MB VRAM)", p.display());
                    Some(m)
                }
                Err(err) => {
                    eprintln!("mtp: warning: failed to load {}: {err}", p.display());
                    None
                }
            },
            None => None,
        };

        let dflash = match tiers.dflash_dir.as_ref() {
            Some(p) => match crate::dflash::DflashModel::load(ctx, p, cap, tiers.dflash_q8) {
                Ok(mut m) => {
                    m.min_p = tiers.dflash_min_p;
                    eprintln!("dflash: loaded the drafter from {} (~1.2 GB VRAM)", p.display());
                    Some(m)
                }
                Err(err) => {
                    eprintln!("dflash: warning: failed to load {}: {err}", p.display());
                    None
                }
            },
            None => None,
        };

        vm("buffers and the drafter, before the expert cache");
        // the expert cache: partitioned per layer at each layer's exact expert size (0 internal fragmentation) when
        // the budget covers at least `n_expert_used` slots per MoE layer, otherwise uniform `slot_bytes` slots.
        let per = |name: String| m.g.tensor(&name).map(|t| (t.nbytes() / t.dims[2]) as usize).unwrap_or(0);
        let l_bytes: Vec<usize> = (0..c.n_layer)
            .map(|l| (per(format!("blk.{l}.ffn_gate_exps.weight")) + per(format!("blk.{l}.ffn_up_exps.weight"))
                + per(format!("blk.{l}.ffn_down_exps.weight"))).next_multiple_of(256))
            .collect();
        let slot_bytes = l_bytes.iter().copied().max().unwrap_or(0);
        let min_per_layer: usize = (0..c.n_layer).filter(|&l| c.is_moe(l)).map(|l| c.n_expert_used * l_bytes[l]).sum();
        let batch_scratch = match (c.arch.as_str(), cache_bytes, slot_bytes) {
            ("flint", _, _) | (_, Some(0), _) | (_, _, 0) => None,
            _ => {
                let n = std::env::var("SHADOW_BATCH_SCRATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(BATCH_SCRATCH_SLOTS);
                (n > 0).then(|| ctx.alloc(n * slot_bytes).map(|b| (b, n))).transpose().map_err(e)?
            }
        };
        let mut budget = match cache_bytes {
            Some(b) => b,
            None => ctx.mem_info().map_err(e)?.0.saturating_sub(vram_reserve()),
        };
        let order = tiers.order.clone().unwrap_or_else(|| uniform_order(c.n_layer, c.n_expert));
        let mut arena = None;
        let mut slot_offset = Vec::new();
        let mut layer_range = vec![0..0; c.n_layer];
        let mut arena_bytes = 0usize;
        let mut slots = 0usize;
        while budget >= slot_bytes.max(1) * c.n_expert_used.max(1) {
            slot_offset.clear();
            layer_range = vec![0..0; c.n_layer];
            if min_per_layer > 0 && budget >= min_per_layer {
                let mut l_slots = vec![0usize; c.n_layer];
                let mut used = 0usize;
                for l in 0..c.n_layer {
                    if c.is_moe(l) {
                        l_slots[l] = c.n_expert_used;
                        used += c.n_expert_used * l_bytes[l];
                    }
                }
                for &(l, _) in &order {
                    if c.is_moe(l) && l_slots[l] < c.n_expert && used + l_bytes[l] <= budget {
                        l_slots[l] += 1;
                        used += l_bytes[l];
                    }
                }
                let mut cur_s = 0usize;
                let mut cur_off = 0u64;
                for l in 0..c.n_layer {
                    let n_s = l_slots[l];
                    layer_range[l] = cur_s..cur_s + n_s;
                    for _ in 0..n_s {
                        slot_offset.push(cur_off);
                        cur_off += l_bytes[l] as u64;
                    }
                    cur_s += n_s;
                }
                arena_bytes = cur_off as usize;
                slots = cur_s;
            } else {
                slots = budget / slot_bytes;
                arena_bytes = slots * slot_bytes;
                for s in 0..slots {
                    slot_offset.push((s * slot_bytes) as u64);
                }
                layer_range = vec![0..slots; c.n_layer];
            }
            match ctx.alloc(arena_bytes) {
                Ok(buf) => {
                    arena = Some(buf);
                    break;
                }
                Err(_) => budget = budget * 7 / 8,
            }
        }
        vm("expert cache arena");
        if arena.is_none() {
            slot_offset.clear();
            layer_range = vec![0..0; c.n_layer];
            arena_bytes = 0;
            slots = 0;
            let need = (k * max_expert).max(4);
            if b.wg.bytes < need {
                b.wg = ctx.alloc(need).map_err(e)?;
                b.wu = ctx.alloc(need).map_err(e)?;
                b.wd = ctx.alloc(need).map_err(e)?;
            }
        }
        let ram_bytes = if tiers.ram_auto {
            let b = crate::pinned::auto_tier_bytes() as usize;
            eprintln!("RAM tier: auto = {:.2} GiB ({:.2} GiB available less a {:.0} GiB margin)", b as f64 / (1u64 << 30) as f64,
                      crate::pinned::available_ram() as f64 / (1u64 << 30) as f64,
                      crate::pinned::auto_margin() as f64 / (1u64 << 30) as f64);
            b
        } else {
            tiers.ram_bytes
        };
        let per_layer_mode = layer_range.iter().filter(|r| !r.is_empty()).count() > 1
            && layer_range.windows(2).any(|w| w[0] != w[1]);
        let mut evict_range = layer_range.clone();
        let mut vram_pinned: Vec<(usize, usize, u64)> = Vec::new();
        let mut vram_pinned_slots: Vec<(usize, usize, usize)> = Vec::new();
        let mut next_in_layer: Vec<usize> = layer_range.iter().map(|r| r.start).collect();
        // When a sizable RAM tier is present and per-layer VRAM slots are active, pin the hottest slots permanently in
        // VRAM (keeping `margin` LRU slots per layer for transient misses) and exclude them from `RamTier` so `RamTier`
        // covers ~1,800+ additional experts without duplicating VRAM-pinned ones.
        if slots > 0 && per_layer_mode && ram_bytes >= (4usize << 30) {
            let margin = tiers.pin_margin.unwrap_or(usize::MAX).max(c.n_expert_used + 2);
            let mut pin_end = vec![0usize; c.n_layer];
            for l in 0..c.n_layer {
                let len = layer_range[l].len();
                let n_pin = len.saturating_sub(margin);
                pin_end[l] = layer_range[l].start + n_pin;
                evict_range[l] = pin_end[l]..layer_range[l].end;
            }
            for &(l, ep) in &order {
                if !c.is_moe(l) || next_in_layer[l] >= pin_end[l] {
                    continue;
                }
                let s = next_in_layer[l];
                next_in_layer[l] += 1;
                vram_pinned.push((l, ep, slot_offset[s]));
                vram_pinned_slots.push((l, ep, s));
            }
        }
        let arena_ptr = arena.as_ref().map_or(0, |a| a.ptr);
        let ram = RamTier::fill(ctx, m, ram_bytes, tiers.order.clone(), &vram_pinned, arena_ptr)?;
        let mut cache = ExpertCache {
            arena, slot_offset, layer_range, evict_range, slot_bytes, arena_bytes, slots, map: HashMap::new(),
            owner: vec![None; slots], stamp: vec![0; slots], clock: 0, hits: 0, misses: 0,
            admit_after: tiers.admit_after, seen: HashMap::new(), scratch_reads: 0,
        };
        for (l, ep, s) in vram_pinned_slots {
            cache.owner[s] = Some((l, ep));
            cache.map.insert((l, ep), s);
        }
        if let Ok(path) = std::env::var("SHADOW_CACHE_DUMP") {
            // per layer: slots, evictable slots, then the pinned experts (tools/cache_sim.py reads it)
            let lines: Vec<String> = (0..c.n_layer).map(|l| {
                let pinned: Vec<String> = cache.layer_range[l].clone().filter(|s| !cache.evict_range[l].contains(s))
                    .filter_map(|s| cache.owner[s].map(|(_, e)| e.to_string())).collect();
                format!("{} {} {} {}", l, cache.layer_range[l].len(), cache.evict_range[l].len(), pinned.join(","))
            }).collect();
            if let Err(err) = std::fs::write(&path, lines.join("
")) {
                eprintln!("SHADOW_CACHE_DUMP {path}: {err}");
            }
        }
        // Pre-warm the remaining (evictable) VRAM expert cache slots from the RAM tier in placement order.
        if let Some(region) = ram.region.as_ref().filter(|_| cache.slots > 0) {
            let names = |l: usize| ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(|r| format!("blk.{l}.{r}.weight"));
            let src = region.as_slice();
            let mut global_s = 0usize;
            for (l, ep) in order {
                if cache.map.contains_key(&(l, ep)) {
                    continue;
                }
                let s = if per_layer_mode {
                    if next_in_layer[l] >= cache.layer_range[l].end {
                        continue;
                    }
                    let s = next_in_layer[l];
                    next_in_layer[l] += 1;
                    s
                } else {
                    if global_s >= cache.slots {
                        break;
                    }
                    let s = global_s;
                    global_s += 1;
                    s
                };
                let Some(&off) = ram.at.get(&(l, ep)) else { continue };
                let total: usize = names(l).iter().map(|n| m.expert_bytes(n)).sum();
                let dst = arena_ptr + cache.slot_offset[s];
                ctx.bind().map_err(e)?;
                let r = unsafe {
                    (ctx.api.cuMemcpyHtoDAsync)(dst, src[off..off + total].as_ptr() as *const std::ffi::c_void, total,
                                                std::ptr::null_mut())
                };
                assert_eq!(r, 0, "cuMemcpyHtoDAsync pre-warm failed ({r})");
                cache.owner[s] = Some((l, ep));
                cache.map.insert((l, ep), s);
            }
            ctx.synchronize().map_err(e)?;
        }
        let copy_stream = ctx.stream().map_err(e)?;
        let copy_stream2 = ctx.stream().map_err(e)?;
        let copy_ev2 = ctx.event().map_err(e)?;
        let copy_event = ctx.event().map_err(e)?;
        let staging = if slot_bytes > 0 && c.n_expert_used > 0 {
            // four tokens' worth: a batched step may read up to 4 * n_expert_used of a layer's misses together
            Some(PinnedRegion::new(ctx, 4 * c.n_expert_used * slot_bytes)?)
        } else {
            None
        };
        let mut s = GpuSession {
            ctx, m, mvp, fw, w, v, b, bb, replay, dscr,
            rparts: Vec::new(), logits_last: false, sub_last: true, sub_first: true,
            devplan: tiers.tensor_prompt && !std::env::var("SHADOW_PREFILL_DEVPLAN").is_ok_and(|v| v == "0"), dp_tidx, dp_cnt, dp_gstart, dp_copied, dp_eaddr, kv_regions, kv_extra: None, kv_extra_reg, prompt_hint: 0, kv_rows: 0, kv_halves: false, router_bias: std::env::var("SHADOW_ROUTER_BIAS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0), batch_scratch, delta, attn, ple_state, tokens: vec![], pictures: vec![], rope_tab: None, rope_on: false, cap, trace: None, stop_after: None,
            resident_bytes, expert_uploads: 0, cache, ram, copy_stream, copy_stream2, copy_ev2, copy_event, staging,
            pending: vec![], routing: RoutingCounts::new(c.n_layer, c.n_expert),
            prof: std::env::var("SHADOW_PROFILE").is_ok_and(|v| v == "1")
                .then(|| Prof { last: std::time::Instant::now(), phases: vec![] }),
            route_log: std::env::var("SHADOW_ROUTE_LOG").ok().map(|p| (p, vec![])),
            cpu: (tiers.cpu_experts > 0 && tiers.cpu_threads > 0)
                .then(|| (shadow_model::cpu_expert::ExpertPool::new(tiers.cpu_threads), tiers.cpu_experts)),
            cpu_served: 0,
            cpu_join: (0, 0.0, 0.0),
            cpu_host: (0.0, 0.0),
            tensor_prompt: tiers.tensor_prompt,
            lent: None,
            sp: None,
            rows: MAX_BATCH,
            small_bb: None,
            pipeline: tiers.pipeline,
            pipe_active: false,
            pipelined_passes: 0,
            plan_lat: (0, 0.0),
            row_shift: 0,
            mapped_event2: ctx.event().map_err(e)?,
            group_event: ctx.event().map_err(e)?,
            ticket_stage: [usize::MAX; 2],
            mapped: PinnedRegion::new(ctx, BIG_BATCH_MAX.max(2) * c.n_expert * 4)?,
            mapped_event: ctx.event().map_err(e)?,
            waits: [0.0; 3],
            stage_ptr: std::ptr::null_mut(),
            stage: PinnedRegion::new(ctx, STAGE_BYTES)?,
            stage_at: std::cell::Cell::new(0),
            prefetch: match tiers.pregate {
                0 => None,
                k => Some(Prefetch { k, buf: ctx.alloc(c.n_expert * 4).map_err(e)?, ahead: Default::default(), wait: false,
                                     issued: 0, used: 0 }),
            },
            pregate: match std::env::var("SHADOW_PREGATE_PROBE").is_ok_and(|v| v == "1") {
                true => Some(Pregate { buf: ctx.alloc(c.n_expert * 4).map_err(e)?, pred: vec![None; c.n_layer],
                                       counts: vec![[0; 5]; c.n_layer] }),
                false => None,
            },
            mtp_probe: std::env::var("SHADOW_MTP_PROBE").is_ok_and(|v| v == "1").then_some((0, 0)),
            mtp,
            dflash,
            graphs: HashMap::new(),
            multi: HashMap::new(),
            fuse: tiers.fuse.or_else(|| std::env::var("SHADOW_FUSE").ok().and_then(|v| v.parse().ok())).unwrap_or(0),
            ltime: std::env::var("SHADOW_LTIME").is_ok_and(|v| v == "1").then(Default::default),
            ple_host: std::cell::Cell::new(0.0),
            ple_calls: std::cell::Cell::new(0),
            timing_no_sync: std::env::var("SHADOW_TIMING_NO_SYNC").is_ok_and(|v| v == "1"),
            stale_logits: HashMap::new(),
            zero_copy: std::env::var("SHADOW_ZERO_COPY").is_ok_and(|v| v == "1"),
            batch_graphs: HashMap::new(),
            fv_on: std::env::var("SHADOW_FILE_VIEWS").is_ok_and(|v| v == "1"),
            fv: None,
            fv_regs: Default::default(),
            fast_order: !std::env::var("SHADOW_REF_ORDER").is_ok_and(|v| v == "1"),
            verify_stats: [0.0; 7],
            tail_skip: tiers.tail_skip,
            verify_resident: tiers.verify_resident,
            verify_rank: tiers.verify_rank,
            tail_skipped: 0,
            accept_moe: tiers.accept_moe,
            accept_beta: tiers.accept_beta,
            accept_scale: tiers.accept_scale,
            accept_budget: tiers.accept_budget,
            commit: vec![],
            graphs_on: std::env::var("SHADOW_GRAPHS").map_or(true, |v| v != "0")
                && !std::env::var("SHADOW_KTIME").is_ok_and(|v| v == "1"),
            overlap: std::env::var("SHADOW_OVERLAP").map_or(true, |v| v != "0"),
            gstream: ctx.blocking_stream().map_err(e)?,
            capturing: std::cell::Cell::new(false),
            batch_max: std::env::var("SHADOW_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_BATCH).clamp(1, MAX_BATCH),
        };
        s.stage_ptr = s.stage.as_mut_slice().as_mut_ptr();
        Ok(s)
    }

    /// Charges the time since the last mark to `phase` (profiling only; synchronizes the device first).
    fn mark(&mut self, phase: &'static str) {
        let Some(p) = self.prof.as_mut() else { return };
        self.ctx.synchronize().unwrap();
        let now = std::time::Instant::now();
        let dt = (now - p.last).as_secs_f64();
        p.last = now;
        match p.phases.iter_mut().find(|(n, _)| *n == phase) {
            Some(x) => x.1 += dt,
            None => p.phases.push((phase, dt)),
        }
    }

    /// (slots, bytes per slot, hits, misses) of the expert cache.
    pub fn cache_stats(&self) -> (usize, usize, u64, u64) {
        (self.cache.slots, self.cache.slot_bytes, self.cache.hits, self.cache.misses)
    }

    pub fn gpu_name(&self) -> &str {
        &self.ctx.name
    }

    pub fn mem_info(&self) -> (usize, usize) {
        self.ctx.mem_info().unwrap_or((0, 0))
    }

    pub fn ram_tier_stats(&self) -> (usize, usize, u64, u64) {
        (self.ram.at.len(), self.ram.region.as_ref().map_or(0, |r| r.len()), self.ram.served, self.ram.drive)
    }

    /// The tokens this session has read, in order: its state is exactly the state after them.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The positions this session was made for.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Back to the state before any token: every DeltaNet state and conv window, the attention caches (as `new` leaves
    /// them: zeroed, no complete blocks) and the n-gram conv state zeroed, the token list emptied.  The weights, the
    /// expert cache and the RAM tier stay.
    /// Pictures for the prompt about to be read: (first token index, rows, the rows as rows x hidden floats).  The tokens there
    /// (the `<|image_pad|>` id, which also gives their per-layer inputs) keep their ids; only their embeddings are the rows.
    pub fn set_pictures(&mut self, pics: Vec<(usize, usize, usize, std::sync::Arc<Vec<f32>>)>) {
        self.pictures = pics;
        self.sync_rope_tab();
    }

    fn picture_row(&self, idx: usize) -> Option<&[f32]> {
        let h = self.m.cfg.hidden;
        self.pictures.iter().find(|(s, n, _, _)| idx >= *s && idx < s + n).map(|(s, _, _, rows)| &rows[(idx - s) * h..(idx - s + 1) * h])
    }

    /// The rotary positions when pictures are in the conversation, as a table over every token index (as a Qwen-VL runtime counts them): a text
    /// token is the next position; a picture row at grid (x, y) is (start, start + y, start + x) for the position `start` the picture begins at;
    /// the token after the picture is `start + max(width, height)`.  No pictures: the table pointer is cleared (the kernels use plain positions).
    fn sync_rope_tab(&mut self) {
        if self.pictures.is_empty() {
            if self.rope_on {
                self.fw.module.set_global_ptr("g_rope_tab", 0).unwrap();
                self.rope_on = false;
            }
            return;
        }
        let cap = self.cap;
        let mut tab = vec![0i32; cap * 3];
        let (mut idx, mut next) = (0usize, 0i32);
        while idx < cap {
            if let Some(&(_, n, nx, _)) = self.pictures.iter().find(|p| p.0 == idx) {
                let (n, nx) = (n.min(cap - idx), nx.max(1));
                for k in 0..n {
                    let (y, x) = ((k / nx) as i32, (k % nx) as i32);
                    tab[(idx + k) * 3..(idx + k) * 3 + 3].copy_from_slice(&[next, next + y, next + x]);
                }
                next += nx.max(n.div_ceil(nx)) as i32;
                idx += n;
            } else {
                tab[idx * 3..idx * 3 + 3].copy_from_slice(&[next; 3]);
                next += 1;
                idx += 1;
            }
        }
        let bytes: Vec<u8> = tab.iter().flat_map(|v| v.to_le_bytes()).collect();
        if self.rope_tab.is_none() {
            self.rope_tab = Some(self.ctx.alloc(cap * 12).unwrap());
        }
        let buf = self.rope_tab.as_ref().unwrap();
        buf.upload(&bytes).unwrap();
        self.fw.module.set_global_ptr("g_rope_tab", buf.ptr).unwrap();
        self.rope_on = true;
    }

    pub fn reset(&mut self) {
        for (s, conv) in self.delta.iter_mut().flatten() {
            s.zero().unwrap();
            conv.zero().unwrap();
        }
        for a in self.attn.iter_mut().flatten() {
            a.k.zero().unwrap();
            a.v.zero().unwrap();
            a.nblocks = 0;
        }
        self.ple_state.zero().unwrap();
        self.tokens.clear();
        self.pictures.clear();
        if let Some(mt) = self.mtp.as_mut() {
            mt.pending = false;
        }
    }

    /// The session's state after its first `tokens().len()` tokens, on the device: every DeltaNet state and conv
    /// window, the n-gram conv state, and - with `with_kv` - each attention layer's keys, values and indexer keys up to
    /// that position.  Without the KV it can only be restored onto a session whose first positions are still the same
    /// tokens (a speculative pass rolled back); with it, onto any (a shared prompt prefix, reused by a new chat).
    pub fn snapshot(&self, with_kv: bool) -> Result<Snapshot<'c>, String> {
        self.snapshot_to(with_kv, false)
    }

    /// `snapshot`, kept in host memory (`host`) or on the device.
    pub fn snapshot_to(&self, with_kv: bool, host: bool) -> Result<Snapshot<'c>, String> {
        let copy = |b: &DeviceBuffer<'c>, bytes: usize| Saved::take(self.ctx, b, bytes, host);
        let n = self.tokens.len();
        let c = &self.m.cfg;
        let kvd = c.n_kv_head * c.head_dim;
        let mut delta = Vec::with_capacity(self.delta.len());
        for d in &self.delta {
            delta.push(match d {
                Some((s, conv)) => Some((copy(s, s.bytes)?, copy(conv, conv.bytes)?)),
                None => None,
            });
        }
        let mut kv = Vec::with_capacity(self.attn.len());
        for a in &self.attn {
            kv.push(match a.as_ref().filter(|_| with_kv) {
                Some(a) => {
                    let id = if a.raw.bytes > 0 { a.raw.bytes / self.cap } else { 0 };
                    Some(SnapKv {
                        k: copy(&a.k, n * kvd * 4)?, v: copy(&a.v, n * kvd * 4)?, raw: copy(&a.raw, n * id)?,
                        blocks: copy(&a.blocks, (a.nblocks * id).min(a.blocks.bytes))?, nblocks: a.nblocks,
                    })
                }
                None => None,
            });
        }
        let mtp = match self.mtp.as_ref() {
            Some(mt) => {
                let kvd = crate::mtp::N_HEAD_KV * crate::mtp::HEAD_DIM * 4;
                let kv = if with_kv { Some((copy(&mt.bufs.kv_k, n * kvd)?, copy(&mt.bufs.kv_v, n * kvd)?)) } else { None };
                Some((copy(&mt.bufs.last_r, mt.bufs.last_r.bytes)?, mt.pending, kv))
            }
            None => None,
        };
        Ok(Snapshot { tokens: self.tokens.clone(), delta, ple: copy(&self.ple_state, self.ple_state.bytes)?, kv,
                      nblocks: self.attn.iter().map(|a| a.as_ref().map_or(0, |a| a.nblocks)).collect(), mtp })
    }

    /// Takes `snap` again at the current position, into its own buffers: a speculative pass's rollback point, once per
    /// pass without allocating (a device snapshot without the KV, from this session; otherwise a new one).
    pub fn snapshot_again(&self, snap: &mut Snapshot<'c>) -> Result<(), String> {
        let reusable = snap.kv.iter().all(|k| k.is_none()) && snap.delta.len() == self.delta.len()
            && matches!(snap.ple, Saved::Dev(..))
            && match (&snap.mtp, &self.mtp) {
                (Some((Saved::Dev(..), _, None)), Some(_)) | (None, None) => true,
                _ => false,
            }
            && snap.delta.iter().zip(&self.delta).all(|(s, d)| match (s, d) {
                (Some((Saved::Dev(a, an), Saved::Dev(b, bn))), Some((st, conv))) => {
                    *an == st.bytes && *bn == conv.bytes && a.bytes >= st.bytes && b.bytes >= conv.bytes
                }
                (None, None) => true,
                _ => false,
            });
        if !reusable {
            *snap = self.snapshot(false)?;
            return Ok(());
        }
        let e = |x: crate::driver::CudaError| x.to_string();
        for (s, d) in snap.delta.iter().zip(&self.delta) {
            if let (Some((Saved::Dev(a, an), Saved::Dev(b, bn))), Some((st, conv))) = (s, d) {
                a.copy_prefix_from(st, *an).map_err(e)?;
                b.copy_prefix_from(conv, *bn).map_err(e)?;
            }
        }
        if let Saved::Dev(p, n) = &snap.ple {
            p.copy_prefix_from(&self.ple_state, *n).map_err(e)?;
        }
        snap.tokens.clone_from(&self.tokens);
        snap.nblocks = self.attn.iter().map(|a| a.as_ref().map_or(0, |a| a.nblocks)).collect();
        if let (Some((Saved::Dev(d, n), pending, _)), Some(mt)) = (snap.mtp.as_mut(), self.mtp.as_ref()) {
            d.copy_prefix_from(&mt.bufs.last_r, *n).map_err(e)?;
            *pending = mt.pending;
        }
        Ok(())
    }

    /// Puts the session back to `snap`: the token list, every recurrent state, and the KV prefix if the snapshot has
    /// it.  Without the KV, the session must still hold the snapshot's tokens as its first positions.
    pub fn restore(&mut self, snap: &Snapshot<'c>) -> Result<(), String> {
        let has_kv = snap.kv.iter().any(|k| k.is_some());
        if !has_kv {
            let n = snap.tokens.len();
            if self.tokens.len() < n || self.tokens[..n] != snap.tokens[..] {
                return Err("restore: this snapshot has no KV and the session no longer starts with its tokens".into());
            }
        }
        for (d, s) in self.delta.iter().zip(&snap.delta) {
            if let (Some((st, conv)), Some((ss, sc))) = (d, s) {
                ss.put(st)?;
                sc.put(conv)?;
            }
        }
        snap.ple.put(&self.ple_state)?;
        for (l, a) in self.attn.iter_mut().enumerate() {
            let Some(a) = a.as_mut() else { continue };
            match &snap.kv[l] {
                Some(k) => {
                    for (saved, buf) in [(&k.k, &a.k), (&k.v, &a.v), (&k.raw, &a.raw), (&k.blocks, &a.blocks)] {
                        if saved.len() > buf.bytes {
                            return Err("restore: the snapshot is from a session with a larger context".into());
                        }
                        saved.put(buf)?;
                    }
                    a.nblocks = k.nblocks;
                }
                None => a.nblocks = a.nblocks.min(snap.nblocks[l]),
            }
        }
        if let (Some(mt), Some((last, pending, kv))) = (self.mtp.as_mut(), snap.mtp.as_ref()) {
            last.put(&mt.bufs.last_r)?;
            mt.pending = *pending;
            if let Some((k, v)) = kv {
                k.put(&mt.bufs.kv_k)?;
                v.put(&mt.bufs.kv_v)?;
            }
        }
        self.tokens = snap.tokens.clone();
        Ok(())
    }

    /// One expert's slice of a fused `[in, out, n_expert]` tensor: its info as a 2-D matrix, without reading it.
    fn expert_info(&self, name: &str) -> TensorInfo {
        let mut t = self.m.g.tensor(name).unwrap_or_else(|| panic!("no tensor {name}")).clone();
        t.dims = vec![t.dims[0], t.dims[1]];
        t
    }

    // -------------------------------------------------------------------------------------------- launch helpers
    /// The stream launches go to: the graph stream while a graph is being captured, else the default stream.
    fn ls(&self) -> Option<&Stream<'c>> {
        self.capturing.get().then_some(&self.gstream)
    }

    /// Runs `f`'s launches as the graph `key`: captured from `f` the first time, replayed after.  `f` must launch the
    /// same kernels with the same arguments every time (no host decisions, uploads or reads inside).
    /// `graph_segment` for the batched step's segments.
    fn batch_graph_segment(&mut self, key: (usize, bool, usize, bool, usize, u64), f: impl FnOnce(&Self)) {
        if !self.batch_graphs.contains_key(&key) {
            self.gstream.begin_capture().unwrap();
            self.capturing.set(true);
            f(self);
            self.capturing.set(false);
            let g = self.gstream.end_capture().unwrap_or_else(|e| panic!("graph capture of layer {} x {}: {e}", key.0, key.2));
            self.batch_graphs.insert(key, g);
        }
        if let Err(e) = self.batch_graphs[&key].launch(&self.gstream) {
            panic!("batch graph (layer {}, rn_ready {}, m {}, on {}, rows {}): {e}", key.0, key.1, key.2, key.3, key.4);
        }
    }

    fn graph_segment(&mut self, key: (usize, bool), f: impl FnOnce(&Self)) {
        if !self.graphs.contains_key(&key) {
            self.gstream.begin_capture().unwrap();
            self.capturing.set(true);
            f(self);
            self.capturing.set(false);
            let g = self.gstream.end_capture().unwrap_or_else(|e| panic!("graph capture of layer {}: {e}", key.0));
            self.graphs.insert(key, g);
        }
        self.graphs[&key].launch(&self.gstream).unwrap();
    }

    /// A point on the default stream for `SHADOW_LTIME`: the time since the previous point is charged to `name`.
    fn lt_point(&self, name: &'static str) {
        if let Some(l) = self.ltime.as_ref() {
            let e = self.ctx.event().unwrap();
            e.record(None).unwrap();
            l.borrow_mut().push((name, e));
        }
    }

    fn k(&self, name: &str, grid: u32, block: u32, args: &[Arg]) {
        // SAFETY: every pointer passed is a buffer of this session sized for the dimensions passed beside it (the
        // registry checks the argument types and count).
        unsafe { self.fw.launch(name, (grid, 1, 1), (block, 1, 1), 0, self.ls(), args) }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
    }

    /// A launch over (gx, gy) blocks: gy = the batch's rows for the `_rows` kernels.
    fn k2(&self, name: &str, gx: u32, gy: u32, block: u32, args: &[Arg]) {
        // SAFETY: as `k`; the `_rows` kernels' row strides are the batch buffers' (BatchBufs).
        unsafe { self.fw.launch(name, (gx, gy, 1), (block, 1, 1), 0, self.ls(), args) }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
    }

    fn k1(&self, name: &str, n: usize, args: &[Arg]) {
        self.k(name, (n as u32).div_ceil(256), 256, args);
    }

    fn matvec(&self, t: &TensorInfo, wptr: u64, rows: usize, x: u64, y: u64) {
        let ty = t.ty.name();
        // the 8-at-a-time kernel where tests/matvec.rs measured it faster than the reference one (every block-quantized
        // type), and for BF16 / F32 too unless the reference sum order is asked for (`fast_order`)
        let fast = (self.fast_order || !matches!(ty.as_str(), "BF16" | "F32")) && t.dims[0].is_multiple_of(8);
        let entry = if fast {
            FAST_KERNELS.iter().find(|q| q.0 == ty).unwrap().1
        } else {
            QUANT_KERNELS.iter().find(|q| q.0 == ty).unwrap_or_else(|| panic!("{}: no matvec kernel for {ty}", t.name)).4
        };
        // SAFETY: `rows` whole rows of t.dims[0] weights at wptr; x holds t.dims[0] floats, y `rows`.
        unsafe {
            self.mvp.launch(entry, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(),
                            &[Arg::Ptr(wptr), Arg::Ptr(x), Arg::Ptr(y), Arg::U32(rows as u32), Arg::U32(t.dims[0] as u32)])
        }
        .unwrap_or_else(|err| panic!("{}: {err}", t.name));
    }

    /// y = W x for a resident dense tensor (applying the companion `.scale` scalar when present).
    fn mv(&self, name: &str, x: u64, y: u64) {
        let (t, bytes) = self.m.dense(name);
        let (qk, qb) = t.ty.block().unwrap();
        let rows = bytes.len() / (t.dims[0] / qk * qb) as usize;
        let wptr = self.w.get(name).unwrap_or_else(|| panic!("{name} is not resident")).ptr;
        let s = self.m.scale(name);
        if s != 1.0 && t.ty.name() == "NVFP4" && t.dims[0].is_multiple_of(8) {
            unsafe {
                self.mvp.launch("matvec_fast_nvfp4_s", ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(),
                                &[Arg::Ptr(wptr), Arg::Ptr(x), Arg::Ptr(y), Arg::U32(rows as u32),
                                  Arg::U32(t.dims[0] as u32), Arg::F32(s)])
            }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
            return;
        }
        self.matvec(t, wptr, rows, x, y);
        if s != 1.0 {
            self.k1("scale_vec", rows, &[Arg::Ptr(y), Arg::U32(rows as u32), Arg::F32(s)]);
        }
    }

    /// Fuses y0 = W0 x and y1 = W1 x into a single launch when W0 and W1 share the column count, unit scale and 8-row
    /// alignment - same quant type (`PAIR_KERNELS`) or a listed pair of types (`PAIR2_KERNELS`); two `mv` otherwise.
    fn mv_pair(&self, name0: &str, y0: u64, name1: &str, y1: u64, x: u64) {
        let (t0, b0) = self.m.dense(name0);
        let (t1, b1) = self.m.dense(name1);
        let ty = t0.ty.name();
        if self.fast_order && matches!(ty.as_str(), "BF16" | "F32") {
            // the pair kernels for these types sum in the reference order
            self.mv(name0, x, y0);
            self.mv(name1, x, y1);
            return;
        }
        let (qk, qb) = t0.ty.block().unwrap();
        let (qk1, qb1) = t1.ty.block().unwrap();
        let r0 = b0.len() / (t0.dims[0] / qk * qb) as usize;
        let r1 = b1.len() / (t1.dims[0] / qk1 * qb1) as usize;
        let ty1 = t1.ty.name();
        let fast_ok = |t: &TensorInfo| matches!(t.ty.name().as_str(), "BF16" | "F32") || t.dims[0].is_multiple_of(8);
        if t0.ty != t1.ty && t0.dims[0] == t1.dims[0] && r0.is_multiple_of(8) && t0.dims[0].is_multiple_of(8)
            && self.m.scale(name0) == 1.0 && self.m.scale(name1) == 1.0
        {
            if let Some(&(_, _, entry)) = PAIR2_KERNELS.iter().find(|q| q.0 == ty && q.1 == ty1) {
                let w0 = self.w.get(name0).unwrap_or_else(|| panic!("{name0} is not resident")).ptr;
                let w1 = self.w.get(name1).unwrap_or_else(|| panic!("{name1} is not resident")).ptr;
                unsafe {
                    self.mvp.launch(entry, (((r0 + r1) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(),
                                    &[Arg::Ptr(w0), Arg::Ptr(y0), Arg::U32(r0 as u32),
                                      Arg::Ptr(w1), Arg::Ptr(y1), Arg::U32(r1 as u32),
                                      Arg::Ptr(x), Arg::U32(t0.dims[0] as u32)])
                }
                .unwrap_or_else(|err| panic!("{name0}+{name1}: {err}"));
                return;
            }
        }
        if t0.ty == t1.ty
            && t0.dims[0] == t1.dims[0]
            && r0.is_multiple_of(8)
            && self.m.scale(name0) == 1.0
            && self.m.scale(name1) == 1.0
            && fast_ok(t0)
        {
            if let Some(&(_, entry)) = PAIR_KERNELS.iter().find(|q| q.0 == ty) {
                let w0 = self.w.get(name0).unwrap_or_else(|| panic!("{name0} is not resident")).ptr;
                let w1 = self.w.get(name1).unwrap_or_else(|| panic!("{name1} is not resident")).ptr;
                unsafe {
                    self.mvp.launch(entry, (((r0 + r1) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(),
                                    &[Arg::Ptr(w0), Arg::Ptr(y0), Arg::U32(r0 as u32),
                                      Arg::Ptr(w1), Arg::Ptr(y1), Arg::U32(r1 as u32),
                                      Arg::Ptr(x), Arg::U32(t0.dims[0] as u32)])
                }
                .unwrap_or_else(|err| panic!("{name0}+{name1}: {err}"));
                return;
            }
        }
        self.mv(name0, x, y0);
        self.mv(name1, x, y1);
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
    /// Device floats to the host without the copy engine: a copy kernel writes each `(ptr, n)` into the mapped region
    /// (one after another), the host waits for the default stream's event and reads them back.
    fn download_mapped(&self, srcs: &[(u64, usize)]) -> Vec<Vec<f32>> {
        let mut at = 0usize;
        for &(ptr, n) in srcs {
            assert!((at + n) * 4 <= self.mapped.len(), "mapped download too big");
            self.k1("copy", n, &[Arg::Ptr(self.mapped.dev_ptr + (at * 4) as u64), Arg::Ptr(ptr), Arg::U32(n as u32)]);
            at += n;
        }
        self.mapped_event.record(None).unwrap();
        self.mapped_event.synchronize().unwrap();
        self.stage_at.set(0); // the default stream has drained: every staged upload has been copied
        let base = self.mapped.as_slice().as_ptr() as *const f32;
        let mut at = 0usize;
        srcs.iter().map(|&(_, n)| {
            // SAFETY: the kernels above wrote n floats at base + at, and finished before the event completed; the
            // device writes nothing there now, and volatile reads keep the compiler from assuming the bytes unchanged.
            let v = (0..n).map(|i| unsafe { std::ptr::read_volatile(base.add(at + i)) }).collect();
            at += n;
            v
        }).collect()
    }

    fn download(&self, ptr: u64, out: &mut [u8]) {
        let api = self.ctx.api;
        self.ctx.bind().unwrap();
        // SAFETY: the address range lies inside a buffer of this session (the callers pass its sizes).
        let r = unsafe { (api.cuMemcpyDtoH)(out.as_mut_ptr() as *mut std::ffi::c_void, ptr, out.len()) };
        assert_eq!(r, 0, "cuMemcpyDtoH failed ({r})");
        self.stage_at.set(0); // synchronous on the default stream: everything staged before has been copied
    }

    fn upload_to(&self, ptr: u64, data: &[u8]) {
        let at = self.stage_at.get();
        if data.len().is_multiple_of(4) && !data.is_empty() && at + data.len() <= STAGE_BYTES && !self.stage_ptr.is_null() {
            // SAFETY: [at, at + len) is inside the staging region and not yet handed to a kernel since the last drain
            // (stage_at only grows until then), so nothing reads it while the host writes.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.stage_ptr.add(at), data.len()) };
            self.stage_at.set((at + data.len()).next_multiple_of(256));
            let n = data.len() / 4;
            self.k1("copy_words", n, &[Arg::Ptr(ptr), Arg::Ptr(self.stage.dev_ptr + at as u64), Arg::U32(n as u32)]);
            return;
        }
        let api = self.ctx.api;
        self.ctx.bind().unwrap();
        // SAFETY: as download.  For small buffers (<= 64 KiB: embedding row, tab, toks) cuMemcpyHtoDAsync stages into
        // the driver's command buffer and returns immediately without stalling the host on prior stream work.
        let r = unsafe { (api.cuMemcpyHtoDAsync)(ptr, data.as_ptr() as *const std::ffi::c_void, data.len(), std::ptr::null_mut()) };
        assert_eq!(r, 0, "cuMemcpyHtoDAsync failed ({r})");
    }

    /// Asynchronous host-to-device DMA from pinned memory (`PinnedRegion`) on `stream` (null = default stream).
    /// Queues `n` copies (device `dsts`, pinned host `srcs`, `lens` bytes) on `stream`: one `cuMemcpyBatchAsync` call when the driver has it (the host
    /// time of 450 single calls was the longest step of a long prompt's layer), else one by one.
    /// `srcs` are the sources' device-mapped addresses (`PinnedRegion::dev_ptr` + offset: the batch call wants those), `host_srcs` the same bytes' host
    /// addresses (for the one-by-one fallback).
    fn upload_batch(&self, dsts: &mut [u64], srcs: &mut [u64], host_srcs: &[u64], lens: &mut [usize], stream: *mut std::ffi::c_void) {
        let api = self.ctx.api;
        if dsts.is_empty() {
            return;
        }
        self.ctx.bind().unwrap();
        static NO_BATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if let Some(batch) = api.cuMemcpyBatchAsync.filter(|_| !*NO_BATCH.get_or_init(|| std::env::var("SHADOW_NO_BATCH_COPY").is_ok_and(|v| v == "1"))) {
            // CUmemcpyAttributes: source access order (STREAM = 1), two location hints (type, id), flags
            let mut attr = [1u32, 0, 0, 0, 0, 0];
            let mut idx = 0usize;
            // SAFETY: `n` device destinations inside this session's buffers and `n` sources inside pinned regions (the callers' slices), `lens`
            // bytes each, one attribute for all of them (index 0); the sources stay put until the stream is done with them (the callers' contract).
            let r = unsafe { batch(dsts.as_mut_ptr(), srcs.as_mut_ptr(), lens.as_mut_ptr(), dsts.len(), attr.as_mut_ptr() as *mut std::ffi::c_void, &mut idx, 1, stream) };
            assert_eq!(r, 0, "cuMemcpyBatchAsync failed ({r})");
            return;
        }
        for i in 0..dsts.len() {
            // SAFETY: as `upload_pinned_on`.
            let r = unsafe { (api.cuMemcpyHtoDAsync)(dsts[i], host_srcs[i] as *const std::ffi::c_void, lens[i], stream) };
            assert_eq!(r, 0, "cuMemcpyHtoDAsync failed ({r})");
        }
    }

    fn upload_pinned_on(&self, ptr: u64, data: &[u8], stream: *mut std::ffi::c_void) {
        let api = self.ctx.api;
        self.ctx.bind().unwrap();
        // SAFETY: `data` lies inside `PinnedRegion` (registered page-locked memory), so `cuMemcpyHtoDAsync` will not
        // stage through pageable memory.
        let r = unsafe { (api.cuMemcpyHtoDAsync)(ptr, data.as_ptr() as *const std::ffi::c_void, data.len(), stream) };
        assert_eq!(r, 0, "cuMemcpyHtoDAsync failed ({r})");
    }

    // -------------------------------------------------------------------------------------------- the forward pass
    pub fn step(&mut self, token: u32) -> Vec<f32> {
        let draft = if self.mtp_probe.is_some() {
            self.mtp_draft(token, 1, 0.0).first().copied()
        } else {
            None
        };
        let logits = self.step_opt(token, true);
        if let (Some(d), Some(p)) = (draft, self.mtp_probe.as_mut()) {
            let top = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32);
            p.0 += 1;
            p.1 += (top == Some(d)) as u64;
        }
        logits
    }

    pub fn step_prefill(&mut self, token: u32) {
        self.step_opt(token, false);
    }

    pub fn step_opt(&mut self, token: u32, with_logits: bool) -> Vec<f32> {
        if self.m.cfg.arch == "flint" {
            return self.step_flint_opt(token, with_logits);
        }
        assert!(self.tokens.len() < self.cap, "this session was made for {} positions", self.cap);
        self.tokens.push(token);
        self.routing.tokens += 1;
        self.mark("between steps (host)");
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden, c.streams);

        let emb = match self.picture_row(self.tokens.len() - 1) {
            Some(r) => r.to_vec(),
            None => {
                let (t, bytes) = self.m.dense("token_embd.weight");
                let (qk, qb) = t.ty.block().unwrap();
                let rb = (t.dims[0] / qk * qb) as usize;
                let mut x = vec![0f32; h];
                dequantize(t.ty, &bytes[token as usize * rb..(token as usize + 1) * rb], &mut x).unwrap();
                x
            }
        };
        self.upload_to(self.b.e.ptr, &f32s(&emb));
        self.k1("broadcast", h * cs, &[Arg::Ptr(self.b.e.ptr), Arg::Ptr(self.b.r.ptr), Arg::U32(h as u32), Arg::U32(cs as u32)]);
        self.rec_host("embed".into(), &emb);
        self.mark("embedding");

        let graphs = self.graphs_on && self.trace.is_none() && self.stop_after.is_none() && self.prof.is_none()
            && self.prefetch.is_none() && self.pregate.is_none() && self.cache.slots > 0;
        let mut rn_ready = false;
        for l in 0..c.n_layer {
            if self.stop_after == Some(l) {
                self.hc_read("output_hc", false, rn_ready);
                self.rec("final_u".into(), self.b.u.ptr, h);
                return vec![];
            }
            if l == c.ple_layer {
                self.ple(l);
                self.k1("add", h * cs, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.b.gv.ptr), Arg::U32((h * cs) as u32)]);
                self.rec(format!("layer{l}.ple"), self.b.gv.ptr, h * cs);
                rn_ready = false;
            }
            if graphs && !c.is_attention(l) && l != c.ple_layer {
                // everything up to the router read is the same launches every token: one graph
                self.graph_segment((l, rn_ready), |s| {
                    s.hc_read(&format!("blk.{l}.hc_attn"), true, rn_ready);
                    s.delta_net(l);
                    s.hc_write_norm(&format!("blk.{l}.hc_ffn_norm.weight"));
                    s.hc_read(&format!("blk.{l}.hc_ffn"), true, true);
                    s.moe_router(l);
                });
                self.moe(l, true);
            } else {
                self.hc_read(&format!("blk.{l}.hc_attn"), true, rn_ready);
                self.rec(format!("layer{l}.attn_hc_u"), self.b.u.ptr, h);
                if c.is_attention(l) {
                    self.attention(l);
                } else {
                    self.delta_net(l);
                }
                self.rec(format!("layer{l}.mixer"), self.b.y.ptr, h);
                self.hc_write_norm(&format!("blk.{l}.hc_ffn_norm.weight"));
                self.hc_read(&format!("blk.{l}.hc_ffn"), true, true);
                self.rec(format!("layer{l}.mlp_hc_u"), self.b.u.ptr, h);
                self.moe(l, false);
            }
            self.rec(format!("layer{l}.moe"), self.b.y.ptr, h);
            if l + 1 == c.ple_layer {
                self.hc_write();
                rn_ready = false;
            } else if l + 1 < c.n_layer && self.stop_after != Some(l + 1) {
                self.hc_write_norm(&format!("blk.{}.hc_attn_norm.weight", l + 1));
                rn_ready = true;
            } else {
                self.hc_write_norm("output_hc_norm.weight");
                rn_ready = true;
            }
            self.rec(format!("layer{l}.out"), self.b.r.ptr, h * cs);
            if self.dflash.is_some() {
                self.df_capture(l, self.b.r.ptr, 1);
            }
        }
        if self.mtp.is_some() {
            self.mtp_observe(self.b.r.ptr, &[token], self.tokens.len() - 1);
        }
        if self.dflash.is_some() {
            self.dflash_observe(1, self.tokens.len() - 1);
        }
        if !with_logits {
            return vec![];
        }
        self.hc_read("output_hc", false, rn_ready);
        self.rec("final_u".into(), self.b.u.ptr, h);
        self.mv("output.weight", self.b.u.ptr, self.b.logits.ptr);
        let mut out = vec![0u8; c.vocab * 4];
        self.download(self.b.logits.ptr, &mut out);
        self.mark("head + logits down");
        floats(&out)
    }

    fn step_flint_opt(&mut self, token: u32, with_logits: bool) -> Vec<f32> {
        assert!(self.tokens.len() < self.cap, "this session was made for {} positions", self.cap);
        self.tokens.push(token);
        self.routing.tokens += 1;
        self.mark("between steps (host)");
        let c = &self.m.cfg;
        let h = c.hidden;

        let emb = {
            let (t, bytes) = self.m.dense("token_embd.weight");
            let (qk, qb) = t.ty.block().unwrap();
            let rb = (t.dims[0] / qk * qb) as usize;
            let mut x = vec![0f32; h];
            dequantize(t.ty, &bytes[token as usize * rb..(token as usize + 1) * rb], &mut x).unwrap();
            let s = self.m.scale("token_embd.weight");
            if s != 1.0 {
                for v in &mut x {
                    *v *= s;
                }
            }
            x
        };
        self.upload_to(self.b.r.ptr, &f32s(&emb));
        self.rec_host("embed".into(), &emb);
        self.mark("embedding");

        for l in 0..c.n_layer {
            if self.stop_after == Some(l) {
                self.k("rmsnorm_groups", 1, 256, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.sv("output_norm.weight")),
                                                   Arg::Ptr(self.b.u.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
                self.rec("final_u".into(), self.b.u.ptr, h);
                return vec![];
            }
            if l == 0 || self.trace.is_some() {
                self.k("rmsnorm_groups", 1, 256, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.sv(&format!("blk.{l}.attn_norm.weight"))),
                                                   Arg::Ptr(self.b.u.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
            }
            self.rec(format!("layer{l}.attn_hc_u"), self.b.u.ptr, h);
            self.attention_flint(l);
            self.rec(format!("layer{l}.mixer"), self.b.y.ptr, h);
            self.k("add_rmsnorm1", 1, 256, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.b.y.ptr),
                                             Arg::Ptr(self.sv(&format!("blk.{l}.ffn_norm.weight"))),
                                             Arg::Ptr(self.b.u.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
            self.rec(format!("layer{l}.mlp_hc_u"), self.b.u.ptr, h);
            if c.is_moe(l) {
                self.moe(l, false);
            } else {
                self.mlp_flint(l);
            }
            self.rec(format!("layer{l}.moe"), self.b.y.ptr, h);
            if self.trace.is_some() || self.stop_after == Some(l + 1) {
                self.k1("add", h, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.b.y.ptr), Arg::U32(h as u32)]);
                self.rec(format!("layer{l}.out"), self.b.r.ptr, h);
            } else if l + 1 < c.n_layer {
                self.k("add_rmsnorm1", 1, 256, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.b.y.ptr),
                                                 Arg::Ptr(self.sv(&format!("blk.{}.attn_norm.weight", l + 1))),
                                                 Arg::Ptr(self.b.u.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
            } else {
                self.k("add_rmsnorm1", 1, 256, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.b.y.ptr),
                                                 Arg::Ptr(self.sv("output_norm.weight")),
                                                 Arg::Ptr(self.b.u.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
            }
        }
        if !with_logits {
            return vec![];
        }
        if self.trace.is_some() {
            self.k("rmsnorm_groups", 1, 256, &[Arg::Ptr(self.b.r.ptr), Arg::Ptr(self.sv("output_norm.weight")),
                                               Arg::Ptr(self.b.u.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
        }
        self.rec("final_u".into(), self.b.u.ptr, h);
        self.mv("output.weight", self.b.u.ptr, self.b.logits.ptr);
        let mut out = vec![0u8; c.vocab * 4];
        self.download(self.b.logits.ptr, &mut out);
        self.mark("head + logits down");
        floats(&out)
    }

    /// The hyper-connection read: u (and g, with `inject`) from the streams r.
    fn hc_read(&self, p: &str, inject: bool, skip_norm: bool) {
        let b = &self.b;
        if !skip_norm {
            self.hc_norm(p, b.r.ptr, b.rn.ptr);
        }
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden as u32, c.streams as u32);
        let r_down = c.hc_rank as u32;
        let (w_inj, r_inj) = if inject {
            (self.w[&format!("{p}_inject.weight")].ptr, cs)
        } else {
            (0, 0)
        };
        unsafe {
            let (entry, per_block) = if self.fast_order { ("hc_down_inject_fast_bf16", 2) } else { ("hc_down_inject_bf16", 8) };
            self.mvp.launch(entry, ((r_down + r_inj).div_ceil(per_block), 1, 1), (256, 1, 1), 0, self.ls(), &[
                Arg::Ptr(self.w[&format!("{p}_down.weight")].ptr), Arg::Ptr(b.down.ptr), Arg::U32(r_down),
                Arg::Ptr(w_inj), Arg::Ptr(b.g.ptr), Arg::U32(r_inj),
                Arg::Ptr(b.rn.ptr), Arg::U32(h * cs), Arg::F32(cs as f32),
            ]).unwrap_or_else(|err| panic!("hc_down_inject_bf16: {err}"));
        }
        if self.fuse & 2 != 0 && self.fast_order {
            self.hc_up_mix(p, b.down.ptr, c.hc_rank, b.rn.ptr, h as usize * cs as usize, b.u.ptr, h as usize, 1);
        } else {
            self.mv(&format!("{p}_up.weight"), b.down.ptr, b.up.ptr);
            self.hc_mix(b.rn.ptr, b.up.ptr, b.u.ptr);
        }
    }

    /// u = the hyper-connection mix of rn and sigmoid(W_up down) for m rows (`hc_up_mix_cols`): the up-projection
    /// never leaves the kernel.  Rows are ds / rs / us floats apart.
    #[allow(clippy::too_many_arguments)]
    fn hc_up_mix(&self, p: &str, down: u64, ds: usize, rn: u64, rs: usize, u: u64, us: usize, m: usize) {
        let c = &self.m.cfg;
        let (h, cs, rank) = (c.hidden as u32, c.streams as u32, c.hc_rank as u32);
        let w = self.w[&format!("{p}_up.weight")].ptr;
        for j0 in (0..m).step_by(COLS_MAX) {
            let n = (m - j0).min(COLS_MAX) as u32;
            // SAFETY: h x streams BF16 rows of `rank` weights; n rows of down (rank floats), rn (h * streams) and u (h).
            unsafe {
                self.mvp.launch("hc_up_mix_cols", (h.div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(), &[
                    Arg::Ptr(w), Arg::Ptr(down + (j0 * ds * 4) as u64), Arg::U32(ds as u32), Arg::Ptr(rn + (j0 * rs * 4) as u64),
                    Arg::U32(rs as u32), Arg::Ptr(u + (j0 * us * 4) as u64), Arg::U32(us as u32), Arg::U32(h), Arg::U32(cs),
                    Arg::U32(rank), Arg::U32(n)])
            }
            .unwrap_or_else(|err| panic!("hc_up_mix_cols: {err}"));
        }
    }

    /// The head-wise gated RMSNorm of the DeltaNet output for the batch's m rows (`bb.o` normalized in place, gated by `bb.z`): for heads of 128 a warp a (head, row) with the
    /// block tree's sum order in registers (bit for bit the same; `SHADOW_GN_W=0` keeps the block-a-head kernel).
    fn gated_norm_rows(&self, weight: &str, d: usize, vh: usize, vd: usize, m: usize) {
        let bb = &self.bb;
        let eps = self.m.cfg.eps;
        if d == 128 && !std::env::var("SHADOW_GN_W").is_ok_and(|v| v == "0") {
            self.k2("gated_rmsnorm_rows_w", (vh as u32).div_ceil(4), m as u32, 128, &[Arg::Ptr(bb.o.ptr), Arg::Ptr(self.sv(weight)), Arg::Ptr(bb.z.ptr),
                                                                                    Arg::U32(d as u32), Arg::F32(eps), Arg::U32(vd as u32), Arg::U32(vh as u32)]);
        } else {
            self.k2("gated_rmsnorm_rows", vh as u32, m as u32, 128, &[Arg::Ptr(bb.o.ptr), Arg::Ptr(self.sv(weight)), Arg::Ptr(bb.z.ptr),
                                                                     Arg::U32(d as u32), Arg::F32(eps), Arg::U32(vd as u32)]);
        }
    }

    fn hc_norm(&self, p: &str, r: u64, rn: u64) {
        let c = &self.m.cfg;
        self.k("rmsnorm_groups", c.streams as u32, 256, &[Arg::Ptr(r), Arg::Ptr(self.sv(&format!("{p}_norm.weight"))),
                                                          Arg::Ptr(rn), Arg::U32(c.hidden as u32), Arg::F32(c.eps)]);
    }

    fn hc_mix(&self, rn: u64, up: u64, u: u64) {
        let c = &self.m.cfg;
        self.k1("hc_mix", c.hidden, &[Arg::Ptr(rn), Arg::Ptr(up), Arg::Ptr(u), Arg::U32(c.hidden as u32),
                                      Arg::U32(c.streams as u32)]);
    }

    fn hc_write(&self) {
        self.hc_write_at(self.b.r.ptr, self.b.g.ptr, self.b.y.ptr);
    }

    fn hc_write_at(&self, r: u64, g: u64, y: u64) {
        let c = &self.m.cfg;
        self.k1("hc_write", c.hidden * c.streams, &[Arg::Ptr(r), Arg::Ptr(g), Arg::Ptr(y), Arg::U32(c.hidden as u32),
                                                   Arg::U32(c.streams as u32)]);
    }

    fn hc_write_norm(&self, next_norm: &str) {
        self.hc_write_norm_at(next_norm, self.b.r.ptr, self.b.g.ptr, self.b.y.ptr, self.b.rn.ptr);
    }

    fn hc_write_norm_at(&self, next_norm: &str, r: u64, g: u64, y: u64, rn: u64) {
        let c = &self.m.cfg;
        self.k("hc_write_rmsnorm_groups", c.streams as u32, 256, &[
            Arg::Ptr(r), Arg::Ptr(g), Arg::Ptr(y), Arg::Ptr(self.sv(next_norm)), Arg::Ptr(rn), Arg::U32(c.hidden as u32),
            Arg::F32(c.eps)]);
    }

    fn delta_net(&self, l: usize) {
        let b = &self.b;
        let p = format!("blk.{l}");
        self.mv_pair(&format!("{p}.attn_qkv.weight"), b.qkv.ptr, &format!("{p}.attn_gate.weight"), b.z.ptr, b.u.ptr);
        self.mv_pair(&format!("{p}.ssm_beta.weight"), b.beta.ptr, &format!("{p}.ssm_alpha.weight"), b.alpha.ptr, b.u.ptr);
        self.delta_mid(l, b.qkv.ptr, b.z.ptr, b.beta.ptr, b.alpha.ptr, b.o.ptr);
        self.mv(&format!("{p}.ssm_out.weight"), b.o.ptr, b.y.ptr);
    }

    /// DeltaNet's per-token part, in order: the conv window, the q/k norms, the state update, the gated norm into o.
    fn delta_mid(&self, l: usize, qkv: u64, z: u64, beta: u64, alpha: u64, o: u64) {
        let c = &self.m.cfg;
        let b = &self.b;
        let p = format!("blk.{l}");
        let (kh, vh, d) = (c.ssm_k_heads, c.ssm_v_heads, c.ssm_head_dim);
        let kd = kh * d;
        let ch = 2 * kd + vh * d;
        let (s, conv) = self.delta[l].as_ref().unwrap();
        self.k1("conv_silu_step", ch, &[Arg::Ptr(qkv), Arg::Ptr(conv.ptr), Arg::Ptr(self.sv(&format!("{p}.ssm_conv1d.weight"))),
                                       Arg::Ptr(b.x.ptr), Arg::U32(ch as u32), Arg::U32(c.conv_k as u32)]);
        self.k("l2norm_qk_heads", (2 * kh) as u32, 128, &[
            Arg::Ptr(b.x.ptr), Arg::U32(kh as u32), Arg::U32(d as u32), Arg::F32((d as f32).sqrt())]);
        self.k("delta_step", vh as u32, d as u32, &[
            Arg::Ptr(b.x.ptr), Arg::Ptr(b.x.ptr + (kd * 4) as u64), Arg::Ptr(b.x.ptr + (2 * kd * 4) as u64),
            Arg::Ptr(beta), Arg::Ptr(alpha), Arg::Ptr(self.sv(&format!("{p}.ssm_a"))),
            Arg::Ptr(self.sv(&format!("{p}.ssm_dt.bias"))), Arg::Ptr(s.ptr), Arg::Ptr(o), Arg::U32(kh as u32),
            Arg::U32(d as u32)]);
        self.k("gated_rmsnorm_heads", vh as u32, 128, &[Arg::Ptr(o), Arg::Ptr(self.sv(&format!("{p}.ssm_norm.weight"))),
                                                       Arg::Ptr(z), Arg::U32(d as u32), Arg::F32(c.eps)]);
    }

    fn attention(&mut self, l: usize) {
        let p = format!("blk.{l}");
        let c = &self.m.cfg;
        let (nkv, hd, id) = (c.n_kv_head, c.head_dim, c.idx_dim);
        let pos = self.tokens.len() - 1;
        let (qg, kk, iq, ao) = (self.b.qg.ptr, self.b.kk.ptr, self.b.iq.ptr, self.b.ao.ptr);
        {
            let b = &self.b;
            let at = self.attn[l].as_ref().unwrap();
            let slot = (pos * nkv * hd * 4) as u64;
            let raw_slot = at.raw.ptr + (pos * id * 4) as u64;
            self.mv(&format!("{p}.attn_q.weight"), b.u.ptr, qg);
            self.mv_pair(&format!("{p}.attn_k.weight"), kk, &format!("{p}.attn_v.weight"), at.v.ptr + slot, b.u.ptr);
            self.mv_pair(&format!("{p}.indexer.q_proj.weight"), iq, &format!("{p}.indexer.k_proj.weight"), raw_slot, b.u.ptr);
        }
        self.attention_mid(l, pos, qg, kk, iq, ao);
        self.mv(&format!("{p}.attn_output.weight"), ao, self.b.y.ptr);
    }

    /// Attention's per-position part, once its projections are in (v and the indexer key already in the caches at
    /// `pos`): q/k norms and RoPE (k into the cache), the indexer's pooled blocks and selection, attention over the
    /// positions <= pos into ao.
    fn attention_mid(&mut self, l: usize, pos: usize, qg: u64, kk: u64, iq: u64, ao: u64) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let (id, ratio) = (c.idx_dim, c.compress[l].max(1));
        let (rd, th, eps) = (c.rope_dims as u32, c.rope_theta as f32, c.eps);
        {
            let b = &self.b;
            let at = self.attn[l].as_ref().unwrap();
            let slot = (pos * nkv * hd * 4) as u64;
            self.k("head_norm_rope_qgk", (nh + nkv) as u32, 128, &[
                Arg::Ptr(qg), Arg::Ptr(b.q.ptr), Arg::Ptr(b.gate.ptr), Arg::Ptr(self.sv(&format!("{p}.attn_q_norm.weight"))),
                Arg::Ptr(kk), Arg::Ptr(at.k.ptr + slot), Arg::Ptr(self.sv(&format!("{p}.attn_k_norm.weight"))),
                Arg::U32(nh as u32), Arg::U32(hd as u32), Arg::F32(eps), Arg::U32(pos as u32), Arg::U32(rd), Arg::F32(th)]);
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
                Arg::Ptr(iq), Arg::U32(id as u32), Arg::Ptr(b.iqn.ptr), Arg::U32(id as u32),
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
                Arg::Ptr(b.gate.ptr), Arg::Ptr(ao), Arg::Ptr(b.scratch.ptr), Arg::U32(self.cap as u32),
                Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32((nh / nkv) as u32)]);
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

    fn attention_flint(&mut self, l: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
        let pos = self.tokens.len() - 1;
        let rd = if c.rope_pattern.get(l).copied().unwrap_or(true) { c.rope_dims as u32 } else { 0 };
        let (th, eps) = (c.rope_theta as f32, c.eps);
        let n = pos + 1;
        let start = if c.sliding_pattern.get(l).copied().unwrap_or(false) {
            n.saturating_sub(c.sliding_window)
        } else {
            0
        };
        let ntok = n - start;
        let b = &self.b;
        let at = self.attn[l].as_ref().unwrap();
        let slot = (pos * nkv * hd * 4) as u64;
        let (qn, kn, vn) = (format!("{p}.attn_q.weight"), format!("{p}.attn_k.weight"), format!("{p}.attn_v.weight"));
        let (rq, rk, rv) = ((nh * hd) as u32, (nkv * hd) as u32, (nkv * hd) as u32);
        unsafe {
            self.mvp.launch("matvec_qkv_nvfp4", ((rq + rk + rv).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(), &[
                Arg::Ptr(self.w[&qn].ptr), Arg::F32(self.m.scale(&qn)), Arg::Ptr(b.qg.ptr), Arg::U32(rq),
                Arg::Ptr(self.w[&kn].ptr), Arg::F32(self.m.scale(&kn)), Arg::Ptr(b.kk.ptr), Arg::U32(rk),
                Arg::Ptr(self.w[&vn].ptr), Arg::F32(self.m.scale(&vn)), Arg::Ptr(at.v.ptr + slot), Arg::U32(rv),
                Arg::Ptr(b.u.ptr), Arg::U32(c.hidden as u32),
            ]).unwrap_or_else(|err| panic!("matvec_qkv_nvfp4: {err}"));
        }
        self.k("head_norm_rope_qk", (nh + nkv) as u32, 128, &[
            Arg::Ptr(b.qg.ptr), Arg::Ptr(b.q.ptr), Arg::Ptr(self.sv(&format!("{p}.attn_q_norm.weight"))),
            Arg::Ptr(b.kk.ptr), Arg::Ptr(at.k.ptr + slot), Arg::Ptr(self.sv(&format!("{p}.attn_k_norm.weight"))),
            Arg::U32(nh as u32), Arg::U32(hd as u32), Arg::F32(eps), Arg::U32(pos as u32), Arg::U32(rd), Arg::F32(th)]);
        self.k("attention_sinks_heads", nh as u32, 256, &[
            Arg::Ptr(b.q.ptr), Arg::Ptr(at.k.ptr), Arg::Ptr(at.v.ptr), Arg::Ptr(self.sv(&format!("{p}.attn_sinks.weight"))),
            Arg::U32(start as u32), Arg::U32(ntok as u32), Arg::Ptr(b.ao.ptr), Arg::Ptr(b.scratch.ptr),
            Arg::U32(self.cap as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32((nh / nkv) as u32)]);
        self.mv(&format!("{p}.attn_output.weight"), b.ao.ptr, b.y.ptr);
    }

    fn mlp_flint(&mut self, l: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let ff = c.dense_ff;
        self.mark("mixers, norms, hc glue, ple");
        let b = &self.b;
        self.mv(&format!("{p}.ffn_gate.weight"), b.u.ptr, b.eg.ptr);
        self.mv(&format!("{p}.ffn_up.weight"), b.u.ptr, b.eu.ptr);
        self.k1("swiglu", ff, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(b.eu.ptr), Arg::Ptr(b.act.ptr), Arg::U32(ff as u32)]);
        self.mv(&format!("{p}.ffn_down.weight"), b.act.ptr, b.y.ptr);
        self.mark("shared expert + combine");
    }

    /// Puts expert e of layer l (its gate, up and down slices) at the three device addresses: a DMA copy from the
    /// pinned RAM tier if it holds the expert, otherwise read from the file (promoting into `RamTier`'s Clock ring if
    /// active, else into the pinned staging buffer) and copied on `copy_stream`.  Those file reads wait in `pending`
    /// for `read_drive`.
    fn fetch_expert(&mut self, l: usize, e: usize, slot_idx: usize, names: &[String; 3], sizes: &[usize], dst: [u64; 3]) {
        if std::env::var_os("SHADOW_TIMING_NO_COPIES").is_some() {
            return; // timing experiment only: the slot keeps whatever it held, so the output is wrong
        }
        let stream = self.copy_stream.raw();
        let contiguous = dst[1] == dst[0] + sizes[0] as u64 && dst[2] == dst[1] + sizes[1] as u64;
        let total = sizes[0] + sizes[1] + sizes[2];
        let part_offs = [0usize, sizes[0], sizes[0] + sizes[1]];
        if let Some(&off) = self.ram.at.get(&(l, e)) {
            if let Some(&r_idx) = self.ram.ring_by_off.get(&off) {
                self.ram.ring_ref[r_idx] = true;
                self.ram.claimed.insert(off);
            }
            let src = self.ram.region.as_ref().unwrap().as_slice();
            if contiguous {
                self.upload_pinned_on(dst[0], &src[off..off + total], stream);
            } else {
                for i in 0..3 {
                    self.upload_pinned_on(dst[i], &src[off + part_offs[i]..off + part_offs[i] + sizes[i]], stream);
                }
            }
            self.ram.served += 1;
            self.mark("expert upload (pinned RAM)");
        } else if let Some(off) = self.ram.promote_slot((l, e), total.next_multiple_of(256)) {
            self.pending.push(PendingRead { e, ring: true, off, dst });
        } else if self.staging.is_some() {
            self.pending.push(PendingRead { e, ring: false, off: slot_idx * self.cache.slot_bytes, dst });
        } else {
            let slices: Vec<Vec<u8>> = names.iter().map(|n| self.m.expert(n, e).1).collect();
            self.mark("expert read (file)");
            for i in 0..3 {
                self.upload_to(dst[i], &slices[i]);
            }
            self.ram.drive += 1;
            self.mark("expert upload (pageable)");
        }
        self.expert_uploads += 3;
    }
    /// Reads every pending expert of this layer from the file at once (`Model::experts_into_many`), then queues their
    /// copies to the device on `copy_stream`.
    fn read_drive(&mut self, names: &[String; 3], sizes: &[usize]) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let total = sizes[0] + sizes[1] + sizes[2];
        let part_offs = [0usize, sizes[0], sizes[0] + sizes[1]];
        {
            let ring = self.ram.region.as_mut().map(|r| r.as_mut_slice());
            let staging = self.staging.as_mut().map(|r| r.as_mut_slice());
            let mut jobs: Vec<(&str, usize, &mut [u8])> = Vec::with_capacity(3 * pending.len());
            for (from_ring, buf) in [(true, ring), (false, staging)] {
                let Some(mut rest) = buf else { continue };
                let mut mine: Vec<&PendingRead> = pending.iter().filter(|p| p.ring == from_ring).collect();
                mine.sort_by_key(|p| p.off);
                let mut at = 0usize;
                for p in mine {
                    // carve the expert's bytes off the front of what is left; the offsets are distinct slots
                    let (_, tail) = std::mem::take(&mut rest).split_at_mut(p.off - at);
                    let (mut slot, tail) = tail.split_at_mut(total);
                    rest = tail;
                    at = p.off + total;
                    // file-offset order: `ffn_down_exps` @ part 2, `ffn_gate_exps` @ part 0, `ffn_up_exps` @ part 1
                    let (gate, more) = std::mem::take(&mut slot).split_at_mut(sizes[0]);
                    let (up, down) = more.split_at_mut(sizes[1]);
                    jobs.push((&names[2], p.e, down));
                    jobs.push((&names[0], p.e, gate));
                    jobs.push((&names[1], p.e, up));
                }
            }
            let t_wait = std::time::Instant::now();
            self.m.experts_into_many(&mut jobs);
            self.waits[2] += t_wait.elapsed().as_secs_f64();
        }
        self.mark("expert read (file)");
        let stream = self.copy_stream.raw();
        for p in &pending {
            let region = if p.ring { self.ram.region.as_ref() } else { self.staging.as_ref() };
            let src = &region.unwrap().as_slice()[p.off..p.off + total];
            if p.dst[1] == p.dst[0] + sizes[0] as u64 && p.dst[2] == p.dst[1] + sizes[1] as u64 {
                self.upload_pinned_on(p.dst[0], src, stream);
            } else {
                for i in 0..3 {
                    self.upload_pinned_on(p.dst[i], &src[part_offs[i]..part_offs[i] + sizes[i]], stream);
                }
            }
            self.ram.drive += 1;
        }
        self.mark("expert upload (pageable)");
    }

    /// Copies layer `l`'s likeliest experts (by its router's `logits` on the previous layer's input) from the RAM tier
    /// into VRAM cache slots on `copy_stream`, without waiting: the cache treats them as resident from now on, and
    /// layer l waits for `copy_event` (or for `copy_stream`) before its experts run.
    fn prefetch_next(&mut self, l: usize, logits: &[f32]) {
        let k = self.prefetch.as_ref().unwrap().k;
        let names = ["gate", "up", "down"].map(|p| format!("blk.{l}.ffn_{p}_exps.weight"));
        let sizes: Vec<usize> = names.iter().map(|n| self.m.expert_bytes(n)).collect();
        let total: usize = sizes.iter().sum();
        let stream = self.copy_stream.raw();
        let mut n = 0u64;
        for (e, _) in route(logits, k) {
            if self.cache.map.contains_key(&(l, e)) {
                continue;
            }
            let Some(&off) = self.ram.at.get(&(l, e)) else { continue };
            if !self.cache.has_victim(l) {
                break;
            }
            if let Some(&r) = self.ram.ring_by_off.get(&off) {
                self.ram.ring_ref[r] = true;
            }
            let s = self.cache.victim(l);
            let base = self.cache.arena.as_ref().unwrap().ptr + self.cache.slot_offset[s];
            let src = &self.ram.region.as_ref().unwrap().as_slice()[off..off + total];
            self.upload_pinned_on(base, src, stream);
            self.cache.owner[s] = Some((l, e));
            self.cache.map.insert((l, e), s);
            self.cache.stamp[s] = self.cache.clock;
            self.prefetch.as_mut().unwrap().ahead.insert((l, e));
            n += 1;
        }
        if n > 0 {
            self.copy_event.record(Some(&self.copy_stream)).unwrap();
            let pf = self.prefetch.as_mut().unwrap();
            pf.wait = true;
            pf.issued += n;
            self.ram.served += n;
            self.expert_uploads += 3 * n as usize;
        }
    }

    /// Lends the tail of the VRAM expert cache (`SHADOW_PREFILL_LEND_MIB`, else room for all of a layer's experts, at most half
    /// of it) to a long prompt as staging: a prompt chunk's whole expert union for a layer is then copied once and run in one
    /// group, instead of in groups of the few free slots.  The evicted entries come back with `restore_cache`.  False when
    /// there is nothing to lend (no cache, a pinned placement).
    /// Tokens a pass can take now: `MAX_BATCH`, more while a long prompt has the cache lent.
    pub fn batch_rows(&self) -> usize {
        self.batch_max
    }

    /// The engine tells the session how many tokens the next long prompt reads (so `lend_kv` knows how much of the window it will fill).
    pub fn hint_prompt(&mut self, n: usize) {
        self.prompt_hint = n;
    }

    /// `SHADOW_KV_LEND=1`: hand the KV window's unfilled tail (past this prompt and 256 positions more) back to the driver and take it as one ordinary buffer for
    /// the prompt's batch buffers; true if there is such a buffer.  The device must be idle (it is: this runs before the prompt).
    fn lend_kv(&mut self, with_halves: bool) -> bool {
        if self.kv_regions.is_empty() || self.kv_extra.is_some() || self.prompt_hint == 0 {
            return false;
        }
        let _ = self.ctx.synchronize();
        let keep = self.tokens.len() + self.prompt_hint + 256;
        let freed: usize = self.kv_regions.iter_mut().map(|r| r.release_tail(keep)).sum();
        let rb = bb_row_floats(&self.m.cfg) * 4;
        // twice the rows when the tail is that big (a long window, a short prompt); the rows are one of two values, so the graphs captured per row count stay valid
        let halves = if with_halves { 2 * self.m.cfg.n_expert * self.cache.slot_bytes } else { 0 };
        let avail = freed.saturating_sub(halves + (1 << 16) + (8 << 20));
        let cap = std::env::var("SHADOW_BIG_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(BIG_BATCH);
        // the rows are one of a few values, so the graphs captured per row count stay few
        let rows = [BIG_BATCH_MAX, 4096, BIG_BATCH, 1024, 512].into_iter().find(|&r| r <= cap.max(512) && rb * r <= avail).unwrap_or(0);
        // with the halves in the arena, a smaller batch than the arena itself can carve (1,600 rows at the 24k window) is no reason to take the KV tail
        // a tail too small for a pass alone still adds to the arena's part (`lend_cache` lends the rest of the rows' buffers from the arena)
        let rows = if !with_halves && rows < 512 { 0 } else { rows };
        let want = halves + rb * rows + (1 << 16);
        if self.kv_extra_reg.is_none() {
            let size = 2 * self.m.cfg.n_expert * self.cache.slot_bytes + rb * BIG_BATCH_MAX + (1 << 16);
            self.kv_extra_reg = crate::kvvmm::VmmRegion::new_empty(self.ctx, size).ok();
        }
        if rows > 0 {
            if let Some(reg) = self.kv_extra_reg.as_mut() {
                if reg.map_bytes(want).is_ok() {
                    // SAFETY: a view of the region's mapped range; the session owns the region.
                    self.kv_extra = Some(unsafe { DeviceBuffer::view(self.ctx, reg.base, want) });
                    self.kv_rows = rows;
                    return true;
                }
                reg.release_tail(0);
            }
        }
        for r in &mut self.kv_regions {
            let _ = r.remap_tail();
        }
        false
    }

    /// Give the borrowed memory back: free the buffer, map the window's tail again (a failure here is fatal: the window would be missing memory).
    fn unlend_kv(&mut self) {
        if self.kv_regions.is_empty() || (self.kv_extra.is_none() && self.kv_regions.iter().all(|r| r.is_full())) {
            return;
        }
        self.kv_extra = None;
        self.kv_halves = false;
        if let Some(reg) = self.kv_extra_reg.as_mut() {
            reg.release_tail(0);
        }
        for r in &mut self.kv_regions {
            r.remap_tail().unwrap_or_else(|err| panic!("KV window: {err}"));
        }
    }

    /// `long`: a prompt long enough (`LONG_PROMPT` tokens) to pay for the layer-ahead pipeline and the big batch (which allocate pinned
    /// buffers and read the evicted experts back from the file afterwards); a shorter one lends the tail only.
    pub fn lend_cache(&mut self, long: bool) -> bool {
        if self.lent.is_some() || self.cache.slots == 0 || self.cache.arena.is_none() || self.cache.layer_range != self.cache.evict_range {
            return false;
        }
        let pipe = long && !std::env::var("SHADOW_PREFILL_PIPE").is_ok_and(|v| v == "0");
        let big = !std::env::var("SHADOW_PREFILL_BIG").is_ok_and(|v| v == "0");
        // the staging halves (two experts-sets of slots) are lent from the arena when it is big enough, else (a long window leaves a small cache) from the KV window's tail with the batch buffers
        let (ne, sb) = (self.m.cfg.n_expert, self.cache.slot_bytes);
        let pct = std::env::var("SHADOW_LEND_PCT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(99);
        let cap_bytes = (self.cache.arena_bytes / 100 * pct).min(self.cache.arena_bytes.saturating_sub(24 * sb));
        let arena_hosts = cap_bytes >= (2 * ne + 6) * sb;
        if pipe && big && !arena_hosts && self.lend_kv(true) {
            let slots = std::env::var("SHADOW_PREFILL_STAGING").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256);
            let need = slots * sb;
            if self.staging.as_ref().is_some_and(|s| s.len() < need) {
                if let Ok(r) = PinnedRegion::new(self.ctx, need) {
                    self.staging = Some(r);
                }
            }
            let ptr = self.kv_extra.as_ref().unwrap().ptr;
            self.kv_halves = true;
            self.lent = Some(Lent { ptr, n: 0, saved: vec![], evict_saved: self.cache.evict_range.clone() });
            self.stage_pipe_init(ptr);
            self.big_batch_on();
            return true;
        }
        let kv_lent = pipe && big && arena_hosts && self.lend_kv(false);
        let big_cap = big_rows_cap();
        // more rows than the KV tail holds (`SHADOW_BIG_ROWS`): the rest of their buffers from the arena, past the halves (256 MiB more for the carving's gaps)
        let extra = if kv_lent {
            let cap = big_rows_cap();
            (big_batch_bytes(&self.m.cfg, cap) + (256 << 20)).saturating_sub(self.kv_extra.as_ref().map_or(0, |x| x.bytes))
        } else { 0 };
        let want = std::env::var("SHADOW_PREFILL_LEND_MIB").ok().and_then(|v| v.parse::<usize>().ok()).map(|m| m << 20)
            .unwrap_or((self.m.cfg.n_expert * if pipe { 2 } else { 1 } + if pipe { 2 } else { 0 }) * self.cache.slot_bytes
                       + if pipe && big && !kv_lent { big_batch_bytes(&self.m.cfg, big_cap) + (1 << 16) } else { extra })
            .min((self.cache.arena_bytes / 100 * std::env::var("SHADOW_LEND_PCT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(99)).min(self.cache.arena_bytes.saturating_sub(24 * self.cache.slot_bytes)));
        let target = self.cache.arena_bytes - want;
        let s0 = self.cache.slot_offset.partition_point(|&o| (o as usize) < target);
        if s0 >= self.cache.slots {
            self.unlend_kv();
            return false;
        }
        let start = self.cache.slot_offset[s0] as usize;
        let n = (self.cache.arena_bytes - start) / self.cache.slot_bytes.max(1);
        if n < 2 {
            self.unlend_kv();
            return false;
        }
        self.ctx.synchronize().unwrap(); // nothing is still reading the entries about to go
        let mut saved = vec![];
        for s in s0..self.cache.slots {
            if let Some((l, e)) = self.cache.owner[s].take() {
                self.cache.map.remove(&(l, e));
                saved.push((s, l, e));
            }
        }
        let evict_saved = self.cache.evict_range.clone();
        for r in &mut self.cache.evict_range {
            if r.start >= s0 {
                *r = r.start..r.start;
            } else if r.end > s0 {
                r.end = s0;
            }
        }
        // the file reads of a layer's misses wait in the staging buffer, a slot each: room for a prompt chunk's share of them
        // (`SHADOW_PREFILL_STAGING` slots), kept after the prompt
        let slots = std::env::var("SHADOW_PREFILL_STAGING").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256);
        let need = slots * self.cache.slot_bytes;
        if self.staging.as_ref().is_some_and(|s| s.len() < need) {
            if let Ok(r) = PinnedRegion::new(self.ctx, need) {
                self.staging = Some(r);
            }
        }
        if std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "1") {
            eprintln!("LEND long {long} pipe {pipe} big {big} kv_lent {kv_lent}: {} of {} slots lent (n {n}, need {} for the pipeline), want {:.2} GiB", self.cache.slots - s0, self.cache.slots, 2 * self.m.cfg.n_expert, want as f64 / 1073741824.0);
        }
        let ptr = self.cache.arena.as_ref().unwrap().ptr + start as u64;
        self.lent = Some(Lent { ptr, n, saved, evict_saved });
        if pipe && n >= 2 * self.m.cfg.n_expert {
            self.stage_pipe_init(ptr);
            if big {
                self.big_batch_on();
            }
        }
        true
    }

    /// Two pinned read-ahead buffers (a slot for the most file-only experts a layer has) and the copy events, for the lent cache at `base`.
    fn stage_pipe_init(&mut self, base: u64) {
        let ne = self.m.cfg.n_expert;
        let mut per_layer = vec![0usize; self.m.cfg.n_layer];
        for &(l, _) in self.ram.at.keys() {
            per_layer[l] += 1;
        }
        let mut slots = per_layer.iter().map(|&n| ne.saturating_sub(n)).max().unwrap_or(ne).max(1);
        // debugging a direct-I/O crash: `SHADOW_STAGE_SLOTS=n` sizes the read-ahead buffers for n experts (512 = every expert of a layer, so no buffer is ever replaced)
        if let Some(n) = std::env::var("SHADOW_STAGE_SLOTS").ok().and_then(|v| v.parse::<usize>().ok()) {
            slots = n.max(1);
        }
        let bytes = slots * self.cache.slot_bytes;
        let (Ok(a), Ok(b), Ok(c3), Ok(e0), Ok(e1)) =
            (PinnedRegion::new(self.ctx, bytes), PinnedRegion::new(self.ctx, bytes), PinnedRegion::new(self.ctx, bytes), self.ctx.event(), self.ctx.event()) else { return };
        let (Ok(d0), Ok(d1), Ok(b0), Ok(b1), Ok(b2)) = (self.ctx.event(), self.ctx.event(), self.ctx.event(), self.ctx.event(), self.ctx.event()) else { return };
        self.sp = Some(StagePipe { base, events: [e0, e1], done: [d0, d1], ready: [usize::MAX; 2], bufs: [a, b, c3], buf_done: [b0, b1, b2], layer: [usize::MAX; 3],
                                   at: [HashMap::new(), HashMap::new(), HashMap::new()], slots, pending: false });
    }

    /// Waits for the read-ahead's file reads.
    fn stage_pipe_wait(&mut self) {
        if let Some(sp) = self.sp.as_mut() {
            if sp.pending {
                let t0 = std::time::Instant::now();
                self.m.experts_wait();
                heap_check("experts_wait (read-ahead complete)");
                self.waits[2] += t0.elapsed().as_secs_f64();
                sp.pending = false;
            }
        }
    }

    /// Starts reading layer `l`'s file-only experts (in neither the VRAM cache nor the RAM tier) on the read pool into the buffer of l's parity.
    fn stage_pipe_read(&mut self, l: usize) {
        if self.sp.is_none() || l >= self.m.cfg.n_layer {
            return;
        }
        self.stage_pipe_wait();
        if self.fv_on {
            // the file-only experts are copied from file views by `stage_copy`: no read-ahead
            let sp = self.sp.as_mut().unwrap();
            sp.layer[l % 3] = l;
            sp.at[l % 3].clear();
            return;
        }
        let names = ["gate", "up", "down"].map(|p| format!("blk.{l}.ffn_{p}_exps.weight"));
        let sizes: Vec<usize> = names.iter().map(|n| self.m.expert_bytes(n)).collect();
        let total: usize = sizes.iter().sum();
        let stride = self.cache.slot_bytes;
        let want: Vec<usize> = (0..self.m.cfg.n_expert)
            .filter(|&e| !self.cache.map.contains_key(&(l, e)) && !self.ram.at.contains_key(&(l, e))).collect();
        let sp = self.sp.as_mut().unwrap();
        let p = l % 3;
        // The buffers were sized for the file-only experts a layer had when the pipeline was set up; the RAM tier changes while a long prompt is read
        // (promotions, evictions), so a layer can have more now.  This half gets a bigger buffer (it is free: what it held was copied out, `stage_pipe_wait`
        // above); cutting `want` short would leave experts in neither place and `stage_copy` would panic.
        if want.len() * stride > sp.bufs[p].len() {
            if let Ok(r) = PinnedRegion::new(self.ctx, want.len() * stride) {
                sp.bufs[p] = r;
            }
        }
        let want = &want[..want.len().min(sp.bufs[p].len() / stride)];
        sp.layer[p] = l;
        sp.at[p] = want.iter().enumerate().map(|(s, &e)| (e, s)).collect();
        if want.is_empty() {
            return;
        }
        let mut jobs: Vec<(&str, usize, &mut [u8])> = Vec::with_capacity(3 * want.len());
        for (&e, slot) in want.iter().zip(sp.bufs[p].as_mut_slice().chunks_mut(stride)) {
            let (gate, more) = slot[..total].split_at_mut(sizes[0]);
            let (up, down) = more.split_at_mut(sizes[1]);
            jobs.push((&names[2], e, down));
            jobs.push((&names[0], e, gate));
            jobs.push((&names[1], e, up));
        }
        if std::env::var("SHADOW_STAGE_NOREAD").is_ok_and(|v| v == "1") {
            // timing experiment only (the output is wrong): the file-only experts are zeros, not read - no file or page-cache traffic beside the copies
            for (_, _, s) in jobs.iter_mut() {
                s.fill(0);
            }
            return;
        }
        // SAFETY: buffer p is not touched again until `stage_pipe_wait` (the next `stage_pipe_read`, the copies out of it in
        // `stage_copy`, `restore_cache`); its previous contents (layer l - 2's) were copied out by a transfer the caller has seen finish.
        unsafe { self.m.experts_submit(&mut jobs) };
        heap_check("experts_submit");
        sp.pending = true;
        if std::env::var("SHADOW_READ_SYNC").is_ok_and(|v| v == "1") {
            // debugging a direct-I/O crash: no read in flight while other work goes on
            self.m.experts_wait();
            sp.pending = false;
        }
    }

    /// Queues the copy of all of layer `l`'s experts that the VRAM cache does not hold into stage half l & 1 (slot = expert number), from the
    /// RAM tier or the read-ahead buffer (waiting for its reads), and records the half's event.
    fn stage_copy(&mut self, l: usize) {
        let t_sc = std::time::Instant::now();
        self.stage_pipe_wait();
        let t_rw = t_sc.elapsed();
        let names = ["gate", "up", "down"].map(|p| format!("blk.{l}.ffn_{p}_exps.weight"));
        let total: usize = names.iter().map(|n| self.m.expert_bytes(n)).sum();
        let stream = self.copy_stream.raw();
        let p = l & 1;
        let rb = l % 3;
        let half = self.sp.as_ref().unwrap().base + (p * self.m.cfg.n_expert * self.cache.slot_bytes) as u64;
        let mut n = 0u64;
        let (mut dsts, mut srcs, mut host_srcs, mut lens): (Vec<u64>, Vec<u64>, Vec<u64>, Vec<usize>) = (vec![], vec![], vec![], vec![]);
        if self.fv_on {
            // the registrations of the copy two layers back are done with (its event was waited for): release them, then register this layer's file-only pieces
            let t_fv = std::time::Instant::now();
            self.sp.as_ref().unwrap().events[p].synchronize().unwrap();
            let t_sync = t_fv.elapsed();
            self.fv_regs[p].release(self.ctx);
            let t_rel = t_fv.elapsed();
            if self.fv.is_none() {
                self.fv = Some(crate::fileview::FileMaps::open(&self.m.shard_paths()).unwrap_or_else(|e| panic!("file views: {e}")));
            }
            let mut pieces: Vec<(usize, u64, usize)> = vec![];
            for e in 0..self.m.cfg.n_expert {
                if self.cache.map.contains_key(&(l, e)) || self.ram.at.contains_key(&(l, e)) {
                    continue;
                }
                for nm in &names {
                    pieces.push(self.m.expert_file_range(nm, e));
                }
            }
            self.fv_regs[p].register(self.ctx, self.fv.as_ref().unwrap(), &pieces).unwrap_or_else(|e| panic!("file views: {e}"));
            if std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "1") {
                eprintln!("FV L{l}: {} pieces, event wait {:.1} ms, release {:.1} ms, map+register {:.1} ms ({} ranges)", pieces.len(), t_sync.as_secs_f64() * 1e3,
                          (t_rel - t_sync).as_secs_f64() * 1e3, (t_fv.elapsed() - t_rel).as_secs_f64() * 1e3, self.fv_regs[p].len());
            }
        }
        for e in 0..self.m.cfg.n_expert {
            if self.cache.map.contains_key(&(l, e)) {
                continue;
            }
            let dst = half + (e * self.cache.slot_bytes) as u64;
            if self.fv_on && !self.ram.at.contains_key(&(l, e)) {
                // a file-only expert: its gate, up and down pieces straight from the registered file view into the slot
                let mut o = 0u64;
                for nm in &names {
                    let (sh, off, len) = self.m.expert_file_range(nm, e);
                    let (dev, host) = self.fv_regs[p].locate(sh, off);
                    dsts.push(dst + o);
                    srcs.push(dev);
                    host_srcs.push(host as u64);
                    lens.push(len);
                    o += len as u64;
                }
                self.ram.drive += 1;
                n += 1;
                continue;
            }
            if let Some(&off) = self.ram.at.get(&(l, e)) {
                let region = self.ram.region.as_ref().unwrap();
                srcs.push(region.dev_ptr + off as u64);
                host_srcs.push(region.as_slice()[off..off + total].as_ptr() as u64);
                self.ram.served += 1;
            } else {
                let sp = self.sp.as_ref().unwrap();
                let slot = *sp.at[rb].get(&e).expect("a staged expert in neither the RAM tier nor the read-ahead");
                debug_assert_eq!(sp.layer[rb], l);
                let at = slot * self.cache.slot_bytes;
                srcs.push(sp.bufs[rb].dev_ptr + at as u64);
                host_srcs.push(sp.bufs[rb].as_slice()[at..at + total].as_ptr() as u64);
                self.ram.drive += 1;
            }
            dsts.push(dst);
            lens.push(total);
            n += 1;
        }
        static TWO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if dsts.len() >= 2 && *TWO.get_or_init(|| std::env::var("SHADOW_COPY_STREAMS").is_ok_and(|v| v == "2")) {
            // half of the layer's experts on each copy stream; stream 1 then waits for stream 2, so the half's event covers both
            let h = dsts.len() / 2;
            let (mut d2, mut s2, mut l2) = (dsts.split_off(h), srcs.split_off(h), lens.split_off(h));
            let h2 = host_srcs[h..].to_vec();
            self.upload_batch(&mut dsts, &mut srcs, &host_srcs[..h], &mut lens, stream);
            self.upload_batch(&mut d2, &mut s2, &h2, &mut l2, self.copy_stream2.raw());
            self.copy_ev2.record(Some(&self.copy_stream2)).unwrap();
            self.ctx.stream_wait_event(Some(&self.copy_stream), &self.copy_ev2).unwrap();
        } else {
            let t_up = std::time::Instant::now();
            self.upload_batch(&mut dsts, &mut srcs, &host_srcs, &mut lens, stream);
            if std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "3") {
                eprintln!("STAGECOPY L{l}: read wait {:.1} ms, build {:.1} ms, upload_batch {:.1} ms ({} pieces, {} file experts of {} bytes)", t_rw.as_secs_f64() * 1e3,
                          (t_up - t_sc).as_secs_f64() * 1e3 - t_rw.as_secs_f64() * 1e3, t_up.elapsed().as_secs_f64() * 1e3, dsts.len(),
                          self.sp.as_ref().unwrap().at[rb].len(), total);
            }
        }
        self.expert_uploads += 3 * n as usize;
        let sp = self.sp.as_mut().unwrap();
        sp.events[p].record(Some(&self.copy_stream)).unwrap();
        sp.buf_done[rb].record(Some(&self.copy_stream)).unwrap();
        sp.ready[p] = l;
        heap_check("stage_copy");
    }

    /// Layer `l`'s routed experts for a long prompt's m tokens, all in one group: the resident ones from the cache, the rest from stage
    /// half l & 1, which the previous layer's `moe_staged` has been copying in while it ran (the first layer of a pass fills it here);
    /// then layer l + 1's copy is queued behind the resident experts' launches and layer l + 2's file reads started.
    fn moe_staged(&mut self, l: usize, m: usize, tops: Option<&[Vec<(usize, f64)>]>, infos: &[TensorInfo], sizes: &[usize], _names: &[String; 3]) {
        let c = &self.m.cfg;
        let (h, k, ne) = (c.hidden, c.n_expert_used, c.n_expert);
        let p = l & 1;
        let stage_time = std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "1");
        heap_check("moe_staged start");
        let t_start = std::time::Instant::now();
        // the copy of layer l + 1 is queued behind layer l's experts (it overlaps them); `SHADOW_STAGE_LEAD=2`: layer l + 2's is queued there instead and waits
        // for layer l's combine, so each copy overlaps a whole layer ahead of its use - a tie at 4,096 rows (docs/STATUS.md 2026-11-04), so opt-in
        let lead2 = std::env::var("SHADOW_STAGE_LEAD").is_ok_and(|v| v == "2");
        if self.sp.as_ref().unwrap().ready[p] != l {
            // the first layer of a pass: read and copy it here, and start layer l + 1's reads
            self.stage_pipe_read(l);
            self.stage_copy(l);
            self.stage_pipe_read(l + 1);
            if lead2 && l + 1 < c.n_layer {
                // two layers in flight: layer l + 1's copy too (half (l + 1) & 1 is free at the start of a pass)
                self.stage_copy(l + 1);
            }
        }
        // the next layer's copy and the read-ahead after it are queued at the start of the layer (its first sub-chunk), not behind its last experts: the half the copy fills
        // was read by layer l - 1, whose combine the copy stream waits for, so the copy runs behind the whole layer's mixers and experts
        // (`SHADOW_STAGE_EARLY=0`: after the layer, as before)
        let early = !lead2 && !std::env::var("SHADOW_STAGE_EARLY").is_ok_and(|v| v == "0");
        if early && self.sub_first && l + 1 < c.n_layer {
            let t0 = std::time::Instant::now();
            self.ctx.stream_wait_event(Some(&self.copy_stream), &self.sp.as_ref().unwrap().done[(l + 1) & 1]).unwrap();
            self.stage_copy(l + 1);
            let t1 = t0.elapsed();
            self.sp.as_ref().unwrap().buf_done[(l + 2) % 3].synchronize().unwrap();
            let t2 = t0.elapsed();
            self.stage_pipe_read(l + 2);
            if std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "3") {
                eprintln!("HOSTWAIT L{l}: stage_copy({}) incl. read wait {:.1} ms, buf_done wait {:.1} ms, read submit {:.1} ms", l + 1, t1.as_secs_f64() * 1e3,
                          (t2 - t1).as_secs_f64() * 1e3, (t0.elapsed() - t2).as_secs_f64() * 1e3);
            }
        }
        let (plan, n_set) = if let Some(tops) = tops {
            let mut seen = vec![false; ne];
            let mut set: Vec<usize> = Vec::new();
            for top in tops {
                for &(e, _) in top {
                    if !seen[e] {
                        seen[e] = true;
                        set.push(e);
                    }
                }
            }
            let arena = self.cache.arena.as_ref().unwrap().ptr;
            let half = self.sp.as_ref().unwrap().base + (p * ne * self.cache.slot_bytes) as u64;
            for &e in &set {
                if self.cache.touch((l, e)).is_some() {
                    self.cache.hits += 1;
                } else {
                    self.cache.misses += 1;
                }
            }
            let part = [0u64, sizes[0] as u64, (sizes[0] + sizes[1]) as u64];
            let addr = |e: usize| match self.cache.map.get(&(l, e)) {
                Some(&s) => arena + self.cache.slot_offset[s],
                None => half + (e * self.cache.slot_bytes) as u64,
            };
            let copied = |e: usize| !self.cache.map.contains_key(&(l, e));
            let chunk_n = if self.mma_experts_ok(infos) { mma_group() } else { 8 };
            let plan = grouped_plan(tops, 0, m, &set, k, self.rows * k, addr, copied, part, chunk_n);
            assert!(plan.bytes.len() <= self.bb.gtab.bytes, "grouped experts: {} bytes of tables", plan.bytes.len());
            self.upload_to(self.bb.gtab.ptr, &plan.bytes);
            (plan, set.len())
        } else {
            // routed and planned on the device: nothing the host decided depends on the tokens' routing
            (self.dev_plan(l, m, sizes), 0)
        };
        let t_plan = t_start.elapsed();
        self.routed_grouped(&plan, 0, 0, m, infos);
        self.lt_point("staged: resident experts done");
        self.ctx.stream_wait_event(None, &self.sp.as_ref().unwrap().events[p]).unwrap();
        self.lt_point("staged: GPU stalled on the copy");
        self.routed_grouped(&plan, 1, 0, m, infos);
        self.lt_point("staged: copied experts done");
        let t_launch = t_start.elapsed();
        heap_check("moe_staged experts launched");
        let bb = &self.bb;
        self.k2("moe_combine_qwen_rows", (h as u32).div_ceil(256), m as u32, 256, &[
            Arg::Ptr(row(&bb.y, h, 0)), Arg::Ptr(row(&bb.eo, k * h, 0)), Arg::Ptr(bb.gtab.ptr + plan.weights),
            Arg::Ptr(row(&bb.so, h, 0)), Arg::Ptr(row(&bb.sgl, 1, 0)), Arg::Ptr(bb.gtab.ptr + plan.counts),
            Arg::U32(h as u32), Arg::U32(k as u32)]);
        if !self.sub_last {
            // not the layer's last sub-chunk: the copy pipeline moves on once, after the last one
            return;
        }
        if !lead2 {
            self.sp.as_ref().unwrap().done[p].record(None).unwrap();
        }
        if early {
            return;
        }
        // now the next layer's copy and the one after's reads, behind this layer's launches: the GPU runs this layer while the host queues them
        if lead2 {
            let n_layer = self.m.cfg.n_layer;
            self.sp.as_ref().unwrap().done[p].record(None).unwrap();
            if l + 2 < n_layer {
                if self.sp.as_ref().unwrap().layer[(l + 2) % 3] != l + 2 {
                    // the buffer held layer l - 1's file-only experts: their copy is done before the buffer is read into again
                    self.sp.as_ref().unwrap().buf_done[(l + 2) % 3].synchronize().unwrap();
                    self.stage_pipe_read(l + 2);
                }
                // the half it fills is the one this layer's experts read: wait for their combine
                self.ctx.stream_wait_event(Some(&self.copy_stream), &self.sp.as_ref().unwrap().done[p]).unwrap();
                self.stage_copy(l + 2);
            }
            if l + 3 < n_layer {
                // the buffer layer l + 3 reads into held layer l's file-only experts: their copy has finished
                self.sp.as_ref().unwrap().buf_done[(l + 3) % 3].synchronize().unwrap();
                self.stage_pipe_read(l + 3);
            }
        } else if l + 1 < self.m.cfg.n_layer {
            // layer l + 1's copy fills the half layer l - 1's experts read: with the host running ahead of the GPU (the device planner: no wait for the router's logits) they may
            // not be done, so the copy stream waits for the event recorded after layer l - 1's combine
            self.ctx.stream_wait_event(Some(&self.copy_stream), &self.sp.as_ref().unwrap().done[(l + 1) & 1]).unwrap();
            self.stage_copy(l + 1);
            let t_copyq = t_start.elapsed();
            // the buffer layer l + 2 reads into held layer l - 1's file-only experts: their copy was queued two layers ago, and is done (or soon)
            self.sp.as_ref().unwrap().buf_done[(l + 2) % 3].synchronize().unwrap();
            let t_sync = t_start.elapsed();
            self.stage_pipe_read(l + 2);
            if stage_time {
                eprintln!("STAGE L{l} m={m} experts={} plan {:.1} launch {:.1} copy_queued {:.1} event_sync {:.1} read_started {:.1} ms (file waits so far {:.2} s)",
                          n_set, t_plan.as_secs_f64() * 1e3, t_launch.as_secs_f64() * 1e3, t_copyq.as_secs_f64() * 1e3, t_sync.as_secs_f64() * 1e3,
                          t_start.elapsed().as_secs_f64() * 1e3, self.waits[2]);
            }
        }
    }

    /// The per-token buffers for `BIG_BATCH` tokens, carved from the lent cache after the stage halves, swapped in for the ones for
    /// `MAX_BATCH` (the buffers that do not grow with the batch are shared); false if the lent room is too small.
    fn big_batch_on(&mut self) -> bool {
        let (Some(lent), true) = (self.lent.as_ref(), self.sp.is_some()) else { return false };
        let c = &self.m.cfg;
        let halves = (2 * c.n_expert * self.cache.slot_bytes) as u64;
        // where the buffers can go: the KV window's lent tail, and the lent part of the arena past the stage halves (both, when the lend took both)
        let ranges: Vec<(u64, u64)> = match self.kv_extra.as_ref() {
            Some(x) => {
                let mut v = vec![(x.ptr + if self.kv_halves { halves } else { 0 }, x.ptr + x.bytes as u64)];
                let (a0, a1) = (self.sp.as_ref().unwrap().base + halves, lent.ptr + (lent.n * self.cache.slot_bytes) as u64);
                if !self.kv_halves && a1 > a0 + (1 << 20) {
                    // the arena's part first: the first buffers carved are the biggest
                    v.insert(0, (a0, a1));
                }
                v
            }
            None => vec![(self.sp.as_ref().unwrap().base + halves, lent.ptr + (lent.n * self.cache.slot_bytes) as u64)],
        };
        // As many rows as the lent room holds, up to BIG_BATCH: a longer `--ctx` leaves the expert cache (and so the room) smaller, and a prompt that
        // read 2,048 tokens at a time must not drop to 256 for want of a few MiB; each pass copies a layer's experts whatever its token count.
        let env_cap = big_rows_cap();
        let cap = if self.kv_halves { env_cap.min(self.kv_rows) } else { env_cap };
        let ch = c.streams * c.hidden;
        let ctx = self.ctx;
        let dummy = |ctx: &'c Context| unsafe { DeviceBuffer::view(ctx, 0, 0) };
        // the carving is first-fit over the ranges, so the rows are found by carving: the most rows (a multiple of 64) whose buffers all fit
        // `total` rows a pass: the temporaries are for `rows`, the residual streams for `total` (when that is more)
        let fits = |rows: usize, total: usize| -> bool {
            let mut cur: Vec<u64> = ranges.iter().map(|r| r.0).collect();
            let mut take = |floats: usize| -> Result<DeviceBuffer<'c>, String> {
                let bytes = (floats.max(1) * 4).next_multiple_of(256) as u64;
                let k = ranges.iter().zip(cur.iter()).position(|(r, &a)| a + bytes <= r.1).ok_or_else(|| "no room".to_string())?;
                cur[k] += bytes;
                Ok(unsafe { DeviceBuffer::view(ctx, 0, 0) })
            };
            let fixed = FixedBufs { att: dummy(ctx), sel_iqn: dummy(ctx), sel_scores: dummy(ctx), sel_toks: dummy(ctx), logits: dummy(ctx) };
            if make_bb(c, rows, &mut take, fixed).is_err() {
                return false;
            }
            total <= rows || (0..total.div_ceil(rows)).all(|_| take(rows * ch).is_ok())
        };
        let sub = sub_rows_cap().min(cap);
        let mut rows = sub / 64 * 64;
        while rows >= 2 * MAX_BATCH && !fits(rows, rows) {
            rows -= 64;
        }
        if rows < 2 * MAX_BATCH {
            return false;
        }
        // the most rows a pass can have with those temporaries (a multiple of 64, at most the cap)
        let mut total = cap / 64 * 64;
        while total > rows && !fits(rows, total) {
            total -= 64;
        }
        let total = total.max(rows);
        if std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "1") {
            eprintln!("BIGBATCH lent {} slots ({:.2} GiB), halves {:.2} GiB, ranges {} ({:.2} GiB), rows {rows} of cap {cap}, pass {total} rows",
                      lent.n, (lent.n * self.cache.slot_bytes) as f64 / 1073741824.0, halves as f64 / 1073741824.0,
                      ranges.len(), ranges.iter().map(|r| r.1 - r.0).sum::<u64>() as f64 / 1073741824.0);
            for r in &ranges {
                eprintln!("BIGBATCH range {:.3} GiB", (r.1 - r.0) as f64 / 1073741824.0);
            }
        }
        let mut cur: Vec<u64> = ranges.iter().map(|r| r.0).collect();
        let mut take = |floats: usize| -> Result<DeviceBuffer<'c>, String> {
            let bytes = floats.max(1) * 4;
            let aligned = bytes.next_multiple_of(256) as u64;
            let k = ranges.iter().zip(cur.iter()).position(|(r, &a)| a + aligned <= r.1).ok_or_else(|| "no room".to_string())?;
            let p = cur[k];
            cur[k] += aligned;
            // SAFETY: inside the lent cache or KV tail, past the stage halves: nothing else uses it until `big_batch_off`, which swaps these away.
            Ok(unsafe { DeviceBuffer::view(ctx, p, bytes) })
        };
        let fixed = FixedBufs {
            att: std::mem::replace(&mut self.bb.att, dummy(ctx)), sel_iqn: std::mem::replace(&mut self.bb.sel_iqn, dummy(ctx)),
            sel_scores: std::mem::replace(&mut self.bb.sel_scores, dummy(ctx)), sel_toks: std::mem::replace(&mut self.bb.sel_toks, dummy(ctx)),
            logits: std::mem::replace(&mut self.bb.logits, dummy(ctx)),
        };
        let built = make_bb(c, rows, &mut take, fixed);
        let rparts: Vec<DeviceBuffer<'c>> = if total > rows { (0..total.div_ceil(rows)).map_while(|_| take(rows * ch).ok()).collect() } else { Vec::new() };
        match built {
            Ok(big) => {
                self.small_bb = Some(std::mem::replace(&mut self.bb, big));
                self.rows = rows;
                self.batch_max = if !rparts.is_empty() && rparts.len() == total.div_ceil(rows) { total } else { rows };
                self.rparts = rparts;
                true
            }
            Err(_) => false, // cannot happen: `take` does not fail; the fixed buffers are dropped views
        }
    }

    /// Back to the buffers for `MAX_BATCH` tokens (the shared ones move back).
    fn big_batch_off(&mut self) {
        let Some(mut small) = self.small_bb.take() else { return };
        for (a, b) in [(&mut small.att, &mut self.bb.att), (&mut small.sel_iqn, &mut self.bb.sel_iqn), (&mut small.sel_scores, &mut self.bb.sel_scores),
                       (&mut small.sel_toks, &mut self.bb.sel_toks), (&mut small.logits, &mut self.bb.logits)] {
            std::mem::swap(a, b);
        }
        self.bb = small;
        self.rparts = Vec::new();
        self.rows = MAX_BATCH;
        self.batch_max = MAX_BATCH;
    }

    /// Gives the lent cache back: the layers' ranges as they were, and the evicted entries read again from the RAM tier
    /// (the ones it holds; the rest warm up by demand).
    pub fn restore_cache(&mut self) {
        let Some(lent) = self.lent.take() else { return };
        self.stage_pipe_wait();
        self.ctx.synchronize().unwrap();
        for r in self.fv_regs.iter_mut() {
            r.release(self.ctx);
        }
        self.big_batch_off();
        self.unlend_kv();
        self.cache.evict_range = lent.evict_saved;
        let arena = self.cache.arena.as_ref().unwrap().ptr;
        let stream = self.copy_stream.raw();
        // the evicted experts the RAM tier does not hold come back from the file, through the pipeline's pinned read-ahead buffer
        if let Some(mut sp) = self.sp.take() {
            let cold: Vec<(usize, usize, usize)> = lent.saved.iter().copied()
                .filter(|&(_, l, e)| !self.cache.map.contains_key(&(l, e)) && !self.ram.at.contains_key(&(l, e))).collect();
            let stride = self.cache.slot_bytes;
            for batch in cold.chunks(sp.slots.max(1)) {
                let mut sizes_of: Vec<[usize; 3]> = vec![];
                let mut names_of: Vec<[String; 3]> = vec![];
                for &(_, l, _) in batch {
                    let names = ["gate", "up", "down"].map(|p| format!("blk.{l}.ffn_{p}_exps.weight"));
                    sizes_of.push([self.m.expert_bytes(&names[0]), self.m.expert_bytes(&names[1]), self.m.expert_bytes(&names[2])]);
                    names_of.push(names);
                }
                {
                    let mut jobs: Vec<(&str, usize, &mut [u8])> = Vec::with_capacity(3 * batch.len());
                    for (((&(_, _, e), names), sz), slot) in batch.iter().zip(&names_of).zip(&sizes_of).zip(sp.bufs[0].as_mut_slice().chunks_mut(stride)) {
                        let (gate, more) = slot[..sz[0] + sz[1] + sz[2]].split_at_mut(sz[0]);
                        let (up, down) = more.split_at_mut(sz[1]);
                        jobs.push((&names[2], e, down));
                        jobs.push((&names[0], e, gate));
                        jobs.push((&names[1], e, up));
                    }
                    self.m.experts_into_many(&mut jobs);
                }
                for (i, &(s, l, e)) in batch.iter().enumerate() {
                    let total: usize = sizes_of[i].iter().sum();
                    let src = &sp.bufs[0].as_slice()[i * stride..i * stride + total];
                    self.upload_pinned_on(arena + self.cache.slot_offset[s], src, stream);
                    self.cache.owner[s] = Some((l, e));
                    self.cache.map.insert((l, e), s);
                    self.cache.stamp[s] = 0;
                }
                self.copy_stream.synchronize().unwrap();
            }
        }
        for (s, l, e) in lent.saved {
            // the prompt may have loaded this expert again into another slot: one owner only
            if self.cache.map.contains_key(&(l, e)) {
                continue;
            }
            let Some(&off) = self.ram.at.get(&(l, e)) else { continue };
            let Some(region) = self.ram.region.as_ref() else { continue };
            let total: usize = ["gate", "up", "down"].iter().map(|p| self.m.expert_bytes(&format!("blk.{l}.ffn_{p}_exps.weight"))).sum();
            if let Some(&r) = self.ram.ring_by_off.get(&off) {
                self.ram.ring_ref[r] = true;
            }
            let src = &region.as_slice()[off..off + total];
            self.upload_pinned_on(arena + self.cache.slot_offset[s], src, stream);
            self.cache.owner[s] = Some((l, e));
            self.cache.map.insert((l, e), s);
            self.cache.stamp[s] = 0;
        }
        self.copy_stream.synchronize().unwrap();
    }

    /// The router's logits (and the shared expert's gate logit) for layer l into rl (and sgl).
    fn moe_router(&self, l: usize) {
        let p = format!("blk.{l}");
        if self.m.cfg.arch != "flint" {
            self.mv_pair(&format!("{p}.ffn_gate_inp.weight"), self.b.rl.ptr,
                         &format!("{p}.ffn_gate_inp_shexp.weight"), self.b.sgl.ptr, self.b.u.ptr);
        } else {
            self.mv(&format!("{p}.ffn_gate_inp.weight"), self.b.u.ptr, self.b.rl.ptr);
        }
    }

    /// The MoE of layer l on u into y (`router_done`: its router already ran, in a graph).
    fn moe(&mut self, l: usize, router_done: bool) {
        self.ram.claimed.clear(); // the previous layer's copies were waited for
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let h = c.hidden;
        let ff = c.expert_ff;
        let sff = c.shared_ff;
        let is_flint = c.arch == "flint";
        let sh_g_name = format!("{p}.ffn_gate_shexp.weight");
        let sh_u_name = format!("{p}.ffn_up_shexp.weight");
        self.mark("mixers, norms, hc glue, ple");
        if !router_done {
            self.moe_router(l);
        }
        let pf_next = !is_flint && self.cache.slots > 0 && l + 1 < c.n_layer && c.is_moe(l + 1) && self.prefetch.is_some();
        if pf_next {
            let buf = self.prefetch.as_ref().unwrap().buf.ptr;
            self.mv(&format!("blk.{}.ffn_gate_inp.weight", l + 1), self.b.u.ptr, buf);
        }
        let mut srcs = vec![(self.b.rl.ptr, c.n_expert)];
        if pf_next {
            srcs.push((self.prefetch.as_ref().unwrap().buf.ptr, c.n_expert));
        }
        let cpu_on = !is_flint && self.cpu.is_some() && self.cache.slots > 0;
        if cpu_on {
            srcs.push((self.b.u.ptr, h)); // the CPU experts' input, in the same read
        }
        let t_wait = std::time::Instant::now();
        let mut got = self.download_mapped(&srcs).into_iter();
        self.waits[0] += t_wait.elapsed().as_secs_f64();
        let logits = got.next().unwrap();
        let next_logits = if pf_next { got.next() } else { None };
        let u_host = if cpu_on { got.next() } else { None };
        // Launch Qwen's shared expert on stream 0 immediately so the GPU SMs compute it while the CPU runs `route`
        // and `copy_stream` DMAs any missing routed experts.
        let sh_launched_early = !is_flint && self.prof.is_none();
        if sh_launched_early {
            let k = c.n_expert_used;
            let sh_eg = self.b.eg.ptr + (2 * k * ff * 4) as u64;
            let sh_eu = self.b.eu.ptr + (k * ff * 4) as u64;
            let sh_act = self.b.act.ptr + (k * ff * 4) as u64;
            let sh_out = self.b.eo.ptr + (k * h * 4) as u64;
            self.mv_pair(&sh_g_name, sh_eg, &sh_u_name, sh_eu, self.b.u.ptr);
            self.k1("swiglu", sff, &[Arg::Ptr(sh_eg), Arg::Ptr(sh_eu), Arg::Ptr(sh_act), Arg::U32(sff as u32)]);
            self.mv(&format!("{p}.ffn_down_shexp.weight"), sh_act, sh_out);
        }
        let top = if is_flint {
            route_sigmoid(&logits, c.n_expert_used)
        } else if self.router_bias != 0.0 {
            route_biased(&logits, &|e| if self.cache.map.contains_key(&(l, e)) { self.router_bias } else { 0.0 }, c.n_expert_used)
        } else {
            route(&logits, c.n_expert_used)
        };
        self.mark("router (logits down, top-10)");
        for &(e, _) in &top {
            self.routing.add(l, e);
        }
        if let Some((_, log)) = self.route_log.as_mut() {
            log.push(l as u16);
            log.extend(top.iter().map(|&(e, _)| e as u16));
        }
        self.rec_host(format!("layer{l}.route_idx"), &top.iter().map(|&(e, _)| e as f32).collect::<Vec<_>>());
        self.rec_host(format!("layer{l}.route_w"), &top.iter().map(|&(_, w)| w as f32).collect::<Vec<_>>());
        let names = [format!("{p}.ffn_gate_exps.weight"), format!("{p}.ffn_up_exps.weight"), format!("{p}.ffn_down_exps.weight")];
        let infos: Vec<TensorInfo> = names.iter().map(|n| self.expert_info(n)).collect();
        let sizes: Vec<usize> = names.iter().map(|n| {
            let t = self.m.g.tensor(n).unwrap();
            (t.nbytes() / t.dims[2]) as usize
        }).collect();
        if !is_flint {
            if let Some(pred) = self.pregate.as_ref().and_then(|pg| pg.pred[l].clone()) {
                let real: Vec<usize> = top.iter().map(|&(e, _)| e).collect();
                let misses: Vec<usize> = real.iter().copied().filter(|e| !self.cache.map.contains_key(&(l, *e))).collect();
                if let Some(pg) = self.pregate.as_mut() {
                    let k = &mut pg.counts[l];
                    k[0] += real.len() as u64;
                    k[1] += real.iter().filter(|e| pred[..10].contains(e)).count() as u64;
                    k[2] += real.iter().filter(|e| pred.contains(e)).count() as u64;
                    k[3] += misses.len() as u64;
                    k[4] += misses.iter().filter(|e| pred.contains(e)).count() as u64;
                }
            }
            if let Some(buf) = self.pregate.as_ref().map(|pg| pg.buf.ptr).filter(|_| l + 1 < c.n_layer && c.is_moe(l + 1)) {
                self.mv(&format!("blk.{}.ffn_gate_inp.weight", l + 1), self.b.u.ptr, buf);
                let mut pb = vec![0u8; c.n_expert * 4];
                self.download(buf, &mut pb);
                let pred: Vec<usize> = route(&floats(&pb), 16).into_iter().map(|(e, _)| e).collect();
                if let Some(pg) = self.pregate.as_mut() {
                    pg.pred[l + 1] = Some(pred);
                }
            }
        }
        // --tail-skip: the low-ranked experts that are not in VRAM drop out (zero contribution, weights unchanged)
        let mut top = top;
        if self.tail_skip > 0 && !is_flint && self.cache.slots > 0 {
            let before = top.len();
            let mut i = 0;
            top.retain(|&(e, _)| {
                i += 1;
                i <= self.tail_skip || self.cache.map.contains_key(&(l, e))
            });
            self.tail_skipped += (before - top.len()) as u64;
        }
        // Hybrid MoE: up to `cpu_experts` of the lowest-weight misses that the RAM tier holds are computed on the CPU,
        // from the pinned bytes, while the GPU copies and runs the rest.  `top` becomes [GPU experts.., CPU experts..].
        let mut cpu_offs: Vec<usize> = vec![];
        if let Some(&(_, max)) = self.cpu.as_ref().filter(|_| !is_flint && self.cache.slots > 0) {
            let mut cpu_set: Vec<(usize, f64, usize)> = vec![];
            for &(e, w) in top.iter().rev() {
                if cpu_set.len() == max {
                    break;
                }
                if !self.cache.map.contains_key(&(l, e)) {
                    if let Some(&off) = self.ram.at.get(&(l, e)) {
                        cpu_set.push((e, w, off));
                    }
                }
            }
            if !cpu_set.is_empty() {
                let ub = u_host.clone().expect("the MoE input comes down with the router logits");
                let base = self.ram.region.as_ref().unwrap().as_slice().as_ptr();
                let refs: Vec<shadow_model::cpu_expert::ExpertRef> = cpu_set.iter().map(|&(_, _, off)| {
                    // SAFETY: the RAM tier holds this expert's [gate | up | down] at off, inside the region.
                    unsafe {
                        shadow_model::cpu_expert::ExpertRef { gate: base.add(off), up: base.add(off + sizes[0]),
                                                              down: base.add(off + sizes[0] + sizes[1]) }
                    }
                }).collect();
                for &(_, _, off) in &cpu_set {
                    if let Some(&r) = self.ram.ring_by_off.get(&off) {
                        self.ram.ring_ref[r] = true; // keep the Clock hand off it this layer
                        self.ram.claimed.insert(off); // and this layer's file reads off its slot
                    }
                }
                let shape = shadow_model::cpu_expert::ExpertShape {
                    hidden: h, ff, gate_ty: infos[0].ty.name(), up_ty: infos[1].ty.name(), down_ty: infos[2].ty.name(),
                };
                let pool = &mut self.cpu.as_mut().unwrap().0;
                // SAFETY: the region lives as long as the session, and nothing rewrites these experts' bytes before
                // `wait` below: read_drive waits first if it is about to reuse one of their ring slots.
                unsafe { pool.submit(&shape, refs, &ub) };
                self.cpu_served += cpu_set.len() as u64;
                top.retain(|&(e, _)| !cpu_set.iter().any(|c| c.0 == e));
                top.extend(cpu_set.iter().map(|&(e, w, _)| (e, w)));
                cpu_offs = cpu_set.iter().map(|c| c.2).collect();
            }
        }
        if !is_flint && sh_launched_early && self.cache.slots > 0 && self.prefetch.is_none()
            && infos[0].ty == infos[1].ty && infos[0].dims == infos[1].dims && self.overlap
        {
            self.routed_overlapped(l, &top, cpu_offs.len(), &names, &infos, &sizes);
            return;
        }
        let ng = top.len() - cpu_offs.len();
        let mut cpu_out: Option<Vec<f32>> = None;
        // where each expert's three slices are on the device this time
        let mut at: Vec<[u64; 3]> = Vec::with_capacity(top.len());
        let mut layer_misses = 0usize;
        if self.cache.slots > 0 {
            self.cache.clock += 1;
            for &(e, _) in &top[..ng] {
                self.cache.touch((l, e)); // protect this token's resident experts from its own misses
            }
            for (i, &(e, _)) in top[..ng].iter().enumerate() {
                let slot = match self.cache.touch((l, e)) {
                    Some(s) => {
                        self.cache.hits += 1;
                        if let Some(pf) = self.prefetch.as_mut() {
                            pf.used += pf.ahead.remove(&(l, e)) as u64;
                        }
                        s
                    }
                    None => {
                        self.cache.misses += 1;
                        layer_misses += 1;
                        let s = self.cache.victim(l);
                        let base = self.cache.arena.as_ref().unwrap().ptr + self.cache.slot_offset[s];
                        let dst = [base, base + sizes[0] as u64, base + (sizes[0] + sizes[1]) as u64];
                        self.fetch_expert(l, e, i, &names, &sizes, dst);
                        self.cache.owner[s] = Some((l, e));
                        self.cache.map.insert((l, e), s);
                        self.cache.stamp[s] = self.cache.clock;
                        s
                    }
                };
                let base = self.cache.arena.as_ref().unwrap().ptr + self.cache.slot_offset[slot];
                at.push([base, base + sizes[0] as u64, base + (sizes[0] + sizes[1]) as u64]);
            }
        }
        if self.cache.slots == 0 {
            // no cache: each routed expert into its own row of the scratch buffers, every time
            let stride = (self.b.wg.bytes / c.n_expert_used) as u64;
            for (i, &(e, _)) in top.iter().enumerate() {
                layer_misses += 1;
                let dst = [self.b.wg.ptr + i as u64 * stride, self.b.wu.ptr + i as u64 * stride,
                           self.b.wd.ptr + i as u64 * stride];
                self.fetch_expert(l, e, i, &names, &sizes, dst);
                at.push(dst);
            }
        }
        if self.pending.iter().any(|p| p.ring && cpu_offs.contains(&p.off)) {
            cpu_out = Some(self.cpu.as_mut().unwrap().0.wait());
        }
        self.read_drive(&names, &sizes);
        if layer_misses > 0 {
            let t_wait = std::time::Instant::now();
            self.copy_stream.synchronize().unwrap();
            self.waits[1] += t_wait.elapsed().as_secs_f64();
        } else if self.prefetch.as_ref().is_some_and(|p| p.wait) {
            // this layer's experts may have been copied ahead: the GPU waits for those copies, the host does not
            self.ctx.stream_wait_event(None, &self.copy_event).unwrap();
        }
        if let Some(pf) = self.prefetch.as_mut() {
            pf.wait = false;
            pf.ahead.retain(|&(pl, _)| pl > l);
        }
        if let Some(nl) = next_logits {
            self.prefetch_next(l + 1, &nl);
        }
        // one table per layer - every expert's gate, up and down address, then the weights (and scales for NVFP4) -
        // and one launch per projection for all of them (with gate+up fused, and Flint's shared expert gate+up fused in too)
        let n = top.len();
        let fuse_shexp = is_flint && sff == 2 * ff && infos[0].ty == infos[1].ty && infos[0].dims == infos[1].dims;
        let n_gu = if fuse_shexp { ng + 2 } else { ng };
        let mut tab = Vec::with_capacity(n_gu * 40);
        for part in 0..2 {
            for a in &at {
                tab.extend_from_slice(&a[part].to_le_bytes());
            }
            if fuse_shexp {
                let base = self.w.get(if part == 0 { &sh_g_name } else { &sh_u_name }).unwrap().ptr;
                tab.extend_from_slice(&base.to_le_bytes());
                tab.extend_from_slice(&(base + sizes[part] as u64).to_le_bytes());
            }
        }
        for a in &at {
            tab.extend_from_slice(&a[2].to_le_bytes());
        }
        for &(_, wgt) in &top {
            tab.extend_from_slice(&(wgt as f32).to_le_bytes());
        }
        if is_flint {
            for (part, name) in names.iter().enumerate().take(2) {
                for &(e, _) in &top {
                    let s = self.m.expert_scale(name, e);
                    tab.extend_from_slice(&s.to_le_bytes());
                }
                if fuse_shexp {
                    let s = self.m.scale(if part == 0 { &sh_g_name } else { &sh_u_name });
                    tab.extend_from_slice(&s.to_le_bytes());
                    tab.extend_from_slice(&s.to_le_bytes());
                }
            }
            for &(e, _) in &top {
                let s = self.m.expert_scale(&names[2], e);
                tab.extend_from_slice(&s.to_le_bytes());
            }
        }
        self.upload_to(self.b.tab.ptr, &tab);
        let b = &self.b;
        let batch = |t: &TensorInfo, tab_off: u64, count: usize, x: u64, xs: usize, y: u64, ys: usize| {
            self.mv_batch(t, self.b.tab.ptr + tab_off, count, x, xs, y, ys)
        };
        let down_tab_off = 16 * n_gu as u64;
        let w_off = down_tab_off + 8 * ng as u64;
        let gs_off = w_off + 4 * n as u64;
        let us_off = gs_off + 4 * n_gu as u64;
        let ds_off = us_off + 4 * n_gu as u64;
        if ng > 0 {
            let eu_ptr = if infos[0].ty == infos[1].ty && infos[0].dims == infos[1].dims {
                // gate and up share the same input vector u and have identical type/dimensions: launch all 2 * n_gu at once
                batch(&infos[0], 0, 2 * n_gu, b.u.ptr, 0, b.eg.ptr, ff);
                b.eg.ptr + (n_gu * ff * 4) as u64
            } else {
                batch(&infos[0], 0, n_gu, b.u.ptr, 0, b.eg.ptr, ff);
                batch(&infos[1], 8 * n_gu as u64, n_gu, b.u.ptr, 0, b.eu.ptr, ff);
                b.eu.ptr
            };
            if is_flint {
                self.k1("swiglu_scaled", n_gu * ff, &[
                    Arg::Ptr(b.eg.ptr), Arg::Ptr(eu_ptr), Arg::Ptr(b.tab.ptr + gs_off),
                    Arg::Ptr(b.tab.ptr + us_off), Arg::Ptr(b.act.ptr), Arg::U32(ff as u32), Arg::U32(n_gu as u32)]);
            } else {
                self.k1("swiglu", ng * ff, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(eu_ptr), Arg::Ptr(b.act.ptr), Arg::U32((ng * ff) as u32)]);
            }
            batch(&infos[2], down_tab_off, ng, b.act.ptr, ff, b.eo.ptr, h);
        }
        if ng < n {
            // the CPU's experts, unweighted, into eo's rows after the GPU's
            let out = cpu_out.take().unwrap_or_else(|| self.cpu.as_mut().unwrap().0.wait());
            self.upload_to(self.b.eo.ptr + (ng * h * 4) as u64, &f32s(&out));
        }
        self.mark("routed expert matvecs");
        let b = &self.b;
        let sh_out_ptr = b.eo.ptr + ((if sh_launched_early { c.n_expert_used } else { n }) * h * 4) as u64;
        if !sh_launched_early {
            let sh_act_ptr = if fuse_shexp {
                b.act.ptr + (n * ff * 4) as u64
            } else {
                self.mv_pair(&sh_g_name, b.eg.ptr, &sh_u_name, b.eu.ptr, b.u.ptr);
                self.k1("swiglu", sff, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(b.eu.ptr), Arg::Ptr(b.act.ptr), Arg::U32(sff as u32)]);
                b.act.ptr
            };
            self.mv(&format!("{p}.ffn_down_shexp.weight"), sh_act_ptr, sh_out_ptr);
        }
        if is_flint {
            self.k1("moe_combine_flint", h, &[
                Arg::Ptr(b.y.ptr), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.tab.ptr + w_off),
                Arg::Ptr(b.tab.ptr + ds_off), Arg::Ptr(sh_out_ptr), Arg::U32(h as u32), Arg::U32(n as u32)]);
        } else {
            self.k1("moe_combine_qwen", h, &[
                Arg::Ptr(b.y.ptr), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.tab.ptr + w_off),
                Arg::Ptr(sh_out_ptr), Arg::Ptr(b.sgl.ptr), Arg::U32(h as u32), Arg::U32(n as u32)]);
        }
        self.mark("shared expert + combine");
    }

    // -------------------------------------------------------------------------------------------- the batched step
    /// Up to `MAX_BATCH` tokens at a time through the model layer-major, and each one's logits: the same numbers, bit
    /// for bit, as `step` on each in turn (tests/reference.rs).  Per layer the projections the tokens share run once for
    /// all of them (`mv_cols`), the recurrences, the KV writes and attention run per token in order, and the MoE
    /// fetches the union of their experts once.  What it does not cover (Flint, tracing, no expert cache, the hybrid
    /// and prefetch experiments) steps one token at a time.
    pub fn step_batch(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.step_batch_opt(tokens, true)
    }

    /// `step_batch` for a caller that uses only the last row's logits (the end of a prompt): the output projection of the rows before it is skipped (they come back empty;
    /// `SHADOW_LOGITS_LAST=0` computes them).  The 1,024 rows of a last sub-chunk were 130 launches of 2.7 ms, 352 ms of a 15k prompt.
    pub fn step_batch_last(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.logits_last = !std::env::var("SHADOW_LOGITS_LAST").is_ok_and(|v| v == "0");
        let out = self.step_batch(tokens);
        self.logits_last = false;
        out
    }

    /// `step_batch` without the logits (reading a prompt).
    pub fn prefill_batch(&mut self, tokens: &[u32]) {
        self.step_batch_opt(tokens, false);
    }

    /// Whether `step_batch` takes this session's tokens through the layers together (else one at a time).
    fn batches(&self) -> bool {
        !(self.m.cfg.arch == "flint" || self.trace.is_some() || self.stop_after.is_some() || self.cache.slots == 0
          || self.prefetch.is_some() || self.pregate.is_some() || self.mtp_probe.is_some())
    }

    /// `step_batch` for a speculative verify pass (2..=MAX_VERIFY tokens): it also keeps what `rollback` needs to
    /// keep a prefix of them without reading it again.
    pub fn step_batch_verify(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let p0 = self.tokens.len();
        let keep = self.batches() && (2..=MAX_VERIFY.min(self.batch_max)).contains(&tokens.len());
        self.replay.on.set(keep);
        let (t0, w0) = (std::time::Instant::now(), self.waits);
        let (n0, ns0) = {
            use std::sync::atomic::Ordering::Relaxed;
            (crate::kernels::LAUNCHES.0.load(Relaxed), crate::kernels::LAUNCHES.1.load(Relaxed))
        };
        let out = self.step_batch(tokens);
        {
            use std::sync::atomic::Ordering::Relaxed;
            let v = &mut self.verify_stats;
            v[0] += 1.0;
            v[1] += tokens.len() as f64;
            v[2] += t0.elapsed().as_secs_f64();
            v[3] += self.waits[0] - w0[0];
            v[4] += self.waits[2] - w0[2];
            v[5] += (crate::kernels::LAUNCHES.1.load(Relaxed) - ns0) as f64 * 1e-9;
            v[6] += (crate::kernels::LAUNCHES.0.load(Relaxed) - n0) as f64;
        }
        self.replay.on.set(false);
        self.replay.pass = keep.then(|| (p0, tokens.to_vec()));
        out
    }

    /// Back to `snap` (taken just before the last `step_batch_verify`) plus the first tokens of that pass (`kept`):
    /// the recurrent states restored and only their updates for those tokens redone, from what the pass kept (the KV
    /// of those positions, the drafter cells and the final residuals are still the ones the pass wrote).  Without
    /// such a pass, restore and read the tokens again.
    /// The engine's word on a verify pass of `m` positions (the anchor and m - 1 drafts) of which `keep` were kept: the
    /// marginal commitment probability of each draft position (`--accept-moe`).
    pub fn note_verify(&mut self, keep: usize, m: usize) {
        if self.commit.len() < m {
            self.commit.resize(m, (0, 0));
        }
        for j in 1..m {
            self.commit[j].1 += 1;
            if keep > j {
                self.commit[j].0 += 1;
            }
        }
    }

    /// AcceptMoE's marginal probability that draft position j (1-based) reaches the output: the observed rate, with a
    /// prior (0.75, 0.56, 0.47, ... what the first runs measured) worth 8 passes.
    fn commit_prob(&self, j: usize) -> f64 {
        const PRIOR: [f64; 7] = [0.75, 0.56, 0.47, 0.39, 0.32, 0.27, 0.22];
        let p = PRIOR[(j - 1).min(PRIOR.len() - 1)];
        let (c, n) = self.commit.get(j).copied().unwrap_or((0, 0));
        (c as f64 + 8.0 * p) / (n as f64 + 8.0)
    }

    /// `--accept-moe` for one MoE layer of a verify pass: tops[0] (the anchor) stays natural; positions 1.. are re-routed
    /// through the block's eligible expert set.
    fn accept_route(&mut self, l: usize, logits: &[Vec<f32>], tops: &mut [Vec<(usize, f64)>]) {
        let k = self.m.cfg.n_expert_used;
        let m = tops.len();
        let nat: Vec<Vec<(usize, f64)>> = tops.to_vec();
        // commitment-weighted demand (a chain: one node a position, so the width discount is 1)
        let mut u: HashMap<usize, f64> = HashMap::new();
        for (j, top) in nat.iter().enumerate() {
            let alpha = if j == 0 { 1.0 } else { self.commit_prob(j + self.row_shift).powf(self.accept_beta) };
            for &(e, w) in top {
                *u.entry(e).or_insert(0.0) += alpha * w;
            }
        }
        let anchor: Vec<usize> = nat[0].iter().map(|&(e, _)| e).collect();
        let mut rest: Vec<(usize, f64)> = u.iter().filter(|(e, _)| !anchor.contains(e)).map(|(&e, &v)| (e, v)).collect();
        rest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let z: f64 = rest.iter().map(|r| r.1).sum();
        let n_er = if z > 0.0 {
            let h: f64 = -rest.iter().map(|r| r.1 / z).filter(|&q| q > 0.0).map(|q| q * q.ln()).sum::<f64>();
            (self.accept_scale * h.exp()).ceil() as usize
        } else {
            0
        };
        let mut set: Vec<usize> = anchor.clone();
        set.extend(rest.iter().take(n_er.max(k.saturating_sub(set.len()))).map(|r| r.0));
        // residency-aware pruning: non-resident members, lowest demand first, within the rerouting budget
        let votes = |e: usize| nat.iter().filter(|t| t.iter().any(|&(x, _)| x == e)).count();
        let displaced: usize = (self.accept_budget * u.keys().filter(|e| !set.contains(e)).map(|&e| votes(e)).sum::<usize>() as f64).ceil() as usize;
        let mut cand: Vec<usize> = set.iter().copied().filter(|e| !self.cache.map.contains_key(&(l, *e))).collect();
        cand.sort_by(|a, b| u.get(a).unwrap_or(&0.0).partial_cmp(u.get(b).unwrap_or(&0.0)).unwrap().then(a.cmp(b)));
        let (mut spent, mut removed) = (0usize, vec![]);
        for e in cand {
            let v = votes(e);
            if spent + v > displaced || set.len() - removed.len() <= k {
                break;
            }
            spent += v;
            removed.push(e);
        }
        set.retain(|e| !removed.contains(e));
        let mut masked = vec![f32::NEG_INFINITY; logits[0].len()];
        for j in 1..m {
            for v in masked.iter_mut() {
                *v = f32::NEG_INFINITY;
            }
            for &e in &set {
                masked[e] = logits[j][e];
            }
            let before = tops[j].len();
            tops[j] = route(&masked, k);
            let _ = before;
        }
        self.tail_skipped += removed.len() as u64;
    }

    pub fn rollback(&mut self, snap: &Snapshot<'c>, kept: &[u32]) -> Result<(), String> {
        let pass = self.replay.pass.take();
        let fits = pass.as_ref().is_some_and(|(p0, toks)| {
            *p0 == snap.tokens.len() && !kept.is_empty() && kept.len() <= toks.len() && toks[..kept.len()] == *kept
                && self.tokens.len() == p0 + toks.len()
        });
        self.restore(snap)?;
        if !fits {
            self.prefill_batch(kept);
            return Ok(());
        }
        let c = &self.m.cfg;
        let (kd, vh) = (c.ssm_k_heads * c.ssm_head_dim, c.ssm_v_heads);
        let chd = 2 * kd + vh * c.ssm_head_dim;
        let ch = c.hidden * c.streams;
        for l in 0..c.n_layer {
            if l == c.ple_layer {
                let p = format!("blk.{l}");
                for j in 0..kept.len() {
                    self.k1("ple_conv_step", ch, &[Arg::Ptr(self.replay.ple.ptr + (j * ch * 4) as u64), Arg::Ptr(self.ple_state.ptr),
                                                  Arg::Ptr(self.sv(&format!("{p}.ple_conv1d.weight"))), Arg::Ptr(self.b.gv.ptr),
                                                  Arg::U32(ch as u32), Arg::U32(c.ple_conv_k as u32), Arg::U32(c.ple_ngram as u32)]);
                }
            }
            if c.is_attention(l) {
                continue;
            }
            let qkv = self.replay.qkv.ptr + (l * MAX_VERIFY * chd * 4) as u64;
            let ba = self.replay.ba.ptr + (l * MAX_VERIFY * 2 * vh * 4) as u64;
            for j in 0..kept.len() {
                self.delta_mid(l, qkv + (j * chd * 4) as u64, self.b.z.ptr, ba + (j * vh * 4) as u64,
                               ba + ((MAX_VERIFY + j) * vh * 4) as u64, self.b.o.ptr);
            }
        }
        self.tokens.extend_from_slice(kept);
        if let Some(mt) = self.mtp.as_mut() {
            let last = mt.bufs.last_r.ptr;
            mt.pending = true;
            self.k1("copy", ch, &[Arg::Ptr(last), Arg::Ptr(row(&self.bb.r, ch, kept.len() - 1)), Arg::U32(ch as u32)]);
        }
        Ok(())
    }

    pub fn step_batch_opt(&mut self, tokens: &[u32], with_logits: bool) -> Vec<Vec<f32>> {
        let per_token = !self.batches();
        let mut out = Vec::new();
        for chunk in tokens.chunks(self.batch_max) {
            if per_token || chunk.len() == 1 {
                for &t in chunk {
                    let logits = self.step_opt(t, with_logits);
                    if with_logits {
                        out.push(logits);
                    }
                }
            } else {
                out.extend(self.batch(chunk, with_logits));
            }
        }
        out
    }

    fn batch(&mut self, toks: &[u32], with_logits: bool) -> Vec<Vec<f32>> {
        #[cfg(windows)]
        {
            // `SHADOW_MAIN_PRIO=N`: the thread that queues a long prompt's kernels runs at THREAD_PRIORITY N (1 above normal, 2 highest), so the read workers' memcpys do not
            // keep it waiting for a core (a launch is 5 us, but 1,000 a prompt took one scheduler quantum, ~16 ms)
            extern "system" {
                fn GetCurrentThread() -> isize;
                fn SetThreadPriority(h: isize, p: i32) -> i32;
            }
            #[link(name = "winmm")]
            extern "system" {
                fn timeBeginPeriod(ms: u32) -> u32;
            }
            static TIMER: std::sync::Once = std::sync::Once::new();
            // `SHADOW_TIMER_MS=N`: the Windows timer resolution (default tick 15.6 ms), for any wait the driver or OS makes in whole ticks
            TIMER.call_once(|| {
                if let Some(ms) = std::env::var("SHADOW_TIMER_MS").ok().and_then(|v| v.parse::<u32>().ok()) {
                    // SAFETY: winmm's timeBeginPeriod, a resolution in milliseconds.
                    unsafe { timeBeginPeriod(ms) };
                }
            });
            static PRIO: std::sync::OnceLock<Option<i32>> = std::sync::OnceLock::new();
            if let Some(p) = *PRIO.get_or_init(|| std::env::var("SHADOW_MAIN_PRIO").ok().and_then(|v| v.parse::<i32>().ok())) {
                // SAFETY: the pseudo-handle of this thread; a priority between -2 and 2.
                unsafe { SetThreadPriority(GetCurrentThread(), p.clamp(-2, 2)) };
            }
        }
        let m = toks.len();
        let p0 = self.tokens.len();
        assert!(p0 + m <= self.cap, "this session was made for {} positions", self.cap);
        self.tokens.extend_from_slice(toks);
        self.routing.tokens += m as u64;
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden, c.streams);
        let ch = h * cs;
        // a pass bigger than the temporaries' rows runs layer by layer in sub-chunks (the first takes the remainder, so the last is a full one), the residual streams
        // of every row in `rbig`; `bb.r` is a view of the sub-chunk being run
        let sub_mode = m > self.rows && !self.rparts.is_empty() && !self.pipeline_ok(m);
        let subs: Vec<(usize, usize)> = if sub_mode {
            let s = self.rows;
            let first = m % s;
            let mut v = vec![];
            if first > 0 {
                v.push((0, first));
            }
            v.extend((first..m).step_by(s).map(|s0| (s0, s)));
            v
        } else {
            vec![(0, m)]
        };
        assert!(sub_mode || m <= self.rows, "a pass of {m} rows with temporaries for {}", self.rows);
        // row j's residual streams are at storage row j + pad (the first sub-chunk, the remainder, ends part 0): a sub-chunk is inside one part
        let pad = if sub_mode { (self.rows - m % self.rows) % self.rows } else { 0 };
        let row_ptr = |j: usize| -> u64 {
            if sub_mode { self.rparts[(j + pad) / self.rows].ptr + (((j + pad) % self.rows) * ch * 4) as u64 } else { self.bb.r.ptr + (j * ch * 4) as u64 }
        };
        {
            let (t, bytes) = self.m.dense("token_embd.weight");
            let (qk, qb) = t.ty.block().unwrap();
            let rb = (t.dims[0] / qk * qb) as usize;
            for (j, &tok) in toks.iter().enumerate() {
                let mut x = vec![0f32; h];
                match self.picture_row(p0 + j) {
                    Some(r) => x.copy_from_slice(r),
                    None => dequantize(t.ty, &bytes[tok as usize * rb..(tok as usize + 1) * rb], &mut x).unwrap(),
                }
                self.upload_to(self.b.e.ptr, &f32s(&x));
                self.k1("broadcast", ch, &[Arg::Ptr(self.b.e.ptr), Arg::Ptr(row_ptr(j)), Arg::U32(h as u32),
                                           Arg::U32(cs as u32)]);
            }
        }
        let mut logs: Vec<Vec<u16>> = vec![vec![]; m];
        let graphs = self.graphs_on && self.trace.is_none() && self.stop_after.is_none() && self.prof.is_none()
            && self.prefetch.is_none() && self.pregate.is_none() && self.cache.slots > 0
            // `SHADOW_BATCH_GRAPHS=0`: no CUDA graphs for the prompt path.  A graph is keyed by buffer addresses and row counts, which a new prompt mostly changes, so a long prompt
            // instantiates ~500 (18 ms each, hidden behind the GPU) and a repeated shape then runs 10% faster (follow-up of 5,609 tokens: 481-509 against 444-451 tok/s);
            // a first run ties (docs/STATUS.md 2026-11-05 (30)).
            && !std::env::var("SHADOW_BATCH_GRAPHS").is_ok_and(|v| v == "0");
        if self.batch_graphs.len() > 4000 {
            // the cache would grow with every new prompt shape
            self.batch_graphs.clear();
        }
        let mut rn_ready = false;
        self.lt_point("(start)");
        let piped = self.pipeline_ok(m);
        if piped {
            rn_ready = self.layers_pipelined(m, p0, &mut logs, graphs);
        }
        let saved_r = if sub_mode { Some(std::mem::replace(&mut self.bb.r, unsafe { self.rparts[0].subview(0, 0) })) } else { None };
        // whether each row's normalized input to the layer is in `rn`; a sub-chunked pass keeps only the residual streams between sub-chunks, so each sub-chunk
        // normalizes its own at the start (`rmsnorm_groups_rows`, 0.2 ms) and `rn` is the sub-chunk's scratch
        let mut rn_layer = rn_ready;
        for l in 0..(if piped { 0 } else { c.n_layer }) {
            for (si, &(s0, ms)) in subs.iter().enumerate() {
                rn_ready = rn_layer && !sub_mode;
                if sub_mode {
                    // SAFETY: rows s0 .. s0 + ms of the residual streams, used by this sub-chunk only until the next view
                    self.bb.r = unsafe { self.rparts[(s0 + pad) / self.rows].subview(((s0 + pad) % self.rows) * ch * 4, ms * ch * 4) };
                    self.sub_last = si + 1 == subs.len();
                    self.sub_first = si == 0;
                }
                let (m, p0) = (ms, p0 + s0);
                if l == c.ple_layer {
                    // every position's rows at once, in parallel
                    let ple_probe = std::env::var("SHADOW_PLE_TIME").is_ok_and(|v| v == "1");
                    if ple_probe {
                        self.ctx.synchronize().unwrap();
                    }
                    let t_ple_block = std::time::Instant::now();
                    let all: Vec<i64> = (0..m).flat_map(|j| ple_rows(c, &self.tokens[..p0 + j + 1])).collect();
                    self.m.ple_prefetch(&all);
                    let t_pref = t_ple_block.elapsed().as_secs_f64();
                    for j in 0..m {
                        let r = row(&self.bb.r, ch, j);
                        self.ple_at(l, p0 + j + 1, r);
                        if self.replay.on.get() {
                            self.k1("copy", ch, &[Arg::Ptr(self.replay.ple.ptr + (j * ch * 4) as u64), Arg::Ptr(self.b.gvn.ptr),
                                                  Arg::U32(ch as u32)]);
                        }
                        self.k1("add", ch, &[Arg::Ptr(r), Arg::Ptr(self.b.gv.ptr), Arg::U32(ch as u32)]);
                    }
                    rn_ready = false;
                    if ple_probe {
                        let t_host = t_ple_block.elapsed().as_secs_f64();
                        self.ctx.synchronize().unwrap();
                        eprintln!("SHADOW_PLE_TIME m={m}: prefetch {:.2} ms, host to queued {:.2} ms, with the GPU done {:.2} ms",
                                  1e3 * t_pref, 1e3 * t_host, 1e3 * t_ple_block.elapsed().as_secs_f64());
                    }
                }
                {
                    let on = self.replay.on.get();
                    if !c.is_attention(l) {
                        self.ensure_multi(l, 0, on);
                    }
                    self.ensure_multi(l, 1, false);
                    self.ensure_multi(l, 2, false);
                }
                heap_check("layer loop top");
                if graphs && !c.is_attention(l) && l != c.ple_layer {
                    // a DeltaNet layer up to its router is the same launches for every batch of m: one graph
                    let on = self.replay.on.get();
                    // the key holds where the batch buffers are (the graph's launches carry their addresses): the same row count at another address - a prompt that lent another stretch of the KV window - is another graph
                    self.batch_graph_segment((l, rn_ready, m, on, self.rows, self.bb.r.ptr), |s| {
                        s.hc_read_batch(&format!("blk.{l}.hc_attn"), true, rn_ready, m);
                        s.delta_batch(l, m);
                        s.hc_write_batch(Some(&format!("blk.{l}.hc_ffn_norm.weight")), m);
                        s.hc_read_batch(&format!("blk.{l}.hc_ffn"), true, true, m);
                        s.moe_router_batch(l, m);
                    });
                    self.lt_point(if graphs { "delta segment (graph)" } else { "delta segment" });
                    heap_check("delta segment (graph) queued");
                    self.moe_batch(l, m, &mut logs[s0..s0 + m], true, None);
                    heap_check("moe_batch after a graph segment");
                    self.lt_point("moe (router wait, fetch, experts)");
                } else {
                    heap_check("layer start (eager)");
                    self.hc_read_batch(&format!("blk.{l}.hc_attn"), true, rn_ready, m);
                    if c.is_attention(l) {
                        self.attention_batch(l, p0, m);
                    } else {
                        self.delta_batch(l, m);
                    }
                    self.hc_write_batch(Some(&format!("blk.{l}.hc_ffn_norm.weight")), m);
                    self.hc_read_batch(&format!("blk.{l}.hc_ffn"), true, true, m);
                    self.mark("batch: mixers, norms, hc glue");
                    self.lt_point(if c.is_attention(l) { "attention layer" } else { "delta segment" });
                    heap_check(if c.is_attention(l) { "attention layer queued" } else { "delta segment queued" });
                    self.moe_batch(l, m, &mut logs[s0..s0 + m], false, None);
                    heap_check("moe_batch (eager)");
                    self.lt_point("moe (router wait, fetch, experts)");
                }
                self.mark("batch: routed experts + combine");
                let next = if l + 1 == c.ple_layer {
                    None
                } else if l + 1 < c.n_layer {
                    Some(format!("blk.{}.hc_attn_norm.weight", l + 1))
                } else {
                    Some("output_hc_norm.weight".to_string())
                };
                self.hc_write_batch(next.as_deref(), m);
                rn_ready = next.is_some();
                if self.dflash.is_some() && (!sub_mode || si + 1 == subs.len()) {
                    // the drafter looks back 2,048 positions: the last sub-chunk's features are the ones kept
                    self.df_capture(l, self.bb.r.ptr, m);
                }
            }
            rn_layer = rn_ready;
        }
        let (m_obs, p0_obs) = { let &(s0, ms) = subs.last().unwrap(); (ms, p0 + s0) };
        if with_logits {
            // the last sub-chunk's output rows (a sub-chunked pass gives logits for its last rows only; the rows before come back empty)
            self.hc_read_batch("output_hc", false, rn_ready, m_obs);
        }
        if let Some(r) = saved_r {
            // the view of the last sub-chunk goes with the buffers: the saved one is `bb.r` again
            let _ = std::mem::replace(&mut self.bb.r, r);
        }
        self.sub_last = true;
        self.sub_first = true;
        if let Some((_, log)) = self.route_log.as_mut() {
            log.extend(logs.into_iter().flatten());
        }
        if self.mtp.is_some() {
            self.mtp_observe(self.bb.r.ptr, toks, p0);
        }
        if self.dflash.is_some() {
            self.dflash_observe(m_obs, p0_obs);
        }
        if !with_logits {
            return vec![];
        }
        let mut logits = Vec::with_capacity(m);
        let first = if self.logits_last { m_obs - 1 } else { 0 };
        logits.resize(m - m_obs + first, vec![]);
        for j0 in (first..m_obs).step_by(COLS_MAX) {
            let n = (m_obs - j0).min(COLS_MAX);
            self.mv_cols("output.weight", row(&self.bb.u, h, j0), h, self.bb.logits.ptr, c.vocab, n);
            let mut out = vec![0u8; n * c.vocab * 4];
            self.download(self.bb.logits.ptr, &mut out);
            logits.extend(out.chunks(c.vocab * 4).map(floats));
        }
        logits
    }

    /// y_j = W x_j for j < m (x_j at x + j * xs floats, y_j at y + j * ys): bit for bit what `mv` gives each input,
    /// in one launch where a matvec_cols kernel computes the same sums (`mv` per input otherwise).
    fn mv_cols(&self, name: &str, x: u64, xs: usize, y: u64, ys: usize, m: usize) {
        let (t, bytes) = self.m.dense(name);
        let (qk, qb) = t.ty.block().unwrap();
        let rows = bytes.len() / (t.dims[0] / qk * qb) as usize;
        let ty = t.ty.name();
        let entry = match ty.as_str() {
            _ if self.m.scale(name) != 1.0 => None,
            // `mv` runs these in the reference order, and so do the _ref kernels
            "BF16" | "F32" if !self.fast_order => Some(format!("matvec_cols_ref_{}", ty.to_lowercase())),
            _ if t.dims[0].is_multiple_of(8) && xs.is_multiple_of(4) && x.is_multiple_of(16) => {
                Some(format!("matvec_cols_{}", ty.to_lowercase()))
            }
            _ => None,
        };
        let Some(entry) = entry else {
            for j in 0..m {
                self.mv(name, x + (j * xs * 4) as u64, y + (j * ys * 4) as u64);
            }
            return;
        };
        let wptr = self.w.get(name).unwrap_or_else(|| panic!("{name} is not resident")).ptr;
        if ty == "F32" && m >= 16 && std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "1") {
            eprintln!("F32 mv_cols {name} rows {rows} cols {} m {m}", t.dims[0]);
        }
        static MMA_BF16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mma_bf16 = ty == "BF16" && *MMA_BF16.get_or_init(|| !std::env::var("SHADOW_MMA_BF16").is_ok_and(|v| v == "0"));
        static MMA_F32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if ty == "F32" && !entry.starts_with("matvec_cols_ref_") && self.tensor_prompt && m >= 16 && t.dims[0].is_multiple_of(64) && *MMA_F32.get_or_init(|| !std::env::var("SHADOW_MMA_F32").is_ok_and(|v| v == "0")) {
            // SAFETY: as the mma_cols launch below: `rows` rows of t.dims[0] f32 weights, m inputs of t.dims[0] floats at x + j * xs, m outputs of `rows` floats at y + j * ys.
            unsafe {
                self.mvp.launch("mma_cols_f32x3", ((rows as u32).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0,
                                self.ls(), &[Arg::Ptr(wptr), Arg::Ptr(x), Arg::U32(xs as u32), Arg::Ptr(y), Arg::U32(ys as u32),
                                             Arg::U32(rows as u32), Arg::U32(t.dims[0] as u32), Arg::U32(m as u32)])
            }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
            return;
        }
        static MMA_NL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // on since the device planner (the pass is no longer paced by the copies: the shared experts' IQ4_NL down projection took 331 ms of a 3.5 s kernel window as 8-column launches); `SHADOW_MMA_NL=0` off
        let mma_nl = ty == "IQ4_NL" && *MMA_NL.get_or_init(|| !std::env::var("SHADOW_MMA_NL").is_ok_and(|v| v == "0"));
        static MMA_QK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mma_qk = matches!(ty.as_str(), "Q4_K" | "Q5_K") && *MMA_QK.get_or_init(|| !std::env::var("SHADOW_MMA_QK").is_ok_and(|v| v == "0"));
        if self.tensor_prompt && m >= 16 && (matches!(ty.as_str(), "Q6_K" | "IQ4_XS") || mma_bf16 || mma_qk || mma_nl) && t.dims[0].is_multiple_of(64)
            && entry.starts_with("matvec_cols_") && !entry.starts_with("matvec_cols_ref_") {
            // SAFETY: as the launches below: `rows` rows of t.dims[0] weights, m inputs of t.dims[0] floats at x + j * xs, m outputs of `rows`
            // floats at y + j * ys; the kernel reads 16-byte groups of the inputs (x and xs checked above) and guards the last tiles.
            unsafe {
                self.mvp.launch(&format!("mma_cols_{}", ty.to_lowercase()), ((rows as u32).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0,
                                self.ls(), &[Arg::Ptr(wptr), Arg::Ptr(x), Arg::U32(xs as u32), Arg::Ptr(y), Arg::U32(ys as u32),
                                             Arg::U32(rows as u32), Arg::U32(t.dims[0] as u32), Arg::U32(m as u32)])
            }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
            return;
        }
        for j0 in (0..m).step_by(COLS_MAX) {
            let (x, y, n) = (x + (j0 * xs * 4) as u64, y + (j0 * ys * 4) as u64, (m - j0).min(COLS_MAX));
            // SAFETY: `rows` whole rows of t.dims[0] weights at wptr; n inputs of t.dims[0] floats at x + j * xs and n
            // outputs of `rows` floats at y + j * ys, inside this session's buffers (the callers' strides are their
            // sizes).
            unsafe {
                self.mvp.launch(&entry, ((rows as u32).div_ceil(cols_rows_per_block(&entry, rows)), 1, 1), (256, 1, 1), 0, self.ls(),
                                &[Arg::Ptr(wptr), Arg::Ptr(x), Arg::U32(xs as u32), Arg::Ptr(y), Arg::U32(ys as u32),
                                  Arg::U32(rows as u32), Arg::U32(t.dims[0] as u32), Arg::U32(n as u32)])
            }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        }
    }

    /// The job table of a group of projections of one input (see `matvec_cols_multi`), or None when a tensor is not
    /// one the kernel covers (then the callers launch them one by one).
    fn multi_jobs(&self, jobs: &[(String, u64, usize)], cols: usize) -> Option<(Vec<u8>, u32)> {
        let mut bytes = Vec::with_capacity(jobs.len() * 32);
        let mut first = 0u32;
        for (name, y, ys) in jobs {
            let (t, b) = self.m.dense(name);
            let (qk, qb) = t.ty.block()?;
            let rows = b.len() / (t.dims[0] / qk * qb) as usize;
            let ty = match t.ty.name().as_str() {
                "Q4_K" => 0u32, "Q5_K" => 1, "Q6_K" => 2, "IQ4_XS" => 3, "BF16" => 4, "F32" => 5, "IQ4_NL" => 6,
                _ => return None,
            };
            if t.dims[0] as usize != cols || !cols.is_multiple_of(8) || self.m.scale(name) != 1.0 || !self.w.contains_key(name) {
                return None;
            }
            bytes.extend_from_slice(&self.w[name].ptr.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            for v in [*ys as u32, rows as u32, ty, first] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            first += (rows as u32).div_ceil(COLS_PLAIN_ROWS);
        }
        Some((bytes, first))
    }

    /// Makes sure the table for (layer, group, rollback) exists (before a graph capture needs it).
    fn ensure_multi(&mut self, l: usize, group: u8, on: bool) {
        if self.multi.contains_key(&(l, group, on)) || self.fuse & 1 == 0 || !self.fast_order {
            return;
        }
        let jobs = self.multi_group(l, group, on);
        if let Some((bytes, _)) = self.multi_jobs(&jobs, self.m.cfg.hidden) {
            let d = self.ctx.alloc(bytes.len()).unwrap();
            d.upload(&bytes).unwrap();
            self.multi.insert((l, group, on), d);
        }
    }

    /// Group 0: a DeltaNet layer's qkv, gate, beta, alpha (outputs: the rollback's slots when `on`).  Group 1: the
    /// router and the shared expert's gate logit (before the host reads the logits).  Group 2: the shared expert's
    /// gate and up (after the read, while the host routes and the copies run).
    fn multi_group(&self, l: usize, group: u8, on: bool) -> Vec<(String, u64, usize)> {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let bb = &self.bb;
        if group == 0 {
            let (vh, d) = (c.ssm_v_heads, c.ssm_head_dim);
            let (kd, vd) = (c.ssm_k_heads * d, vh * d);
            let ch = 2 * kd + vd;
            let (qkv_p, beta_p, alpha_p) = if on {
                let ba = self.replay.ba.ptr + (l * MAX_VERIFY * 2 * vh * 4) as u64;
                (self.replay.qkv.ptr + (l * MAX_VERIFY * ch * 4) as u64, ba, ba + (MAX_VERIFY * vh * 4) as u64)
            } else {
                (bb.qkv.ptr, bb.beta.ptr, bb.alpha.ptr)
            };
            vec![(format!("{p}.attn_qkv.weight"), qkv_p, ch), (format!("{p}.attn_gate.weight"), bb.z.ptr, vd),
                 (format!("{p}.ssm_beta.weight"), beta_p, vh), (format!("{p}.ssm_alpha.weight"), alpha_p, vh)]
        } else {
            if group == 1 {
                vec![(format!("{p}.ffn_gate_inp.weight"), bb.rl.ptr, c.n_expert),
                     (format!("{p}.ffn_gate_inp_shexp.weight"), bb.sgl.ptr, 1)]
            } else {
                vec![(format!("{p}.ffn_gate_shexp.weight"), bb.sg.ptr, c.shared_ff),
                     (format!("{p}.ffn_up_shexp.weight"), bb.su.ptr, c.shared_ff)]
            }
        }
    }

    /// The group's projections of bb.u for the batch's m tokens: one launch per chunk of 8, or one by one.
    fn multi_cols(&self, l: usize, group: u8, on: bool, m: usize) {
        let h = self.m.cfg.hidden;
        let u = self.bb.u.ptr;
        if let Some(tab) = self.multi.get(&(l, group, on)) {
            let jobs = self.multi_group(l, group, on);
            let (_, blocks) = self.multi_jobs(&jobs, h).unwrap();
            for j0 in (0..m).step_by(COLS_MAX) {
                let n = (m - j0).min(COLS_MAX) as u32;
                // SAFETY: the table's jobs are resident matrices of h columns and outputs inside bb / replay.
                unsafe {
                    self.mvp.launch("matvec_cols_multi", (blocks, 1, 1), (256, 1, 1), 0, self.ls(), &[
                        Arg::Ptr(tab.ptr), Arg::U32(jobs.len() as u32), Arg::Ptr(u + (j0 * h * 4) as u64), Arg::U32(h as u32),
                        Arg::U32(h as u32), Arg::U32(n), Arg::U32(j0 as u32)])
                }
                .unwrap_or_else(|err| panic!("matvec_cols_multi: {err}"));
            }
            return;
        }
        for (name, y, ys) in self.multi_group(l, group, on) {
            self.mv_cols(&name, u, h, y, ys, m);
        }
    }

    /// `count` matrices of t's shape from the device table at `tab`, one per blockIdx.y, on inputs x + n * xs
    /// (xs = 0: shared) into y + n * ys.
    #[allow(clippy::too_many_arguments)]
    fn mv_batch(&self, t: &TensorInfo, tab: u64, count: usize, x: u64, xs: usize, y: u64, ys: usize) {
        let ty = t.ty.name();
        let entry = FAST_KERNELS.iter().find(|q| q.0 == ty).unwrap_or_else(|| panic!("{}: no batched kernel", t.name)).2;
        let (rows, cols) = (t.dims[1] as u32, t.dims[0] as u32);
        assert_eq!(cols % 8, 0, "{}: the batched kernel needs whole chunks of 8", t.name);
        // SAFETY: `count` table entries from tab, each a rows x cols matrix as stored; `count` inputs of
        // cols floats (stride xs, or 0 when shared) and `count` outputs of rows floats (stride ys) inside this
        // session's buffers.
        unsafe {
            self.mvp.launch(entry, (rows.div_ceil(8), count as u32, 1), (256, 1, 1), 0, self.ls(),
                            &[Arg::Ptr(tab), Arg::Ptr(x), Arg::U32(xs as u32), Arg::Ptr(y),
                              Arg::U32(ys as u32), Arg::U32(rows), Arg::U32(cols)])
        }
        .unwrap_or_else(|err| panic!("{}: {err}", t.name));
    }

    /// `hc_read` for the batch's m tokens.
    fn hc_read_batch(&self, p: &str, inject: bool, skip_norm: bool, m: usize) {
        let c = &self.m.cfg;
        let (h, cs, rank) = (c.hidden, c.streams, c.hc_rank);
        let ch = h * cs;
        let bb = &self.bb;
        if !skip_norm && m >= 2 {
            // one launch for all rows (the same arithmetic as a launch a row)
            self.k2("rmsnorm_groups_rows", cs as u32, m as u32, 256, &[Arg::Ptr(bb.r.ptr), Arg::Ptr(self.sv(&format!("{p}_norm.weight"))), Arg::Ptr(bb.rn.ptr),
                                                                       Arg::U32(h as u32), Arg::F32(c.eps), Arg::U32(cs as u32)]);
        } else if !skip_norm {
            for j in 0..m {
                self.hc_norm(p, row(&bb.r, ch, j), row(&bb.rn, ch, j));
            }
        }
        let (w_inj, r_inj) = if inject { (self.w[&format!("{p}_inject.weight")].ptr, cs as u32) } else { (0, 0) };
        static MMA_HC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mma_hc = self.tensor_prompt && self.fast_order && m >= 16 && ch.is_multiple_of(64) && *MMA_HC.get_or_init(|| !std::env::var("SHADOW_MMA_HC").is_ok_and(|v| v == "0"));
        if mma_hc {
            // SAFETY: as the launch below, for all m inputs at once; the kernel guards the last row and token tiles.
            unsafe {
                self.mvp.launch("mma_hc_down_inject_bf16", ((rank as u32 + r_inj).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0, self.ls(), &[
                    Arg::Ptr(self.w[&format!("{p}_down.weight")].ptr), Arg::Ptr(bb.down.ptr), Arg::U32(rank as u32),
                    Arg::U32(rank as u32), Arg::Ptr(w_inj), Arg::Ptr(bb.g.ptr), Arg::U32(r_inj), Arg::U32(cs as u32),
                    Arg::Ptr(bb.rn.ptr), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::F32(cs as f32), Arg::U32(m as u32),
                ]).unwrap_or_else(|err| panic!("mma_hc_down_inject_bf16: {err}"));
            }
        }
        for j0 in (0..m).step_by(COLS_MAX).filter(|_| !mma_hc) {
            // SAFETY: rank + r_inj rows of ch BF16 weights; n inputs of ch floats in rn, n outputs of rank (down) and
            // cs (g) floats, from row j0.
            unsafe {
                let (entry, per_block) = if self.fast_order {
                    ("hc_down_inject_cols_fast_bf16", 2)
                } else {
                    ("hc_down_inject_cols_bf16", cols_rows_per_block("hc_down_inject_cols_bf16", rank + r_inj as usize))
                };
                self.mvp.launch(entry, ((rank as u32 + r_inj).div_ceil(per_block), 1, 1), (256, 1, 1), 0, self.ls(), &[
                    Arg::Ptr(self.w[&format!("{p}_down.weight")].ptr), Arg::Ptr(row(&bb.down, rank, j0)), Arg::U32(rank as u32),
                    Arg::U32(rank as u32), Arg::Ptr(w_inj), Arg::Ptr(row(&bb.g, cs, j0)), Arg::U32(r_inj), Arg::U32(cs as u32),
                    Arg::Ptr(row(&bb.rn, ch, j0)), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::F32(cs as f32),
                    Arg::U32((m - j0).min(COLS_MAX) as u32),
                ]).unwrap_or_else(|err| panic!("hc_down_inject_cols_bf16: {err}"));
            }
        }
        if self.fuse & 2 != 0 && self.fast_order {
            self.hc_up_mix(p, bb.down.ptr, rank, bb.rn.ptr, ch, bb.u.ptr, h, m);
            return;
        }
        self.mv_cols(&format!("{p}_up.weight"), bb.down.ptr, rank, bb.up.ptr, ch, m);
        self.k2("hc_mix_rows", (h as u32).div_ceil(256), m as u32, 256, &[Arg::Ptr(bb.rn.ptr), Arg::Ptr(bb.up.ptr), Arg::Ptr(bb.u.ptr),
                                                                         Arg::U32(h as u32), Arg::U32(cs as u32)]);
    }

    /// `hc_write_norm` (with the next norm) or `hc_write` (without) for the batch's m tokens.
    fn hc_write_batch(&self, next_norm: Option<&str>, m: usize) {
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden, c.streams);
        let bb = &self.bb;
        if let Some(n) = next_norm {
            self.k2("hc_write_rmsnorm_rows", cs as u32, m as u32, 256, &[
                Arg::Ptr(bb.r.ptr), Arg::Ptr(bb.g.ptr), Arg::Ptr(bb.y.ptr), Arg::Ptr(self.sv(n)), Arg::Ptr(bb.rn.ptr),
                Arg::U32(h as u32), Arg::F32(c.eps), Arg::U32(cs as u32)]);
            return;
        }
        for j in 0..m {
            self.hc_write_at(row(&bb.r, h * cs, j), row(&bb.g, cs, j), row(&bb.y, h, j));
        }
    }

    fn delta_batch(&self, l: usize, m: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (h, vh, d) = (c.hidden, c.ssm_v_heads, c.ssm_head_dim);
        let (kd, vd) = (c.ssm_k_heads * d, vh * d);
        let ch = 2 * kd + vd;
        let bb = &self.bb;
        // a verify pass writes the recurrence's inputs straight into the rollback's slots for this layer
        let on = self.replay.on.get();
        let (qkv_p, beta_p, alpha_p) = if on {
            let ba = self.replay.ba.ptr + (l * MAX_VERIFY * 2 * vh * 4) as u64;
            (self.replay.qkv.ptr + (l * MAX_VERIFY * ch * 4) as u64, ba, ba + (MAX_VERIFY * vh * 4) as u64)
        } else {
            (bb.qkv.ptr, bb.beta.ptr, bb.alpha.ptr)
        };
        let _ = h;
        self.multi_cols(l, 0, on, m);
        if d == 128 {
            // conv and norms token by token (the conv window moves), then the recurrence for all of them at once
            let (s, conv) = self.delta[l].as_ref().unwrap();
            let kh = c.ssm_k_heads;
            // every (token, channel) its own thread, the same sums in the same order (`SHADOW_CONV_PAR=0`: one thread a channel walking the tokens)
            if m >= 8 && !std::env::var("SHADOW_CONV_PAR").is_ok_and(|v| v == "0") {
                self.k2("conv_silu_steps_par", (ch as u32).div_ceil(256), m as u32, 256, &[Arg::Ptr(qkv_p), Arg::Ptr(conv.ptr),
                                            Arg::Ptr(self.sv(&format!("{p}.ssm_conv1d.weight"))), Arg::Ptr(bb.x.ptr),
                                            Arg::U32(ch as u32), Arg::U32(c.conv_k as u32), Arg::U32(m as u32)]);
                self.k1("conv_state_update", ch, &[Arg::Ptr(qkv_p), Arg::Ptr(conv.ptr), Arg::U32(ch as u32), Arg::U32(c.conv_k as u32), Arg::U32(m as u32)]);
            } else {
            self.k1("conv_silu_steps", ch, &[Arg::Ptr(qkv_p), Arg::Ptr(conv.ptr),
                                            Arg::Ptr(self.sv(&format!("{p}.ssm_conv1d.weight"))), Arg::Ptr(bb.x.ptr),
                                            Arg::U32(ch as u32), Arg::U32(c.conv_k as u32), Arg::U32(m as u32)]);
            }
            self.k2("l2norm_qk_rows", (2 * kh) as u32, m as u32, 128, &[Arg::Ptr(bb.x.ptr), Arg::U32(kh as u32), Arg::U32(d as u32),
                                                                       Arg::F32((d as f32).sqrt()), Arg::U32(ch as u32)]);
            if self.fuse & 4 != 0 {
                // the recurrence and the head's gated RMSNorm in one kernel
                self.k("delta_steps128_gn", vh as u32, 128, &[
                    Arg::Ptr(bb.x.ptr), Arg::Ptr(bb.x.ptr + (kd * 4) as u64), Arg::Ptr(bb.x.ptr + (2 * kd * 4) as u64),
                    Arg::U32(ch as u32), Arg::Ptr(beta_p), Arg::Ptr(alpha_p), Arg::U32(vh as u32),
                    Arg::Ptr(self.sv(&format!("{p}.ssm_a"))), Arg::Ptr(self.sv(&format!("{p}.ssm_dt.bias"))), Arg::Ptr(s.ptr),
                    Arg::Ptr(bb.o.ptr), Arg::U32(vd as u32), Arg::U32(kh as u32), Arg::U32(m as u32), Arg::Ptr(bb.z.ptr),
                    Arg::Ptr(self.sv(&format!("{p}.ssm_norm.weight"))), Arg::F32(c.eps)]);
            } else {
                static DELTA_CHUNK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if m >= 64 && m <= BIG_BATCH && *DELTA_CHUNK.get_or_init(|| std::env::var("SHADOW_DELTA_CHUNK").is_ok_and(|v| v == "1")) {
                    let nch = m.div_ceil(64) as u32;
                    self.k2("delta_chunk_prep", nch, vh as u32, 256, &[
                        Arg::Ptr(bb.x.ptr), Arg::Ptr(bb.x.ptr + (kd * 4) as u64), Arg::Ptr(beta_p), Arg::Ptr(alpha_p),
                        Arg::Ptr(self.sv(&format!("{p}.ssm_a"))), Arg::Ptr(self.sv(&format!("{p}.ssm_dt.bias"))), Arg::Ptr(self.dscr.ptr),
                        Arg::U32(ch as u32), Arg::U32(vh as u32), Arg::U32(kh as u32), Arg::U32(vh as u32), Arg::U32(m as u32)]);
                    self.k2("delta_chunk_run", vh as u32, (d / 16) as u32, 256, &[
                        Arg::Ptr(bb.x.ptr), Arg::Ptr(bb.x.ptr + (kd * 4) as u64), Arg::Ptr(bb.x.ptr + (2 * kd * 4) as u64), Arg::Ptr(self.dscr.ptr),
                        Arg::Ptr(s.ptr), Arg::Ptr(bb.o.ptr), Arg::U32(ch as u32), Arg::U32(vd as u32), Arg::U32(kh as u32), Arg::U32(vh as u32), Arg::U32(m as u32)]);
                    self.gated_norm_rows(&format!("{p}.ssm_norm.weight"), d, vh, vd, m);
                } else {
                static DELTA_X8: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let x8 = m >= 16 && *DELTA_X8.get_or_init(|| std::env::var("SHADOW_DELTA_X8").is_ok_and(|v| v == "1"));
                self.k(if x8 { "delta_steps128_x8" } else { "delta_steps128" }, vh as u32, if x8 { 1024 } else { 128 }, &[
                    Arg::Ptr(bb.x.ptr), Arg::Ptr(bb.x.ptr + (kd * 4) as u64), Arg::Ptr(bb.x.ptr + (2 * kd * 4) as u64),
                    Arg::U32(ch as u32), Arg::Ptr(beta_p), Arg::Ptr(alpha_p), Arg::U32(vh as u32),
                    Arg::Ptr(self.sv(&format!("{p}.ssm_a"))), Arg::Ptr(self.sv(&format!("{p}.ssm_dt.bias"))), Arg::Ptr(s.ptr),
                    Arg::Ptr(bb.o.ptr), Arg::U32(vd as u32), Arg::U32(kh as u32), Arg::U32(m as u32)]);
                self.gated_norm_rows(&format!("{p}.ssm_norm.weight"), d, vh, vd, m);
                }
            }
        } else {
            for j in 0..m {
                self.delta_mid(l, qkv_p + (j * ch * 4) as u64, row(&bb.z, vd, j), beta_p + (j * vh * 4) as u64,
                               alpha_p + (j * vh * 4) as u64, row(&bb.o, vd, j));
            }
        }
        self.mv_cols(&format!("{p}.ssm_out.weight"), bb.o.ptr, vd, bb.y.ptr, h, m);
    }

    /// Attention for the batch's tokens at positions p0..p0 + m: v and the indexer keys go straight into the caches
    /// (consecutive positions are consecutive rows), then each token attends over the positions up to its own.
    fn attention_batch(&mut self, l: usize, p0: usize, m: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let h = c.hidden;
        let (qd, kvd, id) = (c.n_head * c.head_dim, c.n_kv_head * c.head_dim, c.idx_dim);
        let iqd = c.idx_heads * id;
        {
            let bb = &self.bb;
            let at = self.attn[l].as_ref().unwrap();
            self.mv_cols(&format!("{p}.attn_q.weight"), bb.u.ptr, h, bb.qg.ptr, 2 * qd, m);
            self.mv_cols(&format!("{p}.attn_k.weight"), bb.u.ptr, h, bb.kk.ptr, kvd, m);
            self.mv_cols(&format!("{p}.attn_v.weight"), bb.u.ptr, h, at.v.ptr + (p0 * kvd * 4) as u64, kvd, m);
            self.mv_cols(&format!("{p}.indexer.q_proj.weight"), bb.u.ptr, h, bb.iq.ptr, iqd, m);
            self.mv_cols(&format!("{p}.indexer.k_proj.weight"), bb.u.ptr, h, at.raw.ptr + (p0 * id * 4) as u64, id, m);
        }
        let ratio = c.compress[l].max(1);
        if self.trace.is_none() {
            // the tokens before the first one that needs the sparse selection: everything per token in one launch per step, in
            // token order; the rest select their positions sel_rows() at a time (one scores download for each group)
            let (nh, nkv, hd) = (c.n_head, c.n_kv_head, c.head_dim);
            let (rd, th, eps) = (c.rope_dims as u32, c.rope_theta as f32, c.eps);
            let done = self.attn[l].as_ref().unwrap().nblocks;
            static ONE_LAUNCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let one = *ONE_LAUNCH.get_or_init(|| !std::env::var("SHADOW_POOL_BATCH").is_ok_and(|v| v == "0"));
            if !one {
                for blk in done..(p0 + m) / ratio {
                    let at = self.attn[l].as_ref().unwrap();
                    self.k("pool_block", 1, 128, &[
                        Arg::Ptr(at.raw.ptr), Arg::Ptr(at.blocks.ptr), Arg::Ptr(self.sv(&format!("{p}.indexer.k_norm.weight"))),
                        Arg::U32(blk as u32), Arg::U32(ratio as u32), Arg::U32(id as u32), Arg::F32(eps), Arg::U32(rd), Arg::F32(th)]);
                }
            } else if (p0 + m) / ratio > done {
                // one launch for all the blocks this pass completes (it was one each)
                let at = self.attn[l].as_ref().unwrap();
                self.k("pool_blocks", ((p0 + m) / ratio - done) as u32, 128, &[
                    Arg::Ptr(at.raw.ptr), Arg::Ptr(at.blocks.ptr), Arg::Ptr(self.sv(&format!("{p}.indexer.k_norm.weight"))),
                    Arg::U32(done as u32), Arg::U32(ratio as u32), Arg::U32(id as u32), Arg::F32(eps), Arg::U32(rd), Arg::F32(th)]);
            }
            self.attn[l].as_mut().unwrap().nblocks = (p0 + m) / ratio;
            let bb = &self.bb;
            let at = self.attn[l].as_ref().unwrap();
            self.k2("head_norm_rope_qgk_rows", (nh + nkv) as u32, m as u32, 128, &[
                Arg::Ptr(bb.qg.ptr), Arg::Ptr(bb.q.ptr), Arg::Ptr(bb.gate.ptr), Arg::Ptr(self.sv(&format!("{p}.attn_q_norm.weight"))),
                Arg::Ptr(bb.kk.ptr), Arg::Ptr(at.k.ptr), Arg::Ptr(self.sv(&format!("{p}.attn_k_norm.weight"))),
                Arg::U32(nh as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::F32(eps), Arg::U32(p0 as u32),
                Arg::U32(rd), Arg::F32(th)]);
            let keep = c.idx_budget / ratio;
            // the first token with more complete blocks than the budget keeps
            let jt = (0..m).find(|&j| (p0 + j + 1) / ratio > keep).unwrap_or(m);
            // the first positions attend every earlier token: float scores and sums with the key loads shared over `hb` heads a block (`SHADOW_ATT_F32=0`: the double kernel,
            // a block a head); the float scratch holds twice the rows
            let dense_f32 = self.tensor_prompt && !std::env::var("SHADOW_ATT_F32").is_ok_and(|v| v == "0");
            let dense_rows = if dense_f32 { 2 * ATT_ROWS } else { ATT_ROWS };
            for j0 in (0..jt).step_by(dense_rows) {
                let n = (jt - j0).min(dense_rows);
                if dense_f32 {
                    let group = nh / nkv;
                    let hb = std::env::var("SHADOW_ATT_HB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).clamp(1, 8).min(group);
                    self.k2("attention_group_dense_rows_f32", (nkv * group.div_ceil(hb)) as u32, n as u32, 256, &[
                        Arg::Ptr(row(&bb.q, qd, j0)), Arg::Ptr(at.k.ptr), Arg::Ptr(at.v.ptr), Arg::U32((p0 + j0 + 1) as u32),
                        Arg::Ptr(row(&bb.gate, qd, j0)), Arg::Ptr(row(&bb.ao, qd, j0)), Arg::Ptr(bb.att.ptr),
                        Arg::U32(self.cap as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(group as u32),
                        Arg::U32(nh as u32), Arg::U32(hb as u32)]);
                    continue;
                }
                self.k2("attention_heads_rows", nh as u32, n as u32, 256, &[
                    Arg::Ptr(row(&bb.q, qd, j0)), Arg::Ptr(at.k.ptr), Arg::Ptr(at.v.ptr), Arg::U32((p0 + j0 + 1) as u32),
                    Arg::Ptr(row(&bb.gate, qd, j0)), Arg::Ptr(row(&bb.ao, qd, j0)), Arg::Ptr(bb.att.ptr),
                    Arg::U32(self.cap as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32((nh / nkv) as u32),
                    Arg::U32(nh as u32)]);
            }
            // the same kernels, inputs and tie-breaking as `attention_mid`, so the same numbers: per token the indexer's query and
            // block scores, then (after one download for the group) the selection, then attention over the selected positions
            let sstride = self.cap / ratio + 1;
            let tstride = c.idx_budget + ratio;
            let idx_heads = c.idx_heads;
            let sel_time = std::env::var("SHADOW_SEL_TIME").is_ok_and(|v| v == "1");
            // `SHADOW_SEL_CPU=1`: the host selects (download, select_nth, upload per group), as before
            let sel_cpu = std::env::var("SHADOW_SEL_CPU").is_ok_and(|v| v == "1");
            let (mut t_launch, mut t_down, mut t_cpu, mut t_up, mut groups) = (0f64, 0f64, 0f64, 0f64, 0usize);
            let t_all = std::time::Instant::now();
            let mut j = jt;
            while j < m {
                let tg = std::time::Instant::now();
                let e = (j + sel_rows()).min(m);
                // the group's rows in two launches: the indexer queries, then every row's block scores
                self.k2("head_norm_rope_rows", idx_heads as u32, (e - j) as u32, 128, &[
                    Arg::Ptr(row(&bb.iq, iqd, j)), Arg::U32(id as u32), Arg::U32(iqd as u32), Arg::Ptr(bb.sel_iqn.ptr), Arg::U32(id as u32),
                    Arg::U32(iqd as u32), Arg::Ptr(self.sv(&format!("{p}.indexer.q_norm.weight"))), Arg::U32(id as u32), Arg::F32(eps),
                    Arg::U32((p0 + j) as u32), Arg::U32(rd), Arg::F32(th)]);
                let max_nb = (p0 + e) / ratio;
                // float dot products (with `--tensor-prompt`; `SHADOW_IDX_F32=0` keeps the double ones that the card's FP64 rate bounds): a tie while the copies paced the
                // pass, +2-7% with the device planner (docs/STATUS.md 2026-11-04), at the cost of a different (not a worse) reply
                let idx_kernel = if self.tensor_prompt && !std::env::var("SHADOW_IDX_F32").is_ok_and(|v| v == "0") { "index_scores_rows_f32" } else { "index_scores_rows" };
                // register-blocked (4 rows, 2 key blocks a thread; the same scores bit for bit): `SHADOW_IDX_V2=0` keeps the one-thread-a-block kernel
                if idx_kernel == "index_scores_rows_f32" && idx_heads <= 4 && id % 4 == 0 && id <= 256 && !std::env::var("SHADOW_IDX_V2").is_ok_and(|v| v == "0") {
                    self.k2("index_scores_rows_f32v2", (max_nb as u32).div_ceil(256), (e - j).div_ceil(4) as u32, 128, &[
                        Arg::Ptr(bb.sel_iqn.ptr), Arg::Ptr(at.blocks.ptr), Arg::Ptr(bb.sel_scores.ptr), Arg::U32((p0 + j + 1) as u32),
                        Arg::U32(ratio as u32), Arg::U32(idx_heads as u32), Arg::U32(id as u32), Arg::U32(iqd as u32),
                        Arg::U32(sstride as u32), Arg::U32((e - j) as u32)]);
                } else {
                self.k2(idx_kernel, (max_nb as u32).div_ceil(256), (e - j) as u32, 256, &[
                    Arg::Ptr(bb.sel_iqn.ptr), Arg::Ptr(at.blocks.ptr), Arg::Ptr(bb.sel_scores.ptr), Arg::U32((p0 + j + 1) as u32),
                    Arg::U32(ratio as u32), Arg::U32(idx_heads as u32), Arg::U32(id as u32), Arg::U32(iqd as u32),
                    Arg::U32(sstride as u32)]);
                }
                let lens = bb.sel_toks.ptr + (sel_rows() * tstride * 4) as u64;
                if !sel_cpu {
                    // the selection on the GPU (the same choice as the host's below): no download, no host work, the group's launches queue behind each other
                    self.k2("select_blocks_rows", (e - j) as u32, 1, 256, &[
                        Arg::Ptr(bb.sel_scores.ptr), Arg::Ptr(bb.sel_toks.ptr), Arg::Ptr(lens), Arg::U32(sstride as u32), Arg::U32(tstride as u32),
                        Arg::U32((p0 + j + 1) as u32), Arg::U32(ratio as u32), Arg::U32(keep as u32)]);
                } else {
                let mut sb = vec![0u8; (e - j) * sstride * 4];
                t_launch += tg.elapsed().as_secs_f64();
                let td = std::time::Instant::now();
                self.download(bb.sel_scores.ptr, &mut sb);
                t_down += td.elapsed().as_secs_f64();
                let tc = std::time::Instant::now();
                let all = floats(&sb);
                let mut toks_host = vec![0u32; (e - j) * tstride];
                let mut ntoks = Vec::with_capacity(e - j);
                for jj in j..e {
                    let r = jj - j;
                    let n = p0 + jj + 1;
                    let nb = n / ratio;
                    let mut scored: Vec<(f32, usize)> = all[r * sstride..r * sstride + nb].iter().copied().zip(0..nb).collect();
                    let by_score = |a: &(f32, usize), b: &(f32, usize)| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1));
                    // the `keep` best blocks, by score then block number: a total order, so the set is the full sort's
                    scored.select_nth_unstable_by(keep - 1, by_score);
                    let mut toks: Vec<u32> = scored[..keep].iter().flat_map(|&(_, blk)| (blk * ratio) as u32..((blk + 1) * ratio) as u32).collect();
                    toks.extend((nb * ratio) as u32..n as u32);
                    toks.sort_unstable();
                    toks_host[r * tstride..r * tstride + toks.len()].copy_from_slice(&toks);
                    ntoks.push(toks.len());
                }
                // the selections, then their lengths (one word each) after the sel_rows() rows
                t_cpu += tc.elapsed().as_secs_f64();
                let tu = std::time::Instant::now();
                toks_host.resize(sel_rows() * tstride + sel_rows(), 0);
                for (r, &n) in ntoks.iter().enumerate() {
                    toks_host[sel_rows() * tstride + r] = n as u32;
                }
                self.upload_to(bb.sel_toks.ptr, &toks_host.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                t_up += tu.elapsed().as_secs_f64();
                }
                groups += 1;
                // a row needs only its selection's length of scratch (tstride), not the context's (cap): more rows a launch
                let per_launch = (ATT_ROWS * self.cap / tstride).clamp(1, sel_rows());
                for r0 in (0..e - j).step_by(per_launch) {
                    let n = (e - j - r0).min(per_launch);
                    let group = nh / nkv;
                    let args = [
                        Arg::Ptr(row(&bb.q, qd, j + r0)), Arg::Ptr(at.k.ptr), Arg::Ptr(at.v.ptr), Arg::Ptr(row(&bb.sel_toks, tstride, r0)),
                        Arg::U32(tstride as u32), Arg::Ptr(lens + (r0 * 4) as u64), Arg::Ptr(row(&bb.gate, qd, j + r0)),
                        Arg::Ptr(row(&bb.ao, qd, j + r0)), Arg::Ptr(bb.att.ptr), Arg::U32(tstride as u32), Arg::U32(nkv as u32),
                        Arg::U32(hd as u32), Arg::U32(group as u32), Arg::U32(nh as u32)];
                    if hd % 16 == 0 && hd <= 256 {
                        // the heads of a GQA group share each position's K and V reads, `hb` of them a block
                        let hb = std::env::var("SHADOW_ATT_HB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).clamp(1, 8).min(group);
                        let mut a = args.to_vec();
                        a.push(Arg::U32(hb as u32));
                        // float scores, softmax and weighted sum (with `--tensor-prompt`; `SHADOW_ATT_F32=0` keeps the double kernel, FP64-bound at 74% of the pipe): a tie while the
                        // copies paced the pass, +2-7% with the device planner (docs/STATUS.md 2026-11-04)
                        let f32_on = self.tensor_prompt && !std::env::var("SHADOW_ATT_F32").is_ok_and(|v| v == "0");
                        // the profile-built kernel (4 heads a block, 4 tokens a thread, shuffles, exp folded into the weighted sum): `SHADOW_ATT_V2=0` keeps the first float kernel
                        let v2 = f32_on && !std::env::var("SHADOW_ATT_V2").is_ok_and(|v| v == "0");
                        // all of the KV group's heads in one block: K and V are read once for the group, not once for each 4 heads (`SHADOW_ATT_V3=0` off): +3.2-3.8%
                        let v3 = v2 && group <= 12 && !std::env::var("SHADOW_ATT_V3").is_ok_and(|v| v == "0");
                        let kernel = if v3 { "attention_group_sel_rows_v3" } else if v2 { "attention_group_sel_rows_v2" } else if f32_on { "attention_group_sel_rows_f32" } else { "attention_group_sel_rows" };
                        let hb = if v3 { group } else if v2 { 4 } else { hb };
                        self.k2(kernel, (nkv * group.div_ceil(hb)) as u32, n as u32, 256, &a);
                    } else {
                        self.k2("attention_heads_sel_rows", nh as u32, n as u32, 256, &args);
                    }
                }
                j = e;
            }
            if sel_time {
                eprintln!("SEL L{l} m={m} groups {groups}: total {:.1} ms = launch {:.1} + download(wait) {:.1} + cpu select {:.1} + upload {:.1} (rest = attention launches)",
                          t_all.elapsed().as_secs_f64() * 1e3, t_launch * 1e3, t_down * 1e3, t_cpu * 1e3, t_up * 1e3);
            }
        } else {
            for j in 0..m {
                let bb = &self.bb;
                let (qg, kk, iq, ao) = (row(&bb.qg, 2 * qd, j), row(&bb.kk, kvd, j), row(&bb.iq, iqd, j), row(&bb.ao, qd, j));
                self.attention_mid(l, p0 + j, qg, kk, iq, ao);
            }
        }
        self.mv_cols(&format!("{p}.attn_output.weight"), self.bb.ao.ptr, qd, self.bb.y.ptr, h, m);
    }

    /// The MoE for the batch's m tokens: the router and the shared expert for all of them at once, one read of their
    /// router logits, then the union of their routed experts into the VRAM cache (the misses beyond the layer's free
    /// slots into `batch_scratch`) - in groups of consecutive tokens whose misses fit together - and each token's
    /// routed experts and combine.
    /// The router (and the shared expert's gate logit) for the batch's m tokens.
    fn moe_router_batch(&self, l: usize, m: usize) {
        // the router and the shared expert's gate logit: one launch
        self.multi_cols(l, 1, false, m);
    }

    /// Whether this pass's MoE is routed and planned on the device (`SHADOW_PREFILL_DEVPLAN=1`): a long prompt's staged pass, plain routing, the tensor-core expert kernels.
    fn dev_plan_ok(&self, m: usize, infos: &[TensorInfo]) -> bool {
        if std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "2") {
            eprintln!("DEVPLAN? devplan {} sp {} m {} bias {} cpu {} pipe {} accept {} vres {} replay {} rlog {} mma {}", self.devplan, self.sp.is_some(), m, self.router_bias, self.cpu.is_some(),
                      self.pipe_active, self.accept_moe, self.verify_resident, self.replay.on.get(), self.route_log.is_some(), self.mma_experts_ok(infos));
        }
        self.devplan && self.sp.is_some() && m > MAX_VERIFY && m <= BIG_BATCH_MAX && self.router_bias == 0.0 && self.cpu.is_none() && !self.pipe_active
            && !self.replay.on.get() && self.route_log.is_none() && self.mma_experts_ok(infos)
            && self.m.cfg.n_expert <= 512 && self.m.cfg.n_expert_used <= 16
    }

    /// Routes (`route_topk_rows`) and plans (`plan_scan`, `plan_fill`) layer l's m tokens on the device, into `bb.gtab` in the layout `routed_grouped` reads, with room
    /// for the most groups a phase can have; the host sends only the per-expert tables (where each expert is, and whether it is resident), which do not depend on the routing.
    fn dev_plan(&mut self, l: usize, m: usize, sizes: &[usize]) -> GroupedPlan {
        let c = &self.m.cfg;
        let (k, ne) = (c.n_expert_used, c.n_expert);
        let chunk_n = mma_group();
        let g = ne + (m * k).div_ceil(chunk_n);
        let rec = 4 * (1 + 2 * chunk_n);
        let (a0, a1) = (0usize, 3 * g * 8);
        let (r0, r1) = (a1 + 3 * g * 8, a1 + 3 * g * 8 + 3 * g * rec);
        let w = r1 + 3 * g * rec;
        let counts = w + m * k * 4;
        assert!(counts + m * 4 <= self.bb.gtab.bytes, "device plan: {} bytes of tables", counts + m * 4);
        let p = l & 1;
        let arena = self.cache.arena.as_ref().unwrap().ptr;
        let half = self.sp.as_ref().unwrap().base + (p * ne * self.cache.slot_bytes) as u64;
        let part = [0u64, sizes[0] as u64, (sizes[0] + sizes[1]) as u64];
        let mut eaddr: Vec<u8> = Vec::with_capacity(3 * ne * 8);
        let mut copied = vec![0u8; ne.next_multiple_of(4)];
        for pt in 0..3 {
            for e in 0..ne {
                let base = match self.cache.map.get(&(l, e)) {
                    Some(&s) => arena + self.cache.slot_offset[s],
                    None => half + (e * self.cache.slot_bytes) as u64,
                };
                eaddr.extend_from_slice(&(base + part[pt]).to_le_bytes());
            }
        }
        for e in 0..ne {
            copied[e] = !self.cache.map.contains_key(&(l, e)) as u8;
        }
        self.upload_to(self.dp_eaddr.ptr, &eaddr);
        self.upload_to(self.dp_copied.ptr, &copied);
        let (gt, rl) = (self.bb.gtab.ptr, self.bb.rl.ptr);
        let nrec = 2 * 3 * g * (1 + 2 * chunk_n);
        self.k1("zero_words", ne, &[Arg::Ptr(self.dp_cnt.ptr), Arg::U32(ne as u32)]);
        self.k1("zero_words", nrec, &[Arg::Ptr(gt + r0 as u64), Arg::U32(nrec as u32)]);
        assert!(ne <= 512 && k <= 16, "route_topk_rows: {ne} experts, top {k}");
        let route_args = [Arg::Ptr(rl), Arg::Ptr(self.dp_tidx.ptr), Arg::Ptr(gt + w as u64), Arg::Ptr(self.dp_cnt.ptr), Arg::U32(ne as u32), Arg::U32(k as u32), Arg::U32(m as u32)];
        if std::env::var("SHADOW_ROUTE_1T").is_ok_and(|v| v == "1") {
            self.k1("route_topk_rows_1t", m, &route_args);
        } else {
            self.k("route_topk_rows", m.div_ceil(8) as u32, 256, &route_args);
        }

        self.k("plan_scan", 1, 1024, &[Arg::Ptr(self.dp_cnt.ptr), Arg::Ptr(self.dp_copied.ptr), Arg::Ptr(self.dp_gstart.ptr), Arg::U32(ne as u32), Arg::U32(chunk_n as u32)]);
        self.k("plan_fill", ne as u32, 256, &[Arg::Ptr(self.dp_tidx.ptr), Arg::Ptr(self.dp_cnt.ptr), Arg::Ptr(self.dp_gstart.ptr), Arg::Ptr(self.dp_copied.ptr), Arg::Ptr(self.dp_eaddr.ptr),
                                              Arg::Ptr(gt), Arg::U32(ne as u32), Arg::U32(k as u32), Arg::U32(m as u32), Arg::U32(g as u32), Arg::U32((self.rows * k) as u32),
                                              Arg::U32(a0 as u32), Arg::U32(a1 as u32), Arg::U32(r0 as u32), Arg::U32(r1 as u32), Arg::U32(chunk_n as u32)]);
        if std::env::var("SHADOW_DEVPLAN_CHECK").is_ok_and(|v| v == "1") {
            // debugging: the host's routing of the same logits against the device's
            self.ctx.synchronize().unwrap();
            let mut lb = vec![0u8; m * ne * 4];
            self.download(rl, &mut lb);
            let logits = floats(&lb);
            let mut tb = vec![0u8; m * k * 4];
            self.download(self.dp_tidx.ptr, &mut tb);
            let mut wb = vec![0u8; m * k * 4];
            self.download(gt + w as u64, &mut wb);
            let (dw, mut bad_e, mut bad_w) = (floats(&wb), 0usize, 0usize);
            let mut hist = vec![0u32; ne];
            for j in 0..m {
                let top = route(&logits[j * ne..(j + 1) * ne], k);
                for (i, &(e, wt)) in top.iter().enumerate() {
                    let de = u32::from_le_bytes(tb[(j * k + i) * 4..(j * k + i) * 4 + 4].try_into().unwrap()) as usize;
                    hist[e] += 1;
                    bad_e += (de != e) as usize;
                    bad_w += (dw[j * k + i] != wt as f32) as usize;
                }
            }
            let mut cb = vec![0u8; ne * 4];
            self.download(self.dp_cnt.ptr, &mut cb);
            let bad_c = (0..ne).filter(|&e| u32::from_le_bytes(cb[e * 4..e * 4 + 4].try_into().unwrap()) != hist[e]).count();
            eprintln!("DEVPLAN-CHECK L{l} m={m}: experts differing {bad_e}, weights differing {bad_w} of {}, histogram entries differing {bad_c} of {ne}", m * k);
        }
        // the combine's per-token expert counts: all k
        let nc = m;
        self.k1("fill_words", nc, &[Arg::Ptr(gt + counts as u64), Arg::U32(nc as u32), Arg::U32(k as u32)]);
        GroupedPlan { bytes: vec![], phase: [(g, a0 as u64, r0 as u64), (g, a1 as u64, r1 as u64)], weights: w as u64, counts: counts as u64, chunk_n }
    }

    /// `router_done`: `moe_router_batch` already ran (in a graph).
    fn moe_batch(&mut self, l: usize, m: usize, logs: &mut [Vec<u16>], router_done: bool, ticket: Option<usize>) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (h, sff, k, ne) = (c.hidden, c.shared_ff, c.n_expert_used, c.n_expert);
        if !router_done {
            self.moe_router_batch(l, m);
        }
        if self.devplan && self.sp.is_some() && m > MAX_VERIFY {
            let names = [format!("{p}.ffn_gate_exps.weight"), format!("{p}.ffn_up_exps.weight"), format!("{p}.ffn_down_exps.weight")];
            let infos: Vec<TensorInfo> = names.iter().map(|n| self.expert_info(n)).collect();
            if self.dev_plan_ok(m, &infos) {
                let sizes: Vec<usize> = names.iter().map(|n| self.m.expert_bytes(n)).collect();
                {
                    // the shared expert
                    let bb = &self.bb;
                    self.multi_cols(l, 2, false, m);
                    self.k1("swiglu", m * sff, &[Arg::Ptr(bb.sg.ptr), Arg::Ptr(bb.su.ptr), Arg::Ptr(bb.sa.ptr), Arg::U32((m * sff) as u32)]);
                    self.mv_cols(&format!("{p}.ffn_down_shexp.weight"), bb.sa.ptr, sff, bb.so.ptr, h, m);
                }
                self.moe_staged(l, m, None, &infos, &sizes, &names);
                let _ = (logs, ticket, k, ne);
                return;
            }
        }
        let t_wait = std::time::Instant::now();
        // SHADOW_TIMING_NO_SYNC=1 (timing only, wrong output): last time's router logits, no wait for the GPU
        let stale = self.timing_no_sync && self.stale_logits.get(&(l, m)).is_some();
        // the hybrid's CPU experts in a verify pass need the MoE input on the host
        let want_u = self.want_u_rows(m) && !stale;
        let mut u_host: Option<Vec<f32>> = None;
        if self.pipe_active {
            // the copies below reuse slots the previous group's experts may still be reading: the copy stream waits for them
            self.ctx.stream_wait_event(Some(&self.copy_stream), &self.group_event).unwrap();
        }
        let logits: Vec<Vec<f32>> = if let Some(slot) = ticket {
            let (lg, u) = self.wait_router_read(slot, m, want_u);
            u_host = u;
            lg
        } else if stale {
            self.stale_logits[&(l, m)].clone()
        } else {
            let mut srcs = vec![(self.bb.rl.ptr, m * ne)];
            if want_u {
                srcs.push((self.bb.u.ptr, m * h));
            }
            let mut got = self.download_mapped(&srcs);
            if want_u {
                u_host = Some(got.remove(1));
            }
            let all = got.remove(0);
            let lg: Vec<Vec<f32>> = all.chunks(ne).map(|c| c.to_vec()).collect();
            if self.timing_no_sync {
                self.stale_logits.insert((l, m), lg.clone());
            }
            lg
        };
        self.waits[0] += t_wait.elapsed().as_secs_f64();
        let t_woke = std::time::Instant::now();
        self.mark("batch: router + logits down");
        {
            // the shared expert, on the GPU while the host routes and the copies run
            let bb = &self.bb;
            self.multi_cols(l, 2, false, m);
            self.k1("swiglu", m * sff, &[Arg::Ptr(bb.sg.ptr), Arg::Ptr(bb.su.ptr), Arg::Ptr(bb.sa.ptr), Arg::U32((m * sff) as u32)]);
            self.mv_cols(&format!("{p}.ffn_down_shexp.weight"), bb.sa.ptr, sff, bb.so.ptr, h, m);
        }
        let t_routed0 = t_woke.elapsed();
        let mut tops: Vec<Vec<(usize, f64)>> = route_all(&logits, k, if self.router_bias != 0.0 { Some((&self.cache.map, l, self.router_bias)) } else { None });
        for (j, top) in tops.iter().enumerate() {
            for &(e, _) in top {
                self.routing.add(l, e);
            }
            if self.route_log.is_some() {
                logs[j].push(l as u16);
                logs[j].extend(top.iter().map(|&(e, _)| e as u16));
            }
        }
        if self.accept_moe && self.replay.on.get() && m >= 2 {
            self.accept_route(l, &logits, &mut tops);
        }
        if self.verify_resident && !self.accept_moe && self.replay.on.get() {
            // a verify pass: the drafted positions keep only their resident experts
            for top in tops.iter_mut().skip(1) {
                let before = top.len();
                let mut rank = 0;
                top.retain(|&(e, _)| {
                    rank += 1;
                    rank <= self.verify_rank || self.cache.map.contains_key(&(l, e))
                });
                self.tail_skipped += (before - top.len()) as u64;
            }
        }
        if self.tail_skip > 0 && m <= MAX_VERIFY {
            // --tail-skip over the whole pass: an expert not in VRAM that every token routes only at rank >= R
            let mut low: HashMap<usize, bool> = HashMap::new();
            for top in &tops {
                for (i, &(e, _)) in top.iter().enumerate() {
                    let tail = low.entry(e).or_insert(true);
                    *tail &= i >= self.tail_skip;
                }
            }
            for top in tops.iter_mut() {
                let before = top.len();
                top.retain(|&(e, _)| !low[&e] || self.cache.map.contains_key(&(l, e)));
                self.tail_skipped += (before - top.len()) as u64;
            }
        }
        let names = [format!("{p}.ffn_gate_exps.weight"), format!("{p}.ffn_up_exps.weight"), format!("{p}.ffn_down_exps.weight")];
        let infos: Vec<TensorInfo> = names.iter().map(|n| self.expert_info(n)).collect();
        let sizes: Vec<usize> = names.iter().map(|n| self.m.expert_bytes(n)).collect();
        // Hybrid MoE in a verify pass: up to `cpu_experts` of the layer's cache misses that the RAM tier holds and that
        // ONE token routes to (the CPU computes an expert per token; a copy serves every token that uses it) are computed
        // on the CPU from the pinned bytes while the GPU runs the resident experts and the copies run.  Their rank rows of
        // eo are filled from the host before the combine; tops keeps them, so the combine weights them as usual.
        let mut cpu_skip: Vec<usize> = vec![];
        let mut cpu_pairs: Vec<(usize, usize)> = vec![];
        let mut cpu_offs: Vec<usize> = vec![];
        let mut cpu_pending = false;
        let t_plan = std::time::Instant::now();
        let mut cpu_submit = std::time::Instant::now();
        if let (Some(ub), Some(max)) = (u_host.as_ref(), self.cpu.as_ref().map(|c| c.1)) {
            if infos[0].ty == infos[1].ty && infos[0].dims == infos[1].dims && self.ram.region.is_some() {
                let mut users: HashMap<usize, Vec<(usize, usize, f64)>> = HashMap::new();
                for (j, top) in tops.iter().enumerate() {
                    for (i, &(e, w)) in top.iter().enumerate() {
                        users.entry(e).or_default().push((j, i, w));
                    }
                }
                // (weight, expert, token, rank, offset in the RAM tier), lowest weight first
                let mut cand: Vec<(f64, usize, usize, usize, usize)> = vec![];
                for (&e, us) in &users {
                    if us.len() == 1 && !self.cache.map.contains_key(&(l, e)) {
                        if let Some(&off) = self.ram.at.get(&(l, e)) {
                            cand.push((us[0].2, e, us[0].0, us[0].1, off));
                        }
                    }
                }
                cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
                cand.truncate(max);
                if !cand.is_empty() {
                    let mut toks: Vec<usize> = cand.iter().map(|c| c.2).collect();
                    toks.sort_unstable();
                    toks.dedup();
                    let xs: Vec<f32> = toks.iter().flat_map(|&j| ub[j * h..(j + 1) * h].iter().copied()).collect();
                    let xidx: Vec<usize> = cand.iter().map(|c| toks.iter().position(|&j| j == c.2).unwrap()).collect();
                    let base = self.ram.region.as_ref().unwrap().as_slice().as_ptr();
                    let refs: Vec<shadow_model::cpu_expert::ExpertRef> = cand.iter().map(|c| {
                        // SAFETY: the RAM tier holds this expert's [gate | up | down] at its offset, inside the region.
                        unsafe {
                            shadow_model::cpu_expert::ExpertRef { gate: base.add(c.4), up: base.add(c.4 + sizes[0]),
                                                                  down: base.add(c.4 + sizes[0] + sizes[1]) }
                        }
                    }).collect();
                    for c in &cand {
                        cpu_skip.push(c.1);
                        cpu_pairs.push((c.2, c.3));
                        cpu_offs.push(c.4);
                    }
                    let shape = shadow_model::cpu_expert::ExpertShape {
                        hidden: h, ff: self.m.cfg.expert_ff, gate_ty: infos[0].ty.name(), up_ty: infos[1].ty.name(),
                        down_ty: infos[2].ty.name(),
                    };
                    // SAFETY: the region lives as long as the session, and nothing rewrites these experts' bytes before
                    // the `wait` below: the group loop re-claims their ring slots after every `claimed.clear()`.
                    unsafe { self.cpu.as_mut().unwrap().0.submit_multi(&shape, refs, xidx, &xs) };
                    self.cpu_served += cpu_pairs.len() as u64;
                    cpu_pending = true;
                    cpu_submit = std::time::Instant::now();
                }
            }
        }
        if cpu_pending {
            self.cpu_host.0 += t_plan.elapsed().as_secs_f64();
        }
        if self.sp.is_some() && m > MAX_VERIFY && !cpu_pending {
            let t_pre = t_woke.elapsed();
            self.moe_staged(l, m, Some(&tops), &infos, &sizes, &names);
            if m >= 2048 && std::env::var("SHADOW_STAGE_TIME").is_ok_and(|v| v == "1") {
                eprintln!("HOST L{l} m={m}: after the logits arrived: conversion {:.1} ms, route+bookkeeping+plan prep {:.1} ms, moe_staged {:.1} ms (GPU has only the shared expert meanwhile)",
                          t_routed0.as_secs_f64() * 1e3, (t_pre - t_routed0).as_secs_f64() * 1e3, (t_woke.elapsed() - t_pre).as_secs_f64() * 1e3);
            }
            return;
        }
        // misses read from the file may wait in the staging buffer, a slot each
        let max_file = self.staging.as_ref().map_or(usize::MAX, |s| s.len() / self.cache.slot_bytes.max(1));
        let scratch = match self.lent.as_ref() {
            Some(l) => (l.ptr, l.n),
            None => self.batch_scratch.as_ref().map_or((0, 0), |(b, n)| (b.ptr, *n)),
        };
        let arena = self.cache.arena.as_ref().unwrap().ptr;
        let mut start = 0;
        while start < m {
            let mut set: Vec<usize> = Vec::new();
            let mut end = start;
            while end < m {
                let mut cand = set.clone();
                for &(e, _) in &tops[end] {
                    if !cand.contains(&e) && !cpu_skip.contains(&e) {
                        cand.push(e);
                    }
                }
                let missing = || cand.iter().filter(|&&e| !self.cache.map.contains_key(&(l, e)));
                let file = missing().filter(|&&e| !self.ram.at.contains_key(&(l, e))).count();
                let free = self.cache.evict_range[l].clone()
                    .filter(|&i| self.cache.owner[i].is_none_or(|(ol, oe)| ol != l || !cand.contains(&oe)))
                    .count();
                if end > start && (missing().count() > free + scratch.1 || file > max_file) {
                    break;
                }
                set = cand;
                end += 1;
            }
            if start > 0 {
                // the previous group's matvecs may still be reading slots this group's misses take
                self.ctx.synchronize().unwrap();
            }
            self.ram.claimed.clear(); // the previous group's copies were waited for
            if cpu_pending {
                for &off in &cpu_offs {
                    if let Some(&r) = self.ram.ring_by_off.get(&off) {
                        self.ram.ring_ref[r] = true; // keep the Clock hand off the CPU's experts
                    }
                    self.ram.claimed.insert(off);
                }
            }
            self.cache.clock += 1;
            for &e in &set {
                self.cache.touch((l, e));
            }
            // the group's misses get their slots (cache victims, else scratch) first
            let mut copying: Vec<(usize, u64)> = Vec::new();
            let mut extra: Vec<(usize, u64)> = Vec::new();
            for &e in &set {
                if self.cache.map.contains_key(&(l, e)) {
                    self.cache.hits += 1;
                    continue;
                }
                self.cache.misses += 1;
                let times = if self.cache.admit_after > 0 {
                    let n = self.cache.seen.entry((l, e)).or_insert(0);
                    *n += 1;
                    *n as usize
                } else {
                    usize::MAX
                };
                let cold = times <= self.cache.admit_after && extra.len() < scratch.1;
                if cold {
                    self.cache.scratch_reads += 1;
                }
                let base = if !cold && self.cache.has_victim(l) {
                    let s = self.cache.victim(l);
                    self.cache.owner[s] = Some((l, e));
                    self.cache.map.insert((l, e), s);
                    self.cache.stamp[s] = self.cache.clock;
                    arena + self.cache.slot_offset[s]
                } else {
                    let b = scratch.0 + (extra.len() * self.cache.slot_bytes) as u64;
                    extra.push((e, b));
                    b
                };
                copying.push((e, base));
            }
            // grouped and overlapped: each union expert once, on all of the group's tokens that route to it - the
            // resident ones while the misses copy, then (the GPU waiting for the copies) the copied ones
            let overlap = self.overlap && self.prof.is_none() && infos[0].ty == infos[1].ty && infos[0].dims == infos[1].dims;
            let part = [0u64, sizes[0] as u64, (sizes[0] + sizes[1]) as u64];
            let mut plan = None;
            if overlap {
                let addr = |e: usize| match self.cache.map.get(&(l, e)) {
                    Some(&s) => arena + self.cache.slot_offset[s],
                    None => extra.iter().find(|x| x.0 == e).expect("a routed expert neither cached nor fetched").1,
                };
                let copied = |e: usize| copying.iter().any(|c| c.0 == e);
                let chunk_n = if end - start >= 64 && self.mma_experts_ok(&infos) { mma_group() } else { 8 };
                let p = grouped_plan(&tops, start, end, &set, k, self.rows * k, addr, copied, part, chunk_n);
                assert!(p.bytes.len() <= self.bb.gtab.bytes, "grouped experts: {} bytes of tables", p.bytes.len());
                self.upload_to(self.bb.gtab.ptr, &p.bytes);
                if start == 0 {
                    self.plan_lat.0 += 1;
                    self.plan_lat.1 += t_woke.elapsed().as_secs_f64();
                }
                self.routed_grouped(&p, 0, start, end, &infos);
                plan = Some(p);
            }
            // the group's RAM-tier misses keep their ring slots: a promotion for one of its file reads must not
            // evict an expert the group is about to copy (it would turn into a file read past the staging slots)
            for &(e, _) in &copying {
                if let Some(&off) = self.ram.at.get(&(l, e)).filter(|off| self.ram.ring_by_off.contains_key(off)) {
                    self.ram.claimed.insert(off);
                }
            }
            let mut n_file = 0;
            if self.pipe_active && copying.iter().any(|&(e, _)| !self.ram.at.contains_key(&(l, e))) {
                // a file read may reuse a pinned slot an earlier group's copy is still reading
                self.copy_stream.synchronize().unwrap();
            }
            for &(e, base) in &copying {
                let from_file = !self.ram.at.contains_key(&(l, e));
                self.fetch_expert(l, e, n_file, &names, &sizes, [base, base + part[1], base + part[2]]);
                n_file += from_file as usize;
            }
            self.read_drive(&names, &sizes);
            if cpu_pending {
                let t_join = std::time::Instant::now();
                let out = self.cpu.as_mut().unwrap().0.wait();
                self.cpu_join.0 += 1;
                self.cpu_join.1 += t_join.elapsed().as_secs_f64();
                self.cpu_join.2 += cpu_submit.elapsed().as_secs_f64();
                let t_up = std::time::Instant::now();
                for (n, &(j, i)) in cpu_pairs.iter().enumerate() {
                    self.upload_to(row(&self.bb.eo, k * h, j) + (i * h * 4) as u64, &f32s(&out[n * h..(n + 1) * h]));
                }
                self.cpu_host.1 += t_up.elapsed().as_secs_f64();
                cpu_pending = false;
            }
            if let Some(p) = plan.as_ref() {
                if !copying.is_empty() {
                    self.copy_event.record(Some(&self.copy_stream)).unwrap();
                    self.ctx.stream_wait_event(None, &self.copy_event).unwrap();
                }
                self.routed_grouped(p, 1, start, end, &infos);
                let bb = &self.bb;
                self.k2("moe_combine_qwen_rows", (h as u32).div_ceil(256), (end - start) as u32, 256, &[
                    Arg::Ptr(row(&bb.y, h, start)), Arg::Ptr(row(&bb.eo, k * h, start)),
                    Arg::Ptr(bb.gtab.ptr + p.weights + (start * k * 4) as u64), Arg::Ptr(row(&bb.so, h, start)),
                    Arg::Ptr(row(&bb.sgl, 1, start)), Arg::Ptr(bb.gtab.ptr + p.counts + (start * 4) as u64),
                    Arg::U32(h as u32), Arg::U32(k as u32)]);
            } else {
                if !copying.is_empty() {
                    let t_wait = std::time::Instant::now();
                    self.copy_stream.synchronize().unwrap();
                    self.waits[1] += t_wait.elapsed().as_secs_f64();
                }
                self.mark("batch: shared expert + expert fetch");
                for (j, top) in tops.iter().enumerate().take(end).skip(start) {
                    self.routed_qwen(l, j, top, &extra, &infos, &sizes);
                }
            }
            start = end;
        }
        if self.pipe_active {
            self.group_event.record(None).unwrap();
        }
    }

    /// Whether a layer's three expert matrices can run on the tensor cores (`--tensor-prompt`: types with an FP16 decode, columns a multiple of 64).
    fn mma_experts_ok(&self, infos: &[TensorInfo]) -> bool {
        self.tensor_prompt
            && infos.iter().all(|t| matches!(t.ty.name().as_str(), "Q6_K" | "IQ4_XS" | "IQ4_NL" | "Q2_0" | "IQ2_S" | "IQ3_XXS" | "IQ3_S")
                                    && t.dims[0] % 64 == 0 && t.dims[1] % 8 == 0)
    }

    /// Phase `ph` (0: the resident experts, 1: the copied ones) of `grouped_plan`'s routed experts for tokens
    /// start..end: gate+up grouped, swiglu over the tokens' rows, down grouped, into bb.eo row (token, rank).
    fn routed_grouped(&self, p: &GroupedPlan, ph: usize, start: usize, end: usize, infos: &[TensorInfo]) {
        let c = &self.m.cfg;
        let (h, ff, k) = (c.hidden, c.expert_ff, c.n_expert_used);
        let bb = &self.bb;
        let (n, w, g) = p.phase[ph];
        if n == 0 {
            return;
        }
        let up = (self.rows * k * ff * 4) as u64;
        let grouped = |t: &TensorInfo, wt: u64, gt: u64, count: usize, x: u64, xs: usize, y: u64, ys: usize| {
            let ty = t.ty.name().to_lowercase();
            let (rows, cols) = (t.dims[1] as u32, t.dims[0] as u32);
            // SAFETY: `count` table entries at gtab + wt (matrix addresses) and + gt (17 u32 each: count, input rows,
            // output rows; 65 for the 32-token records), each a rows x cols matrix in the cache or the scratch slots; the rows
            // they name are inside the batch's buffers at the strides given.
            let (kernel, grid, block) = if p.chunk_n >= 32 {
                (if p.chunk_n == 64 { format!("mma_grouped64_{ty}") } else { format!("mma_grouped_{ty}") }, (rows.div_ceil(64), count as u32, 1), 128)
            } else {
                // 8 rows a block: matvec.cu's GROUPED_RW = 1
                (format!("matvec_grouped_{ty}"), (rows.div_ceil(8), count as u32, 1), 256)
            };
            unsafe {
                self.mvp.launch(&kernel, grid, (block, 1, 1), 0, self.ls(),
                                &[Arg::Ptr(bb.gtab.ptr + wt), Arg::Ptr(bb.gtab.ptr + gt), Arg::Ptr(x), Arg::U32(xs as u32),
                                  Arg::Ptr(y), Arg::U32(ys as u32), Arg::U32(rows), Arg::U32(cols)])
            }
            .unwrap_or_else(|err| panic!("{}: {err}", t.name));
        };
        grouped(&infos[0], w, g, 2 * n, bb.u.ptr, h, bb.eg.ptr, ff);
        let rows = (start * k * ff * 4) as u64;
        let len = (end - start) * k * ff;
        self.k1("swiglu", len, &[Arg::Ptr(bb.eg.ptr + rows), Arg::Ptr(bb.eg.ptr + up + rows), Arg::Ptr(bb.act.ptr + rows),
                                 Arg::U32(len as u32)]);
        grouped(&infos[2], w + 16 * n as u64, g + (2 * n * 4 * (1 + 2 * p.chunk_n)) as u64, n, bb.act.ptr, ff, bb.eo.ptr, h);
    }

    /// Token j's routed experts (resident in the cache, or at `extra`'s scratch addresses) and the combine with its
    /// shared expert, into its row of y.
    #[allow(clippy::too_many_arguments)]
    fn routed_qwen(&self, l: usize, j: usize, top: &[(usize, f64)], extra: &[(usize, u64)], infos: &[TensorInfo],
                   sizes: &[usize]) {
        let c = &self.m.cfg;
        let (h, ff) = (c.hidden, c.expert_ff);
        let n = top.len();
        let arena = self.cache.arena.as_ref().unwrap().ptr;
        let at: Vec<[u64; 3]> = top.iter().map(|&(e, _)| {
            let base = match self.cache.map.get(&(l, e)) {
                Some(&s) => arena + self.cache.slot_offset[s],
                None => extra.iter().find(|x| x.0 == e).expect("a routed expert neither cached nor fetched").1,
            };
            [base, base + sizes[0] as u64, base + (sizes[0] + sizes[1]) as u64]
        }).collect();
        // the layout `moe` uses: gate, up and down addresses, then the weights
        let mut tab = Vec::with_capacity(n * 28);
        for part in 0..3 {
            for a in &at {
                tab.extend_from_slice(&a[part].to_le_bytes());
            }
        }
        for &(_, w) in top {
            tab.extend_from_slice(&(w as f32).to_le_bytes());
        }
        self.upload_to(self.b.tab.ptr, &tab);
        let b = &self.b;
        let u = row(&self.bb.u, h, j);
        let eu = if infos[0].ty == infos[1].ty && infos[0].dims == infos[1].dims {
            self.mv_batch(&infos[0], b.tab.ptr, 2 * n, u, 0, b.eg.ptr, ff);
            b.eg.ptr + (n * ff * 4) as u64
        } else {
            self.mv_batch(&infos[0], b.tab.ptr, n, u, 0, b.eg.ptr, ff);
            self.mv_batch(&infos[1], b.tab.ptr + 8 * n as u64, n, u, 0, b.eu.ptr, ff);
            b.eu.ptr
        };
        self.k1("swiglu", n * ff, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(eu), Arg::Ptr(b.act.ptr), Arg::U32((n * ff) as u32)]);
        self.mv_batch(&infos[2], b.tab.ptr + 16 * n as u64, n, b.act.ptr, ff, b.eo.ptr, h);
        self.k1("moe_combine_qwen", h, &[
            Arg::Ptr(row(&self.bb.y, h, j)), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.tab.ptr + 24 * n as u64),
            Arg::Ptr(row(&self.bb.so, h, j)), Arg::Ptr(row(&self.bb.sgl, 1, j)), Arg::U32(h as u32), Arg::U32(n as u32)]);
    }

    /// One token's routed experts with their copies overlapped: the resident ones run at once, the misses' copies go
    /// out on copy_stream, and the default stream - not the host - waits for them before the misses run, so the host
    /// goes on queueing the layer meanwhile.  eo's rows come out resident first; the combine reads them in routing
    /// order (`perm`), so y is `moe`'s to the bit.
    /// The last `ncpu` entries of `top` are the CPU's (submitted to the pool): their outputs come into eo after the
    /// GPU's, before the combine.
    fn routed_overlapped(&mut self, l: usize, top: &[(usize, f64)], ncpu: usize, names: &[String; 3], infos: &[TensorInfo],
                         sizes: &[usize]) {
        let all = top;
        let top = &all[..all.len() - ncpu];
        let c = &self.m.cfg;
        let (h, ff, n) = (c.hidden, c.expert_ff, top.len());
        let arena = self.cache.arena.as_ref().unwrap().ptr;
        self.cache.clock += 1;
        for &(e, _) in top {
            self.cache.touch((l, e)); // protect this token's resident experts from its own misses
        }
        // (rank in `top`, expert, slot address)
        let (mut hits, mut misses): (Vec<(usize, usize, u64)>, Vec<(usize, usize, u64)>) = (vec![], vec![]);
        for (i, &(e, _)) in top.iter().enumerate() {
            match self.cache.touch((l, e)) {
                Some(s) => {
                    self.cache.hits += 1;
                    hits.push((i, e, arena + self.cache.slot_offset[s]));
                }
                None if self.zero_copy && self.ram.at.contains_key(&(l, e)) => {
                    // read in place over PCIe by the matvec kernels: no copy, no cache slot
                    let off = self.ram.at[&(l, e)];
                    self.ram.zc_reads += 1;
                    if let Some(&r) = self.ram.ring_by_off.get(&off) {
                        self.ram.ring_ref[r] = true;
                    }
                    self.ram.claimed.insert(off);
                    hits.push((i, e, self.ram.region.as_ref().unwrap().dev_ptr + off as u64));
                }
                None => {
                    self.cache.misses += 1;
                    let s = self.cache.victim(l);
                    self.cache.owner[s] = Some((l, e));
                    self.cache.map.insert((l, e), s);
                    self.cache.stamp[s] = self.cache.clock;
                    misses.push((i, e, arena + self.cache.slot_offset[s]));
                }
            }
        }
        let nh = hits.len();
        let part = [0u64, sizes[0] as u64, (sizes[0] + sizes[1]) as u64];
        // [gate, up] of the resident ones, [gate, up] of the misses, down in that order, weights in routing order, perm
        let mut tab = Vec::with_capacity(n * 32);
        for group in [&hits, &misses] {
            for p in &part[..2] {
                for x in group.iter() {
                    tab.extend_from_slice(&(x.2 + p).to_le_bytes());
                }
            }
        }
        let mut perm: Vec<u32> = (0..all.len() as u32).collect(); // the CPU's rows follow the GPU's, in their order
        for (row, x) in hits.iter().chain(&misses).enumerate() {
            tab.extend_from_slice(&(x.2 + part[2]).to_le_bytes());
            perm[x.0] = row as u32;
        }
        let na = all.len();
        for &(_, w) in all {
            tab.extend_from_slice(&(w as f32).to_le_bytes());
        }
        tab.extend(perm.iter().flat_map(|p| p.to_le_bytes()));
        self.upload_to(self.b.tab.ptr, &tab);
        // rows first..first + cnt of eo: gate+up, swiglu, down
        let run = |s: &Self, first: usize, cnt: usize| {
            if cnt == 0 {
                return;
            }
            let b = &s.b;
            let eg = b.eg.ptr + (2 * first * ff * 4) as u64;
            let act = b.act.ptr + (first * ff * 4) as u64;
            s.mv_batch(&infos[0], b.tab.ptr + 16 * first as u64, 2 * cnt, b.u.ptr, 0, eg, ff);
            s.k1("swiglu", cnt * ff, &[Arg::Ptr(eg), Arg::Ptr(eg + (cnt * ff * 4) as u64), Arg::Ptr(act),
                                      Arg::U32((cnt * ff) as u32)]);
            s.mv_batch(&infos[2], b.tab.ptr + (16 * n + 8 * first) as u64, cnt, act, ff, b.eo.ptr + (first * h * 4) as u64, h);
        };
        run(self, 0, nh);
        for &(i, e, base) in &misses {
            self.fetch_expert(l, e, i, names, sizes, [base, base + part[1], base + part[2]]);
        }
        self.read_drive(names, sizes);
        if !misses.is_empty() {
            self.copy_event.record(Some(&self.copy_stream)).unwrap();
            self.ctx.stream_wait_event(None, &self.copy_event).unwrap();
        }
        run(self, nh, misses.len());
        if ncpu > 0 {
            let out = self.cpu.as_mut().unwrap().0.wait();
            self.upload_to(self.b.eo.ptr + ((na - ncpu) * h * 4) as u64, &f32s(&out));
        }
        let b = &self.b;
        self.k1("moe_combine_qwen_perm", h, &[
            Arg::Ptr(b.y.ptr), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.tab.ptr + 24 * n as u64), Arg::Ptr(b.tab.ptr + (24 * n + 4 * na) as u64),
            // the shared expert's output is row n_expert_used of eo (`moe` launches it there), whatever n is
            Arg::Ptr(b.eo.ptr + (c.n_expert_used * h * 4) as u64), Arg::Ptr(b.sgl.ptr), Arg::U32(h as u32), Arg::U32(na as u32)]);
    }

    /// The n-gram memory at layer l, into gv (the caller adds it to r).
    fn ple(&self, l: usize) {
        self.ple_at(l, self.tokens.len(), self.b.r.ptr);
    }

    /// The n-gram memory at layer l for the token at position n - 1 with streams r, into gv.
    fn ple_at(&self, l: usize, n: usize, r: u64) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (h, cs) = (c.hidden, c.streams);
        let ch = h * cs;
        let t_ple = std::time::Instant::now();
        let rows = ple_rows(c, &self.tokens[..n]);
        self.m.ple_prefetch(&rows);
        let e: Vec<f32> = rows.into_iter().flat_map(|row| self.m.ple_row(row)).collect();
        self.ple_host.set(self.ple_host.get() + t_ple.elapsed().as_secs_f64());
        self.ple_calls.set(self.ple_calls.get() + 1);
        let b = &self.b;
        self.upload_to(b.pe.ptr, &f32s(&e));
        self.mv_pair(&format!("{p}.ple_key.weight"), b.key.ptr, &format!("{p}.ple_value.weight"), b.val.ptr, b.pe.ptr);
        self.k("rmsnorm_groups", cs as u32, 256, &[Arg::Ptr(b.key.ptr), Arg::Ptr(self.sv(&format!("{p}.ple_norm_key.weight"))),
                                                  Arg::Ptr(b.keyn.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
        self.k("rmsnorm_groups", cs as u32, 256, &[Arg::Ptr(r), Arg::Ptr(self.sv(&format!("{p}.ple_norm_query.weight"))),
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

impl Drop for GpuSession<'_, '_> {
    fn drop(&mut self) {
        // little-endian: u32 token count, the tokens (u32), u32 record width k, then (layer, k expert ids) as u16s
        if let Some((path, log)) = self.route_log.take() {
            let k = self.m.cfg.n_expert_used as u32;
            let mut b: Vec<u8> = (self.tokens.len() as u32).to_le_bytes().to_vec();
            b.extend(self.tokens.iter().flat_map(|t| t.to_le_bytes()));
            b.extend(k.to_le_bytes());
            b.extend(log.iter().flat_map(|v| v.to_le_bytes()));
            if let Err(e) = std::fs::write(&path, b) {
                eprintln!("SHADOW_ROUTE_LOG {path}: {e}");
            }
        }
    }
}

impl Stepper for GpuSession<'_, '_> {
    fn step(&mut self, token: u32) -> Vec<f32> {
        GpuSession::step(self, token)
    }
    fn step_prefill(&mut self, token: u32) {
        GpuSession::step_prefill(self, token);
    }
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.logits_last = !std::env::var("SHADOW_LOGITS_LAST").is_ok_and(|v| v == "0");
        let out = self.step_batch(tokens).pop().unwrap_or_default();
        self.logits_last = false;
        out
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
    fn routing(&self) -> Option<&RoutingCounts> {
        Some(&self.routing)
    }
    fn report(&self) -> String {
        let (slots, bytes, hits, misses) = self.cache_stats();
        let arena_gib = self.cache.arena_bytes as f64 / (1u64 << 30) as f64;
        let avg_mib = if slots > 0 { (self.cache.arena_bytes as f64 / slots as f64) / (1u64 << 20) as f64 } else { bytes as f64 / (1u64 << 20) as f64 };
        format!("expert cache: {slots} slots x {avg_mib:.2} MiB = {arena_gib:.2} GiB; {hits} hits, {misses} misses ({:.1}% hits); {} slice uploads",
                100.0 * hits as f64 / (hits + misses).max(1) as f64, self.expert_uploads)
            + &(if std::env::var("SHADOW_VRAM_MAP").is_ok() { format!("; vram free now {} MiB", self.ctx.mem_info().map_or(0, |x| x.0 >> 20)) } else { String::new() })
            + &(if self.cache.admit_after > 0 { format!("; admit-after {}: {} misses kept out of the cache", self.cache.admit_after, self.cache.scratch_reads) } else { String::new() })
            + &format!("; RAM tier: {} experts, {:.2} GiB pinned (filled in {:.1} s), misses served {} from RAM, {} from the file",
                       self.ram.at.len(), self.ram.region.as_ref().map_or(0, |r| r.len()) as f64 / (1u64 << 30) as f64,
                       self.ram.fill_secs, self.ram.served, self.ram.drive)
            + &self.cpu.as_ref().map(|c| format!("; CPU experts: up to {} per layer, {} computed", c.1, self.cpu_served))
                .unwrap_or_default()
            + &(if self.pipeline { format!("; pipelined verify passes: {}", self.pipelined_passes) } else { String::new() })
            + &(if self.plan_lat.0 > 0 {
                format!("; MoE host latency, router logits received to first expert kernel queued: {:.0} us a layer",
                        1e6 * self.plan_lat.1 / self.plan_lat.0 as f64)
            } else { String::new() })
            + &self.cpu.as_ref().filter(|_| self.cpu_join.0 > 0).map(|_| {
                let (n, join, job) = self.cpu_join;
                format!("; verify-pass CPU joins: {n}, host waited {:.0} us each, submit-to-result {:.0} us each, choosing and submitting {:.0} us, uploading the outputs {:.0} us",
                        1e6 * join / n as f64, 1e6 * job / n as f64, 1e6 * self.cpu_host.0 / n as f64,
                        1e6 * self.cpu_host.1 / n as f64)
            }).unwrap_or_default()
            + &{
                let v = &self.verify_stats;
                if v[0] > 0.0 {
                    let n = v[0];
                    format!("; verify passes {:.0} ({:.1} tokens): {:.1} ms each, GPU waits {:.1} ms, file {:.1} ms, launches {:.0}                              ({:.1} ms of host)", n, v[1] / n, 1e3 * v[2] / n, 1e3 * v[3] / n, 1e3 * v[4] / n, v[6] / n, 1e3 * v[5] / n)
                } else {
                    String::new()
                }
            }
            + &{
                match self.ltime.as_ref() {
                    Some(l) if l.borrow().len() > 1 => {
                        let _ = self.ctx.synchronize();
                        let v = l.borrow();
                        let mut sum: Vec<(&str, f64, usize)> = vec![];
                        for w in v.windows(2) {
                            if w[1].0 == "(start)" {
                                continue;
                            }
                            let ms = w[1].1.since(&w[0].1).unwrap_or(0.0) as f64;
                            match sum.iter_mut().find(|x| x.0 == w[1].0) {
                                Some(x) => { x.1 += ms; x.2 += 1; }
                                None => sum.push((w[1].0, ms, 1)),
                            }
                        }
                        let passes = v.iter().filter(|x| x.0 == "(start)").count().max(1) as f64;
                        format!("; per batched pass: {}", sum.iter().map(|(n, ms, c)| format!("{n} {:.1} ms ({:.0}x)", ms / passes, *c as f64 / passes))
                            .collect::<Vec<_>>().join(", "))
                    }
                    _ => String::new(),
                }
            }
            + &(if self.ple_calls.get() > 0 {
                format!("; n-gram rows: {:.2} ms a token on the host", 1e3 * self.ple_host.get() / self.ple_calls.get() as f64)
            } else {
                String::new()
            })
            + &(if self.zero_copy { format!("; {} experts read in place over PCIe", self.ram.zc_reads) } else { String::new() })
            + &(if self.tail_skip > 0 { format!("; tail skip (rank >= {}): {} routed experts skipped", self.tail_skip, self.tail_skipped) } else { String::new() })
            + &format!("; host waits per step: router {:.1} ms, copies {:.1} ms, file {:.1} ms",
                       1e3 * self.waits[0] / self.tokens.len().max(1) as f64, 1e3 * self.waits[1] / self.tokens.len().max(1) as f64,
                       1e3 * self.waits[2] / self.tokens.len().max(1) as f64)
            + &{
                use std::sync::atomic::Ordering::Relaxed;
                let n = self.tokens.len().max(1) as f64;
                format!("; kernel launches per step {:.0}, {:.1} ms of host time in them",
                        crate::kernels::LAUNCHES.0.load(Relaxed) as f64 / n, crate::kernels::LAUNCHES.1.load(Relaxed) as f64 / 1e6 / n)
            }
            + &self.prefetch.as_ref().map(|p| format!("; pre-gated prefetch: top {}, {} copied ahead, {} used ({:.1}%)", p.k,
                                                       p.issued, p.used, 100.0 * p.used as f64 / p.issued.max(1) as f64))
                .unwrap_or_default()
            + &self.pregate.as_ref().map(|p| {
                let t = p.counts.iter().fold([0u64; 5], |a, k| std::array::from_fn(|i| a[i] + k[i]));
                let pc = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
                format!("; pre-gating probe: top-10 {:.1}%, top-16 {:.1}% of routed experts; {:.1}% of misses in the top 16                          ({} misses)", pc(t[1], t[0]), pc(t[2], t[0]), pc(t[4], t[3]), t[3])
            }).unwrap_or_default()
            + &self.mtp_probe.map(|(n, hit)| format!("; MTP probe: {hit}/{n} drafts right ({:.1}%)",
                                                       100.0 * hit as f64 / n.max(1) as f64)).unwrap_or_default()
            + if self.mtp.is_some() { "; MTP: active (speculative decoding with tuned basalt head)" } else { "" }
            + &{
                let mut kt: Vec<_> = self.mvp.kernel_times().into_iter().chain(self.fw.kernel_times()).collect();
                kt.sort_by(|a, b| b.1.total_cmp(&a.1));
                let n = self.tokens.len().max(1) as f64;
                if kt.is_empty() { String::new() } else {
                    let total: f64 = kt.iter().map(|k| k.1).sum();
                    format!("
  GPU kernel time per token {:.2} ms: {}", total / n, kt.iter().take(16)
                        .map(|(k, ms, c)| format!("{k} {:.2} ms ({:.0}x)", ms / n, *c as f64 / n)).collect::<Vec<_>>().join(", "))
                }
            }
            + &self.prof.as_ref().map(|p| {
                let n = self.tokens.len().max(1) as f64;
                let total: f64 = p.phases.iter().map(|x| x.1).sum();
                let parts: Vec<String> = p.phases.iter()
                    .map(|(name, t)| format!("{name} {:.1} ms ({:.0}%)", 1e3 * t / n, 100.0 * t / total.max(1e-9)))
                    .collect();
                format!("\n  profile over {} steps, {:.1} ms/step (synchronized): {}", n, 1e3 * total / n, parts.join(", "))
            }).unwrap_or_default()
    }
}
