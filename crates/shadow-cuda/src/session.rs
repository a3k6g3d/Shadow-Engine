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
use shadow_model::forward::{ple_rows, route, route_sigmoid, Model, Stepper};
use shadow_model::gguf::TensorInfo;
use shadow_model::placement::{uniform_order, RoutingCounts};
use shadow_model::quant::dequantize;
use std::collections::{BTreeMap, HashMap};

struct Attn<'c> {
    k: DeviceBuffer<'c>,
    v: DeviceBuffer<'c>,
    raw: DeviceBuffer<'c>,
    blocks: DeviceBuffer<'c>,
    nblocks: usize,
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
pub const MAX_BATCH: usize = 64;
const COLS_MAX: usize = 8;

/// Expert slots outside the cache for a batched step's misses beyond the layer's free cache slots (a layer keeps only
/// a dozen evictable slots next to its pinned hot experts, and a prompt misses most of its experts); `SHADOW_BATCH_SCRATCH`
/// overrides it.
const BATCH_SCRATCH_SLOTS: usize = 32;

/// `step_batch`'s per-token intermediates, MAX_BATCH rows each, row j of a buffer at ptr + j * its stride: what goes
/// into or comes out of a projection all m tokens share (the rest stay in `Bufs`, used one token at a time).
struct BatchBufs<'c> {
    r: DeviceBuffer<'c>,
    rn: DeviceBuffer<'c>,
    u: DeviceBuffer<'c>,
    g: DeviceBuffer<'c>,
    y: DeviceBuffer<'c>,
    down: DeviceBuffer<'c>,
    up: DeviceBuffer<'c>,
    qkv: DeviceBuffer<'c>,
    z: DeviceBuffer<'c>,
    beta: DeviceBuffer<'c>,
    alpha: DeviceBuffer<'c>,
    o: DeviceBuffer<'c>,
    qg: DeviceBuffer<'c>,
    kk: DeviceBuffer<'c>,
    iq: DeviceBuffer<'c>,
    ao: DeviceBuffer<'c>,
    rl: DeviceBuffer<'c>,
    sgl: DeviceBuffer<'c>,
    sg: DeviceBuffer<'c>,
    su: DeviceBuffer<'c>,
    sa: DeviceBuffer<'c>,
    so: DeviceBuffer<'c>,
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
    bb: BatchBufs<'c>,
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
    /// Expert slots of `cache.slot_bytes` for `moe_batch`'s overflow misses (not cached), and how many.
    batch_scratch: Option<(DeviceBuffer<'c>, usize)>,
    ram: RamTier<'c>,
    copy_stream: Stream<'c>,
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
    /// The most tokens `step_batch` takes through the layers together: `MAX_BATCH`, or `SHADOW_BATCH=n` (1: one token
    /// at a time, for comparisons).
    pub batch_max: usize,
    /// CUDA graphs of the launches between two router reads that do not depend on the position (a DeltaNet layer's
    /// hc glue, mixer and router), keyed by (layer, whether its first norm was fused into the previous write); captured
    /// on first use on `gstream`, which orders with the default stream.  `SHADOW_GRAPHS=0`: none.
    graphs: HashMap<(usize, bool), Graph<'c>>,
    graphs_on: bool,
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
    /// The pinned RAM tier in bytes (0: none).  Refused if it would leave less than `pinned::RAM_FLOOR` available.
    pub ram_bytes: usize,
    /// Size the RAM tier as `pinned::auto_tier_bytes()` says, measured when the session fills it - after the dense
    /// weights are loaded and uploaded, so the figure is what is really left (`ram_bytes` is then ignored).
    pub ram_auto: bool,
    /// Which experts the RAM tier holds, best first; `None`: `placement::uniform_order`.
    pub order: Option<Vec<(usize, usize)>>,
    /// Path to tuned MTP head directory (e.g. models/mtp-basalt-ft1).
    pub mtp_dir: Option<std::path::PathBuf>,
    /// Hybrid MoE: up to this many of a layer's misses that the RAM tier holds are computed on the CPU instead of
    /// copied to the GPU (0: off), on `cpu_threads` workers.
    pub cpu_experts: usize,
    pub cpu_threads: usize,
    /// Pre-gated prefetch: layer l+1's router applied to layer l's MoE input; its top `pregate` experts that the RAM
    /// tier holds and VRAM does not are copied up while layer l computes (0: off).
    pub pregate: usize,
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
        Ok(Tiers { vram_cache: gib("--expert-cache-gib")?, ram_bytes, ram_auto, order, mtp_dir,
                   cpu_experts: num("--cpu-experts", 0)?, cpu_threads: num("--cpu-threads", 8)?,
                   pregate: num("--pregate", 0)? })
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
            claimed: Default::default(), served: 0, drive: 0, fill_secs: 0.0,
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
pub const VRAM_RESERVE: usize = 384 << 20;

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
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
        let ab = |n: usize| a(MAX_BATCH * n);
        let bb = BatchBufs {
            r: ab(ch)?, rn: ab(ch)?, u: ab(h)?, g: ab(cs)?, y: ab(h)?, down: ab(c.hc_rank)?, up: ab(ch)?,
            qkv: ab(2 * kd + vd)?, z: ab(vd)?, beta: ab(c.ssm_v_heads)?, alpha: ab(c.ssm_v_heads)?, o: ab(vd)?,
            qg: ab(2 * qd)?, kk: ab(kvd)?, iq: ab(c.idx_heads * id)?, ao: ab(qd)?, rl: ab(c.n_expert)?, sgl: ab(1)?,
            sg: ab(c.shared_ff)?, su: ab(c.shared_ff)?, sa: ab(c.shared_ff)?, so: ab(h)?, logits: a(COLS_MAX * c.vocab)?,
        };
        let mut delta = Vec::new();
        let mut attn = Vec::new();
        for l in 0..c.n_layer {
            if c.is_attention(l) {
                delta.push(None);
                let blocks = if id > 0 { cap / c.compress[l].max(1) + 1 } else { 0 };
                let mut at = Attn { k: a(cap * kvd)?, v: a(cap * kvd)?, raw: a(cap * id)?, blocks: a(blocks * id)?, nblocks: 0 };
                at.k.zero().map_err(e)?;
                at.v.zero().map_err(e)?;
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
            None => ctx.mem_info().map_err(e)?.0.saturating_sub(VRAM_RESERVE),
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
                      crate::pinned::AUTO_MARGIN as f64 / (1u64 << 30) as f64);
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
            let margin = (c.n_expert_used + 2).max(12);
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
        let copy_event = ctx.event().map_err(e)?;
        let staging = if slot_bytes > 0 && c.n_expert_used > 0 {
            // four tokens' worth: a batched step may read up to 4 * n_expert_used of a layer's misses together
            Some(PinnedRegion::new(ctx, 4 * c.n_expert_used * slot_bytes)?)
        } else {
            None
        };
        let mut s = GpuSession {
            ctx, m, mvp, fw, w, v, b, bb, batch_scratch, delta, attn, ple_state, tokens: vec![], cap, trace: None, stop_after: None,
            resident_bytes, expert_uploads: 0, cache, ram, copy_stream, copy_event, staging,
            pending: vec![], routing: RoutingCounts::new(c.n_layer, c.n_expert),
            prof: std::env::var("SHADOW_PROFILE").is_ok_and(|v| v == "1")
                .then(|| Prof { last: std::time::Instant::now(), phases: vec![] }),
            route_log: std::env::var("SHADOW_ROUTE_LOG").ok().map(|p| (p, vec![])),
            cpu: (tiers.cpu_experts > 0 && tiers.cpu_threads > 0)
                .then(|| (shadow_model::cpu_expert::ExpertPool::new(tiers.cpu_threads), tiers.cpu_experts)),
            cpu_served: 0,
            mapped: PinnedRegion::new(ctx, MAX_BATCH.max(2) * c.n_expert * 4)?,
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
            graphs: HashMap::new(),
            graphs_on: std::env::var("SHADOW_GRAPHS").map_or(true, |v| v != "0"),
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
        Ok(Snapshot { tokens: self.tokens.clone(), delta, ple: copy(&self.ple_state, self.ple_state.bytes)?, kv,
                      nblocks: self.attn.iter().map(|a| a.as_ref().map_or(0, |a| a.nblocks)).collect() })
    }

    /// Takes `snap` again at the current position, into its own buffers: a speculative pass's rollback point, once per
    /// pass without allocating (a device snapshot without the KV, from this session; otherwise a new one).
    pub fn snapshot_again(&self, snap: &mut Snapshot<'c>) -> Result<(), String> {
        let reusable = snap.kv.iter().all(|k| k.is_none()) && snap.delta.len() == self.delta.len()
            && matches!(snap.ple, Saved::Dev(..))
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

    fn k(&self, name: &str, grid: u32, block: u32, args: &[Arg]) {
        // SAFETY: every pointer passed is a buffer of this session sized for the dimensions passed beside it (the
        // registry checks the argument types and count).
        unsafe { self.fw.launch(name, (grid, 1, 1), (block, 1, 1), 0, self.ls(), args) }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
    }

    fn k1(&self, name: &str, n: usize, args: &[Arg]) {
        self.k(name, (n as u32).div_ceil(256), 256, args);
    }

    fn matvec(&self, t: &TensorInfo, wptr: u64, rows: usize, x: u64, y: u64) {
        let ty = t.ty.name();
        // the 8-at-a-time kernel where tests/matvec.rs measured it faster than the reference one (every block-quantized type)
        let fast = !matches!(ty.as_str(), "BF16" | "F32") && t.dims[0].is_multiple_of(8);
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
    fn upload_pinned_on(&self, ptr: u64, data: &[u8], stream: *mut std::ffi::c_void) {
        let api = self.ctx.api;
        self.ctx.bind().unwrap();
        // SAFETY: `data` lies inside `PinnedRegion` (registered page-locked memory), so `cuMemcpyHtoDAsync` will not
        // stage through pageable memory.
        let r = unsafe { (api.cuMemcpyHtoDAsync)(ptr, data.as_ptr() as *const std::ffi::c_void, data.len(), stream) };
        assert_eq!(r, 0, "cuMemcpyHtoDAsync failed ({r})");
    }

    // -------------------------------------------------------------------------------------------- the forward pass
    pub fn step_draft(&mut self, last_token: u32) -> Result<u32, String> {
        let (pos, r_in) = (self.tokens.len().saturating_sub(1), self.b.r.ptr);
        let (t_emb, emb_bytes) = self.m.dense("token_embd.weight");
        let (qk, qb) = t_emb.ty.block().unwrap();
        let rb = (t_emb.dims[0] / qk * qb) as usize;
        let mut emb = vec![0f32; self.m.cfg.hidden];
        shadow_model::quant::dequantize(t_emb.ty, &emb_bytes[last_token as usize * rb..(last_token as usize + 1) * rb], &mut emb)?;

        let (t_out, _) = self.m.dense("output.weight");
        let out_ty = t_out.ty.name();
        let out_rows = t_out.dims[1] as usize;
        let out_ptr = self.w.get("output.weight").unwrap().ptr;

        if let Some(ref mut mtp) = self.mtp {
            mtp.bufs.en.upload(&f32s(&emb)).map_err(|e| e.to_string())?;
            mtp.draft_step(
                &self.mvp,
                &self.fw,
                0,
                &out_ty,
                self.m.cfg.hidden,
                out_ptr,
                &out_ty,
                out_rows,
                r_in,
                last_token,
                pos,
            )
        } else {
            Err("mtp: draft model is not loaded".into())
        }
    }

    pub fn step(&mut self, token: u32) -> Vec<f32> {
        let draft = if self.mtp_probe.is_some() && self.mtp.is_some() && !self.tokens.is_empty() {
            self.step_draft(token).ok()
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
        self.mark("embedding");

        let graphs = self.graphs_on && self.trace.is_none() && self.stop_after.is_none() && self.prof.is_none()
            && self.cpu.is_none() && self.prefetch.is_none() && self.pregate.is_none() && self.mtp.is_none()
            && self.cache.slots > 0;
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
            self.mvp.launch("hc_down_inject_bf16", ((r_down + r_inj).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(), &[
                Arg::Ptr(self.w[&format!("{p}_down.weight")].ptr), Arg::Ptr(b.down.ptr), Arg::U32(r_down),
                Arg::Ptr(w_inj), Arg::Ptr(b.g.ptr), Arg::U32(r_inj),
                Arg::Ptr(b.rn.ptr), Arg::U32(h * cs), Arg::F32(cs as f32),
            ]).unwrap_or_else(|err| panic!("hc_down_inject_bf16: {err}"));
        }
        self.mv(&format!("{p}_up.weight"), b.down.ptr, b.up.ptr);
        self.hc_mix(b.rn.ptr, b.up.ptr, b.u.ptr);
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
        let t_wait = std::time::Instant::now();
        let mut got = self.download_mapped(&srcs).into_iter();
        self.waits[0] += t_wait.elapsed().as_secs_f64();
        let logits = got.next().unwrap();
        let next_logits = got.next();
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
        // Hybrid MoE: up to `cpu_experts` of the lowest-weight misses that the RAM tier holds are computed on the CPU,
        // from the pinned bytes, while the GPU copies and runs the rest.  `top` becomes [GPU experts.., CPU experts..].
        let mut top = top;
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
                let mut ub = vec![0u8; h * 4];
                self.download(self.b.u.ptr, &mut ub);
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
                    }
                }
                let shape = shadow_model::cpu_expert::ExpertShape {
                    hidden: h, ff, gate_ty: infos[0].ty.name(), up_ty: infos[1].ty.name(), down_ty: infos[2].ty.name(),
                };
                let pool = &mut self.cpu.as_mut().unwrap().0;
                // SAFETY: the region lives as long as the session, and nothing rewrites these experts' bytes before
                // `wait` below: read_drive waits first if it is about to reuse one of their ring slots.
                unsafe { pool.submit(&shape, refs, &floats(&ub)) };
                self.cpu_served += cpu_set.len() as u64;
                top.retain(|&(e, _)| !cpu_set.iter().any(|c| c.0 == e));
                top.extend(cpu_set.iter().map(|&(e, w, _)| (e, w)));
                cpu_offs = cpu_set.iter().map(|c| c.2).collect();
            }
        }
        if !is_flint && sh_launched_early && self.cache.slots > 0 && cpu_offs.is_empty() && self.prefetch.is_none()
            && infos[0].ty == infos[1].ty && infos[0].dims == infos[1].dims && self.overlap
        {
            self.routed_overlapped(l, &top, &names, &infos, &sizes);
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
            self.mv_batch(t, tab_off, count, x, xs, y, ys)
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
        let sh_out_ptr = b.eo.ptr + (n * h * 4) as u64;
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

    /// `step_batch` without the logits (reading a prompt).
    pub fn prefill_batch(&mut self, tokens: &[u32]) {
        self.step_batch_opt(tokens, false);
    }

    pub fn step_batch_opt(&mut self, tokens: &[u32], with_logits: bool) -> Vec<Vec<f32>> {
        let per_token = self.m.cfg.arch == "flint" || self.trace.is_some() || self.stop_after.is_some()
            || self.cache.slots == 0 || self.cpu.is_some() || self.prefetch.is_some() || self.pregate.is_some()
            || self.mtp_probe.is_some();
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
        let m = toks.len();
        let p0 = self.tokens.len();
        assert!(p0 + m <= self.cap, "this session was made for {} positions", self.cap);
        self.tokens.extend_from_slice(toks);
        self.routing.tokens += m as u64;
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden, c.streams);
        let ch = h * cs;
        {
            let (t, bytes) = self.m.dense("token_embd.weight");
            let (qk, qb) = t.ty.block().unwrap();
            let rb = (t.dims[0] / qk * qb) as usize;
            for (j, &tok) in toks.iter().enumerate() {
                let mut x = vec![0f32; h];
                dequantize(t.ty, &bytes[tok as usize * rb..(tok as usize + 1) * rb], &mut x).unwrap();
                self.upload_to(self.b.e.ptr, &f32s(&x));
                self.k1("broadcast", ch, &[Arg::Ptr(self.b.e.ptr), Arg::Ptr(row(&self.bb.r, ch, j)), Arg::U32(h as u32),
                                           Arg::U32(cs as u32)]);
            }
        }
        let mut logs: Vec<Vec<u16>> = vec![vec![]; m];
        let mut rn_ready = false;
        for l in 0..c.n_layer {
            if l == c.ple_layer {
                for j in 0..m {
                    let r = row(&self.bb.r, ch, j);
                    self.ple_at(l, p0 + j + 1, r);
                    self.k1("add", ch, &[Arg::Ptr(r), Arg::Ptr(self.b.gv.ptr), Arg::U32(ch as u32)]);
                }
                rn_ready = false;
            }
            self.hc_read_batch(&format!("blk.{l}.hc_attn"), true, rn_ready, m);
            if c.is_attention(l) {
                self.attention_batch(l, p0, m);
            } else {
                self.delta_batch(l, m);
            }
            self.hc_write_batch(Some(&format!("blk.{l}.hc_ffn_norm.weight")), m);
            self.hc_read_batch(&format!("blk.{l}.hc_ffn"), true, true, m);
            self.moe_batch(l, m, &mut logs);
            let next = if l + 1 == c.ple_layer {
                None
            } else if l + 1 < c.n_layer {
                Some(format!("blk.{}.hc_attn_norm.weight", l + 1))
            } else {
                Some("output_hc_norm.weight".to_string())
            };
            self.hc_write_batch(next.as_deref(), m);
            rn_ready = next.is_some();
        }
        if let Some((_, log)) = self.route_log.as_mut() {
            log.extend(logs.into_iter().flatten());
        }
        if !with_logits {
            return vec![];
        }
        self.hc_read_batch("output_hc", false, rn_ready, m);
        let mut logits = Vec::with_capacity(m);
        for j0 in (0..m).step_by(COLS_MAX) {
            let n = (m - j0).min(COLS_MAX);
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
            "BF16" => Some("matvec_cols_ref_bf16".to_string()),
            "F32" => Some("matvec_cols_ref_f32".to_string()),
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
        for j0 in (0..m).step_by(COLS_MAX) {
            let (x, y, n) = (x + (j0 * xs * 4) as u64, y + (j0 * ys * 4) as u64, (m - j0).min(COLS_MAX));
            // SAFETY: `rows` whole rows of t.dims[0] weights at wptr; n inputs of t.dims[0] floats at x + j * xs and n
            // outputs of `rows` floats at y + j * ys, inside this session's buffers (the callers' strides are their
            // sizes).
            unsafe {
                self.mvp.launch(&entry, ((rows as u32).div_ceil(32), 1, 1), (256, 1, 1), 0, self.ls(),
                                &[Arg::Ptr(wptr), Arg::Ptr(x), Arg::U32(xs as u32), Arg::Ptr(y), Arg::U32(ys as u32),
                                  Arg::U32(rows as u32), Arg::U32(t.dims[0] as u32), Arg::U32(n as u32)])
            }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        }
    }

    /// `count` matrices of t's shape from the device table at tab + tab_off, one per blockIdx.y, on inputs x + n * xs
    /// (xs = 0: shared) into y + n * ys.
    #[allow(clippy::too_many_arguments)]
    fn mv_batch(&self, t: &TensorInfo, tab_off: u64, count: usize, x: u64, xs: usize, y: u64, ys: usize) {
        let ty = t.ty.name();
        let entry = FAST_KERNELS.iter().find(|q| q.0 == ty).unwrap_or_else(|| panic!("{}: no batched kernel", t.name)).2;
        let (rows, cols) = (t.dims[1] as u32, t.dims[0] as u32);
        assert_eq!(cols % 8, 0, "{}: the batched kernel needs whole chunks of 8", t.name);
        // SAFETY: `count` table entries from tab + tab_off, each a rows x cols matrix as stored; `count` inputs of
        // cols floats (stride xs, or 0 when shared) and `count` outputs of rows floats (stride ys) inside this
        // session's buffers.
        unsafe {
            self.mvp.launch(entry, (rows.div_ceil(8), count as u32, 1), (256, 1, 1), 0, self.ls(),
                            &[Arg::Ptr(self.b.tab.ptr + tab_off), Arg::Ptr(x), Arg::U32(xs as u32), Arg::Ptr(y),
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
        if !skip_norm {
            for j in 0..m {
                self.hc_norm(p, row(&bb.r, ch, j), row(&bb.rn, ch, j));
            }
        }
        let (w_inj, r_inj) = if inject { (self.w[&format!("{p}_inject.weight")].ptr, cs as u32) } else { (0, 0) };
        for j0 in (0..m).step_by(COLS_MAX) {
            // SAFETY: rank + r_inj rows of ch BF16 weights; n inputs of ch floats in rn, n outputs of rank (down) and
            // cs (g) floats, from row j0.
            unsafe {
                self.mvp.launch("hc_down_inject_cols_bf16", ((rank as u32 + r_inj).div_ceil(32), 1, 1), (256, 1, 1), 0, self.ls(), &[
                    Arg::Ptr(self.w[&format!("{p}_down.weight")].ptr), Arg::Ptr(row(&bb.down, rank, j0)), Arg::U32(rank as u32),
                    Arg::U32(rank as u32), Arg::Ptr(w_inj), Arg::Ptr(row(&bb.g, cs, j0)), Arg::U32(r_inj), Arg::U32(cs as u32),
                    Arg::Ptr(row(&bb.rn, ch, j0)), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::F32(cs as f32),
                    Arg::U32((m - j0).min(COLS_MAX) as u32),
                ]).unwrap_or_else(|err| panic!("hc_down_inject_cols_bf16: {err}"));
            }
        }
        self.mv_cols(&format!("{p}_up.weight"), bb.down.ptr, rank, bb.up.ptr, ch, m);
        for j in 0..m {
            self.hc_mix(row(&bb.rn, ch, j), row(&bb.up, ch, j), row(&bb.u, h, j));
        }
    }

    /// `hc_write_norm` (with the next norm) or `hc_write` (without) for the batch's m tokens.
    fn hc_write_batch(&self, next_norm: Option<&str>, m: usize) {
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden, c.streams);
        let bb = &self.bb;
        for j in 0..m {
            let (r, g, y) = (row(&bb.r, h * cs, j), row(&bb.g, cs, j), row(&bb.y, h, j));
            match next_norm {
                Some(n) => self.hc_write_norm_at(n, r, g, y, row(&bb.rn, h * cs, j)),
                None => self.hc_write_at(r, g, y),
            }
        }
    }

    fn delta_batch(&self, l: usize, m: usize) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (h, vh, d) = (c.hidden, c.ssm_v_heads, c.ssm_head_dim);
        let (kd, vd) = (c.ssm_k_heads * d, vh * d);
        let ch = 2 * kd + vd;
        let bb = &self.bb;
        self.mv_cols(&format!("{p}.attn_qkv.weight"), bb.u.ptr, h, bb.qkv.ptr, ch, m);
        self.mv_cols(&format!("{p}.attn_gate.weight"), bb.u.ptr, h, bb.z.ptr, vd, m);
        self.mv_cols(&format!("{p}.ssm_beta.weight"), bb.u.ptr, h, bb.beta.ptr, vh, m);
        self.mv_cols(&format!("{p}.ssm_alpha.weight"), bb.u.ptr, h, bb.alpha.ptr, vh, m);
        for j in 0..m {
            self.delta_mid(l, row(&bb.qkv, ch, j), row(&bb.z, vd, j), row(&bb.beta, vh, j), row(&bb.alpha, vh, j),
                           row(&bb.o, vd, j));
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
        for j in 0..m {
            let bb = &self.bb;
            let (qg, kk, iq, ao) = (row(&bb.qg, 2 * qd, j), row(&bb.kk, kvd, j), row(&bb.iq, iqd, j), row(&bb.ao, qd, j));
            self.attention_mid(l, p0 + j, qg, kk, iq, ao);
        }
        self.mv_cols(&format!("{p}.attn_output.weight"), self.bb.ao.ptr, qd, self.bb.y.ptr, h, m);
    }

    /// The MoE for the batch's m tokens: the router and the shared expert for all of them at once, one read of their
    /// router logits, then the union of their routed experts into the VRAM cache (the misses beyond the layer's free
    /// slots into `batch_scratch`) - in groups of consecutive tokens whose misses fit together - and each token's
    /// routed experts and combine.
    fn moe_batch(&mut self, l: usize, m: usize, logs: &mut [Vec<u16>]) {
        let c = &self.m.cfg;
        let p = format!("blk.{l}");
        let (h, sff, k, ne) = (c.hidden, c.shared_ff, c.n_expert_used, c.n_expert);
        let u = self.bb.u.ptr;
        self.mv_cols(&format!("{p}.ffn_gate_inp.weight"), u, h, self.bb.rl.ptr, ne, m);
        self.mv_cols(&format!("{p}.ffn_gate_inp_shexp.weight"), u, h, self.bb.sgl.ptr, 1, m);
        let srcs: Vec<(u64, usize)> = (0..m).map(|j| (row(&self.bb.rl, ne, j), ne)).collect();
        let t_wait = std::time::Instant::now();
        let logits = self.download_mapped(&srcs);
        self.waits[0] += t_wait.elapsed().as_secs_f64();
        {
            // the shared expert, on the GPU while the host routes and the copies run
            let bb = &self.bb;
            self.mv_cols(&format!("{p}.ffn_gate_shexp.weight"), u, h, bb.sg.ptr, sff, m);
            self.mv_cols(&format!("{p}.ffn_up_shexp.weight"), u, h, bb.su.ptr, sff, m);
            self.k1("swiglu", m * sff, &[Arg::Ptr(bb.sg.ptr), Arg::Ptr(bb.su.ptr), Arg::Ptr(bb.sa.ptr), Arg::U32((m * sff) as u32)]);
            self.mv_cols(&format!("{p}.ffn_down_shexp.weight"), bb.sa.ptr, sff, bb.so.ptr, h, m);
        }
        let tops: Vec<Vec<(usize, f64)>> = logits.iter().map(|lg| route(lg, k)).collect();
        for (j, top) in tops.iter().enumerate() {
            for &(e, _) in top {
                self.routing.add(l, e);
            }
            if self.route_log.is_some() {
                logs[j].push(l as u16);
                logs[j].extend(top.iter().map(|&(e, _)| e as u16));
            }
        }
        let names = [format!("{p}.ffn_gate_exps.weight"), format!("{p}.ffn_up_exps.weight"), format!("{p}.ffn_down_exps.weight")];
        let infos: Vec<TensorInfo> = names.iter().map(|n| self.expert_info(n)).collect();
        let sizes: Vec<usize> = names.iter().map(|n| self.m.expert_bytes(n)).collect();
        // misses read from the file may wait in the staging buffer, a slot each
        let max_file = self.staging.as_ref().map_or(usize::MAX, |s| s.len() / self.cache.slot_bytes.max(1));
        let scratch = self.batch_scratch.as_ref().map_or((0, 0), |(b, n)| (b.ptr, *n));
        let arena = self.cache.arena.as_ref().unwrap().ptr;
        let mut start = 0;
        while start < m {
            let mut set: Vec<usize> = Vec::new();
            let mut end = start;
            while end < m {
                let mut cand = set.clone();
                for &(e, _) in &tops[end] {
                    if !cand.contains(&e) {
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
            self.cache.clock += 1;
            for &e in &set {
                self.cache.touch((l, e));
            }
            let (mut n_miss, mut n_file) = (0, 0);
            let mut extra: Vec<(usize, u64)> = Vec::new();
            for &e in &set {
                if self.cache.map.contains_key(&(l, e)) {
                    self.cache.hits += 1;
                    continue;
                }
                self.cache.misses += 1;
                let base = if self.cache.has_victim(l) {
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
                let dst = [base, base + sizes[0] as u64, base + (sizes[0] + sizes[1]) as u64];
                let from_file = !self.ram.at.contains_key(&(l, e));
                self.fetch_expert(l, e, n_file, &names, &sizes, dst);
                n_file += from_file as usize;
                n_miss += 1;
            }
            self.read_drive(&names, &sizes);
            if n_miss > 0 {
                let t_wait = std::time::Instant::now();
                self.copy_stream.synchronize().unwrap();
                self.waits[1] += t_wait.elapsed().as_secs_f64();
            }
            for (j, top) in tops.iter().enumerate().take(end).skip(start) {
                self.routed_qwen(l, j, top, &extra, &infos, &sizes);
            }
            start = end;
        }
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
            self.mv_batch(&infos[0], 0, 2 * n, u, 0, b.eg.ptr, ff);
            b.eg.ptr + (n * ff * 4) as u64
        } else {
            self.mv_batch(&infos[0], 0, n, u, 0, b.eg.ptr, ff);
            self.mv_batch(&infos[1], 8 * n as u64, n, u, 0, b.eu.ptr, ff);
            b.eu.ptr
        };
        self.k1("swiglu", n * ff, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(eu), Arg::Ptr(b.act.ptr), Arg::U32((n * ff) as u32)]);
        self.mv_batch(&infos[2], 16 * n as u64, n, b.act.ptr, ff, b.eo.ptr, h);
        self.k1("moe_combine_qwen", h, &[
            Arg::Ptr(row(&self.bb.y, h, j)), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.tab.ptr + 24 * n as u64),
            Arg::Ptr(row(&self.bb.so, h, j)), Arg::Ptr(row(&self.bb.sgl, 1, j)), Arg::U32(h as u32), Arg::U32(n as u32)]);
    }

    /// One token's routed experts with their copies overlapped: the resident ones run at once, the misses' copies go
    /// out on copy_stream, and the default stream - not the host - waits for them before the misses run, so the host
    /// goes on queueing the layer meanwhile.  eo's rows come out resident first; the combine reads them in routing
    /// order (`perm`), so y is `moe`'s to the bit.
    fn routed_overlapped(&mut self, l: usize, top: &[(usize, f64)], names: &[String; 3], infos: &[TensorInfo],
                         sizes: &[usize]) {
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
        let mut perm = vec![0u32; n];
        for (row, x) in hits.iter().chain(&misses).enumerate() {
            tab.extend_from_slice(&(x.2 + part[2]).to_le_bytes());
            perm[x.0] = row as u32;
        }
        for &(_, w) in top {
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
            s.mv_batch(&infos[0], 16 * first as u64, 2 * cnt, b.u.ptr, 0, eg, ff);
            s.k1("swiglu", cnt * ff, &[Arg::Ptr(eg), Arg::Ptr(eg + (cnt * ff * 4) as u64), Arg::Ptr(act),
                                      Arg::U32((cnt * ff) as u32)]);
            s.mv_batch(&infos[2], (16 * n + 8 * first) as u64, cnt, act, ff, b.eo.ptr + (first * h * 4) as u64, h);
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
        let b = &self.b;
        self.k1("moe_combine_qwen_perm", h, &[
            Arg::Ptr(b.y.ptr), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.tab.ptr + 24 * n as u64), Arg::Ptr(b.tab.ptr + 28 * n as u64),
            Arg::Ptr(b.eo.ptr + (n * h * 4) as u64), Arg::Ptr(b.sgl.ptr), Arg::U32(h as u32), Arg::U32(n as u32)]);
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
        let e: Vec<f32> = ple_rows(c, &self.tokens[..n]).into_iter().flat_map(|row| self.m.ple_row(row)).collect();
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
        self.step_batch(tokens).pop().unwrap_or_default()
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
            + &format!("; RAM tier: {} experts, {:.2} GiB pinned (filled in {:.1} s), misses served {} from RAM, {} from the file",
                       self.ram.at.len(), self.ram.region.as_ref().map_or(0, |r| r.len()) as f64 / (1u64 << 30) as f64,
                       self.ram.fill_secs, self.ram.served, self.ram.drive)
            + &self.cpu.as_ref().map(|c| format!("; CPU experts: up to {} per layer, {} computed", c.1, self.cpu_served))
                .unwrap_or_default()
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
