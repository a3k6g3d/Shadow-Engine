//! Routed experts on the CPU, straight from the pinned RAM tier: the other half of the hybrid MoE.
//!
//! On this PC a missed expert reaches the GPU at ~22 GB/s (PCIe 4 x16, already at its limit), and expert movement is
//! most of a step.  The CPU can read the same pinned bytes from RAM at the same time, so a few of each layer's misses
//! are computed here while the GPU takes the rest: two paths to the weights instead of one.
//!
//! Each quant block is decoded to int8 values and one f32 scale per sub-block, exactly the factors `quant::dequantize`
//! multiplies (so the weights are the same numbers), then dotted with the f32 input on AVX2 + FMA.  Only the order of
//! the sums differs from the GPU and the reference.
//!
//! `ExpertPool` keeps `threads` workers that spin briefly between jobs (a layer's job comes ~1 ms after the last), then
//! park.  A job is gate+up+SwiGLU for every (expert, block of rows), a barrier, then the down projection: the same
//! `silu(gate) * up` and `down` the GPU runs, unweighted (the GPU's combine applies the router weights).

use crate::half::f16_to_f32;
use crate::quant_tables::{IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES, KSIGNS};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// For each sign byte, a mask with 0xff in every byte whose bit is set.
const SIGN8: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        let mut m = 0u64;
        let mut k = 0;
        while k < 8 {
            if (i >> k) & 1 == 1 {
                m |= 0xffu64 << (8 * k);
            }
            k += 1;
        }
        t[i] = m;
        i += 1;
    }
    t
};
const ONES: u64 = 0x0101_0101_0101_0101;

/// Negates the bytes of `v` that `m` marks.  Every grid value is 1..=127, so `(v ^ 0xff) + 1` never carries.
#[inline(always)]
fn neg8(v: u64, m: u64) -> u64 {
    (v ^ m).wrapping_add(m & ONES)
}

#[inline(always)]
fn put8(q: &mut [i8], at: usize, v: u64) {
    q[at..at + 8].copy_from_slice(&(v.to_le_bytes().map(|b| b as i8)));
}

fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// A quant type's block as int8 values and per-sub-block scales: (weights per block, bytes per block, sub-block size,
/// decoder).  The decoder fills `q[..weights]` and `s[..weights / sub]`.
pub type Decode = fn(&[u8], &mut [i8; 256], &mut [f32; 16]);

pub fn layout(ty: &str) -> Option<(usize, usize, usize, Decode)> {
    Some(match ty {
        "IQ2_S" => (256, 82, 16, iq2_s as Decode),
        "IQ3_XXS" => (256, 98, 32, iq3_xxs),
        "IQ3_S" => (256, 110, 32, iq3_s),
        "IQ4_XS" => (256, 136, 32, iq4_xs),
        "IQ4_NL" => (32, 18, 32, iq4_nl),
        "Q2_0" => (64, 18, 32, q2_0),
        _ => return None,
    })
}

fn iq2_s(b: &[u8], q: &mut [i8; 256], s: &mut [f32; 16]) {
    let d = f16_at(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for j in 0..32 {
        let idx = qs[j] as usize | ((((qh[j / 4] >> (2 * (j % 4))) & 3) as usize) << 8);
        put8(q, j * 8, neg8(u64::from_le_bytes(IQ2S_GRID[idx]), SIGN8[signs[j] as usize]));
    }
    for (i, v) in s.iter_mut().enumerate() {
        *v = d * (0.5 + ((scales[i / 2] >> (4 * (i % 2))) & 0x0f) as f32) * 0.25;
    }
}

fn iq3_xxs(b: &[u8], q: &mut [i8; 256], s: &mut [f32; 16]) {
    let d = f16_at(b, 0);
    let (qs, sc) = (&b[2..66], &b[66..98]);
    for g in 0..8 {
        let s32 = u32::from_le_bytes(sc[g * 4..g * 4 + 4].try_into().unwrap());
        s[g] = d * (0.5 + (s32 >> 28) as f32) * 0.5;
        for k in 0..4 {
            let signs = KSIGNS[((s32 >> (7 * k)) & 0x7f) as usize];
            let (a, c) = (IQ3XXS_GRID[qs[g * 8 + k * 2] as usize], IQ3XXS_GRID[qs[g * 8 + k * 2 + 1] as usize]);
            let v = u32::from_le_bytes(a) as u64 | ((u32::from_le_bytes(c) as u64) << 32);
            put8(q, g * 32 + k * 8, neg8(v, SIGN8[signs as usize]));
        }
    }
}

fn iq3_s(b: &[u8], q: &mut [i8; 256], s: &mut [f32; 16]) {
    let d = f16_at(b, 0);
    let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
    let idx = |j: usize| qs[j] as usize | ((((qh[j / 8] >> (j % 8)) & 1) as usize) << 8);
    for p in 0..32 {
        let v = u32::from_le_bytes(IQ3S_GRID[idx(2 * p)]) as u64 | ((u32::from_le_bytes(IQ3S_GRID[idx(2 * p + 1)]) as u64) << 32);
        put8(q, p * 8, neg8(v, SIGN8[signs[p] as usize]));
    }
    for (g, v) in s.iter_mut().enumerate().take(8) {
        *v = d * (1 + 2 * ((scales[g / 2] >> (4 * (g % 2))) & 0x0f)) as f32;
    }
}

fn iq4_xs(b: &[u8], q: &mut [i8; 256], s: &mut [f32; 16]) {
    let d = f16_at(b, 0);
    let sh = u16::from_le_bytes([b[2], b[3]]);
    let (sl, qs) = (&b[4..8], &b[8..136]);
    for g in 0..8 {
        let lo = (sl[g / 2] >> (4 * (g % 2))) & 0x0f;
        let hi = ((sh >> (2 * g)) & 3) as u8;
        s[g] = d * ((lo | (hi << 4)) as i8).wrapping_sub(32) as f32;
        for c in 0..16 {
            let v = qs[g * 16 + c];
            q[g * 32 + c] = IQ4NL_VALUES[(v & 0x0f) as usize];
            q[g * 32 + 16 + c] = IQ4NL_VALUES[(v >> 4) as usize];
        }
    }
}

fn iq4_nl(b: &[u8], q: &mut [i8; 256], s: &mut [f32; 16]) {
    s[0] = f16_at(b, 0);
    for c in 0..16 {
        q[c] = IQ4NL_VALUES[(b[2 + c] & 0x0f) as usize];
        q[16 + c] = IQ4NL_VALUES[(b[2 + c] >> 4) as usize];
    }
}

fn q2_0(b: &[u8], q: &mut [i8; 256], s: &mut [f32; 16]) {
    let d = f16_at(b, 0);
    (s[0], s[1]) = (d, d);
    for (j, v) in q.iter_mut().enumerate().take(64) {
        *v = ((b[2 + j / 4] >> ((j % 4) * 2)) & 0x03) as i8 - 1;
    }
}

/// sum over sub-blocks g of s[g] * (q . x) for one decoded block of `n` weights in sub-blocks of `sub`.
fn block_dot_scalar(q: &[i8], s: &[f32], x: &[f32], n: usize, sub: usize) -> f32 {
    let mut total = 0f32;
    for (g, &sg) in s.iter().enumerate().take(n / sub) {
        let mut acc = 0f32;
        for i in g * sub..(g + 1) * sub {
            acc += q[i] as f32 * x[i];
        }
        total += sg * acc;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_dot_avx2(q: &[i8], s: &[f32], x: &[f32], n: usize, sub: usize) -> f32 {
    use std::arch::x86_64::*;
    debug_assert!(q.len() >= n && x.len() >= n && sub.is_multiple_of(8));
    // SAFETY (whole body): every load reads 8 i8 or 8 f32 at i < n, inside q[..n] and x[..n].
    unsafe {
        let mut total = _mm256_setzero_ps();
        for (g, &sg) in s.iter().enumerate().take(n / sub) {
            let mut acc = _mm256_setzero_ps();
            let mut i = g * sub;
            while i < (g + 1) * sub {
                let qi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(q.as_ptr().add(i) as *const __m128i));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(qi), _mm256_loadu_ps(x.as_ptr().add(i)), acc);
                i += 8;
            }
            total = _mm256_fmadd_ps(acc, _mm256_set1_ps(sg), total);
        }
        let hi = _mm256_extractf128_ps(total, 1);
        let lo = _mm256_castps256_ps128(total);
        let v = _mm_add_ps(hi, lo);
        let v = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
        _mm_cvtss_f32(v)
    }
}

fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static YES: OnceLock<bool> = OnceLock::new();
        *YES.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// One stored row (whole blocks of `ty`) dotted with `x`.
pub fn row_dot(ty: &str, row: &[u8], x: &[f32]) -> f32 {
    let (n, bytes, sub, decode) = layout(ty).unwrap_or_else(|| panic!("{ty}: no CPU expert kernel"));
    assert_eq!(row.len() / bytes * n, x.len(), "{ty}: {} bytes for {} inputs", row.len(), x.len());
    let (mut q, mut s) = ([0i8; 256], [0f32; 16]);
    let avx2 = has_avx2();
    let mut total = 0f32;
    for (b, xb) in row.chunks_exact(bytes).zip(x.chunks_exact(n)) {
        decode(b, &mut q, &mut s);
        total += if avx2 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: avx2 and fma were detected at run time; q and xb hold n values.
            unsafe { block_dot_avx2(&q, &s, xb, n, sub) }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!()
        } else {
            block_dot_scalar(&q, &s, xb, n, sub)
        };
    }
    total
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ------------------------------------------------------------------------------------------------------- the pool
/// One expert of a job: its gate, up and down matrices as stored (pointers into memory that outlives the job).
#[derive(Clone, Copy)]
pub struct ExpertRef {
    pub gate: *const u8,
    pub up: *const u8,
    pub down: *const u8,
}

/// The shapes and types of one layer's experts.
#[derive(Clone)]
pub struct ExpertShape {
    pub hidden: usize,
    pub ff: usize,
    pub gate_ty: String,
    pub up_ty: String,
    pub down_ty: String,
}

const ROWS_A: usize = 16; // gate/up rows per work item
const ROWS_B: usize = 64; // down rows per work item

struct Job {
    shape: ExpertShape,
    experts: Vec<ExpertRef>,
    x: Vec<f32>,
    act: Vec<std::cell::UnsafeCell<f32>>,
    out: Vec<std::cell::UnsafeCell<f32>>,
    items_a: usize,
    items_b: usize,
    next: AtomicUsize,
    done_a: AtomicUsize,
    done_b: AtomicUsize,
}

// SAFETY: the raw pointers are read-only expert bytes the submitter keeps alive until `wait`; `act` and `out` cells are
// written by exactly one work item each (disjoint rows), and read only after the barrier (`done_a`) or `wait`.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    /// Runs work items until there are none left.  Items 0..items_a are gate/up rows, the rest down rows; a down item
    /// first waits for every gate/up item to finish.
    fn work(&self) {
        let (h, ff) = (self.shape.hidden, self.shape.ff);
        let (gl, gb, ..) = layout(&self.shape.gate_ty).unwrap();
        let (ul, ub, ..) = layout(&self.shape.up_ty).unwrap();
        let (dl, db, ..) = layout(&self.shape.down_ty).unwrap();
        let (grow, urow, drow) = (h / gl * gb, h / ul * ub, ff / dl * db);
        let per_a = ff.div_ceil(ROWS_A);
        let per_b = h.div_ceil(ROWS_B);
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i < self.items_a {
                let (e, blk) = (i / per_a, i % per_a);
                let ex = self.experts[e];
                for r in blk * ROWS_A..((blk + 1) * ROWS_A).min(ff) {
                    // SAFETY: row r of a ff x h matrix as stored lies inside the expert's slice (ExpertRef's contract).
                    let (g, u) = unsafe {
                        (std::slice::from_raw_parts(ex.gate.add(r * grow), grow),
                         std::slice::from_raw_parts(ex.up.add(r * urow), urow))
                    };
                    let v = silu(row_dot(&self.shape.gate_ty, g, &self.x)) * row_dot(&self.shape.up_ty, u, &self.x);
                    // SAFETY: this item alone writes act[e * ff + r].
                    unsafe { *self.act[e * ff + r].get() = v };
                }
                self.done_a.fetch_add(1, Ordering::Release);
            } else if i < self.items_a + self.items_b {
                while self.done_a.load(Ordering::Acquire) < self.items_a {
                    std::hint::spin_loop();
                }
                let j = i - self.items_a;
                let (e, blk) = (j / per_b, j % per_b);
                let ex = self.experts[e];
                // SAFETY: every act cell was written before done_a reached items_a (Acquire above); none is written again.
                let a: Vec<f32> = (0..ff).map(|r| unsafe { *self.act[e * ff + r].get() }).collect();
                for r in blk * ROWS_B..((blk + 1) * ROWS_B).min(h) {
                    // SAFETY: row r of the h x ff down matrix as stored lies inside the expert's slice.
                    let d = unsafe { std::slice::from_raw_parts(ex.down.add(r * drow), drow) };
                    // SAFETY: this item alone writes out[e * h + r].
                    unsafe { *self.out[e * h + r].get() = row_dot(&self.shape.down_ty, d, &a) };
                }
                self.done_b.fetch_add(1, Ordering::Release);
            } else {
                return;
            }
        }
    }
}

struct Shared {
    job: Mutex<Option<Arc<Job>>>,
    epoch: AtomicU64,
    stop: AtomicBool,
}

pub struct ExpertPool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    current: Option<Arc<Job>>,
}

/// How long an idle worker spins before it parks.
const SPIN: std::time::Duration = std::time::Duration::from_millis(3);

impl ExpertPool {
    pub fn new(threads: usize) -> ExpertPool {
        let shared = Arc::new(Shared { job: Mutex::new(None), epoch: AtomicU64::new(0), stop: AtomicBool::new(false) });
        let threads = (0..threads).map(|w| {
            let sh = shared.clone();
            std::thread::Builder::new().name(format!("shadow-cpu-expert-{w}")).spawn(move || {
                let mut seen = 0u64;
                loop {
                    let t0 = std::time::Instant::now();
                    let mut spins = 0u32;
                    while sh.epoch.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Relaxed) {
                        spins = spins.wrapping_add(1);
                        if spins.is_multiple_of(1024) && t0.elapsed() > SPIN {
                            std::thread::park_timeout(std::time::Duration::from_millis(50));
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                    if sh.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    seen = sh.epoch.load(Ordering::Acquire);
                    let job = sh.job.lock().unwrap().clone();
                    if let Some(j) = job {
                        j.work();
                    }
                }
            }).expect("cpu expert worker")
        }).collect();
        ExpertPool { shared, threads, current: None }
    }

    /// Starts `silu(gate . x) * (up . x)` then `down . act` for every expert, in the background.
    ///
    /// # Safety
    /// Every pointer in `experts` must stay valid and unchanged, for its whole matrix of `shape`, until `wait` returns.
    pub unsafe fn submit(&mut self, shape: &ExpertShape, experts: Vec<ExpertRef>, x: &[f32]) {
        assert!(self.current.is_none(), "one job at a time");
        assert_eq!(x.len(), shape.hidden);
        let n = experts.len();
        let job = Arc::new(Job {
            items_a: n * shape.ff.div_ceil(ROWS_A),
            items_b: n * shape.hidden.div_ceil(ROWS_B),
            act: (0..n * shape.ff).map(|_| std::cell::UnsafeCell::new(0.0)).collect(),
            out: (0..n * shape.hidden).map(|_| std::cell::UnsafeCell::new(0.0)).collect(),
            shape: shape.clone(),
            experts,
            x: x.to_vec(),
            next: AtomicUsize::new(0),
            done_a: AtomicUsize::new(0),
            done_b: AtomicUsize::new(0),
        });
        *self.shared.job.lock().unwrap() = Some(job.clone());
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for t in &self.threads {
            t.thread().unpark();
        }
        self.current = Some(job);
    }

    /// Whether a job is in flight.
    pub fn busy(&self) -> bool {
        self.current.is_some()
    }

    /// Waits for the job (helping with what is left) and returns each expert's down output, `hidden` floats apiece.
    pub fn wait(&mut self) -> Vec<f32> {
        let Some(job) = self.current.take() else { return vec![] };
        job.work();
        while job.done_b.load(Ordering::Acquire) < job.items_b {
            std::hint::spin_loop();
        }
        *self.shared.job.lock().unwrap() = None;
        // SAFETY: every out cell was written before done_b reached items_b (Acquire above).
        job.out.iter().map(|c| unsafe { *c.get() }).collect()
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        self.wait();
        self.shared.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Random blocks of each type: the decoded int8 x scale products are exactly what `dequantize` gives, and the
    /// row dot is the f64 dot of the dequantized row to f32 rounding.
    #[test]
    fn decode_matches_dequantize() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for (ty, id) in [("IQ2_S", 22), ("IQ3_XXS", 18), ("IQ3_S", 21), ("IQ4_XS", 23), ("IQ4_NL", 20), ("Q2_0", 42)] {
            let (n, bytes, sub, decode) = layout(ty).unwrap();
            let gt = crate::gguf::GgmlType::from_id(id);
            for _ in 0..200 {
                let mut b: Vec<u8> = (0..bytes).map(|_| rng(&mut seed) as u8).collect();
                // a small finite f16 scale (0x2c00 ~ 0.0625 times a little noise)
                b[0] = rng(&mut seed) as u8;
                b[1] = 0x2c;
                let mut want = vec![0f32; n];
                crate::quant::dequantize(gt, &b, &mut want).unwrap();
                let (mut q, mut s) = ([0i8; 256], [0f32; 16]);
                decode(&b, &mut q, &mut s);
                for i in 0..n {
                    let got = s[i / sub] * q[i] as f32;
                    assert!((got - want[i]).abs() <= 1e-6 * want[i].abs().max(1e-3), "{ty} weight {i}: {got} vs {}", want[i]);
                }
                let x: Vec<f32> = (0..n).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
                let exact: f64 = want.iter().zip(&x).map(|(&w, &v)| w as f64 * v as f64).sum();
                let mag: f64 = want.iter().zip(&x).map(|(&w, &v)| (w as f64 * v as f64).abs()).sum();
                let got = row_dot(ty, &b, &x) as f64;
                assert!((got - exact).abs() <= 1e-5 * mag.max(1e-9), "{ty}: dot {got} vs {exact}");
            }
        }
    }

    #[test]
    fn pool_matches_serial() {
        let shape = ExpertShape { hidden: 512, ff: 64, gate_ty: "IQ3_S".into(), up_ty: "IQ3_S".into(), down_ty: "IQ4_NL".into() };
        let mut seed = 99u64;
        let gate_bytes = shape.ff * (shape.hidden / 256 * 110);
        let down_bytes = shape.hidden * (shape.ff / 32 * 18);
        let mut mk = |n: usize, scale_at: usize| -> Vec<u8> {
            let mut v: Vec<u8> = (0..n).map(|_| rng(&mut seed) as u8).collect();
            for blk in v.chunks_mut(scale_at) {
                blk[1] = 0x24; // keep the f16 scales small and finite
            }
            v
        };
        let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..3).map(|_| (mk(gate_bytes, 110), mk(gate_bytes, 110), mk(down_bytes, 18))).collect();
        let x: Vec<f32> = (0..shape.hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let mut pool = ExpertPool::new(3);
        let refs: Vec<ExpertRef> = mats.iter().map(|(g, u, d)| ExpertRef { gate: g.as_ptr(), up: u.as_ptr(), down: d.as_ptr() }).collect();
        // SAFETY: mats outlives the wait below.
        unsafe { pool.submit(&shape, refs, &x) };
        let got = pool.wait();
        for (e, (g, u, d)) in mats.iter().enumerate() {
            let grow = gate_bytes / shape.ff;
            let act: Vec<f32> = (0..shape.ff).map(|r| silu(row_dot("IQ3_S", &g[r * grow..(r + 1) * grow], &x))
                * row_dot("IQ3_S", &u[r * grow..(r + 1) * grow], &x)).collect();
            let drow = down_bytes / shape.hidden;
            for r in 0..shape.hidden {
                let want = row_dot("IQ4_NL", &d[r * drow..(r + 1) * drow], &act);
                assert_eq!(got[e * shape.hidden + r].to_bits(), want.to_bits(), "expert {e} row {r}");
            }
        }
    }
}
