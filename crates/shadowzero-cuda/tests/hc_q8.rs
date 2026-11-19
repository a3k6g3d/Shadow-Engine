//! Unit tests and microbenchmarks for packed per-32 symmetric `hc8` hyper-connection kernels.

use shadowzero_cuda::kernels::{file, Arg, Program};
use shadowzero_cuda::session::quantize_bf16_hc8;
use shadowzero_cuda::{nvrtc, Context, DeviceBuffer};
use shadowzero_model::half::f32_to_bf16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn gpu_ctx() -> Option<Context> {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            return None;
        }
    };
    if let Err(e) = nvrtc::version() {
        eprintln!("SKIPPED, not passed: no NVRTC ({e})");
        return None;
    }
    Some(ctx)
}

fn to_bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn download_f32(buf: &DeviceBuffer, n_floats: usize) -> Vec<f32> {
    let mut b = vec![0u8; buf.bytes];
    buf.download(&mut b).unwrap();
    b[..n_floats * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn make_bf16_matrix(rng: &mut Rng, rows: usize, cols: usize, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 2);
    for _ in 0..rows * cols {
        let u = f32_to_bf16(rng.unit() * scale);
        out.extend_from_slice(&u.to_le_bytes());
    }
    out
}

fn make_f32_vec(rng: &mut Rng, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| rng.unit() * scale).collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
    }
    (num / den.max(1e-24)).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-24)
}

fn check_close(got: &[f32], want: &[f32], name: &str, max_rel: f64) {
    assert_eq!(got.len(), want.len());
    for (i, &v) in got.iter().enumerate() {
        assert!(v.is_finite(), "{name}: non-finite at {i}: {v}");
    }
    let rl2 = rel_l2(got, want);
    let cos = cosine(got, want);
    assert!(rl2 < max_rel && cos > 0.9998, "{name}: rel_l2={rl2:.6}, cosine={cos:.6}");
}

#[test]
fn test_hc8_down_inject_cols() {
    let Some(ctx) = gpu_ctx() else { return };
    let mvp = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng(0x9e3779b97f4a7c15);

    let (r_down, r_inj, cols, streams) = (320usize, 4usize, 10240usize, 4usize);
    let max_m = 8usize;
    let w_down_bf16 = make_bf16_matrix(&mut rng, r_down, cols, 0.05);
    let w_inj_bf16 = make_bf16_matrix(&mut rng, r_inj, cols, 0.05);
    let w_down_hc8 = quantize_bf16_hc8(&w_down_bf16, r_down, cols);
    let w_inj_hc8 = quantize_bf16_hc8(&w_inj_bf16, r_inj, cols);
    let rn = make_f32_vec(&mut rng, max_m * cols, 1.0);

    let d_wdown_bf = ctx.alloc(w_down_bf16.len()).unwrap();
    let d_winj_bf = ctx.alloc(w_inj_bf16.len()).unwrap();
    let d_wdown_q8 = ctx.alloc(w_down_hc8.len()).unwrap();
    let d_winj_q8 = ctx.alloc(w_inj_hc8.len()).unwrap();
    let d_rn = ctx.alloc(rn.len() * 4).unwrap();
    let d_down_ref = ctx.alloc(max_m * r_down * 4).unwrap();
    let d_g_ref = ctx.alloc(max_m * r_inj * 4).unwrap();
    let d_down_q8 = ctx.alloc(max_m * r_down * 4).unwrap();
    let d_g_q8 = ctx.alloc(max_m * r_inj * 4).unwrap();

    d_wdown_bf.upload(&w_down_bf16).unwrap();
    d_winj_bf.upload(&w_inj_bf16).unwrap();
    d_wdown_q8.upload(&w_down_hc8).unwrap();
    d_winj_q8.upload(&w_inj_hc8).unwrap();
    d_rn.upload(&to_bytes_f32(&rn)).unwrap();

    for m in [1usize, 3, 4, 8] {
        let grid = ((r_down + r_inj) as u32).div_ceil(2);
        let q8_k = if m == 1 { "hc8_down_inject_m1" } else if m <= 4 { "hc8_down_inject_m4" } else { "hc8_down_inject_cols" };
        unsafe {
            mvp.launch("hc_down_inject_cols_fast_bf16", (grid, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wdown_bf.ptr), Arg::Ptr(d_down_ref.ptr), Arg::U32(r_down as u32), Arg::U32(r_down as u32),
                Arg::Ptr(d_winj_bf.ptr), Arg::Ptr(d_g_ref.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                Arg::Ptr(d_rn.ptr), Arg::U32(cols as u32), Arg::U32(cols as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
            ]).unwrap();
            mvp.launch(q8_k, (grid, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wdown_q8.ptr), Arg::Ptr(d_down_q8.ptr), Arg::U32(r_down as u32), Arg::U32(r_down as u32),
                Arg::Ptr(d_winj_q8.ptr), Arg::Ptr(d_g_q8.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                Arg::Ptr(d_rn.ptr), Arg::U32(cols as u32), Arg::U32(cols as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
            ]).unwrap();
        }
        ctx.synchronize().unwrap();
        let down_ref = download_f32(&d_down_ref, m * r_down);
        let down_q8 = download_f32(&d_down_q8, m * r_down);
        let g_ref = download_f32(&d_g_ref, m * r_inj);
        let g_q8 = download_f32(&d_g_q8, m * r_inj);
        check_close(&down_q8, &down_ref, &format!("hc8_down m={m}"), 0.012);
        check_close(&g_q8, &g_ref, &format!("hc8_inj m={m}"), 0.012);
    }
}

#[test]
fn test_mma_hc8_down_inject() {
    let Some(ctx) = gpu_ctx() else { return };
    let mvp = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng(0x243f6a8885a308d3);

    let (r_down, r_inj, cols, streams) = (320usize, 4usize, 10240usize, 4usize);
    let max_m = 40usize;
    let w_down_bf16 = make_bf16_matrix(&mut rng, r_down, cols, 0.05);
    let w_inj_bf16 = make_bf16_matrix(&mut rng, r_inj, cols, 0.05);
    let w_down_hc8 = quantize_bf16_hc8(&w_down_bf16, r_down, cols);
    let w_inj_hc8 = quantize_bf16_hc8(&w_inj_bf16, r_inj, cols);
    let rn = make_f32_vec(&mut rng, max_m * cols, 1.0);

    let d_wdown_bf = ctx.alloc(w_down_bf16.len()).unwrap();
    let d_winj_bf = ctx.alloc(w_inj_bf16.len()).unwrap();
    let d_wdown_q8 = ctx.alloc(w_down_hc8.len()).unwrap();
    let d_winj_q8 = ctx.alloc(w_inj_hc8.len()).unwrap();
    let d_rn = ctx.alloc(rn.len() * 4).unwrap();
    let d_down_ref = ctx.alloc(max_m * r_down * 4).unwrap();
    let d_g_ref = ctx.alloc(max_m * r_inj * 4).unwrap();
    let d_down_q8 = ctx.alloc(max_m * r_down * 4).unwrap();
    let d_g_q8 = ctx.alloc(max_m * r_inj * 4).unwrap();

    d_wdown_bf.upload(&w_down_bf16).unwrap();
    d_winj_bf.upload(&w_inj_bf16).unwrap();
    d_wdown_q8.upload(&w_down_hc8).unwrap();
    d_winj_q8.upload(&w_inj_hc8).unwrap();
    d_rn.upload(&to_bytes_f32(&rn)).unwrap();

    for m in [16usize, 40] {
        let gx = ((r_down + r_inj) as u32).div_ceil(64);
        let gy = (m as u32).div_ceil(32);
        unsafe {
            mvp.launch("mma_hc_down_inject_bf16", (gx, gy, 1), (128, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wdown_bf.ptr), Arg::Ptr(d_down_ref.ptr), Arg::U32(r_down as u32), Arg::U32(r_down as u32),
                Arg::Ptr(d_winj_bf.ptr), Arg::Ptr(d_g_ref.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                Arg::Ptr(d_rn.ptr), Arg::U32(cols as u32), Arg::U32(cols as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
            ]).unwrap();
            mvp.launch("mma_hc8_down_inject", (gx, gy, 1), (128, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wdown_q8.ptr), Arg::Ptr(d_down_q8.ptr), Arg::U32(r_down as u32), Arg::U32(r_down as u32),
                Arg::Ptr(d_winj_q8.ptr), Arg::Ptr(d_g_q8.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                Arg::Ptr(d_rn.ptr), Arg::U32(cols as u32), Arg::U32(cols as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
            ]).unwrap();
        }
        ctx.synchronize().unwrap();
        let down_ref = download_f32(&d_down_ref, m * r_down);
        let down_q8 = download_f32(&d_down_q8, m * r_down);
        let g_ref = download_f32(&d_g_ref, m * r_inj);
        let g_q8 = download_f32(&d_g_q8, m * r_inj);
        check_close(&down_q8, &down_ref, &format!("mma_hc8_down m={m}"), 0.012);
        check_close(&g_q8, &g_ref, &format!("mma_hc8_inj m={m}"), 0.012);
    }
}

#[test]
fn test_hc8_up_mix_cols_and_mma() {
    let Some(ctx) = gpu_ctx() else { return };
    let mvp = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let fw = Program::load(&ctx, file("forward.cu").unwrap()).unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng(0x13198a2e03707344);

    let (h, streams, rank) = (2560usize, 4usize, 320usize);
    let ch = h * streams;
    let max_m = 40usize;
    let w_up_bf16 = make_bf16_matrix(&mut rng, ch, rank, 0.15);
    let w_up_hc8 = quantize_bf16_hc8(&w_up_bf16, ch, rank);
    let down = make_f32_vec(&mut rng, max_m * rank, 1.0);
    let rn = make_f32_vec(&mut rng, max_m * ch, 1.0);

    let d_wup_bf = ctx.alloc(w_up_bf16.len()).unwrap();
    let d_wup_q8 = ctx.alloc(w_up_hc8.len()).unwrap();
    let d_down = ctx.alloc(down.len() * 4).unwrap();
    let d_rn = ctx.alloc(rn.len() * 4).unwrap();
    let d_up_ref = ctx.alloc(max_m * ch * 4).unwrap();
    let d_up_q8 = ctx.alloc(max_m * ch * 4).unwrap();
    let d_u_ref = ctx.alloc(max_m * h * 4).unwrap();
    let d_u_q8 = ctx.alloc(max_m * h * 4).unwrap();

    d_wup_bf.upload(&w_up_bf16).unwrap();
    d_wup_q8.upload(&w_up_hc8).unwrap();
    d_down.upload(&to_bytes_f32(&down)).unwrap();
    d_rn.upload(&to_bytes_f32(&rn)).unwrap();

    for m in [1usize, 3, 4, 8] {
        let gx = (h as u32).div_ceil(8);
        let q8_k = if m == 1 { "hc8_up_mix_m1" } else if m <= 4 { "hc8_up_mix_m4" } else { "hc8_up_mix_cols" };
        unsafe {
            mvp.launch("hc_up_mix_cols", (gx, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wup_bf.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::Ptr(d_rn.ptr),
                Arg::U32(ch as u32), Arg::Ptr(d_u_ref.ptr), Arg::U32(h as u32), Arg::U32(h as u32),
                Arg::U32(streams as u32), Arg::U32(rank as u32), Arg::U32(m as u32),
            ]).unwrap();
            mvp.launch(q8_k, (gx, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wup_q8.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::Ptr(d_rn.ptr),
                Arg::U32(ch as u32), Arg::Ptr(d_u_q8.ptr), Arg::U32(h as u32), Arg::U32(h as u32),
                Arg::U32(streams as u32), Arg::U32(rank as u32), Arg::U32(m as u32),
            ]).unwrap();
        }
        ctx.synchronize().unwrap();
        let u_ref = download_f32(&d_u_ref, m * h);
        let u_q8 = download_f32(&d_u_q8, m * h);
        check_close(&u_q8, &u_ref, &format!("hc8_up_mix m={m}"), 0.012);
    }

    for m in [16usize, 40] {
        for j0 in (0..m).step_by(8) {
            let n = (m - j0).min(8) as u32;
            unsafe {
                mvp.launch("matvec_cols_bf16", ((ch as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wup_bf.ptr), Arg::Ptr(d_down.ptr + (j0 * rank * 4) as u64), Arg::U32(rank as u32),
                    Arg::Ptr(d_up_ref.ptr + (j0 * ch * 4) as u64), Arg::U32(ch as u32), Arg::U32(ch as u32),
                    Arg::U32(rank as u32), Arg::U32(n),
                ]).unwrap();
            }
        }
        unsafe {
            fw.launch("hc_mix_rows", ((h as u32).div_ceil(256), m as u32, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_rn.ptr), Arg::Ptr(d_up_ref.ptr), Arg::Ptr(d_u_ref.ptr),
                Arg::U32(h as u32), Arg::U32(streams as u32),
            ]).unwrap();
            mvp.launch("mma_hc8_cols", ((ch as u32).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wup_q8.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32),
                Arg::Ptr(d_up_q8.ptr), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::U32(rank as u32), Arg::U32(m as u32),
            ]).unwrap();
            fw.launch("hc_mix_rows", ((h as u32).div_ceil(256), m as u32, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_rn.ptr), Arg::Ptr(d_up_q8.ptr), Arg::Ptr(d_u_q8.ptr),
                Arg::U32(h as u32), Arg::U32(streams as u32),
            ]).unwrap();
        }
        ctx.synchronize().unwrap();
        let u_ref = download_f32(&d_u_ref, m * h);
        let u_q8 = download_f32(&d_u_q8, m * h);
        check_close(&u_q8, &u_ref, &format!("mma_hc8_up_mix m={m}"), 0.012);
    }
}

#[test]
#[ignore]
fn bench_hc8_vs_bf16() {
    let Some(ctx) = gpu_ctx() else { return };
    let mvp = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let fw = Program::load(&ctx, file("forward.cu").unwrap()).unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng(0x55aa55aa12345678);

    let (h, streams, rank, r_inj) = (2560usize, 4usize, 320usize, 4usize);
    let ch = h * streams;
    let max_m = 512usize;
    let w_down_bf16 = make_bf16_matrix(&mut rng, rank, ch, 0.05);
    let w_inj_bf16 = make_bf16_matrix(&mut rng, r_inj, ch, 0.05);
    let w_up_bf16 = make_bf16_matrix(&mut rng, ch, rank, 0.15);
    let w_down_hc8 = quantize_bf16_hc8(&w_down_bf16, rank, ch);
    let w_inj_hc8 = quantize_bf16_hc8(&w_inj_bf16, r_inj, ch);
    let w_up_hc8 = quantize_bf16_hc8(&w_up_bf16, ch, rank);

    let d_wdown_bf = ctx.alloc(w_down_bf16.len()).unwrap();
    let d_winj_bf = ctx.alloc(w_inj_bf16.len()).unwrap();
    let d_wup_bf = ctx.alloc(w_up_bf16.len()).unwrap();
    let d_wdown_q8 = ctx.alloc(w_down_hc8.len()).unwrap();
    let d_winj_q8 = ctx.alloc(w_inj_hc8.len()).unwrap();
    let d_wup_q8 = ctx.alloc(w_up_hc8.len()).unwrap();
    let d_rn = ctx.alloc(max_m * ch * 4).unwrap();
    let d_down = ctx.alloc(max_m * rank * 4).unwrap();
    let d_g = ctx.alloc(max_m * r_inj * 4).unwrap();
    let d_up = ctx.alloc(max_m * ch * 4).unwrap();
    let d_u = ctx.alloc(max_m * h * 4).unwrap();

    d_wdown_bf.upload(&w_down_bf16).unwrap();
    d_winj_bf.upload(&w_inj_bf16).unwrap();
    d_wup_bf.upload(&w_up_bf16).unwrap();
    d_wdown_q8.upload(&w_down_hc8).unwrap();
    d_winj_q8.upload(&w_inj_hc8).unwrap();
    d_wup_q8.upload(&w_up_hc8).unwrap();
    d_rn.upload(&to_bytes_f32(&make_f32_vec(&mut rng, max_m * ch, 1.0))).unwrap();

    let t_ev = |f: &dyn Fn()| -> f64 {
        for _ in 0..20 {
            f();
        }
        ctx.synchronize().unwrap();
        let ev0 = ctx.event().unwrap();
        let ev1 = ctx.event().unwrap();
        ev0.record(Some(&stream)).unwrap();
        for _ in 0..200 {
            f();
        }
        ev1.record(Some(&stream)).unwrap();
        ev1.synchronize().unwrap();
        (ev1.since(&ev0).unwrap() as f64 / 200.0) * 1e3 // microseconds
    };

    for m in [1usize, 3, 4, 8, 64, 512] {
        let bf_down = t_ev(&|| unsafe {
            if m < 16 {
                mvp.launch("hc_down_inject_cols_fast_bf16", (((rank + r_inj) as u32).div_ceil(2), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wdown_bf.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::U32(rank as u32),
                    Arg::Ptr(d_winj_bf.ptr), Arg::Ptr(d_g.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                    Arg::Ptr(d_rn.ptr), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
                ]).unwrap();
            } else {
                mvp.launch("mma_hc_down_inject_bf16", (((rank + r_inj) as u32).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wdown_bf.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::U32(rank as u32),
                    Arg::Ptr(d_winj_bf.ptr), Arg::Ptr(d_g.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                    Arg::Ptr(d_rn.ptr), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
                ]).unwrap();
            }
        });
        let q8_down = t_ev(&|| unsafe {
            if m < 16 {
                let entry = if m == 1 { "hc8_down_inject_m1" } else if m <= 4 { "hc8_down_inject_m4" } else { "hc8_down_inject_cols" };
                mvp.launch(entry, (((rank + r_inj) as u32).div_ceil(2), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wdown_q8.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::U32(rank as u32),
                    Arg::Ptr(d_winj_q8.ptr), Arg::Ptr(d_g.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                    Arg::Ptr(d_rn.ptr), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
                ]).unwrap();
            } else {
                mvp.launch("mma_hc8_down_inject", (((rank + r_inj) as u32).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wdown_q8.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::U32(rank as u32),
                    Arg::Ptr(d_winj_q8.ptr), Arg::Ptr(d_g.ptr), Arg::U32(r_inj as u32), Arg::U32(r_inj as u32),
                    Arg::Ptr(d_rn.ptr), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::F32(streams as f32), Arg::U32(m as u32),
                ]).unwrap();
            }
        });
        let bf_up = t_ev(&|| unsafe {
            if m < 16 {
                mvp.launch("matvec_cols_bf16", ((ch as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wup_bf.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::Ptr(d_up.ptr),
                    Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::U32(rank as u32), Arg::U32(m as u32),
                ]).unwrap();
                fw.launch("hc_mix_rows", ((h as u32).div_ceil(256), m as u32, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_rn.ptr), Arg::Ptr(d_up.ptr), Arg::Ptr(d_u.ptr), Arg::U32(h as u32), Arg::U32(streams as u32),
                ]).unwrap();
            } else {
                for j0 in (0..m).step_by(8) {
                    mvp.launch("matvec_cols_bf16", ((ch as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                        Arg::Ptr(d_wup_bf.ptr), Arg::Ptr(d_down.ptr + (j0 * rank * 4) as u64), Arg::U32(rank as u32),
                        Arg::Ptr(d_up.ptr + (j0 * ch * 4) as u64), Arg::U32(ch as u32), Arg::U32(ch as u32),
                        Arg::U32(rank as u32), Arg::U32((m - j0).min(8) as u32),
                    ]).unwrap();
                }
                fw.launch("hc_mix_rows", ((h as u32).div_ceil(256), m as u32, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_rn.ptr), Arg::Ptr(d_up.ptr), Arg::Ptr(d_u.ptr), Arg::U32(h as u32), Arg::U32(streams as u32),
                ]).unwrap();
            }
        });
        let q8_up = t_ev(&|| unsafe {
            if m < 16 {
                let entry = if m == 1 { "hc8_up_mix_m1" } else if m <= 4 { "hc8_up_mix_m4" } else { "hc8_up_mix_cols" };
                mvp.launch(entry, ((h as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wup_q8.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32), Arg::Ptr(d_rn.ptr),
                    Arg::U32(ch as u32), Arg::Ptr(d_u.ptr), Arg::U32(h as u32), Arg::U32(h as u32),
                    Arg::U32(streams as u32), Arg::U32(rank as u32), Arg::U32(m as u32),
                ]).unwrap();
            } else {
                mvp.launch("mma_hc8_cols", ((ch as u32).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_wup_q8.ptr), Arg::Ptr(d_down.ptr), Arg::U32(rank as u32),
                    Arg::Ptr(d_up.ptr), Arg::U32(ch as u32), Arg::U32(ch as u32), Arg::U32(rank as u32), Arg::U32(m as u32),
                ]).unwrap();
                fw.launch("hc_mix_rows", ((h as u32).div_ceil(256), m as u32, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_rn.ptr), Arg::Ptr(d_up.ptr), Arg::Ptr(d_u.ptr), Arg::U32(h as u32), Arg::U32(streams as u32),
                ]).unwrap();
            }
        });
        eprintln!(
            "m={m:2}: down bf={bf_down:5.2} q8={q8_down:5.2} ({:.2}x) | up bf={bf_up:5.2} q8={q8_up:5.2} ({:.2}x) | tot {:.2}x",
            bf_down / q8_down,
            bf_up / q8_up,
            (bf_down + bf_up) / (q8_down + q8_up),
        );
    }
}
