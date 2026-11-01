//! The reference GPU matvec against the CPU, on real tensors of the model: for every quant type, the first tensor of
//! that type (one expert's slice for a routed-expert tensor), a pseudo-random x, and y = W x on both sides.  The GPU
//! sums in f32 in a different order than the CPU's f64, so the bound is relative to the size of the sum:
//! |gpu - cpu| <= 1e-4 * sum |w x| + 1e-6.  Also prints each kernel's time and effective bandwidth - the reference
//! kernel decodes per weight and is slow on purpose; this is the number the fast one will be measured against.

use shadow_cuda::kernels::{file, Arg, Program, FAST_KERNELS, QUANT_KERNELS};
use shadow_cuda::{nvrtc, Context};
use shadow_model::gguf::{Gguf, TensorInfo};
use shadow_model::quant::dequantize;
use std::io::{Read, Seek, SeekFrom};

const MODEL: &str =
    "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";

fn read(g: &Gguf, t: &TensorInfo, bytes: usize) -> Vec<u8> {
    let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
    let mut b = vec![0u8; bytes];
    f.seek(SeekFrom::Start(t.offset)).unwrap();
    f.read_exact(&mut b).unwrap();
    b
}

#[test]
fn gpu_matvec_matches_the_cpu_on_real_tensors() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    if let Err(e) = nvrtc::version() {
        return eprintln!("SKIPPED, not passed: no NVRTC ({e})");
    }
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let prog = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    for &(ty, _, _, _, entry) in QUANT_KERNELS {
        // a 2-D tensor of this type if there is one, else one expert's slice of a 3-D one
        let Some(t) = g.tensors.iter().filter(|t| t.ty.name() == ty && t.dims.len() >= 2 && t.dims[1] > 1)
            .min_by_key(|t| (t.dims.len(), t.name.contains("token_embd") || t.name.contains("output.")))
        else {
            continue;
        };
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let nbytes = (t.nbytes() / if t.dims.len() == 3 { t.dims[2] } else { 1 }) as usize;
        let w = read(&g, t, nbytes);
        let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();

        let mut row = vec![0f32; cols];
        let rb = nbytes / rows;
        let (mut cpu, mut mag) = (vec![0f64; rows], vec![0f64; rows]);
        for r in 0..rows {
            dequantize(t.ty, &w[r * rb..(r + 1) * rb], &mut row).unwrap();
            cpu[r] = row.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
            mag[r] = row.iter().zip(&x).map(|(&a, &b)| (a as f64 * b as f64).abs()).sum();
        }

        let dw = ctx.alloc(w.len()).unwrap();
        let dx = ctx.alloc(cols * 4).unwrap();
        let dy = ctx.alloc(rows * 4).unwrap();
        dw.upload(&w).unwrap();
        dx.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let (start, end) = (ctx.event().unwrap(), ctx.event().unwrap());
        let launch = || {
            // SAFETY: rows x cols weights in whole rows, cols f32 in, rows f32 out; the kernel bounds-checks rows.
            unsafe {
                prog.launch(entry, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(rows as u32), Arg::U32(cols as u32)])
                    .unwrap();
            }
        };
        launch(); // warm
        start.record(None).unwrap();
        for _ in 0..10 {
            launch();
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let ms = end.since(&start).unwrap() / 10.0;
        let mut yb = vec![0u8; rows * 4];
        dy.download(&mut yb).unwrap();
        let gpu: Vec<f32> = yb.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut worst = 0f64;
        for r in 0..rows {
            let err = (gpu[r] as f64 - cpu[r]).abs();
            let bound = 1e-4 * mag[r] + 1e-6;
            assert!(err <= bound, "{ty} {} row {r}: gpu {} cpu {} (err {err:e} > {bound:e})", t.name, gpu[r], cpu[r]);
            worst = worst.max(err / (mag[r] + 1e-30));
        }
        eprintln!("{ty:<8} {:<34} {rows:>6} x {cols:<6} {:>7.3} ms {:>7.1} GB/s  worst rel err {worst:.1e}", t.name, ms,
                  w.len() as f64 / (ms as f64 * 1e6));

        // the fast kernel: same bound
        let &(_, fast, batch) = FAST_KERNELS.iter().find(|f| f.0 == ty).expect("a fast kernel for every type");
        let launch_fast = || {
            // SAFETY: as above.
            unsafe {
                prog.launch(fast, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(rows as u32), Arg::U32(cols as u32)])
                    .unwrap();
            }
        };
        launch_fast();
        start.record(None).unwrap();
        for _ in 0..10 {
            launch_fast();
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let ms_fast = end.since(&start).unwrap() / 10.0;
        dy.download(&mut yb).unwrap();
        let worst_fast = check(ty, &t.name, &yb, &cpu, &mag);
        eprintln!("{:<8} {:<34} {:>15} {:>7.3} ms {:>7.1} GB/s  worst rel err {worst_fast:.1e}  ({:.1}x)", "  fast", "", "",
                  ms_fast, w.len() as f64 / (ms_fast as f64 * 1e6), ms / ms_fast);

        // the batched kernel: 10 copies of the matrix, each output must match
        let n = 10usize;
        let tab = ctx.alloc(n * 8).unwrap();
        tab.upload(&(0..n).flat_map(|_| dw.ptr.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let dyn_ = ctx.alloc(n * rows * 4).unwrap();
        let launch_batch = || {
            // SAFETY: n table entries, each a rows x cols matrix; x shared (stride 0); n outputs of `rows` floats.
            unsafe {
                prog.launch(batch, ((rows as u32).div_ceil(8), n as u32, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(tab.ptr), Arg::Ptr(dx.ptr), Arg::U32(0), Arg::Ptr(dyn_.ptr), Arg::U32(rows as u32),
                              Arg::U32(rows as u32), Arg::U32(cols as u32)])
                    .unwrap();
            }
        };
        launch_batch();
        start.record(None).unwrap();
        for _ in 0..10 {
            launch_batch();
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let ms_batch = end.since(&start).unwrap() / 10.0 / n as f32;
        eprintln!("{:<8} {:<34} {:>15} {:>7.3} ms {:>7.1} GB/s  per matrix, 10 in one launch  ({:.1}x)", "  batch", "", "",
                  ms_batch, w.len() as f64 / (ms_batch as f64 * 1e6), ms / ms_batch);
        let mut yn = vec![0u8; n * rows * 4];
        dyn_.download(&mut yn).unwrap();
        for k in 0..n {
            check(ty, &t.name, &yn[k * rows * 4..(k + 1) * rows * 4], &cpu, &mag);
        }
    }
}

/// Every row within the bound; returns the worst relative error.
fn check(ty: &str, name: &str, yb: &[u8], cpu: &[f64], mag: &[f64]) -> f64 {
    let gpu: Vec<f32> = yb.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut worst = 0f64;
    for r in 0..cpu.len() {
        let err = (gpu[r] as f64 - cpu[r]).abs();
        let bound = 1e-4 * mag[r] + 1e-6;
        assert!(err <= bound, "{ty} {name} row {r}: gpu {} cpu {} (err {err:e} > {bound:e})", gpu[r], cpu[r]);
        worst = worst.max(err / (mag[r] + 1e-30));
    }
    worst
}

/// matvec_cols_* (one weight read for up to 8 inputs) against matvec_fast_* run once per input: bit for bit, on a
/// real tensor of every fast type, for m = 1..8; and the time for 8 inputs at once against 8 separate launches.
#[test]
fn several_inputs_at_once_match_one_at_a_time() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    if let Err(e) = nvrtc::version() {
        return eprintln!("SKIPPED, not passed: no NVRTC ({e})");
    }
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let prog = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mut tested = 0;
    for &(ty, fast, _) in FAST_KERNELS {
        let Some(t) = g.tensors.iter().filter(|t| t.ty.name() == ty && t.dims.len() >= 2 && t.dims[1] > 1)
            .min_by_key(|t| (t.dims.len(), t.name.contains("token_embd") || t.name.contains("output.")))
        else {
            continue;
        };
        let cols_entry = format!("matvec_cols_{}", ty.to_lowercase());
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        if cols % 8 != 0 {
            continue;
        }
        let nbytes = (t.nbytes() / if t.dims.len() == 3 { t.dims[2] } else { 1 }) as usize;
        let w = read(&g, t, nbytes);
        let x: Vec<f32> = (0..8 * cols).map(|_| rnd()).collect();
        let dw = ctx.alloc(w.len()).unwrap();
        let dx = ctx.alloc(8 * cols * 4).unwrap();
        let (dy1, dy8) = (ctx.alloc(8 * rows * 4).unwrap(), ctx.alloc(8 * rows * 4).unwrap());
        dw.upload(&w).unwrap();
        dx.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let one = |j: usize| unsafe {
            // SAFETY: rows x cols weights, input j of cols floats, output j of rows floats, all inside the buffers.
            prog.launch(fast, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                        &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr + (j * cols * 4) as u64),
                          Arg::Ptr(dy1.ptr + (j * rows * 4) as u64), Arg::U32(rows as u32), Arg::U32(cols as u32)]).unwrap();
        };
        let many = |m: usize| unsafe {
            // SAFETY: m inputs of cols floats at stride cols, m outputs of rows floats at stride rows.
            prog.launch(&cols_entry, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                        &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr), Arg::U32(cols as u32), Arg::Ptr(dy8.ptr), Arg::U32(rows as u32),
                          Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32)]).unwrap();
        };
        for j in 0..8 {
            one(j);
        }
        let mut want = vec![0u8; 8 * rows * 4];
        ctx.synchronize().unwrap();
        dy1.download(&mut want).unwrap();
        for m in 1..=8 {
            many(m);
            ctx.synchronize().unwrap();
            let mut got = vec![0u8; 8 * rows * 4];
            dy8.download(&mut got).unwrap();
            assert_eq!(&got[..m * rows * 4], &want[..m * rows * 4], "{cols_entry} m={m}: differs from {fast}");
        }
        let (start, end) = (ctx.event().unwrap(), ctx.event().unwrap());
        start.record(None).unwrap();
        for _ in 0..20 {
            for j in 0..8 {
                one(j);
            }
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let t8 = end.since(&start).unwrap() / 20.0;
        start.record(None).unwrap();
        for _ in 0..20 {
            many(8);
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let tc = end.since(&start).unwrap() / 20.0;
        eprintln!("{ty:8} {} ({rows}x{cols}): 8 inputs {:.3} ms one at a time, {:.3} ms at once ({:.1}x)", t.name, t8, tc,
                  t8 / tc);
        tested += 1;
    }
    assert!(tested >= 6, "only {tested} types found");
}

/// `mma_cols_*` (tensor cores, FP16 operands, the inexact prompt path) against `matvec_cols_*` (the reference sums), 64 inputs, on a real tensor:
/// the relative error of each output vector, and the time of 64 inputs as 8 launches of 8 against one launch.
#[test]
fn tensor_core_projections_are_close_to_the_exact_ones() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    if nvrtc::version().is_err() {
        return eprintln!("SKIPPED, not passed: no NVRTC");
    }
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let prog = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let m = 64usize;
    for (ty, exact, mma) in [("Q6_K", "matvec_cols_q6_k", "mma_cols_q6_k"), ("IQ4_XS", "matvec_cols_iq4_xs", "mma_cols_iq4_xs")] {
        let Some(t) = g.tensors.iter().filter(|t| t.ty.name() == ty && t.dims.len() == 2 && t.dims[0] % 64 == 0 && !t.name.contains("token_embd"))
            .max_by_key(|t| t.dims[1] * t.dims[0])
        else {
            continue;
        };
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let w = read(&g, t, t.nbytes() as usize);
        let x: Vec<f32> = (0..m * cols).map(|_| rnd()).collect();
        let dw = ctx.alloc(w.len()).unwrap();
        let dx = ctx.alloc(m * cols * 4).unwrap();
        let (dy_e, dy_m) = (ctx.alloc(m * rows * 4).unwrap(), ctx.alloc(m * rows * 4).unwrap());
        dw.upload(&w).unwrap();
        dx.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let run_exact = || unsafe {
            for j0 in (0..m).step_by(8) {
                // SAFETY: 8 inputs of cols floats, 8 outputs of rows floats, inside the buffers.
                prog.launch(exact, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr + (j0 * cols * 4) as u64), Arg::U32(cols as u32),
                              Arg::Ptr(dy_e.ptr + (j0 * rows * 4) as u64), Arg::U32(rows as u32), Arg::U32(rows as u32),
                              Arg::U32(cols as u32), Arg::U32(8)]).unwrap();
            }
        };
        let run_mma = || unsafe {
            // SAFETY: as above, all 64 inputs at once.
            prog.launch(mma, ((rows as u32).div_ceil(64), (m as u32).div_ceil(32), 1), (128, 1, 1), 0, None,
                        &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr), Arg::U32(cols as u32), Arg::Ptr(dy_m.ptr), Arg::U32(rows as u32),
                          Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32)]).unwrap();
        };
        run_exact();
        run_mma();
        ctx.synchronize().unwrap();
        let (mut ge, mut gm) = (vec![0u8; m * rows * 4], vec![0u8; m * rows * 4]);
        dy_e.download(&mut ge).unwrap();
        dy_m.download(&mut gm).unwrap();
        let (fe, fm): (Vec<f32>, Vec<f32>) = (
            ge.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            gm.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
        let mut worst = 0f64;
        for j in 0..m {
            let (a, b) = (&fe[j * rows..(j + 1) * rows], &fm[j * rows..(j + 1) * rows]);
            let num: f64 = a.iter().zip(b).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
            let den: f64 = a.iter().map(|p| (*p as f64).powi(2)).sum();
            worst = worst.max((num / den.max(1e-30)).sqrt());
        }
        let (start, end) = (ctx.event().unwrap(), ctx.event().unwrap());
        start.record(None).unwrap();
        for _ in 0..20 {
            run_exact();
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let te = end.since(&start).unwrap() / 20.0;
        start.record(None).unwrap();
        for _ in 0..20 {
            run_mma();
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let tm = end.since(&start).unwrap() / 20.0;
        eprintln!("{ty:7} {} ({rows}x{cols}): relative error {:.2e}; 64 inputs {:.3} ms exact (8 launches), {:.3} ms tensor cores ({:.1}x)", t.name, worst, te, tm, te / tm);
        assert!(worst < 5e-3, "{ty}: relative error {worst}");
    }
}

/// Decode-shape speed of the plain `matvec_cols_*` kernels on the model's real tensors: 4 inputs (a verify pass), the achieved weight bandwidth per type
/// and the share of the card's memory bandwidth (`cargo test --release -p shadow-cuda --test matvec -- --ignored --nocapture cols_bandwidth`).
#[test]
#[ignore]
fn cols_bandwidth() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED: no CUDA device ({e})"),
    };
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED: no model at {p}");
    }
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let prog = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let m = 4usize;
    let mut seen = std::collections::HashSet::new();
    for t in g.tensors.iter().filter(|t| t.dims.len() == 2 && t.dims[0] % 256 == 0 && t.nbytes() > (1 << 20) && !t.name.contains("exps")) {
        let ty = t.ty.name();
        if !seen.insert((ty.clone(), t.dims[0], t.dims[1])) || seen.len() > 24 {
            continue;
        }
        let entry = match ty.as_str() {
            "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_NL" | "IQ4_XS" | "Q8_0" | "BF16" | "F32" => format!("matvec_cols_{}", ty.to_lowercase()),
            _ => continue,
        };
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let w = read(&g, t, t.nbytes() as usize);
        let dw = ctx.alloc(w.len()).unwrap();
        dw.upload(&w).unwrap();
        let dx = ctx.alloc(m * cols * 4).unwrap();
        let dy = ctx.alloc(m * rows * 4).unwrap();
        dx.upload(&vec![1u8; m * cols * 4]).unwrap();
        let run = || unsafe {
            // SAFETY: rows x cols weights, m inputs of cols floats, m outputs of rows floats.
            prog.launch(&entry, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                        &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr), Arg::U32(cols as u32), Arg::Ptr(dy.ptr), Arg::U32(rows as u32),
                          Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32)]).unwrap();
        };
        run();
        ctx.synchronize().unwrap();
        let (start, end) = (ctx.event().unwrap(), ctx.event().unwrap());
        start.record(None).unwrap();
        for _ in 0..50 {
            run();
        }
        end.record(None).unwrap();
        end.synchronize().unwrap();
        let ms = end.since(&start).unwrap() as f64 / 50.0;
        let gbs = w.len() as f64 / (ms / 1e3) / 1e9;
        println!("{ty:7} {:36} {rows:6}x{cols:5} {:7.1} MB  {:7.1} us  {:6.1} GB/s  ({:4.1}% of 672)", t.name, w.len() as f64 / 1e6, ms * 1e3, gbs, 100.0 * gbs / 672.0);
    }
}
