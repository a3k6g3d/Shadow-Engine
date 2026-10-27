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

        let mut dw = ctx.alloc(w.len()).unwrap();
        let mut dx = ctx.alloc(cols * 4).unwrap();
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
        let mut tab = ctx.alloc(n * 8).unwrap();
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
