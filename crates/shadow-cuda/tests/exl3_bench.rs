//! Timing of the EXL3 kernels on the shapes a decode token launches (`#[ignore]`d: `cargo test --release -p shadow-cuda --test
//! exl3_bench -- --ignored --nocapture`).  Random blobs; every shape is launched 50 times between two events.  Prints microseconds a launch,
//! G weights a second and the bandwidth that implies (K bits a weight), against the card's memory roof.

use shadow_cuda::kernels::{file, Arg, Program};
use shadow_cuda::{nvrtc, Context};
use shadow_model::exl3::blob_len;

fn blob(inf: usize, outf: usize, bits: usize) -> Vec<u8> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut b = vec![0u8; blob_len(inf, outf, bits)];
    for i in 0..inf + outf {
        let v: u16 = if i < inf { 0x3c00 } else { 0x3800 }; // 1.0, 0.5
        b[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
    }
    let t0 = 2 * inf + 2 * outf + 4;
    for c in b[t0..].chunks_exact_mut(8) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        c.copy_from_slice(&seed.to_le_bytes());
    }
    b
}

#[test]
#[ignore]
fn time_the_decode_shapes() {
    let Ok(ctx) = Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    if nvrtc::version().is_err() {
        return eprintln!("SKIPPED, not passed: no NVRTC");
    }
    let prog = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
    let suffix = std::env::var("EXL3_KERNEL").unwrap_or_default(); // e.g. "_v2": matvec_exl3_v2_4
    // (name, in, out, K, matrices in the table (0: a single-matrix launch))
    let shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("J qkv", 2560, 10240, 4, 0), ("J z", 2560, 6144, 4, 0), ("J out", 6144, 2560, 4, 0), ("K q", 2560, 12288, 4, 0), ("K o", 6144, 2560, 4, 0),
        ("K k/v", 2560, 512, 4, 0), ("shexp gate/up", 2560, 640, 4, 0), ("shexp down", 640, 2560, 4, 0), ("lm_head", 2560, 248320, 5, 0),
        ("experts gate+up x20", 2560, 640, 3, 20), ("experts down x10", 640, 2560, 3, 10),
    ];
    let cols_m: u32 = std::env::var("EXL3_COLS").ok().and_then(|v| v.parse().ok()).unwrap_or(0); // >0: the multi-input kernels with that many inputs
    let mma = std::env::var("EXL3_MMA").is_ok_and(|v| v == "1"); // with EXL3_COLS: the tensor-core kernel
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    eprintln!("{:22} {:>9} {:>10} {:>9}", "shape", "us/launch", "G weights/s", "GB/s");
    for &(name, inf, outf, bits, count) in shapes {
        let n = count.max(1);
        let bl = blob(inf, outf, bits);
        let bufs: Vec<_> = (0..n.min(4)).map(|_| { let w = ctx.alloc(bl.len()).unwrap(); w.upload(&bl).unwrap(); w }).collect();
        let tab: Vec<u8> = (0..n).flat_map(|i| bufs[i % bufs.len()].ptr.to_le_bytes()).collect();
        let dt = ctx.alloc(tab.len()).unwrap();
        dt.upload(&tab).unwrap();
        let dx = ctx.alloc(inf * 4 * n.max(cols_m as usize).max(8)).unwrap();
        dx.upload(&vec![0x3du8; inf * 4 * n.max(cols_m as usize).max(8)]).unwrap();
        let dy = ctx.alloc(outf * 4 * n.max(cols_m as usize).max(8)).unwrap();
        let launch = || {
            // SAFETY: random blobs of this shape; inf floats in, outf floats out (a table: a matrix a blockIdx.y).
            unsafe {
                if count == 0 && cols_m > 0 && mma {
                    prog.launch(&format!("mma_cols_exl3{suffix}_{bits}"), ((outf / 128) as u32, cols_m.div_ceil(64), 1), (256, 1, 1), 0, None,
                                &[Arg::Ptr(bufs[0].ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(cols_m)]).unwrap()
                } else if count == 0 && cols_m > 0 {
                    prog.launch(&format!("matvec_cols_exl3{suffix}_{bits}"), ((outf / 128) as u32, 1, 1), (threads, 1, 1), 0, None,
                                &[Arg::Ptr(bufs[0].ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(cols_m)]).unwrap()
                } else if count == 0 {
                    prog.launch(&format!("matvec_exl3{suffix}_{bits}"), ((outf / 128) as u32, 1, 1), (threads, 1, 1), (inf * 4) as u32, None,
                                &[Arg::Ptr(bufs[0].ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap()
                } else {
                    prog.launch(&format!("matvec_batch_exl3{suffix}_{bits}"), ((outf / 128) as u32, count as u32, 1), (threads, 1, 1), (inf * 4) as u32, None,
                                &[Arg::Ptr(dt.ptr), Arg::Ptr(dx.ptr), Arg::U32(0), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap()
                }
            }
        };
        for _ in 0..5 {
            launch();
        }
        ctx.synchronize().unwrap();
        let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
        e0.record(None).unwrap();
        let reps = 50;
        for _ in 0..reps {
            launch();
        }
        e1.record(None).unwrap();
        e1.synchronize().unwrap();
        let us = e1.since(&e0).unwrap() as f64 * 1e3 / reps as f64;
        let weights = (inf * outf * n) as f64;
        eprintln!("{name:22} {us:9.1} {:10.1} {:9.1}", weights / us / 1e3, weights * bits as f64 / 8.0 / us / 1e3);
    }
}
