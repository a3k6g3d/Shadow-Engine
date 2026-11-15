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
    let split: usize = std::env::var("EXL3_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(0); // with EXL3_COLS: split-K over that many blocks
    let pre = std::env::var("EXL3_PRE").is_ok_and(|v| v == "1"); // with EXL3_MMA: the rotation pre-pass + the pre kernel
    let mma = std::env::var("EXL3_MMA").is_ok_and(|v| v == "1"); // with EXL3_COLS: the tensor-core kernel
    let threads: u32 = std::env::var("EXL3_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
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
        let dpart = ctx.alloc(8 * 8 * outf * 4).unwrap();
        let dxr = ctx.alloc(inf * 2 * cols_m.max(1) as usize).unwrap();
        let dy = ctx.alloc(outf * 4 * n.max(cols_m as usize).max(8)).unwrap();
        let launch = || {
            // SAFETY: random blobs of this shape; inf floats in, outf floats out (a table: a matrix a blockIdx.y).
            unsafe {
                if count == 0 && cols_m > 0 && !mma && split > 0 {
                    // split-K: `nsplit` blocks along the input a 128-output slice, then the combine (nsplit = min(EXL3_SPLIT, the input's 128-blocks / 4))
                    let ns = split.min((inf / 128).div_ceil(4)).max(1) as u32;
                    prog.launch(&format!("matvec_cols_exl3_sk4_{bits}"), ((outf / 128) as u32, ns, 1), (1024, 1, 1), 0, None,
                                &[Arg::Ptr(bufs[0].ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dpart.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(cols_m)]).unwrap();
                    prog.launch("exl3_splitk_combine", ((outf / 128) as u32, 1, 1), (256, 1, 1), 0, None,
                                &[Arg::Ptr(bufs[0].ptr), Arg::Ptr(dpart.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(cols_m), Arg::U32(ns)]).unwrap()
                } else if count == 0 && cols_m > 0 && mma && pre {
                    // the rotation once (a pre-pass), then the tensor-core kernel on the rotated rows
                    prog.launch("exl3_rotate_rows", (((cols_m as usize * (inf / 128)) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                                &[Arg::Ptr(bufs[0].ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dxr.ptr), Arg::U32(inf as u32), Arg::U32(cols_m)]).unwrap();
                    prog.launch(&format!("mma_cols_exl3_pre_{bits}"), ((outf / 128) as u32, cols_m.div_ceil(64), 1), (256, 1, 1), 0, None,
                                &[Arg::Ptr(bufs[0].ptr), Arg::Ptr(dxr.ptr), Arg::U32(0), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(cols_m)]).unwrap()
                } else if count == 0 && cols_m > 0 && mma {
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

/// The routed-expert groups of a verify pass: `n` groups (experts) of `tokens` tokens each, gate+up (2n groups, 2,560 -> 640) and down (n groups, 640 -> 2,560), K = 2.
/// `EXL3_KERNEL=_s1|_s2|_s4` picks `matvec_grouped_exl3{suffix}_2`; threads 256 x the suffix's slices.
#[test]
#[ignore]
fn time_grouped_experts() {
    let Ok(ctx) = Context::new(0) else { return eprintln!("SKIPPED, not passed: no CUDA device") };
    if nvrtc::version().is_err() {
        return eprintln!("SKIPPED, not passed: no NVRTC");
    }
    let prog = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
    let suffix = std::env::var("EXL3_KERNEL").unwrap_or_else(|_| "_s4".into());
    let threads: u32 = 256 * suffix.rsplit('s').next().and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
    let toks: usize = std::env::var("EXL3_TOK").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let n = 40usize;
    eprintln!("{:22} {:>9}", "shape", "us/launch");
    for &(name, inf, outf, count) in &[("gate+up", 2560usize, 640usize, 2 * n), ("down", 640, 2560, n)] {
        let bl = blob(inf, outf, 2);
        let bufs: Vec<_> = (0..4).map(|_| { let w = ctx.alloc(bl.len()).unwrap(); w.upload(&bl).unwrap(); w }).collect();
        let tab: Vec<u8> = (0..count).flat_map(|i| bufs[i % 4].ptr.to_le_bytes()).collect();
        let dt = ctx.alloc(tab.len()).unwrap();
        dt.upload(&tab).unwrap();
        let mut rec = vec![0u32; count * 17];
        for g in 0..count {
            rec[g * 17] = toks as u32;
            for u in 0..toks {
                rec[g * 17 + 1 + u] = ((g * toks + u) % 8) as u32;
                rec[g * 17 + 9 + u] = (g * toks + u) as u32;
            }
        }
        let dg = ctx.alloc(rec.len() * 4).unwrap();
        dg.upload(&rec.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let dx = ctx.alloc(inf * 4 * 8).unwrap();
        dx.upload(&vec![0x3du8; inf * 4 * 8]).unwrap();
        let dy = ctx.alloc(outf * 4 * count * toks).unwrap();
        let launch = || {
            // SAFETY: `count` table entries and 17-u32 records; input rows < 8 of dx, output rows < count * toks of dy.
            unsafe {
                prog.launch(&format!("matvec_grouped_exl3{suffix}_2"), ((outf / 128) as u32, count as u32, 1), (threads, 1, 1), 0, None,
                            &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap()
            }
        };
        for _ in 0..5 {
            launch();
        }
        ctx.synchronize().unwrap();
        let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
        e0.record(None).unwrap();
        for _ in 0..50 {
            launch();
        }
        e1.record(None).unwrap();
        e1.synchronize().unwrap();
        eprintln!("{name:22} {:9.1}", e1.since(&e0).unwrap() as f64 * 1e3 / 50.0);
    }
}
