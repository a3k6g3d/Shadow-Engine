//! Timing of the EXL3 kernels on the shapes a decode token launches (`#[ignore]`d: `cargo test --release -p shadowzero-cuda --test
//! exl3_bench -- --ignored --nocapture`).  Random blobs; every shape is launched 50 times between two events.  Prints microseconds a launch,
//! G weights a second and the bandwidth that implies (K bits a weight), against the card's memory roof.

use shadowzero_cuda::kernels::{file, Arg, Program};
use shadowzero_cuda::{nvrtc, Context};
use shadowzero_model::exl3::blob_len;

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

/// Sweep `M=4`/`M=8` and slice counts across RTX 5070 (dev 0) and RTX 5060 Ti (dev 1) on real `mixD18` decode shapes.
#[test]
#[ignore]
fn bench_dual_gpu_shapes() {
    let devs: Vec<i32> = std::env::var("EXL3_DEVS").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<i32>| !v.is_empty())
        .unwrap_or_else(|| vec![0, 1]);
    for dev in devs {
        let Ok(ctx) = Context::new(dev) else { continue };
        if nvrtc::version().is_err() {
            return;
        }
        eprintln!("\n=== dev {dev}: {} ({} SMs) ===", ctx.name, ctx.sms);
        let prog = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
        // 1. Dense cols shapes: lm_head (K=5) and mixD18 dense projections (K=4,5)
        for &(label, inf, outf, bits) in &[
            ("lm_head 2560->248320", 2560usize, 248320usize, 5usize),
            ("proj    2560->12288 ", 2560, 12288, 4),
            ("proj    2560->12288 ", 2560, 12288, 5),
            ("proj    2560->10240 ", 2560, 10240, 4),
            ("proj    2560->6144  ", 2560, 6144, 4),
            ("proj    6144->2560  ", 6144, 2560, 4),
            ("proj     640->2560  ", 640, 2560, 4),
            ("proj    2560->512   ", 2560, 512, 4),
        ] {
            if dev == 1 && outf == 248320 {
                continue; // lm_head runs on dev 0 only
            }
            let bl = blob(inf, outf, bits);
            let w = ctx.alloc(bl.len()).unwrap();
            w.upload(&bl).unwrap();
            let dx = ctx.alloc(inf * 4 * 8).unwrap();
            dx.upload(&vec![0x3du8; inf * 4 * 8]).unwrap();
            let dxr = ctx.alloc(inf * 4 * 8).unwrap();
            let dy = ctx.alloc(outf * 4 * 8).unwrap();
            let dpart = ctx.alloc(8 * 8 * outf * 4).unwrap();
            let grid = (outf / 128) as u32;
            let blocks = (outf / 128).max(1);
            let nsplit = if blocks <= 24 && inf >= 2048 { (64 / blocks).min(8).min((inf / 128).div_ceil(4)).max(1) } else { 1 };
            let ms: &[usize] = if outf == 248320 { &[1, 2, 3, 4] } else { &[1, 3, 4] };
            for &m in ms {
                eprintln!("  {label} K={bits} m={m}:");
                let variants: Vec<(&str, u32)> = vec![("s4", 1024), ("m4s4", 1024)];
                for (suf, thr) in variants {
                    let kname = format!("matvec_cols_exl3_{suf}_{bits}");
                    let launch = || unsafe {
                        prog.launch(&kname, (grid, 1, 1), (thr, 1, 1), 0, None,
                                    &[Arg::Ptr(w.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dy.ptr), Arg::U32(outf as u32),
                                      Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32)]).unwrap();
                    };
                    for _ in 0..5 { launch(); }
                    ctx.synchronize().unwrap();
                    let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
                    e0.record(None).unwrap();
                    let reps = 40;
                    for _ in 0..reps { launch(); }
                    e1.record(None).unwrap();
                    e1.synchronize().unwrap();
                    let us = e1.since(&e0).unwrap() as f64 * 1e3 / reps as f64;
                    let wbytes = (inf * outf * bits) as f64 / 8.0;
                    eprintln!("    {:10} {:8.1} us  ({:6.1} GB/s)", suf, us, wbytes / us / 1e3);
                }
                if outf == 248320 {
                    let kname = format!("matvec_cols_exl3_pre_m4s1_{bits}");
                    let launch = || unsafe {
                        prog.launch("exl3_rotate_rows_f32", (((m * (inf / 128)) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                                    &[Arg::Ptr(w.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dxr.ptr), Arg::U32(inf as u32), Arg::U32(m as u32)]).unwrap();
                        prog.launch(&kname, (grid, 1, 1), (256, 1, 1), 0, None,
                                    &[Arg::Ptr(w.ptr), Arg::Ptr(dxr.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32),
                                      Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32)]).unwrap();
                    };
                    for _ in 0..5 { launch(); }
                    ctx.synchronize().unwrap();
                    let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
                    e0.record(None).unwrap();
                    let reps = 40;
                    for _ in 0..reps { launch(); }
                    e1.record(None).unwrap();
                    e1.synchronize().unwrap();
                    let us = e1.since(&e0).unwrap() as f64 * 1e3 / reps as f64;
                    let wbytes = (inf * outf * bits) as f64 / 8.0;
                    eprintln!("    {:10} {:8.1} us  ({:6.1} GB/s)", "pre_m4s1", us, wbytes / us / 1e3);
                }
                if nsplit > 1 {
                    for suf in ["sk4", "sk4_m4"] {
                        let kname = format!("matvec_cols_exl3_{suf}_{bits}");
                        let launch = || unsafe {
                            prog.launch(&kname, (grid, nsplit as u32, 1), (1024, 1, 1), 0, None,
                                        &[Arg::Ptr(w.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dpart.ptr),
                                          Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32)]).unwrap();
                            prog.launch("exl3_splitk_combine", (grid, 1, 1), (256, 1, 1), 0, None,
                                        &[Arg::Ptr(w.ptr), Arg::Ptr(dpart.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32),
                                          Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32), Arg::U32(nsplit as u32)]).unwrap();
                        };
                        for _ in 0..5 { launch(); }
                        ctx.synchronize().unwrap();
                        let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
                        e0.record(None).unwrap();
                        let reps = 40;
                        for _ in 0..reps { launch(); }
                        e1.record(None).unwrap();
                        e1.synchronize().unwrap();
                        let us = e1.since(&e0).unwrap() as f64 * 1e3 / reps as f64;
                        let wbytes = (inf * outf * bits) as f64 / 8.0;
                        eprintln!("    {:10} {:8.1} us  ({:6.1} GB/s)", suf, us, wbytes / us / 1e3);
                    }
                }
                if m == 1 {
                    for ks in [1u32, 2, 4] {
                        let kname = format!("matvec_exl3_s{ks}_{bits}");
                        let launch = || unsafe {
                            prog.launch(&kname, (grid, 1, 1), (256 * ks, 1, 1), (inf * 4) as u32, None,
                                        &[Arg::Ptr(w.ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap();
                        };
                        for _ in 0..5 { launch(); }
                        ctx.synchronize().unwrap();
                        let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
                        e0.record(None).unwrap();
                        let reps = 40;
                        for _ in 0..reps { launch(); }
                        e1.record(None).unwrap();
                        e1.synchronize().unwrap();
                        let us = e1.since(&e0).unwrap() as f64 * 1e3 / reps as f64;
                        let wbytes = (inf * outf * bits) as f64 / 8.0;
                        eprintln!("    mv2_s{:<5} {:8.1} us  ({:6.1} GB/s)", ks, us, wbytes / us / 1e3);
                    }
                }
            }
        }
        // 2. Prompt MMA workloads (grouped MoE pre-rotated + dense pre-rotated)
        for &(label, inf, outf, bits, count) in &[
            ("mma_grp_sparse 2560->640 ", 2560usize, 640usize, 2usize, 256usize),
            ("mma_grp_sparse  640->2560", 640, 2560, 2, 128),
        ] {
            let bl = blob(inf, outf, bits);
            let bufs: Vec<_> = (0..8).map(|_| { let w = ctx.alloc(bl.len()).unwrap(); w.upload(&bl).unwrap(); w }).collect();
            let mut tab = Vec::with_capacity(count * 8);
            let mut rec = vec![0u32; count * 129];
            let mut row_cursor = 0usize;
            for g in 0..count {
                // realistic 2,048-row prompt sparse histogram across 512 experts
                let n = match g % 10 {
                    0 => 0,
                    1..=4 => 4 + (g % 11),      // 1..16 (ntt=1)
                    5..=7 => 18 + (g % 13),     // 17..32 (ntt=2)
                    8 => 35 + (g % 12),         // 33..48 (ntt=3)
                    _ => 52 + (g % 13),         // 49..64 (ntt=4)
                };
                tab.extend_from_slice(&(if n == 0 { 0u64 } else { bufs[g % 8].ptr }).to_le_bytes());
                rec[g * 129] = n as u32;
                for u in 0..n {
                    rec[g * 129 + 1 + u] = ((g * 13 + u * 7) % 2048) as u32;
                    rec[g * 129 + 65 + u] = row_cursor as u32;
                    row_cursor += 1;
                }
            }
            let dt = ctx.alloc(tab.len()).unwrap();
            dt.upload(&tab).unwrap();
            let dg = ctx.alloc(rec.len() * 4).unwrap();
            dg.upload(&rec.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            let dx = ctx.alloc(2048 * inf * 4).unwrap();
            dx.upload(&vec![0x3du8; 2048 * inf * 4]).unwrap();
            let dxr = ctx.alloc(count * 64 * inf * 2).unwrap();
            let dy = ctx.alloc(row_cursor.max(1) * outf * 4).unwrap();
            let launch = || unsafe {
                prog.launch("exl3_rotate_groups", ((64 * (inf as u32 / 128)).div_ceil(8), count as u32, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dxr.ptr), Arg::U32(inf as u32)]).unwrap();
                prog.launch(&format!("mma_grouped_exl3_pre_{bits}"), ((outf / 128) as u32, count as u32, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dxr.ptr), Arg::U32(0), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap();
            };
            for _ in 0..3 { launch(); }
            ctx.synchronize().unwrap();
            let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
            e0.record(None).unwrap();
            let reps = 20;
            for _ in 0..reps { launch(); }
            e1.record(None).unwrap();
            e1.synchronize().unwrap();
            let us = e1.since(&e0).unwrap() as f64 * 1e3 / reps as f64;
            eprintln!("  {label} K={bits} count={count}: {:8.1} us", us);
        }
        // 3. Grouped MoE shapes: gate+up (2560->640, 80 groups) and down (640->2560, 40 groups), mixD18 K=2,3,4
        for &(label, inf, outf, count, bits) in &[
            ("gate+up 2560->640 ", 2560usize, 640usize, 80usize, 2usize),
            ("down     640->2560", 640, 2560, 40, 2),
            ("gate+up 2560->640 ", 2560, 640, 80, 3),
            ("down     640->2560", 640, 2560, 40, 3),
            ("gate+up 2560->640 ", 2560, 640, 80, 4),
            ("down     640->2560", 640, 2560, 40, 4),
        ] {
            let bl = blob(inf, outf, bits);
            let bufs: Vec<_> = (0..8).map(|_| { let w = ctx.alloc(bl.len()).unwrap(); w.upload(&bl).unwrap(); w }).collect();
            let tab: Vec<u8> = (0..count).flat_map(|i| bufs[i % 8].ptr.to_le_bytes()).collect();
            let dt = ctx.alloc(tab.len()).unwrap();
            dt.upload(&tab).unwrap();
            let dx = ctx.alloc(inf * 4 * 8).unwrap();
            dx.upload(&vec![0x3du8; inf * 4 * 8]).unwrap();
            let dy = ctx.alloc(outf * 4 * count * 4).unwrap();
            for &(pat, maxg) in &[("1tok", 1usize), ("mix2", 2), ("mix4", 4)] {
                let mut rec = vec![0u32; count * 17];
                for g in 0..count {
                    let n = match (maxg, g % 10) {
                        (1, _) => 1,
                        (2, 0 | 5) => 2,
                        (2, _) => 1,
                        (4, 0) => 4,
                        (4, 3) => 3,
                        (4, 6 | 9) => 2,
                        _ => 1,
                    };
                    rec[g * 17] = n as u32;
                    for u in 0..n {
                        rec[g * 17 + 1 + u] = ((g + u) % 4) as u32;
                        rec[g * 17 + 9 + u] = (g * 4 + u) as u32;
                    }
                }
                let dg = ctx.alloc(rec.len() * 4).unwrap();
                dg.upload(&rec.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                eprintln!("  {label} K={bits} {pat}:");
                let mut vars: Vec<(&str, u32)> = vec![("s4", 1024), ("m4s4", 1024)];
                if maxg <= 2 {
                    vars.push(("m2s4", 1024));
                }
                for (suf, thr) in vars {
                    let kname = format!("matvec_grouped_exl3_{suf}_{bits}");
                    let launch = || unsafe {
                        prog.launch(&kname, ((outf / 128) as u32, count as u32, 1), (thr, 1, 1), 0, None,
                                    &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dy.ptr),
                                      Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap();
                    };
                    for _ in 0..5 { launch(); }
                    ctx.synchronize().unwrap();
                    let (e0, e1) = (ctx.event().unwrap(), ctx.event().unwrap());
                    e0.record(None).unwrap();
                    let reps = 40;
                    for _ in 0..reps { launch(); }
                    e1.record(None).unwrap();
                    e1.synchronize().unwrap();
                    let us = e1.since(&e0).unwrap() as f64 * 1e3 / reps as f64;
                    eprintln!("    {:8} {:8.1} us", suf, us);
                }
            }
        }
    }
}
