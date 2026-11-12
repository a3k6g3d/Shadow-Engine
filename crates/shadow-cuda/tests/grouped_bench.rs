//! The grouped expert kernels (`matvec_grouped_*`, kernels/matvec.cu) of a verify pass, on real expert tensors of the model (docs/STATUS.md 114).
//!
//! For each (quantization, role: gate / up / down) of the model's routed experts: the first tensor of it, 28 of its experts read from the file, a verify pass's group table (28 groups: 18 of one
//! token, 7 of two, 2 of three, 1 of four, the inputs 4 token rows), and the launch the engine makes (grid = rows / 8 x groups, block 256).  The repo's kernel is timed with the L2
//! flushed between calls (the weights come from DRAM, as they do in the engine) and checked against an f64 CPU reference on sampled rows; `GROUPED_SRC=path` names a variant of
//! matvec.cu, compiled the same way, timed alongside and compared bit for bit with the repo's output.  Needs the model and a card; skips without them.  Run under tools/run_capped.ps1.

use shadow_cuda::kernels::{file, Arg, Program};
use shadow_cuda::Context;
use shadow_model::gguf::Gguf;
use shadow_model::quant::dequantize;
use std::io::{Read, Seek, SeekFrom};

const MODEL: &str = "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";
/// the token count of each group, a verify pass of 4 rows
const COUNTS: [usize; 28] = [1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1, 1, 2, 1, 4, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1];

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn grouped_kernels_on_real_experts() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let base = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    // a variant of matvec.cu: the registry's entries and headers with the source of the file named by GROUPED_SRC (leaked: the program wants a 'static file)
    let variant = std::env::var("GROUPED_SRC").ok().map(|path| {
        let src: &'static str = Box::leak(std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}")).into_boxed_str());
        let f = file("matvec.cu").unwrap();
        let v: &'static shadow_cuda::kernels::File = Box::leak(Box::new(shadow_cuda::kernels::File { file: "matvec_variant.cu", source: src, headers: f.headers, entries: f.entries }));
        Program::load(&ctx, v).unwrap()
    });
    let only = std::env::var("GROUPED_ONLY").unwrap_or_default();
    // rows a block of the variant covers (8 warps x its rows a warp: 8, 16, 32)
    let vrows: usize = std::env::var("GROUPED_VARIANT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let flush = std::cell::RefCell::new(ctx.alloc(160 << 20).unwrap());
    let (start, end) = (ctx.event().unwrap(), ctx.event().unwrap());
    let mut seen: Vec<(String, &str)> = vec![];
    let mut failures = vec![];
    let ng = COUNTS.len();
    eprintln!("{ng} groups, {} token-expert pairs, 4 token rows", COUNTS.iter().sum::<usize>());
    for t in g.tensors.iter().filter(|t| t.dims.len() == 3 && t.name.contains("_exps.weight")) {
        let role = if t.name.contains("ffn_gate") { "gate" } else if t.name.contains("ffn_up") { "up" } else { "down" };
        let ty = t.ty.name();
        if seen.iter().any(|(a, b)| *a == ty && *b == role) || !(only.is_empty() || only.contains(&ty)) {
            continue;
        }
        seen.push((ty.clone(), role));
        let entry = format!("matvec_grouped_{}", ty.to_lowercase());
        let (cols, rows, ne) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
        let per = t.nbytes() as usize / ne;
        // 28 experts, spread over the tensor
        let mut w = vec![0u8; ng * per];
        let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
        for i in 0..ng {
            f.seek(SeekFrom::Start(t.offset + ((i * 7 + 3) % ne * per) as u64)).unwrap();
            f.read_exact(&mut w[i * per..(i + 1) * per]).unwrap();
        }
        let dw = ctx.alloc(w.len()).unwrap();
        dw.upload(&w).unwrap();
        let wtab: Vec<u8> = (0..ng).flat_map(|i| (dw.ptr + (i * per) as u64).to_le_bytes()).collect();
        let dtab = ctx.alloc(wtab.len()).unwrap();
        dtab.upload(&wtab).unwrap();
        // the group table: 17 u32 a group (count, 8 input rows, 8 output rows); output row = group * 8 + slot
        let mut gt: Vec<u32> = vec![];
        for (gi, &n) in COUNTS.iter().enumerate() {
            let mut rec = vec![0u32; 17];
            rec[0] = n as u32;
            for s in 0..n {
                rec[1 + s] = ((gi + s) % 4) as u32;
                rec[9 + s] = (gi * 8 + s) as u32;
            }
            gt.extend(rec);
        }
        let dgt = ctx.alloc(gt.len() * 4).unwrap();
        dgt.upload(&gt.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let x: Vec<f32> = (0..4 * cols).map(|_| rnd()).collect();
        let dx = ctx.alloc(x.len() * 4).unwrap();
        dx.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let ybytes = ng * 8 * rows * 4;
        let dy = ctx.alloc(ybytes).unwrap();
        let dy2 = ctx.alloc(ybytes).unwrap();
        let launch = |prog: &Program, dst: &shadow_cuda::driver::DeviceBuffer, rows_per_block: usize| {
            // SAFETY: ng matrices (the table's addresses) of rows x cols values of this type, ng records of 17 u32, 4 input rows of cols floats (16-byte aligned, xs = cols), output rows
            // group * 8 + slot of `rows` floats inside dst.
            unsafe {
                prog.launch(&entry, ((rows as u32).div_ceil(rows_per_block as u32), ng as u32, 1), (256, 1, 1), 0, None,
                            &[Arg::Ptr(dtab.ptr), Arg::Ptr(dgt.ptr), Arg::Ptr(dx.ptr), Arg::U32(cols as u32), Arg::Ptr(dst.ptr), Arg::U32(rows as u32), Arg::U32(rows as u32),
                              Arg::U32(cols as u32)])
                    .unwrap();
            }
        };
        let time = |prog: &Program, dst: &shadow_cuda::driver::DeviceBuffer, rpb: usize| -> f32 {
            launch(prog, dst, rpb);
            let mut sum = 0f32;
            for _ in 0..20 {
                flush.borrow_mut().zero().unwrap();
                start.record(None).unwrap();
                launch(prog, dst, rpb);
                end.record(None).unwrap();
                end.synchronize().unwrap();
                sum += end.since(&start).unwrap();
            }
            sum / 20.0 * 1000.0
        };
        let us = time(&base, &dy, 8);
        let mut yb = vec![0u8; ybytes];
        dy.download(&mut yb).unwrap();
        let y = floats(&yb);
        // an f64 reference for sampled rows of the first and the busiest group
        let mut worst = 0f64;
        for &gi in &[0usize, 15] {
            for r in (0..rows).step_by((rows / 24).max(1)) {
                let mut row = vec![0f32; cols];
                dequantize(t.ty, &w[gi * per + r * (per / rows)..gi * per + (r + 1) * (per / rows)], &mut row).unwrap();
                for s in 0..COUNTS[gi] {
                    let xr = &x[((gi + s) % 4) * cols..((gi + s) % 4 + 1) * cols];
                    let (mut sum, mut mag) = (0f64, 0f64);
                    for c in 0..cols {
                        let v = row[c] as f64 * xr[c] as f64;
                        sum += v;
                        mag += v.abs();
                    }
                    worst = worst.max((y[(gi * 8 + s) * rows + r] as f64 - sum).abs() / (mag + 1e-30));
                }
            }
        }
        let mut line = format!("{ty:<8} {role:<5} {rows:>5} rows x {cols:<5} {:>6.1} MB: repo {us:>7.1} us ({:>5.0} GB/s), err {worst:.1e}", (ng * per) as f64 / 1e6, (ng * per) as f64 / (us as f64 * 1e3));
        if worst > 1e-4 {
            failures.push(format!("{ty} {role}: err {worst:e}"));
        }
        if let Some(v) = variant.as_ref() {
            let us2 = time(v, &dy2, vrows);
            let mut yb2 = vec![0u8; ybytes];
            dy2.download(&mut yb2).unwrap();
            // only the rows the groups write are compared (the rest of the buffer is whatever it was)
            let mut diff = 0usize;
            for (gi, &n) in COUNTS.iter().enumerate() {
                for s in 0..n {
                    let a = (gi * 8 + s) * rows * 4;
                    diff += yb[a..a + rows * 4].iter().zip(&yb2[a..a + rows * 4]).filter(|(p, q)| p != q).count();
                }
            }
            line += &format!("; variant {us2:>7.1} us ({:+.1}%), {} bytes differ", 100.0 * (us2 as f64 / us as f64 - 1.0), diff);
        }
        eprintln!("{line}");
    }
    assert!(failures.is_empty(), "{failures:?}");
}
