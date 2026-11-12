//! The int8 activation paths of the dense projections against the float `matvec_cols_*` kernels, on real tensors of the model (docs/STATUS.md 109).
//!
//! For one real 2-D tensor of each quantization (the first rows of it), m = 1..=8 and 20 pseudo-random inputs, y_j = W x_j is computed on the CPU in f64 (the reference), by the
//! float `matvec_cols_*` kernel (Shadow's own), and by the native library's int8 paths (`SHADOW_MMQ_DLL=native/mmq/shadow_mmq.dll`): llama.cpp's MMQ (`gemm`) and, when the DLL
//! has it, its `mul_mat_vec_q` (`mmvq`).  Prints per path the time of one call (events, 20 calls), the worst |error| / sum |w x| and any NaN / Inf; the int8 paths are held to
//! `INT8_BOUND` of the sum's magnitude (activations quantized to int8 in blocks of 32 cost ~0.4%), the float kernel to 1e-4.  Needs the model and a card; skips without them.
//! Run under tools/run_capped.ps1 (it reads tensors from the model file).

use shadow_cuda::kernels::{file, Arg, Program};
use shadow_cuda::Context;
use shadow_model::gguf::{Gguf, TensorInfo};
use shadow_model::quant::dequantize;
use std::io::{Read, Seek, SeekFrom};

const MODEL: &str = "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf";
const INT8_BOUND: f64 = 2e-2;
fn rows_max() -> usize {
    std::env::var("INT8_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(2048)
}

fn read(g: &Gguf, t: &TensorInfo, bytes: usize) -> Vec<u8> {
    let mut f = std::fs::File::open(&g.shards[t.shard].path).unwrap();
    let mut b = vec![0u8; bytes];
    f.seek(SeekFrom::Start(t.offset)).unwrap();
    f.read_exact(&mut b).unwrap();
    b
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn int8_paths_match_the_float_kernels_on_real_tensors() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    let p = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| MODEL.into());
    if !std::path::Path::new(&p).is_file() {
        return eprintln!("SKIPPED, not passed: no model at {p}");
    }
    let dll = std::env::var("SHADOW_MMQ_DLL").unwrap_or_default();
    let mq = if dll.is_empty() { None } else { shadow_cuda::mmq::Mmq::load() };
    if mq.is_none() {
        eprintln!("no native DLL (set SHADOW_MMQ_DLL): only the float kernel is checked");
    }
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let prog = Program::load(&ctx, file("matvec.cu").unwrap()).unwrap();
    let which = std::env::var("INT8_PATHS").unwrap_or_else(|_| "mmq,mmvq".into());
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let (start, end) = (ctx.event().unwrap(), ctx.event().unwrap());
    // a buffer bigger than the card's L2 (48 MB), written between calls so each timed call reads its weights from DRAM as the engine's do
    let flush = std::cell::RefCell::new(ctx.alloc(160 << 20).unwrap());
    let mut failures = vec![];
    for ty in ["Q6_K", "IQ4_XS", "Q4_K", "Q5_K", "Q8_0"] {
        let Some(t) = g.tensors.iter().filter(|t| t.ty.name() == ty && t.dims.len() == 2 && t.dims[1] > 64 && t.dims[0] % 256 == 0 && !t.name.contains("token_embd") && !t.name.contains("output."))
            .min_by_key(|t| t.name.clone())
        else {
            eprintln!("{ty}: no 2-D tensor in this model");
            continue;
        };
        let (cols, all_rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let rows = all_rows.min(rows_max());
        let rb = t.nbytes() as usize / all_rows;
        let w = read(&g, t, rb * rows);
        let dw = ctx.alloc(w.len()).unwrap();
        dw.upload(&w).unwrap();
        let mut wf = vec![0f32; rows * cols];
        for r in 0..rows {
            dequantize(t.ty, &w[r * rb..(r + 1) * rb], &mut wf[r * cols..(r + 1) * cols]).unwrap();
        }
        eprintln!("== {ty} {} ({rows} of {all_rows} rows x {cols})", t.name);
        for m in [1usize, 2, 3, 4, 5, 8, 20] {
            let x: Vec<f32> = (0..m * cols).map(|_| rnd()).collect();
            let (mut cpu, mut mag) = (vec![0f64; m * rows], vec![0f64; m * rows]);
            for j in 0..m {
                for r in 0..rows {
                    let (mut s, mut a) = (0f64, 0f64);
                    for c in 0..cols {
                        let v = wf[r * cols + c] as f64 * x[j * cols + c] as f64;
                        s += v;
                        a += v.abs();
                    }
                    cpu[j * rows + r] = s;
                    mag[j * rows + r] = a;
                }
            }
            let dx = ctx.alloc(m * cols * 4).unwrap();
            dx.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            let dy = ctx.alloc(m * rows * 4 + 4096).unwrap();
            let check = |name: &str, bound: f64, ms: f32, failures: &mut Vec<String>| {
                let mut yb = vec![0u8; m * rows * 4];
                dy.download(&mut yb).unwrap();
                let y = floats(&yb);
                let (mut worst, mut bad) = (0f64, 0usize);
                for i in 0..m * rows {
                    if !y[i].is_finite() {
                        bad += 1;
                        continue;
                    }
                    worst = worst.max((y[i] as f64 - cpu[i]).abs() / (mag[i] + 1e-30));
                }
                let verdict = if bad > 0 || worst > bound { "FAIL" } else { "ok" };
                eprintln!("  m={m:<2} {name:<14} {ms:>7.1} us  worst err/|sum| {worst:.1e}  non-finite {bad}  {verdict}");
                if bad > 0 || worst > bound {
                    failures.push(format!("{ty} m={m} {name}: worst {worst:e}, non-finite {bad}"));
                }
            };
            // the mean of 20 calls, each after the L2 was flushed (cold weights), in microseconds
            let time = |f: &dyn Fn()| -> f32 {
                f();
                let mut sum = 0f32;
                for _ in 0..20 {
                    flush.borrow_mut().zero().unwrap();
                    start.record(None).unwrap();
                    f();
                    end.record(None).unwrap();
                    end.synchronize().unwrap();
                    sum += end.since(&start).unwrap();
                }
                sum / 20.0 * 1000.0
            };
            // the float kernel (m <= 8 at a time, as the engine launches it)
            let entry = format!("matvec_cols_{}", ty.to_lowercase());
            let float = || {
                for j0 in (0..m).step_by(8) {
                    let n = (m - j0).min(8);
                    // SAFETY: rows x cols weights, n inputs of cols floats at x + j * cols, n outputs of rows floats at y + j * rows; 8 rows a block.
                    unsafe {
                        prog.launch(&entry, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                                    &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr + (j0 * cols * 4) as u64), Arg::U32(cols as u32), Arg::Ptr(dy.ptr + (j0 * rows * 4) as u64),
                                      Arg::U32(rows as u32), Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(n as u32)])
                            .unwrap();
                    }
                }
            };
            let ms = time(&float);
            check("float", 1e-4, ms, &mut failures);
            let Some(mq) = mq.as_ref() else { continue };
            if which.contains("mmq") {
                if let Some(gty) = shadow_cuda::mmq::Mmq::ggml_type(ty) {
                    let yq = ctx.alloc(mq.y_bytes(cols, m) + 4096).unwrap();
                    let run = || {
                        // SAFETY: x holds m rows of cols floats; yq has y_bytes; the weights are rows rows of cols values; dy holds m rows of `rows` floats.
                        unsafe {
                            mq.quantize(dx.ptr, yq.ptr, cols, m, cols, gty, 0).unwrap();
                            mq.gemm(gty, dw.ptr, yq.ptr, dy.ptr, rows, cols, m, rows, 0).unwrap();
                        }
                    };
                    dy.upload(&vec![0u8; m * rows * 4]).unwrap();
                    let ms = time(&run);
                    check("mmq+quant", INT8_BOUND, ms, &mut failures);
                }
            }
            if which.contains("mmvq") && mq.has_mmvq() && m <= 8 {
                if let Some(gty) = shadow_cuda::mmq::Mmq::mmvq_type(ty) {
                    let yq = ctx.alloc(mq.mmvq_y_bytes(cols, m) + 4096).unwrap();
                    let run = || {
                        // SAFETY: as above; yq has mmvq_y_bytes (plain q8_1 blocks).
                        unsafe {
                            mq.mmvq_quantize(dx.ptr, yq.ptr, cols, m, cols, 0).unwrap();
                            mq.mmvq(gty, dw.ptr, yq.ptr, dy.ptr, rows, cols, m, rows, 0).unwrap();
                        }
                    };
                    dy.upload(&vec![0u8; m * rows * 4]).unwrap();
                    let ms = time(&run);
                    check("mmvq+quant", INT8_BOUND, ms, &mut failures);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures, first: {:?}", failures.len(), &failures[..failures.len().min(5)]);
}
