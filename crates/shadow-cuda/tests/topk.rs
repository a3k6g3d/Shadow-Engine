//! `topk_slices` + `topk_merge` (kernels/forward.cu, STATUS 159) against a CPU sort: the 64 largest logits of every row, largest first, ties to the lower id, for rows of the
//! model's vocabulary size (248,320) with many equal values, -inf and NaN.  Skips with no CUDA device.

use shadow_cuda::kernels::{file, Arg, Program};
use shadow_cuda::Context;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

#[test]
fn the_device_top_64_is_the_cpu_top_64_ties_included() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    let prog = Program::load(&ctx, file("forward.cu").unwrap()).unwrap();
    let (vocab, rows) = (248_320usize, 5usize);
    let nb = vocab.div_ceil(4096);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut logits = vec![0f32; rows * vocab];
    for r in 0..rows {
        for i in 0..vocab {
            let u = (rng(&mut seed) >> 40) as f32 / (1u64 << 24) as f32;
            logits[r * vocab + i] = match r {
                // plain: a spread like a model's logits
                0 => u * 20.0 - 10.0,
                // heavy ties: only 40 distinct values, so ranks 60..70 are decided by the ids
                1 => ((rng(&mut seed) % 40) as f32) * 0.5,
                // every value equal: the 64 lowest ids
                2 => 1.25,
                // -inf everywhere but a few, and a NaN
                3 => if i % 5000 == 7 { u } else if i == 12_345 { f32::NAN } else { f32::NEG_INFINITY },
                // the largest values in the last slice (the vocabulary is not a multiple of 4,096)
                _ => if i >= vocab - 100 { 50.0 + u } else { u },
            };
        }
    }
    let dl = ctx.alloc(rows * vocab * 4).unwrap();
    dl.upload(&logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let sv = ctx.alloc(rows * nb * 64 * 4).unwrap();
    let si = ctx.alloc(rows * nb * 64 * 4).unwrap();
    let out = ctx.alloc(rows * 128 * 4).unwrap();
    // SAFETY: logits holds rows x vocab floats; the slice buffers rows x nb x 64 each; out rows x 128 words.
    unsafe {
        prog.launch("topk_slices", ((rows * nb) as u32, 1, 1), (256, 1, 1), 0, None,
                    &[Arg::Ptr(dl.ptr), Arg::U32(vocab as u32), Arg::U32(vocab as u32), Arg::U32(nb as u32), Arg::Ptr(sv.ptr), Arg::Ptr(si.ptr)])
            .unwrap();
        prog.launch("topk_merge", (rows as u32, 1, 1), (256, 1, 1), 0, None, &[Arg::Ptr(sv.ptr), Arg::Ptr(si.ptr), Arg::U32(nb as u32), Arg::Ptr(out.ptr)]).unwrap();
    }
    ctx.synchronize().unwrap();
    let mut bytes = vec![0u8; rows * 128 * 4];
    out.download(&mut bytes).unwrap();
    let words: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    for r in 0..rows {
        // the CPU: largest first, ties to the lower id (a NaN is -inf)
        let mut v: Vec<(f32, u32)> = (0..vocab).map(|i| { let x = logits[r * vocab + i]; (if x.is_nan() { f32::NEG_INFINITY } else { x }, i as u32) }).collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for t in 0..64 {
            let got = (f32::from_bits(words[r * 128 + t]), words[r * 128 + 64 + t]);
            assert_eq!(got, v[t], "row {r} rank {t}");
        }
    }
}
