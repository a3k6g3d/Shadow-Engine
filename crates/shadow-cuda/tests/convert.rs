//! The prelude's f16/bf16 conversions on the GPU against the Rust reference (shadow_model::half): all 65,536 bit
//! patterns widened, and ~1M f32 values (every f16 and bf16 value, their midpoints, and pseudo-random bits) narrowed.
//! Exact match required; NaN only has to stay NaN.

use shadow_cuda::kernels::{file, Arg, Program};
use shadow_cuda::{nvrtc, Context};
use shadow_model::half;

fn run_u16_to_f32(ctx: &Context, prog: &Program, name: &str, input: &[u16]) -> Vec<f32> {
    let n = input.len();
    let mut din = ctx.alloc(n * 2).unwrap();
    let dout = ctx.alloc(n * 4).unwrap();
    din.upload(&input.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    // SAFETY: n u16 in, n f32 out, bounds-checked in the kernel.
    unsafe {
        prog.launch(name, ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, None,
                    &[Arg::Ptr(din.ptr), Arg::Ptr(dout.ptr), Arg::U32(n as u32)]).unwrap();
    }
    ctx.synchronize().unwrap();
    let mut b = vec![0u8; n * 4];
    dout.download(&mut b).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn run_f32_to_u16(ctx: &Context, prog: &Program, name: &str, input: &[f32]) -> Vec<u16> {
    let n = input.len();
    let mut din = ctx.alloc(n * 4).unwrap();
    let dout = ctx.alloc(n * 2).unwrap();
    din.upload(&input.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    // SAFETY: n f32 in, n u16 out, bounds-checked in the kernel.
    unsafe {
        prog.launch(name, ((n as u32).div_ceil(256), 1, 1), (256, 1, 1), 0, None,
                    &[Arg::Ptr(din.ptr), Arg::Ptr(dout.ptr), Arg::U32(n as u32)]).unwrap();
    }
    ctx.synchronize().unwrap();
    let mut b = vec![0u8; n * 2];
    dout.download(&mut b).unwrap();
    b.chunks(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn half_conversions_match_the_reference() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return eprintln!("SKIPPED, not passed: no CUDA device ({e})"),
    };
    if let Err(e) = nvrtc::version() {
        return eprintln!("SKIPPED, not passed: no NVRTC ({e})");
    }
    let prog = Program::load(&ctx, file("convert.cu").unwrap()).unwrap();

    let all: Vec<u16> = (0..=u16::MAX).collect();
    let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    for (name, reference) in [("f16_to_f32_all", half::f16_to_f32 as fn(u16) -> f32),
                              ("bf16_to_f32_all", half::bf16_to_f32 as fn(u16) -> f32)] {
        let got = run_u16_to_f32(&ctx, &prog, name, &all);
        let bad: Vec<u16> = all.iter().copied().filter(|&h| !same(got[h as usize], reference(h))).collect();
        assert!(bad.is_empty(), "{name}: {} of 65536 differ, first {:#06x}: gpu {} cpu {}", bad.len(), bad[0],
                got[bad[0] as usize], reference(bad[0]));
    }

    // Narrowing: every representable value, every midpoint (the ties), and pseudo-random bit patterns.
    let mut xs: Vec<f32> = Vec::new();
    for h in 0..=u16::MAX {
        let (a, b) = (half::f16_to_f32(h), half::bf16_to_f32(h));
        xs.extend([a, b]);
        let next = |f: f32| f32::from_bits(f.to_bits() + 1);
        xs.extend([next(a), next(b)]);
    }
    for h in 0..u16::MAX {
        let (a, b) = (half::f16_to_f32(h), half::f16_to_f32(h + 1));
        if a.is_finite() && b.is_finite() {
            xs.push((a as f64 * 0.5 + b as f64 * 0.5) as f32);
        }
    }
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..600_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        xs.push(f32::from_bits(s as u32));
    }
    for (name, reference) in [("f32_to_f16_all", half::f32_to_f16 as fn(f32) -> u16),
                              ("f32_to_bf16_all", half::f32_to_bf16 as fn(f32) -> u16)] {
        let got = run_f32_to_u16(&ctx, &prog, name, &xs);
        let widen = if name.contains("bf16") { half::bf16_to_f32 } else { half::f16_to_f32 };
        let bad: Vec<usize> = (0..xs.len())
            .filter(|&i| {
                let (g, r) = (got[i], reference(xs[i]));
                !(g == r || (widen(g).is_nan() && widen(r).is_nan()))
            })
            .collect();
        assert!(bad.is_empty(), "{name}: {} of {} differ, first {} ({:#010x}): gpu {:#06x} cpu {:#06x}", bad.len(),
                xs.len(), xs[bad[0]], xs[bad[0]].to_bits(), got[bad[0]], reference(xs[bad[0]]));
    }
    eprintln!("checked 2 x 65,536 widenings and 2 x {} narrowings", xs.len());
}
