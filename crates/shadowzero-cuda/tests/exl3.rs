//! The EXL3 GPU kernels (`kernels/exl3.cu`) against the CPU reference (`shadowzero_model::exl3`): the codebook bit for bit over all
//! 65,536 states, and the matvec in its three shapes (one input, up to 8 inputs, a table of matrices) at every bit rate on
//! random blobs and on the two real tensors of `eval/fixtures/exl3_sample.safetensors`.  The GPU sums in f32 in another order
//! than the CPU's f64, so the bound is relative to the size of the sum.  Skips with no CUDA device.

use shadowzero_cuda::kernels::{file, Arg, Program};
use shadowzero_cuda::{nvrtc, Context};
use shadowzero_model::exl3::{blob_len, codebook, matvec_blob, Exl3Linear, Rate};
use shadowzero_model::half::{f16_to_f32, f32_to_f16};
use shadowzero_model::safetensors::SafeTensors;

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

fn random_blob(rng: &mut Rng, inf: usize, outf: usize, rate: impl Into<Rate>) -> Vec<u8> {
    let rate: Rate = rate.into();
    let mut b = Vec::with_capacity(blob_len(inf, outf, rate));
    for i in 0..inf + outf {
        // sign vectors are +-1 (suh); the output scales (svh) are small positive numbers
        let v = if i < inf { if rng.next() & 1 == 0 { 1.0 } else { -1.0 } } else { 0.3 + (rng.next() % 100) as f32 / 100.0 };
        b.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    b.extend_from_slice(&1i32.to_le_bytes());
    for _ in 0..inf * outf * (2 * rate.bits + rate.half as usize) / 32 {
        b.extend_from_slice(&(rng.next() as u16).to_le_bytes());
    }
    assert_eq!(b.len(), blob_len(inf, outf, rate));
    b
}

fn check(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 2e-4 * scale, "{what}: worst difference {worst} against a largest output of {scale}");
}

fn gpu_ctx() -> Option<Context> {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => return { eprintln!("SKIPPED, not passed: no CUDA device ({e})"); None },
    };
    if let Err(e) = nvrtc::version() {
        eprintln!("SKIPPED, not passed: no NVRTC ({e})");
        return None;
    }
    Some(ctx)
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn the_gpu_codebook_is_the_cpu_codebook_bit_for_bit() {
    let Some(ctx) = gpu_ctx() else { return };
    let prog = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
    let out = ctx.alloc(65536 * 4).unwrap();
    // SAFETY: 65,536 floats at `out`, one thread each.
    unsafe { prog.launch("exl3_codebook", (256, 1, 1), (256, 1, 1), 0, None, &[Arg::Ptr(out.ptr), Arg::U32(65536)]).unwrap() };
    ctx.synchronize().unwrap();
    let mut b = vec![0u8; 65536 * 4];
    out.download(&mut b).unwrap();
    for (s, g) in floats(&b).iter().enumerate() {
        assert_eq!(g.to_bits(), codebook::cb2(s as u32).to_bits(), "state {s}: gpu {g}, cpu {}", codebook::cb2(s as u32));
    }
}

#[test]
fn the_gpu_matvec_matches_the_cpu_on_random_blobs_at_every_bit_rate() {
    let Some(ctx) = gpu_ctx() else { return };
    let prog = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    // every integer rate and the half-integer rates K + 0.5 (tag "2h" = 2.5 bits)
    for (bits, half) in (1..=8usize).map(|b| (b, false)).chain((1..=3usize).map(|b| (b, true))) {
        let tag = if half { format!("{bits}h") } else { bits.to_string() };
        let rate = (bits, half);
        for (inf, outf) in [(256usize, 128usize), (640, 256), (384, 384)] {
            let blob = random_blob(&mut rng, inf, outf, rate);
            let w = ctx.alloc(blob.len()).unwrap();
            w.upload(&blob).unwrap();
            let m = 5usize;
            let x: Vec<f32> = (0..m * inf).map(|_| rng.unit()).collect();
            let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let dx = ctx.alloc(xb.len()).unwrap();
            dx.upload(&xb).unwrap();
            let dy = ctx.alloc(m * outf * 4).unwrap();
            let want: Vec<Vec<f32>> = (0..m).map(|j| matvec_blob(inf, outf, rate, 2, &blob, &x[j * inf..(j + 1) * inf], 2)).collect();
            let grid = (outf / 128) as u32;
            let read = |dy: &shadowzero_cuda::DeviceBuffer, n: usize| -> Vec<f32> {
                let mut o = vec![0u8; n * 4];
                dy.download(&mut o).unwrap();
                floats(&o)
            };
            // one input
            // SAFETY: a blob of this projection at w, `inf` floats at dx, `outf` at dy.
            unsafe { prog.launch(&format!("matvec_exl3_{tag}"), (grid, 1, 1), (256, 1, 1), 0, None, &[Arg::Ptr(w.ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
            ctx.synchronize().unwrap();
            check(&read(&dy, outf), &want[0], &format!("matvec_exl3_{tag} {inf}x{outf}"));
            // the k-sliced kernels (1, 2 and 4 slices of 8 warps; dynamic shared memory for the rotated input)
            for ks in [1u32, 2, 4] {
                dy.upload(&vec![0u8; outf * 4]).unwrap();
                // SAFETY: as above; 256 ks threads, 4 inf bytes of shared memory.
                unsafe { prog.launch(&format!("matvec_exl3_s{ks}_{tag}"), (grid, 1, 1), (256 * ks, 1, 1), (inf * 4) as u32, None, &[Arg::Ptr(w.ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
                ctx.synchronize().unwrap();
                check(&read(&dy, outf), &want[0], &format!("matvec_exl3_s{ks}_{tag} {inf}x{outf}"));
            }
            // m inputs at once, strides wider than the rows
            let (xs, ys) = (inf + 16, outf + 32);
            let xpad: Vec<u8> = (0..m).flat_map(|j| x[j * inf..(j + 1) * inf].iter().copied().chain(std::iter::repeat(0.0).take(16))).flat_map(|v: f32| v.to_le_bytes()).collect();
            let dxp = ctx.alloc(xpad.len()).unwrap();
            dxp.upload(&xpad).unwrap();
            let dyp = ctx.alloc(m * ys * 4).unwrap();
            // SAFETY: m rows of inf floats xs apart at dxp, m rows of outf floats ys apart at dyp.
            unsafe { prog.launch(&format!("matvec_cols_exl3_{tag}"), (grid, 1, 1), (256, 1, 1), 0, None,
                                 &[Arg::Ptr(w.ptr), Arg::Ptr(dxp.ptr), Arg::U32(xs as u32), Arg::Ptr(dyp.ptr), Arg::U32(ys as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32)]).unwrap() };
            ctx.synchronize().unwrap();
            let got = read(&dyp, m * ys);
            for j in 0..m {
                check(&got[j * ys..j * ys + outf], &want[j], &format!("matvec_cols_exl3_{tag} {inf}x{outf} input {j}"));
            }
            for ks in [2u32, 4] {
                dyp.upload(&vec![0u8; m * ys * 4]).unwrap();
                // SAFETY: as the cols launch above, with the k-sliced kernel.
                unsafe { prog.launch(&format!("matvec_cols_exl3_s{ks}_{tag}"), (grid, 1, 1), (256 * ks, 1, 1), 0, None,
                                     &[Arg::Ptr(w.ptr), Arg::Ptr(dxp.ptr), Arg::U32(xs as u32), Arg::Ptr(dyp.ptr), Arg::U32(ys as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32)]).unwrap() };
                ctx.synchronize().unwrap();
                let got = read(&dyp, m * ys);
                for j in 0..m {
                    check(&got[j * ys..j * ys + outf], &want[j], &format!("matvec_cols_exl3_s{ks}_{tag} {inf}x{outf} input {j}"));
                }
            }
            // the tensor-core prompt kernel: 5 and 70 tokens (a second, partial token block); f16 activations, so a looser bound than the f32 kernels
            for mt in [5usize, 70] {
                let xm: Vec<f32> = (0..mt * inf).map(|_| rng.unit()).collect();
                let dxm = ctx.alloc(mt * inf * 4).unwrap();
                dxm.upload(&xm.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                let dym = ctx.alloc(mt * outf * 4).unwrap();
                dym.upload(&vec![0u8; mt * outf * 4]).unwrap();
                // SAFETY: mt inputs of inf floats at dxm, mt outputs of outf at dym.
                unsafe { prog.launch(&format!("mma_cols_exl3_{tag}"), (grid, mt.div_ceil(64) as u32, 1), (256, 1, 1), 0, None,
                                     &[Arg::Ptr(w.ptr), Arg::Ptr(dxm.ptr), Arg::U32(inf as u32), Arg::Ptr(dym.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(mt as u32)]).unwrap() };
                ctx.synchronize().unwrap();
                let got = read(&dym, mt * outf);
                // the same through the rotation pre-pass: bit for bit what the fused kernel gives
                {
                    let dxr = ctx.alloc(mt * inf * 2).unwrap();
                    let dyp = ctx.alloc(mt * outf * 4).unwrap();
                    dyp.upload(&vec![0u8; mt * outf * 4]).unwrap();
                    // SAFETY: mt inputs of inf floats at dxm, mt rotated rows of inf f16 at dxr, mt outputs of outf at dyp.
                    unsafe {
                        prog.launch("exl3_rotate_rows", (((mt * (inf / 128)) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                                    &[Arg::Ptr(w.ptr), Arg::Ptr(dxm.ptr), Arg::U32(inf as u32), Arg::Ptr(dxr.ptr), Arg::U32(inf as u32), Arg::U32(mt as u32)]).unwrap();
                        prog.launch(&format!("mma_cols_exl3_pre_{tag}"), (grid, mt.div_ceil(64) as u32, 1), (256, 1, 1), 0, None,
                                    &[Arg::Ptr(w.ptr), Arg::Ptr(dxr.ptr), Arg::U32(0), Arg::Ptr(dyp.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(mt as u32)]).unwrap();
                    }
                    ctx.synchronize().unwrap();
                    assert_eq!(read(&dyp, mt * outf), got, "mma_cols_exl3_pre_{tag} {inf}x{outf} m={mt}: not what the fused kernel gives");
                }
                for j in 0..mt {
                    let wantm = matvec_blob(inf, outf, rate, 2, &blob, &xm[j * inf..(j + 1) * inf], 2);
                    let scale = wantm.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
                    let worst = got[j * outf..(j + 1) * outf].iter().zip(&wantm).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
                    assert!(worst <= 3e-3 * scale, "mma_cols_exl3_{tag} {inf}x{outf} m={mt} input {j}: worst difference {worst} against a largest output of {scale}");
                }
            }
            // a routed expert's group: n inputs picked from rows of x (stride xs), outputs scattered to rows of y (stride ys), two groups on two matrices
            {
                let blob2 = random_blob(&mut rng, inf, outf, rate);
                let w2 = ctx.alloc(blob2.len()).unwrap();
                w2.upload(&blob2).unwrap();
                let tab: Vec<u8> = [w.ptr, w2.ptr].iter().flat_map(|p| p.to_le_bytes()).collect();
                let dt = ctx.alloc(tab.len()).unwrap();
                dt.upload(&tab).unwrap();
                // records: n, 8 input rows, 8 output rows (17 u32)
                let groups: [(usize, [u32; 8], [u32; 8]); 2] = [(5, [4, 0, 3, 1, 2, 0, 0, 0], [7, 1, 0, 9, 3, 0, 0, 0]), (2, [1, 4, 0, 0, 0, 0, 0, 0], [2, 5, 0, 0, 0, 0, 0, 0])];
                let rec: Vec<u8> = groups.iter().flat_map(|(n, xi, yi)| std::iter::once(*n as u32).chain(xi.iter().copied()).chain(yi.iter().copied())).flat_map(|v| v.to_le_bytes()).collect();
                let dg = ctx.alloc(rec.len()).unwrap();
                dg.upload(&rec).unwrap();
                let (xs, ys) = (inf + 16, outf + 32);
                let xpad: Vec<u8> = (0..m).flat_map(|j| x[j * inf..(j + 1) * inf].iter().copied().chain(std::iter::repeat(0.0).take(16))).flat_map(|v: f32| v.to_le_bytes()).collect();
                let dxp = ctx.alloc(xpad.len()).unwrap();
                dxp.upload(&xpad).unwrap();
                let dyg = ctx.alloc(10 * ys * 4).unwrap();
                dyg.upload(&vec![0u8; 10 * ys * 4]).unwrap();
                // SAFETY: two table entries, two 17-u32 records, input rows < 5 of dxp, output rows < 10 of dyg.
                unsafe { prog.launch(&format!("matvec_grouped_exl3_s4_{tag}"), (grid, 2, 1), (1024, 1, 1), 0, None,
                                     &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dxp.ptr), Arg::U32(xs as u32), Arg::Ptr(dyg.ptr), Arg::U32(ys as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
                ctx.synchronize().unwrap();
                let got = read(&dyg, 10 * ys);
                for (gi, (n, xi, yi)) in groups.iter().enumerate() {
                    let b = if gi == 0 { &blob } else { &blob2 };
                    for u in 0..*n {
                        let wantg = matvec_blob(inf, outf, rate, 2, b, &x[xi[u] as usize * inf..(xi[u] as usize + 1) * inf], 2);
                        let r = yi[u] as usize;
                        check(&got[r * ys..r * ys + outf], &wantg, &format!("matvec_grouped_exl3_s4_{tag} {inf}x{outf} group {gi} input {u}"));
                    }
                }
            }
            // split-K: the input split over `ns` blocks (partial sums in a scratch), then the combine
            for ns in [2u32, 3] {
                if ns as usize > (inf / 128).max(1) {
                    continue;
                }
                let dpart = ctx.alloc(ns as usize * 8 * outf * 4).unwrap();
                let dys = ctx.alloc(m * outf * 4).unwrap();
                dys.upload(&vec![0u8; m * outf * 4]).unwrap();
                // SAFETY: m inputs of inf floats at dx, partial sums for ns slices of 8 inputs of outf floats at dpart, m outputs of outf at dys.
                unsafe {
                    prog.launch(&format!("matvec_cols_exl3_sk4_{tag}"), (grid, ns, 1), (1024, 1, 1), 0, None,
                                &[Arg::Ptr(w.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dpart.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32)]).unwrap();
                    prog.launch("exl3_splitk_combine", (grid, 1, 1), (256, 1, 1), 0, None,
                                &[Arg::Ptr(w.ptr), Arg::Ptr(dpart.ptr), Arg::Ptr(dys.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32), Arg::U32(m as u32), Arg::U32(ns)]).unwrap();
                }
                ctx.synchronize().unwrap();
                let got = read(&dys, m * outf);
                for j in 0..m {
                    check(&got[j * outf..(j + 1) * outf], &want[j], &format!("split-K x{ns} {tag} {inf}x{outf} input {j}"));
                }
            }
            // the group kernels made for groups of at most 2 and 4 tokens (`m2s4`, `m4s4`), on groups of that size
            for (kname, sizes) in [("m2s4", [2usize, 1]), ("m4s4", [4, 3])] {
                let blob2 = random_blob(&mut rng, inf, outf, rate);
                let w2 = ctx.alloc(blob2.len()).unwrap();
                w2.upload(&blob2).unwrap();
                let tab: Vec<u8> = [w.ptr, w2.ptr].iter().flat_map(|p| p.to_le_bytes()).collect();
                let dt = ctx.alloc(tab.len()).unwrap();
                dt.upload(&tab).unwrap();
                let mut rec = vec![0u32; 2 * 17];
                for (g, &n) in sizes.iter().enumerate() {
                    rec[g * 17] = n as u32;
                    for u in 0..n {
                        rec[g * 17 + 1 + u] = ((g * 3 + u) % 5) as u32; // input rows of the 5 of x
                        rec[g * 17 + 9 + u] = (g * 4 + u) as u32; // distinct output rows
                    }
                }
                let dg = ctx.alloc(rec.len() * 4).unwrap();
                dg.upload(&rec.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                let dyg = ctx.alloc(8 * outf * 4).unwrap();
                dyg.upload(&vec![0u8; 8 * outf * 4]).unwrap();
                // SAFETY: two table entries and 17-u32 records, input rows < 5 of dx (inf floats a row), output rows < 8 of dyg.
                unsafe { prog.launch(&format!("matvec_grouped_exl3_{kname}_{tag}"), (grid, 2, 1), (1024, 1, 1), 0, None,
                                     &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dx.ptr), Arg::U32(inf as u32), Arg::Ptr(dyg.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
                ctx.synchronize().unwrap();
                let got = read(&dyg, 8 * outf);
                for (g, &n) in sizes.iter().enumerate() {
                    let b = if g == 0 { &blob } else { &blob2 };
                    for u in 0..n {
                        let (xi, yi) = (rec[g * 17 + 1 + u] as usize, rec[g * 17 + 9 + u] as usize);
                        let wantg = matvec_blob(inf, outf, rate, 2, b, &x[xi * inf..(xi + 1) * inf], 2);
                        check(&got[yi * outf..(yi + 1) * outf], &wantg, &format!("matvec_grouped_exl3_{kname}_{tag} {inf}x{outf} group {g} input {u}"));
                    }
                }
            }
            // the tensor-core group kernel: groups of up to 64 tokens (records of 129 u32: n, 64 input rows, 64 output rows)
            {
                let blob2 = random_blob(&mut rng, inf, outf, rate);
                let w2 = ctx.alloc(blob2.len()).unwrap();
                w2.upload(&blob2).unwrap();
                let tab: Vec<u8> = [w.ptr, w2.ptr].iter().flat_map(|p| p.to_le_bytes()).collect();
                let dt = ctx.alloc(tab.len()).unwrap();
                dt.upload(&tab).unwrap();
                let (n0, n1) = (37usize, 3usize);
                let mut recs = vec![0u32; 2 * 129];
                recs[0] = n0 as u32;
                for u in 0..n0 {
                    recs[1 + u] = (u * 7 % 20) as u32; // inputs: rows of a 20-row buffer, with repeats
                    recs[65 + u] = (u * 3 + 1) as u32; // outputs: distinct rows of a 120-row buffer
                }
                recs[129] = n1 as u32;
                for u in 0..n1 {
                    recs[129 + 1 + u] = (19 - u) as u32;
                    recs[129 + 65 + u] = (112 + u) as u32;
                }
                let dg = ctx.alloc(recs.len() * 4).unwrap();
                dg.upload(&recs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                let xm: Vec<f32> = (0..20 * inf).map(|_| rng.unit()).collect();
                let dxm = ctx.alloc(xm.len() * 4).unwrap();
                dxm.upload(&xm.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                let dyg = ctx.alloc(120 * outf * 4).unwrap();
                dyg.upload(&vec![0u8; 120 * outf * 4]).unwrap();
                // SAFETY: two table entries, two 129-u32 records, input rows < 20 of dxm, output rows < 120 of dyg.
                unsafe { prog.launch(&format!("mma_grouped_exl3_{tag}"), (grid, 2, 1), (256, 1, 1), 0, None,
                                     &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dxm.ptr), Arg::U32(inf as u32), Arg::Ptr(dyg.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
                ctx.synchronize().unwrap();
                let got = read(&dyg, 120 * outf);
                // the same through the groups' rotation pre-pass: bit for bit the fused kernel's
                {
                    let dxr = ctx.alloc(2 * 64 * inf * 2).unwrap();
                    let dyq = ctx.alloc(120 * outf * 4).unwrap();
                    dyq.upload(&vec![0u8; 120 * outf * 4]).unwrap();
                    // SAFETY: two table entries and records, rows < 20 of dxm, two groups of 64 rotated rows of inf f16 at dxr, output rows < 120 of dyq.
                    unsafe {
                        prog.launch("exl3_rotate_groups", ((64 * (inf as u32 / 128)).div_ceil(8), 2, 1), (256, 1, 1), 0, None,
                                    &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dxm.ptr), Arg::U32(inf as u32), Arg::Ptr(dxr.ptr), Arg::U32(inf as u32)]).unwrap();
                        prog.launch(&format!("mma_grouped_exl3_pre_{tag}"), (grid, 2, 1), (256, 1, 1), 0, None,
                                    &[Arg::Ptr(dt.ptr), Arg::Ptr(dg.ptr), Arg::Ptr(dxr.ptr), Arg::U32(0), Arg::Ptr(dyq.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap();
                    }
                    ctx.synchronize().unwrap();
                    assert_eq!(read(&dyq, 120 * outf), got, "mma_grouped_exl3_pre_{tag} {inf}x{outf}: not what the fused group kernel gives");
                }
                for gi in 0..2usize {
                    let b = if gi == 0 { &blob } else { &blob2 };
                    for u in 0..recs[gi * 129] as usize {
                        let (xi, yi) = (recs[gi * 129 + 1 + u] as usize, recs[gi * 129 + 65 + u] as usize);
                        let wantg = matvec_blob(inf, outf, rate, 2, b, &xm[xi * inf..(xi + 1) * inf], 2);
                        let scale = wantg.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
                        let worst = got[yi * outf..(yi + 1) * outf].iter().zip(&wantg).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
                        assert!(worst <= 3e-3 * scale, "mma_grouped_exl3_{tag} {inf}x{outf} group {gi} input {u}: worst difference {worst} against a largest output of {scale}");
                    }
                }
            }
            // a table of two different matrices of this shape, one shared input and one input each
            let blob2 = random_blob(&mut rng, inf, outf, rate);
            let w2 = ctx.alloc(blob2.len()).unwrap();
            w2.upload(&blob2).unwrap();
            let tab: Vec<u8> = [w.ptr, w2.ptr].iter().flat_map(|p| p.to_le_bytes()).collect();
            let dt = ctx.alloc(tab.len()).unwrap();
            dt.upload(&tab).unwrap();
            let dyb = ctx.alloc(2 * outf * 4).unwrap();
            let want2: Vec<Vec<f32>> = [&blob, &blob2].iter().map(|b| matvec_blob(inf, outf, rate, 2, b, &x[..inf], 2)).collect();
            for ks in [1u32, 2, 4] {
                for (xstride, name) in [(0usize, "shared input"), (inf, "an input each")] {
                    // SAFETY: as below, with the k-sliced kernel.
                    unsafe { prog.launch(&format!("matvec_batch_exl3_s{ks}_{tag}"), (grid, 2, 1), (256 * ks, 1, 1), (inf * 4) as u32, None,
                                         &[Arg::Ptr(dt.ptr), Arg::Ptr(dx.ptr), Arg::U32(xstride as u32), Arg::Ptr(dyb.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
                    ctx.synchronize().unwrap();
                    let got = read(&dyb, 2 * outf);
                    for b in 0..2 {
                        let wantb = if xstride == 0 { want2[b].clone() } else { matvec_blob(inf, outf, rate, 2, if b == 0 { &blob } else { &blob2 }, &x[b * inf..(b + 1) * inf], 2) };
                        check(&got[b * outf..(b + 1) * outf], &wantb, &format!("matvec_batch_exl3_s{ks}_{tag} {inf}x{outf} {name} matrix {b}"));
                    }
                }
            }
            for (xstride, name) in [(0usize, "shared input"), (inf, "an input each")] {
                // SAFETY: a table of two blobs, the inputs at dx (stride xstride), the outputs outf apart at dyb.
                unsafe { prog.launch(&format!("matvec_batch_exl3_{tag}"), (grid, 2, 1), (256, 1, 1), 0, None,
                                     &[Arg::Ptr(dt.ptr), Arg::Ptr(dx.ptr), Arg::U32(xstride as u32), Arg::Ptr(dyb.ptr), Arg::U32(outf as u32), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
                ctx.synchronize().unwrap();
                let got = read(&dyb, 2 * outf);
                for b in 0..2 {
                    let wantb = if xstride == 0 { want2[b].clone() } else { matvec_blob(inf, outf, rate, 2, if b == 0 { &blob } else { &blob2 }, &x[b * inf..(b + 1) * inf], 2) };
                    check(&got[b * outf..(b + 1) * outf], &wantb, &format!("matvec_batch_exl3_{tag} {inf}x{outf} {name} matrix {b}"));
                }
            }
        }
    }
}

#[test]
fn the_gpu_matvec_matches_the_cpu_on_real_tensors() {
    let Some(ctx) = gpu_ctx() else { return };
    let prog = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
    let st = SafeTensors::open(&[concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/fixtures/exl3_sample.safetensors").into()]).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for prefix in ["model.language_model.layers.0.mlp.experts.0.gate_proj", "model.language_model.layers.0.mlp.shared_expert.gate_proj"] {
        let lin = Exl3Linear::load(&st, prefix).unwrap();
        let blob = lin.to_blob();
        let (inf, outf) = (lin.in_features, lin.out_features);
        let x: Vec<f32> = (0..inf).map(|_| rng.unit()).collect();
        let want = matvec_blob(inf, outf, lin.bits, 2, &blob, &x, 4);
        // and against the reconstructed matrix, which does not use the kernels' route at all
        let w = lin.reconstruct();
        let direct: Vec<f32> = (0..outf).map(|o| (0..inf).map(|i| x[i] as f64 * w[i * outf + o] as f64).sum::<f64>() as f32).collect();
        check(&want, &direct, &format!("{prefix}: cpu blob matvec against the matrix"));
        let dw = ctx.alloc(blob.len()).unwrap();
        dw.upload(&blob).unwrap();
        let dx = ctx.alloc(inf * 4).unwrap();
        dx.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let dy = ctx.alloc(outf * 4).unwrap();
        // SAFETY: the blob of this projection at dw, inf floats at dx, outf at dy.
        unsafe { prog.launch(&format!("matvec_exl3_{}", lin.bits), ((outf / 128) as u32, 1, 1), (256, 1, 1), 0, None, &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
        ctx.synchronize().unwrap();
        let mut o = vec![0u8; outf * 4];
        dy.download(&mut o).unwrap();
        check(&floats(&o), &want, &format!("{prefix} K={}", lin.bits));
        for ks in [1u32, 2, 4] {
            // SAFETY: as above, with the k-sliced kernel.
            unsafe { prog.launch(&format!("matvec_exl3_s{ks}_{}", lin.bits), ((outf / 128) as u32, 1, 1), (256 * ks, 1, 1), (inf * 4) as u32, None, &[Arg::Ptr(dw.ptr), Arg::Ptr(dx.ptr), Arg::Ptr(dy.ptr), Arg::U32(outf as u32), Arg::U32(inf as u32)]).unwrap() };
            ctx.synchronize().unwrap();
            dy.download(&mut o).unwrap();
            check(&floats(&o), &want, &format!("{prefix} K={} k-sliced x{ks}", lin.bits));
        }
        let _ = f16_to_f32;
    }
}
