//! Unit tests for the packed 16-byte-aligned symmetric Q8_KV cache kernels in `kernels/forward.cu`.

use shadowzero_cuda::kernels::{file, Arg, Program};
use shadowzero_cuda::{nvrtc, Context};
use shadowzero_model::half::f16_to_f32;

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

fn gpu_ctx() -> Option<Context> {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            return None;
        }
    };
    if let Err(e) = nvrtc::version() {
        eprintln!("SKIPPED, not passed: no NVRTC ({e})");
        return None;
    }
    Some(ctx)
}

fn to_bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn to_bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn kv_q8_head_bytes(hd: usize) -> usize {
    hd + (((hd >> 5) * 2 + 15) & !15)
}

fn dequant_q8_rows(raw: &[u8], pos0: usize, rows: usize, nkv: usize, hd: usize) -> Vec<f32> {
    let hs = kv_q8_head_bytes(hd);
    let rs = nkv * hs;
    let mut out = vec![0f32; rows * nkv * hd];
    for j in 0..rows {
        for kh in 0..nkv {
            let base = (pos0 + j) * rs + kh * hs;
            let ng = hd >> 5;
            // Verify zeroed padding bytes
            for pad_b in &raw[base + hd + ng * 2..base + hs] {
                assert_eq!(*pad_b, 0, "padding byte must be zeroed for deterministic snapshots");
            }
            for grp in 0..ng {
                let sc_bits = u16::from_le_bytes([raw[base + hd + grp * 2], raw[base + hd + grp * 2 + 1]]);
                let sc = f16_to_f32(sc_bits);
                for i in 0..32 {
                    let q = raw[base + grp * 32 + i] as i8;
                    out[(j * nkv + kh) * hd + grp * 32 + i] = (q as f32) * sc;
                }
            }
        }
    }
    out
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
    }
    (num / den.max(1e-24)).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-24)
}

fn check_attn_out(got: &[f32], want: &[f32], name: &str) {
    assert_eq!(got.len(), want.len());
    for (i, &v) in got.iter().enumerate() {
        assert!(v.is_finite(), "{name}: non-finite output at {i}: {v}");
    }
    let max_abs = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    let cos = cosine(got, want);
    assert!(max_abs < 2e-2, "{name}: max_abs {max_abs:.5} >= 2e-2 (cos={cos:.6})");
    assert!(cos > 0.9995, "{name}: cosine {cos:.6} <= 0.9995 (max_abs={max_abs:.5})");
}

#[test]
fn kv_quant_q8_roundtrip_and_edge_cases() {
    let Some(ctx) = gpu_ctx() else { return };
    let prog = Program::load(&ctx, file("forward.cu").unwrap()).expect("compile forward.cu");
    let mut rng = Rng(0x5a7e_1234_99ab_cdef);

    for &hd in &[64usize, 128, 256] {
        let nkv = 2usize;
        let rows = 5usize;
        let pos0 = 3usize;
        let cap = pos0 + rows + 2;
        let rs = nkv * kv_q8_head_bytes(hd);
        let mut src = vec![0f32; rows * nkv * hd];
        for j in 0..rows {
            for kh in 0..nkv {
                for i in 0..hd {
                    let v = match j {
                        0 => 0.0,                        // exact zeros
                        1 => rng.unit() * 1e-9,          // tiny subnormals -> underflow fp16 scale cleanly
                        2 => rng.unit() * 12.0,          // large values
                        _ => rng.unit() * 2.5,           // normal activations
                    };
                    src[(j * nkv + kh) * hd + i] = v;
                }
            }
        }
        let d_src = ctx.alloc(src.len() * 4).unwrap();
        d_src.upload(&to_bytes_f32(&src)).unwrap();
        let d_dst = ctx.alloc(cap * rs).unwrap();
        // Fill with 0xAA to prove padding gets explicitly zeroed
        d_dst.upload(&vec![0xAAu8; cap * rs]).unwrap();

        unsafe {
            prog.launch(
                "kv_quant_q8_rows",
                (nkv as u32, rows as u32, 1),
                (128, 1, 1),
                0,
                None,
                &[
                    Arg::Ptr(d_src.ptr),
                    Arg::Ptr(d_dst.ptr),
                    Arg::U32(pos0 as u32),
                    Arg::U32(nkv as u32),
                    Arg::U32(hd as u32),
                    Arg::U32(rows as u32),
                ],
            )
            .unwrap();
        }
        ctx.synchronize().unwrap();
        let mut raw = vec![0u8; cap * rs];
        d_dst.download(&mut raw).unwrap();
        let deq = dequant_q8_rows(&raw, pos0, rows, nkv, hd);
        for (i, &v) in deq.iter().enumerate() {
            assert!(v.is_finite(), "hd={hd}: non-finite at {i}: {v}");
        }
        // Row 0 (zeros) must be exact 0
        assert!(deq[..nkv * hd].iter().all(|&x| x == 0.0));
        // Rows 2..rows normal & large must have rel_l2 < 0.015
        let rl2 = rel_l2(&deq[2 * nkv * hd..], &src[2 * nkv * hd..]);
        assert!(rl2 < 0.015, "hd={hd}: rel_l2 {rl2:.5} >= 0.015");
    }
}

#[test]
fn head_norm_rope_qgk_rows_q8_matches_f32() {
    let Some(ctx) = gpu_ctx() else { return };
    let prog = Program::load(&ctx, file("forward.cu").unwrap()).expect("compile forward.cu");
    let mut rng = Rng(0xfeed_face_0123_4567);

    let (nh, nkv, hd, rows, pos0) = (24usize, 2usize, 256usize, 6usize, 17usize);
    let cap = pos0 + rows + 4;
    let qd = nh * hd;
    let kd = nkv * hd;
    let rs = nkv * kv_q8_head_bytes(hd);

    let qg: Vec<f32> = (0..rows * 2 * qd).map(|_| rng.unit() * 2.0).collect();
    let kk: Vec<f32> = (0..rows * kd).map(|_| rng.unit() * 2.0).collect();
    let wq: Vec<f32> = (0..hd).map(|_| 0.5 + rng.unit().abs()).collect();
    let wk: Vec<f32> = (0..hd).map(|_| 0.5 + rng.unit().abs()).collect();

    let d_qg = ctx.alloc(qg.len() * 4).unwrap(); d_qg.upload(&to_bytes_f32(&qg)).unwrap();
    let d_kk = ctx.alloc(kk.len() * 4).unwrap(); d_kk.upload(&to_bytes_f32(&kk)).unwrap();
    let d_wq = ctx.alloc(wq.len() * 4).unwrap(); d_wq.upload(&to_bytes_f32(&wq)).unwrap();
    let d_wk = ctx.alloc(wk.len() * 4).unwrap(); d_wk.upload(&to_bytes_f32(&wk)).unwrap();

    let d_q_ref = ctx.alloc(rows * qd * 4).unwrap();
    let d_g_ref = ctx.alloc(rows * qd * 4).unwrap();
    let mut d_k_ref = ctx.alloc(cap * kd * 4).unwrap(); d_k_ref.zero().unwrap();

    let d_q_q8 = ctx.alloc(rows * qd * 4).unwrap();
    let d_g_q8 = ctx.alloc(rows * qd * 4).unwrap();
    let mut d_k_q8 = ctx.alloc(cap * rs).unwrap(); d_k_q8.zero().unwrap();

    let eps = 1e-6f32;
    let rd = 64u32;
    let theta = 10_000_000.0f32;

    for &(fast, ref_k, q8_k) in &[
        (false, "head_norm_rope_qgk_rows", "head_norm_rope_qgk_rows_q8"),
        (true, "head_norm_rope_qgk_rows_f", "head_norm_rope_qgk_rows_fq8"),
    ] {
        if fast {
            unsafe {
                prog.launch("rope_pow_table", (1, 1, 1), (128, 1, 1), 0, None, &[Arg::F32(theta), Arg::U32(rd)]).unwrap();
            }
        }
        unsafe {
            prog.launch(
                ref_k,
                ((nh + nkv) as u32, rows as u32, 1),
                (128, 1, 1),
                0,
                None,
                &[
                    Arg::Ptr(d_qg.ptr), Arg::Ptr(d_q_ref.ptr), Arg::Ptr(d_g_ref.ptr), Arg::Ptr(d_wq.ptr),
                    Arg::Ptr(d_kk.ptr), Arg::Ptr(d_k_ref.ptr), Arg::Ptr(d_wk.ptr),
                    Arg::U32(nh as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32),
                    Arg::F32(eps), Arg::U32(pos0 as u32), Arg::U32(rd), Arg::F32(theta),
                ],
            ).unwrap();
            prog.launch(
                q8_k,
                ((nh + nkv) as u32, rows as u32, 1),
                (128, 1, 1),
                0,
                None,
                &[
                    Arg::Ptr(d_qg.ptr), Arg::Ptr(d_q_q8.ptr), Arg::Ptr(d_g_q8.ptr), Arg::Ptr(d_wq.ptr),
                    Arg::Ptr(d_kk.ptr), Arg::Ptr(d_k_q8.ptr), Arg::Ptr(d_wk.ptr),
                    Arg::U32(nh as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32),
                    Arg::F32(eps), Arg::U32(pos0 as u32), Arg::U32(rd), Arg::F32(theta),
                ],
            ).unwrap();
        }
        ctx.synchronize().unwrap();

        let (mut q1, mut q2, mut g1, mut g2) = (vec![0u8; rows * qd * 4], vec![0u8; rows * qd * 4], vec![0u8; rows * qd * 4], vec![0u8; rows * qd * 4]);
        d_q_ref.download(&mut q1).unwrap(); d_q_q8.download(&mut q2).unwrap();
        d_g_ref.download(&mut g1).unwrap(); d_g_q8.download(&mut g2).unwrap();
        assert_eq!(q1, q2, "{q8_k}: q output must be bit-exact");
        assert_eq!(g1, g2, "{q8_k}: gate output must be bit-exact");

        let mut kr_b = vec![0u8; cap * kd * 4]; d_k_ref.download(&mut kr_b).unwrap();
        let kr = from_bytes_f32(&kr_b);
        let mut kq_b = vec![0u8; cap * rs]; d_k_q8.download(&mut kq_b).unwrap();
        let kq = dequant_q8_rows(&kq_b, pos0, rows, nkv, hd);
        let rl2 = rel_l2(&kq, &kr[pos0 * kd..(pos0 + rows) * kd]);
        assert!(rl2 < 0.015, "{q8_k}: dequantized K rel_l2 {rl2:.5} >= 0.015");
    }
}

#[test]
fn all_seven_q8_attention_kernels_match_f32() {
    let Some(ctx) = gpu_ctx() else { return };
    let prog = Program::load(&ctx, file("forward.cu").unwrap()).expect("compile forward.cu");
    let mut rng = Rng(0x1357_9bdf_2468_ace0);

    let (nh, nkv, hd) = (24usize, 2usize, 256usize);
    let group = nh / nkv; // 12
    let qd = nh * hd;
    let kd = nkv * hd;
    let cap = 320usize;
    let rows = 4usize;
    let ntok0 = 280usize;
    let rs = nkv * kv_q8_head_bytes(hd);

    let q: Vec<f32> = (0..rows * qd).map(|_| rng.unit() * 0.8).collect();
    let gate: Vec<f32> = (0..rows * qd).map(|_| rng.unit() * 1.5).collect();
    let kc: Vec<f32> = (0..cap * kd).map(|_| rng.unit() * 0.9).collect();
    let vc: Vec<f32> = (0..cap * kd).map(|_| rng.unit() * 0.9).collect();

    let d_q = ctx.alloc(q.len() * 4).unwrap(); d_q.upload(&to_bytes_f32(&q)).unwrap();
    let d_gate = ctx.alloc(gate.len() * 4).unwrap(); d_gate.upload(&to_bytes_f32(&gate)).unwrap();
    let d_kc_f32 = ctx.alloc(kc.len() * 4).unwrap(); d_kc_f32.upload(&to_bytes_f32(&kc)).unwrap();
    let d_vc_f32 = ctx.alloc(vc.len() * 4).unwrap(); d_vc_f32.upload(&to_bytes_f32(&vc)).unwrap();

    let d_kc_q8 = ctx.alloc(cap * rs).unwrap();
    let d_vc_q8 = ctx.alloc(cap * rs).unwrap();
    unsafe {
        prog.launch("kv_quant_q8_rows", (nkv as u32, cap as u32, 1), (128, 1, 1), 0, None, &[
            Arg::Ptr(d_kc_f32.ptr), Arg::Ptr(d_kc_q8.ptr), Arg::U32(0), Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(cap as u32),
        ]).unwrap();
        prog.launch("kv_quant_q8_rows", (nkv as u32, cap as u32, 1), (128, 1, 1), 0, None, &[
            Arg::Ptr(d_vc_f32.ptr), Arg::Ptr(d_vc_q8.ptr), Arg::U32(0), Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(cap as u32),
        ]).unwrap();
    }

    let d_scr = ctx.alloc(rows * nh * cap * 8).unwrap();
    let d_o_ref = ctx.alloc(rows * qd * 4).unwrap();
    let mut d_o_q8 = ctx.alloc(rows * qd * 4).unwrap();

    // 1. attention_heads_q8 (single token, dense & sparse)
    let sel_single: Vec<u32> = (0..180u32).map(|i| (i * 3) % (ntok0 as u32)).collect();
    let d_toks1 = ctx.alloc(sel_single.len() * 4).unwrap();
    d_toks1.upload(&to_bytes_u32(&sel_single)).unwrap();
    for &(tptr, ntok, tag) in &[(0u64, ntok0, "attention_heads_q8(dense)"), (d_toks1.ptr, sel_single.len(), "attention_heads_q8(sparse)")] {
        unsafe {
            prog.launch("attention_heads", (nh as u32, 1, 1), (256, 1, 1), 0, None, &[
                Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_f32.ptr), Arg::Ptr(d_vc_f32.ptr), Arg::Ptr(tptr), Arg::U32(ntok as u32),
                Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_ref.ptr), Arg::Ptr(d_scr.ptr), Arg::U32(cap as u32),
                Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(group as u32),
            ]).unwrap();
            prog.launch("attention_heads_q8", (nh as u32, 1, 1), (256, 1, 1), 0, None, &[
                Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_q8.ptr), Arg::Ptr(d_vc_q8.ptr), Arg::Ptr(tptr), Arg::U32(ntok as u32),
                Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_q8.ptr), Arg::Ptr(d_scr.ptr), Arg::U32(cap as u32),
                Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(group as u32),
            ]).unwrap();
        }
        ctx.synchronize().unwrap();
        let (mut br, mut bq) = (vec![0u8; qd * 4], vec![0u8; qd * 4]);
        d_o_ref.download(&mut br).unwrap(); d_o_q8.download(&mut bq).unwrap();
        check_attn_out(&from_bytes_f32(&bq), &from_bytes_f32(&br), tag);
    }

    // 2. Dense batched: attention_heads_rows_q8, attention_group_dense_rows_{f32,tc}_q8
    unsafe {
        prog.launch("attention_heads_rows", (nh as u32, rows as u32, 1), (256, 1, 1), 0, None, &[
            Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_f32.ptr), Arg::Ptr(d_vc_f32.ptr), Arg::U32(ntok0 as u32),
            Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_ref.ptr), Arg::Ptr(d_scr.ptr), Arg::U32(cap as u32),
            Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(group as u32), Arg::U32(nh as u32),
        ]).unwrap();
    }
    ctx.synchronize().unwrap();
    let mut dense_ref_b = vec![0u8; rows * qd * 4];
    d_o_ref.download(&mut dense_ref_b).unwrap();
    let dense_ref = from_bytes_f32(&dense_ref_b);

    unsafe {
        prog.launch("attention_heads_rows_q8", (nh as u32, rows as u32, 1), (256, 1, 1), 0, None, &[
            Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_q8.ptr), Arg::Ptr(d_vc_q8.ptr), Arg::U32(ntok0 as u32),
            Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_q8.ptr), Arg::Ptr(d_scr.ptr), Arg::U32(cap as u32),
            Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(group as u32), Arg::U32(nh as u32),
        ]).unwrap();
    }
    ctx.synchronize().unwrap();
    let mut bq = vec![0u8; rows * qd * 4];
    d_o_q8.download(&mut bq).unwrap();
    check_attn_out(&from_bytes_f32(&bq), &dense_ref, "attention_heads_rows_q8");

    for &(kname, hb) in &[("attention_group_dense_rows_f32_q8", 4usize), ("attention_group_dense_rows_tc_q8", 8usize)] {
        d_o_q8.zero().unwrap();
        unsafe {
            prog.launch(kname, ((nkv * group.div_ceil(hb)) as u32, rows as u32, 1), (256, 1, 1), 0, None, &[
                Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_q8.ptr), Arg::Ptr(d_vc_q8.ptr), Arg::U32(ntok0 as u32),
                Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_q8.ptr), Arg::Ptr(d_scr.ptr), Arg::U32(cap as u32),
                Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32(group as u32), Arg::U32(nh as u32), Arg::U32(hb as u32),
            ]).unwrap();
        }
        ctx.synchronize().unwrap();
        d_o_q8.download(&mut bq).unwrap();
        check_attn_out(&from_bytes_f32(&bq), &dense_ref, kname);
    }

    // 3. Sparse batched: attention_heads_sel_rows_q8, attention_group_sel_rows_{v3,tc}_q8
    let tstride = 264usize;
    let ntoks: Vec<u32> = vec![261, 197, 256, 203];
    let mut toks = vec![0u32; rows * tstride];
    for r in 0..rows {
        for i in 0..ntoks[r] as usize {
            toks[r * tstride + i] = ((i + r * 11) % (ntok0 + r)) as u32;
        }
    }
    let d_toks = ctx.alloc(toks.len() * 4).unwrap(); d_toks.upload(&to_bytes_u32(&toks)).unwrap();
    let d_ntoks = ctx.alloc(ntoks.len() * 4).unwrap(); d_ntoks.upload(&to_bytes_u32(&ntoks)).unwrap();

    unsafe {
        prog.launch("attention_heads_sel_rows", (nh as u32, rows as u32, 1), (256, 1, 1), 0, None, &[
            Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_f32.ptr), Arg::Ptr(d_vc_f32.ptr), Arg::Ptr(d_toks.ptr),
            Arg::U32(tstride as u32), Arg::Ptr(d_ntoks.ptr), Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_ref.ptr),
            Arg::Ptr(d_scr.ptr), Arg::U32(tstride as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32),
            Arg::U32(group as u32), Arg::U32(nh as u32),
        ]).unwrap();
    }
    ctx.synchronize().unwrap();
    let mut sel_ref_b = vec![0u8; rows * qd * 4];
    d_o_ref.download(&mut sel_ref_b).unwrap();
    let sel_ref = from_bytes_f32(&sel_ref_b);

    d_o_q8.zero().unwrap();
    unsafe {
        prog.launch("attention_heads_sel_rows_q8", (nh as u32, rows as u32, 1), (256, 1, 1), 0, None, &[
            Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_q8.ptr), Arg::Ptr(d_vc_q8.ptr), Arg::Ptr(d_toks.ptr),
            Arg::U32(tstride as u32), Arg::Ptr(d_ntoks.ptr), Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_q8.ptr),
            Arg::Ptr(d_scr.ptr), Arg::U32(tstride as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32),
            Arg::U32(group as u32), Arg::U32(nh as u32),
        ]).unwrap();
    }
    ctx.synchronize().unwrap();
    d_o_q8.download(&mut bq).unwrap();
    check_attn_out(&from_bytes_f32(&bq), &sel_ref, "attention_heads_sel_rows_q8");

    for &kname in &["attention_group_sel_rows_v3_q8", "attention_group_sel_rows_tc_q8"] {
        d_o_q8.zero().unwrap();
        unsafe {
            prog.launch(kname, (nkv as u32, rows as u32, 1), (256, 1, 1), 0, None, &[
                Arg::Ptr(d_q.ptr), Arg::Ptr(d_kc_q8.ptr), Arg::Ptr(d_vc_q8.ptr), Arg::Ptr(d_toks.ptr),
                Arg::U32(tstride as u32), Arg::Ptr(d_ntoks.ptr), Arg::Ptr(d_gate.ptr), Arg::Ptr(d_o_q8.ptr),
                Arg::Ptr(d_scr.ptr), Arg::U32(tstride as u32), Arg::U32(nkv as u32), Arg::U32(hd as u32),
                Arg::U32(group as u32), Arg::U32(nh as u32), Arg::U32(group as u32),
            ]).unwrap();
        }
        ctx.synchronize().unwrap();
        d_o_q8.download(&mut bq).unwrap();
        check_attn_out(&from_bytes_f32(&bq), &sel_ref, kname);
    }
}
