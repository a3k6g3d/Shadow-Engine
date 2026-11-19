//! Unit tests and microbenchmarks for `lm_head` Q2_P dp4a proxy screening and sparse EXL3 rescoring.

use shadowzero_cuda::kernels::{file, Arg, Program};
use shadowzero_cuda::{nvrtc, Context, DeviceBuffer};
use shadowzero_model::half::{f16_to_f32, f32_to_f16};

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

fn download_f32(buf: &DeviceBuffer, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; buf.bytes];
    buf.download(&mut b).unwrap();
    b[..n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn download_u32(buf: &DeviceBuffer, n: usize) -> Vec<u32> {
    let mut b = vec![0u8; buf.bytes];
    buf.download(&mut b).unwrap();
    b[..n * 4]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn make_exl3_k5_blob(rng: &mut Rng, rows: usize, cols: usize, hot_rows: &[usize]) -> Vec<u8> {
    let kb = 10usize; // K=5
    let nt = rows / 16;
    let kt = cols / 16;
    let tr_words = nt * kt * 4 * kb;
    let header_bytes = 2 * cols + 2 * rows + 4;
    let mut out = vec![0u8; header_bytes + tr_words * 4];
    for c in 0..cols {
        let h = f32_to_f16(0.8 + 0.4 * rng.unit().abs());
        out[c * 2..c * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    for r in 0..rows {
        let sc = if hot_rows.contains(&r) { 6.0 } else { 0.8 + 0.4 * rng.unit().abs() };
        let h = f32_to_f16(sc);
        let o = 2 * cols + r * 2;
        out[o..o + 2].copy_from_slice(&h.to_le_bytes());
    }
    // 4-byte marker left 0 (mul1)
    for w_i in 0..tr_words {
        let v = (rng.next() & 0xffff_ffff) as u32;
        let o = header_bytes + w_i * 4;
        out[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn test_q2_proxy_and_sparse_topk_exactness() {
    let Some(ctx) = gpu_ctx() else { return };
    let ex3 = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
    let fw = Program::load(&ctx, file("forward.cu").unwrap()).unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng(0xabcdef0123456789);

    let (rows, cols, m, k_blk) = (4096usize, 2560usize, 4usize, 12usize);
    let n_ob = rows / 128;
    let n_b = cols / 128;
    let hot: [usize; 0] = [];
    let blob = make_exl3_k5_blob(&mut rng, rows, cols, &hot);

    let d_w = ctx.alloc(blob.len()).unwrap();
    d_w.upload(&blob).unwrap();
    let d_wq2 = ctx.alloc(rows * cols / 4).unwrap();
    let d_wsc = ctx.alloc(rows * n_b * 2).unwrap();

    unsafe {
        ex3.launch("exl3_build_q2_proxy_5", (n_ob as u32, n_b as u32, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_w.ptr), Arg::Ptr(d_wq2.ptr), Arg::Ptr(d_wsc.ptr), Arg::U32(rows as u32), Arg::U32(cols as u32),
        ]).unwrap();
    }

    let x: Vec<f32> = (0..m * cols).map(|_| rng.unit()).collect();
    let d_x = ctx.alloc(m * cols * 4).unwrap();
    let d_xr = ctx.alloc(m * cols * 4).unwrap();
    let d_y_dense = ctx.alloc(m * rows * 4).unwrap();
    let d_y_sparse = ctx.alloc(m * rows * 4).unwrap();
    let d_xq = ctx.alloc(m * n_b * 32 * 4).unwrap();
    let d_xsc = ctx.alloc(m * n_b * 4).unwrap();
    let d_xcorr = ctx.alloc(m * n_b * 4).unwrap();
    let d_bmax = ctx.alloc(m * n_ob * 4).unwrap();
    let mut d_seen = ctx.alloc(n_ob.div_ceil(32) * 4).unwrap();
    let max_sel = (m * k_blk).min(n_ob);
    let d_sel = ctx.alloc((1 + max_sel) * 4).unwrap();
    d_x.upload(&to_bytes_f32(&x)).unwrap();
    d_seen.zero().unwrap();

    unsafe {
        ex3.launch("exl3_rotate_rows_f32", (((m * n_b) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_w.ptr), Arg::Ptr(d_x.ptr), Arg::U32(cols as u32), Arg::Ptr(d_xr.ptr), Arg::U32(cols as u32), Arg::U32(m as u32),
        ]).unwrap();
        ex3.launch("matvec_cols_exl3_pre_m4s1_5", (n_ob as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_w.ptr), Arg::Ptr(d_xr.ptr), Arg::Ptr(d_y_dense.ptr), Arg::U32(rows as u32),
            Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32),
        ]).unwrap();
        fw.launch("proxy_q2_quant_x", (n_b as u32, m as u32, 1), (128, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_xr.ptr), Arg::U32(cols as u32), Arg::Ptr(d_xq.ptr), Arg::Ptr(d_xsc.ptr), Arg::Ptr(d_xcorr.ptr), Arg::U32(cols as u32),
        ]).unwrap();
        fw.launch("proxy_q2_dp4a_bmax_m4", (n_ob as u32, 1, 1), (128, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_wq2.ptr), Arg::Ptr(d_wsc.ptr), Arg::Ptr(d_xq.ptr), Arg::Ptr(d_xsc.ptr), Arg::Ptr(d_xcorr.ptr),
            Arg::Ptr(d_bmax.ptr), Arg::U32(n_ob as u32), Arg::U32(cols as u32), Arg::U32(m as u32),
        ]).unwrap();
        fw.launch("proxy_pick_rows", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_bmax.ptr), Arg::U32(n_ob as u32), Arg::U32(k_blk as u32), Arg::Ptr(d_seen.ptr),
        ]).unwrap();
        fw.launch("proxy_compact_seen", (1, 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_seen.ptr), Arg::U32(n_ob as u32), Arg::U32(max_sel as u32), Arg::Ptr(d_sel.ptr),
        ]).unwrap();
        ex3.launch("matvec_cols_exl3_pre_sparse_m4s1_5", (max_sel as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_w.ptr), Arg::Ptr(d_xr.ptr), Arg::Ptr(d_y_sparse.ptr), Arg::U32((max_sel * 128) as u32),
            Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32), Arg::Ptr(d_sel.ptr),
        ]).unwrap();
    }

    // Also test topk_slices_blks + topk_merge and df_top16_blks on the sparse logits vs CPU reference over selected blocks
    let nb_sp = max_sel.div_ceil(16);
    let d_sv = ctx.alloc(m * nb_sp * 64 * 4).unwrap();
    let d_si = ctx.alloc(m * nb_sp * 64 * 4).unwrap();
    let d_fin = ctx.alloc(m * 128 * 4).unwrap();
    let d_df_id = ctx.alloc(m * 16 * 4).unwrap();
    let d_df_val = ctx.alloc(m * 16 * 4).unwrap();
    unsafe {
        fw.launch("topk_slices_blks", ((m * nb_sp) as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_y_sparse.ptr), Arg::U32((max_sel * 128) as u32), Arg::Ptr(d_sel.ptr),
            Arg::U32(nb_sp as u32), Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr),
        ]).unwrap();
        fw.launch("topk_merge_blks", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr), Arg::U32(nb_sp as u32), Arg::Ptr(d_fin.ptr),
        ]).unwrap();
        fw.launch("df_top16_blks", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_y_sparse.ptr), Arg::U32((max_sel * 128) as u32), Arg::Ptr(d_sel.ptr),
            Arg::Ptr(d_df_id.ptr), Arg::Ptr(d_df_val.ptr),
        ]).unwrap();
    }
    ctx.synchronize().unwrap();

    let sel = download_u32(&d_sel, 1 + max_sel);
    let n_sel = sel[0] as usize;
    assert!(n_sel >= k_blk && n_sel <= max_sel, "unexpected n_sel={n_sel}");
    let dense = download_f32(&d_y_dense, m * rows);
    let bmax_got = download_f32(&d_bmax, m * n_ob);
    let mut bmax_want = vec![-3.0e38f32; m * n_ob];
    for j in 0..m {
        for ob in 0..n_ob {
            let mx = dense[j * rows + ob * 128..j * rows + (ob + 1) * 128].iter().copied().fold(-3.0e38f32, f32::max);
            bmax_want[j * n_ob + ob] = mx;
        }
        let g = &bmax_got[j * n_ob..(j + 1) * n_ob];
        let w = &bmax_want[j * n_ob..(j + 1) * n_ob];
        let (mut dot, mut ng, mut nw) = (0f64, 0f64, 0f64);
        for (&a, &b) in g.iter().zip(w) {
            dot += (a as f64) * (b as f64);
            ng += (a as f64) * (a as f64);
            nw += (b as f64) * (b as f64);
        }
        let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-24);
        eprintln!("bmax j={j} cos={cos:.5} got0..3={:.3?} want0..3={:.3?}", &g[..4], &w[..4]);
        assert!(cos > 0.85, "proxy bmax cosine too low ({cos:.5})! Quantized proxy layout bug!");
    }
    let sparse = download_f32(&d_y_sparse, m * max_sel * 128);
    let fin = download_u32(&d_fin, m * 128);
    let df_ids = download_u32(&d_df_id, m * 16);
    let df_vals = download_f32(&d_df_val, m * 16);

    // Every selected block in sparse must be bit-identical to dense!
    for j in 0..m {
        for k in 0..n_sel {
            let ob = sel[1 + k] as usize;
            for i in 0..128 {
                let got = sparse[j * max_sel * 128 + k * 128 + i];
                let want = dense[j * rows + ob * 128 + i];
                assert_eq!(got.to_bits(), want.to_bits(), "sparse/dense mismatch j={j} ob={ob} i={i}");
            }
        }
        // Top-16 and Top-64 on selected blocks must match exact reference over selected blocks
        let mut pairs: Vec<(f32, u32)> = Vec::new();
        for &ob in &sel[1..=n_sel] {
            for i in 0..128 {
                let tid = ob * 128 + i as u32;
                pairs.push((dense[j * rows + tid as usize], tid));
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for r in 0..64 {
            let gv = f32::from_bits(fin[j * 128 + r]);
            let gi = fin[j * 128 + 64 + r];
            assert_eq!((gv.to_bits(), gi), (pairs[r].0.to_bits(), pairs[r].1), "topk64 mismatch j={j} r={r}");
        }
        for r in 0..16 {
            assert_eq!((df_vals[j * 16 + r].to_bits(), df_ids[j * 16 + r]), (pairs[r].0.to_bits(), pairs[r].1), "df_top16 mismatch j={j} r={r}");
        }
        // Check hot blocks survived proxy screen
        let mut dense_full: Vec<(f32, u32)> = dense[j * rows..(j + 1) * rows].iter().copied().zip(0..rows as u32).collect();
        dense_full.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let top1_id = dense_full[0].1;
        assert!(sel[1..=n_sel].contains(&(top1_id / 128)), "proxy missed top1 row {top1_id} for j={j}");
    }
    let _ = f16_to_f32(f32_to_f16(1.0));
}

#[test]
#[ignore]
fn bench_head_proxy_vs_exl3() {
    let Some(ctx) = gpu_ctx() else { return };
    let ex3 = Program::load(&ctx, file("exl3.cu").unwrap()).unwrap();
    let fw = Program::load(&ctx, file("forward.cu").unwrap()).unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = Rng(0xdeadbeefcafe1234);

    let (rows, cols, k_blk) = (248320usize, 2560usize, 128usize);
    let n_ob = rows / 128;
    let n_b = cols / 128;
    let blob = make_exl3_k5_blob(&mut rng, rows, cols, &[1000, 50000, 120000, 220000]);

    let d_w = ctx.alloc(blob.len()).unwrap();
    d_w.upload(&blob).unwrap();
    let d_wq2 = ctx.alloc(rows * cols / 4).unwrap();
    let d_wsc = ctx.alloc(rows * n_b * 2).unwrap();

    let ev0 = ctx.event().unwrap();
    let ev1 = ctx.event().unwrap();
    ev0.record(Some(&stream)).unwrap();
    unsafe {
        ex3.launch("exl3_build_q2_proxy_5", (n_ob as u32, n_b as u32, 1), (256, 1, 1), 0, Some(&stream), &[
            Arg::Ptr(d_w.ptr), Arg::Ptr(d_wq2.ptr), Arg::Ptr(d_wsc.ptr), Arg::U32(rows as u32), Arg::U32(cols as u32),
        ]).unwrap();
    }
    ev1.record(Some(&stream)).unwrap();
    ev1.synchronize().unwrap();
    eprintln!("exl3_build_q2_proxy_5 startup time: {:.2} ms", ev1.since(&ev0).unwrap());

    let max_m = 4usize;
    let d_x = ctx.alloc(max_m * cols * 4).unwrap();
    let d_xr = ctx.alloc(max_m * cols * 4).unwrap();
    let d_logits = ctx.alloc(max_m * rows * 4).unwrap();
    let d_xq = ctx.alloc(max_m * n_b * 32 * 4).unwrap();
    let d_xsc = ctx.alloc(max_m * n_b * 4).unwrap();
    let d_xcorr = ctx.alloc(max_m * n_b * 4).unwrap();
    let d_bmax = ctx.alloc(max_m * n_ob * 4).unwrap();
    let mut d_seen = ctx.alloc(n_ob.div_ceil(32) * 4).unwrap();
    let d_sel = ctx.alloc((1 + max_m * k_blk) * 4).unwrap();
    let nb_full = rows.div_ceil(4096);
    let d_sv = ctx.alloc(max_m * nb_full * 64 * 4).unwrap();
    let d_si = ctx.alloc(max_m * nb_full * 64 * 4).unwrap();
    let d_fin = ctx.alloc(max_m * 128 * 4).unwrap();

    d_x.upload(&to_bytes_f32(&(0..max_m * cols).map(|_| rng.unit()).collect::<Vec<f32>>())).unwrap();
    d_seen.zero().unwrap();

    let t_ev = |f: &dyn Fn()| -> f64 {
        for _ in 0..10 { f(); }
        ctx.synchronize().unwrap();
        ev0.record(Some(&stream)).unwrap();
        for _ in 0..100 { f(); }
        ev1.record(Some(&stream)).unwrap();
        ev1.synchronize().unwrap();
        (ev1.since(&ev0).unwrap() as f64 / 100.0) * 1e3
    };

    for m in [1usize, 3, 4] {
        let dense_us = t_ev(&|| unsafe {
            if m == 1 {
                ex3.launch("matvec_exl3_s1_5", (n_ob as u32, 1, 1), (256, 1, 1), (cols * 4) as u32, Some(&stream), &[
                    Arg::Ptr(d_w.ptr), Arg::Ptr(d_x.ptr), Arg::Ptr(d_logits.ptr), Arg::U32(rows as u32), Arg::U32(cols as u32),
                ]).unwrap();
            } else {
                ex3.launch("exl3_rotate_rows_f32", (((m * n_b) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_w.ptr), Arg::Ptr(d_x.ptr), Arg::U32(cols as u32), Arg::Ptr(d_xr.ptr), Arg::U32(cols as u32), Arg::U32(m as u32),
                ]).unwrap();
                ex3.launch("matvec_cols_exl3_pre_m4s1_5", (n_ob as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                    Arg::Ptr(d_w.ptr), Arg::Ptr(d_xr.ptr), Arg::Ptr(d_logits.ptr), Arg::U32(rows as u32),
                    Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32),
                ]).unwrap();
            }
            fw.launch("topk_slices", ((m * nb_full) as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_logits.ptr), Arg::U32(rows as u32), Arg::U32(rows as u32), Arg::U32(nb_full as u32),
                Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr),
            ]).unwrap();
            fw.launch("topk_merge", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr), Arg::U32(nb_full as u32), Arg::Ptr(d_fin.ptr),
            ]).unwrap();
        });

        let max_sel = (m * k_blk).min(n_ob);
        let nb_sp = max_sel.div_ceil(16);
        let sparse_us = t_ev(&|| unsafe {
            ex3.launch("exl3_rotate_rows_f32", (((m * n_b) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_w.ptr), Arg::Ptr(d_x.ptr), Arg::U32(cols as u32), Arg::Ptr(d_xr.ptr), Arg::U32(cols as u32), Arg::U32(m as u32),
            ]).unwrap();
            fw.launch("proxy_q2_quant_x", (n_b as u32, m as u32, 1), (128, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_xr.ptr), Arg::U32(cols as u32), Arg::Ptr(d_xq.ptr), Arg::Ptr(d_xsc.ptr), Arg::Ptr(d_xcorr.ptr), Arg::U32(cols as u32),
            ]).unwrap();
            let p_k = if m == 1 { "proxy_q2_dp4a_bmax_m1" } else { "proxy_q2_dp4a_bmax_m4" };
            fw.launch(p_k, (n_ob as u32, 1, 1), (128, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wq2.ptr), Arg::Ptr(d_wsc.ptr), Arg::Ptr(d_xq.ptr), Arg::Ptr(d_xsc.ptr), Arg::Ptr(d_xcorr.ptr),
                Arg::Ptr(d_bmax.ptr), Arg::U32(n_ob as u32), Arg::U32(cols as u32), Arg::U32(m as u32),
            ]).unwrap();
            fw.launch("proxy_pick_rows", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_bmax.ptr), Arg::U32(n_ob as u32), Arg::U32(k_blk as u32), Arg::Ptr(d_seen.ptr),
            ]).unwrap();
            fw.launch("proxy_compact_seen", (1, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_seen.ptr), Arg::U32(n_ob as u32), Arg::U32(max_sel as u32), Arg::Ptr(d_sel.ptr),
            ]).unwrap();
            ex3.launch("matvec_cols_exl3_pre_sparse_m4s1_5", (max_sel as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_w.ptr), Arg::Ptr(d_xr.ptr), Arg::Ptr(d_logits.ptr), Arg::U32((max_sel * 128) as u32),
                Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32), Arg::Ptr(d_sel.ptr),
            ]).unwrap();
            fw.launch("topk_slices_blks", ((m * nb_sp) as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_logits.ptr), Arg::U32((max_sel * 128) as u32), Arg::Ptr(d_sel.ptr),
                Arg::U32(nb_sp as u32), Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr),
            ]).unwrap();
            fw.launch("topk_merge_blks", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr), Arg::U32(nb_sp as u32), Arg::Ptr(d_fin.ptr),
            ]).unwrap();
        });
        let p1 = t_ev(&|| unsafe {
            ex3.launch("exl3_rotate_rows_f32", (((m * n_b) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_w.ptr), Arg::Ptr(d_x.ptr), Arg::U32(cols as u32), Arg::Ptr(d_xr.ptr), Arg::U32(cols as u32), Arg::U32(m as u32),
            ]).unwrap();
            fw.launch("proxy_q2_quant_x", (n_b as u32, m as u32, 1), (128, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_xr.ptr), Arg::U32(cols as u32), Arg::Ptr(d_xq.ptr), Arg::Ptr(d_xsc.ptr), Arg::Ptr(d_xcorr.ptr), Arg::U32(cols as u32),
            ]).unwrap();
            let p_k = if m == 1 { "proxy_q2_dp4a_bmax_m1" } else { "proxy_q2_dp4a_bmax_m4" };
            fw.launch(p_k, (n_ob as u32, 1, 1), (128, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_wq2.ptr), Arg::Ptr(d_wsc.ptr), Arg::Ptr(d_xq.ptr), Arg::Ptr(d_xsc.ptr), Arg::Ptr(d_xcorr.ptr),
                Arg::Ptr(d_bmax.ptr), Arg::U32(n_ob as u32), Arg::U32(cols as u32), Arg::U32(m as u32),
            ]).unwrap();
        });
        let p2 = t_ev(&|| unsafe {
            fw.launch("proxy_pick_rows", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_bmax.ptr), Arg::U32(n_ob as u32), Arg::U32(k_blk as u32), Arg::Ptr(d_seen.ptr),
            ]).unwrap();
            fw.launch("proxy_compact_seen", (1, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_seen.ptr), Arg::U32(n_ob as u32), Arg::U32(max_sel as u32), Arg::Ptr(d_sel.ptr),
            ]).unwrap();
        });
        let p3 = t_ev(&|| unsafe {
            ex3.launch("matvec_cols_exl3_pre_sparse_m4s1_5", (max_sel as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_w.ptr), Arg::Ptr(d_xr.ptr), Arg::Ptr(d_logits.ptr), Arg::U32((max_sel * 128) as u32),
                Arg::U32(rows as u32), Arg::U32(cols as u32), Arg::U32(m as u32), Arg::Ptr(d_sel.ptr),
            ]).unwrap();
        });
        let p4 = t_ev(&|| unsafe {
            fw.launch("topk_slices_blks", ((m * nb_sp) as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_logits.ptr), Arg::U32((max_sel * 128) as u32), Arg::Ptr(d_sel.ptr),
                Arg::U32(nb_sp as u32), Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr),
            ]).unwrap();
            fw.launch("topk_merge_blks", (m as u32, 1, 1), (256, 1, 1), 0, Some(&stream), &[
                Arg::Ptr(d_sv.ptr), Arg::Ptr(d_si.ptr), Arg::U32(nb_sp as u32), Arg::Ptr(d_fin.ptr),
            ]).unwrap();
        });
        let sel_host = download_u32(&d_sel, 1);
        eprintln!(
            "m={m}: dense+topk={dense_us:7.2} us | q2p+sparse+topk={sparse_us:7.2} us ({:.2}x) [proxy={p1:.1} pick={p2:.1} exl3_sp={p3:.1} (n_sel={}/{max_sel}) topk_sp={p4:.1}]",
            dense_us / sparse_us, sel_host[0]
        );
    }
}
