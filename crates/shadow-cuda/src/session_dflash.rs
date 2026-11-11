//! The DFlash 2 drafter (`crate::dflash`) on the session's kernels: K/V for every position the main model reads (from
//! the five feature taps), and a block of drafts in one pass.  See dflash.rs for the model.

use super::{cols_rows_per_block, row, GpuSession, COLS_MAX};
use crate::dflash::*;
use crate::kernels::Arg;
use shadow_model::quant::dequantize;

/// `SHADOW_DF_KEEP=0`: the old behaviour, any snapshot restore switches the drafter off until a conversation is read from position 0
/// (STATUS 98: decode -17% for the rest of the session).  On by default.
fn df_keep() -> bool {
    static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| !std::env::var("SHADOW_DF_KEEP").is_ok_and(|v| v == "0"))
}

impl<'c, 'm> GpuSession<'c, 'm> {
    /// y_j = W x_j (a BF16 matrix of `rows` x `cols` at `w`) for j < m, x_j at x + j * xs floats, y_j at y + j * ys.
    fn df_mv_ptr(&self, w: u64, q8: bool, rows: usize, cols: usize, x: u64, xs: usize, y: u64, ys: usize, m: usize) {
        let entry = if q8 { "matvec_cols_q8_0" } else { "matvec_cols_bf16" };
        for j0 in (0..m).step_by(COLS_MAX) {
            let n = (m - j0).min(COLS_MAX);
            // SAFETY: `rows` rows of `cols` BF16 weights at w; n inputs of `cols` floats at x + j * xs and n outputs of
            // `rows` floats at y + j * ys, inside the drafter's buffers (the callers' strides are their sizes).
            unsafe {
                self.mvp.launch(entry, ((rows as u32).div_ceil(cols_rows_per_block(entry, rows)), 1, 1), (256, 1, 1), 0, self.ls(),
                                &[Arg::Ptr(w), Arg::Ptr(x + (j0 * xs * 4) as u64), Arg::U32(xs as u32),
                                  Arg::Ptr(y + (j0 * ys * 4) as u64), Arg::U32(ys as u32), Arg::U32(rows as u32),
                                  Arg::U32(cols as u32), Arg::U32(n as u32)])
            }
            .unwrap_or_else(|err| panic!("dflash matvec: {err}"));
        }
    }

    fn df_mv(&self, name: &str, x: u64, xs: usize, y: u64, ys: usize, m: usize) {
        let df = self.dflash.as_ref().unwrap();
        let (rows, cols) = df.dims(name);
        self.df_mv_ptr(df.ptr(name), df.tensors[name].q8, rows, cols, x, xs, y, ys, m);
    }

    /// The layer-`l` output residual of the m rows at `r` (streams h apart in each row), if `l` is one of the taps: its
    /// mean over the streams into the feature rows.
    pub(super) fn df_capture(&self, l: usize, r: u64, m: usize) {
        let Some(slot) = TAPS.iter().position(|&t| t == l) else { return };
        let c = &self.m.cfg;
        // a pipelined group writes its rows of the feature buffer, not the first ones
        let feat = self.dflash.as_ref().unwrap().bufs.feat.ptr + (self.row_shift * NFEAT * 4) as u64;
        // a pass of more rows than the buffers hold: the last ones (the drafter looks back WINDOW positions, `dflash_observe` takes the same)
        let skip = m.saturating_sub(DF_ROWS);
        let r = r + (skip * c.streams * c.hidden * 4) as u64;
        let m = m - skip;
        self.k2("df_capture", (c.hidden as u32).div_ceil(256), m as u32, 256,
                &[Arg::Ptr(r), Arg::Ptr(feat), Arg::U32(slot as u32), Arg::U32(c.hidden as u32), Arg::U32(c.streams as u32),
                  Arg::U32(NFEAT as u32)]);
    }

    /// K and V of the m positions p0 .. p0 + m in layer i from the context rows `ctxh`, into the rings.
    fn df_kv(&self, i: usize, src: u64, pos0: usize, m: usize) {
        let df = self.dflash.as_ref().unwrap();
        let b = &df.bufs;
        let p = format!("layers.{i}.self_attn");
        self.df_mv(&format!("{p}.k_proj.weight"), src, HIDDEN, b.kp.ptr, KV_WIDTH, m);
        self.df_mv(&format!("{p}.v_proj.weight"), src, HIDDEN, b.vp.ptr, KV_WIDTH, m);
        self.df_kv_finish(i, pos0, m);
    }

    /// K normed and roped, K and V put in layer i's rings, from b.kp / b.vp (m rows).
    fn df_kv_finish(&self, i: usize, pos0: usize, m: usize) {
        let df = self.dflash.as_ref().unwrap();
        let b = &df.bufs;
        let p = format!("layers.{i}.self_attn");
        self.k("df_rmsnorm", (m * N_HEAD_KV) as u32, 256,
               &[Arg::Ptr(b.kp.ptr), Arg::Ptr(df.ptr(&format!("{p}.k_norm.weight"))), Arg::Ptr(b.kp.ptr), Arg::U32(HEAD_DIM as u32),
                 Arg::F32(EPS)]);
        self.k2("df_rope", N_HEAD_KV as u32, m as u32, 64, &[Arg::Ptr(b.kp.ptr), Arg::U32(pos0 as u32), Arg::U32(KV_WIDTH as u32),
                                                             Arg::F32(ROPE_THETA)]);
        for (src, ring) in [(b.kp.ptr, df.kring[i].ptr), (b.vp.ptr, df.vring[i].ptr)] {
            self.k2("df_ring_put", (KV_WIDTH as u32).div_ceil(256), m as u32, 256,
                    &[Arg::Ptr(src), Arg::Ptr(ring), Arg::U32(pos0 as u32), Arg::U32(RING as u32), Arg::U32(KV_WIDTH as u32)]);
        }
    }

    /// The main model has read positions p0 .. p0 + m, their features in the feature rows (`df_capture`): the drafter's
    /// context K/V for them.
    pub(super) fn dflash_observe(&mut self, m: usize, p0: usize) {
        // only the last DF_ROWS rows of a long pass have features (`df_capture`)
        let (valid, skip) = (p0 + m, m.saturating_sub(DF_ROWS));
        let (m, p0) = (m - skip, p0 + skip);
        let (feat, ctxh) = {
            let df = self.dflash.as_ref().unwrap();
            (df.bufs.feat.ptr, df.bufs.ctxh.ptr)
        };
        self.df_mv("fc.weight", feat, NFEAT, ctxh, HIDDEN, m);
        let hn = self.dflash.as_ref().unwrap().ptr("hidden_norm.weight");
        self.k("df_rmsnorm", m as u32, 256, &[Arg::Ptr(ctxh), Arg::Ptr(hn), Arg::Ptr(ctxh), Arg::U32(HIDDEN as u32), Arg::F32(EPS)]);
        for i in 0..LAYERS {
            self.df_kv(i, ctxh, p0, m);
        }
        if let Ok(dir) = std::env::var("SHADOW_DF_DUMP") {
            // the features of the rows just read, appended: (p0, m) as u32s then m x NFEAT floats
            let mut bytes = vec![0u8; m * NFEAT * 4];
            self.download(feat, &mut bytes);
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/feats.bin")).unwrap();
            f.write_all(&(p0 as u32).to_le_bytes()).unwrap();
            f.write_all(&(m as u32).to_le_bytes()).unwrap();
            for t in &self.tokens[p0..p0 + m] {
                f.write_all(&t.to_le_bytes()).unwrap();
            }
            f.write_all(&bytes).unwrap();
        }
        let df = self.dflash.as_mut().unwrap();
        let (valid_old, lo_old) = (df.valid, df.lo);
        df.valid = valid;
        if p0 == 0 {
            df.broken = false;
        }
        if df_keep() {
            // positions p0 .. valid now hold their own K/V.  Read from the start, or far past the end of what was usable, or below its start:
            // nothing earlier connects to them, the usable range starts here; read on from the end of the usable range: it grows.  A hole of a
            // few positions is read on from too: the last token a reply sampled is never stepped, so a chat's next prompt starts one position
            // past `valid` (the first version of this reset the range at every turn: accept 0.595 against 0.625, STATUS 98)
            if p0 == 0 || p0 > valid_old + 2 * BLOCK || p0 < lo_old {
                df.lo = p0;
            }
        }
    }

    /// The session was restored from a snapshot made elsewhere: the drafter's K/V for its positions may be another
    /// conversation's.
    pub fn dflash_break(&mut self) {
        if let Some(df) = self.dflash.as_mut() {
            df.broken = true;
        }
    }

    /// `restore` has put the session back to a snapshot of `n_s` positions (called with the old session still in `self.tokens`; `same` =
    /// its first `n_s` tokens are the snapshot's, no pictures involved).  The drafter keeps what is still its own: when the old session
    /// starts with the snapshot's tokens, its K/V ring holds those positions' features - except the slots the old session overwrote
    /// (every position it observed past the ring, and the `BLOCK` rows of its last draft), which `lo` leaves out of the window; when it was
    /// another conversation, nothing below the restore point is usable and the window starts there.  Before STATUS 98 any restore switched
    /// the drafter off for the rest of the session (`SHADOW_DF_KEEP=0` brings that back).
    pub(super) fn dflash_restored(&mut self, same: bool, n_s: usize) {
        let Some(df) = self.dflash.as_mut() else { return };
        if !df_keep() {
            df.broken = true;
            return;
        }
        if same && !df.broken && df.valid >= n_s {
            let stale = (df.valid + BLOCK).saturating_sub(RING);
            df.lo = df.lo.max(stale).min(n_s);
        } else {
            df.lo = n_s;
        }
        df.valid = n_s;
    }

    fn df_embed(&self, tok: u32) -> Vec<f32> {
        let (t, bytes) = self.m.dense("token_embd.weight");
        let (qk, qb) = t.ty.block().unwrap();
        let rb = (t.dims[0] / qk * qb) as usize;
        let mut x = vec![0f32; HIDDEN];
        dequantize(t.ty, &bytes[tok as usize * rb..(tok as usize + 1) * rb], &mut x).unwrap();
        x
    }

    /// Up to k (at most BLOCK - 1) drafts for the tokens after `next` (the token about to be stepped, which follows the
    /// session's last position): the block (`next`, then mask tokens) through the drafter's layers against the K/V of
    /// everything before it, 16 candidates a slot from the main model's head, and the selector's best path.  Empty
    /// without features for every position so far.
    pub fn dflash_draft(&mut self, next: u32, k: usize, min_p: f32) -> Vec<u32> {
        let n = self.tokens.len();
        let Some(df) = self.dflash.as_ref() else { return vec![] };
        let k = k.min(BLOCK - 1).min(self.cap.saturating_sub(n + 1));
        if k == 0 || n == 0 || df.valid < n || df.broken {
            return vec![];
        }
        if self.dflash.as_ref().unwrap().mask_emb.is_none() {
            let e = self.df_embed(MASK_TOKEN);
            self.dflash.as_mut().unwrap().mask_emb = Some(e);
        }
        let df = self.dflash.as_ref().unwrap();
        let mut emb = self.df_embed(next);
        for _ in 1..BLOCK {
            emb.extend_from_slice(df.mask_emb.as_ref().unwrap());
        }
        let b = &df.bufs;
        let prof = df.prof.is_some();
        let mut laps: Vec<(&'static str, f64)> = vec![];
        let mut t0 = std::time::Instant::now();
        let mut lap = |name: &'static str| {
            if prof {
                self.ctx.synchronize().unwrap();
                laps.push((name, t0.elapsed().as_secs_f64()));
                t0 = std::time::Instant::now();
            }
        };
        self.upload_to(b.emb.ptr, &super::f32s(&emb));
        let (h, nb) = (HIDDEN as u32, BLOCK as u32);
        let w = |name: &str| Arg::Ptr(df.ptr(name));
        let f = |x: u64| Arg::Ptr(x);
        self.k1("copy", BLOCK * HIDDEN, &[f(b.res.ptr), f(b.emb.ptr), Arg::U32(h * nb)]);
        let tp = [f(b.res.ptr), f(b.y2.ptr)];
        for i in 0..LAYERS {
            let pre = format!("layers.{i}");
            if i == 0 {
                self.k("df_rmsnorm", nb, 256, &[tp[0], w(&format!("{pre}.input_layernorm.weight")), f(b.xn.ptr), Arg::U32(h), Arg::F32(EPS)]);
            } else {
                self.k("df_add_rmsnorm", nb, 256, &[tp[0], tp[1], w(&format!("{pre}.input_layernorm.weight")), f(b.xn.ptr), Arg::U32(h),
                                                    Arg::F32(EPS)]);
            }
            // attention: the convolution on its input, q/k/v, the block's K/V into the rings, attention, o_proj
            let conv = |delta: u64, base: &str, side: u32, x: u64, y: u64| {
                self.k2("df_conv", h.div_ceil(256), nb, 256, &[f(x), f(delta), w(base), f(y), Arg::U32(side), Arg::U32(h),
                                                               Arg::U32(CONV_GROUP as u32), Arg::U32(N_GROUPS as u32)]);
            };
            self.df_mv(&format!("{pre}.attention_conv.kernel_projection.weight"), b.xn.ptr, HIDDEN, b.delta_a.ptr, 4 * N_GROUPS, BLOCK);
            conv(b.delta_a.ptr, &format!("{pre}.attention_conv.base_kernel"), 0, b.xn.ptr, b.xc.ptr);
            let p = format!("{pre}.self_attn");
            self.df_mv(&format!("{p}.q_proj.weight"), b.xc.ptr, HIDDEN, b.q.ptr, N_HEAD * HEAD_DIM, BLOCK);
            self.df_mv(&format!("{p}.k_proj.weight"), b.xc.ptr, HIDDEN, b.kp.ptr, KV_WIDTH, BLOCK);
            self.df_mv(&format!("{p}.v_proj.weight"), b.xc.ptr, HIDDEN, b.vp.ptr, KV_WIDTH, BLOCK);
            self.k("df_rmsnorm", (BLOCK * N_HEAD) as u32, 256, &[f(b.q.ptr), w(&format!("{p}.q_norm.weight")), f(b.q.ptr),
                                                                 Arg::U32(HEAD_DIM as u32), Arg::F32(EPS)]);
            self.k2("df_rope", N_HEAD as u32, nb, 64, &[f(b.q.ptr), Arg::U32(n as u32), Arg::U32((N_HEAD * HEAD_DIM) as u32),
                                                        Arg::F32(ROPE_THETA)]);
            self.df_kv_finish(i, n, BLOCK);
            lap("layer: qkv, norms, rope");
            // SAFETY: q is BLOCK x 4096 floats, the rings RING x 1024, ao BLOCK x 4096; the kernel reads slots
            // (lo + j) % RING for the window and the block, all written (context rows and this block's).
            unsafe {
                self.fw.launch("df_attention", (N_HEAD as u32, nb, 1), (128, 1, 1), ((WINDOW + BLOCK) * 4) as u32, self.ls(), &[
                    f(b.q.ptr), f(df.kring[i].ptr), f(df.vring[i].ptr), f(b.ao.ptr), Arg::U32(n as u32), Arg::U32(nb),
                    Arg::U32(WINDOW as u32), Arg::U32(RING as u32), Arg::F32((HEAD_DIM as f32).powf(-0.5)), Arg::U32(df.lo.min(n) as u32)])
            }
            .unwrap_or_else(|err| panic!("df_attention: {err}"));
            lap("layer: attention");
            self.df_mv(&format!("{p}.o_proj.weight"), b.ao.ptr, N_HEAD * HEAD_DIM, b.y.ptr, HIDDEN, BLOCK);
            conv(b.delta_a.ptr, &format!("{pre}.attention_conv.base_kernel"), 1, b.y.ptr, b.y2.ptr);
            self.k("df_add_rmsnorm", nb, 256, &[tp[0], tp[1], w(&format!("{pre}.post_attention_layernorm.weight")), f(b.xn.ptr), Arg::U32(h),
                                                Arg::F32(EPS)]);
            lap("layer: o_proj, conv, norm");
            // MLP with its own convolution
            self.df_mv(&format!("{pre}.mlp_conv.kernel_projection.weight"), b.xn.ptr, HIDDEN, b.delta_m.ptr, 4 * N_GROUPS, BLOCK);
            conv(b.delta_m.ptr, &format!("{pre}.mlp_conv.base_kernel"), 0, b.xn.ptr, b.xc.ptr);
            self.df_mv(&format!("{pre}.mlp.gate_proj.weight"), b.xc.ptr, HIDDEN, b.g.ptr, FF, BLOCK);
            self.df_mv(&format!("{pre}.mlp.up_proj.weight"), b.xc.ptr, HIDDEN, b.u.ptr, FF, BLOCK);
            self.k1("df_silu_mul", BLOCK * FF, &[f(b.g.ptr), f(b.u.ptr), Arg::U32((BLOCK * FF) as u32)]);
            self.df_mv(&format!("{pre}.mlp.down_proj.weight"), b.g.ptr, FF, b.y.ptr, HIDDEN, BLOCK);
            conv(b.delta_m.ptr, &format!("{pre}.mlp_conv.base_kernel"), 1, b.y.ptr, b.y2.ptr);
            lap("layer: mlp + convs");
        }
        self.k("df_add_rmsnorm", nb, 256, &[tp[0], tp[1], w("norm.weight"), f(b.hid.ptr), Arg::U32(h), Arg::F32(EPS)]);
        // slots 1 .. BLOCK - 1 predict the tokens after the anchor: the main model's head, 16 candidates each, the walk
        let slots = BLOCK - 1;
        let vocab = self.m.cfg.vocab;
        lap("final norm");
        let hid1 = row(&b.hid, HIDDEN, 1);
        match df.draft_ids.as_ref() {
            // the draft vocabulary's rows only, every row decoded once for all the slots
            Some((ids, n)) => {
                let (t_out, _) = self.m.dense_meta("output.weight");
                let head_ty = t_out.ty.name().to_lowercase();
                // SAFETY: n rows named by ids (all < vocab) of the head as stored; hid1 holds `slots` rows of HIDDEN floats,
                // logits slots x n floats (n <= vocab).
                unsafe {
                    self.mvp.launch(&format!("matvec_ids_cols_{head_ty}"), ((*n as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(),
                                    &[Arg::Ptr(self.w["output.weight"].ptr), Arg::Ptr(ids.ptr), Arg::Ptr(hid1), Arg::U32(HIDDEN as u32),
                                      Arg::Ptr(self.bb.logits.ptr), Arg::U32(*n as u32), Arg::U32(*n as u32),
                                      Arg::U32(t_out.dims[0] as u32), Arg::U32(slots as u32)])
                }
                .unwrap_or_else(|err| panic!("dflash head: {err}"));
                lap("head (draft vocabulary)");
                self.k("df_top16", slots as u32, 256, &[f(self.bb.logits.ptr), Arg::U32(*n as u32), Arg::U32(*n as u32), f(ids.ptr),
                                                        f(b.cand.ptr), f(b.unary.ptr)]);
            }
            None => {
                self.mv_cols("output.weight", hid1, HIDDEN, self.bb.logits.ptr, vocab, slots);
                lap("head (output.weight x 7)");
                self.k("df_top16", slots as u32, 256, &[f(self.bb.logits.ptr), Arg::U32(vocab as u32), Arg::U32(vocab as u32), Arg::Ptr(0),
                                                        f(b.cand.ptr), f(b.unary.ptr)]);
            }
        }
        lap("top16");
        self.df_mv("candidate_selector.hidden_projection.weight", hid1, HIDDEN, b.hp.ptr, RANK, slots);
        self.k("df_walk", 1, 512, &[f(b.cand.ptr), f(b.unary.ptr), f(b.hp.ptr), w("candidate_selector.predecessor_codebook"),
                                    w("candidate_selector.successor_codebook"), Arg::U32(next), Arg::U32(slots as u32),
                                    Arg::U32(RANK as u32), f(b.out.ptr)]);
        lap("hidden_projection + walk");
        let mut out = vec![0u8; 2 * slots * 4];
        self.download(b.out.ptr, &mut out);
        lap("download");
        let words: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        // the chain stops after a draft the selector is less sure of than min_p (that draft is kept)
        let mut toks = Vec::with_capacity(k);
        for e in 0..k {
            toks.push(words[e]);
            if f32::from_bits(words[slots + e]) < min_p {
                break;
            }
        }
        if let (Ok(dir), true) = (std::env::var("SHADOW_DF_DUMP"), std::env::var("SHADOW_DF_DUMP_DRAFTS").is_ok_and(|v| v == "1")) {
            // one draft: n, next, then the block's input rows, its final hidden, the candidates (ids, logits), hp, picks
            let rd = |ptr: u64, nf: usize| {
                let mut b = vec![0u8; nf * 4];
                self.download(ptr, &mut b);
                b
            };
            use std::io::Write;
            let d = self.dflash.as_ref().unwrap();
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/drafts.bin")).unwrap();
            f.write_all(&(n as u32).to_le_bytes()).unwrap();
            f.write_all(&next.to_le_bytes()).unwrap();
            for (ptr, nf) in [(d.bufs.emb.ptr, BLOCK * HIDDEN), (d.bufs.hid.ptr, BLOCK * HIDDEN), (d.bufs.cand.ptr, slots * TOP_K),
                              (d.bufs.unary.ptr, slots * TOP_K), (d.bufs.hp.ptr, slots * RANK), (d.bufs.out.ptr, slots)] {
                f.write_all(&rd(ptr, nf)).unwrap();
            }
        }
        let df = self.dflash.as_mut().unwrap();
        df.stats.0 += 1;
        if let Some(p) = df.prof.as_mut() {
            for (name, s) in laps {
                let e = p.entry(name).or_default();
                e.0 += s;
                e.1 += 1;
            }
            if df.stats.0 % 40 == 0 {
                let rounds = df.stats.0 as f64;
                let line: Vec<String> = p.iter().map(|(n, (s, _))| format!("{n}: {:.2} ms", 1e3 * s / rounds)).collect();
                eprintln!("dflash: per draft round after {rounds} rounds: {}", line.join("; "));
            }
        }
        toks
    }
}
