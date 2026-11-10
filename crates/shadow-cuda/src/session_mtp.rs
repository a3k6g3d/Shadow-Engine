//! The MTP draft layer (`crate::mtp`) on the session's kernels: its K/V for every position the main model reads, and
//! drafts.  Cell c comes from the main model's final residual at position c and the token at c + 1; a draft chain
//! starts from the last position read and the token about to be stepped.  Follows Strata's `src/core/mtp.cpp`.

use super::{f32s, GpuSession};
use crate::kernels::Arg;
use crate::mtp::{BLOB_BYTES, EXPERT_FF, HEAD_DIM, N_EXPERTS, N_HEAD, N_HEAD_KV, TOP_K};
use shadow_model::forward::route;
use shadow_model::quant::dequantize;

impl<'c, 'm> GpuSession<'c, 'm> {
    fn mtp_w(&self, name: &str) -> u64 {
        self.mtp.as_ref().unwrap().tensor_ptr(name).unwrap_or_else(|| panic!("mtp: no tensor {name}"))
    }

    /// y = W x for one of the draft layer's dense tensors (Q8_0 through the fast kernel, BF16/F32 the reference one).
    fn mtp_mv(&self, name: &str, x: u64, y: u64) {
        let t = &self.mtp.as_ref().unwrap().tensors[name];
        let entry = match t.kind.as_str() {
            "q8_0" => "matvec_fast_q8_0",
            "bf16" => "matvec_bf16",
            "f32" => "matvec_f32",
            k => panic!("mtp: {name}: no matvec for {k}"),
        };
        // SAFETY: t.rows rows of t.cols weights at the tensor's offset in dense.bin; x holds t.cols floats, y t.rows.
        unsafe {
            self.mvp.launch(entry, ((t.rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(),
                            &[Arg::Ptr(self.mtp_w(name)), Arg::Ptr(x), Arg::Ptr(y), Arg::U32(t.rows as u32),
                              Arg::U32(t.cols as u32)])
        }
        .unwrap_or_else(|err| panic!("mtp {name}: {err}"));
    }

    /// The draft layer's hyper-connection read `p` (`hc_norm`, `input_mix_weight_down` / `_up`, and with `inject` the
    /// block's `block_inject_weight` into it): u from its streams r (`skip_norm`: rn already holds their norm).
    fn mtp_hc_read(&self, p: &str, inject: Option<u64>, skip_norm: bool, u: u64) {
        let c = &self.m.cfg;
        let (h, cs, rank) = (c.hidden, c.streams, c.hc_rank as u32);
        let b = &self.mtp.as_ref().unwrap().bufs;
        if !skip_norm {
            self.k("rmsnorm_groups", cs as u32, 256, &[Arg::Ptr(b.r.ptr), Arg::Ptr(self.mtp_w(&format!("{p}.hc_norm.weight"))),
                                                      Arg::Ptr(b.rn.ptr), Arg::U32(h as u32), Arg::F32(c.eps)]);
        }
        let (w_inj, g, r_inj) = match inject {
            Some(g) => (self.mtp_w(&format!("{p}.block_inject_weight.weight")), g, cs as u32),
            None => (0, 0, 0),
        };
        // SAFETY: rank + r_inj BF16 rows of h * cs weights; rn holds h * cs floats, down rank, g cs.
        unsafe {
            self.mvp.launch("hc_down_inject_bf16", ((rank + r_inj).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(), &[
                Arg::Ptr(self.mtp_w(&format!("{p}.input_mix_weight_down.weight"))), Arg::Ptr(b.down.ptr), Arg::U32(rank),
                Arg::Ptr(w_inj), Arg::Ptr(g), Arg::U32(r_inj), Arg::Ptr(b.rn.ptr), Arg::U32((h * cs) as u32),
                Arg::F32(cs as f32)])
        }
        .unwrap_or_else(|err| panic!("mtp hc read {p}: {err}"));
        self.mtp_mv(&format!("{p}.input_mix_weight_up.weight"), b.down.ptr, b.up.ptr);
        self.hc_mix(b.rn.ptr, b.up.ptr, u);
    }

    /// One row of the draft layer: `rin` (h * streams floats: the main model's residual at position `cell`, or in a
    /// chain the layer's own) and the token after it.  Its K/V go into `cell`; with `full` the rest of the layer runs
    /// and returns its guess at the token after that and the guess's probability over the draft vocabulary (the
    /// layer's residual stays in bufs.r for the next link).
    fn mtp_row(&self, rin: u64, tok: u32, cell: usize, full: bool) -> Option<(u32, f32)> {
        let c = &self.m.cfg;
        let (h, cs) = (c.hidden, c.streams);
        let ch = h * cs;
        let (nh, nkv, hd) = (N_HEAD, N_HEAD_KV, HEAD_DIM);
        let (rd, th, eps) = (c.rope_dims as u32, c.rope_theta as f32, c.eps);
        let mt = self.mtp.as_ref().unwrap();
        let b = &mt.bufs;
        let emb = {
            let (t, bytes) = self.m.dense("token_embd.weight");
            let (qk, qb) = t.ty.block().unwrap();
            let rb = (t.dims[0] / qk * qb) as usize;
            let mut x = vec![0f32; h];
            dequantize(t.ty, &bytes[tok as usize * rb..(tok as usize + 1) * rb], &mut x).unwrap();
            x
        };
        // the two inputs: rms(embed(tok)) through fc_embedding, rms(rin) through fc_hidden per stream; R = their sum
        self.upload_to(b.en.ptr, &f32s(&emb));
        self.k("rmsnorm_groups", 1, 256, &[Arg::Ptr(b.en.ptr), Arg::Ptr(self.mtp_w("pre_fc_norm_embedding.weight")),
                                           Arg::Ptr(b.en.ptr), Arg::U32(h as u32), Arg::F32(eps)]);
        self.mtp_mv("fc_embedding.weight", b.en.ptr, b.e2.ptr);
        self.k("rmsnorm_groups", 1, 256, &[Arg::Ptr(rin), Arg::Ptr(self.mtp_w("pre_fc_norm_hidden.weight")),
                                           Arg::Ptr(b.hn.ptr), Arg::U32(ch as u32), Arg::F32(eps)]);
        for s in 0..cs as u64 {
            self.mtp_mv("fc_hidden.weight", b.hn.ptr + s * h as u64 * 4, b.h2.ptr + s * h as u64 * 4);
        }
        self.k1("broadcast", ch, &[Arg::Ptr(b.e2.ptr), Arg::Ptr(b.r.ptr), Arg::U32(h as u32), Arg::U32(cs as u32)]);
        self.k1("add", ch, &[Arg::Ptr(b.r.ptr), Arg::Ptr(b.h2.ptr), Arg::U32(ch as u32)]);
        // attention: K/V into the cell
        self.mtp_hc_read("attn_hyper_connection", Some(b.inj.ptr), false, b.u.ptr);
        let slot = (cell * nkv * hd * 4) as u64;
        self.mtp_mv("self_attn.k_proj.weight", b.u.ptr, b.kk.ptr);
        self.mtp_mv("self_attn.v_proj.weight", b.u.ptr, b.kv_v.ptr + slot);
        if !full {
            self.k("head_norm_rope", nkv as u32, 128, &[
                Arg::Ptr(b.kk.ptr), Arg::U32(hd as u32), Arg::Ptr(b.kv_k.ptr + slot), Arg::U32(hd as u32),
                Arg::Ptr(self.mtp_w("self_attn.k_norm.weight")), Arg::U32(hd as u32), Arg::F32(eps), Arg::U32(cell as u32),
                Arg::U32(rd), Arg::F32(th)]);
            return None;
        }
        self.mtp_mv("self_attn.q_proj.weight", b.u.ptr, b.qg.ptr);
        self.k("head_norm_rope_qgk", (nh + nkv) as u32, 128, &[
            Arg::Ptr(b.qg.ptr), Arg::Ptr(b.q.ptr), Arg::Ptr(b.gate.ptr), Arg::Ptr(self.mtp_w("self_attn.q_norm.weight")),
            Arg::Ptr(b.kk.ptr), Arg::Ptr(b.kv_k.ptr + slot), Arg::Ptr(self.mtp_w("self_attn.k_norm.weight")),
            Arg::U32(nh as u32), Arg::U32(hd as u32), Arg::F32(eps), Arg::U32(cell as u32), Arg::U32(rd), Arg::F32(th)]);
        self.k("attention_heads", nh as u32, 256, &[
            Arg::Ptr(b.q.ptr), Arg::Ptr(b.kv_k.ptr), Arg::Ptr(b.kv_v.ptr), Arg::Ptr(0), Arg::U32(cell as u32 + 1),
            Arg::Ptr(b.gate.ptr), Arg::Ptr(b.ao.ptr), Arg::Ptr(b.scratch.ptr), Arg::U32(mt.cap as u32),
            Arg::U32(nkv as u32), Arg::U32(hd as u32), Arg::U32((nh / nkv) as u32)]);
        self.mtp_mv("self_attn.o_proj.weight", b.ao.ptr, b.bo.ptr);
        // the MLP hyper-connection, the attention's write folded in
        self.k("hc_write_rmsnorm_groups", cs as u32, 256, &[
            Arg::Ptr(b.r.ptr), Arg::Ptr(b.inj.ptr), Arg::Ptr(b.bo.ptr), Arg::Ptr(self.mtp_w("mlp_hyper_connection.hc_norm.weight")),
            Arg::Ptr(b.rn.ptr), Arg::U32(h as u32), Arg::F32(eps)]);
        self.mtp_hc_read("mlp_hyper_connection", Some(b.inj2.ptr), true, b.u.ptr);
        // the MoE: router and the gated shared expert, the top 10 of its 512 resident experts, the combine
        self.mtp_mv("mlp.gate.weight", b.u.ptr, b.rl.ptr);
        self.mtp_mv("mlp.shared_expert_gate.weight", b.u.ptr, b.sgl.ptr);
        self.mtp_mv("mlp.shared_expert.gate_proj.weight", b.u.ptr, b.sg.ptr);
        self.mtp_mv("mlp.shared_expert.up_proj.weight", b.u.ptr, b.su.ptr);
        self.k1("swiglu", EXPERT_FF, &[Arg::Ptr(b.sg.ptr), Arg::Ptr(b.su.ptr), Arg::Ptr(b.sa.ptr), Arg::U32(EXPERT_FF as u32)]);
        self.mtp_mv("mlp.shared_expert.down_proj.weight", b.sa.ptr, b.so.ptr);
        let logits = self.download_mapped(&[(b.rl.ptr, N_EXPERTS)]).remove(0);
        let top = route(&logits, TOP_K);
        let part = [0u64, (BLOB_BYTES / 3) as u64, (2 * BLOB_BYTES / 3) as u64];
        let mut tab = Vec::with_capacity(TOP_K * 28);
        for p in part {
            for &(e, _) in &top {
                tab.extend_from_slice(&(mt.experts.ptr + (e * BLOB_BYTES) as u64 + p).to_le_bytes());
            }
        }
        for &(_, w) in &top {
            tab.extend_from_slice(&(w as f32).to_le_bytes());
        }
        self.upload_to(b.tab.ptr, &tab);
        let k = TOP_K as u64;
        let batch = |tab_off: u64, count: usize, x: u64, xs: usize, y: u64, ys: usize, rows: usize, cols: usize| {
            // SAFETY: `count` Q2_0 matrices of rows x cols at the table's addresses (inside `experts`); inputs and
            // outputs inside the layer's buffers at the strides given.
            unsafe {
                self.mvp.launch("matvec_batch_q2_0", ((rows as u32).div_ceil(8), count as u32, 1), (256, 1, 1), 0, self.ls(),
                                &[Arg::Ptr(b.tab.ptr + tab_off), Arg::Ptr(x), Arg::U32(xs as u32), Arg::Ptr(y),
                                  Arg::U32(ys as u32), Arg::U32(rows as u32), Arg::U32(cols as u32)])
            }
            .unwrap_or_else(|err| panic!("mtp experts: {err}"));
        };
        batch(0, 2 * TOP_K, b.u.ptr, 0, b.eg.ptr, EXPERT_FF, EXPERT_FF, h);
        self.k1("swiglu", TOP_K * EXPERT_FF, &[Arg::Ptr(b.eg.ptr), Arg::Ptr(b.eg.ptr + k * EXPERT_FF as u64 * 4),
                                               Arg::Ptr(b.act.ptr), Arg::U32((TOP_K * EXPERT_FF) as u32)]);
        batch(16 * k, TOP_K, b.act.ptr, EXPERT_FF, b.eo.ptr, h, h, EXPERT_FF);
        self.k1("moe_combine_qwen", h, &[Arg::Ptr(b.y.ptr), Arg::Ptr(b.eo.ptr), Arg::Ptr(b.tab.ptr + 24 * k),
                                         Arg::Ptr(b.so.ptr), Arg::Ptr(b.sgl.ptr), Arg::U32(h as u32), Arg::U32(TOP_K as u32)]);
        self.hc_write_at(b.r.ptr, b.inj2.ptr, b.y.ptr);
        // the final mixer and the main model's head, over the draft vocabulary
        self.mtp_hc_read("hyper_connection_mixer", None, false, b.sample.ptr);
        // the head over the draft vocabulary's rows only (when its type has the row-index kernel), the pick on the GPU
        // (smallest id on a tie) and its probability there
        let (t_out, _) = self.m.dense_meta("output.weight");
        let head_ty = t_out.ty.name().to_lowercase();
        match (&mt.draft_ids, matches!(head_ty.as_str(), "q6_k" | "q4_k" | "iq4_xs") && self.m.scale("output.weight") == 1.0) {
            (Some(d), true) => {
                let n = mt.draft_vocab.as_ref().map_or(0, |v| v.len());
                // SAFETY: n rows named by d (all < vocab) of the head as stored; sample holds t_out.dims[0] floats, logits n.
                unsafe {
                    self.mvp.launch(&format!("matvec_ids_{head_ty}"), ((n as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, self.ls(),
                                    &[Arg::Ptr(self.w["output.weight"].ptr), Arg::Ptr(d.ptr), Arg::Ptr(b.sample.ptr),
                                      Arg::Ptr(self.b.logits.ptr), Arg::U32(n as u32), Arg::U32(t_out.dims[0] as u32)])
                }
                .unwrap_or_else(|err| panic!("mtp head: {err}"));
                self.k("argmax_compact", 1, 1024, &[Arg::Ptr(self.b.logits.ptr), Arg::Ptr(d.ptr), Arg::U32(n as u32), Arg::Ptr(b.pick.ptr)]);
            }
            _ => {
                self.mv("output.weight", b.sample.ptr, self.b.logits.ptr);
                let (ids, n) = match &mt.draft_ids {
                    Some(d) => (d.ptr, mt.draft_vocab.as_ref().map_or(0, |v| v.len())),
                    None => (0, c.vocab),
                };
                self.k("argmax_subset", 1, 1024, &[Arg::Ptr(self.b.logits.ptr), Arg::Ptr(ids), Arg::U32(n as u32), Arg::Ptr(b.pick.ptr)]);
            }
        }
        let pick = self.download_mapped(&[(b.pick.ptr, 2)]).remove(0);
        Some((pick[0] as u32, pick[1]))
    }

    /// The main model has read positions p0 .. p0 + toks.len() (`toks`), their final residuals in rows (row j at
    /// rows + j * h * streams floats): the drafter's K/V for cells p0 - 1 .. p0 + n - 2, and the last row kept pending.
    pub(super) fn mtp_observe(&mut self, rows: u64, toks: &[u32], p0: usize) {
        let ch = self.m.cfg.hidden * self.m.cfg.streams;
        let (pending, last) = {
            let mt = self.mtp.as_ref().unwrap();
            (mt.pending, mt.bufs.last_r.ptr)
        };
        for (j, &t) in toks.iter().enumerate() {
            if j == 0 {
                if pending && p0 > 0 {
                    self.mtp_row(last, t, p0 - 1, false);
                }
            } else {
                self.mtp_row(rows + ((j - 1) * ch * 4) as u64, t, p0 + j - 1, false);
            }
        }
        self.k1("copy", ch, &[Arg::Ptr(last), Arg::Ptr(rows + ((toks.len() - 1) * ch * 4) as u64), Arg::U32(ch as u32)]);
        self.mtp.as_mut().unwrap().pending = true;
    }

    /// Up to k drafts for the tokens after `next` (the token about to be stepped, which follows the session's last
    /// position): a chain of the draft layer from the main model's last residual, stopped after a draft the layer
    /// gives less than `min_p` (the draft is kept: it is the chain's last).  None without the layer or a position to
    /// start from.  The cells the chain writes past the last position are rewritten when the main model reads those
    /// positions.
    pub fn mtp_draft(&mut self, next: u32, k: usize, min_p: f32) -> Vec<u32> {
        let Some(mt) = self.mtp.as_ref().filter(|m| m.pending) else { return vec![] };
        let (last, r, cap) = (mt.bufs.last_r.ptr, mt.bufs.r.ptr, mt.cap);
        let cell0 = self.tokens.len() - 1;
        let k = k.min(cap.saturating_sub(cell0 + 2));
        let mut out = Vec::with_capacity(k);
        let (mut rin, mut tok) = (last, next);
        for j in 0..k {
            let (d, p) = self.mtp_row(rin, tok, cell0 + j, true).unwrap();
            out.push(d);
            if p < min_p {
                break;
            }
            (rin, tok) = (r, d);
        }
        out
    }
}
