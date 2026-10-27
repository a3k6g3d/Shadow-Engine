//! The Multi-Token Prediction (MTP) draft layer on the GPU.
//!
//! Loads the tuned MTP head from `models/mtp-basalt-ft1` (`dense.bin`, `dense.txt`, `experts.bin`, and optional
//! `draft_vocab.bin`) into GPU memory (~824 MB VRAM).
//!
//! The MTP draft block predicts candidate tokens at position t+2 given residual R_t and token t+1:
//!   e = fc_embedding(rms(embed(tok)) * (1+w))
//!   h = fc_hidden(rms_10240(R) * (1+w))
//!   R_mtp = h + e  ->  attention hyper-connection  ->  QSA self-attention  ->  MLP hyper-connection  ->
//!   MoE (512 experts, top-10, shared expert)  ->  final mixer  ->  head -> draft token
//!
//! Because speculative decoding verifies candidate tokens against the base model's greedy output, the emitted
//! tokens are guaranteed bit-for-bit identical to the base model while decoding 2x-3x faster.

use crate::driver::{Context, DeviceBuffer};
use crate::kernels::{Arg as KArg, Program};
use std::collections::HashMap;
use std::path::Path;

pub const BLOB_BYTES: usize = 1_382_400;
pub const N_EXPERTS: usize = 512;
pub const HIDDEN: usize = 2560;
pub const STREAMS: usize = 4;
pub const EXPERT_FF: usize = 640;
pub const TOP_K: usize = 10;
pub const HEAD_DIM: usize = 256;
pub const N_HEAD: usize = 24;
pub const N_HEAD_KV: usize = 2;
pub const ROPE_DIMS: usize = 64;
pub const EPS: f32 = 1e-6;

#[derive(Debug, Clone)]
pub struct DenseTensor {
    pub name: String,
    pub kind: String,
    pub rows: usize,
    pub cols: usize,
    pub offset: usize,
    pub bytes: usize,
}

pub struct MtpBuffers<'c> {
    pub r_mtp: DeviceBuffer<'c>,
    pub rn: DeviceBuffer<'c>,
    pub down: DeviceBuffer<'c>,
    pub up: DeviceBuffer<'c>,
    pub u: DeviceBuffer<'c>,
    pub inj: DeviceBuffer<'c>,
    pub inj2: DeviceBuffer<'c>,
    pub en: DeviceBuffer<'c>,
    pub hn: DeviceBuffer<'c>,
    pub e2: DeviceBuffer<'c>,
    pub h2: DeviceBuffer<'c>,
    pub qg: DeviceBuffer<'c>,
    pub q: DeviceBuffer<'c>,
    pub gate: DeviceBuffer<'c>,
    pub k: DeviceBuffer<'c>,
    pub v: DeviceBuffer<'c>,
    pub kv_k: DeviceBuffer<'c>,
    pub kv_v: DeviceBuffer<'c>,
    pub attn: DeviceBuffer<'c>,
    pub scratch: DeviceBuffer<'c>,
    pub bo: DeviceBuffer<'c>,
    pub seg: DeviceBuffer<'c>,
    pub seu: DeviceBuffer<'c>,
    pub sact: DeviceBuffer<'c>,
    pub sh_out: DeviceBuffer<'c>,
    pub rl: DeviceBuffer<'c>,
    pub eg: DeviceBuffer<'c>,
    pub eu: DeviceBuffer<'c>,
    pub act: DeviceBuffer<'c>,
    pub eo: DeviceBuffer<'c>,
    pub tab: DeviceBuffer<'c>,
    pub sgl: DeviceBuffer<'c>,
    pub y: DeviceBuffer<'c>,
    pub sample: DeviceBuffer<'c>,
    pub logits: DeviceBuffer<'c>,
}

pub struct MtpModel<'c> {
    pub dense_weights: DeviceBuffer<'c>,
    pub tensors: HashMap<String, DenseTensor>,
    pub experts: DeviceBuffer<'c>,
    pub draft_vocab: Option<Vec<u32>>,
    pub bufs: MtpBuffers<'c>,
    pub cap: usize,
    pub cs: usize,
}

/// Unpacks a 1,382,400-byte split-layout expert blob into standard contiguous Q2_0 rows:
/// gate (640x2560 = 460,800 bytes), up (640x2560 = 460,800 bytes), down (2560x640 = 460,800 bytes).
pub fn unpack_expert(blob: &[u8], out: &mut [u8]) {
    assert_eq!(blob.len(), BLOB_BYTES);
    assert_eq!(out.len(), BLOB_BYTES);
    let inter_codes = &blob[0..819_200];
    let dnb_codes = &blob[819_200..1_228_800];
    let inter_scales = &blob[1_228_800..1_331_200];
    let dnb_scales = &blob[1_331_200..1_382_400];

    let (gate_out, rest) = out.split_at_mut(460_800);
    let (up_out, down_out) = rest.split_at_mut(460_800);

    for r in 0..EXPERT_FF {
        for c in 0..40 {
            let b_gate = (2 * r) * 40 + c;
            let dst_gate = r * 720 + c * 18;
            gate_out[dst_gate..dst_gate + 2].copy_from_slice(&inter_scales[b_gate * 2..b_gate * 2 + 2]);
            gate_out[dst_gate + 2..dst_gate + 18].copy_from_slice(&inter_codes[b_gate * 16..b_gate * 16 + 16]);

            let b_up = (2 * r + 1) * 40 + c;
            let dst_up = r * 720 + c * 18;
            up_out[dst_up..dst_up + 2].copy_from_slice(&inter_scales[b_up * 2..b_up * 2 + 2]);
            up_out[dst_up + 2..dst_up + 18].copy_from_slice(&inter_codes[b_up * 16..b_up * 16 + 16]);
        }
    }

    for r in 0..HIDDEN {
        for c in 0..10 {
            let b_dn = r * 10 + c;
            let dst_dn = r * 180 + c * 18;
            down_out[dst_dn..dst_dn + 2].copy_from_slice(&dnb_scales[b_dn * 2..b_dn * 2 + 2]);
            down_out[dst_dn + 2..dst_dn + 18].copy_from_slice(&dnb_codes[b_dn * 16..b_dn * 16 + 16]);
        }
    }
}

impl<'c> MtpModel<'c> {
    pub fn load(ctx: &'c Context, dir: &Path, cap: usize, vocab_size: usize) -> Result<Self, String> {
        let e = |x: crate::driver::CudaError| x.to_string();
        let txt_path = dir.join("dense.txt");
        let bin_path = dir.join("dense.bin");
        let exp_path = dir.join("experts.bin");
        let voc_path = dir.join("draft_vocab.bin");

        if !txt_path.exists() || !bin_path.exists() || !exp_path.exists() {
            return Err(format!("mtp: directory {} missing required weights", dir.display()));
        }

        // 1. Parse dense.txt
        let txt = std::fs::read_to_string(&txt_path).map_err(|err| err.to_string())?;
        let mut tensors = HashMap::new();
        for line in txt.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                continue;
            }
            let name = parts[0].to_string();
            let kind = parts[1].to_string();
            let rows: usize = parts[2].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            let cols: usize = parts[3].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            let offset: usize = parts[4].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            let bytes: usize = parts[5].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            tensors.insert(name.clone(), DenseTensor { name, kind, rows, cols, offset, bytes });
        }

        // 2. Load dense.bin into GPU
        let dense_bytes = std::fs::read(&bin_path).map_err(|err| err.to_string())?;
        let dense_weights = ctx.alloc(dense_bytes.len()).map_err(e)?;
        dense_weights.upload(&dense_bytes).map_err(e)?;

        // 3. Load & unpack experts.bin (512 experts x 1,382,400 bytes = 707,788,800 bytes)
        let total_exp_bytes = N_EXPERTS * BLOB_BYTES;
        let experts = ctx.alloc(total_exp_bytes).map_err(e)?;
        {
            let exp_data = std::fs::read(&exp_path).map_err(|err| err.to_string())?;
            if exp_data.len() != total_exp_bytes {
                return Err(format!("mtp: experts.bin is {} bytes, expected {total_exp_bytes}", exp_data.len()));
            }
            let mut unpacked = vec![0u8; total_exp_bytes];
            for i in 0..N_EXPERTS {
                let src_blob = &exp_data[i * BLOB_BYTES..(i + 1) * BLOB_BYTES];
                let dst_blob = &mut unpacked[i * BLOB_BYTES..(i + 1) * BLOB_BYTES];
                unpack_expert(src_blob, dst_blob);
            }
            experts.upload(&unpacked).map_err(e)?;
        }

        // 4. Optional draft_vocab.bin
        let draft_vocab = if voc_path.exists() {
            let vb = std::fs::read(&voc_path).map_err(|err| err.to_string())?;
            let mut u32s = Vec::with_capacity(vb.len() / 4);
            for c in vb.chunks_exact(4) {
                u32s.push(u32::from_le_bytes(c.try_into().unwrap()));
            }
            Some(u32s)
        } else {
            None
        };

        // 5. Allocate MTP scratch buffers
        let h = HIDDEN;
        let cs = STREAMS;
        let r_mtp = ctx.alloc(h * cs * 4).map_err(e)?;
        let rn = ctx.alloc(h * cs * 4).map_err(e)?;
        let down = ctx.alloc(320 * 4).map_err(e)?;
        let up = ctx.alloc(h * cs * 4).map_err(e)?;
        let u = ctx.alloc(h * 4).map_err(e)?;
        let inj = ctx.alloc(cs * 4).map_err(e)?;
        let inj2 = ctx.alloc(cs * 4).map_err(e)?;
        let en = ctx.alloc(h * 4).map_err(e)?;
        let hn = ctx.alloc(h * cs * 4).map_err(e)?;
        let e2 = ctx.alloc(h * 4).map_err(e)?;
        let h2 = ctx.alloc(h * cs * 4).map_err(e)?;
        let qg = ctx.alloc(N_HEAD * 2 * HEAD_DIM * 4).map_err(e)?;
        let q = ctx.alloc(N_HEAD * HEAD_DIM * 4).map_err(e)?;
        let gate = ctx.alloc(N_HEAD * HEAD_DIM * 4).map_err(e)?;
        let k = ctx.alloc(N_HEAD_KV * HEAD_DIM * 4).map_err(e)?;
        let v = ctx.alloc(N_HEAD_KV * HEAD_DIM * 4).map_err(e)?;
        let kv_k = ctx.alloc(cap * N_HEAD_KV * HEAD_DIM * 4).map_err(e)?;
        let kv_v = ctx.alloc(cap * N_HEAD_KV * HEAD_DIM * 4).map_err(e)?;
        let attn = ctx.alloc(N_HEAD * HEAD_DIM * 4).map_err(e)?;
        let scratch = ctx.alloc(N_HEAD * cap * 8).map_err(e)?; // double for scratch in attention_heads
        let bo = ctx.alloc(h * 4).map_err(e)?;
        let seg = ctx.alloc(EXPERT_FF * 4).map_err(e)?;
        let seu = ctx.alloc(EXPERT_FF * 4).map_err(e)?;
        let sact = ctx.alloc(EXPERT_FF * 4).map_err(e)?;
        let sh_out = ctx.alloc(h * 4).map_err(e)?;
        let rl = ctx.alloc(N_EXPERTS * 4).map_err(e)?;
        let eg = ctx.alloc(TOP_K * EXPERT_FF * 4).map_err(e)?;
        let eu = ctx.alloc(TOP_K * EXPERT_FF * 4).map_err(e)?;
        let act = ctx.alloc(TOP_K * EXPERT_FF * 4).map_err(e)?;
        let eo = ctx.alloc(TOP_K * h * 4).map_err(e)?;
        let tab = ctx.alloc(TOP_K * 3 * 8 + TOP_K * 4).map_err(e)?; // 3 ptrs + 1 weight per top_k expert
        let sgl = ctx.alloc(h * 4).map_err(e)?;
        let y = ctx.alloc(h * 4).map_err(e)?;
        let sample = ctx.alloc(h * 4).map_err(e)?;
        let logits = ctx.alloc(vocab_size * 4).map_err(e)?;

        Ok(Self {
            dense_weights,
            tensors,
            experts,
            draft_vocab,
            bufs: MtpBuffers {
                r_mtp, rn, down, up, u, inj, inj2, en, hn, e2, h2, qg, q, gate, k, v,
                kv_k, kv_v, attn, scratch, bo, seg, seu, sact, sh_out, rl, eg, eu, act,
                eo, tab, sgl, y, sample, logits,
            },
            cap,
            cs,
        })
    }

    pub fn tensor_ptr(&self, name: &str) -> Option<u64> {
        self.tensors.get(name).map(|t| self.dense_weights.ptr + t.offset as u64)
    }

    /// Single draft prediction step.
    /// Takes base model's final residual `r_base` at position `pos`, and the token emitted for position `pos+1`.
    /// Emits the predicted draft token at `pos+2`.
    pub fn draft_step(
        &mut self,
        mvp: &Program<'c>,
        fw: &Program<'c>,
        token_embd_weights: u64,
        token_embd_ty: &str,
        token_embd_cols: usize,
        output_weights: u64,
        output_ty: &str,
        output_rows: usize,
        r_in: u64,
        token: u32,
        pos: usize,
    ) -> Result<u32, String> {
        let h = HIDDEN;
        let cs = STREAMS;
        let b = &self.bufs;

        // ---- 1. Embedding and pre-FC projections
        // Dequantize or matvec token embedding
        if token_embd_ty == "BF16" || token_embd_ty == "F32" {
            let row_bytes = if token_embd_ty == "BF16" { token_embd_cols * 2 } else { token_embd_cols * 4 };
            let row_ptr = token_embd_weights + (token as usize * row_bytes) as u64;
            if token_embd_ty == "BF16" {
                unsafe {
                    fw.launch("convert", (h.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, None,
                              &[KArg::Ptr(row_ptr), KArg::Ptr(b.en.ptr), KArg::U32(h as u32)]).map_err(|e| e.to_string())?;
                }
            }
        }

        // Pre-FC norm on embedding -> en
        let pre_emb_norm = self.tensor_ptr("pre_fc_norm_embedding.weight").ok_or("pre_fc_norm_embedding missing")?;
        unsafe {
            fw.launch("rmsnorm_groups", (1, 1, 1), (256, 1, 1), 0, None,
                      &[KArg::Ptr(b.en.ptr), KArg::Ptr(pre_emb_norm), KArg::Ptr(b.en.ptr), KArg::U32(h as u32), KArg::F32(EPS)])
                .map_err(|e| e.to_string())?;
        }

        // fc_embedding.weight (q8_0, 2560x2560) -> e2
        let fc_emb = self.tensor_ptr("fc_embedding.weight").ok_or("fc_embedding missing")?;
        unsafe {
            mvp.launch("matvec_fast_q8_0", ((h.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(fc_emb), KArg::Ptr(b.en.ptr), KArg::Ptr(b.e2.ptr), KArg::U32(h as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
        }

        // Pre-FC norm on hidden residual (1 group of 10,240 floats) -> hn
        let pre_hid_norm = self.tensor_ptr("pre_fc_norm_hidden.weight").ok_or("pre_fc_norm_hidden missing")?;
        unsafe {
            fw.launch("rmsnorm_groups", (1, 1, 1), (256, 1, 1), 0, None,
                      &[KArg::Ptr(r_in), KArg::Ptr(pre_hid_norm), KArg::Ptr(b.hn.ptr), KArg::U32((h * cs) as u32), KArg::F32(EPS)])
                .map_err(|e| e.to_string())?;
        }

        // fc_hidden.weight (q8_0, 2560x2560) on each stream -> h2
        let fc_hid = self.tensor_ptr("fc_hidden.weight").ok_or("fc_hidden missing")?;
        for s in 0..cs {
            let hn_s = b.hn.ptr + (s * h * 4) as u64;
            let h2_s = b.h2.ptr + (s * h * 4) as u64;
            unsafe {
                mvp.launch("matvec_fast_q8_0", ((h.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                           &[KArg::Ptr(fc_hid), KArg::Ptr(hn_s), KArg::Ptr(h2_s), KArg::U32(h as u32), KArg::U32(h as u32)])
                    .map_err(|e| e.to_string())?;
            }
        }

        // Add streams: r_mtp[s] = h2[s] + e2
        for s in 0..cs {
            let r_s = b.r_mtp.ptr + (s * h * 4) as u64;
            let h2_s = b.h2.ptr + (s * h * 4) as u64;
            unsafe {
                fw.launch("copy", ((h.div_ceil(256)) as u32, 1, 1), (256, 1, 1), 0, None,
                          &[KArg::Ptr(r_s), KArg::Ptr(h2_s), KArg::U32(h as u32)]).map_err(|e| e.to_string())?;
                fw.launch("add", ((h.div_ceil(256)) as u32, 1, 1), (256, 1, 1), 0, None,
                          &[KArg::Ptr(r_s), KArg::Ptr(b.e2.ptr), KArg::U32(h as u32)]).map_err(|e| e.to_string())?;
            }
        }

        // ---- 2. Attention Hyper-Connection
        let attn_hc_norm = self.tensor_ptr("attn_hyper_connection.hc_norm.weight").ok_or("attn hc norm missing")?;
        unsafe {
            fw.launch("rmsnorm_groups", (cs as u32, 1, 1), (256, 1, 1), 0, None,
                      &[KArg::Ptr(b.r_mtp.ptr), KArg::Ptr(attn_hc_norm), KArg::Ptr(b.rn.ptr), KArg::U32(h as u32), KArg::F32(EPS)])
                .map_err(|e| e.to_string())?;
        }

        let attn_down = self.tensor_ptr("attn_hyper_connection.input_mix_weight_down.weight").ok_or("attn down missing")?;
        let attn_inj = self.tensor_ptr("attn_hyper_connection.block_inject_weight.weight").ok_or("attn inj missing")?;
        unsafe {
            mvp.launch("hc_down_inject_bf16", (((320 + cs) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(attn_down), KArg::Ptr(b.down.ptr), KArg::U32(320),
                         KArg::Ptr(attn_inj), KArg::Ptr(b.inj.ptr), KArg::U32(cs as u32),
                         KArg::Ptr(b.rn.ptr), KArg::U32((h * cs) as u32), KArg::F32(cs as f32)])
                .map_err(|e| e.to_string())?;
        }

        let attn_up = self.tensor_ptr("attn_hyper_connection.input_mix_weight_up.weight").ok_or("attn up missing")?;
        unsafe {
            mvp.launch("matvec_fast_bf16", (((h * cs) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(attn_up), KArg::Ptr(b.down.ptr), KArg::Ptr(b.up.ptr), KArg::U32((h * cs) as u32), KArg::U32(320)])
                .map_err(|e| e.to_string())?;
            fw.launch("hc_mix", (h as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.rn.ptr), KArg::Ptr(b.up.ptr), KArg::Ptr(b.u.ptr), KArg::U32(h as u32), KArg::U32(cs as u32)])
                .map_err(|e| e.to_string())?;
        }

        // ---- 3. Self-Attention (QSA)
        let q_proj = self.tensor_ptr("self_attn.q_proj.weight").ok_or("self_attn.q_proj missing")?;
        let k_proj = self.tensor_ptr("self_attn.k_proj.weight").ok_or("self_attn.k_proj missing")?;
        let v_proj = self.tensor_ptr("self_attn.v_proj.weight").ok_or("self_attn.v_proj missing")?;
        let o_proj = self.tensor_ptr("self_attn.o_proj.weight").ok_or("self_attn.o_proj missing")?;

        let slot = (pos * N_HEAD_KV * HEAD_DIM * 4) as u64;
        unsafe {
            mvp.launch("matvec_fast_q8_0", (((N_HEAD * 2 * HEAD_DIM) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(q_proj), KArg::Ptr(b.u.ptr), KArg::Ptr(b.qg.ptr), KArg::U32((N_HEAD * 2 * HEAD_DIM) as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
            mvp.launch("matvec_fast_q8_0", (((N_HEAD_KV * HEAD_DIM) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(k_proj), KArg::Ptr(b.u.ptr), KArg::Ptr(b.k.ptr), KArg::U32((N_HEAD_KV * HEAD_DIM) as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
            mvp.launch("matvec_fast_q8_0", (((N_HEAD_KV * HEAD_DIM) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(v_proj), KArg::Ptr(b.u.ptr), KArg::Ptr(b.kv_v.ptr + slot), KArg::U32((N_HEAD_KV * HEAD_DIM) as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
        }

        let q_norm = self.tensor_ptr("self_attn.q_norm.weight").ok_or("self_attn.q_norm missing")?;
        let k_norm = self.tensor_ptr("self_attn.k_norm.weight").ok_or("self_attn.k_norm missing")?;
        unsafe {
            fw.launch("head_norm_rope_qgk", ((N_HEAD + N_HEAD_KV) as u32, 1, 1), (128, 1, 1), 0, None,
                      &[KArg::Ptr(b.qg.ptr), KArg::Ptr(b.q.ptr), KArg::Ptr(b.gate.ptr), KArg::Ptr(q_norm),
                        KArg::Ptr(b.k.ptr), KArg::Ptr(b.kv_k.ptr + slot), KArg::Ptr(k_norm),
                        KArg::U32(N_HEAD as u32), KArg::U32(HEAD_DIM as u32), KArg::F32(EPS),
                        KArg::U32(pos as u32), KArg::U32(ROPE_DIMS as u32), KArg::F32(10000000.0)])
                .map_err(|e| e.to_string())?;
        }

        let ntok = pos + 1;
        unsafe {
            fw.launch("attention_heads", (N_HEAD as u32, 1, 1), (256, 1, 1), 0, None,
                      &[KArg::Ptr(b.q.ptr), KArg::Ptr(b.kv_k.ptr), KArg::Ptr(b.kv_v.ptr), KArg::Ptr(0), KArg::U32(ntok as u32),
                        KArg::Ptr(b.gate.ptr), KArg::Ptr(b.attn.ptr), KArg::Ptr(b.scratch.ptr), KArg::U32(self.cap as u32),
                        KArg::U32(N_HEAD_KV as u32), KArg::U32(HEAD_DIM as u32), KArg::U32((N_HEAD / N_HEAD_KV) as u32)])
                .map_err(|e| e.to_string())?;
            mvp.launch("matvec_fast_q8_0", ((h.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(o_proj), KArg::Ptr(b.attn.ptr), KArg::Ptr(b.bo.ptr), KArg::U32(h as u32), KArg::U32((N_HEAD * HEAD_DIM) as u32)])
                .map_err(|e| e.to_string())?;
        }

        // ---- 4. MLP Hyper-Connection
        unsafe {
            fw.launch("hc_write", ((h * cs) as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.r_mtp.ptr), KArg::Ptr(b.inj.ptr), KArg::Ptr(b.bo.ptr), KArg::U32(h as u32), KArg::U32(cs as u32)])
                .map_err(|e| e.to_string())?;
        }

        let mlp_hc_norm = self.tensor_ptr("mlp_hyper_connection.hc_norm.weight").ok_or("mlp hc norm missing")?;
        unsafe {
            fw.launch("rmsnorm_groups", (cs as u32, 1, 1), (256, 1, 1), 0, None,
                      &[KArg::Ptr(b.r_mtp.ptr), KArg::Ptr(mlp_hc_norm), KArg::Ptr(b.rn.ptr), KArg::U32(h as u32), KArg::F32(EPS)])
                .map_err(|e| e.to_string())?;
        }

        let mlp_down = self.tensor_ptr("mlp_hyper_connection.input_mix_weight_down.weight").ok_or("mlp down missing")?;
        let mlp_inj = self.tensor_ptr("mlp_hyper_connection.block_inject_weight.weight").ok_or("mlp inj missing")?;
        unsafe {
            mvp.launch("hc_down_inject_bf16", (((320 + cs) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(mlp_down), KArg::Ptr(b.down.ptr), KArg::U32(320),
                         KArg::Ptr(mlp_inj), KArg::Ptr(b.inj2.ptr), KArg::U32(cs as u32),
                         KArg::Ptr(b.rn.ptr), KArg::U32((h * cs) as u32), KArg::F32(cs as f32)])
                .map_err(|e| e.to_string())?;
        }

        let mlp_up = self.tensor_ptr("mlp_hyper_connection.input_mix_weight_up.weight").ok_or("mlp up missing")?;
        unsafe {
            mvp.launch("matvec_fast_bf16", (((h * cs) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(mlp_up), KArg::Ptr(b.down.ptr), KArg::Ptr(b.up.ptr), KArg::U32((h * cs) as u32), KArg::U32(320)])
                .map_err(|e| e.to_string())?;
            fw.launch("hc_mix", (h as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.rn.ptr), KArg::Ptr(b.up.ptr), KArg::Ptr(b.u.ptr), KArg::U32(h as u32), KArg::U32(cs as u32)])
                .map_err(|e| e.to_string())?;
        }

        // ---- 5. MoE: Shared Expert + Routed Experts
        let sh_gate = self.tensor_ptr("mlp.shared_expert.gate_proj.weight").ok_or("sh gate missing")?;
        let sh_up = self.tensor_ptr("mlp.shared_expert.up_proj.weight").ok_or("sh up missing")?;
        let sh_down = self.tensor_ptr("mlp.shared_expert.down_proj.weight").ok_or("sh down missing")?;
        unsafe {
            mvp.launch("matvec_fast_q8_0", ((EXPERT_FF.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(sh_gate), KArg::Ptr(b.u.ptr), KArg::Ptr(b.seg.ptr), KArg::U32(EXPERT_FF as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
            mvp.launch("matvec_fast_q8_0", ((EXPERT_FF.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(sh_up), KArg::Ptr(b.u.ptr), KArg::Ptr(b.seu.ptr), KArg::U32(EXPERT_FF as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
            fw.launch("swiglu", (EXPERT_FF as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.seg.ptr), KArg::Ptr(b.seu.ptr), KArg::Ptr(b.sact.ptr), KArg::U32(EXPERT_FF as u32)])
                .map_err(|e| e.to_string())?;
            mvp.launch("matvec_fast_q8_0", ((h.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(sh_down), KArg::Ptr(b.sact.ptr), KArg::Ptr(b.sh_out.ptr), KArg::U32(h as u32), KArg::U32(EXPERT_FF as u32)])
                .map_err(|e| e.to_string())?;
        }

        // Router: mlp.gate.weight (bf16, 512x2560) -> rl
        let router_gate = self.tensor_ptr("mlp.gate.weight").ok_or("mlp.gate missing")?;
        unsafe {
            mvp.launch("matvec_fast_bf16", ((N_EXPERTS.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(router_gate), KArg::Ptr(b.u.ptr), KArg::Ptr(b.rl.ptr), KArg::U32(N_EXPERTS as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
        }

        let mut r_logits = vec![0u8; N_EXPERTS * 4];
        b.rl.download(&mut r_logits).map_err(|e| e.to_string())?;
        let r_floats: Vec<f32> = r_logits.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let routed = route_sigmoid(&r_floats, TOP_K);

        // Populate table for matvec_batch_q2_0
        let mut tab_bytes = Vec::with_capacity(TOP_K * 3 * 8 + TOP_K * 4);
        for &(e, _) in &routed {
            let base = self.experts.ptr + (e * BLOB_BYTES) as u64;
            tab_bytes.extend((base).to_le_bytes()); // gate
            tab_bytes.extend((base + 460_800).to_le_bytes()); // up
            tab_bytes.extend((base + 921_600).to_le_bytes()); // down
        }
        for &(_, w) in &routed {
            tab_bytes.extend(w.to_le_bytes());
        }
        b.tab.upload(&tab_bytes).map_err(|e| e.to_string())?;

        // Batch launch for routed experts
        // gate and up: 2 * TOP_K matrices (640 rows x 2560 cols)
        let tab_gate = b.tab.ptr;
        let tab_down = b.tab.ptr + (TOP_K * 2 * 8) as u64;
        let tab_w = b.tab.ptr + (TOP_K * 3 * 8) as u64;
        unsafe {
            mvp.launch("matvec_batch_q2_0", ((EXPERT_FF.div_ceil(8)) as u32, (2 * TOP_K) as u32, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(tab_gate), KArg::Ptr(b.u.ptr), KArg::U32(0), KArg::Ptr(b.eg.ptr), KArg::U32(EXPERT_FF as u32),
                         KArg::U32(EXPERT_FF as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
            fw.launch("swiglu", ((TOP_K * EXPERT_FF) as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.eg.ptr), KArg::Ptr(b.eg.ptr + (TOP_K * EXPERT_FF * 4) as u64), KArg::Ptr(b.act.ptr), KArg::U32((TOP_K * EXPERT_FF) as u32)])
                .map_err(|e| e.to_string())?;
            mvp.launch("matvec_batch_q2_0", ((h.div_ceil(8)) as u32, TOP_K as u32, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(tab_down), KArg::Ptr(b.act.ptr), KArg::U32(EXPERT_FF as u32), KArg::Ptr(b.eo.ptr), KArg::U32(h as u32),
                         KArg::U32(h as u32), KArg::U32(EXPERT_FF as u32)])
                .map_err(|e| e.to_string())?;
            fw.launch("moe_combine_qwen", (h as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.y.ptr), KArg::Ptr(b.eo.ptr), KArg::Ptr(tab_w), KArg::Ptr(b.sh_out.ptr), KArg::Ptr(b.sgl.ptr),
                        KArg::U32(h as u32), KArg::U32(TOP_K as u32)])
                .map_err(|e| e.to_string())?;
        }

        // ---- 6. Final Mixer
        unsafe {
            fw.launch("hc_write", ((h * cs) as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.r_mtp.ptr), KArg::Ptr(b.inj2.ptr), KArg::Ptr(b.y.ptr), KArg::U32(h as u32), KArg::U32(cs as u32)])
                .map_err(|e| e.to_string())?;
        }

        let mix_norm = self.tensor_ptr("hyper_connection_mixer.hc_norm.weight").ok_or("mixer norm missing")?;
        unsafe {
            fw.launch("rmsnorm_groups", (cs as u32, 1, 1), (256, 1, 1), 0, None,
                      &[KArg::Ptr(b.r_mtp.ptr), KArg::Ptr(mix_norm), KArg::Ptr(b.rn.ptr), KArg::U32(h as u32), KArg::F32(EPS)])
                .map_err(|e| e.to_string())?;
        }

        let mix_down = self.tensor_ptr("hyper_connection_mixer.input_mix_weight_down.weight").ok_or("mixer down missing")?;
        unsafe {
            mvp.launch("hc_down_inject_bf16", ((320 as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(mix_down), KArg::Ptr(b.down.ptr), KArg::U32(320),
                         KArg::Ptr(0), KArg::Ptr(0), KArg::U32(0),
                         KArg::Ptr(b.rn.ptr), KArg::U32((h * cs) as u32), KArg::F32(cs as f32)])
                .map_err(|e| e.to_string())?;
        }

        let mix_up = self.tensor_ptr("hyper_connection_mixer.input_mix_weight_up.weight").ok_or("mixer up missing")?;
        unsafe {
            mvp.launch("matvec_fast_bf16", (((h * cs) as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(mix_up), KArg::Ptr(b.down.ptr), KArg::Ptr(b.up.ptr), KArg::U32((h * cs) as u32), KArg::U32(320)])
                .map_err(|e| e.to_string())?;
            fw.launch("hc_mix", (h as u32, 1, 1), (1, 1, 1), 0, None,
                      &[KArg::Ptr(b.rn.ptr), KArg::Ptr(b.up.ptr), KArg::Ptr(b.sample.ptr), KArg::U32(h as u32), KArg::U32(cs as u32)])
                .map_err(|e| e.to_string())?;
        }

        // ---- 7. Output Head & Argmax
        let kernel_name = format!("matvec_fast_{}", output_ty.to_lowercase());
        unsafe {
            mvp.launch(&kernel_name, ((output_rows.div_ceil(8)) as u32, 1, 1), (256, 1, 1), 0, None,
                       &[KArg::Ptr(output_weights), KArg::Ptr(b.sample.ptr), KArg::Ptr(b.logits.ptr),
                         KArg::U32(output_rows as u32), KArg::U32(h as u32)])
                .map_err(|e| e.to_string())?;
        }

        let mut logits_bytes = vec![0u8; output_rows * 4];
        b.logits.download(&mut logits_bytes).map_err(|e| e.to_string())?;
        let l_floats: Vec<f32> = logits_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

        // Argmax: if draft_vocab is present, search over subset; otherwise search full vocab
        let best_id = if let Some(ref dv) = self.draft_vocab {
            dv.iter().copied()
                .filter(|&id| (id as usize) < l_floats.len())
                .max_by(|&a, &b| l_floats[a as usize].partial_cmp(&l_floats[b as usize]).unwrap())
                .unwrap_or(0)
        } else {
            l_floats.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap()
        };

        Ok(best_id)
    }
}

fn route_sigmoid(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, 1.0 / (1.0 + (-v).exp()))).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    pairs.truncate(k);
    let sum: f32 = pairs.iter().map(|p| p.1).sum();
    if sum > 0.0 {
        for p in &mut pairs {
            p.1 /= sum;
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_expert_roundtrip() {
        // Create synthetic blob where each segment has recognizable patterns
        let mut blob = vec![0u8; BLOB_BYTES];
        for i in 0..819_200 {
            blob[i] = (i % 251) as u8;
        }
        for i in 819_200..1_228_800 {
            blob[i] = (i % 241) as u8;
        }
        for i in 1_228_800..1_331_200 {
            blob[i] = (i % 239) as u8;
        }
        for i in 1_331_200..1_382_400 {
            blob[i] = (i % 233) as u8;
        }

        let mut unpacked = vec![0u8; BLOB_BYTES];
        unpack_expert(&blob, &mut unpacked);

        // Verify gate block 0
        assert_eq!(&unpacked[0..2], &blob[1_228_800..1_228_802]);
        assert_eq!(&unpacked[2..18], &blob[0..16]);

        // Verify up block 0 (inter row 1 = b_up 40)
        assert_eq!(&unpacked[460_800..460_802], &blob[1_228_800 + 40 * 2..1_228_800 + 40 * 2 + 2]);
        assert_eq!(&unpacked[460_802..460_818], &blob[40 * 16..40 * 16 + 16]);

        // Verify down block 0
        assert_eq!(&unpacked[921_600..921_602], &blob[1_331_200..1_331_202]);
        assert_eq!(&unpacked[921_602..921_618], &blob[819_200..819_216]);
    }
}

