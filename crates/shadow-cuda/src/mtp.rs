//! The Multi-Token Prediction (MTP) draft layer's weights: `models/mtp-basalt-ft1` (`dense.txt` + `dense.bin`, the
//! 512 routed experts in `experts.bin`, the draft head's token subset in `draft_vocab.bin`), ~830 MB of VRAM, and its
//! buffers.  The layer itself runs in `session::mtp_layer`, on the session's kernels.
//!
//! Drafter cell c is computed from the main model's final residual streams at position c and the token at c + 1:
//!   R = fc_hidden(rms(R_c)) + fc_embedding(rms(embed(x_{c+1})))  ->  hc read  ->  gated attention over its own K/V
//!   (q/k norm, RoPE at c)  ->  hc write + read  ->  MoE (512 experts, top 10, gated shared expert)  ->  hc write
//!   ->  final mixer  ->  the main model's head  ->  a guess at x_{c+2}.
//! A chain feeds the layer's own R and its guess back in for the next cell.  The verify pass checks every draft
//! against the main model, so speculation built on it cannot change an answer.  Layout and semantics follow Strata's
//! `src/core/mtp.cpp` (`record_front` / `record_rest`, `prefill`, `draft`).

use crate::driver::{Context, DeviceBuffer};
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

/// The draft layer's device buffers (one row at a time).
pub struct MtpBuffers<'c> {
    /// The layer's residual streams (after a full row: the input of the next chained draft).
    pub r: DeviceBuffer<'c>,
    pub rn: DeviceBuffer<'c>,
    pub down: DeviceBuffer<'c>,
    pub up: DeviceBuffer<'c>,
    pub u: DeviceBuffer<'c>,
    pub inj: DeviceBuffer<'c>,
    pub inj2: DeviceBuffer<'c>,
    pub en: DeviceBuffer<'c>,
    pub e2: DeviceBuffer<'c>,
    pub hn: DeviceBuffer<'c>,
    pub h2: DeviceBuffer<'c>,
    pub qg: DeviceBuffer<'c>,
    pub q: DeviceBuffer<'c>,
    pub gate: DeviceBuffer<'c>,
    pub kk: DeviceBuffer<'c>,
    /// The layer's own K/V, one cell per position of the main model.
    pub kv_k: DeviceBuffer<'c>,
    pub kv_v: DeviceBuffer<'c>,
    pub ao: DeviceBuffer<'c>,
    pub scratch: DeviceBuffer<'c>,
    pub bo: DeviceBuffer<'c>,
    pub sg: DeviceBuffer<'c>,
    pub su: DeviceBuffer<'c>,
    pub sa: DeviceBuffer<'c>,
    pub so: DeviceBuffer<'c>,
    pub rl: DeviceBuffer<'c>,
    pub sgl: DeviceBuffer<'c>,
    /// gate and up of the routed experts (2 x TOP_K rows of EXPERT_FF), their activations, their outputs
    pub eg: DeviceBuffer<'c>,
    pub act: DeviceBuffer<'c>,
    pub eo: DeviceBuffer<'c>,
    pub tab: DeviceBuffer<'c>,
    pub y: DeviceBuffer<'c>,
    pub sample: DeviceBuffer<'c>,
    /// The main model's final residual streams at the last position it read (`MtpModel::pending`).
    pub last_r: DeviceBuffer<'c>,
    /// argmax_subset's output: the draft token (as a float) and its probability
    pub pick: DeviceBuffer<'c>,
}

pub struct MtpModel<'c> {
    pub dense_weights: DeviceBuffer<'c>,
    pub tensors: HashMap<String, DenseTensor>,
    /// The 512 routed experts unpacked to Q2_0, gate | up | down rows each (BLOB_BYTES an expert).
    pub experts: DeviceBuffer<'c>,
    /// The token ids the draft head picks from (draft_vocab.bin), or the whole vocabulary.
    pub draft_vocab: Option<Vec<u32>>,
    /// draft_vocab on the device (for argmax_subset)
    pub draft_ids: Option<DeviceBuffer<'c>>,
    pub bufs: MtpBuffers<'c>,
    pub cap: usize,
    /// Whether `bufs.last_r` holds the residual of the session's last position, whose drafter K/V cell waits for the
    /// token after it.
    pub pending: bool,
    /// Drafts made and drafts the main model agreed with (`SHADOW_MTP_PROBE`, the speculation loop).
    pub stats: (u64, u64),
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
            let u32s: Vec<u32> = vb.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            Some(u32s)
        } else {
            None
        };

        // 5. The layer's buffers
        let (h, cs) = (HIDDEN, STREAMS);
        let a = |n: usize| ctx.alloc(n.max(1) * 4).map_err(e);
        let bufs = MtpBuffers {
            r: a(h * cs)?, rn: a(h * cs)?, down: a(320)?, up: a(h * cs)?, u: a(h)?, inj: a(cs)?, inj2: a(cs)?,
            en: a(h)?, e2: a(h)?, hn: a(h * cs)?, h2: a(h * cs)?, qg: a(N_HEAD * 2 * HEAD_DIM)?,
            q: a(N_HEAD * HEAD_DIM)?, gate: a(N_HEAD * HEAD_DIM)?, kk: a(N_HEAD_KV * HEAD_DIM)?,
            kv_k: a(cap * N_HEAD_KV * HEAD_DIM)?, kv_v: a(cap * N_HEAD_KV * HEAD_DIM)?, ao: a(N_HEAD * HEAD_DIM)?,
            scratch: ctx.alloc(N_HEAD * cap * 8).map_err(e)?, bo: a(h)?, sg: a(EXPERT_FF)?, su: a(EXPERT_FF)?,
            sa: a(EXPERT_FF)?, so: a(h)?, rl: a(N_EXPERTS)?, sgl: a(1)?, eg: a(2 * TOP_K * EXPERT_FF)?,
            act: a(TOP_K * EXPERT_FF)?, eo: a(TOP_K * h)?, tab: a(TOP_K * 7)?, y: a(h)?, sample: a(h)?,
            last_r: a(h * cs)?, pick: a(2)?,
        };
        let draft_ids = match draft_vocab.as_ref() {
            Some(v) => {
                let d = ctx.alloc(v.len() * 4).map_err(e)?;
                d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).map_err(e)?;
                Some(d)
            }
            None => None,
        };
        let _ = vocab_size;
        Ok(Self { dense_weights, tensors, experts, draft_vocab, draft_ids, bufs, cap, pending: false, stats: (0, 0) })
    }

    pub fn tensor_ptr(&self, name: &str) -> Option<u64> {
        self.tensors.get(name).map(|t| self.dense_weights.ptr + t.offset as u64)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_expert_roundtrip() {
        // Create synthetic blob where each segment has recognizable patterns
        let mut blob = vec![0u8; BLOB_BYTES];
        for (i, b) in blob.iter_mut().enumerate() {
            let m = match i {
                0..819_200 => 251,
                819_200..1_228_800 => 241,
                1_228_800..1_331_200 => 239,
                _ => 233,
            };
            *b = (i % m) as u8;
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

