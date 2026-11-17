//! The routed experts' rows on AVX-512 (BW + VBMI): int8 activations, integer dot products, one float multiply per
//! block - Strata's `src/kernels/cpu/iq_avx512.cpp` (IQ2_S, IQ3_XXS, IQ3_S) ported, plus IQ4_XS, IQ4_NL and Q2_0.
//! The activations are quantized (`Q8`), so these sums are not `cpu_expert::row_dot`'s: an approximation, used only
//! where the hybrid MoE runs (`--cpu-experts`).
//!
//! Grids give 8 (IQ2_S) or 4 (IQ3) unsigned magnitudes per index; signs go onto the activations (a masked subtract),
//! `maddubs` multiplies u8 magnitudes by i8 activations into i16 pairs, `madd` applies the sub-block scales into i32.
//! The 4-bit formats look their values up (`shuffle_epi8`), and multiply |w| by the activation carrying w's sign.

#![allow(clippy::missing_safety_doc)]

use crate::half::f16_to_f32;
use crate::quant_tables::{IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES, KSIGNS};
use std::arch::x86_64::*;

/// Activations as int8: per block of `blk` values a scale `d` (amax / 127) and `qs = round(x / d)`; with `sums` the
/// sum of each block's qs (for Q2_0's offset).
pub struct Q8 {
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub sums: Vec<i32>,
    pub blk: usize,
}

pub fn quantize(x: &[f32], blk: usize) -> Q8 {
    assert!(x.len().is_multiple_of(blk));
    let mut q = Q8 { qs: vec![0; x.len()], d: Vec::with_capacity(x.len() / blk), sums: Vec::with_capacity(x.len() / blk), blk };
    for (b, xb) in x.chunks_exact(blk).enumerate() {
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut s = 0i32;
        for (i, &v) in xb.iter().enumerate() {
            let qi = (v * id).round().clamp(-127.0, 127.0) as i8;
            q.qs[b * blk + i] = qi;
            s += qi as i32;
        }
        q.d.push(d);
        q.sums.push(s);
    }
    q
}

/// Whether this CPU runs these kernels.
pub fn available() -> bool {
    use std::sync::OnceLock;
    static YES: OnceLock<bool> = OnceLock::new();
    *YES.get_or_init(|| {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vbmi")
    })
}

/// The block size `quantize` must use for an input of `ty`.
pub fn act_block(ty: &str) -> Option<usize> {
    match ty {
        "IQ2_S" | "IQ3_XXS" | "IQ3_S" | "IQ4_XS" => Some(256),
        "IQ4_NL" | "Q2_0" => Some(32),
        _ => None,
    }
}

fn h(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// int16 lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn scales4(s0: i32, s1: i32, s2: i32, s3: i32) -> __m512i {
    let idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    let packed = (s0 as u16 as u64) | ((s1 as u16 as u64) << 16) | ((s2 as u16 as u64) << 32) | ((s3 as u16 as u64) << 48);
    _mm512_permutexvar_epi16(idx, _mm512_set1_epi64(packed as i64))
}

#[inline]
fn ks(v: u32) -> u64 {
    KSIGNS[(v & 127) as usize] as u64
}

#[inline]
fn signs4(a: u32) -> u64 {
    ks(a) | (ks(a >> 7) << 8) | (ks(a >> 14) << 16) | (ks(a >> 21) << 24)
}

/// 64 values (chunk j of a 256-value block): grid magnitudes, sign mask, int16 scales.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn decode_grid(ty: u8, b: &[u8], j: usize) -> (__m512i, u64, __m512i) {
    let g2 = |i: usize| i64::from_le_bytes(IQ2S_GRID[i]);
    let g3x = |i: usize| u32::from_le_bytes(IQ3XXS_GRID[i]) as i32;
    let g3s = |i: usize| u32::from_le_bytes(IQ3S_GRID[i]) as i32;
    match ty {
        22 => {
            // IQ2_S: d, qs[32] grid bytes, signs[32], qh[8], scales[8]
            let qs = &b[2 + 8 * j..];
            let (h0, h1) = (b[66 + 2 * j] as usize, b[66 + 2 * j + 1] as usize);
            let g = _mm512_set_epi64(g2(qs[7] as usize | ((h1 << 2) & 0x300)), g2(qs[6] as usize | ((h1 << 4) & 0x300)),
                                     g2(qs[5] as usize | ((h1 << 6) & 0x300)), g2(qs[4] as usize | ((h1 << 8) & 0x300)),
                                     g2(qs[3] as usize | ((h0 << 2) & 0x300)), g2(qs[2] as usize | ((h0 << 4) & 0x300)),
                                     g2(qs[1] as usize | ((h0 << 6) & 0x300)), g2(qs[0] as usize | ((h0 << 8) & 0x300)));
            let m = u64::from_le_bytes(b[34 + 8 * j..42 + 8 * j].try_into().unwrap());
            let (s0, s1) = (b[74 + 2 * j] as i32, b[74 + 2 * j + 1] as i32);
            (g, m, scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1))
        }
        18 => {
            // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
            let q = &b[2 + 16 * j..];
            let g = _mm512_set_epi32(g3x(q[15] as usize), g3x(q[14] as usize), g3x(q[13] as usize), g3x(q[12] as usize),
                                     g3x(q[11] as usize), g3x(q[10] as usize), g3x(q[9] as usize), g3x(q[8] as usize),
                                     g3x(q[7] as usize), g3x(q[6] as usize), g3x(q[5] as usize), g3x(q[4] as usize),
                                     g3x(q[3] as usize), g3x(q[2] as usize), g3x(q[1] as usize), g3x(q[0] as usize));
            let a = u32::from_le_bytes(b[66 + 8 * j..70 + 8 * j].try_into().unwrap());
            let c = u32::from_le_bytes(b[70 + 8 * j..74 + 8 * j].try_into().unwrap());
            let (sa, sb) = (2 * (a >> 28) as i32 + 1, 2 * (c >> 28) as i32 + 1);
            (g, signs4(a) | (signs4(c) << 32), scales4(sa, sa, sb, sb))
        }
        _ => {
            // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
            let q = &b[2 + 16 * j..];
            let (h0, h1) = (b[66 + 2 * j] as usize, b[66 + 2 * j + 1] as usize);
            let gi = |k: usize, hh: usize, kk: usize| g3s(q[k] as usize | (((hh >> kk) & 1) << 8));
            let g = _mm512_set_epi32(gi(15, h1, 7), gi(14, h1, 6), gi(13, h1, 5), gi(12, h1, 4), gi(11, h1, 3), gi(10, h1, 2),
                                     gi(9, h1, 1), gi(8, h1, 0), gi(7, h0, 7), gi(6, h0, 6), gi(5, h0, 5), gi(4, h0, 4),
                                     gi(3, h0, 3), gi(2, h0, 2), gi(1, h0, 1), gi(0, h0, 0));
            let m = u64::from_le_bytes(b[74 + 8 * j..82 + 8 * j].try_into().unwrap());
            let s = b[106 + j] as i32;
            let (sa, sb) = (2 * (s & 15) + 1, 2 * (s >> 4) + 1);
            (g, m, scales4(sa, sa, sb, sb))
        }
    }
}

/// Sum of |w| * (y carrying w's sign) over 64 bytes, as 16 i32 lanes (4 bytes each).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn signed_dot(w: __m512i, y: __m512i) -> __m512i {
    let neg = _mm512_movepi8_mask(w);
    let ys = _mm512_mask_sub_epi8(y, neg, _mm512_setzero_si512(), y);
    _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_abs_epi8(w), ys), _mm512_set1_epi16(1))
}

/// 32 nibbles of two 16-byte runs `a`, `b` (an IQ4_NL block each, or two IQ4_XS sub-blocks) as 64 int8 values, in
/// value order: a's lows, a's highs, b's lows, b's highs.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn nibbles(a: &[u8], b: &[u8]) -> __m512i {
    // SAFETY: IQ4NL_VALUES is 16 bytes.
    let lut = _mm512_broadcast_i32x4(unsafe { _mm_loadu_si128(IQ4NL_VALUES.as_ptr() as *const __m128i) });
    // SAFETY: a and b hold at least 16 bytes each.
    let (va, vb) = unsafe { (_mm_loadu_si128(a.as_ptr() as *const __m128i), _mm_loadu_si128(b.as_ptr() as *const __m128i)) };
    let lo = _mm_set1_epi8(0x0f);
    let parts = [_mm_and_si128(va, lo), _mm_and_si128(_mm_srli_epi16(va, 4), lo), _mm_and_si128(vb, lo),
                 _mm_and_si128(_mm_srli_epi16(vb, 4), lo)];
    let idx = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set_m128i(parts[1], parts[0])), _mm256_set_m128i(parts[3], parts[2]), 1);
    _mm512_shuffle_epi8(lut, idx)
}

/// One stored row of `ty` dotted with quantized activations (`y.blk` as `act_block(ty)` says).
///
/// # Safety
/// `available()` must be true; `row` must hold whole blocks covering `y.qs.len()` values.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn row_dot(ty: &str, row: &[u8], y: &Q8) -> f32 {
    let n = y.qs.len();
    let yq = y.qs.as_ptr();
    // SAFETY (all loads below): offsets stay inside y.qs (n values) and inside `row` (whole blocks for n values).
    unsafe {
        match ty {
            "IQ2_S" | "IQ3_XXS" | "IQ3_S" => {
                let (code, bytes, k) = match ty {
                    "IQ2_S" => (22u8, 82usize, 0.125f32),
                    "IQ3_XXS" => (18, 98, 0.25),
                    _ => (21, 110, 1.0),
                };
                let mut accf = _mm512_setzero_ps();
                for i in 0..n / 256 {
                    let blk = &row[i * bytes..(i + 1) * bytes];
                    let mut acci = _mm512_setzero_si512();
                    for j in 0..4 {
                        let (g, m, sc) = decode_grid(code, blk, j);
                        let yv = _mm512_loadu_si512(yq.add(256 * i + 64 * j) as *const __m512i);
                        let ys = _mm512_mask_sub_epi8(yv, m, _mm512_setzero_si512(), yv);
                        acci = _mm512_add_epi32(acci, _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
                    }
                    accf = _mm512_fmadd_ps(_mm512_set1_ps(h(blk, 0) * k * y.d[i]), _mm512_cvtepi32_ps(acci), accf);
                }
                _mm512_reduce_add_ps(accf)
            }
            "IQ4_XS" => {
                // d, scales_h (u16), scales_l[4], qs[128]: 8 sub-blocks of 32, scale ((l | h << 4) - 32) * d
                let mut accf = _mm512_setzero_ps();
                for i in 0..n / 256 {
                    let blk = &row[i * 136..(i + 1) * 136];
                    let d = h(blk, 0) * y.d[i];
                    let sh = u16::from_le_bytes([blk[2], blk[3]]);
                    let sc = |g: usize| (((blk[4 + g / 2] >> (4 * (g % 2))) & 0x0f) as i32 | ((((sh >> (2 * g)) & 3) as i32) << 4)) - 32;
                    for p in 0..4 {
                        let w = nibbles(&blk[8 + 32 * p..], &blk[8 + 32 * p + 16..]);
                        let yv = _mm512_loadu_si512(yq.add(256 * i + 64 * p) as *const __m512i);
                        let dot = signed_dot(w, yv);
                        let (s0, s1) = (sc(2 * p) as f32 * d, sc(2 * p + 1) as f32 * d);
                        let sv = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(s0), _mm512_set1_ps(s1));
                        accf = _mm512_fmadd_ps(sv, _mm512_cvtepi32_ps(dot), accf);
                    }
                }
                _mm512_reduce_add_ps(accf)
            }
            "IQ4_NL" => {
                // two 32-value blocks (d, 16 nibble bytes) at a time
                let mut accf = _mm512_setzero_ps();
                for p in 0..n / 64 {
                    let (b0, b1) = (&row[p * 36..p * 36 + 18], &row[p * 36 + 18..p * 36 + 36]);
                    let w = nibbles(&b0[2..], &b1[2..]);
                    let yv = _mm512_loadu_si512(yq.add(64 * p) as *const __m512i);
                    let dot = signed_dot(w, yv);
                    let sv = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(h(b0, 0) * y.d[2 * p]), _mm512_set1_ps(h(b1, 0) * y.d[2 * p + 1]));
                    accf = _mm512_fmadd_ps(sv, _mm512_cvtepi32_ps(dot), accf);
                }
                _mm512_reduce_add_ps(accf)
            }
            "Q2_0" => {
                // d, 16 bytes of 2-bit codes (value (c - 1) * d, element e at byte e / 4, bits 2 * (e % 4))
                // 128-bit lane k of the broadcast, shifted by 2k, holds element 4i + k at byte i; vpermb puts it at e
                let shifts = _mm512_set_epi16(6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0);
                let mut perm = [0i8; 64];
                for (e, v) in perm.iter_mut().enumerate() {
                    *v = ((e % 4) * 16 + e / 4) as i8;
                }
                let perm = _mm512_loadu_si512(perm.as_ptr() as *const __m512i);
                let three = _mm512_set1_epi8(3);
                let mut accf = _mm512_setzero_ps();
                let mut off = 0f32;
                for i in 0..n / 64 {
                    let blk = &row[i * 18..(i + 1) * 18];
                    let codes = _mm512_broadcast_i32x4(_mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i));
                    let q = _mm512_and_si512(_mm512_srlv_epi16(codes, shifts), three);
                    let q = _mm512_permutexvar_epi8(perm, q);
                    let yv = _mm512_loadu_si512(yq.add(64 * i) as *const __m512i);
                    let dot = _mm512_madd_epi16(_mm512_maddubs_epi16(q, yv), _mm512_set1_epi16(1));
                    let d = h(blk, 0);
                    let (d0, d1) = (d * y.d[2 * i], d * y.d[2 * i + 1]);
                    accf = _mm512_fmadd_ps(_mm512_mask_blend_ps(0xff00, _mm512_set1_ps(d0), _mm512_set1_ps(d1)), _mm512_cvtepi32_ps(dot), accf);
                    off += d0 * y.sums[2 * i] as f32 + d1 * y.sums[2 * i + 1] as f32;
                }
                _mm512_reduce_add_ps(accf) - off
            }
            _ => panic!("{ty}: no AVX-512 expert kernel"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Random rows of every type: the int8 dot is the f64 dot of the dequantized row with the quantized-back
    /// activations, to f32 rounding (the activation quantization is the only approximation).
    #[test]
    fn rows_match_dequantize() {
        if !available() {
            return eprintln!("SKIPPED, not passed: no AVX-512 BW + VBMI");
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (ty, id, n, bytes, blk) in [("IQ2_S", 22, 256, 82, 256), ("IQ3_XXS", 18, 256, 98, 256), ("IQ3_S", 21, 256, 110, 256),
                                        ("IQ4_XS", 23, 256, 136, 256), ("IQ4_NL", 20, 32, 18, 32), ("Q2_0", 42, 64, 18, 32)] {
            let gt = crate::gguf::GgmlType::from_id(id);
            for _ in 0..100 {
                let len = if n == 256 { 2560 } else { 640 };
                let mut row: Vec<u8> = (0..len / n * bytes).map(|_| rng(&mut seed) as u8).collect();
                for b in row.chunks_mut(bytes) {
                    b[1] = 0x2c; // small finite f16 scales
                }
                let mut w = vec![0f32; len];
                crate::quant::dequantize(gt, &row, &mut w).unwrap();
                let x: Vec<f32> = (0..len).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
                let y = quantize(&x, blk);
                let xq: Vec<f64> = (0..len).map(|i| y.qs[i] as f64 * y.d[i / blk] as f64).collect();
                let exact: f64 = w.iter().zip(&xq).map(|(&a, &b)| a as f64 * b).sum();
                let mag: f64 = w.iter().zip(&xq).map(|(&a, &b)| (a as f64 * b).abs()).sum();
                // SAFETY: available() checked above; row holds whole blocks for len values.
                let got = unsafe { row_dot(ty, &row, &y) } as f64;
                assert!((got - exact).abs() <= 1e-4 * mag.max(1e-9), "{ty}: {got} vs {exact} (mag {mag})");
            }
        }
    }
}
