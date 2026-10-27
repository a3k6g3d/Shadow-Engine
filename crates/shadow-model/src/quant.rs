//! Dequantization, in Rust, for every type this model uses: the CPU reference (stage 1) and the oracle every GPU
//! dequant kernel is tested against.
//!
//! Each format follows its ggml definition. The arithmetic is written in the same order as the gguf Python
//! package's numpy implementation, so that with no fused multiply-adds (Rust never fuses on its own) the two agree bit
//! for bit; `tests/dequant.rs` checks that on real blocks of the model. Q2_0 has no such check: the gguf package (0.19) does
//! not implement it, so it is written from upstream ggml's `dequantize_row_q2_0` and tested by hand and for sanity.

use crate::gguf::GgmlType;
use crate::half::{bf16_to_f32, f16_to_f32};
use crate::quant_tables::{IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES, KSIGNS};

const QK_K: usize = 256;

fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

fn sign(bit: u8) -> f32 {
    if bit & 1 == 0 { 1.0 } else { -1.0 }
}

/// Q4_K and Q5_K's 6-bit scales and mins, eight of each, packed in 12 bytes.
fn scale_min_k4(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    let (mut sc, mut mn) = ([0u8; 8], [0u8; 8]);
    for j in 0..4 {
        sc[j] = s[j] & 0x3f;
        mn[j] = s[4 + j] & 0x3f;
        sc[4 + j] = (s[8 + j] & 0x0f) | ((s[j] >> 2) & 0x30);
        mn[4 + j] = (s[8 + j] >> 4) | ((s[4 + j] >> 2) & 0x30);
    }
    (sc, mn)
}

fn q4_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let (sc, mn) = scale_min_k4(&b[4..16]);
    let qs = &b[16..144];
    for i in 0..8 {
        let (ds, dm) = (d * sc[i] as f32, dmin * mn[i] as f32);
        let (g, shift) = (i / 2, (i % 2) * 4);
        for c in 0..32 {
            let q = (qs[g * 32 + c] >> shift) & 0x0f;
            y[i * 32 + c] = ds * q as f32 - dm;
        }
    }
}

fn q5_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let (sc, mn) = scale_min_k4(&b[4..16]);
    let qh = &b[16..48];
    let qs = &b[48..176];
    for i in 0..8 {
        let (ds, dm) = (d * sc[i] as f32, dmin * mn[i] as f32);
        let (g, shift) = (i / 2, (i % 2) * 4);
        for c in 0..32 {
            let q = ((qs[g * 32 + c] >> shift) & 0x0f) | (((qh[c] >> i) & 1) << 4);
            y[i * 32 + c] = ds * q as f32 - dm;
        }
    }
}

fn q6_k(b: &[u8], y: &mut [f32]) {
    let (ql, qh, scales) = (&b[0..128], &b[128..192], &b[192..208]);
    let d = f16_at(b, 208);
    for r in 0..8 {
        let (g, h, k, s) = (r / 4, (r % 4) / 2, r % 2, r % 4);
        for c in 0..32 {
            let lo = (ql[g * 64 + k * 32 + c] >> (h * 4)) & 0x0f;
            let hi = (qh[g * 32 + c] >> (2 * s)) & 0x03;
            let q = ((lo | (hi << 4)) as i8).wrapping_sub(32);
            let e = r * 32 + c;
            y[e] = (d * scales[e / 16] as i8 as f32) * q as f32;
        }
    }
}

fn iq2_s(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for j in 0..32 {
        let idx = qs[j] as usize | ((((qh[j / 4] >> (2 * (j % 4))) & 3) as usize) << 8);
        for (k, &v) in IQ2S_GRID[idx].iter().enumerate() {
            let e = j * 8 + k;
            let s = (scales[e / 32] >> (4 * ((e / 16) % 2))) & 0x0f;
            let db = d * (0.5 + s as f32) * 0.25;
            y[e] = db * v as f32 * sign(signs[e / 8] >> (e % 8));
        }
    }
}

fn iq3_xxs(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, sc) = (&b[2..66], &b[66..98]);
    for g in 0..8 {
        let s32 = u32::from_le_bytes(sc[g * 4..g * 4 + 4].try_into().unwrap());
        let db = d * (0.5 + (s32 >> 28) as f32) * 0.5;
        for k in 0..4 {
            let signs = KSIGNS[((s32 >> (7 * k)) & 0x7f) as usize];
            for m in 0..8 {
                let e = g * 32 + k * 8 + m;
                let v = IQ3XXS_GRID[qs[e / 4] as usize][e % 4];
                y[e] = db * v as f32 * sign(signs >> m);
            }
        }
    }
}

fn iq3_s(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
    for e in 0..QK_K {
        let j = e / 4;
        let idx = qs[j] as usize | ((((qh[j / 8] >> (j % 8)) & 1) as usize) << 8);
        let g = e / 32;
        let s = (scales[g / 2] >> (4 * (g % 2))) & 0x0f;
        let db = d * (1 + 2 * s) as f32;
        y[e] = db * IQ3S_GRID[idx][e % 4] as f32 * sign(signs[e / 8] >> (e % 8));
    }
}

/// Q2_0 (upstream ggml, `dequantize_row_q2_0`): an f16 scale, then 64 two-bit codes, four per byte, lowest bits first;
/// code `q` is `(q - 1) * d`, so the four levels are -d, 0, d, 2d.
fn q2_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (j, v) in y.iter_mut().enumerate().take(64) {
        let q = (b[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
        *v = (q as i32 - 1) as f32 * d;
    }
}

/// NVFP4 (upstream ggml, `dequantize_row_nvfp4`): 4 unsigned E4M3 scales (bias 7, halved for the doubled E2M1 table),
/// then 4 sub-blocks of 8 bytes each (64 weights total); in each 8-byte sub-block the 8 low nibbles are weights 0..8
/// and the 8 high nibbles are weights 8..16.
fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for i in 0..32 {
        y[i] = d * (b[2 + i] as i8 as f32);
    }
}

fn nvfp4(b: &[u8], y: &mut [f32]) {
    const KVALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    for g in 0..4 {
        let sb = b[g];
        let d = if sb == 0 || sb == 0x7f {
            0.0f32
        } else {
            let exp = ((sb >> 3) & 0x0f) as u32;
            let man = (sb & 0x07) as u32;
            if exp == 0 {
                (man as f32) * (1.0 / 1024.0)
            } else {
                (8 + man) as f32 * f32::from_bits((116 + exp) << 23)
            }
        };
        let qs = &b[4 + g * 8..4 + (g + 1) * 8];
        for c in 0..8 {
            y[g * 16 + c] = d * KVALUES[(qs[c] & 0x0f) as usize] as f32;
            y[g * 16 + 8 + c] = d * KVALUES[(qs[c] >> 4) as usize] as f32;
        }
    }
}

fn iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for c in 0..16 {
        y[c] = d * IQ4NL_VALUES[(b[2 + c] & 0x0f) as usize] as f32;
        y[16 + c] = d * IQ4NL_VALUES[(b[2 + c] >> 4) as usize] as f32;
    }
}

fn iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let sh = u16::from_le_bytes([b[2], b[3]]);
    let (sl, qs) = (&b[4..8], &b[8..136]);
    for g in 0..8 {
        let lo = (sl[g / 2] >> (4 * (g % 2))) & 0x0f;
        let hi = ((sh >> (2 * g)) & 3) as u8;
        let dl = d * ((lo | (hi << 4)) as i8).wrapping_sub(32) as f32;
        for c in 0..16 {
            let q = qs[g * 16 + c];
            y[g * 32 + c] = dl * IQ4NL_VALUES[(q & 0x0f) as usize] as f32;
            y[g * 32 + 16 + c] = dl * IQ4NL_VALUES[(q >> 4) as usize] as f32;
        }
    }
}

/// Dequantizes `data` (whole blocks of `ty`) into `out`, which must hold exactly the number of weights `data` carries.
pub fn dequantize(ty: GgmlType, data: &[u8], out: &mut [f32]) -> Result<(), String> {
    let name = ty.name();
    let (block, bytes) = ty.block().ok_or_else(|| format!("{name}: unknown block layout"))?;
    let (block, bytes) = (block as usize, bytes as usize);
    if !data.len().is_multiple_of(bytes) || out.len() != data.len() / bytes * block {
        return Err(format!("{name}: {} bytes for {} weights", data.len(), out.len()));
    }
    let f: fn(&[u8], &mut [f32]) = match name.as_str() {
        "F32" => |b, y| y[0] = f32::from_le_bytes(b[..4].try_into().unwrap()),
        "F16" => |b, y| y[0] = f16_at(b, 0),
        "BF16" => |b, y| y[0] = bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        "Q4_K" => q4_k,
        "Q5_K" => q5_k,
        "Q6_K" => q6_k,
        "IQ2_S" => iq2_s,
        "IQ3_XXS" => iq3_xxs,
        "IQ3_S" => iq3_s,
        "IQ4_NL" => iq4_nl,
        "IQ4_XS" => iq4_xs,
        "NVFP4" => nvfp4,
        "Q2_0" => q2_0,
        "Q8_0" => q8_0,
        _ => return Err(format!("{name}: dequantization not implemented")),
    };
    for (b, y) in data.chunks_exact(bytes).zip(out.chunks_exact_mut(block)) {
        f(b, y);
    }
    Ok(())
}
