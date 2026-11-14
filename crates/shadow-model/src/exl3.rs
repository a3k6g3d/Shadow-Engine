//! EXL3 (ExLlamaV3, https://github.com/turboderp-org/exllamav3, MIT) weights: the CPU reference decoder.
//!
//! A linear layer of an EXL3 model is four tensors: `trellis` (int16, `[in/16, out/16, 16 K]`), `suh` (f16, `[in]`),
//! `svh` (f16, `[out]`) and a scalar `mul1` or `mcg` that names the codebook (neither: codebook 0).  K is the bit rate:
//! 16 K int16 words a 16 x 16 tile of weights, so 256 K bits; for the half-integer rates of the `mul1` codebook (1.5,
//! 2.5, 3.5) it is `16 KA + 8` words.
//!
//! **Tile.**  The 256 K bits of a tile are one cyclic bit stream, most significant bit first inside each little-endian
//! u32 word.  Weight `t` (0..255 in the tile's own order) has the 16-bit window of the stream that ends at bit
//! `(t + 1) K` (mod 256 K): a tail-biting trellis, where consecutive weights' windows overlap by 16 - K bits.  The
//! window is a state; the codebook ("3INST", QTIP) turns the state into an f16 value without a table: a multiply by a
//! constant and a few byte or bit operations (`codebook` below).  Weight position `p = 8 L + j` (lane L 0..31, j 0..7)
//! belongs to the tensor-core fragment layout and sits at row `2 (L mod 4) + (j mod 2) + 8 ((j >> 1) mod 2)`, column
//! `2 (L >> 3) + ((L >> 2) mod 2) + 8 (j >> 2)` of the 16 x 16 tile (`tile_pos`); row = input index, column = output.
//!
//! **Basis.**  What the trellis holds is the weight in a rotated basis, `W_hat[in][out]`.  The weight of the model is
//! `W = diag(suh) H W_hat H diag(svh)` where `H` is the 128 x 128 Sylvester-Hadamard matrix divided by sqrt(128),
//! applied on each 128-block of the input index and of the output index; `y = x W`.  At inference the same is done to
//! the activations (`x diag(suh) H`, then `y' = . W_hat`, then `y' H diag(svh)`), which is what the kernels do.
//!
//! Everything here follows exllamav3's `reconstruct.cu`, `exl3_dq.cuh`, `codebook.cuh` and `modules/quant/exl3.py`
//! (the licence is in `NOTICE`); the f16 arithmetic of the codebooks is reproduced exactly (`round_f16`).

use crate::half::f16_to_f32;
use crate::safetensors::{Error, SafeTensors};

/// Rounds to the nearest f16 (ties to even) and returns it as an f32: what a half-precision operation gives.
pub fn round_f16(x: f64) -> f32 {
    f16_to_f32(f64_to_f16_bits(x))
}

pub fn f64_to_f16_bits(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 6.103515625e-5 {
        // below the smallest normal 2^-14: multiples of 2^-24; 1024 of them is the smallest normal, whose bits it is
        let n = (a * 16777216.0).round_ties_even() as u16;
        return sign | n;
    }
    let mut e = a.log2().floor() as i32;
    // log2 can be off by one at the edges of a binade
    if 2f64.powi(e) > a {
        e -= 1;
    } else if 2f64.powi(e + 1) <= a {
        e += 1;
    }
    let mant = a / 2f64.powi(e); // [1, 2)
    let mut m = ((mant - 1.0) * 1024.0).round_ties_even() as u32;
    if m == 1024 {
        m = 0;
        e += 1;
    }
    if e + 15 >= 31 {
        return sign | 0x7c00;
    }
    sign | (((e + 15) as u16) << 10) | m as u16
}

/// The three codebooks: a 16-bit state to the f16 value of one weight (as an f32 holding exactly that f16).
pub mod codebook {
    use super::*;

    /// `lop3(x, 0x8fff8fff, 0x3b603b60, 0x6a)` = `(x & 0x8fff8fff) ^ 0x3b603b60`: two f16 bit patterns; their sum in f16.
    fn two_halves_sum(x: u32) -> f32 {
        let x = (x & 0x8fff_8fff) ^ 0x3b60_3b60;
        round_f16(f16_to_f32((x & 0xffff) as u16) as f64 + f16_to_f32((x >> 16) as u16) as f64)
    }

    /// Codebook 0.
    pub fn cb0(state: u32) -> f32 {
        two_halves_sum(state.wrapping_mul(89226354).wrapping_add(64248484))
    }

    /// Codebook 1 ("mcg").
    pub fn cb1(state: u32) -> f32 {
        two_halves_sum(state.wrapping_mul(0xCBAC1FED))
    }

    /// Codebook 2 ("mul1"): the four bytes of the product, summed on top of 0x6400 (the f16 1024.0), are an f16 in
    /// [1024, 2048) that is mapped linearly, in f16 arithmetic and fused, to about [-10.4, 10.4].
    pub fn cb2(state: u32) -> f32 {
        let x = state.wrapping_mul(0x83DCD12D);
        let sum = 0x6400u32 + (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
        let h = f16_to_f32(sum as u16) as f64;
        let k_inv = f16_to_f32(0x1eee) as f64;
        let k_bias = f16_to_f32(0xc931) as f64;
        round_f16(h * k_inv + k_bias)
    }

    pub fn decode(cb: u8, state: u32) -> f32 {
        match cb {
            0 => cb0(state),
            1 => cb1(state),
            _ => cb2(state),
        }
    }
}

/// Row and column of weight position `p` (0..255, trellis order) inside its 16 x 16 tile.
pub fn tile_pos(p: usize) -> (usize, usize) {
    let (l, j) = (p / 8, p % 8);
    let row = 2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1);
    let col = 2 * (l >> 3) + ((l >> 2) & 1) + 8 * (j >> 2);
    (row, col)
}

/// The bit rate of a trellis whose last dimension is `last`: (bits, half) for K = bits (+ 0.5 when half).
pub fn bits_from_last_dim(last: usize) -> Option<(usize, bool)> {
    if last % 16 == 0 && (1..=8).contains(&(last / 16)) {
        Some((last / 16, false))
    } else if last % 16 == 8 && (1..=3).contains(&(last / 16)) {
        Some((last / 16, true))
    } else {
        None
    }
}

/// Bit `s` (counting from the most significant bit of word 0) of a tile's stream of `words` u32 (two int16 each).
#[inline]
fn stream_bit(tile: &[u16], s: usize) -> u32 {
    let w = s / 32;
    let word = tile[2 * w] as u32 | ((tile[2 * w + 1] as u32) << 16);
    (word >> (31 - s % 32)) & 1
}

/// The 16-bit state of weight `t` of a tile with integer bit rate `bits`.
pub fn state_int(tile: &[u16], bits: usize, t: usize) -> u32 {
    let n = 256 * bits;
    let end = (t + 1) * bits;
    let mut w = 0u32;
    for b in 0..16 {
        w = (w << 1) | stream_bit(tile, (end + n - 16 + b) % n);
    }
    w
}

/// The 256 weights of one tile, in trellis order, as the exact f16 values the codebook gives.
pub fn decode_tile(tile: &[u16], bits: usize, cb: u8) -> [f32; 256] {
    let mut out = [0f32; 256];
    for (t, o) in out.iter_mut().enumerate() {
        *o = codebook::decode(cb, state_int(tile, bits, t));
    }
    out
}

/// One EXL3 linear layer, as read from a checkpoint.
#[derive(Clone, Debug)]
pub struct Exl3Linear {
    pub in_features: usize,
    pub out_features: usize,
    /// K: bits a weight; `half` means K + 0.5.
    pub bits: usize,
    pub half: bool,
    pub cb: u8,
    /// `[in/16][out/16][16 K (+8)]`, as stored.
    pub trellis: Vec<u16>,
    pub suh: Vec<f32>,
    pub svh: Vec<f32>,
}

impl Exl3Linear {
    /// Words of one tile.
    pub fn tile_words(&self) -> usize {
        16 * self.bits + if self.half { 8 } else { 0 }
    }

    /// Reads the tensors `{prefix}.trellis`, `.suh`, `.svh` and `.mul1` / `.mcg`.
    pub fn load(st: &SafeTensors, prefix: &str) -> Result<Exl3Linear, Error> {
        let name = |s: &str| format!("{prefix}.{s}");
        let t = st.get(&name("trellis")).ok_or_else(|| Error(format!("no tensor {}", name("trellis"))))?;
        if t.shape.len() != 3 || t.dtype != "I16" {
            return Err(Error(format!("{}: expected a 3-D I16 trellis, found {} {:?}", name("trellis"), t.dtype, t.shape)));
        }
        let (bits, half) = bits_from_last_dim(t.shape[2]).ok_or_else(|| Error(format!("{}: {} words a tile is not a bit rate", name("trellis"), t.shape[2])))?;
        let cb = if st.get(&name("mcg")).is_some() { 1 } else if st.get(&name("mul1")).is_some() { 2 } else { 0 };
        if half && cb != 2 {
            return Err(Error(format!("{prefix}: a half-integer bit rate needs the mul1 codebook")));
        }
        let suh = st.read_f32(&name("suh"))?;
        let svh = st.read_f32(&name("svh"))?;
        let (in_features, out_features) = (t.shape[0] * 16, t.shape[1] * 16);
        if suh.len() != in_features || svh.len() != out_features {
            return Err(Error(format!("{prefix}: suh/svh lengths {}/{} do not match {in_features} x {out_features}", suh.len(), svh.len())));
        }
        Ok(Exl3Linear { in_features, out_features, bits, half, cb, trellis: st.read_u16(&name("trellis"))?, suh, svh })
    }

    /// `W_hat[in][out]` (row-major, f32 holding f16 values): the weight in the rotated basis, straight from the trellis.
    pub fn reconstruct_inner(&self) -> Vec<f32> {
        assert!(!self.half, "half-integer bit rates are not decoded yet");
        let (kt, nt, tw) = (self.in_features / 16, self.out_features / 16, self.tile_words());
        let mut w = vec![0f32; self.in_features * self.out_features];
        for k in 0..kt {
            for n in 0..nt {
                let tile = &self.trellis[(k * nt + n) * tw..(k * nt + n + 1) * tw];
                let vals = decode_tile(tile, self.bits, self.cb);
                for (p, v) in vals.iter().enumerate() {
                    let (r, c) = tile_pos(p);
                    w[(k * 16 + r) * self.out_features + n * 16 + c] = *v;
                }
            }
        }
        w
    }

    /// `W[in][out]`: the weight of the model, `diag(suh) H W_hat H diag(svh)`.
    pub fn reconstruct(&self) -> Vec<f32> {
        let (ni, no) = (self.in_features, self.out_features);
        let mut w = self.reconstruct_inner();
        // H on the input index: each 128-block of rows, every column
        let mut col = [0f32; 128];
        for b in 0..ni / 128 {
            for c in 0..no {
                for (i, v) in col.iter_mut().enumerate() {
                    *v = w[(b * 128 + i) * no + c];
                }
                wht128(&mut col);
                for (i, v) in col.iter().enumerate() {
                    w[(b * 128 + i) * no + c] = *v;
                }
            }
        }
        for (i, s) in self.suh.iter().enumerate() {
            for v in &mut w[i * no..(i + 1) * no] {
                *v *= s;
            }
        }
        // H on the output index: each 128-block of columns, every row
        for r in 0..ni {
            for b in 0..no / 128 {
                let row: &mut [f32; 128] = (&mut w[r * no + b * 128..r * no + (b + 1) * 128]).try_into().unwrap();
                wht128(row);
            }
        }
        for r in 0..ni {
            for (c, s) in self.svh.iter().enumerate() {
                w[r * no + c] *= s;
            }
        }
        w
    }
}

/// Re-layouts that need no decode: 128 outputs or inputs are 8 tile columns or rows and one 128-chunk of `svh` / `suh`,
/// and 128 is the Hadamard block, so moving whole 128-blocks moves whole blocks of the rotation.
impl Exl3Linear {
    /// Output block `first + i` of the result is output block `first + map[i]` of `self` (blocks of 128 outputs).
    pub fn permute_out_blocks(&mut self, first: usize, map: &[usize]) {
        let (kt, nt, tw) = (self.in_features / 16, self.out_features / 16, self.tile_words());
        assert!((first + map.len()) * 8 <= nt, "permute_out_blocks: {} blocks from {first} in {nt} tile columns", map.len());
        let src = self.trellis.clone();
        let svh = self.svh.clone();
        for (i, &m) in map.iter().enumerate() {
            for c in 0..8 {
                let (nd, ns) = ((first + i) * 8 + c, (first + m) * 8 + c);
                for k in 0..kt {
                    let (d, s) = ((k * nt + nd) * tw, (k * nt + ns) * tw);
                    self.trellis[d..d + tw].copy_from_slice(&src[s..s + tw]);
                }
            }
            self.svh[(first + i) * 128..(first + i + 1) * 128].copy_from_slice(&svh[(first + m) * 128..(first + m + 1) * 128]);
        }
    }

    /// Input block `first + i` of the result is input block `first + map[i]` of `self` (blocks of 128 inputs).
    pub fn permute_in_blocks(&mut self, first: usize, map: &[usize]) {
        let (kt, nt, tw) = (self.in_features / 16, self.out_features / 16, self.tile_words());
        assert!((first + map.len()) * 8 <= kt, "permute_in_blocks: {} blocks from {first} in {kt} tile rows", map.len());
        let src = self.trellis.clone();
        let suh = self.suh.clone();
        let row = nt * tw;
        for (i, &m) in map.iter().enumerate() {
            for r in 0..8 {
                let (kd, ks) = ((first + i) * 8 + r, (first + m) * 8 + r);
                self.trellis[kd * row..(kd + 1) * row].copy_from_slice(&src[ks * row..(ks + 1) * row]);
            }
            self.suh[(first + i) * 128..(first + i + 1) * 128].copy_from_slice(&suh[(first + m) * 128..(first + m + 1) * 128]);
        }
    }

    /// The linear layer restricted to output blocks `first .. first + count` (128 outputs each).
    pub fn slice_out_blocks(&self, first: usize, count: usize) -> Exl3Linear {
        let (kt, nt, tw) = (self.in_features / 16, self.out_features / 16, self.tile_words());
        assert!((first + count) * 8 <= nt, "slice_out_blocks: blocks {first}..{} of {nt} tile columns", first + count);
        let nt2 = count * 8;
        let mut trellis = Vec::with_capacity(kt * nt2 * tw);
        for k in 0..kt {
            for n in first * 8..(first + count) * 8 {
                trellis.extend_from_slice(&self.trellis[(k * nt + n) * tw..(k * nt + n + 1) * tw]);
            }
        }
        Exl3Linear {
            in_features: self.in_features, out_features: count * 128, bits: self.bits, half: self.half, cb: self.cb, trellis,
            suh: self.suh.clone(), svh: self.svh[first * 128..(first + count) * 128].to_vec(),
        }
    }
}

/// In-place Walsh-Hadamard transform of 128 values in the natural (Sylvester) order, divided by sqrt(128).
pub fn wht128(v: &mut [f32; 128]) {
    let mut h = 1;
    while h < 128 {
        for i in (0..128).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = 0.088_388_35_f32; // 1 / sqrt(128)
    for x in v.iter_mut() {
        *x *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_rounding_matches_known_values() {
        assert_eq!(f64_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f64_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f64_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f64_to_f16_bits(65520.0), 0x7c00);
        assert_eq!(f64_to_f16_bits(6.103515625e-5), 0x0400);
        assert_eq!(f64_to_f16_bits(5.960464477539063e-8), 0x0001);
        // 1 + 2^-11 is a tie between 1.0 and 1 + 2^-10: to even is 1.0
        assert_eq!(f64_to_f16_bits(1.0 + 2f64.powi(-11)), 0x3c00);
        assert_eq!(f64_to_f16_bits(1.0 + 3.0 * 2f64.powi(-11)), 0x3c02);
        for bits in 0u16..0x7c00 {
            assert_eq!(f64_to_f16_bits(f16_to_f32(bits) as f64), bits, "{bits:#x}");
        }
    }

    #[test]
    fn tile_positions_are_a_bijection_onto_the_tile() {
        let mut seen = [false; 256];
        for p in 0..256 {
            let (r, c) = tile_pos(p);
            assert!(r < 16 && c < 16);
            assert!(!std::mem::replace(&mut seen[r * 16 + c], true), "position {p} repeats a cell");
        }
    }

    #[test]
    fn hadamard_is_its_own_inverse() {
        let mut v = [0f32; 128];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i * 37 % 11) as f32) - 5.0;
        }
        let orig = v;
        wht128(&mut v);
        wht128(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// The codebooks are made to look like a unit Gaussian; a wrong constant or bit pattern would not.
    #[test]
    fn every_codebook_is_roughly_a_unit_gaussian() {
        for cb in 0..3u8 {
            let (mut s, mut s2) = (0f64, 0f64);
            for state in 0..65536u32 {
                let v = codebook::decode(cb, state) as f64;
                assert!(v.is_finite(), "cb {cb} state {state}");
                s += v;
                s2 += v * v;
            }
            let (mean, var) = (s / 65536.0, s2 / 65536.0);
            eprintln!("codebook {cb}: mean {mean:.4} std {:.4}", (var - mean * mean).sqrt());
            assert!(mean.abs() < 0.1, "cb {cb} mean {mean}");
            assert!((var - mean * mean).sqrt() > 0.7 && (var - mean * mean).sqrt() < 1.3, "cb {cb} std");
        }
    }
}
