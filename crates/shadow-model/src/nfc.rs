//! Unicode NFC, because the checkpoint's tokenizer.json normalizes every text to NFC before tokenizing it and the GGUF
//! does not say so.  Rust's standard library has no normalization; the tables are generated (`nfc_table.rs`).
//!
//! The standard algorithm (UAX #15): fully decompose, put each run of combining marks in canonical order, then
//! recompose.  Hangul syllables are algorithmic.  ASCII text is returned untouched, which is nearly all of it.

use crate::nfc_table::{CCC, COMPOSE, DECOMP};
use std::borrow::Cow;

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn ccc(c: u32) -> u8 {
    match CCC.binary_search_by(|&(a, b, _)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => CCC[i].2,
        Err(_) => 0,
    }
}

fn decompose(c: u32, out: &mut Vec<u32>) {
    if (S_BASE..S_BASE + S_COUNT).contains(&c) {
        let s = c - S_BASE;
        out.push(L_BASE + s / N_COUNT);
        out.push(V_BASE + (s % N_COUNT) / T_COUNT);
        if s % T_COUNT != 0 {
            out.push(T_BASE + s % T_COUNT);
        }
        return;
    }
    match DECOMP.binary_search_by_key(&c, |&(cp, _, _)| cp) {
        Ok(i) => {
            let (_, a, b) = DECOMP[i];
            decompose(a, out);
            if b != 0 {
                decompose(b, out);
            }
        }
        Err(_) => out.push(c),
    }
}

fn compose(a: u32, b: u32) -> Option<u32> {
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a) && (a - S_BASE) % T_COUNT == 0
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return Some(a + (b - T_BASE));
    }
    COMPOSE.binary_search_by_key(&(a, b), |&(x, y, _)| (x, y)).ok().map(|i| COMPOSE[i].2)
}

pub fn nfc(s: &str) -> Cow<'_, str> {
    if s.is_ascii() {
        return Cow::Borrowed(s);
    }
    let mut d = Vec::with_capacity(s.len());
    for ch in s.chars() {
        decompose(ch as u32, &mut d);
    }
    // canonical ordering: each run of non-starters, stably sorted by combining class
    let mut i = 0;
    while i < d.len() {
        if ccc(d[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < d.len() && ccc(d[i]) != 0 {
            i += 1;
        }
        d[start..i].sort_by_key(|&c| ccc(c));
    }
    // canonical composition
    let mut out: Vec<u32> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    let mut last_class = 0u16;
    for &c in &d {
        let class = ccc(c) as u16;
        if let Some(si) = starter {
            if last_class < class || last_class == 0 {
                if let Some(comp) = compose(out[si], c) {
                    out[si] = comp;
                    continue;
                }
            }
        }
        if class == 0 {
            starter = Some(out.len());
        }
        // a leading non-starter blocks nothing it is not composed with
        last_class = if starter.is_none() { 256 } else { class };
        out.push(c);
    }
    let r: String = out.into_iter().map(|c| char::from_u32(c).unwrap_or('\u{FFFD}')).collect();
    if r == s { Cow::Borrowed(s) } else { Cow::Owned(r) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_cases() {
        assert_eq!(nfc("plain ascii"), "plain ascii");
        assert_eq!(nfc("e\u{301}"), "\u{e9}");
        assert_eq!(nfc("\u{e9}"), "\u{e9}");
        assert_eq!(nfc("a\u{323}\u{307}"), "\u{1ea1}\u{307}", "canonical order: dot below (220) before dot above (230)");
        assert_eq!(nfc("a\u{307}\u{323}"), "\u{1ea1}\u{307}");
        assert_eq!(nfc("\u{212b}"), "\u{c5}", "the Angstrom sign is a singleton for A-ring");
        assert_eq!(nfc("\u{1112}\u{1161}\u{11ab}"), "\u{d55c}", "Hangul jamo compose");
        assert_eq!(nfc("\u{d55c}"), "\u{d55c}");
        assert_eq!(nfc("\u{fb01}"), "\u{fb01}", "compatibility forms are NFC-stable");
        assert_eq!(nfc("\u{301}a"), "\u{301}a", "a leading mark has nothing to attach to");
    }
}
