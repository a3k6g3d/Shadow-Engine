//! The tokenizer, from the GGUF alone: byte-level BPE (`tokenizer.ggml.model = "gpt2"`) with the vocabulary and the
//! merges the file carries, the `qwen35` pre-tokenizer split written by hand, and the special tokens
//! (token types 3 and 4) matched as whole tokens before anything else.
//!
//! The `qwen35` split, as published in upstream llama.cpp (src/llama-vocab.cpp), one alternation tried left to right:
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//! | \p{N}
//! |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//! ```
//!
//! It is implemented as a hand-written matcher, branch by branch, rather than a regex engine. `\p{..}` comes from
//! `unicode_table.rs` (generated); `\s` is Unicode White_Space, which is what `char::is_whitespace` tests.

use crate::gguf::{Gguf, Value};
use crate::nfc::nfc;
use crate::unicode_table::RANGES;
use std::collections::HashMap;

const LU_LT: u8 = 1;
const LL: u8 = 2;
const LM_LO: u8 = 4;
const L: u8 = LU_LT | LL | LM_LO;
const M: u8 = 8;
const N: u8 = 16;

fn cat(c: char) -> u8 {
    let cp = c as u32;
    match RANGES.binary_search_by(|&(a, b, _)| {
        if b < cp {
            std::cmp::Ordering::Less
        } else if a > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => RANGES[i].2,
        Err(_) => 0,
    }
}

fn is_l(c: char) -> bool {
    cat(c) & L != 0
}
fn is_n(c: char) -> bool {
    cat(c) & N != 0
}
fn is_lm(c: char) -> bool {
    cat(c) & (L | M) != 0
}
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}
fn contraction_len(cs: &[(usize, char)], i: usize) -> usize {
    let at = |k: usize| cs.get(k).map(|&(_, c)| c);
    if at(i) == Some('\'') {
        if let Some(d) = at(i + 1) {
            let d = d.to_ascii_lowercase();
            let e = at(i + 2).map(|x| x.to_ascii_lowercase());
            if matches!((d, e), ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l'))) {
                return 3;
            }
            if matches!(d, 's' | 't' | 'm' | 'd') {
                return 2;
            }
        }
    }
    0
}

/// The `qwen35` pre-tokenizer: `text` cut into the pieces BPE then works on, each a slice of `text`.
pub fn split_qwen35(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| cs.get(i).map(|&(_, c)| c);
    let byte = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let end = (|| {
            // 1. contractions
            let cl = contraction_len(&cs, i);
            if cl > 0 {
                return i + cl;
            }
            // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  - the optional prefix first, then without it
            let run = |mut j: usize| {
                while j < n && is_lm(cs[j].1) {
                    j += 1;
                }
                j
            };
            if !is_nl(c) && !is_l(c) && !is_n(c) && at(i + 1).is_some_and(is_lm) {
                return run(i + 1);
            }
            if is_lm(c) {
                return run(i);
            }
            // 3. \p{N}, one at a time
            if is_n(c) {
                return i + 1;
            }
            // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            let other = |x: char| !x.is_whitespace() && !is_lm(x) && !is_n(x);
            let start = if c == ' ' && at(i + 1).is_some_and(other) { i + 1 } else { i };
            if at(start).is_some_and(other) {
                let mut j = start;
                while j < n && other(cs[j].1) {
                    j += 1;
                }
                while j < n && is_nl(cs[j].1) {
                    j += 1;
                }
                return j;
            }
            // whitespace from here on: the maximal run [i, ws)
            let mut ws = i;
            while ws < n && cs[ws].1.is_whitespace() {
                ws += 1;
            }
            if ws > i {
                // 5. \s*[\r\n]+  - ends just after the last newline in the run
                if let Some(last_nl) = (i..ws).rev().find(|&j| is_nl(cs[j].1)) {
                    return last_nl + 1;
                }
                // 6. \s+(?!\S)  - the whole run at the end of the text, else all but its last character
                if ws == n {
                    return ws;
                }
                if ws - i >= 2 {
                    return ws - 1;
                }
                // 7. \s+
                return ws;
            }
            i + 1
        })();
        out.push(&text[byte(i)..byte(end)]);
        i = end;
    }
    out
}

/// GPT-2's byte <-> printable-character map: the 188 printable bytes map to themselves, the other 68 to U+0100 on.
fn byte_chars() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..256u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    map
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizer {
    Qwen35,
}

pub struct Tokenizer {
    pub tokens: Vec<String>,
    pub token_type: Vec<i32>,
    pub pre: PreTokenizer,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    /// Special tokens (types 3 and 4), longest first, so `<|im_start|>` wins over any prefix of it.
    specials: Vec<(String, u32)>,
    byte_char: [char; 256],
    char_byte: HashMap<char, u8>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eog: Vec<u32>,
}

#[derive(Debug)]
pub struct TokenizerError(pub String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TokenizerError {}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, TokenizerError> {
        let e = |m: &str| TokenizerError(m.to_string());
        let model = g.get("tokenizer.ggml.model").and_then(Value::as_str).ok_or_else(|| e("no tokenizer.ggml.model"))?;
        if model != "gpt2" {
            return Err(e(&format!("tokenizer model {model:?}: only byte-level BPE (gpt2) is implemented")));
        }
        let pre_str = g.get("tokenizer.ggml.pre").and_then(Value::as_str).unwrap_or("");
        let pre = match pre_str {
            "qwen35" => PreTokenizer::Qwen35,
            _ => return Err(e(&format!("pre-tokenizer {pre_str:?}: only qwen35 is implemented (this build runs the Qwen3.5-family model only)"))),
        };
        let strings = |k: &str| -> Result<Vec<String>, TokenizerError> {
            g.get(k).and_then(Value::as_array).ok_or_else(|| e(&format!("no {k}")))?
                .iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| e(&format!("{k}: not a string"))))
                .collect()
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let merges = strings("tokenizer.ggml.merges")?;
        let token_type: Vec<i32> = g.get("tokenizer.ggml.token_type").and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as i32).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);
        let bos = g.get("tokenizer.ggml.bos_token_id").and_then(Value::as_u64).map(|v| v as u32);
        let eos = g.get("tokenizer.ggml.eos_token_id").and_then(Value::as_u64).map(|v| v as u32);
        let mut eog: Vec<u32> = g.get("tokenizer.ggml.eog_token_ids").and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as u32).collect())
            .unwrap_or_default();
        if let Some(id) = eos {
            if !eog.contains(&id) {
                eog.push(id);
            }
        }
        let mut tok = Self::new_with_pre(tokens, token_type, &merges, bos, eos, pre);
        tok.eog = eog;
        Ok(tok)
    }

    pub fn new(tokens: Vec<String>, token_type: Vec<i32>, merges: &[String], bos: Option<u32>, eos: Option<u32>)
               -> Tokenizer {
        Self::new_with_pre(tokens, token_type, merges, bos, eos, PreTokenizer::Qwen35)
    }

    pub fn new_with_pre(
        tokens: Vec<String>,
        token_type: Vec<i32>,
        merges: &[String],
        bos: Option<u32>,
        eos: Option<u32>,
        pre: PreTokenizer,
    ) -> Tokenizer {
        let ids = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let ranks = merges.iter().enumerate()
            .filter_map(|(r, m)| m.split_once(' ').map(|(a, b)| ((a.to_string(), b.to_string()), r as u32)))
            .collect();
        let mut specials: Vec<(String, u32)> = tokens.iter().enumerate()
            .filter(|(i, _)| matches!(token_type.get(*i), Some(3) | Some(4)))
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let byte_char = byte_chars();
        let char_byte = byte_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let eog = eos.into_iter().collect();
        Tokenizer { tokens, token_type, pre, ids, ranks, specials, byte_char, char_byte, bos, eos, eog }
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    /// The strings of the special tokens (`<|im_start|>`, `<think>`, ...): what `encode(text, true)` turns into one token each.
    pub fn special_strings(&self) -> impl Iterator<Item = &str> {
        self.specials.iter().map(|(s, _)| s.as_str())
    }

    /// BPE over one pre-tokenizer piece.
    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mapped: String = piece.bytes().map(|b| self.byte_char[b as usize]).collect();
        if let Some(&id) = self.ids.get(&mapped) {
            out.push(id);
            return;
        }
        let mut syms: Vec<String> = mapped.chars().map(String::from).collect();
        loop {
            let best = syms.windows(2).enumerate()
                .filter_map(|(i, w)| self.ranks.get(&(w[0].clone(), w[1].clone())).map(|&r| (r, i)))
                .min();
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }
        for s in syms {
            match self.ids.get(&s) {
                Some(&id) => out.push(id),
                // A symbol the vocabulary does not have: fall back to its single bytes, which every byte-level vocab
                // has.  Counted by the tests; it should not happen on a well-formed vocabulary.
                None => out.extend(s.chars().filter_map(|c| self.ids.get(&c.to_string()).copied())),
            }
        }
    }

    fn encode_plain(&self, plain: &str, out: &mut Vec<u32>) {
        match self.pre {
            PreTokenizer::Qwen35 => {
                let norm = nfc(plain);
                for piece in split_qwen35(&norm) {
                    self.bpe(piece, out);
                }
            }
        }
    }

    /// Text to token ids.  For `qwen35`, plain text is NFC-normalized first, as the checkpoint's tokenizer.json does.
    /// With `special`, special-token strings in the text become their ids (a chat template's `<|im_start|>`); without it they are tokenized as the plain text they are.
    pub fn encode(&self, text: &str, special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let hit = if special {
                // the earliest special token in the rest, longest at a tie
                self.specials.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id)))
                    .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
            } else {
                None
            };
            let (plain, after) = match hit {
                Some((p, len, id)) => {
                    self.encode_plain(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + len..];
                    continue;
                }
                None => (rest, ""),
            };
            self.encode_plain(plain, &mut out);
            rest = after;
        }
        out
    }

    /// Token ids to bytes, special tokens as their text.  Bytes, not a String: a token can end inside a UTF-8
    /// character, and only the whole sequence is guaranteed to be valid.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let Some(t) = self.tokens.get(id as usize) else { continue };
            if matches!(self.token_type.get(id as usize), Some(3) | Some(4)) {
                out.extend(t.as_bytes());
            } else {
                out.extend(t.chars().filter_map(|c| self.char_byte.get(&c).copied()));
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_follows_the_pattern() {
        fn s(t: &str) -> Vec<&str> {
            split_qwen35(t)
        }
        assert_eq!(s("Hello world"), ["Hello", " world"]);
        assert_eq!(s("don't"), ["don", "'t"]);
        assert_eq!(s("I'LL go"), ["I", "'LL", " go"]);
        assert_eq!(s("12345"), ["1", "2", "3", "4", "5"]);
        assert_eq!(s("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(s("x  y"), ["x", " ", " y"], "the run gives its last space to the word");
        assert_eq!(s("x   "), ["x", "   "], "a run at the end stays whole");
        assert_eq!(s("hi!!\n\nok"), ["hi", "!!\n\n", "ok"]);
        assert_eq!(s(" ?!"), [" ?!"]);
        assert_eq!(s("fn(x)"), ["fn", "(x", ")"], "the prefix branch takes one non-letter before letters");
        assert_eq!(s("  \n  \n  x"), ["  \n  \n", " ", " x"]);
        assert_eq!(s("naïve café"), ["naïve", " café"]);
        assert_eq!(s("e\u{301}t\u{e9}"), ["e\u{301}t\u{e9}"], "combining marks stay with their letters");
        assert_eq!(s("日本語です"), ["日本語です"]);
        assert_eq!(s("x²"), ["x", "²"], "superscript two is \\p{N}");
        assert_eq!(s(""), Vec::<&str>::new());
        for t in ["Hello world", "  \n  \n  x", "a\tb\r\nc", "🙂 ok", "x=1;\n\ty += 2"] {
            assert_eq!(s(t).concat(), t, "pieces must cover the text exactly");
        }
    }

    #[test]
    fn byte_map_is_a_bijection() {
        let m = byte_chars();
        let set: std::collections::HashSet<char> = m.iter().copied().collect();
        assert_eq!(set.len(), 256);
        assert_eq!(m[b' ' as usize], 'Ġ');
        assert_eq!(m[b'\n' as usize], 'Ċ');
        assert_eq!(m[b'A' as usize], 'A');
    }

    #[test]
    fn bpe_on_a_toy_vocabulary() {
        let tokens: Vec<String> = ["a", "b", "c", "Ġ", "ab", "abc", "Ġa", "<|s|>"].iter().map(|s| s.to_string()).collect();
        let types = vec![1, 1, 1, 1, 1, 1, 1, 3];
        let merges: Vec<String> = ["a b", "ab c", "Ġ a"].iter().map(|s| s.to_string()).collect();
        let t = Tokenizer::new(tokens, types, &merges, None, None);
        assert_eq!(t.encode("abc", false), [5]);
        assert_eq!(t.encode("abc a", false), [5, 6]);
        assert_eq!(t.encode("ab<|s|>c", true), [4, 7, 2]);
        assert_eq!(t.decode(&[5, 6, 7]), "abc a<|s|>");
    }
}
