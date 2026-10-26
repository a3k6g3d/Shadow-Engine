//! The smallest JSON reader this crate needs: one value per line of `eval/cases.jsonl`, the memory store, and the
//! embedder's reply.  No serde, no serde_json — the workspace has two packages in its lock file on purpose, and a
//! yardstick with a dependency tree is a yardstick that can break on an unrelated `cargo update`.
//!
//! It is a real parser (strings with escapes, numbers, nesting, null/bool), not a regex over quotes, because the
//! store contains whatever the last session typed into it — quotes, backslashes, emoji — and a line that does not
//! parse must not silently rank the wrong record.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(BTreeMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Num(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str_list(&self) -> Option<Vec<&str>> {
        self.as_arr().map(|v| v.iter().filter_map(|x| x.as_str()).collect())
    }
}

pub fn parse(src: &str) -> Result<Value, String> {
    let b: Vec<char> = src.chars().collect();
    let mut p = Parser { b: &b, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing text at byte {}", p.byte_of(p.i)));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [char],
    i: usize,
}

impl<'a> Parser<'a> {
    fn byte_of(&self, at: usize) -> usize {
        self.b[..at].iter().map(|c| c.len_utf8()).sum()
    }

    fn err(&self, msg: &str) -> String {
        format!("{msg} at byte {}", self.byte_of(self.i))
    }

    fn peek(&self) -> Option<char> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(c) if c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &str) -> bool {
        let w: Vec<char> = word.chars().collect();
        if self.b[self.i..].starts_with(&w[..]) {
            self.i += w.len();
            return true;
        }
        false
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Value::Str(self.string()?)),
            Some('t') if self.lit("true") => Ok(Value::Bool(true)),
            Some('f') if self.lit("false") => Ok(Value::Bool(false)),
            Some('n') if self.lit("null") => Ok(Value::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(self.err(&format!("unexpected {c:?}"))),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = BTreeMap::new();
        self.ws();
        if self.peek() == Some('}') {
            self.i += 1;
            return Ok(Value::Obj(out));
        }
        loop {
            self.ws();
            if self.peek() != Some('"') {
                return Err(self.err("expected a key"));
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(':') {
                return Err(self.err("expected `:`"));
            }
            self.i += 1;
            self.ws();
            out.insert(key, self.value()?);
            self.ws();
            match self.peek() {
                Some(',') => {
                    self.i += 1;
                }
                Some('}') => {
                    self.i += 1;
                    return Ok(Value::Obj(out));
                }
                _ => return Err(self.err("expected `,` or `}`")),
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Ok(Value::Arr(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(',') => {
                    self.i += 1;
                }
                Some(']') => {
                    self.i += 1;
                    return Ok(Value::Arr(out));
                }
                _ => return Err(self.err("expected `,` or `]`")),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.i += 1;
            if c == '"' {
                return Ok(out);
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let e = self.peek().ok_or_else(|| self.err("dangling escape"))?;
            self.i += 1;
            match e {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let hi = self.hex4()?;
                    let cp = if (0xD800..0xDC00).contains(&hi) {
                        if self.peek() == Some('\\') && self.b.get(self.i + 1) == Some(&'u') {
                            self.i += 2;
                            let lo = self.hex4()?;
                            if (0xDC00..0xE000).contains(&lo) {
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            }
                        } else {
                            hi
                        }
                    } else {
                        hi
                    };
                    out.push(char::from_u32(cp).ok_or_else(|| self.err("not a character"))?);
                }
                _ => return Err(self.err("unknown escape")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.peek().ok_or_else(|| self.err("short \\u escape"))?;
            let d = c.to_digit(16).ok_or_else(|| self.err("not a hex digit"))?;
            v = v * 16 + d;
            self.i += 1;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        if self.peek() == Some('-') {
            self.i += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
            self.i += 1;
        }
        let text: String = self.b[start..self.i].iter().collect();
        text.parse::<f64>().map(Value::Num).map_err(|_| self.err("not a number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_case_line_parses_to_the_fields_the_scorer_uses() {
        let v = parse(r#"{"id": "c01", "ask": "why \"quotes\"", "gold_sub": "x", "must_contain": ["abi", "link"]}"#).unwrap();
        assert_eq!(v.get("id").and_then(|x| x.as_str()), Some("c01"));
        assert_eq!(v.get("ask").and_then(|x| x.as_str()), Some(r#"why "quotes""#));
        assert_eq!(v.get("must_contain").and_then(|x| x.as_str_list()), Some(vec!["abi", "link"]));
    }

    #[test]
    fn a_store_line_parses_to_text_and_a_vector() {
        let v = parse(r#"{"text": "tab\there é \u00e9", "vec": [-0.042579, 1e3, 0.0]}"#).unwrap();
        assert_eq!(v.get("text").and_then(|x| x.as_str()), Some("tab\there é é"));
        let nums: Vec<f64> = v.get("vec").unwrap().as_arr().unwrap().iter().filter_map(Value::as_f64).collect();
        assert_eq!(nums, vec![-0.042579, 1000.0, 0.0]);
    }

    #[test]
    fn escapes_and_surrogates_survive_the_round_trip() {
        let v = parse(r#"{"k": "A\u00e9B😀C"}"#).unwrap();
        assert_eq!(v.get("k").and_then(|x| x.as_str()), Some("AéB😀C"));
    }

    #[test]
    fn a_broken_line_fails_loudly_instead_of_ranking_the_wrong_record() {
        assert!(parse(r#"{"id": "c01",}"#).is_err());
        assert!(parse(r#"{"id": c01}"#).is_err());
        assert!(parse("{}trailing").is_err());
        assert!(parse(r#"{"a": "unterminated}"#).is_err());
    }
}
