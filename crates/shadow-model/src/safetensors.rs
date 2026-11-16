//! A reader for safetensors files (the format of Hugging Face model repos): an 8-byte little-endian header length, a JSON
//! header `{name: {dtype, shape, data_offsets: [start, end]}, "__metadata__": {...}}`, then the tensor bytes.  Written
//! here, like the GGUF reader, so the model crate stays free of outside dependencies; the header's JSON is tiny and
//! regular, so a small parser is enough.  Used for EXL3 models (`exl3.rs`), whose repos are safetensors.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct StTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Index into `SafeTensors::files`.
    pub file: usize,
    /// Absolute byte range in that file.
    pub start: u64,
    pub end: u64,
}

impl StTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Every tensor of one or more safetensors files, by name.
#[derive(Debug, Default)]
pub struct SafeTensors {
    pub files: Vec<PathBuf>,
    pub tensors: BTreeMap<String, StTensor>,
}

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl SafeTensors {
    /// Reads the headers of `paths` (a name in two files is an error).
    pub fn open(paths: &[PathBuf]) -> Result<SafeTensors, Error> {
        let mut st = SafeTensors::default();
        for p in paths {
            st.add_file(p)?;
        }
        Ok(st)
    }

    /// All `*.safetensors` files of a directory, in name order.
    pub fn open_dir(dir: &Path) -> Result<SafeTensors, Error> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| Error(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        SafeTensors::open(&paths)
    }

    pub fn add_file(&mut self, path: &Path) -> Result<(), Error> {
        let err = |m: String| Error(format!("{}: {m}", path.display()));
        let mut f = std::fs::File::open(path).map_err(|e| err(e.to_string()))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| err(format!("header length: {e}")))?;
        let hlen = u64::from_le_bytes(len);
        if hlen > 1 << 30 {
            return Err(err(format!("header of {hlen} bytes is not a safetensors header")));
        }
        let mut h = vec![0u8; hlen as usize];
        f.read_exact(&mut h).map_err(|e| err(format!("header: {e}")))?;
        let text = std::str::from_utf8(&h).map_err(|e| err(format!("header is not UTF-8: {e}")))?;
        let file = self.files.len();
        let data0 = 8 + hlen;
        let size = f.metadata().map_err(|e| err(e.to_string()))?.len();
        for (name, dtype, shape, s, e) in parse_header(text).map_err(err)? {
            if data0 + e > size {
                return Err(err(format!("tensor {name} ends at {} but the file has {size} bytes", data0 + e)));
            }
            let t = StTensor { dtype, shape, file, start: data0 + s, end: data0 + e };
            if self.tensors.insert(name.clone(), t).is_some() {
                return Err(err(format!("tensor {name} appears twice")));
            }
        }
        self.files.push(path.to_path_buf());
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&StTensor> {
        self.tensors.get(name)
    }

    /// Takes the tensors of `other`'s `model-*.safetensors` files whose names `keep` accepts in place of the ones of the same name here (a mix of two quants of the same weights,
    /// STATUS 165); returns how many.  The other files are appended to `files`, so every index already handed out stays valid.
    pub fn override_from(&mut self, other: &Path, keep: impl Fn(&str) -> bool) -> Result<usize, Error> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(other)
            .map_err(|e| Error(format!("{}: {e}", other.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors")))
            .collect();
        paths.sort();
        let o = SafeTensors::open(&paths)?;
        let base = self.files.len();
        self.files.extend(o.files.iter().cloned());
        let mut n = 0;
        for (name, t) in o.tensors {
            if keep(&name) && self.tensors.contains_key(&name) {
                self.tensors.insert(name, StTensor { file: t.file + base, ..t });
                n += 1;
            }
        }
        Ok(n)
    }

    /// The raw bytes of a tensor.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        let t = self.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        self.read_range(t, 0, t.end - t.start)
    }

    /// `len` bytes from byte `off` inside a tensor.
    pub fn read_range(&self, t: &StTensor, off: u64, len: u64) -> Result<Vec<u8>, Error> {
        if off + len > t.end - t.start {
            return Err(Error(format!("read of {len} bytes at {off} is outside a tensor of {} bytes", t.end - t.start)));
        }
        let p = &self.files[t.file];
        let mut f = std::fs::File::open(p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        f.seek(SeekFrom::Start(t.start + off)).map_err(|e| Error(e.to_string()))?;
        let mut b = vec![0u8; len as usize];
        f.read_exact(&mut b).map_err(|e| Error(format!("{}: {e}", p.display())))?;
        Ok(b)
    }

    /// A tensor as little-endian u16 words (I16 / F16 / BF16 storage).
    pub fn read_u16(&self, name: &str) -> Result<Vec<u16>, Error> {
        let b = self.read(name)?;
        Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    /// An F16, BF16 or F32 tensor as f32.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let t = self.get(name).ok_or_else(|| Error(format!("no tensor {name}")))?;
        let b = self.read(name)?;
        match t.dtype.as_str() {
            "F32" => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            "F16" => Ok(b.chunks_exact(2).map(|c| crate::half::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            "BF16" => Ok(b.chunks_exact(2).map(|c| crate::half::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            d => Err(Error(format!("{name}: {d} is not a float type read_f32 knows"))),
        }
    }
}

// ---- the header's JSON ------------------------------------------------------------------------------------------

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let h = std::str::from_utf8(self.s.get(self.i..self.i + 4).ok_or("short \\u escape")?).map_err(|e| e.to_string())?;
                            let cp = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                            self.i += 4;
                            let ch = char::from_u32(cp).unwrap_or('\u{fffd}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(format!("bad escape \\{}", e as char)),
                    }
                }
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|e| e.to_string())
    }

    fn number(&mut self) -> Result<u64, String> {
        self.ws();
        let a = self.i;
        while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
            self.i += 1;
        }
        std::str::from_utf8(&self.s[a..self.i]).unwrap().parse::<u64>().map_err(|e| format!("number at byte {a}: {e}"))
    }

    /// Skips any JSON value (the `__metadata__` object, unknown keys).
    fn skip(&mut self) -> Result<(), String> {
        match self.peek().ok_or("unexpected end")? {
            b'"' => self.string().map(|_| ()),
            b'{' => {
                self.eat(b'{')?;
                if self.peek() == Some(b'}') {
                    return self.eat(b'}');
                }
                loop {
                    self.string()?;
                    self.eat(b':')?;
                    self.skip()?;
                    if self.peek() == Some(b',') {
                        self.eat(b',')?;
                    } else {
                        return self.eat(b'}');
                    }
                }
            }
            b'[' => {
                self.eat(b'[')?;
                if self.peek() == Some(b']') {
                    return self.eat(b']');
                }
                loop {
                    self.skip()?;
                    if self.peek() == Some(b',') {
                        self.eat(b',')?;
                    } else {
                        return self.eat(b']');
                    }
                }
            }
            _ => {
                while self.i < self.s.len() && !matches!(self.s[self.i], b',' | b'}' | b']') && !self.s[self.i].is_ascii_whitespace() {
                    self.i += 1;
                }
                Ok(())
            }
        }
    }

    fn num_array(&mut self) -> Result<Vec<u64>, String> {
        self.eat(b'[')?;
        let mut v = Vec::new();
        if self.peek() == Some(b']') {
            self.eat(b']')?;
            return Ok(v);
        }
        loop {
            v.push(self.number()?);
            if self.peek() == Some(b',') {
                self.eat(b',')?;
            } else {
                self.eat(b']')?;
                return Ok(v);
            }
        }
    }
}

/// (name, dtype, shape, start, end) of every tensor in a header; offsets are relative to the start of the data.
type HeaderTensor = (String, String, Vec<usize>, u64, u64);

fn parse_header(text: &str) -> Result<Vec<HeaderTensor>, String> {
    let mut p = P { s: text.as_bytes(), i: 0 };
    let mut out = Vec::new();
    p.eat(b'{')?;
    if p.peek() == Some(b'}') {
        return Ok(out);
    }
    loop {
        let name = p.string()?;
        p.eat(b':')?;
        if name == "__metadata__" {
            p.skip()?;
        } else {
            p.eat(b'{')?;
            let (mut dtype, mut shape, mut offs) = (None, None, None);
            loop {
                let key = p.string()?;
                p.eat(b':')?;
                match key.as_str() {
                    "dtype" => dtype = Some(p.string()?),
                    "shape" => shape = Some(p.num_array()?),
                    "data_offsets" => offs = Some(p.num_array()?),
                    _ => p.skip()?,
                }
                if p.peek() == Some(b',') {
                    p.eat(b',')?;
                } else {
                    p.eat(b'}')?;
                    break;
                }
            }
            let (dtype, shape, offs) = (dtype.ok_or(format!("{name}: no dtype"))?, shape.ok_or(format!("{name}: no shape"))?, offs.ok_or(format!("{name}: no data_offsets"))?);
            if offs.len() != 2 || offs[1] < offs[0] {
                return Err(format!("{name}: bad data_offsets {offs:?}"));
            }
            out.push((name, dtype, shape.into_iter().map(|d| d as usize).collect(), offs[0], offs[1]));
        }
        if p.peek() == Some(b',') {
            p.eat(b',')?;
        } else {
            p.eat(b'}')?;
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_with_metadata_scalars_and_escapes() {
        let h = r#"{"__metadata__":{"format":"pt"},"a.b":{"dtype":"I16","shape":[2,3],"data_offsets":[0,12]},"s\"x":{"dtype":"I32","shape":[],"data_offsets":[12,16]}}"#;
        let t = parse_header(h).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], ("a.b".to_string(), "I16".to_string(), vec![2, 3], 0, 12));
        assert_eq!(t[1], ("s\"x".to_string(), "I32".to_string(), vec![], 12, 16));
    }

    #[test]
    fn a_broken_header_is_an_error_not_a_panic() {
        assert!(parse_header("{\"a\":{\"dtype\":\"F16\"}}").is_err());
        assert!(parse_header("{\"a\":").is_err());
        assert!(parse_header("").is_err());
    }
}
