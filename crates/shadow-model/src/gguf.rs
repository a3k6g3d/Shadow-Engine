//! A GGUF reader of our own: the header, the key-values and the tensor table, nothing else.
//!
//! It never maps or reads tensor data.  Where each byte of a weight lives is the tier's decision (docs/SHADOW_PLAN.md,
//! stage 3), so this module only says *where in which file* a tensor is and how many bytes it takes.  A split model
//! (`-00001-of-00002.gguf`) is opened from any one shard and the others are found beside it by name.
//!
//! Format, version 3: magic `GGUF`, u32 version, u64 tensor count, u64 key-value count, the key-values, the tensor
//! infos (name, dims, type, offset relative to the data section), then the data section, aligned to
//! `general.alignment` (default 32).  All little-endian.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 4] = b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// A key-value's value.  Arrays keep their element type, because a 248,320-entry token list and a 248,320-entry
/// score list are read very differently downstream.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    /// Any integer type, as u64, when it is non-negative.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// A short rendering for `shadow inspect`: arrays print their length and first few elements, not all of them.
    pub fn brief(&self) -> String {
        match self {
            Value::Str(s) if s.len() > 80 => format!("{:?}... ({} bytes)", &s[..s.floor_char_boundary(80)], s.len()),
            Value::Str(s) => format!("{s:?}"),
            Value::Array(a) => {
                let head: Vec<String> = a.iter().take(4).map(|v| v.brief()).collect();
                format!("[{}{}] ({} items)", head.join(", "), if a.len() > 4 { ", ..." } else { "" }, a.len())
            }
            v => format!("{v:?}").split(['(', ')']).nth(1).unwrap_or("?").to_string(),
        }
    }
}

/// ggml's tensor types, by their on-disk id.  Only the ones whose block layout is fixed are listed with a size; an id
/// this table does not know is kept as `Unknown(id)` and its size is taken from the offsets instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GgmlType {
    Known(u32),
    Unknown(u32),
}

/// (id, name, elements per block, bytes per block)
const TYPES: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98),
    (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 56),
    (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54),
    (35, "TQ2_0", 256, 66),
    (39, "MXFP4", 32, 17),
];

impl GgmlType {
    pub fn from_id(id: u32) -> GgmlType {
        if TYPES.iter().any(|t| t.0 == id) { GgmlType::Known(id) } else { GgmlType::Unknown(id) }
    }

    pub fn id(self) -> u32 {
        match self {
            GgmlType::Known(i) | GgmlType::Unknown(i) => i,
        }
    }

    fn entry(self) -> Option<&'static (u32, &'static str, u64, u64)> {
        match self {
            GgmlType::Known(i) => TYPES.iter().find(|t| t.0 == i),
            GgmlType::Unknown(_) => None,
        }
    }

    pub fn name(self) -> String {
        self.entry().map(|e| e.1.to_string()).unwrap_or_else(|| format!("type{}", self.id()))
    }

    /// (elements per block, bytes per block), when the layout is known.
    pub fn block(self) -> Option<(u64, u64)> {
        self.entry().map(|e| (e.2, e.3))
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// One tensor: where it is and what it is.  `offset` is absolute in `shard`'s file, not relative to the data section.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the fastest-moving (the row length).
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub shard: usize,
    pub offset: u64,
    /// The byte size from the type's block layout, or None when the type or the shape does not allow it.
    pub nbytes_layout: Option<u64>,
    /// The byte size from the gap to the next tensor in the same shard (padding included), or to the end of the file.
    pub nbytes_span: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// The size this tensor takes: the layout's when it is known, else the span's.
    pub fn nbytes(&self) -> u64 {
        self.nbytes_layout.unwrap_or(self.nbytes_span)
    }
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub path: PathBuf,
    pub version: u32,
    pub file_len: u64,
    pub data_start: u64,
    pub alignment: u64,
}

/// A whole model: every shard's key-values (shard 0's are the model's; the others' are kept for the split keys) and
/// one tensor table across all shards.
#[derive(Debug, Clone)]
pub struct Gguf {
    pub shards: Vec<Shard>,
    pub kv: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
}

#[derive(Debug)]
pub enum Error {
    Io(PathBuf, io::Error),
    Format(PathBuf, String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Error::Format(p, m) => write!(f, "{}: not a GGUF this reader understands: {m}", p.display()),
        }
    }
}

impl std::error::Error for Error {}

struct R<'a> {
    r: BufReader<File>,
    path: &'a Path,
}

// Limits that stop a corrupt header from asking for absurd allocations.  Generous: this model's vocabulary is 248,320
// strings and its chat template 9,993 bytes.
const MAX_STR: u64 = 64 << 20;
const MAX_ARRAY: u64 = 1 << 26;
const MAX_DIMS: u32 = 8;

impl R<'_> {
    fn io(&self, e: io::Error) -> Error {
        Error::Io(self.path.to_path_buf(), e)
    }
    fn bad(&self, m: impl Into<String>) -> Error {
        Error::Format(self.path.to_path_buf(), m.into())
    }
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b).map_err(|e| self.io(e))?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String, Error> {
        let n = self.u64()?;
        if n > MAX_STR {
            return Err(self.bad(format!("a string of {n} bytes")));
        }
        let mut b = vec![0u8; n as usize];
        self.r.read_exact(&mut b).map_err(|e| self.io(e))?;
        // Token strings are not all valid UTF-8 on every model; keep them lossless enough to see, and let the tokenizer
        // read the raw bytes through its own path if it ever needs to.
        Ok(String::from_utf8(b).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, Error> {
        Ok(match ty {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(self.bytes::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(self.bad("arrays nested more than two deep"));
                }
                let et = self.u32()?;
                let n = self.u64()?;
                if n > MAX_ARRAY {
                    return Err(self.bad(format!("an array of {n} items")));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            t => return Err(self.bad(format!("value type {t}"))),
        })
    }
}

type ShardHeader = (Shard, BTreeMap<String, Value>, Vec<TensorInfo>);

/// One shard's header: its key-values and its tensors, offsets made absolute.
fn read_shard(path: &Path, index: usize) -> Result<ShardHeader, Error> {
    let f = File::open(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
    let file_len = f.metadata().map_err(|e| Error::Io(path.to_path_buf(), e))?.len();
    let mut r = R { r: BufReader::with_capacity(1 << 20, f), path };
    if &r.bytes::<4>()? != MAGIC {
        return Err(r.bad("no GGUF magic"));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(r.bad(format!("version {version} (2 and 3 are read)")));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > 1 << 20 || n_kv > 1 << 20 {
        return Err(r.bad(format!("{n_tensors} tensors, {n_kv} key-values")));
    }
    let mut kv = BTreeMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        let v = r.value(ty, 0)?;
        kv.insert(key, v);
    }
    let mut raw = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        if nd == 0 || nd > MAX_DIMS {
            return Err(r.bad(format!("tensor {name} has {nd} dimensions")));
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(r.u64()?);
        }
        let ty = GgmlType::from_id(r.u32()?);
        let off = r.u64()?;
        raw.push((name, dims, ty, off));
    }
    let alignment = kv.get("general.alignment").and_then(Value::as_u64).unwrap_or(DEFAULT_ALIGNMENT);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(r.bad(format!("general.alignment {alignment}")));
    }
    let header_end = r.r.stream_position().map_err(|e| r.io(e))?;
    let data_start = header_end.div_ceil(alignment) * alignment;

    // Spans: sort by offset, each tensor runs to the next one's start (or the end of the file).
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by_key(|&i| raw[i].3);
    let mut span = vec![0u64; raw.len()];
    for (k, &i) in order.iter().enumerate() {
        let start = data_start + raw[i].3;
        let end = order.get(k + 1).map(|&j| data_start + raw[j].3).unwrap_or(file_len);
        if start > file_len || end < start {
            return Err(r.bad(format!("tensor {} at {start} is outside the file ({file_len} bytes)", raw[i].0)));
        }
        span[i] = end - start;
    }
    let tensors = raw
        .into_iter()
        .enumerate()
        .map(|(i, (name, dims, ty, off))| {
            let elements: u64 = dims.iter().product();
            let nbytes_layout = ty.block().and_then(|(be, bb)| {
                (dims[0] % be == 0).then(|| elements / be * bb)
            });
            TensorInfo { name, dims, ty, shard: index, offset: data_start + off, nbytes_layout, nbytes_span: span[i] }
        })
        .collect::<Vec<_>>();
    for t in &tensors {
        if let Some(n) = t.nbytes_layout {
            if n > t.nbytes_span {
                return Err(r.bad(format!("tensor {} needs {n} bytes but has {}", t.name, t.nbytes_span)));
            }
        }
    }
    Ok((Shard { path: path.to_path_buf(), version, file_len, data_start, alignment }, kv, tensors))
}

/// The shard file names of a split model, from any one of them: `X-00001-of-00002.gguf` gives both.  A path that
/// does not follow the pattern is a single-file model.
pub fn shard_paths(path: &Path) -> Vec<PathBuf> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parse = || -> Option<(String, usize, usize)> {
        let stem = name.strip_suffix(".gguf")?;
        let (head, total) = stem.rsplit_once("-of-")?;
        let (base, _idx) = head.rsplit_once('-')?;
        let width = total.len();
        Some((base.to_string(), total.parse().ok()?, width))
    };
    match parse() {
        Some((base, total, width)) if total > 0 => (1..=total)
            .map(|i| path.with_file_name(format!("{base}-{i:0width$}-of-{total:0width$}.gguf")))
            .collect(),
        _ => vec![path.to_path_buf()],
    }
}

impl Gguf {
    /// Opens a model from any of its shards.
    pub fn open(path: &Path) -> Result<Gguf, Error> {
        let paths = shard_paths(path);
        let mut shards = Vec::new();
        let mut kv = BTreeMap::new();
        let mut tensors = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            let (s, k, t) = read_shard(p, i)?;
            if i == 0 {
                kv = k;
            } else {
                for (key, v) in k {
                    kv.entry(key).or_insert(v);
                }
            }
            shards.push(s);
            tensors.extend(t);
        }
        if let Some(n) = kv.get("split.count").and_then(Value::as_u64) {
            if n as usize != shards.len() {
                return Err(Error::Format(path.to_path_buf(), format!("split.count {n}, found {} shards", shards.len())));
            }
        }
        if let Some(n) = kv.get("split.tensors.count").and_then(Value::as_u64) {
            if n as usize != tensors.len() {
                return Err(Error::Format(path.to_path_buf(), format!("split.tensors.count {n}, read {}", tensors.len())));
            }
        }
        Ok(Gguf { shards, kv, tensors })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    /// `<arch>.<key>`, the way per-architecture keys are stored.
    pub fn arch_key(&self, key: &str) -> Option<&Value> {
        let arch = self.get("general.architecture")?.as_str()?;
        self.get(&format!("{arch}.{key}"))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The sum of every tensor's size.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(TensorInfo::nbytes).sum()
    }

    /// Reads one tensor's bytes.  Only for small tensors and tests: the engine reads weights through the I/O ring.
    pub fn read_tensor(&self, t: &TensorInfo) -> Result<Vec<u8>, Error> {
        let p = &self.shards[t.shard].path;
        let mut f = File::open(p).map_err(|e| Error::Io(p.clone(), e))?;
        f.seek(SeekFrom::Start(t.offset)).map_err(|e| Error::Io(p.clone(), e))?;
        let mut b = vec![0u8; t.nbytes() as usize];
        f.read_exact(&mut b).map_err(|e| Error::Io(p.clone(), e))?;
        Ok(b)
    }
}

/// The layer index in a `blk.N.` tensor name.
pub fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }

    /// A two-tensor GGUF written by hand, so the reader is checked against the format and not against itself.
    fn tiny(path: &Path) {
        let mut b = Vec::new();
        b.extend(MAGIC);
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes()); // tensors
        b.extend(3u64.to_le_bytes()); // kv
        put_str(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, "toy");
        put_str(&mut b, "toy.block_count");
        b.extend(4u32.to_le_bytes());
        b.extend(2u32.to_le_bytes());
        put_str(&mut b, "tokenizer.ggml.tokens");
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        put_str(&mut b, "a");
        put_str(&mut b, "bc");
        // blk.0.w: 32 x 2 F32 = 256 bytes at 0; blk.1.q: 256 x 1 IQ3_S = 110 bytes at 256
        put_str(&mut b, "blk.0.w");
        b.extend(2u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        put_str(&mut b, "blk.1.q");
        b.extend(1u32.to_le_bytes());
        b.extend(256u64.to_le_bytes());
        b.extend(21u32.to_le_bytes());
        b.extend(256u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat_n(7u8, 256 + 110));
        File::create(path).unwrap().write_all(&b).unwrap();
    }

    #[test]
    fn reads_a_hand_written_file() {
        let dir = std::env::temp_dir().join(format!("shadow-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("toy.gguf");
        tiny(&p);
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.get("general.architecture").and_then(Value::as_str), Some("toy"));
        assert_eq!(g.arch_key("block_count").and_then(Value::as_u64), Some(2));
        assert_eq!(g.get("tokenizer.ggml.tokens").and_then(Value::as_array).map(|a| a.len()), Some(2));
        let w = g.tensor("blk.0.w").unwrap();
        assert_eq!((w.dims.clone(), w.ty.name(), w.nbytes()), (vec![32, 2], "F32".to_string(), 256));
        let q = g.tensor("blk.1.q").unwrap();
        assert_eq!((q.ty.name(), q.nbytes_layout, q.nbytes_span), ("IQ3_S".to_string(), Some(110), 110));
        assert_eq!(g.tensor_bytes(), 366);
        assert_eq!(g.read_tensor(q).unwrap(), vec![7u8; 110]);
        assert_eq!(block_index("blk.1.q"), Some(1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_truncated_file() {
        let dir = std::env::temp_dir().join(format!("shadow-gguf-t-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("cut.gguf");
        tiny(&p);
        let full = std::fs::read(&p).unwrap();
        std::fs::write(&p, &full[..full.len() - 200]).unwrap();
        assert!(Gguf::open(&p).is_err(), "a tensor that runs past the end of the file must be refused");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn finds_the_other_shards_by_name() {
        let p = Path::new("m/x-IQ3_S-00002-of-00002.gguf");
        let v = shard_paths(p);
        assert_eq!(v, vec![PathBuf::from("m/x-IQ3_S-00001-of-00002.gguf"), PathBuf::from("m/x-IQ3_S-00002-of-00002.gguf")]);
        assert_eq!(shard_paths(Path::new("m/single.gguf")), vec![PathBuf::from("m/single.gguf")]);
    }
}
