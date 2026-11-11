//! The DFlash 2 drafter's weights (`models/dflash2`, a `model.safetensors` of BF16 tensors, ~1.2 GB of VRAM) and
//! buffers.  The drafter is a 5-layer block-diffusion model: it reads features the main model produces (the mean
//! over the hyper-connection streams of layers 5, 15, 25, 35 and 45's outputs), keeps per-layer K/V for every
//! position, and drafts a block of BLOCK - 1 tokens in one pass: the anchor (the token about to be stepped) and
//! BLOCK - 1 mask tokens go through the layers together, the main model's head gives each mask slot 16 candidates, and
//! a low-rank bilinear selector walks the best path through them.  The layer itself runs in `session::dflash_layer`,
//! on the session's kernels.  Follows SGLang's `dflash.py` (DFlash2DraftModel) and `worker.py`.

use crate::driver::{Context, DeviceBuffer};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const HIDDEN: usize = 2560;
pub const BLOCK: usize = 8;
pub const LAYERS: usize = 5;
pub const TAPS: [usize; LAYERS] = [5, 15, 25, 35, 45];
pub const NFEAT: usize = TAPS.len() * HIDDEN;
pub const N_HEAD: usize = 32;
pub const N_HEAD_KV: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const FF: usize = 7680;
pub const KV_WIDTH: usize = N_HEAD_KV * HEAD_DIM;
pub const CONV_GROUP: usize = 16;
pub const N_GROUPS: usize = HIDDEN / CONV_GROUP;
pub const RANK: usize = 256;
pub const TOP_K: usize = 16;
pub const WINDOW: usize = 2048;
/// rows of the drafter's feature and context buffers: a long pass's last WINDOW positions
pub const DF_ROWS: usize = WINDOW;
/// K/V slots kept per layer (a position p lives in slot p % RING): the window and a block.
pub const RING: usize = WINDOW + 2 * BLOCK;
pub const MASK_TOKEN: u32 = 248077;
pub const ROPE_THETA: f32 = 10000.0;
pub const EPS: f32 = 1e-6;

#[derive(Debug, Clone)]
pub struct DfTensor {
    /// byte offset in the weights buffer
    pub offset: usize,
    /// stored as Q8_0 blocks (34 bytes a 32 weights) instead of BF16
    pub q8: bool,
    pub rows: usize,
    pub cols: usize,
}

/// The drafter's activations (BLOCK rows each unless said).
pub struct DflashBufs<'c> {
    /// the main model's features for up to `feat_rows` positions (NFEAT floats each)
    pub feat: DeviceBuffer<'c>,
    pub feat_rows: usize,
    /// fc output / hidden_norm (feat_rows x HIDDEN)
    pub ctxh: DeviceBuffer<'c>,
    /// context K/V projections for up to feat_rows positions
    pub kp: DeviceBuffer<'c>,
    pub vp: DeviceBuffer<'c>,
    /// block activations
    pub emb: DeviceBuffer<'c>,
    pub res: DeviceBuffer<'c>,
    pub xn: DeviceBuffer<'c>,
    pub xc: DeviceBuffer<'c>,
    pub delta_a: DeviceBuffer<'c>,
    pub delta_m: DeviceBuffer<'c>,
    pub q: DeviceBuffer<'c>,
    pub ao: DeviceBuffer<'c>,
    pub y: DeviceBuffer<'c>,
    pub y2: DeviceBuffer<'c>,
    pub g: DeviceBuffer<'c>,
    pub u: DeviceBuffer<'c>,
    /// final hidden (BLOCK x HIDDEN)
    pub hid: DeviceBuffer<'c>,
    pub hp: DeviceBuffer<'c>,
    pub cand: DeviceBuffer<'c>,
    pub unary: DeviceBuffer<'c>,
    pub out: DeviceBuffer<'c>,
}

pub struct DflashModel<'c> {
    pub weights: DeviceBuffer<'c>,
    pub tensors: HashMap<String, DfTensor>,
    pub bufs: DflashBufs<'c>,
    /// per layer: K and V rings (RING x KV_WIDTH floats)
    pub kring: Vec<DeviceBuffer<'c>>,
    pub vring: Vec<DeviceBuffer<'c>>,
    /// positions 0 .. valid have K/V from the main model's features
    pub valid: usize,
    /// the lowest position whose ring slot holds its own K/V: positions `lo .. valid` are usable (0 in a conversation read from the
    /// start; after a restore, the part of the old session's ring that the old session had not overwritten, or the restore point
    /// itself when the old session was another conversation: STATUS 98)
    pub lo: usize,
    /// the old behaviour, `SHADOW_DF_KEEP=0`: the session was restored from a snapshot the drafter has no features for: no
    /// drafts until a conversation starts from position 0 again
    pub broken: bool,
    /// the token ids the candidates are picked from (`draft_vocab.bin` beside the weights: the main model's likeliest
    /// ~43% of the vocabulary), on the device; None: the whole vocabulary
    pub draft_ids: Option<(DeviceBuffer<'c>, usize)>,
    /// a chain stops after a draft the selector gives less than this probability
    pub min_p: f32,
    /// `SHADOW_DF_TIME=1`: seconds and rounds per phase of the draft round (synchronizes between phases)
    pub prof: Option<std::collections::BTreeMap<&'static str, (f64, u64)>>,
    /// the mask token's embedding row (set when the session first drafts)
    pub mask_emb: Option<Vec<f32>>,
    /// rounds drafted, drafted tokens, tokens the main model agreed with (the speculation loop)
    pub stats: (u64, u64, u64),
}

/// BF16 weights (little-endian u16, rows of 32-multiples) as GGML Q8_0 blocks: an f16 scale then 32 int8, x = d * q.
fn quantize_q8_0(bf16: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bf16.len() / 64 * 34);
    for blk in bf16.chunks_exact(64) {
        let x: Vec<f32> = blk.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&shadow_model::half::f32_to_f16(d).to_le_bytes());
        out.extend(x.iter().map(|v| (v * id).round().clamp(-127.0, 127.0) as i8 as u8));
    }
    out
}

/// A tiny reader for the safetensors header: {"name": {"dtype": "BF16", "shape": [r, c], "data_offsets": [s, e]}, ...}.
struct Json<'a> {
    s: &'a [u8],
    at: usize,
}

impl Json<'_> {
    fn ws(&mut self) {
        while self.at < self.s.len() && self.s[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.at) == Some(&c) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("safetensors header: expected {:?} at {}", c as char, self.at))
        }
    }
    fn peek(&mut self) -> u8 {
        self.ws();
        self.s.get(self.at).copied().unwrap_or(0)
    }
    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let start = self.at;
        while self.at < self.s.len() && self.s[self.at] != b'"' {
            self.at += 1;
        }
        let out = String::from_utf8_lossy(&self.s[start..self.at]).to_string();
        self.eat(b'"')?;
        Ok(out)
    }
    fn number(&mut self) -> Result<usize, String> {
        self.ws();
        let start = self.at;
        while self.at < self.s.len() && self.s[self.at].is_ascii_digit() {
            self.at += 1;
        }
        std::str::from_utf8(&self.s[start..self.at]).unwrap().parse().map_err(|_| "safetensors header: bad number".to_string())
    }
    fn numbers(&mut self) -> Result<Vec<usize>, String> {
        self.eat(b'[')?;
        let mut v = vec![];
        while self.peek() != b']' {
            v.push(self.number()?);
            if self.peek() == b',' {
                self.at += 1;
            }
        }
        self.eat(b']')?;
        Ok(v)
    }
    /// Skips a value that is not an object of tensors (the "__metadata__" entry).
    fn skip(&mut self) -> Result<(), String> {
        let mut depth = 0i32;
        loop {
            match self.peek() {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b'"' => {
                    self.string()?;
                    if depth == 0 {
                        return Ok(());
                    }
                    continue;
                }
                0 => return Err("safetensors header: unexpected end".into()),
                _ => {}
            }
            self.at += 1;
            if depth == 0 {
                return Ok(());
            }
        }
    }
}

/// (name, shape, byte range in the data section) for each BF16 tensor of a safetensors file.
fn parse_header(h: &[u8]) -> Result<Vec<(String, Vec<usize>, (usize, usize))>, String> {
    let mut j = Json { s: h, at: 0 };
    let mut out = vec![];
    j.eat(b'{')?;
    while j.peek() != b'}' {
        let name = j.string()?;
        j.eat(b':')?;
        if name == "__metadata__" {
            j.skip()?;
        } else {
            j.eat(b'{')?;
            let (mut shape, mut range, mut dtype) = (vec![], (0, 0), String::new());
            while j.peek() != b'}' {
                let key = j.string()?;
                j.eat(b':')?;
                match key.as_str() {
                    "dtype" => dtype = j.string()?,
                    "shape" => shape = j.numbers()?,
                    "data_offsets" => {
                        let r = j.numbers()?;
                        range = (r[0], r[1]);
                    }
                    _ => j.skip()?,
                }
                if j.peek() == b',' {
                    j.at += 1;
                }
            }
            j.eat(b'}')?;
            if dtype != "BF16" {
                return Err(format!("dflash: tensor {name} is {dtype}, expected BF16"));
            }
            out.push((name, shape, range));
        }
        if j.peek() == b',' {
            j.at += 1;
        }
    }
    Ok(out)
}

impl<'c> DflashModel<'c> {
    /// `q8`: the matrices (not the norms, convolution bases or selector codebooks) as Q8_0, half the VRAM.
    pub fn load(ctx: &'c Context, dir: &Path, cap: usize, q8: bool) -> Result<Self, String> {
        let e = |x: crate::driver::CudaError| x.to_string();
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::open(&path).map_err(|err| format!("{}: {err}", path.display()))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|err| err.to_string())?;
        let hlen = u64::from_le_bytes(len) as usize;
        let mut head = vec![0u8; hlen];
        f.read_exact(&mut head).map_err(|err| err.to_string())?;
        let entries = parse_header(&head)?;
        let data0 = 8 + hlen as u64;
        let quantize = |name: &str, shape: &[usize]| q8 && shape.len() == 2 && shape[1] % 32 == 0 && name.ends_with(".weight")
            && !name.ends_with("norm.weight");
        let stored = |name: &str, shape: &[usize], bytes: usize| if quantize(name, shape) { bytes / 64 * 34 } else { bytes };
        let total: usize = entries.iter().map(|(n, s, r)| stored(n, s, r.1 - r.0).next_multiple_of(256)).sum();
        let weights = ctx.alloc(total).map_err(e)?;
        let mut tensors = HashMap::new();
        let mut at = 0usize;
        let mut buf = vec![];
        for (name, shape, (s, en)) in entries {
            buf.resize(en - s, 0u8);
            f.seek(SeekFrom::Start(data0 + s as u64)).map_err(|err| err.to_string())?;
            f.read_exact(&mut buf).map_err(|err| err.to_string())?;
            let is_q8 = quantize(&name, &shape);
            if is_q8 {
                buf = quantize_q8_0(&buf);
            }
            // SAFETY: [at, at + buf.len()) is inside the buffer allocated for the sum of the padded tensor sizes.
            let r = unsafe {
                (ctx.api.cuMemcpyHtoD)(weights.ptr + at as u64, buf.as_ptr() as *const std::ffi::c_void, buf.len())
            };
            if r != 0 {
                return Err(format!("dflash: upload of {name} failed ({r})"));
            }
            let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, shape.iter().product()) };
            tensors.insert(name, DfTensor { offset: at, rows, cols, q8: is_q8 });
            at += buf.len().next_multiple_of(256);
        }
        for need in ["fc.weight", "hidden_norm.weight", "norm.weight", "candidate_selector.predecessor_codebook"] {
            if !tensors.contains_key(need) {
                return Err(format!("dflash: {} has no tensor {need}", path.display()));
            }
        }
        let a = |n: usize| ctx.alloc(n.max(1) * 4).map_err(e);
        // the drafter looks back WINDOW positions: a pass of more rows than that gives it the last DF_ROWS only (`df_capture`), whatever the pass's size
        let fr = DF_ROWS;
        let bufs = DflashBufs {
            feat: a(fr * NFEAT)?,
            feat_rows: fr,
            ctxh: a(fr * HIDDEN)?,
            kp: a(fr * KV_WIDTH)?,
            vp: a(fr * KV_WIDTH)?,
            emb: a(BLOCK * HIDDEN)?,
            res: a(BLOCK * HIDDEN)?,
            xn: a(BLOCK * HIDDEN)?,
            xc: a(BLOCK * HIDDEN)?,
            delta_a: a(BLOCK * 4 * N_GROUPS)?,
            delta_m: a(BLOCK * 4 * N_GROUPS)?,
            q: a(BLOCK * N_HEAD * HEAD_DIM)?,
            ao: a(BLOCK * N_HEAD * HEAD_DIM)?,
            y: a(BLOCK * HIDDEN)?,
            y2: a(BLOCK * HIDDEN)?,
            g: a(BLOCK * FF)?,
            u: a(BLOCK * FF)?,
            hid: a(BLOCK * HIDDEN)?,
            hp: a(BLOCK * RANK)?,
            cand: a(BLOCK * TOP_K)?,
            unary: a(BLOCK * TOP_K)?,
            out: a(2 * BLOCK)?,
        };
        let _ = cap;
        let draft_ids = match std::fs::read(dir.join("draft_vocab.bin")) {
            Ok(vb) if std::env::var("SHADOW_DF_FULL_HEAD").is_err_and(|_| true) => {
                let ids: Vec<u8> = vb.chunks_exact(4).flat_map(|c| c.to_vec()).collect();
                let d = ctx.alloc(ids.len()).map_err(e)?;
                d.upload(&ids).map_err(e)?;
                Some((d, vb.len() / 4))
            }
            _ => None,
        };
        let ring = |_| ctx.alloc(RING * KV_WIDTH * 4).map_err(e);
        let kring = (0..LAYERS).map(ring).collect::<Result<Vec<_>, _>>()?;
        let vring = (0..LAYERS).map(ring).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { weights, tensors, bufs, kring, vring, valid: 0, lo: 0, broken: false, draft_ids, min_p: 0.0, prof: std::env::var("SHADOW_DF_TIME").is_ok_and(|v| v == "1").then(Default::default), mask_emb: None, stats: (0, 0, 0) })
    }

    pub fn ptr(&self, name: &str) -> u64 {
        let t = self.tensors.get(name).unwrap_or_else(|| panic!("dflash: no tensor {name}"));
        self.weights.ptr + t.offset as u64
    }

    pub fn dims(&self, name: &str) -> (usize, usize) {
        let t = &self.tensors[name];
        (t.rows, t.cols)
    }
}
