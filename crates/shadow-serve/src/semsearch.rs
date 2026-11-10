//! Search over the project's files and the session's tool outputs (docs/STATUS.md 78, 91): the `search` tool, by keywords (BM25, built in, ready at once, no helper process) and by
//! meaning (the embedder below), the two rankings fused; with the embedder missing or failing it answers by keywords alone.
//!
//! The text is cut into chunks (about 40 lines / 900 characters, 3 lines of overlap), each chunk is turned into a 384-number vector by the embedder the repo already has
//! (`bin/llama-embedding.exe` running `models/embed-minilm/all-MiniLM-L6-v2-Q8_0.gguf`, else `models/bge-small-en-v1.5-q8_0.gguf`, on the CPU - the GPU is full), the vector is normalized and
//! stored as 384 bytes (this embedder build manages ~1.2 ms a token with 8 threads on MiniLM, twice that on BGE: a first index of a repo is half an hour of
//! low-priority background work, then only what changed is embedded again), and a query is answered by the chunks whose vectors are closest to the query's.  The index lives on disk (`semindex-<root>.bin` next to the tool-output
//! log), is refreshed file by file when a file's modified time or size changed, and is built on a background thread: a search made while it is still building answers
//! from what is there and says so.  Nothing here touches the GPU, and the embedder is told to use 2 threads so the engine keeps its CPU.
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, UNIX_EPOCH};

pub const DIM: usize = 384;
const SEP: &str = "<#sep#>";
const MAX_FILE_BYTES: u64 = 400_000;
const CHUNK_CHARS: usize = 900;
const CHUNK_LINES: usize = 40;
const OVERLAP_LINES: usize = 3;
const LINE_CLIP: usize = 300;
/// MiniLM and BGE have 512 positions: a text of 511 or more WordPiece tokens makes `llama-embedding` abort (`GGML_ASSERT(i01 >= 0 && i01 < ne01)` in the position lookup, exit 0xC0000409;
/// docs/STATUS.md 91) - and dense code reaches that in 900 characters (every symbol is a token).  Chunks are cut, and texts cut again before they are embedded, at this many
/// *estimated* tokens (`est_tokens` over-counts, so the real count stays under the limit with room to spare).
const CHUNK_TOKENS: usize = 340;
const EMBED_TOKENS: usize = 400;
const BATCH_TEXTS: usize = 32;
const BATCH_CHARS: usize = 20_000;
const MAGIC: &[u8; 8] = b"SHSIDX01";
const EXTS: &[&str] = &[
    "rs", "md", "py", "js", "ts", "json", "toml", "txt", "cu", "cuh", "c", "cpp", "h", "hpp",
    "bat", "ps1", "sh", "html", "css", "yml", "yaml",
];
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "models",
    "bin",
    "bench",
    "memory",
    "__pycache__",
    "ShadowConvCache",
    "toolcache",
    "fixtures",
    "AppData",
    "Windows",
    "Program Files",
    "Program Files (x86)",
    "ProgramData",
    "$Recycle.Bin",
];
/// More candidate files than this: the working folder is not a project (a home folder, a drive), and nothing is indexed.
const MAX_FILES: usize = 5_000;
/// Files above this size are embedded after everything else.
const BIG_FILE: u64 = 150_000;

/// Something that turns texts into vectors (the real one runs `llama-embedding.exe`; the tests use a fake one).
pub trait Embedder: Send + Sync {
    /// False for the stand-in used when there is no embedder: nothing is embedded and the tool answers by keywords.
    fn available(&self) -> bool {
        true
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;

    /// A question to search with (BGE models want a fixed instruction in front of a short query; MiniLM does not).
    fn embed_query(&self, q: &str) -> Result<Vec<f32>, String> {
        self.embed(&[q.to_string()])?
            .pop()
            .ok_or_else(|| "no vector for the query".to_string())
    }
}

/// No embedder (not found): `available()` is false and nothing calls `embed`.
struct NoEmbedder;
impl Embedder for NoEmbedder {
    fn available(&self) -> bool {
        false
    }
    fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Err("no embedder".into())
    }
}

/// `llama-embedding.exe` with the BGE-small model, run as a subprocess.
pub struct Llama {
    exe: PathBuf,
    model: PathBuf,
    threads: usize,
}

const MODELS: &[&str] = &[
    "models/embed-minilm/all-MiniLM-L6-v2-Q8_0.gguf",
    "models/bge-small-en-v1.5-q8_0.gguf",
];
const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

impl Llama {
    /// `SHADOW_EMBED_EXE` / `SHADOW_EMBED_MODEL`, else the repo's `bin/` and `models/` found by walking up from the executable's and the working folder.
    pub fn find() -> Option<Llama> {
        let env = |k: &str| {
            std::env::var_os(k)
                .map(PathBuf::from)
                .filter(|p| p.is_file())
        };
        let (exe, model) = (env("SHADOW_EMBED_EXE"), env("SHADOW_EMBED_MODEL"));
        let threads = std::env::var("SHADOW_EMBED_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4usize)
            .clamp(1, 16);
        if let (Some(exe), Some(model)) = (exe.clone(), model.clone()) {
            return Some(Llama {
                exe,
                model,
                threads,
            });
        }
        let starts = [std::env::current_exe().ok(), std::env::current_dir().ok()];
        for start in starts.into_iter().flatten() {
            let mut d: Option<&Path> = Some(start.as_path());
            while let Some(p) = d {
                let e = p.join("bin").join("llama-embedding.exe");
                let m = MODELS.iter().map(|f| p.join(f)).find(|f| f.is_file());
                if let (true, Some(m)) = (e.is_file(), m) {
                    return Some(Llama {
                        exe: exe.unwrap_or(e),
                        model: model.unwrap_or(m),
                        threads,
                    });
                }
                d = p.parent();
            }
        }
        None
    }
}

impl Llama {
    /// The model's file name: the index of one model's vectors is not another's.
    pub fn model_id(&self) -> String {
        self.model
            .file_name()
            .map_or_else(|| "model".to_string(), |n| n.to_string_lossy().to_string())
    }
}

impl Embedder for Llama {
    fn embed_query(&self, q: &str) -> Result<Vec<f32>, String> {
        let text = if self.model_id().to_lowercase().contains("bge") {
            format!("{BGE_QUERY_PREFIX}{q}")
        } else {
            q.to_string()
        };
        self.embed(&[text])?
            .pop()
            .ok_or_else(|| "no vector for the query".to_string())
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(texts.len());
        let mut i = 0;
        while i < texts.len() {
            let mut j = i;
            let mut chars = 0;
            while j < texts.len()
                && j - i < BATCH_TEXTS
                && (j == i || chars + texts[j].len() < BATCH_CHARS)
            {
                chars += texts[j].len();
                j += 1;
            }
            // the separator is the only thing the protocol reads literally; an empty text would be skipped
            let joined = texts[i..j]
                .iter()
                .map(|t| {
                    let t = fit_tokens(&t.replace(SEP, " "), EMBED_TOKENS);
                    if t.trim().is_empty() {
                        " ".to_string()
                    } else {
                        t
                    }
                })
                .collect::<Vec<_>>()
                .join(SEP);
            let mut cmd = std::process::Command::new(&self.exe);
            cmd.args(["-m"])
                .arg(&self.model)
                .args(["-p"])
                .arg(&joined)
                .args([
                    "--embd-separator",
                    SEP,
                    "--embd-output-format",
                    "json",
                    "-t",
                ])
                .arg(self.threads.to_string())
                .stderr(std::process::Stdio::null());
            // below normal priority and no console window: it runs beside the engine and yields the CPU to it
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x0000_4000 | 0x0800_0000);
            }
            let o = cmd
                .output()
                .map_err(|e| format!("could not run the embedder: {e}"))?;
            if !o.status.success() {
                return Err(format!("the embedder failed (exit {:?})", o.status.code()));
            }
            let v: Value = serde_json::from_slice(&o.stdout)
                .map_err(|e| format!("the embedder's reply is not JSON: {e}"))?;
            let data = v
                .get("data")
                .and_then(Value::as_array)
                .ok_or("the embedder's reply has no data")?;
            if data.len() != j - i {
                return Err(format!(
                    "the embedder returned {} vectors for {} texts",
                    data.len(),
                    j - i
                ));
            }
            for item in data {
                let e = item
                    .get("embedding")
                    .and_then(Value::as_array)
                    .ok_or("an item has no embedding")?;
                let mut vec: Vec<f32> = e
                    .iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect();
                if vec.len() != DIM {
                    return Err(format!("a vector has {} numbers, not {DIM}", vec.len()));
                }
                normalize(&mut vec);
                out.push(vec);
            }
            i = j;
        }
        Ok(out)
    }
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn quantize(v: &[f32]) -> Vec<i8> {
    v.iter()
        .map(|x| (x * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Cosine of a normalized query and a stored (normalized, quantized) vector.
fn score(q: &[f32], c: &[i8]) -> f32 {
    q.iter().zip(c).map(|(a, b)| a * (*b as f32)).sum::<f32>() / 127.0
}

/// An over-estimate of the WordPiece tokens a BERT-family embedder makes of `s`: a run of letters and digits ceil(len / 3), anything else that is not white space (punctuation, symbols,
/// accented and non-Latin letters) one each.
fn est_tokens(s: &str) -> usize {
    let (mut n, mut run) = (0usize, 0usize);
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            run += 1;
        } else {
            if run > 0 {
                n += run.div_ceil(3);
                run = 0;
            }
            if !c.is_whitespace() {
                n += 1;
            }
        }
    }
    n + run.div_ceil(3)
}

/// The longest prefix of `s` (cut at a character) whose estimate is at most `max` tokens.
fn fit_tokens(s: &str, max: usize) -> String {
    if est_tokens(s) <= max {
        return s.to_string();
    }
    let (mut n, mut run, mut end) = (0usize, 0usize, 0usize);
    for (i, c) in s.char_indices() {
        let before = n + run.div_ceil(3);
        if c.is_ascii_alphanumeric() {
            run += 1;
        } else {
            if run > 0 {
                n += run.div_ceil(3);
                run = 0;
            }
            if !c.is_whitespace() {
                n += 1;
            }
        }
        if n + run.div_ceil(3) > max {
            end = i;
            let _ = before;
            break;
        }
        end = i + c.len_utf8();
    }
    s[..end].to_string()
}

/// `text` cut into chunks of ~`CHUNK_LINES` lines / `CHUNK_CHARS` characters with `OVERLAP_LINES` of overlap: (first line number, 1-indexed; the chunk's text).
pub fn chunk_text(text: &str) -> Vec<(u32, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let clip = |l: &str| -> String { l.chars().take(LINE_CLIP).collect() };
    let mut out = vec![];
    let mut i = 0;
    while i < lines.len() {
        let (mut j, mut chars, mut toks) = (i, 0usize, 0usize);
        while j < lines.len()
            && j - i < CHUNK_LINES
            && (j == i
                || (chars + lines[j].chars().count().min(LINE_CLIP) < CHUNK_CHARS
                    && toks + est_tokens(&clip(lines[j])) <= CHUNK_TOKENS))
        {
            chars += lines[j].chars().count().min(LINE_CLIP) + 1;
            toks += est_tokens(&clip(lines[j]));
            j += 1;
        }
        let body = lines[i..j]
            .iter()
            .map(|l| clip(l))
            .collect::<Vec<_>>()
            .join("\n");
        if !body.trim().is_empty() {
            out.push((i as u32 + 1, body));
        }
        if j >= lines.len() {
            break;
        }
        i = if j - i > OVERLAP_LINES {
            j - OVERLAP_LINES
        } else {
            j
        };
    }
    out
}

struct Chunk {
    src: String,
    line: u32,
    text: String,
    /// The normalized, quantized vector; empty while the chunk has none (not embedded yet, or the embedder failed on it).
    v: Vec<i8>,
    /// Keyword statistics (BM25): (hash of a term, how often it occurs) sorted by hash, over the source's name and the text, and the number of terms.
    tf: Vec<(u64, u16)>,
    len: u32,
}

impl Chunk {
    fn new(src: String, line: u32, text: String, v: Vec<i8>) -> Chunk {
        let mut all: Vec<u64> = terms(&src).iter().chain(terms(&text).iter()).map(|t| term_hash(t)).collect();
        let len = all.len() as u32;
        all.sort_unstable();
        let mut tf: Vec<(u64, u16)> = Vec::new();
        for h in all {
            match tf.last_mut() {
                Some((p, n)) if *p == h => *n = n.saturating_add(1),
                _ => tf.push((h, 1)),
            }
        }
        Chunk { src, line, text, v, tf, len }
    }

    fn has_vector(&self) -> bool {
        self.v.len() == DIM
    }
}

const STOP: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "was", "with", "that", "this", "from", "have", "has", "its", "into", "than", "then", "what", "when", "which", "how",
    "does", "did", "why", "who", "where", "of", "to", "in", "on", "at", "is", "it", "as", "be", "by", "or", "an", "if", "we", "do", "so",
];

/// Lower-case a word and drop a plural 's' (`experts` -> `expert`): the same on both sides of the match.
fn stem(w: &str) -> String {
    let w = w.to_lowercase();
    if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") && !w.ends_with("us") && !w.ends_with("is") {
        w[..w.len() - 1].to_string()
    } else {
        w
    }
}

/// The search terms of a text: every run of letters, digits and `_` whole (`shadow_fused_moe`), and, when it is an identifier, its parts (`shadow`, `fused`, `moe`: split at `_`, at
/// lower-to-UPPER and at letter-digit changes), stop words dropped, one-character terms dropped.
pub fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let push = |w: &str, out: &mut Vec<String>| {
        let t = stem(w);
        if t.chars().count() >= 2 && !STOP.contains(&t.as_str()) {
            out.push(t);
        }
    };
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if word.is_empty() {
            continue;
        }
        push(word, &mut out);
        let mut parts: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut prev: Option<char> = None;
        for c in word.chars() {
            let split = c == '_'
                || prev.is_some_and(|p| (p.is_lowercase() && c.is_uppercase()) || (p.is_alphabetic() && c.is_ascii_digit()) || (p.is_ascii_digit() && c.is_alphabetic()));
            if split && !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            if c != '_' {
                cur.push(c);
            }
            prev = Some(c);
        }
        if !cur.is_empty() {
            parts.push(cur);
        }
        if parts.len() > 1 {
            for p in &parts {
                push(p, &mut out);
            }
        }
    }
    out
}

fn term_hash(t: &str) -> u64 {
    t.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// One answer of `hybrid`.
pub struct Hit<'a> {
    pub src: &'a str,
    pub line: u32,
    pub text: &'a str,
    pub keywords: Option<f32>,
    pub meaning: Option<f32>,
}

/// What is indexed: the chunks, and for each source (a file's relative path, or "output #n") its modified time and size when it was embedded.
#[derive(Default)]
pub struct Index {
    chunks: Vec<Chunk>,
    seen: HashMap<String, (u64, u64)>,
}

impl Index {
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    fn replace(&mut self, src: &str, stamp: (u64, u64), new: Vec<Chunk>) {
        self.chunks.retain(|c| c.src != src);
        self.chunks.extend(new);
        self.seen.insert(src.to_string(), stamp);
    }

    fn forget(&mut self, src: &str) {
        self.chunks.retain(|c| c.src != src);
        self.seen.remove(src);
    }

    /// The best `k` chunks for a normalized query vector: (score, source, first line, text); only sources starting with `under` when it is given.
    pub fn search(&self, q: &[f32], k: usize, under: Option<&str>) -> Vec<(f32, &str, u32, &str)> {
        let mut hits: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .filter(|c| under.is_none_or(|u| c.src.starts_with(u)))
            .map(|c| (score(q, &c.v), c))
            .collect();
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hits.into_iter()
            .take(k)
            .map(|(s, c)| (s, c.src.as_str(), c.line, c.text.as_str()))
            .collect()
    }

    /// BM25 over the chunks (only sources starting with `under` when given): the best 60 as (score, chunk index), best first.  The terms are the query's (`terms`, at most 24).
    fn keyword(&self, query: &str, under: Option<&str>) -> Vec<(f32, usize)> {
        let mut q: Vec<u64> = terms(query).iter().map(|t| term_hash(t)).collect();
        q.sort_unstable();
        q.dedup();
        q.truncate(24);
        let live: Vec<usize> = (0..self.chunks.len()).filter(|&i| under.is_none_or(|u| self.chunks[i].src.starts_with(u))).collect();
        if q.is_empty() || live.is_empty() {
            return vec![];
        }
        let n = live.len() as f32;
        let avg = live.iter().map(|&i| self.chunks[i].len as f32).sum::<f32>() / n;
        let tf_of = |c: &Chunk, h: u64| c.tf.binary_search_by_key(&h, |x| x.0).map_or(0, |k| c.tf[k].1) as f32;
        let idf: Vec<f32> = q
            .iter()
            .map(|&h| {
                let df = live.iter().filter(|&&i| tf_of(&self.chunks[i], h) > 0.0).count() as f32;
                ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
            })
            .collect();
        let (k1, b) = (1.2f32, 0.75f32);
        let mut hits: Vec<(f32, usize)> = live
            .iter()
            .filter_map(|&i| {
                let c = &self.chunks[i];
                let norm = 1.0 - b + b * c.len as f32 / avg.max(1.0);
                let sc: f32 = q.iter().zip(&idf).map(|(&h, w)| {
                    let f = tf_of(c, h);
                    if f > 0.0 { w * f * (k1 + 1.0) / (f + k1 * norm) } else { 0.0 }
                }).sum();
                (sc > 0.0).then_some((sc, i))
            })
            .collect();
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(60);
        hits
    }

    /// The best `k` chunks for a query: the keyword ranking (`keyword`) and, with a query vector, the meaning ranking (cosine, over the chunks that have a vector), fused by reciprocal
    /// rank (a chunk both put high wins; a chunk only one found still counts).  Each hit: the chunk and its keyword score and cosine where it has them.
    pub fn hybrid(&self, query: &str, qv: Option<&[f32]>, k: usize, under: Option<&str>) -> Vec<Hit<'_>> {
        let kw = self.keyword(query, under);
        let mut sem: Vec<(f32, usize)> = match qv {
            Some(q) => (0..self.chunks.len())
                .filter(|&i| self.chunks[i].has_vector() && under.is_none_or(|u| self.chunks[i].src.starts_with(u)))
                .map(|i| (score(q, &self.chunks[i].v), i))
                .collect(),
            None => vec![],
        };
        sem.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        sem.truncate(60);
        let mut fused: HashMap<usize, (f32, Option<f32>, Option<f32>)> = HashMap::new();
        for (r, (s, i)) in kw.iter().enumerate() {
            let e = fused.entry(*i).or_insert((0.0, None, None));
            e.0 += 1.0 / (60.0 + r as f32);
            e.1 = Some(*s);
        }
        for (r, (s, i)) in sem.iter().enumerate() {
            let e = fused.entry(*i).or_insert((0.0, None, None));
            e.0 += 1.0 / (60.0 + r as f32);
            e.2 = Some(*s);
        }
        let mut v: Vec<(usize, (f32, Option<f32>, Option<f32>))> = fused.into_iter().collect();
        v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        v.into_iter()
            .take(k)
            .map(|(i, (_, kw, sem))| Hit { src: &self.chunks[i].src, line: self.chunks[i].line, text: &self.chunks[i].text, keywords: kw, meaning: sem })
            .collect()
    }

    /// (chunks, chunks with a vector).
    pub fn counts(&self) -> (usize, usize) {
        (self.chunks.len(), self.chunks.iter().filter(|c| c.has_vector()).count())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut b: Vec<u8> = Vec::with_capacity(self.chunks.len() * (DIM + 600));
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&(self.seen.len() as u32).to_le_bytes());
        for (s, (m, l)) in &self.seen {
            b.extend_from_slice(&(s.len() as u32).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
            b.extend_from_slice(&m.to_le_bytes());
            b.extend_from_slice(&l.to_le_bytes());
        }
        b.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        for c in &self.chunks {
            b.extend_from_slice(&(c.src.len() as u32).to_le_bytes());
            b.extend_from_slice(c.src.as_bytes());
            b.extend_from_slice(&c.line.to_le_bytes());
            b.extend_from_slice(&(c.text.len() as u32).to_le_bytes());
            b.extend_from_slice(c.text.as_bytes());
            if c.has_vector() {
                b.extend(c.v.iter().map(|x| *x as u8));
            } else {
                b.extend(std::iter::repeat_n(0u8, DIM)); // no vector yet: zeros, read back as none
            }
        }
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &b)
            .and_then(|_| std::fs::rename(&tmp, path))
            .map_err(|e| format!("could not write the index: {e}"))
    }

    pub fn load(path: &Path) -> Option<Index> {
        let b = std::fs::read(path).ok()?;
        if b.len() < 12 || &b[..8] != MAGIC {
            return None;
        }
        let mut p = 8usize;
        let u32_at = |p: &mut usize| -> Option<u32> {
            let v = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?);
            *p += 4;
            Some(v)
        };
        let u64_at = |p: &mut usize| -> Option<u64> {
            let v = u64::from_le_bytes(b.get(*p..*p + 8)?.try_into().ok()?);
            *p += 8;
            Some(v)
        };
        let str_at = |p: &mut usize, n: usize| -> Option<String> {
            let s = String::from_utf8(b.get(*p..*p + n)?.to_vec()).ok()?;
            *p += n;
            Some(s)
        };
        let mut idx = Index::default();
        for _ in 0..u32_at(&mut p)? {
            let n = u32_at(&mut p)? as usize;
            let s = str_at(&mut p, n)?;
            let (m, l) = (u64_at(&mut p)?, u64_at(&mut p)?);
            idx.seen.insert(s, (m, l));
        }
        for _ in 0..u32_at(&mut p)? {
            let n = u32_at(&mut p)? as usize;
            let src = str_at(&mut p, n)?;
            let line = u32_at(&mut p)?;
            let tn = u32_at(&mut p)? as usize;
            let text = str_at(&mut p, tn)?;
            let mut v: Vec<i8> = b.get(p..p + DIM)?.iter().map(|x| *x as i8).collect();
            p += DIM;
            if v.iter().all(|x| *x == 0) {
                v.clear();
            }
            idx.chunks.push(Chunk::new(src, line, text, v));
        }
        Some(idx)
    }
}

/// Progress of a build, read by the tool while it runs.
#[derive(Default)]
pub struct Progress {
    pub building: AtomicBool,
    pub done: AtomicUsize,
    pub total: AtomicUsize,
    pub error: Mutex<Option<String>>,
    /// The text of every changed file is in the index (keyword search works); the vectors follow.
    pub text_ready: AtomicBool,
    /// Embedder batches in a row that gave no vector at all: after `MAX_BAD_STREAK` the embedder is left alone (`embed_off`) and the tool answers by keywords.
    pub bad_streak: AtomicUsize,
    pub embed_off: AtomicBool,
    /// Texts the embedder could not embed even alone, and the sources they were in (not tried again in this process).
    pub embed_failures: AtomicUsize,
    pub failed: Mutex<std::collections::HashSet<String>>,
}

const MAX_BAD_STREAK: usize = 3;
const MAX_SPLIT_ERRORS: usize = 12;

fn stamp_of(meta: &std::fs::Metadata) -> (u64, u64) {
    let m = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    (m, meta.len())
}

/// The files of the project worth indexing under `root`: (relative path with `/`, full path, stamp).
pub fn scan_files(root: &Path) -> Vec<(String, PathBuf, (u64, u64))> {
    let mut out = vec![];
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let Ok(meta) = e.metadata() else { continue };
            if meta.is_dir() {
                if SKIP_DIRS.contains(&name.as_str())
                    || name.starts_with("target-")
                    || name.starts_with('.')
                {
                    continue;
                }
                stack.push(path);
            } else if meta.len() > 0 && meta.len() <= MAX_FILE_BYTES && !name.contains("tables") {
                let ext = path
                    .extension()
                    .and_then(|x| x.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if EXTS.contains(&ext.as_str()) {
                    let rel = path
                        .strip_prefix(root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.push((rel, path, stamp_of(&meta)));
                    if out.len() > MAX_FILES {
                        return out;
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Whether a chunk has enough words in it to be worth a vector (a table of numbers has none).
fn wordy(t: &str) -> bool {
    let (mut alpha, mut all) = (0usize, 0usize);
    for c in t.chars().filter(|c| !c.is_whitespace()) {
        all += 1;
        alpha += c.is_alphabetic() as usize;
    }
    all > 0 && alpha * 100 >= all * 35
}

/// A source's text cut into chunks (without vectors, except those `old` holds for the same chunk text: only what changed in a file is embedded again).  A table of numbers is not
/// worth a vector, and has no words to search for either: it is left out.
fn chunks_of(src: &str, text: &str, old: &HashMap<String, Vec<i8>>) -> Vec<Chunk> {
    chunk_text(text)
        .into_iter()
        .filter(|(_, t)| wordy(t))
        .map(|(line, t)| {
            let v = old.get(&t).cloned().unwrap_or_default();
            Chunk::new(src.to_string(), line, t, v)
        })
        .collect()
}

/// The vectors the index already holds for a source, by chunk text.
fn old_vectors(ix: &Index, src: &str) -> HashMap<String, Vec<i8>> {
    ix.chunks.iter().filter(|c| c.src == src && c.has_vector()).map(|c| (c.text.clone(), c.v.clone())).collect()
}

/// `embed`, tolerant: a batch the embedder fails on is halved until the text that makes it fail is alone, which is retried cut in half (the estimate of its tokens may have been low) and then
/// given up (None); the rest of the batch gets its vectors.  At most `MAX_SPLIT_ERRORS` failed calls for one batch, then all that is left is None.
fn embed_tolerant(emb: &dyn Embedder, texts: &[String], progress: &Progress) -> Vec<Option<Vec<f32>>> {
    fn go(emb: &dyn Embedder, texts: &[String], errors: &mut usize) -> Vec<Option<Vec<f32>>> {
        if texts.is_empty() {
            return vec![];
        }
        match emb.embed(texts) {
            Ok(v) if v.len() == texts.len() => v.into_iter().map(Some).collect(),
            _ => {
                *errors += 1;
                if *errors > MAX_SPLIT_ERRORS {
                    return vec![None; texts.len()];
                }
                if texts.len() == 1 {
                    let n = texts[0].chars().count();
                    if n > 40 && *errors <= MAX_SPLIT_ERRORS / 2 {
                        let half: String = texts[0].chars().take(n / 2).collect();
                        return go(emb, &[half], errors);
                    }
                    return vec![None];
                }
                let (a, b) = texts.split_at(texts.len() / 2);
                let mut out = go(emb, a, errors);
                out.extend(go(emb, b, errors));
                out
            }
        }
    }
    let mut errors = 0;
    let out = go(emb, texts, &mut errors);
    let lost = out.iter().filter(|v| v.is_none()).count();
    progress.embed_failures.fetch_add(lost, Ordering::Relaxed);
    if !out.is_empty() && lost == out.len() {
        if progress.bad_streak.fetch_add(1, Ordering::Relaxed) + 1 >= MAX_BAD_STREAK {
            progress.embed_off.store(true, Ordering::Relaxed);
        }
    } else if !out.is_empty() {
        progress.bad_streak.store(0, Ordering::Relaxed);
    }
    out
}

/// Whether to use the embedder now (there is one, and it has not failed `MAX_BAD_STREAK` batches in a row).
fn usable(emb: &dyn Embedder, progress: &Progress) -> bool {
    emb.available() && !progress.embed_off.load(Ordering::Relaxed)
}

/// Gives the chunks of source `src` that have no vector one (up to `max` chunks); returns how many it embedded.  The index is locked only to read and to write.
fn fill_vectors(index: &Mutex<Index>, src: &str, emb: &dyn Embedder, progress: &Progress, max: usize) -> usize {
    if !usable(emb, progress) || progress.failed.lock().unwrap().contains(src) {
        return 0;
    }
    let need: Vec<(u32, String)> = {
        let ix = index.lock().unwrap();
        ix.chunks.iter().filter(|c| c.src == src && !c.has_vector()).take(max).map(|c| (c.line, c.text.clone())).collect()
    };
    if need.is_empty() {
        return 0;
    }
    let texts: Vec<String> = need.iter().map(|(_, t)| format!("{src}\n{t}")).collect();
    let vecs = embed_tolerant(emb, &texts, progress);
    let mut done = 0;
    let mut ix = index.lock().unwrap();
    for ((line, text), v) in need.iter().zip(vecs) {
        match v {
            Some(v) => {
                if let Some(c) = ix.chunks.iter_mut().find(|c| c.src == src && c.line == *line && c.text == *text) {
                    c.v = quantize(&v);
                    done += 1;
                }
            }
            None => {
                progress.failed.lock().unwrap().insert(src.to_string());
            }
        }
    }
    done
}

/// Brings `index` up to date with the files under `root`.  First the text of everything new or changed goes into the index (keyword search works from then on: seconds), then the
/// vectors are made, file by file, recently changed first, with `pause` slept between files (the build runs beside the engine); the embedder failing on a text leaves that chunk without
/// a vector and does not stop the build.  Drops what is gone.  Returns the number of files whose text was (re)indexed.
pub fn build(
    index: &Mutex<Index>,
    progress: &Progress,
    root: &Path,
    emb: &dyn Embedder,
    pause: Duration,
    save_to: Option<&Path>,
) -> Result<usize, String> {
    let files = scan_files(root);
    if files.len() > MAX_FILES {
        return Err(format!("the working folder ({}) holds more than {MAX_FILES} files to index: start Shadow from a project folder", root.display()));
    }
    let (todo, gone): (Vec<_>, Vec<String>) = {
        let ix = index.lock().unwrap();
        let mut todo = files.iter().filter(|(rel, _, st)| ix.seen.get(rel) != Some(st)).cloned().collect::<Vec<_>>();
        // what was changed lately first (it is what is being worked on), the big files last
        todo.sort_by(|a, b| (a.2 .1 > BIG_FILE).cmp(&(b.2 .1 > BIG_FILE)).then(b.2 .0.cmp(&a.2 .0)));
        let gone = ix.seen.keys().filter(|k| !k.starts_with("output #") && !files.iter().any(|(r, _, _)| r == *k)).cloned().collect();
        (todo, gone)
    };
    progress.text_ready.store(false, Ordering::SeqCst);
    progress.total.store(todo.len(), Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    {
        let mut ix = index.lock().unwrap();
        for g in &gone {
            ix.forget(g);
        }
    }
    // phase 1: the text (and the vectors the index already has for unchanged chunks)
    let mut order: Vec<(String, (u64, u64))> = Vec::new();
    for (rel, path, st) in &todo {
        match std::fs::read(path) {
            Ok(b) if !b.contains(&0) => {
                let text = String::from_utf8_lossy(&b).to_string();
                let mut ix = index.lock().unwrap();
                let old = old_vectors(&ix, rel);
                let chunks = chunks_of(rel, &text, &old);
                let all_have = chunks.iter().all(|c| c.has_vector());
                ix.chunks.retain(|c| &c.src != rel);
                ix.chunks.extend(chunks);
                if all_have {
                    ix.seen.insert(rel.clone(), *st);
                } else {
                    order.push((rel.clone(), *st));
                }
            }
            _ => index.lock().unwrap().replace(rel, *st, vec![]), // binary or unreadable: remembered, not indexed
        }
    }
    progress.text_ready.store(true, Ordering::SeqCst);
    // files whose text is current but that still lack vectors (the embedder was away, or failed): tried again unless this process saw it fail on them
    if usable(emb, progress) {
        let ix = index.lock().unwrap();
        let mut missing: Vec<String> = ix.chunks.iter().filter(|c| !c.has_vector()).map(|c| c.src.clone()).collect();
        missing.sort();
        missing.dedup();
        for m in missing {
            if !order.iter().any(|(r, _)| *r == m) && !m.starts_with("output #") && !progress.failed.lock().unwrap().contains(&m) {
                let st = ix.seen.get(&m).copied().unwrap_or((0, 0));
                order.push((m, st));
            }
        }
    }
    // phase 2: the vectors
    let mut last_save = Instant::now();
    for (rel, st) in &order {
        while usable(emb, progress) && fill_vectors(index, rel, emb, progress, CHUNKS_PER_CALL) > 0 {}
        index.lock().unwrap().seen.insert(rel.clone(), *st);
        progress.done.fetch_add(1, Ordering::Relaxed);
        // a restart in the middle of a long first build must not lose it
        if let Some(path) = save_to {
            if last_save.elapsed() > Duration::from_secs(60) {
                let _ = index.lock().unwrap().save(path);
                last_save = Instant::now();
            }
        }
        if !pause.is_zero() {
            std::thread::sleep(pause);
        }
    }
    progress.done.store(todo.len(), Ordering::Relaxed);
    Ok(todo.len())
}

const CHUNKS_PER_CALL: usize = 64;

/// Puts the tool outputs of the session's log (`ids`, read with `load`) that are not in the index yet into it as text (keyword search finds them at once); returns how many chunks it added.
pub fn add_outputs(index: &Mutex<Index>, ids: &[u64], load: &dyn Fn(u64) -> Option<String>) -> usize {
    let mut added = 0;
    for id in ids {
        let src = format!("output #{id}");
        if index.lock().unwrap().seen.contains_key(&src) {
            continue;
        }
        let Some(text) = load(*id) else { continue };
        let chunks = chunks_of(&src, &text, &HashMap::new());
        added += chunks.len();
        index.lock().unwrap().replace(&src, (0, text.len() as u64), chunks);
    }
    added
}

/// Embeds the chunks of session outputs that have no vector yet, at most `max_chunks` in one call (they are on the path of a search: the rest follow on later calls).
pub fn embed_outputs(index: &Mutex<Index>, emb: &dyn Embedder, progress: &Progress, max_chunks: usize) -> usize {
    if !usable(emb, progress) {
        return 0;
    }
    let srcs: Vec<String> = {
        let ix = index.lock().unwrap();
        let mut v: Vec<String> = ix.chunks.iter().filter(|c| c.src.starts_with("output #") && !c.has_vector()).map(|c| c.src.clone()).collect();
        v.dedup();
        v
    };
    let mut done = 0;
    for s in srcs {
        if done >= max_chunks {
            break;
        }
        done += fill_vectors(index, &s, emb, progress, max_chunks - done);
    }
    done
}

/// A query answered: the best chunks as text.
pub fn answer(index: &Index, query: &str, qv: Option<&[f32]>, k: usize, under: Option<&str>, note: &str) -> String {
    let hits = index.hybrid(query, qv, k, under);
    if hits.is_empty() {
        return format!("nothing indexed matches{}{}", under.map_or(String::new(), |u| format!(" under {u}")), note);
    }
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        let shown: Vec<&str> = h.text.lines().take(8).collect();
        let how = match (h.keywords, h.meaning) {
            (Some(k), Some(m)) => format!("keywords {k:.1}, meaning {m:.2}"),
            (Some(k), None) => format!("keywords {k:.1}"),
            (None, Some(m)) => format!("meaning {m:.2}"),
            (None, None) => String::new(),
        };
        out.push_str(&format!(
            "{}. {}:{}  (score {})\n{}\n",
            i + 1,
            h.src,
            h.line,
            how,
            shown.iter().map(|l| format!("   {l}")).collect::<Vec<_>>().join("\n")
        ));
    }
    out.push_str("(read more of a file with read_file path=..., start=..., end=...; of an output with recall id=N)");
    out.push_str(note);
    out
}

// ---- the tool: one index per working folder, built on a background thread

struct Global {
    index: Mutex<Index>,
    progress: Progress,
    root: PathBuf,
    cache: PathBuf,
    last_scan: Mutex<Option<Instant>>,
}

static GLOBAL: OnceLock<Global> = OnceLock::new();

fn cache_path(root: &Path, model_id: &str) -> PathBuf {
    let base = std::env::var_os("SHADOW_TOOL_LOG_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("LOCALAPPDATA")
                .map(|a| PathBuf::from(a).join("Shadow").join("toolcache"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("shadow-toolcache"));
    let h = root
        .to_string_lossy()
        .bytes()
        .chain(model_id.bytes())
        .fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        });
    base.join(format!("semindex-{h:016x}.bin"))
}

/// The `search` tool: `query`, optional `max_results` (default 6) and `path` (only under this folder or file).  By keywords (BM25: ready as soon as the files are read) and by meaning
/// (the embedder, when there is one and it works), fused; the answer says which it is.
pub fn tool(args: &Value, output_ids: &[u64], load_output: &dyn Fn(u64) -> Option<String>) -> Result<String, String> {
    let query = args.get("query").and_then(Value::as_str).unwrap_or("").trim().to_string();
    if query.is_empty() {
        return Err("give a text to search for (query)".into());
    }
    let k = args.get("max_results").and_then(Value::as_u64).unwrap_or(6).clamp(1, 15) as usize;
    let under = args.get("path").and_then(Value::as_str).map(|p| p.replace('\\', "/").trim_start_matches("./").to_string()).filter(|p| !p.is_empty());
    static EMB: OnceLock<Option<Llama>> = OnceLock::new();
    let llama: Option<&'static Llama> = EMB.get_or_init(Llama::find).as_ref();
    static NONE: NoEmbedder = NoEmbedder;
    let emb: &'static dyn Embedder = match llama {
        Some(l) => l,
        None => &NONE,
    };
    let g = GLOBAL.get_or_init(|| {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        // the index of a model's vectors is not another's; with no embedder the cache is the keyword-only one
        let cache = cache_path(&root, &llama.map_or("keywords".to_string(), |l| l.model_id()));
        Global { index: Mutex::new(Index::load(&cache).unwrap_or_default()), progress: Progress::default(), root, cache, last_scan: Mutex::new(None) }
    });
    // refresh: a scan at most every 30 s, on a thread; the first call waits a while for the text to be in, later ones for a short while
    let stale = g.last_scan.lock().unwrap().is_none_or(|t| t.elapsed() > Duration::from_secs(30));
    if stale && !g.progress.building.swap(true, Ordering::SeqCst) {
        *g.last_scan.lock().unwrap() = Some(Instant::now());
        std::thread::spawn(move || {
            let r = build(&g.index, &g.progress, &g.root, emb, Duration::from_millis(30), Some(&g.cache));
            match r {
                Ok(n) => {
                    *g.progress.error.lock().unwrap() = None;
                    if n > 0 || !g.cache.exists() {
                        let _ = g.index.lock().unwrap().save(&g.cache);
                    }
                }
                Err(e) => *g.progress.error.lock().unwrap() = Some(e),
            }
            g.progress.building.store(false, Ordering::SeqCst);
        });
    }
    let wait = if g.index.lock().unwrap().is_empty() { 12 } else { 6 };
    let t0 = Instant::now();
    while g.progress.building.load(Ordering::SeqCst) && !g.progress.text_ready.load(Ordering::SeqCst) && t0.elapsed() < Duration::from_secs(wait) {
        std::thread::sleep(Duration::from_millis(200));
    }
    add_outputs(&g.index, output_ids, load_output);
    embed_outputs(&g.index, emb, &g.progress, 40);
    // the query's vector, when the embedder can give one; if not the answer is by keywords and says so
    let (qv, why_not): (Option<Vec<f32>>, Option<String>) = if !emb.available() {
        (None, Some("the embedder was not found".to_string()))
    } else if g.progress.embed_off.load(Ordering::Relaxed) {
        (None, Some("the embedder failed several times in a row and was switched off until Shadow restarts".to_string()))
    } else {
        match emb.embed_query(&fit_tokens(&query, EMBED_TOKENS)) {
            Ok(v) => (Some(v), None),
            Err(e) => (None, Some(format!("the embedder failed ({e})"))),
        }
    };
    let ix = g.index.lock().unwrap();
    let (n, nv) = ix.counts();
    let mut note = String::new();
    if let Some(w) = &why_not {
        note.push_str(&format!("\n(keyword results only: {w})"));
    } else if nv < n {
        note.push_str(&format!("\n(meaning search covers {nv} of {n} chunks so far; the rest are found by keywords only)"));
    }
    if g.progress.building.load(Ordering::SeqCst) {
        note.push_str(&format!(
            "\n(the index is still being built: {} of {} changed files done, so results may be incomplete; ask again in a minute)",
            g.progress.done.load(Ordering::Relaxed),
            g.progress.total.load(Ordering::Relaxed)
        ));
    } else if let Some(e) = g.progress.error.lock().unwrap().clone() {
        note.push_str(&format!("\n(the last refresh of the index failed: {e})"));
    }
    Ok(answer(&ix, &query, qv.as_deref(), k, under.as_deref(), &note))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words into the dimensions (a bag of words): the same words, the same direction.
    struct Fake;
    impl Embedder for Fake {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0f32; DIM];
                    for w in t
                        .to_lowercase()
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|w| w.len() > 2)
                    {
                        let h = w
                            .bytes()
                            .fold(7usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
                        v[h % DIM] += 1.0;
                    }
                    normalize(&mut v);
                    v
                })
                .collect())
        }
    }

    /// Fails on any batch that holds a text with "POISON" in it (what the real embedder does on a text of 511 tokens), else a bag-of-words vector.
    struct Poison;
    impl Embedder for Poison {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            if texts.iter().any(|t| t.contains("POISON")) {
                return Err("the embedder failed (exit Some(-1073740791))".into());
            }
            Fake.embed(texts)
        }
    }

    struct Dead;
    impl Embedder for Dead {
        fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            Err("always".into())
        }
    }

    #[test]
    fn identifiers_are_searched_whole_and_by_their_parts() {
        let t = terms("SHADOW_FUSED_MOE=0 turns off fused_experts; the ShadowFusedMoe flag, experts, expert");
        for w in ["shadow_fused_moe", "shadow", "fused", "moe", "fused_expert", "expert", "shadowfusedmoe"] {
            assert!(t.contains(&w.to_string()), "{w} in {t:?}");
        }
        assert!(!t.contains(&"the".to_string()), "stop words are not terms");
        assert_eq!(stem("Experts"), stem("expert"));
        assert_eq!(stem("class"), "class", "a double s is not a plural");
    }

    #[test]
    fn keyword_search_ranks_by_bm25_and_needs_no_embedder() {
        let d = tmp("kw");
        std::fs::create_dir_all(d.join("src")).unwrap();
        std::fs::write(d.join("src/fused.rs"), "// SHADOW_FUSED_MOE=0 turns the fused int8 experts off\nfn fused_experts() {}\n").unwrap();
        std::fs::write(d.join("src/pinned.rs"), "// the pinned memory tier holds experts in RAM\nfn tier() {}\n").unwrap();
        std::fs::write(d.join("notes.md"), "the quick brown fox and a lazy dog talk about nothing in particular\n").unwrap();
        let ix = Mutex::new(Index::default());
        let pg = Progress::default();
        assert_eq!(build(&ix, &pg, &d, &NoEmbedder, Duration::ZERO, None).unwrap(), 3);
        assert!(pg.text_ready.load(Ordering::SeqCst));
        let g = ix.lock().unwrap();
        assert_eq!(g.counts(), (3, 0), "text only: no vectors, and the tool still answers");
        let h = g.hybrid("what turns off the SHADOW_FUSED_MOE flag", None, 3, None);
        assert_eq!(h[0].src, "src/fused.rs");
        assert!(h[0].keywords.is_some() && h[0].meaning.is_none());
        let h = g.hybrid("fused experts", None, 3, None);
        assert_eq!(h[0].src, "src/fused.rs", "the parts of an identifier match");
        let h = g.hybrid("pinned memory", None, 3, Some("src/p"));
        assert!(h.iter().all(|x| x.src == "src/pinned.rs") && !h.is_empty());
        assert!(g.hybrid("zebra", None, 3, None).is_empty(), "no term in common, no answer");
        assert!(answer(&g, "zebra", None, 3, None, "").starts_with("nothing indexed matches"));
        let a = answer(&g, "fused experts", None, 2, None, "\n(note)");
        assert!(a.starts_with("1. src/fused.rs:1  (score keywords ") && a.ends_with("(note)"), "{a}");
        drop(g);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn keywords_and_meaning_are_fused_and_a_chunk_without_a_vector_is_still_found() {
        let ix = Mutex::new(Index::default());
        {
            let mut g = ix.lock().unwrap();
            let v = |t: &str| quantize(&Fake.embed(&[t.to_string()]).unwrap()[0]);
            let a = "the dashboard shows latency percentiles";
            let b = "pinned memory tier of experts";
            g.chunks.push(Chunk::new("a.rs".into(), 1, a.into(), v(a)));
            g.chunks.push(Chunk::new("b.rs".into(), 1, b.into(), v(b)));
            g.chunks.push(Chunk::new("c.rs".into(), 1, "latency percentiles are computed from the recent requests".into(), vec![])); // no vector
        }
        let g = ix.lock().unwrap();
        let q = Fake.embed(&["latency percentiles".to_string()]).unwrap().pop().unwrap();
        let h = g.hybrid("latency percentiles", Some(&q), 3, None);
        let srcs: Vec<&str> = h.iter().map(|x| x.src).collect();
        assert!(srcs.contains(&"a.rs") && srcs.contains(&"c.rs"), "{srcs:?}: the vectorless chunk is found by its words");
        assert!(h.iter().find(|x| x.src == "a.rs").unwrap().meaning.is_some());
        assert!(h.iter().find(|x| x.src == "c.rs").unwrap().meaning.is_none());
        assert_eq!(g.counts(), (3, 2));
    }

    #[test]
    fn dense_code_is_cut_and_trimmed_under_the_embedders_token_limit() {
        // 511 tokens aborted llama-embedding: 900 characters of symbols were enough
        let line = "(\"Q6_K\", 256, 210, \"dequant_q6_k\", \"matvec_q6_k\", [0, 1, 2, 3]),";
        let text: String = (0..40).map(|_| format!("{line}\n")).collect();
        let chunks = chunk_text(&text);
        assert!(chunks.len() > 3, "dense lines are cut into more, smaller chunks: {}", chunks.len());
        for (_, c) in &chunks {
            assert!(est_tokens(c) <= CHUNK_TOKENS + est_tokens(line), "{} estimated tokens", est_tokens(c));
        }
        let cut = fit_tokens(&"(".repeat(900), EMBED_TOKENS);
        assert!(cut.len() == EMBED_TOKENS && est_tokens(&cut) <= EMBED_TOKENS);
        assert_eq!(fit_tokens("short text", EMBED_TOKENS), "short text");
        assert!(est_tokens(&"x".repeat(60)) == 20 && est_tokens("a, b") == 3);
    }

    #[test]
    fn a_text_the_embedder_fails_on_costs_only_its_own_vector() {
        let texts: Vec<String> = (0..9).map(|i| format!("ordinary text number {i} about experts")).chain(["POISON text".to_string()]).collect();
        let pg = Progress::default();
        let out = embed_tolerant(&Poison, &texts, &pg);
        assert_eq!(out.iter().filter(|v| v.is_some()).count(), 9, "the rest of the batch has its vectors");
        assert!(out[9].is_none());
        assert_eq!(pg.embed_failures.load(Ordering::Relaxed), 1);
        assert!(!pg.embed_off.load(Ordering::Relaxed), "one bad text is not an embedder that is down");
        // a build goes on past the bad file: the others get vectors and the bad one keeps its words
        let d = tmp("poison");
        std::fs::write(d.join("a.md"), "the thinking budget closes the think block\n").unwrap();
        std::fs::write(d.join("b.md"), "POISON the emitter pushes tokens\n").unwrap();
        std::fs::write(d.join("c.md"), "the ring promotes experts from the file\n").unwrap();
        let ix = Mutex::new(Index::default());
        let pg = Progress::default();
        assert_eq!(build(&ix, &pg, &d, &Poison, Duration::ZERO, None).unwrap(), 3);
        let g = ix.lock().unwrap();
        assert_eq!(g.counts(), (3, 2));
        assert_eq!(g.hybrid("emitter tokens", None, 1, None)[0].src, "b.md", "found by keywords though it has no vector");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_embedder_that_keeps_failing_is_switched_off_and_the_tool_goes_on_by_keywords() {
        let d = tmp("dead");
        for i in 0..6 {
            std::fs::write(d.join(format!("f{i}.md")), format!("file number {i} talks about the staging ring and the pinned tier\n")).unwrap();
        }
        let ix = Mutex::new(Index::default());
        let pg = Progress::default();
        assert_eq!(build(&ix, &pg, &d, &Dead, Duration::ZERO, None).unwrap(), 6, "the build finishes");
        assert!(pg.embed_off.load(Ordering::Relaxed), "after {MAX_BAD_STREAK} batches with no vector at all");
        let g = ix.lock().unwrap();
        assert_eq!(g.counts().1, 0);
        assert!(!g.hybrid("staging ring", None, 3, None).is_empty());
        drop(g);
        // vectors that are missing are saved as such and read back as such
        let file = d.join("idx.bin");
        ix.lock().unwrap().save(&file).unwrap();
        let loaded = Index::load(&file).unwrap();
        assert_eq!(loaded.counts(), (6, 0));
        assert_eq!(loaded.hybrid("staging ring", None, 3, None).len(), 3);
        let _ = std::fs::remove_dir_all(&d);
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("shadow_sem_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn text_is_cut_into_overlapping_chunks_with_their_first_line() {
        let text: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        let c = chunk_text(&text);
        assert!(c.len() >= 3, "{}", c.len());
        assert_eq!(c[0].0, 1);
        assert_eq!(
            c[1].0, 38,
            "40 lines, 3 of overlap: the second chunk starts at line 38"
        );
        assert!(
            c[1].1.starts_with("line 38\n")
                && c[0].1.contains("line 40")
                && c[1].1.contains("line 40")
        );
        assert!(chunk_text("").is_empty() && chunk_text("\n\n  \n").is_empty());
        // one very long line is clipped, not dropped
        let long = format!("{}\nshort", "x".repeat(5000));
        let c = chunk_text(&long);
        assert_eq!(c[0].1.lines().next().unwrap().len(), LINE_CLIP);
    }

    #[test]
    fn the_right_file_is_found_by_the_words_of_the_question() {
        let d = tmp("find");
        std::fs::create_dir_all(d.join("src")).unwrap();
        std::fs::create_dir_all(d.join("target")).unwrap();
        std::fs::write(d.join("src/dash.rs"), "fn latency_percentiles() {\n    // the dashboard shows p50 and p95 latency percentiles\n}\n").unwrap();
        std::fs::write(
            d.join("src/mem.rs"),
            "fn pinned_memory() {\n    // the pinned host memory tier holds experts\n}\n",
        )
        .unwrap();
        std::fs::write(d.join("target/skip.rs"), "latency percentiles dashboard\n").unwrap();
        std::fs::write(d.join("src/image.bin"), [0u8, 1, 2]).unwrap();
        let ix = Mutex::new(Index::default());
        let pg = Progress::default();
        assert_eq!(
            build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap(),
            2,
            "target/ and .bin are not indexed"
        );
        let q = Fake
            .embed(&["where does the dashboard show latency percentiles".to_string()])
            .unwrap()
            .pop()
            .unwrap();
        let ixg = ix.lock().unwrap();
        let hits = ixg.search(&q, 3, None);
        assert_eq!(hits[0].1, "src/dash.rs");
        assert!(hits[0].0 > hits[1].0);
        // restricted to a folder or a file
        assert!(ixg
            .search(&q, 3, Some("src/mem"))
            .iter()
            .all(|h| h.1 == "src/mem.rs"));
        assert!(answer(&ixg, "where does the dashboard show latency percentiles", Some(&q), 2, None, "").starts_with("1. src/dash.rs:1  (score "));
        drop(ixg);
        // nothing changed: nothing is embedded again; a change is
        assert_eq!(build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap(), 0);
        std::fs::write(
            d.join("src/mem.rs"),
            "fn pinned_memory() {\n    // changed text about the staging buffers and the ring\n}\n",
        )
        .unwrap();
        assert_eq!(build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap(), 1);
        // a deleted file leaves the index
        std::fs::remove_file(d.join("src/dash.rs")).unwrap();
        build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap();
        assert!(ix
            .lock()
            .unwrap()
            .search(&q, 5, None)
            .iter()
            .all(|h| h.1 != "src/dash.rs"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Counts the texts it is asked to embed.
    struct Counting(std::sync::atomic::AtomicUsize);
    impl Embedder for Counting {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            self.0.fetch_add(texts.len(), Ordering::Relaxed);
            Fake.embed(texts)
        }
    }

    #[test]
    fn a_changed_file_embeds_only_its_changed_chunks_and_tables_of_numbers_are_skipped() {
        let d = tmp("reuse");
        let words: String = (0..120)
            .map(|i| format!("this is line number {i} about the dashboard and its panels\n"))
            .collect();
        std::fs::write(d.join("big.md"), &words).unwrap();
        std::fs::write(
            d.join("nums.txt"),
            (0..200)
                .map(|i| format!("{i} {} {} {}\n", i * 3, i * 7, i * 11))
                .collect::<String>(),
        )
        .unwrap();
        let ix = Mutex::new(Index::default());
        let c = Counting(Default::default());
        build(&ix, &Progress::default(), &d, &c, Duration::ZERO, None).unwrap();
        let first = c.0.load(Ordering::Relaxed);
        assert!(first >= 3, "the 120-line file is several chunks: {first}");
        assert!(
            ix.lock()
                .unwrap()
                .search(&Fake.embed(&["x".to_string()]).unwrap()[0], 50, None)
                .iter()
                .all(|h| h.1 == "big.md"),
            "the table of numbers has no chunks"
        );
        // a line added at the end: the chunks before it keep their vectors
        std::thread::sleep(Duration::from_millis(1100));
        std::fs::write(
            d.join("big.md"),
            format!("{words}and one more line about queues\n"),
        )
        .unwrap();
        build(&ix, &Progress::default(), &d, &c, Duration::ZERO, None).unwrap();
        let second = c.0.load(Ordering::Relaxed) - first;
        assert!(
            second >= 1 && second < first,
            "only the last chunk or two are embedded again: {second} of {first}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_folder_that_is_not_a_project_is_refused_and_hidden_folders_are_skipped() {
        let d = tmp("cap");
        std::fs::create_dir_all(d.join(".hidden")).unwrap();
        std::fs::write(d.join(".hidden/secret.md"), "hidden text\n").unwrap();
        std::fs::write(d.join("ok.md"), "plain text about the dashboard\n").unwrap();
        let found: Vec<String> = scan_files(&d).into_iter().map(|f| f.0).collect();
        assert_eq!(
            found,
            vec!["ok.md".to_string()],
            "a hidden folder is not indexed"
        );
        std::fs::create_dir_all(d.join("many")).unwrap();
        for i in 0..(MAX_FILES + 5) {
            std::fs::write(d.join("many").join(format!("f{i}.md")), "x y z\n").unwrap();
        }
        let ix = Mutex::new(Index::default());
        let e = build(&ix, &Progress::default(), &d, &Fake, Duration::ZERO, None).unwrap_err();
        assert!(e.contains("start Shadow from a project folder"), "{e}");
        assert!(ix.lock().unwrap().is_empty(), "nothing was embedded");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_index_survives_a_save_and_a_load_and_outputs_are_added_once() {
        let d = tmp("save");
        std::fs::write(
            d.join("a.md"),
            "# notes\nthe thinking budget closes the think block\n",
        )
        .unwrap();
        let ix = Mutex::new(Index::default());
        build(&ix, &Progress::default(), &d, &Fake, Duration::ZERO, None).unwrap();
        let load = |id: u64| {
            (id == 5).then(|| "(output #5)\nthe emitter pushes tokens to the client\n".to_string())
        };
        assert_eq!(add_outputs(&ix, &[5, 6], &load), 1, "output 6 is not there: skipped");
        assert_eq!(add_outputs(&ix, &[5], &load), 0, "an output is added once");
        assert_eq!(embed_outputs(&ix, &Fake, &Progress::default(), 40), 1, "its vector is made on the way of a search");
        assert_eq!(embed_outputs(&ix, &Fake, &Progress::default(), 40), 0, "once");
        let file = d.join("idx.bin");
        ix.lock().unwrap().save(&file).unwrap();
        let loaded = Index::load(&file).expect("loads");
        assert_eq!(loaded.len(), ix.lock().unwrap().len());
        let q = Fake
            .embed(&["who pushes tokens to the client".to_string()])
            .unwrap()
            .pop()
            .unwrap();
        let hits = loaded.search(&q, 2, None);
        assert_eq!(hits[0].1, "output #5");
        assert!(Index::load(&d.join("missing.bin")).is_none());
        std::fs::write(d.join("bad.bin"), b"not an index").unwrap();
        assert!(Index::load(&d.join("bad.bin")).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The real embedder on the dense code that used to abort it (STATUS 91): every chunk of the kernel tables and CUDA sources gets a vector, nothing fails, no batch is lost.
    /// `cargo test --release -p shadow-serve real_embedder_survives_dense_code -- --ignored` (a minute or two).
    #[test]
    #[ignore]
    fn real_embedder_survives_dense_code() {
        let real = Llama::find().expect("the embedder is in this checkout");
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let d = tmp("dense");
        for (from, to) in [
            ("crates/shadow-cuda/src/kernels.rs", "kernels.rs"),
            ("crates/shadow-cuda/src/pinned.rs", "pinned.rs"),
            ("kernels/quants.cuh", "quants.cuh"),
            ("kernels/iq_tables.cuh", "iq_tables.cuh"),
            ("kernels/matvec.cu", "matvec.cu"),
        ] {
            if let Ok(t) = std::fs::read_to_string(root.join(from)) {
                if t.len() < 400_000 {
                    std::fs::write(d.join(to), t).unwrap();
                }
            }
        }
        let ix = Mutex::new(Index::default());
        let pg = Progress::default();
        build(&ix, &pg, &d, &real, Duration::ZERO, None).unwrap();
        let (n, nv) = ix.lock().unwrap().counts();
        assert!(n > 50, "{n} chunks");
        assert_eq!(pg.embed_failures.load(Ordering::Relaxed), 0, "no text the embedder failed on");
        assert!(!pg.embed_off.load(Ordering::Relaxed));
        assert_eq!(nv, n, "every chunk has a vector");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The whole tool on this crate's own source, with the real embedder (a minute or so the first time): `cargo test --release -p shadow-serve search_this_crate -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn search_this_crate_with_the_real_tool() {
        let ask = |q: &str| -> String {
            let t0 = Instant::now();
            loop {
                let r = tool(
                    &serde_json::json!({"query": q, "max_results": 3}),
                    &[],
                    &|_| None,
                )
                .expect("the tool answers");
                if !r.contains("still being built") || t0.elapsed() > Duration::from_secs(600) {
                    return r;
                }
            }
        };
        let t0 = Instant::now();
        for (q, want) in [
            (
                "where does the server stop a reply that starts repeating itself",
                "engine.rs",
            ),
            (
                "how are older tool outputs shortened when the conversation is too long",
                "engine.rs",
            ),
            (
                "which web pages are fetched and turned into plain text",
                "webtools.rs",
            ),
        ] {
            let r = ask(q);
            println!(
                "--- {q}  ({:.1} s since the start)
{r}",
                t0.elapsed().as_secs_f64()
            );
            assert!(
                r.lines().next().unwrap().contains(want),
                "{q}: {}",
                r.lines().next().unwrap()
            );
        }
    }

    #[test]
    fn the_real_embedder_ranks_a_question_to_the_file_that_answers_it() {
        let Some(real) = Llama::find() else {
            eprintln!("no embedder here: skipped");
            return;
        };
        let d = tmp("real");
        std::fs::write(d.join("dash.md"), "The dashboard shows how fast the engine reads prompts and writes replies, as tokens per second, with percentiles of the latency.\n").unwrap();
        std::fs::write(d.join("cache.md"), "The expert cache keeps the most used mixture-of-experts weights in video memory and evicts the least recently used slot.\n").unwrap();
        std::fs::write(d.join("tools.md"), "The file reading tool returns numbered lines of a source file between a start and an end line.\n").unwrap();
        let ix = Mutex::new(Index::default());
        build(&ix, &Progress::default(), &d, &real, Duration::ZERO, None).unwrap();
        for (q, want) in [
            (
                "how quickly does the server generate text, shown on a chart",
                "dash.md",
            ),
            ("which weights stay on the graphics card", "cache.md"),
            ("read some lines from a code file", "tools.md"),
        ] {
            let qv = real.embed(&[q.to_string()]).unwrap().pop().unwrap();
            let hits = ix
                .lock()
                .unwrap()
                .search(&qv, 1, None)
                .iter()
                .map(|h| h.1.to_string())
                .collect::<Vec<_>>();
            assert_eq!(hits[0], want, "query {q:?}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
