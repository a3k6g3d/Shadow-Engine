//! Search by meaning over the project's files and the session's tool outputs (docs/STATUS.md 78): the `search` tool.
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
const BATCH_TEXTS: usize = 32;
const BATCH_CHARS: usize = 20_000;
const MAGIC: &[u8; 8] = b"SHSIDX01";
const EXTS: &[&str] = &["rs", "md", "py", "js", "ts", "json", "toml", "txt", "cu", "cuh", "c", "cpp", "h", "hpp", "bat", "ps1", "sh", "html", "css", "yml", "yaml"];
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "models", "bin", "bench", "memory", "__pycache__", "ShadowConvCache", "toolcache", "fixtures", "AppData", "Windows", "Program Files", "Program Files (x86)", "ProgramData", "$Recycle.Bin"];
/// More candidate files than this: the working folder is not a project (a home folder, a drive), and nothing is indexed.
const MAX_FILES: usize = 5_000;
/// Files above this size are embedded after everything else.
const BIG_FILE: u64 = 150_000;

/// Something that turns texts into vectors (the real one runs `llama-embedding.exe`; the tests use a fake one).
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;

    /// A question to search with (BGE models want a fixed instruction in front of a short query; MiniLM does not).
    fn embed_query(&self, q: &str) -> Result<Vec<f32>, String> {
        self.embed(&[q.to_string()])?.pop().ok_or_else(|| "no vector for the query".to_string())
    }
}

/// `llama-embedding.exe` with the BGE-small model, run as a subprocess.
pub struct Llama {
    exe: PathBuf,
    model: PathBuf,
    threads: usize,
}

const MODELS: &[&str] = &["models/embed-minilm/all-MiniLM-L6-v2-Q8_0.gguf", "models/bge-small-en-v1.5-q8_0.gguf"];
const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

impl Llama {
    /// `SHADOW_EMBED_EXE` / `SHADOW_EMBED_MODEL`, else the repo's `bin/` and `models/` found by walking up from the executable's and the working folder.
    pub fn find() -> Option<Llama> {
        let env = |k: &str| std::env::var_os(k).map(PathBuf::from).filter(|p| p.is_file());
        let (exe, model) = (env("SHADOW_EMBED_EXE"), env("SHADOW_EMBED_MODEL"));
        let threads = std::env::var("SHADOW_EMBED_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize).clamp(1, 16);
        if let (Some(exe), Some(model)) = (exe.clone(), model.clone()) {
            return Some(Llama { exe, model, threads });
        }
        let starts = [std::env::current_exe().ok(), std::env::current_dir().ok()];
        for start in starts.into_iter().flatten() {
            let mut d: Option<&Path> = Some(start.as_path());
            while let Some(p) = d {
                let e = p.join("bin").join("llama-embedding.exe");
                let m = MODELS.iter().map(|f| p.join(f)).find(|f| f.is_file());
                if let (true, Some(m)) = (e.is_file(), m) {
                    return Some(Llama { exe: exe.unwrap_or(e), model: model.unwrap_or(m), threads });
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
        self.model.file_name().map_or_else(|| "model".to_string(), |n| n.to_string_lossy().to_string())
    }
}

impl Embedder for Llama {
    fn embed_query(&self, q: &str) -> Result<Vec<f32>, String> {
        let text = if self.model_id().to_lowercase().contains("bge") { format!("{BGE_QUERY_PREFIX}{q}") } else { q.to_string() };
        self.embed(&[text])?.pop().ok_or_else(|| "no vector for the query".to_string())
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(texts.len());
        let mut i = 0;
        while i < texts.len() {
            let mut j = i;
            let mut chars = 0;
            while j < texts.len() && j - i < BATCH_TEXTS && (j == i || chars + texts[j].len() < BATCH_CHARS) {
                chars += texts[j].len();
                j += 1;
            }
            // the separator is the only thing the protocol reads literally; an empty text would be skipped
            let joined = texts[i..j].iter().map(|t| { let t = t.replace(SEP, " "); if t.trim().is_empty() { " ".to_string() } else { t } }).collect::<Vec<_>>().join(SEP);
            let mut cmd = std::process::Command::new(&self.exe);
            cmd.args(["-m"]).arg(&self.model)
                .args(["-p"]).arg(&joined)
                .args(["--embd-separator", SEP, "--embd-output-format", "json", "-t"]).arg(self.threads.to_string())
                .stderr(std::process::Stdio::null());
            // below normal priority and no console window: it runs beside the engine and yields the CPU to it
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                cmd.creation_flags(0x0000_4000 | 0x0800_0000);
            }
            let o = cmd.output().map_err(|e| format!("could not run the embedder: {e}"))?;
            if !o.status.success() {
                return Err(format!("the embedder failed (exit {:?})", o.status.code()));
            }
            let v: Value = serde_json::from_slice(&o.stdout).map_err(|e| format!("the embedder's reply is not JSON: {e}"))?;
            let data = v.get("data").and_then(Value::as_array).ok_or("the embedder's reply has no data")?;
            if data.len() != j - i {
                return Err(format!("the embedder returned {} vectors for {} texts", data.len(), j - i));
            }
            for item in data {
                let e = item.get("embedding").and_then(Value::as_array).ok_or("an item has no embedding")?;
                let mut vec: Vec<f32> = e.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
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
    v.iter().map(|x| (x * 127.0).round().clamp(-127.0, 127.0) as i8).collect()
}

/// Cosine of a normalized query and a stored (normalized, quantized) vector.
fn score(q: &[f32], c: &[i8]) -> f32 {
    q.iter().zip(c).map(|(a, b)| a * (*b as f32)).sum::<f32>() / 127.0
}

/// `text` cut into chunks of ~`CHUNK_LINES` lines / `CHUNK_CHARS` characters with `OVERLAP_LINES` of overlap: (first line number, 1-indexed; the chunk's text).
pub fn chunk_text(text: &str) -> Vec<(u32, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let clip = |l: &str| -> String { l.chars().take(LINE_CLIP).collect() };
    let mut out = vec![];
    let mut i = 0;
    while i < lines.len() {
        let (mut j, mut chars) = (i, 0usize);
        while j < lines.len() && j - i < CHUNK_LINES && (j == i || chars + lines[j].chars().count().min(LINE_CLIP) < CHUNK_CHARS) {
            chars += lines[j].chars().count().min(LINE_CLIP) + 1;
            j += 1;
        }
        let body = lines[i..j].iter().map(|l| clip(l)).collect::<Vec<_>>().join("\n");
        if !body.trim().is_empty() {
            out.push((i as u32 + 1, body));
        }
        if j >= lines.len() {
            break;
        }
        i = if j - i > OVERLAP_LINES { j - OVERLAP_LINES } else { j };
    }
    out
}

struct Chunk {
    src: String,
    line: u32,
    text: String,
    v: Vec<i8>,
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
        let mut hits: Vec<(f32, &Chunk)> = self.chunks.iter().filter(|c| under.is_none_or(|u| c.src.starts_with(u))).map(|c| (score(q, &c.v), c)).collect();
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hits.into_iter().take(k).map(|(s, c)| (s, c.src.as_str(), c.line, c.text.as_str())).collect()
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
            b.extend(c.v.iter().map(|x| *x as u8));
        }
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &b).and_then(|_| std::fs::rename(&tmp, path)).map_err(|e| format!("could not write the index: {e}"))
    }

    pub fn load(path: &Path) -> Option<Index> {
        let b = std::fs::read(path).ok()?;
        if b.len() < 12 || &b[..8] != MAGIC {
            return None;
        }
        let mut p = 8usize;
        let u32_at = |p: &mut usize| -> Option<u32> { let v = u32::from_le_bytes(b.get(*p..*p + 4)?.try_into().ok()?); *p += 4; Some(v) };
        let u64_at = |p: &mut usize| -> Option<u64> { let v = u64::from_le_bytes(b.get(*p..*p + 8)?.try_into().ok()?); *p += 8; Some(v) };
        let str_at = |p: &mut usize, n: usize| -> Option<String> { let s = String::from_utf8(b.get(*p..*p + n)?.to_vec()).ok()?; *p += n; Some(s) };
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
            let v: Vec<i8> = b.get(p..p + DIM)?.iter().map(|x| *x as i8).collect();
            p += DIM;
            idx.chunks.push(Chunk { src, line, text, v });
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
}

fn stamp_of(meta: &std::fs::Metadata) -> (u64, u64) {
    let m = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    (m, meta.len())
}

/// The files of the project worth indexing under `root`: (relative path with `/`, full path, stamp).
pub fn scan_files(root: &Path) -> Vec<(String, PathBuf, (u64, u64))> {
    let mut out = vec![];
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            let Ok(meta) = e.metadata() else { continue };
            if meta.is_dir() {
                if SKIP_DIRS.contains(&name.as_str()) || name.starts_with("target-") || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
            } else if meta.len() > 0 && meta.len() <= MAX_FILE_BYTES && !name.contains("tables") {
                let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
                if EXTS.contains(&ext.as_str()) {
                    let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
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

/// Chunks and embeds one source's text; a chunk whose text is the same as one `old` holds keeps its vector (only what changed in a file is embedded again).
fn embed_source(src: &str, text: &str, emb: &dyn Embedder, old: &HashMap<String, Vec<i8>>) -> Result<Vec<Chunk>, String> {
    let parts: Vec<(u32, String)> = chunk_text(text).into_iter().filter(|(_, t)| wordy(t)).collect();
    if parts.is_empty() {
        return Ok(vec![]);
    }
    // the file's name in front of each chunk, so a query that names a file or a module finds it
    let fresh: Vec<usize> = (0..parts.len()).filter(|i| !old.contains_key(&parts[*i].1)).collect();
    let texts: Vec<String> = fresh.iter().map(|i| format!("{src}\n{}", parts[*i].1)).collect();
    let mut vecs = if texts.is_empty() { vec![] } else { emb.embed(&texts)? }.into_iter();
    let mut by_index: HashMap<usize, Vec<i8>> = fresh.iter().map(|i| (*i, quantize(&vecs.next().unwrap_or_default()))).collect();
    Ok(parts.into_iter().enumerate().map(|(i, (line, text))| {
        let v = by_index.remove(&i).or_else(|| old.get(&text).cloned()).unwrap_or_default();
        Chunk { src: src.to_string(), line, text, v }
    }).collect())
}

/// The vectors the index already holds for a source, by chunk text.
fn old_vectors(ix: &Index, src: &str) -> HashMap<String, Vec<i8>> {
    ix.chunks.iter().filter(|c| c.src == src && c.v.len() == DIM).map(|c| (c.text.clone(), c.v.clone())).collect()
}

/// Brings `index` up to date with the files under `root`: embeds what is new or changed, drops what is gone.  The index is locked only to read and to write, not while
/// the embedder runs.  `pause` is slept between files (the build runs beside the engine).  Returns the number of files embedded.
pub fn build(index: &Mutex<Index>, progress: &Progress, root: &Path, emb: &dyn Embedder, pause: Duration, save_to: Option<&Path>) -> Result<usize, String> {
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
    progress.total.store(todo.len(), Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    {
        let mut ix = index.lock().unwrap();
        for g in &gone {
            ix.forget(g);
        }
    }
    let mut embedded = 0;
    let mut last_save = Instant::now();
    for (rel, path, st) in todo {
        match std::fs::read(&path) {
            Ok(b) if !b.contains(&0) => {
                let text = String::from_utf8_lossy(&b).to_string();
                let old = old_vectors(&index.lock().unwrap(), &rel);
                let chunks = embed_source(&rel, &text, emb, &old)?;
                index.lock().unwrap().replace(&rel, st, chunks);
                embedded += 1;
            }
            _ => index.lock().unwrap().replace(&rel, st, vec![]), // binary or unreadable: remembered, not indexed
        }
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
    Ok(embedded)
}

/// Embeds the tool outputs of the session's log (`ids`, read with `load`) that are not in the index yet, at most `max_chunks` chunks' worth in one call.
pub fn add_outputs(index: &Mutex<Index>, ids: &[u64], load: &dyn Fn(u64) -> Option<String>, emb: &dyn Embedder, max_chunks: usize) -> Result<usize, String> {
    let mut added = 0;
    for id in ids {
        let src = format!("output #{id}");
        if index.lock().unwrap().seen.contains_key(&src) {
            continue;
        }
        let Some(text) = load(*id) else { continue };
        let n = chunk_text(&text).len();
        if added > 0 && added + n > max_chunks {
            break;
        }
        let chunks = embed_source(&src, &text, emb, &HashMap::new())?;
        index.lock().unwrap().replace(&src, (0, text.len() as u64), chunks);
        added += n;
    }
    Ok(added)
}

/// A query answered: the best chunks as text.
pub fn answer(index: &Index, qv: &[f32], k: usize, under: Option<&str>, note: &str) -> String {
    let hits = index.search(qv, k, under);
    if hits.is_empty() {
        return format!("nothing indexed matches{}{}", under.map_or(String::new(), |u| format!(" under {u}")), note);
    }
    let mut out = String::new();
    for (i, (s, src, line, text)) in hits.iter().enumerate() {
        let shown: Vec<&str> = text.lines().take(8).collect();
        out.push_str(&format!("{}. {src}:{line}  (score {s:.2})\n{}\n", i + 1, shown.iter().map(|l| format!("   {l}")).collect::<Vec<_>>().join("\n")));
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
    let base = std::env::var_os("SHADOW_TOOL_LOG_DIR").map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(|a| PathBuf::from(a).join("Shadow").join("toolcache")))
        .unwrap_or_else(|| std::env::temp_dir().join("shadow-toolcache"));
    let h = root.to_string_lossy().bytes().chain(model_id.bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    base.join(format!("semindex-{h:016x}.bin"))
}

/// The `search` tool: `query`, optional `max_results` (default 6) and `path` (only under this folder or file).
pub fn tool(args: &Value, output_ids: &[u64], load_output: &dyn Fn(u64) -> Option<String>) -> Result<String, String> {
    let query = args.get("query").and_then(Value::as_str).unwrap_or("").trim().to_string();
    if query.is_empty() {
        return Err("give a text to search for (query)".into());
    }
    let k = args.get("max_results").and_then(Value::as_u64).unwrap_or(6).clamp(1, 15) as usize;
    let under = args.get("path").and_then(Value::as_str).map(|p| p.replace('\\', "/").trim_start_matches("./").to_string()).filter(|p| !p.is_empty());
    static EMB: OnceLock<Option<Llama>> = OnceLock::new();
    let llama: &'static Llama = match EMB.get_or_init(Llama::find) {
        Some(l) => l,
        None => return Err("the embedder was not found (bin/llama-embedding.exe and models/embed-minilm/all-MiniLM-L6-v2-Q8_0.gguf or models/bge-small-en-v1.5-q8_0.gguf, or SHADOW_EMBED_EXE / SHADOW_EMBED_MODEL)".into()),
    };
    let emb: &'static dyn Embedder = llama;
    let g = GLOBAL.get_or_init(|| {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let cache = cache_path(&root, &llama.model_id());
        Global { index: Mutex::new(Index::load(&cache).unwrap_or_default()), progress: Progress::default(), root, cache, last_scan: Mutex::new(None) }
    });
    // refresh: a scan at most every 30 s, on a thread; the first call waits a while for it, later ones for a short while
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
    while g.progress.building.load(Ordering::SeqCst) && t0.elapsed() < Duration::from_secs(wait) {
        std::thread::sleep(Duration::from_millis(200));
    }
    add_outputs(&g.index, output_ids, load_output, emb, 40)?;
    let qv = emb.embed_query(&query)?;
    let note = if g.progress.building.load(Ordering::SeqCst) {
        format!("\n(the index is still being built: {} of {} changed files done, so results may be incomplete; ask again in a minute)", g.progress.done.load(Ordering::Relaxed), g.progress.total.load(Ordering::Relaxed))
    } else if let Some(e) = g.progress.error.lock().unwrap().clone() {
        format!("\n(the last refresh of the index failed: {e})")
    } else {
        String::new()
    };
    let ix = g.index.lock().unwrap();
    Ok(answer(&ix, &qv, k, under.as_deref(), &note))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words into the dimensions (a bag of words): the same words, the same direction.
    struct Fake;
    impl Embedder for Fake {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts.iter().map(|t| {
                let mut v = vec![0f32; DIM];
                for w in t.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
                    let h = w.bytes().fold(7usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
                    v[h % DIM] += 1.0;
                }
                normalize(&mut v);
                v
            }).collect())
        }
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
        assert_eq!(c[1].0, 38, "40 lines, 3 of overlap: the second chunk starts at line 38");
        assert!(c[1].1.starts_with("line 38\n") && c[0].1.contains("line 40") && c[1].1.contains("line 40"));
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
        std::fs::write(d.join("src/mem.rs"), "fn pinned_memory() {\n    // the pinned host memory tier holds experts\n}\n").unwrap();
        std::fs::write(d.join("target/skip.rs"), "latency percentiles dashboard\n").unwrap();
        std::fs::write(d.join("src/image.bin"), [0u8, 1, 2]).unwrap();
        let ix = Mutex::new(Index::default());
        let pg = Progress::default();
        assert_eq!(build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap(), 2, "target/ and .bin are not indexed");
        let q = Fake.embed(&["where does the dashboard show latency percentiles".to_string()]).unwrap().pop().unwrap();
        let ixg = ix.lock().unwrap();
        let hits = ixg.search(&q, 3, None);
        assert_eq!(hits[0].1, "src/dash.rs");
        assert!(hits[0].0 > hits[1].0);
        // restricted to a folder or a file
        assert!(ixg.search(&q, 3, Some("src/mem")).iter().all(|h| h.1 == "src/mem.rs"));
        assert!(answer(&ixg, &q, 2, None, "").starts_with("1. src/dash.rs:1  (score "));
        drop(ixg);
        // nothing changed: nothing is embedded again; a change is
        assert_eq!(build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap(), 0);
        std::fs::write(d.join("src/mem.rs"), "fn pinned_memory() {\n    // changed text about the staging buffers and the ring\n}\n").unwrap();
        assert_eq!(build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap(), 1);
        // a deleted file leaves the index
        std::fs::remove_file(d.join("src/dash.rs")).unwrap();
        build(&ix, &pg, &d, &Fake, Duration::ZERO, None).unwrap();
        assert!(ix.lock().unwrap().search(&q, 5, None).iter().all(|h| h.1 != "src/dash.rs"));
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
        let words: String = (0..120).map(|i| format!("this is line number {i} about the dashboard and its panels\n")).collect();
        std::fs::write(d.join("big.md"), &words).unwrap();
        std::fs::write(d.join("nums.txt"), (0..200).map(|i| format!("{i} {} {} {}\n", i * 3, i * 7, i * 11)).collect::<String>()).unwrap();
        let ix = Mutex::new(Index::default());
        let c = Counting(Default::default());
        build(&ix, &Progress::default(), &d, &c, Duration::ZERO, None).unwrap();
        let first = c.0.load(Ordering::Relaxed);
        assert!(first >= 3, "the 120-line file is several chunks: {first}");
        assert!(ix.lock().unwrap().search(&Fake.embed(&["x".to_string()]).unwrap()[0], 50, None).iter().all(|h| h.1 == "big.md"), "the table of numbers has no chunks");
        // a line added at the end: the chunks before it keep their vectors
        std::thread::sleep(Duration::from_millis(1100));
        std::fs::write(d.join("big.md"), format!("{words}and one more line about queues\n")).unwrap();
        build(&ix, &Progress::default(), &d, &c, Duration::ZERO, None).unwrap();
        let second = c.0.load(Ordering::Relaxed) - first;
        assert!(second >= 1 && second < first, "only the last chunk or two are embedded again: {second} of {first}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_folder_that_is_not_a_project_is_refused_and_hidden_folders_are_skipped() {
        let d = tmp("cap");
        std::fs::create_dir_all(d.join(".hidden")).unwrap();
        std::fs::write(d.join(".hidden/secret.md"), "hidden text\n").unwrap();
        std::fs::write(d.join("ok.md"), "plain text about the dashboard\n").unwrap();
        let found: Vec<String> = scan_files(&d).into_iter().map(|f| f.0).collect();
        assert_eq!(found, vec!["ok.md".to_string()], "a hidden folder is not indexed");
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
        std::fs::write(d.join("a.md"), "# notes\nthe thinking budget closes the think block\n").unwrap();
        let ix = Mutex::new(Index::default());
        build(&ix, &Progress::default(), &d, &Fake, Duration::ZERO, None).unwrap();
        let load = |id: u64| (id == 5).then(|| "(output #5)\nthe emitter pushes tokens to the client\n".to_string());
        assert_eq!(add_outputs(&ix, &[5, 6], &load, &Fake, 40).unwrap(), 1, "output 6 is not there: skipped");
        assert_eq!(add_outputs(&ix, &[5], &load, &Fake, 40).unwrap(), 0, "an output is embedded once");
        let file = d.join("idx.bin");
        ix.lock().unwrap().save(&file).unwrap();
        let loaded = Index::load(&file).expect("loads");
        assert_eq!(loaded.len(), ix.lock().unwrap().len());
        let q = Fake.embed(&["who pushes tokens to the client".to_string()]).unwrap().pop().unwrap();
        let hits = loaded.search(&q, 2, None);
        assert_eq!(hits[0].1, "output #5");
        assert!(Index::load(&d.join("missing.bin")).is_none());
        std::fs::write(d.join("bad.bin"), b"not an index").unwrap();
        assert!(Index::load(&d.join("bad.bin")).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The whole tool on this crate's own source, with the real embedder (a minute or so the first time): `cargo test --release -p shadow-serve search_this_crate -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn search_this_crate_with_the_real_tool() {
        let ask = |q: &str| -> String {
            let t0 = Instant::now();
            loop {
                let r = tool(&serde_json::json!({"query": q, "max_results": 3}), &[], &|_| None).expect("the tool answers");
                if !r.contains("still being built") || t0.elapsed() > Duration::from_secs(600) {
                    return r;
                }
            }
        };
        let t0 = Instant::now();
        for (q, want) in [("where does the server stop a reply that starts repeating itself", "engine.rs"), ("how are older tool outputs shortened when the conversation is too long", "engine.rs"), ("which web pages are fetched and turned into plain text", "webtools.rs")] {
            let r = ask(q);
            println!("--- {q}  ({:.1} s since the start)
{r}", t0.elapsed().as_secs_f64());
            assert!(r.lines().next().unwrap().contains(want), "{q}: {}", r.lines().next().unwrap());
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
        for (q, want) in [("how quickly does the server generate text, shown on a chart", "dash.md"), ("which weights stay on the graphics card", "cache.md"), ("read some lines from a code file", "tools.md")] {
            let qv = real.embed(&[q.to_string()]).unwrap().pop().unwrap();
            let hits = ix.lock().unwrap().search(&qv, 1, None).iter().map(|h| h.1.to_string()).collect::<Vec<_>>();
            assert_eq!(hits[0], want, "query {q:?}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
