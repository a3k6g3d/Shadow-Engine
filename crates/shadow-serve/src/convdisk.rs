//! Conversation snapshots on disk (the NVMe the model is on, by default), written by a background thread after the prompt has been read, so a restart (or a chat
//! left for an hour) keeps them and the RAM only holds the token lists.
//!
//! **Format v2: the rows are stored once.**  A snapshot is the session at a token boundary: the attention layers' per-position rows (keys, values, indexer rows,
//! pooled blocks; ~57 KiB a token) and a recurrent part (the DeltaNet states, the PLE state: tens of MiB, whatever the length).  The rows at a position depend
//! only on the tokens before it, so every conversation that starts the same way has the same rows for that start; v1 wrote them again in every file (44
//! snapshots of branches of a few conversations, 20 GiB).  Now:
//! - a **segment** file (`s<id>.seg`) holds the rows of positions `start..end` of some token sequence, once;
//! - a **state** file (`t<stamp>-<n>.cst`) holds one snapshot's tokens, its recurrent part, and the list of **pieces** (segment, positions) whose rows make up
//!   positions `0..n`.  A new snapshot reuses the longest start it shares with a stored one (a piece of an old segment, cut where the tokens diverge) and
//!   writes a segment only for the rest.
//!
//! An older snapshot whose tokens are a start of a newer one is dropped as before (its state file; its segments stay while the newer one uses them); a segment goes
//! when no state uses it; over the budget, the oldest state goes.  Writing is one pass over the new rows (a `Vec` of them is the only copy made); reading one pass
//! into the snapshot's own buffers.  Files live in `ShadowConvCache/<model id>/`; a lock file keeps a second `shadow serve` on the same model off the directory.
//! Files of the first format (`.cnv`, one file a snapshot) are of no use any more and are deleted when found.

use shadow_cuda::session::Snapshot;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

const OLD_MAGIC: &[u8; 8] = b"SHDWCNV1";
const SEG_MAGIC: &[u8; 8] = b"SHDWSG02";
const ST_MAGIC: &[u8; 8] = b"SHDWST02";
/// A start shorter than this is not worth sharing (a state of its own is simpler to keep and to lose).
const MIN_SHARE: usize = 64;
/// The most bytes of snapshots waiting for the writer (more are dropped, not queued: the RAM matters more than one more file).
const MAX_PENDING: u64 = 2 << 30;

#[derive(Clone, Debug, PartialEq)]
struct Piece {
    seg: u64,
    a: usize,
    b: usize,
}

struct Seg {
    path: PathBuf,
    bytes: u64,
    start: usize,
    end: usize,
    /// false while the writer has not finished the file (a later snapshot may already share its rows: the writer works in order)
    ready: bool,
}

struct State {
    tokens: Vec<u32>,
    path: PathBuf,
    bytes: u64,
    stamp: u64,
    pieces: Vec<Piece>,
    /// false while the writer has not finished the file: it can be shared (a later snapshot is planned on it) but not restored from
    ready: bool,
}

#[derive(Default)]
struct Index {
    states: Vec<State>,
    segs: HashMap<u64, Seg>,
}

impl Index {
    fn total(&self) -> u64 {
        self.states.iter().map(|s| s.bytes).sum::<u64>() + self.segs.values().map(|s| s.bytes).sum::<u64>()
    }

    /// Deletes the segments no state uses (a snapshot still being written counts: it names its segments from the moment it is planned).
    fn collect(&mut self) {
        let used: HashSet<u64> = self.states.iter().flat_map(|s| s.pieces.iter().map(|p| p.seg)).collect();
        let dead: Vec<u64> = self.segs.iter().filter(|(id, _)| !used.contains(id)).map(|(id, _)| *id).collect();
        for id in dead {
            if let Some(s) = self.segs.remove(&id) {
                let _ = std::fs::remove_file(&s.path);
            }
        }
    }

    /// Deletes the oldest written states (and what only they used) until the files fit `budget`; the newest state always stays.
    fn trim(&mut self, budget: u64) {
        while self.states.iter().filter(|s| s.ready).count() > 1 && self.total() > budget {
            let oldest = self.states.iter().enumerate().filter(|(_, s)| s.ready).min_by_key(|(_, s)| s.stamp).map(|(i, _)| i).unwrap();
            let s = self.states.remove(oldest);
            let _ = std::fs::remove_file(&s.path);
            self.collect();
        }
    }
}

/// What `ConvDisk::plan` decided for a new snapshot: the positions `0..m` come from pieces of segments already stored (or being written), the rest is a segment of
/// its own.  The snapshot is in the index from then on, so the next one is planned on it even before the writer is done.
pub struct Plan {
    /// The first position whose rows the snapshot's own segment must hold (`Snapshot::disk_parts(m)`).
    pub m: usize,
    /// the snapshot's own segment (positions `m..n`), when it has one
    seg: Option<(u64, PathBuf)>,
    state_path: PathBuf,
    stamp: u64,
}

enum Job {
    Write { tokens: Vec<u32>, plan: Plan, state: Vec<u8>, rows: Vec<u8> },
    /// Answers when every job before it is done (for tests and for a clean exit).
    Flush(Sender<()>),
}

pub struct ConvDisk {
    dir: PathBuf,
    model_id: u64,
    index: Arc<Mutex<Index>>,
    tx: Sender<Job>,
    pending: Arc<AtomicU64>,
    next_id: AtomicU64,
    /// Held for the life of the process: a second `shadow serve` on this model finds the directory locked.
    _lock: Option<std::fs::File>,
}

fn fnv(s: &[u8]) -> u64 {
    s.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

fn mtime_ms(md: &std::fs::Metadata) -> u64 {
    md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_millis() as u64)
}

fn rd_u64(f: &mut std::fs::File) -> Option<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

/// The header of a segment file: (id, start, end), or None for a file that is not a whole segment of this model.
fn read_seg_header(path: &Path, model_id: u64) -> Option<(u64, usize, usize, u64)> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut m = [0u8; 8];
    f.read_exact(&mut m).ok()?;
    if &m != SEG_MAGIC || rd_u64(&mut f)? != model_id {
        return None;
    }
    let (id, start, end, len) = (rd_u64(&mut f)?, rd_u64(&mut f)? as usize, rd_u64(&mut f)? as usize, rd_u64(&mut f)?);
    (f.metadata().ok()?.len() == 48 + len && start <= end).then_some((id, start, end, len))
}

/// A state file's header: (tokens, pieces, the state part's offset and length), or None for a file that is not a whole state of this model.
fn read_state_header(path: &Path, model_id: u64) -> Option<(Vec<u32>, Vec<Piece>, u64, u64)> {
    let mut f = std::fs::File::open(path).ok()?;
    let flen = f.metadata().ok()?.len();
    let mut m = [0u8; 8];
    f.read_exact(&mut m).ok()?;
    if &m != ST_MAGIC || rd_u64(&mut f)? != model_id {
        return None;
    }
    let n = rd_u64(&mut f)? as usize;
    if n > (1 << 24) {
        return None;
    }
    let mut t = vec![0u8; n * 4];
    f.read_exact(&mut t).ok()?;
    let tokens: Vec<u32> = t.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let np = rd_u64(&mut f)? as usize;
    if np > (1 << 20) {
        return None;
    }
    let mut pieces = Vec::with_capacity(np);
    for _ in 0..np {
        pieces.push(Piece { seg: rd_u64(&mut f)?, a: rd_u64(&mut f)? as usize, b: rd_u64(&mut f)? as usize });
    }
    let len = rd_u64(&mut f)?;
    let at = 8 + 8 + 8 + 4 * n as u64 + 8 + 24 * np as u64 + 8;
    (flen == at + len).then_some((tokens, pieces, at, len))
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Whether `pieces` tile `0..n` and each lies inside its segment's range.
fn pieces_valid(pieces: &[Piece], n: usize, segs: &HashMap<u64, Seg>) -> bool {
    let mut at = 0;
    for p in pieces {
        let Some(s) = segs.get(&p.seg) else { return false };
        if p.a != at || p.b <= p.a || p.a < s.start || p.b > s.end {
            return false;
        }
        at = p.b;
    }
    at == n
}

impl ConvDisk {
    /// `SHADOW_CONV_DIR` (default: `ShadowConvCache` at the root of the model's drive) holds a directory a model; at most `SHADOW_CONV_DISK_GIB` (default 20, per
    /// model; 0 = off) of files in it.
    pub fn open(model: &Path) -> Option<ConvDisk> {
        let budget = (std::env::var("SHADOW_CONV_DISK_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(20.0) * 1073741824.0) as u64;
        if budget == 0 {
            return None;
        }
        let root = match std::env::var("SHADOW_CONV_DIR") {
            Ok(d) if !d.is_empty() => PathBuf::from(d),
            _ => model.ancestors().last().map(Path::to_path_buf).unwrap_or_default().join("ShadowConvCache"),
        };
        let len = std::fs::metadata(model).map_or(0, |m| m.len());
        let model_id = fnv(model.to_string_lossy().as_bytes()) ^ len.rotate_left(17);
        Self::open_in(&root, model_id, budget, std::time::Duration::from_secs(15))
    }

    /// The cache of model `model_id` under `root`, waiting up to `lock_wait` for another process to let go of it (a `shadow serve` that is being replaced).
    pub fn open_in(root: &Path, model_id: u64, budget: u64, lock_wait: std::time::Duration) -> Option<ConvDisk> {
        let dir = root.join(format!("{model_id:016x}"));
        std::fs::create_dir_all(&dir).ok()?;
        let lock = Self::lock(&dir, lock_wait)?;
        Self::tidy_root(root, model_id);
        // everything this directory holds: whole files are indexed, the rest goes
        let mut index = Index::default();
        let mut states: Vec<(PathBuf, Vec<u32>, Vec<Piece>, u64, u64)> = vec![];
        for e in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = e.path();
            let Ok(md) = e.metadata() else { continue };
            match p.extension().and_then(|x| x.to_str()) {
                Some("seg") => match read_seg_header(&p, model_id) {
                    Some((id, start, end, _)) => {
                        index.segs.insert(id, Seg { path: p, bytes: md.len(), start, end, ready: true });
                    }
                    None => {
                        let _ = std::fs::remove_file(&p);
                    }
                },
                Some("cst") => match read_state_header(&p, model_id) {
                    Some((tokens, pieces, _, _)) => states.push((p, tokens, pieces, md.len(), mtime_ms(&md))),
                    None => {
                        let _ = std::fs::remove_file(&p);
                    }
                },
                Some("tmp") => {
                    let _ = std::fs::remove_file(&p); // we hold the lock: nobody is writing it
                }
                _ => {}
            }
        }
        // a state whose segments are gone or do not fit is no use; of exact duplicates the newest stays
        states.sort_by_key(|s| std::cmp::Reverse(s.4));
        let mut seen: HashSet<Vec<u32>> = HashSet::new();
        for (path, tokens, pieces, bytes, stamp) in states {
            if !pieces_valid(&pieces, tokens.len(), &index.segs) || !seen.insert(tokens.clone()) {
                let _ = std::fs::remove_file(&path);
                continue;
            }
            index.states.push(State { tokens, path, bytes, stamp, pieces, ready: true });
        }
        index.collect();
        index.trim(budget);
        if !index.states.is_empty() {
            eprintln!(
                "shadow serve: {} conversation snapshot(s) on disk in {} ({:.2} GiB, {} shared row segment(s))",
                index.states.len(),
                dir.display(),
                index.total() as f64 / 1073741824.0,
                index.segs.len()
            );
        }
        let index = Arc::new(Mutex::new(index));
        let pending = Arc::new(AtomicU64::new(0));
        let (tx, rx) = channel::<Job>();
        let (ix, d2, pend) = (index.clone(), dir.clone(), pending.clone());
        std::thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                match job {
                    Job::Flush(done) => {
                        let _ = done.send(());
                    }
                    Job::Write { tokens, plan, state, rows } => {
                        let bytes = (state.len() + rows.len()) as u64;
                        let t0 = std::time::Instant::now();
                        match Self::write_job(&ix, model_id, budget, &tokens, &plan, &state, &rows) {
                            Ok((n_states, total)) => eprintln!(
                                "shadow serve: conversation snapshot of {} tokens written to disk ({:.2} GiB new in {:.2} s; {} on disk, {:.2} GiB)",
                                tokens.len(),
                                bytes as f64 / 1073741824.0,
                                t0.elapsed().as_secs_f64(),
                                n_states,
                                total as f64 / 1073741824.0
                            ),
                            Err(e) => {
                                eprintln!("shadow serve: could not write a conversation snapshot to {}: {e}", d2.display());
                                Self::abandon(&ix, &plan);
                            }
                        }
                        pend.fetch_sub(bytes, Ordering::SeqCst);
                    }
                }
            }
        });
        Some(ConvDisk { dir, model_id, index, tx, pending, next_id: AtomicU64::new(1), _lock: lock })
    }

    /// The directory's lock (Windows: a file opened with no sharing, released by the system when the process ends, however it ends).
    fn lock(dir: &Path, wait: std::time::Duration) -> Option<Option<std::fs::File>> {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            let path = dir.join(".lock");
            let t0 = std::time::Instant::now();
            loop {
                match std::fs::OpenOptions::new().write(true).create(true).truncate(false).share_mode(0).open(&path) {
                    Ok(f) => return Some(Some(f)),
                    Err(_) if t0.elapsed() < wait => std::thread::sleep(std::time::Duration::from_millis(250)),
                    Err(_) => {
                        eprintln!("shadow serve: another shadow serve is using the conversation cache of this model ({}): this one runs without the disk copy", dir.display());
                        return None;
                    }
                }
            }
        }
        #[cfg(not(windows))]
        {
            let _ = (dir, wait);
            Some(None)
        }
    }

    /// The cache root's old leftovers: first-format snapshot files of this model (unreadable now) and `.tmp` files a writer died on (older than ten minutes: no
    /// live writer takes that long).
    fn tidy_root(root: &Path, model_id: u64) {
        let Ok(rd) = std::fs::read_dir(root) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if !e.file_type().is_ok_and(|t| t.is_file()) {
                continue;
            }
            match p.extension().and_then(|x| x.to_str()) {
                Some("cnv") => {
                    // the first format's file of this model (of no use now), or a broken one; another model's stays until that model's server tidies it
                    let mut h = [0u8; 16];
                    let head = std::fs::File::open(&p).and_then(|mut f| f.read_exact(&mut h)).is_ok();
                    if !head || &h[0..8] != OLD_MAGIC || u64::from_le_bytes(h[8..16].try_into().unwrap()) == model_id {
                        let _ = std::fs::remove_file(&p);
                    }
                }
                Some("tmp") => {
                    let old = e.metadata().ok().is_some_and(|m| now_ms().saturating_sub(mtime_ms(&m)) > 600_000);
                    if old {
                        let _ = std::fs::remove_file(&p);
                    }
                }
                _ => {}
            }
        }
    }

    /// Writes a planned snapshot (its own segment, then its state), marks both written, and drops what it makes stale and what the budget cannot hold.
    fn write_job(ix: &Arc<Mutex<Index>>, model_id: u64, budget: u64, tokens: &[u32], plan: &Plan, state: &[u8], rows: &[u8]) -> Result<(usize, u64), String> {
        let n = tokens.len();
        let write = |path: &Path, parts: &[&[u8]]| -> Result<(), String> {
            let tmp = path.with_extension("tmp");
            let r = (|| -> std::io::Result<()> {
                let mut f = std::fs::File::create(&tmp)?;
                for p in parts {
                    f.write_all(p)?;
                }
                f.flush()?;
                drop(f);
                std::fs::rename(&tmp, path)
            })();
            if r.is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
            r.map_err(|e| e.to_string())
        };
        let pieces = {
            let g = ix.lock().unwrap();
            g.states.iter().find(|s| s.path == plan.state_path).map(|s| s.pieces.clone()).ok_or("the snapshot was dropped before it was written")?
        };
        let mut seg_bytes = 0u64;
        if let Some((id, path)) = &plan.seg {
            let mut head = Vec::with_capacity(48);
            head.extend_from_slice(SEG_MAGIC);
            for v in [model_id, *id, plan.m as u64, n as u64, rows.len() as u64] {
                head.extend_from_slice(&v.to_le_bytes());
            }
            write(path, &[&head, rows])?;
            seg_bytes = (head.len() + rows.len()) as u64;
        }
        let mut head = Vec::with_capacity(32 + tokens.len() * 4 + pieces.len() * 24 + 8);
        head.extend_from_slice(ST_MAGIC);
        head.extend_from_slice(&model_id.to_le_bytes());
        head.extend_from_slice(&(n as u64).to_le_bytes());
        for t in tokens {
            head.extend_from_slice(&t.to_le_bytes());
        }
        head.extend_from_slice(&(pieces.len() as u64).to_le_bytes());
        for p in &pieces {
            for v in [p.seg, p.a as u64, p.b as u64] {
                head.extend_from_slice(&v.to_le_bytes());
            }
        }
        head.extend_from_slice(&(state.len() as u64).to_le_bytes());
        write(&plan.state_path, &[&head, state])?;
        let mut g = ix.lock().unwrap();
        if let Some((id, _)) = &plan.seg {
            if let Some(sg) = g.segs.get_mut(id) {
                sg.ready = true;
                sg.bytes = seg_bytes;
            }
        }
        if let Some(st) = g.states.iter_mut().find(|s| s.path == plan.state_path) {
            st.ready = true;
            st.bytes = (head.len() + state.len()) as u64;
        }
        // an older snapshot of this conversation (its tokens a start of these) goes, its rows stay with this one
        let me = plan.state_path.clone();
        g.states.retain(|e| {
            let stale = e.ready && e.path != me && e.tokens.len() <= n && tokens[..e.tokens.len()] == e.tokens[..];
            if stale {
                let _ = std::fs::remove_file(&e.path);
            }
            !stale
        });
        g.collect();
        g.trim(budget);
        Ok((g.states.iter().filter(|s| s.ready).count(), g.total()))
    }

    /// A planned snapshot that could not be written: out of the index again (a later one that was planned on its rows will fail to load and be forgotten).
    fn abandon(ix: &Arc<Mutex<Index>>, plan: &Plan) {
        let mut g = ix.lock().unwrap();
        g.states.retain(|s| s.path != plan.state_path);
        if let Some((id, path)) = &plan.seg {
            g.segs.remove(id);
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_file(&plan.state_path);
        g.collect();
    }

    /// How a snapshot of `tokens` is stored: the start it shares with a snapshot already stored, or planned, is reused (pieces of its segments) and the rest is a new
    /// segment.  The snapshot enters the index now, unwritten.  `None`: these very tokens are stored already.
    pub fn plan(&self, tokens: &[u32]) -> Option<Plan> {
        let n = tokens.len();
        let mut ix = self.index.lock().unwrap();
        if let Some(s) = ix.states.iter_mut().find(|s| s.tokens == tokens) {
            s.stamp = now_ms();
            return None;
        }
        let (best, m) = ix.states.iter().enumerate().map(|(i, s)| (i, lcp(&s.tokens, tokens))).max_by_key(|&(_, m)| m).unwrap_or((0, 0));
        let mut pieces = vec![];
        let mut m = if m >= MIN_SHARE { m } else { 0 };
        if m > 0 {
            let mut at = 0;
            for p in &ix.states[best].pieces {
                if at >= m {
                    break;
                }
                let b = p.b.min(m);
                pieces.push(Piece { seg: p.seg, a: p.a, b });
                at = b;
            }
            if pieces.is_empty() || at != m {
                m = 0;
                pieces.clear();
            }
        }
        let stamp = now_ms();
        let id = (stamp << 16) ^ self.next_id.fetch_add(1, Ordering::SeqCst);
        let seg = (m < n).then(|| (id, self.dir.join(format!("s{id:016x}.seg"))));
        if let Some((sid, path)) = &seg {
            pieces.push(Piece { seg: *sid, a: m, b: n });
            ix.segs.insert(*sid, Seg { path: path.clone(), bytes: 0, start: m, end: n, ready: false });
        }
        let state_path = self.dir.join(format!("t{stamp:016x}-{n}-{:04x}.cst", id & 0xffff));
        ix.states.push(State { tokens: tokens.to_vec(), path: state_path.clone(), bytes: 0, stamp, pieces, ready: false });
        Some(Plan { m, seg, state_path, stamp })
    }

    /// Hand a snapshot (a host snapshot with the KV) to the writer thread: planned, split (`Snapshot::disk_parts`), queued.
    pub fn save_snapshot(&self, snap: &Snapshot<'_>) -> Result<(), String> {
        let Some(plan) = self.plan(snap.tokens()) else { return Ok(()) };
        let (state, rows) = match snap.disk_parts(plan.m) {
            Ok(p) => p,
            Err(e) => {
                Self::abandon(&self.index, &plan);
                return Err(e);
            }
        };
        let bytes = (state.len() + rows.len()) as u64;
        if self.pending.load(Ordering::SeqCst) > 0 && self.pending.load(Ordering::SeqCst) + bytes > MAX_PENDING {
            Self::abandon(&self.index, &plan);
            return Err("the writer is behind: this snapshot is not written".into());
        }
        self.pending.fetch_add(bytes, Ordering::SeqCst);
        let abandoned = Plan { m: plan.m, seg: plan.seg.clone(), state_path: plan.state_path.clone(), stamp: plan.stamp };
        self.tx.send(Job::Write { tokens: snap.tokens().to_vec(), plan, state, rows }).map_err(|_| {
            self.pending.fetch_sub(bytes, Ordering::SeqCst);
            Self::abandon(&self.index, &abandoned);
            "the writer has stopped".to_string()
        })
    }

    /// Blocks until everything handed to the writer so far is on disk.
    pub fn flush(&self) {
        let (tx, rx) = channel();
        if self.tx.send(Job::Flush(tx)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// The longest stored snapshot whose tokens are a start of `ids` (and shorter than it, and longer than `min`): its length and path.
    pub fn best(&self, ids: &[u32], min: usize) -> Option<(usize, PathBuf)> {
        let es = self.index.lock().unwrap();
        es.states
            .iter()
            .filter(|e| e.ready && e.tokens.len() > min && e.tokens.len() < ids.len() && ids[..e.tokens.len()] == e.tokens[..])
            .max_by_key(|e| e.tokens.len())
            .map(|e| (e.tokens.len(), e.path.clone()))
    }

    /// A stored snapshot, rebuilt from its state file and the pieces of segments it names (one pass into the snapshot's own buffers).  A snapshot that cannot be
    /// read (a segment gone, a file cut short) is forgotten, and its files with it.
    pub fn load<'c>(&self, path: &Path) -> Result<Snapshot<'c>, String> {
        let r = self.load_inner(path);
        if r.is_err() {
            let mut ix = self.index.lock().unwrap();
            ix.states.retain(|s| s.path != path);
            let _ = std::fs::remove_file(path);
            ix.collect();
        } else if let Ok(f) = std::fs::File::options().write(true).open(path) {
            let _ = f.set_modified(std::time::SystemTime::now());
            let mut ix = self.index.lock().unwrap();
            if let Some(s) = ix.states.iter_mut().find(|s| s.path == path) {
                s.stamp = now_ms();
            }
        }
        r
    }

    fn load_inner<'c>(&self, path: &Path) -> Result<Snapshot<'c>, String> {
        let (tokens, pieces, at, len) = read_state_header(path, self.model_id).ok_or("not this model's snapshot, or a partial file")?;
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        std::io::Seek::seek(&mut f, std::io::SeekFrom::Start(at)).map_err(|e| e.to_string())?;
        let mut state = vec![0u8; len as usize];
        f.read_exact(&mut state).map_err(|e| e.to_string())?;
        let n = tokens.len();
        let mut b = Snapshot::builder_from_disk(&state, tokens)?;
        drop(state);
        let paths: Vec<Option<PathBuf>> = {
            let ix = self.index.lock().unwrap();
            pieces.iter().map(|p| ix.segs.get(&p.seg).map(|s| s.path.clone())).collect()
        };
        let mut covered = 0;
        for (p, sp) in pieces.iter().zip(paths) {
            let sp = sp.ok_or("a row segment of this snapshot is gone")?;
            let bytes = std::fs::read(&sp).map_err(|e| e.to_string())?;
            if bytes.len() < 48 || &bytes[0..8] != SEG_MAGIC {
                return Err("a row segment is not one".into());
            }
            let rd = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
            let (mid, sid, start, end, plen) = (rd(8), rd(16), rd(24) as usize, rd(32) as usize, rd(40) as usize);
            if mid != self.model_id || sid != p.seg || bytes.len() != 48 + plen || p.a < start || p.b > end || p.a != covered {
                return Err("a row segment does not fit its snapshot".into());
            }
            b.add_segment_rows(p.a, p.b, start, end, &bytes[48..])?;
            covered = p.b;
        }
        if covered != n {
            return Err("the stored pieces stop short of the snapshot".into());
        }
        b.finish()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// (snapshots, segments, bytes) now on disk.
    pub fn stats(&self) -> (usize, usize, u64) {
        let ix = self.index.lock().unwrap();
        (ix.states.iter().filter(|s| s.ready).count(), ix.segs.values().filter(|s| s.ready).count(), ix.total())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shadow_cuda::session::test_snapshot;
    use std::time::Duration;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("shadow-convdisk-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// `n` tokens: the first `shared` are those of conversation 0, the rest depend on `branch`.
    fn toks(shared: usize, branch: u32, n: usize) -> Vec<u32> {
        (0..n).map(|i| if i < shared { (i as u32 * 13) % 997 } else { 1000 + branch * 100_000 + i as u32 }).collect()
    }

    fn same(a: &Snapshot, b: &Snapshot) -> bool {
        a.tokens() == b.tokens() && a.host_bytes() == b.host_bytes() && a.disk_parts(0).unwrap() == b.disk_parts(0).unwrap()
    }

    #[test]
    fn branches_share_their_start_and_every_snapshot_reads_back() {
        let root = tmpdir("share");
        let d = ConvDisk::open_in(&root, 7, 1 << 40, Duration::ZERO).unwrap();
        let a = test_snapshot(toks(600, 0, 600));
        let b = test_snapshot(toks(400, 1, 700)); // shares 400 positions with `a`
        let c = test_snapshot(toks(400, 2, 650)); // and so does `c`
        for s in [&a, &b, &c] {
            d.save_snapshot(s).unwrap();
            d.flush();
        }
        let (states, segs, bytes) = d.stats();
        assert_eq!((states, segs), (3, 3), "three snapshots, one segment each (a: 0..600, b: 400..700, c: 400..650)");
        let alone = a.disk_parts(0).unwrap().1.len() + b.disk_parts(0).unwrap().1.len() + c.disk_parts(0).unwrap().1.len();
        assert!((bytes as usize) < alone, "the shared 400 positions are stored once: {bytes} against {alone}");
        for (s, t) in [(&a, toks(600, 0, 600)), (&b, toks(400, 1, 700)), (&c, toks(400, 2, 650))] {
            let mut ids = t.clone();
            ids.push(5);
            let (n, path) = d.best(&ids, 0).expect("stored");
            assert_eq!(n, t.len());
            let back = d.load(&path).unwrap();
            assert!(same(&back, s), "the snapshot read back is the one written");
        }
        // a restart finds them all
        drop(d);
        let d = ConvDisk::open_in(&root, 7, 1 << 40, Duration::ZERO).unwrap();
        assert_eq!(d.stats().0, 3);
        let mut ids = toks(400, 1, 700);
        ids.push(1);
        let back = d.load(&d.best(&ids, 0).unwrap().1).unwrap();
        assert!(same(&back, &b));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_snapshot_is_planned_on_one_still_being_written() {
        let root = tmpdir("inflight");
        let d = ConvDisk::open_in(&root, 14, 1 << 40, Duration::ZERO).unwrap();
        let a = test_snapshot(toks(500, 0, 500));
        let b = test_snapshot(toks(450, 1, 520));
        // both handed over back to back, no flush between: b is planned while a's files are not on disk
        d.save_snapshot(&a).unwrap();
        d.save_snapshot(&b).unwrap();
        d.flush();
        let (states, segs, _) = d.stats();
        assert_eq!((states, segs), (2, 2));
        let sizes: Vec<u64> = std::fs::read_dir(d.dir()).unwrap().flatten().filter(|e| e.path().extension().is_some_and(|x| x == "seg")).map(|e| e.metadata().unwrap().len()).collect();
        let full = a.disk_parts(0).unwrap().1.len() as u64;
        assert!(sizes.iter().any(|&x| x < full / 4), "b's own segment holds only its 70 positions beyond the 450 it shares: {sizes:?}");
        let mut ids = toks(450, 1, 520);
        ids.push(3);
        assert!(same(&d.load(&d.best(&ids, 0).unwrap().1).unwrap(), &b));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_longer_snapshot_of_the_conversation_replaces_the_older_but_keeps_its_rows() {
        let root = tmpdir("stale");
        let d = ConvDisk::open_in(&root, 8, 1 << 40, Duration::ZERO).unwrap();
        let a = test_snapshot(toks(500, 0, 500));
        let b = test_snapshot(toks(500, 0, 900)); // the conversation grew (same first 500 tokens)
        d.save_snapshot(&a).unwrap();
        d.flush();
        d.save_snapshot(&b).unwrap();
        d.flush();
        let (states, segs, _) = d.stats();
        assert_eq!((states, segs), (1, 2), "only the longer snapshot, on the two segments it is made of");
        let mut ids = toks(500, 0, 900);
        ids.push(9);
        assert!(same(&d.load(&d.best(&ids, 0).unwrap().1).unwrap(), &b));
        // the same tokens again: nothing is written
        d.save_snapshot(&b).unwrap();
        d.flush();
        assert_eq!(d.stats().1, 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn over_the_budget_the_oldest_goes_with_the_rows_only_it_used() {
        let root = tmpdir("budget");
        let one = test_snapshot(toks(0, 0, 400)).disk_parts(0).map(|(s, r)| s.len() + r.len()).unwrap() as u64;
        let d = ConvDisk::open_in(&root, 9, one * 5 / 2, Duration::ZERO).unwrap(); // room for two and a half
        for k in 0..4u32 {
            d.save_snapshot(&test_snapshot(toks(0, k + 1, 400))).unwrap();
            d.flush();
            std::thread::sleep(Duration::from_millis(5));
        }
        let (states, segs, bytes) = d.stats();
        assert!(states <= 3 && states >= 2, "states {states}");
        assert_eq!(segs, states, "no segment outlives the snapshots that used it");
        assert!(bytes <= one * 5 / 2 + 8, "{bytes} over the budget");
        // the files on disk are exactly what the index says
        let files = std::fs::read_dir(root.join(format!("{:016x}", 9))).unwrap().flatten().filter(|e| e.path().extension().is_some_and(|x| x == "seg" || x == "cst")).count();
        assert_eq!(files, states + segs);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn leftovers_are_tidied_at_start_and_a_second_process_stays_out() {
        let root = tmpdir("tidy");
        let dir = root.join(format!("{:016x}", 11));
        std::fs::create_dir_all(&dir).unwrap();
        // a dead writer's temporary file, a first-format file of this model and one of another, a cut segment
        std::fs::write(dir.join("t0000-1.tmp"), vec![1u8; 100]).unwrap();
        std::fs::write(dir.join("s00.seg"), b"SHDWSG02 cut").unwrap();
        let mut old = Vec::new();
        old.extend_from_slice(OLD_MAGIC);
        old.extend_from_slice(&11u64.to_le_bytes());
        old.extend_from_slice(&0u64.to_le_bytes());
        old.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(root.join("0001-1.cnv"), &old).unwrap();
        let mut other = old.clone();
        other[8..16].copy_from_slice(&12u64.to_le_bytes());
        std::fs::write(root.join("0002-1.cnv"), &other).unwrap();
        let d = ConvDisk::open_in(&root, 11, 1 << 40, Duration::ZERO).unwrap();
        assert!(!dir.join("t0000-1.tmp").exists() && !dir.join("s00.seg").exists());
        assert!(!root.join("0001-1.cnv").exists(), "this model's first-format file is deleted");
        assert!(root.join("0002-1.cnv").exists(), "another model's is not ours to delete");
        #[cfg(windows)]
        assert!(ConvDisk::open_in(&root, 11, 1 << 40, Duration::ZERO).is_none(), "a second opener of the same model's cache is refused");
        // another model's cache is another directory
        assert!(ConvDisk::open_in(&root, 12, 1 << 40, Duration::ZERO).is_some());
        drop(d);
        assert!(ConvDisk::open_in(&root, 11, 1 << 40, Duration::ZERO).is_some(), "the lock goes with its holder");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_broken_snapshot_is_forgotten_not_retried() {
        let root = tmpdir("broken");
        let d = ConvDisk::open_in(&root, 13, 1 << 40, Duration::ZERO).unwrap();
        d.save_snapshot(&test_snapshot(toks(0, 0, 300))).unwrap();
        d.flush();
        let mut ids = toks(0, 0, 300);
        ids.push(1);
        let (_, path) = d.best(&ids, 0).unwrap();
        // lose its segment
        for e in std::fs::read_dir(d.dir()).unwrap().flatten() {
            if e.path().extension().is_some_and(|x| x == "seg") {
                std::fs::remove_file(e.path()).unwrap();
            }
        }
        assert!(d.load(&path).is_err());
        assert!(d.best(&ids, 0).is_none());
        assert_eq!(d.stats().0, 0);
        let _ = std::fs::remove_dir_all(&root);
    }
}
