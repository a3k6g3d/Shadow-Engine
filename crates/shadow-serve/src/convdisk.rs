//! Conversation snapshots on disk (the NVMe the model is on, by default): a file a snapshot (`Snapshot::to_bytes`) with its tokens in front, written by a background
//! thread after the prompt has been read, listed in an index that is rebuilt from the files' headers at start, so a restart keeps them and the RAM only holds the token lists.
//! File: "SHDWCNV1", the model's id (u64), the token count (u64), the tokens (u32 each), the payload length (u64), the payload.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

const MAGIC: &[u8; 8] = b"SHDWCNV1";

struct Entry {
    tokens: Vec<u32>,
    path: PathBuf,
    bytes: u64,
    stamp: u64,
}

pub struct ConvDisk {
    dir: PathBuf,
    model_id: u64,
    entries: Arc<Mutex<Vec<Entry>>>,
    tx: Sender<(Vec<u32>, Vec<u8>)>,
}

fn fnv(s: &[u8]) -> u64 {
    s.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// The header of a snapshot file: (model id, tokens), or None for a file that is not one of ours.
fn read_header(path: &Path) -> Option<(u64, Vec<u32>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = [0u8; 24];
    f.read_exact(&mut h).ok()?;
    if &h[0..8] != MAGIC {
        return None;
    }
    let id = u64::from_le_bytes(h[8..16].try_into().unwrap());
    let n = u64::from_le_bytes(h[16..24].try_into().unwrap()) as usize;
    if n > (1 << 24) {
        return None;
    }
    let mut t = vec![0u8; n * 4];
    f.read_exact(&mut t).ok()?;
    Some((id, t.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()))
}

impl ConvDisk {
    /// `SHADOW_CONV_DIR` (default: `ShadowConvCache` at the root of the model's drive), at most `SHADOW_CONV_DISK_GIB` (default 20; 0 = off) of files.
    pub fn open(model: &Path) -> Option<ConvDisk> {
        let budget = (std::env::var("SHADOW_CONV_DISK_GIB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(20.0) * 1073741824.0) as u64;
        if budget == 0 {
            return None;
        }
        let dir = match std::env::var("SHADOW_CONV_DIR") {
            Ok(d) if !d.is_empty() => PathBuf::from(d),
            _ => {
                let root = model.ancestors().last().map(Path::to_path_buf).unwrap_or_default();
                root.join("ShadowConvCache")
            }
        };
        std::fs::create_dir_all(&dir).ok()?;
        let len = std::fs::metadata(model).map_or(0, |m| m.len());
        let model_id = fnv(model.to_string_lossy().as_bytes()) ^ len.rotate_left(17);
        let mut found = vec![];
        for e in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "cnv") {
                continue;
            }
            match read_header(&p) {
                Some((id, tokens)) if id == model_id => {
                    let md = e.metadata().ok()?;
                    let stamp = md.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_millis() as u64);
                    found.push(Entry { tokens, path: p, bytes: md.len(), stamp });
                }
                _ => {
                    // another model's file stays; a broken or partial one goes
                    if read_header(&p).is_none() {
                        let _ = std::fs::remove_file(&p);
                    }
                }
            }
        }
        if !found.is_empty() {
            eprintln!("shadow serve: {} conversation snapshot(s) on disk in {} ({:.2} GiB)", found.len(), dir.display(),
                      found.iter().map(|e| e.bytes).sum::<u64>() as f64 / 1073741824.0);
        }
        let entries = Arc::new(Mutex::new(found));
        let (tx, rx) = channel::<(Vec<u32>, Vec<u8>)>();
        let (e2, d2) = (entries.clone(), dir.clone());
        std::thread::spawn(move || {
            while let Ok((tokens, payload)) = rx.recv() {
                let stamp = now_ms();
                let path = d2.join(format!("{stamp:016x}-{}.cnv", tokens.len()));
                let tmp = path.with_extension("tmp");
                let mut head = Vec::with_capacity(32 + tokens.len() * 4);
                head.extend_from_slice(MAGIC);
                head.extend_from_slice(&model_id.to_le_bytes());
                head.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
                for t in &tokens {
                    head.extend_from_slice(&t.to_le_bytes());
                }
                head.extend_from_slice(&(payload.len() as u64).to_le_bytes());
                let t0 = std::time::Instant::now();
                let ok = {
                    use std::io::Write;
                    std::fs::File::create(&tmp).and_then(|mut f| f.write_all(&head).and_then(|_| f.write_all(&payload)).and_then(|_| f.flush())).is_ok()
                        && std::fs::rename(&tmp, &path).is_ok()
                };
                if !ok {
                    let _ = std::fs::remove_file(&tmp);
                    eprintln!("shadow serve: could not write a conversation snapshot to {}", d2.display());
                    continue;
                }
                let bytes = (head.len() + payload.len()) as u64;
                let mut es = e2.lock().unwrap();
                // an older snapshot of this conversation (its tokens a start of these) is replaced
                es.retain(|e| {
                    let stale = e.tokens.len() <= tokens.len() && tokens[..e.tokens.len()] == e.tokens[..];
                    if stale {
                        let _ = std::fs::remove_file(&e.path);
                    }
                    !stale
                });
                es.push(Entry { tokens, path, bytes, stamp });
                while es.len() > 1 && es.iter().map(|e| e.bytes).sum::<u64>() > budget {
                    let oldest = es.iter().enumerate().min_by_key(|(_, e)| e.stamp).map(|(i, _)| i).unwrap();
                    let _ = std::fs::remove_file(&es[oldest].path);
                    es.remove(oldest);
                }
                eprintln!("shadow serve: conversation snapshot written to disk ({:.2} GiB in {:.2} s; {} on disk)", bytes as f64 / 1073741824.0, t0.elapsed().as_secs_f64(), es.len());
            }
        });
        Some(ConvDisk { dir, model_id, entries, tx })
    }

    /// Hand a snapshot (payload from `Snapshot::to_bytes`) for these tokens to the writer thread.
    pub fn save(&self, tokens: &[u32], payload: Vec<u8>) {
        let _ = self.tx.send((tokens.to_vec(), payload));
    }

    /// The longest stored snapshot whose tokens are a start of `ids` (and shorter than it, and longer than `min`): its length and path.
    pub fn best(&self, ids: &[u32], min: usize) -> Option<(usize, PathBuf)> {
        let es = self.entries.lock().unwrap();
        es.iter().filter(|e| e.tokens.len() > min && e.tokens.len() < ids.len() && ids[..e.tokens.len()] == e.tokens[..])
            .max_by_key(|e| e.tokens.len()).map(|e| (e.tokens.len(), e.path.clone()))
    }

    /// A snapshot file's payload.
    pub fn load(&self, path: &Path) -> Result<Vec<u8>, String> {
        let b = std::fs::read(path).map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(b.get(16..24).ok_or("short file")?.try_into().unwrap()) as usize;
        let at = 24 + n * 4;
        let len = u64::from_le_bytes(b.get(at..at + 8).ok_or("short file")?.try_into().unwrap()) as usize;
        let id = u64::from_le_bytes(b[8..16].try_into().unwrap());
        if id != self.model_id || b.len() != at + 8 + len {
            return Err("not this model's snapshot, or a partial file".into());
        }
        Ok(b[at + 8..].to_vec())
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}
