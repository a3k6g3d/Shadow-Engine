//! Positional reads spread over a few worker threads, each with its own handles on the model's files.
//!
//! Windows serializes I/O on one synchronous file handle (docs/STATUS.md, stage 3: threads sharing `Model`'s handle
//! only added spawn cost), so every worker opens the shards itself.  The reads stay buffered: expert reads repeat, and
//! the file cache serves the repeats at memory speed.  The workers live as long as the pool, so a batch costs a few
//! channel messages, not thread spawns.

use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;

/// How a batch is cut up: pieces of at most this many bytes go to the workers round-robin.
pub const CHUNK: usize = 256 << 10;

struct Job {
    shard: usize,
    offset: u64,
    ptr: usize,
    len: usize,
}

pub struct ReadPool {
    work: Vec<Sender<Job>>,
    done: Mutex<Receiver<std::io::Result<()>>>,
    /// answers still owed for pieces sent by `submit` (taken, in `done`'s lock order, by `wait_async` or the next `read_many`)
    owed: Mutex<usize>,
    threads: Vec<JoinHandle<()>>,
}

impl ReadPool {
    /// `workers` threads, each with its own handle on every one of `paths`.
    pub fn new(paths: &[PathBuf], workers: usize) -> Result<ReadPool, String> {
        let (done_tx, done) = channel();
        let mut work = Vec::with_capacity(workers);
        let mut threads = Vec::with_capacity(workers);
        for w in 0..workers {
            let files = paths.iter().map(|p| File::open(p).map_err(|e| format!("{}: {e}", p.display())))
                .collect::<Result<Vec<_>, _>>()?;
            let (tx, rx) = channel::<Job>();
            let done_tx = done_tx.clone();
            let t = std::thread::Builder::new().name(format!("shadow-read-{w}")).spawn(move || {
                for j in rx {
                    // SAFETY: `read_many` holds the `&mut` borrow these bytes came from until every job of its batch
                    // has answered on `done`, so the slice is live and nothing else touches it meanwhile.
                    let buf = unsafe { std::slice::from_raw_parts_mut(j.ptr as *mut u8, j.len) };
                    if done_tx.send(crate::forward::read_at(&files[j.shard], buf, j.offset)).is_err() {
                        break;
                    }
                }
            }).map_err(|e| format!("read worker: {e}"))?;
            work.push(tx);
            threads.push(t);
        }
        Ok(ReadPool { work, done: Mutex::new(done), owed: Mutex::new(0), threads })
    }

    /// Reads every `(shard, offset, buffer)` in full, in parallel, and returns when all of them are in.
    pub fn read_many(&self, jobs: &mut [(usize, u64, &mut [u8])]) -> std::io::Result<()> {
        let done = self.done.lock().unwrap_or_else(|p| p.into_inner());
        let mut result = self.drain(&done);
        let mut sent = 0usize;
        'send: for (shard, offset, buf) in jobs.iter_mut() {
            for (k, piece) in buf.chunks_mut(CHUNK).enumerate() {
                let job = Job { shard: *shard, offset: *offset + (k * CHUNK) as u64, ptr: piece.as_mut_ptr() as usize,
                                len: piece.len() };
                if self.work[sent % self.work.len()].send(job).is_err() {
                    result = Err(std::io::Error::other("read worker gone"));
                    break 'send;
                }
                sent += 1;
            }
        }
        for _ in 0..sent {
            // Waiting for every answer, errors included, is what keeps the borrows above sound.
            let r = done.recv().expect("read worker gone");
            if result.is_ok() {
                result = r;
            }
        }
        result
    }
}

impl ReadPool {
    /// The answers owed for `submit`ted pieces.
    fn drain(&self, done: &Receiver<std::io::Result<()>>) -> std::io::Result<()> {
        let owed = std::mem::take(&mut *self.owed.lock().unwrap_or_else(|p| p.into_inner()));
        let mut result = Ok(());
        for _ in 0..owed {
            let r = done.recv().expect("read worker gone");
            if result.is_ok() {
                result = r;
            }
        }
        result
    }

    /// `read_many` without the wait: queues every `(shard, offset, buffer)` and returns.
    ///
    /// # Safety
    /// The buffers must stay allocated and untouched until `wait_async` (or any `read_many`, which waits first) has returned.
    pub unsafe fn submit(&self, jobs: &mut [(usize, u64, &mut [u8])]) {
        let _done = self.done.lock().unwrap_or_else(|p| p.into_inner());
        let mut owed = self.owed.lock().unwrap_or_else(|p| p.into_inner());
        let mut sent = *owed;
        for (shard, offset, buf) in jobs.iter_mut() {
            for (k, piece) in buf.chunks_mut(CHUNK).enumerate() {
                let job = Job { shard: *shard, offset: *offset + (k * CHUNK) as u64, ptr: piece.as_mut_ptr() as usize,
                                len: piece.len() };
                self.work[sent % self.work.len()].send(job).expect("read worker gone");
                sent += 1;
                *owed += 1;
            }
        }
    }

    /// Returns when every `submit`ted piece is in.
    pub fn wait_async(&self) -> std::io::Result<()> {
        let done = self.done.lock().unwrap_or_else(|p| p.into_inner());
        self.drain(&done)
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        self.work.clear(); // closes the channels, so every worker's loop ends
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
