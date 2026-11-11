//! Positional reads spread over a few worker threads, each with its own handles on the model's files.
//!
//! Windows serializes I/O on one synchronous file handle (docs/STATUS.md, stage 3: threads sharing `Model`'s handle
//! only added spawn cost), so every worker opens the shards itself.  The reads stay buffered: expert reads repeat, and
//! the file cache serves the repeats at memory speed.  The workers live as long as the pool, so a batch costs a few
//! channel messages, not thread spawns.
//!
//! Opt-in direct-I/O path (SHADOW_DIRECT_IO=1): genuinely cold bytes (drive-tier experts) are read unbuffered straight
//! into pinned staging buffers via an overlapped event pool (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED), skipping
//! one round-trip across the DRAM bus (two touches instead of three).

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
    #[cfg(windows)]
    direct: Option<DirectIoPool>,
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

        #[cfg(windows)]
        let direct = if std::env::var("SHADOW_DIRECT_IO").as_deref() == Ok("1") {
            match DirectIoPool::new(paths, 64) {
                Ok(d) => {
                    eprintln!("shadow-model: Direct I/O enabled (SHADOW_DIRECT_IO=1, unbuffered overlapped event pool)");
                    Some(d)
                }
                Err(e) => {
                    eprintln!("shadow-model: Direct I/O failed to open ({e}), falling back to buffered");
                    None
                }
            }
        } else {
            None
        };

        Ok(ReadPool {
            work,
            done: Mutex::new(done),
            owed: Mutex::new(0),
            threads,
            #[cfg(windows)]
            direct,
        })
    }

    /// Reads every `(shard, offset, buffer)` in full, in parallel, and returns when all of them are in.
    pub fn read_many(&self, jobs: &mut [(usize, u64, &mut [u8])]) -> std::io::Result<()> {
        #[cfg(windows)]
        if let Some(ref direct) = self.direct {
            match direct.read_many(jobs) {
                Ok(()) => return Ok(()),
                // the direct pool has drained what it queued (no read is left writing into these buffers): read them buffered
                Err(e) => direct_failed(&e),
            }
        }

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
        #[cfg(windows)]
        if let Some(ref direct) = self.direct {
            match unsafe { direct.submit(jobs) } {
                Ok(()) => return,
                Err(e) => direct_failed(&e), // drained; queue the batch on the buffered workers below
            }
        }

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
        // a batch may be on either path (the buffered one after a direct failure): wait for both
        #[cfg(windows)]
        let direct_result = match self.direct {
            Some(ref direct) => direct.wait_async(),
            None => Ok(()),
        };
        let done = self.done.lock().unwrap_or_else(|p| p.into_inner());
        let result = self.drain(&done);
        #[cfg(windows)]
        direct_result?;
        result
    }
}

/// A direct-I/O batch failed (e.g. the pages of an unbuffered read could not be locked with most of RAM pinned): say so once, then read buffered.
#[cfg(windows)]
fn direct_failed(e: &std::io::Error) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("shadow-model: a direct-I/O read failed ({e}); this and later failing batches are read buffered"));
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        self.work.clear(); // closes the channels, so every worker's loop ends
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Direct I/O (unbuffered, overlapped event pool) on Windows (SHADOW_DIRECT_IO=1)
// ------------------------------------------------------------------------------------------------

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    pub type Handle = *mut c_void;
    pub const INVALID_HANDLE_VALUE: Handle = -1isize as Handle;
    pub const GENERIC_READ: u32 = 0x8000_0000;
    pub const FILE_SHARE_ALL: u32 = 0x0000_0007; // READ | WRITE | DELETE
    pub const OPEN_EXISTING: u32 = 3;
    pub const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    pub const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const ERROR_IO_PENDING: u32 = 997;

    #[repr(C)]
    pub struct Overlapped {
        pub internal: usize,
        pub internal_high: usize,
        pub offset: u32,
        pub offset_high: u32,
        pub event: Handle,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateFileW(
            filename: *const u16,
            access: u32,
            share: u32,
            security: *mut c_void,
            disposition: u32,
            flags: u32,
            template: Handle,
        ) -> Handle;
        pub fn CloseHandle(handle: Handle) -> i32;
        pub fn CreateEventW(security: *mut c_void, manual_reset: i32, initial_state: i32, name: *const u16) -> Handle;
        pub fn ResetEvent(event: Handle) -> i32;
        pub fn ReadFile(
            file: Handle,
            buffer: *mut c_void,
            bytes_to_read: u32,
            bytes_read: *mut u32,
            overlapped: *mut Overlapped,
        ) -> i32;
        pub fn GetOverlappedResult(
            file: Handle,
            overlapped: *mut Overlapped,
            bytes_transferred: *mut u32,
            wait: i32,
        ) -> i32;
        pub fn GetLastError() -> u32;
        pub fn VirtualAlloc(address: *mut c_void, size: usize, allocation_type: u32, protect: u32) -> *mut c_void;
        pub fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> i32;
    }
}

#[cfg(windows)]
const DIRECT_SLOT_SIZE: usize = 1024 * 1024; // 1 MiB per slot, 4096-aligned
/// The most slots (reads in flight) the pool keeps: one slot a 512 KiB piece, kept for reuse, so without a cap a batch of thousands of small reads (the n-gram
/// rows) committed thousands of MiB and, with most of RAM pinned, the commit ran out (docs/STATUS.md 2026-11-04).  At the cap the oldest read is finished first.
const DIRECT_MAX_SLOTS: usize = 128;

#[cfg(windows)]
struct PendingDirect {
    slot: usize,
    shard: usize,
    dst_ptr: usize,
    lead: usize,
    len: usize,
}

#[cfg(windows)]
struct DirectSlots {
    events: Vec<win::Handle>,
    ovs: Vec<Box<win::Overlapped>>,
    buffers: Vec<*mut u8>,
    free: Vec<usize>,
    pending: Vec<PendingDirect>,
}

#[cfg(windows)]
impl DirectSlots {
    fn alloc_slot(&mut self) -> Result<usize, std::io::Error> {
        if let Some(s) = self.free.pop() {
            return Ok(s);
        }
        let buf = unsafe {
            win::VirtualAlloc(
                std::ptr::null_mut(),
                DIRECT_SLOT_SIZE,
                win::MEM_COMMIT | win::MEM_RESERVE,
                win::PAGE_READWRITE,
            )
        } as *mut u8;
        if buf.is_null() {
            return Err(std::io::Error::other("VirtualAlloc failed for direct-I/O slot"));
        }
        let ev = unsafe { win::CreateEventW(std::ptr::null_mut(), 1, 0, std::ptr::null()) };
        if ev.is_null() {
            unsafe { win::VirtualFree(buf as *mut std::ffi::c_void, 0, win::MEM_RELEASE) };
            return Err(std::io::Error::other("CreateEventW failed for direct-I/O slot"));
        }
        let idx = self.buffers.len();
        self.buffers.push(buf);
        self.events.push(ev);
        self.ovs.push(Box::new(win::Overlapped {
            internal: 0,
            internal_high: 0,
            offset: 0,
            offset_high: 0,
            event: ev,
        }));
        Ok(idx)
    }
}

#[cfg(windows)]
pub struct DirectIoPool {
    files: Vec<win::Handle>,
    slots: Mutex<DirectSlots>,
}

#[cfg(windows)]
unsafe impl Send for DirectIoPool {}
#[cfg(windows)]
unsafe impl Sync for DirectIoPool {}

#[cfg(windows)]
impl DirectIoPool {
    pub fn new(paths: &[PathBuf], initial_depth: usize) -> Result<DirectIoPool, String> {
        use std::os::windows::ffi::OsStrExt;
        let mut files = Vec::with_capacity(paths.len());
        for p in paths {
            let mut wide: Vec<u16> = p.as_os_str().encode_wide().collect();
            wide.push(0);
            let h = unsafe {
                win::CreateFileW(
                    wide.as_ptr(),
                    win::GENERIC_READ,
                    win::FILE_SHARE_ALL,
                    std::ptr::null_mut(),
                    win::OPEN_EXISTING,
                    win::FILE_FLAG_NO_BUFFERING | win::FILE_FLAG_OVERLAPPED,
                    std::ptr::null_mut(),
                )
            };
            if h == win::INVALID_HANDLE_VALUE {
                let err = unsafe { win::GetLastError() };
                for &f in &files {
                    unsafe { win::CloseHandle(f) };
                }
                return Err(format!("DirectIoPool: CreateFileW({}): error {err}", p.display()));
            }
            files.push(h);
        }
        let mut slots = DirectSlots {
            events: Vec::with_capacity(initial_depth),
            ovs: Vec::with_capacity(initial_depth),
            buffers: Vec::with_capacity(initial_depth),
            free: Vec::with_capacity(initial_depth),
            pending: Vec::with_capacity(initial_depth),
        };
        for _ in 0..initial_depth {
            let idx = slots.alloc_slot().map_err(|e| e.to_string())?;
            slots.free.push(idx);
        }
        Ok(DirectIoPool {
            files,
            slots: Mutex::new(slots),
        })
    }

    /// Queues unbuffered overlapped direct-I/O reads into the event pool.  On an error it first waits for every read it queued (of this batch and
    /// before), so no read is left to land in a buffer the caller frees once it sees the error (that left `ple_prefetch`'s freed row buffers to be
    /// written by the next call's wait: heap corruption, docs/STATUS.md 2026-11-04).
    pub unsafe fn submit(&self, jobs: &mut [(usize, u64, &mut [u8])]) -> Result<(), std::io::Error> {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        if !slots.pending.is_empty() {
            self.wait_locked(&mut slots)?;
        }
        let queued = unsafe { self.submit_locked(&mut slots, jobs) };
        if queued.is_err() {
            let _ = self.wait_locked(&mut slots);
        }
        queued
    }

    unsafe fn submit_locked(&self, slots: &mut DirectSlots, jobs: &mut [(usize, u64, &mut [u8])]) -> Result<(), std::io::Error> {
        const MAX_CHUNK: usize = 512 << 10; // 512 KiB max piece
        for (shard, offset, buf) in jobs.iter_mut() {
            let total_len = buf.len();
            let mut done = 0usize;
            while done < total_len {
                let chunk_len = (total_len - done).min(MAX_CHUNK);
                let piece_offset = *offset + done as u64;
                let sector_offset = piece_offset & !4095;
                let lead = (piece_offset - sector_offset) as usize;
                let sector_end = (piece_offset + chunk_len as u64 + 4095) & !4095;
                let sector_len = (sector_end - sector_offset) as usize;
if sector_len > DIRECT_SLOT_SIZE {
                    // Checked, not asserted: a panic mid-batch unwinds past submit's drain-on-error
                    // and lets reads queued earlier in this batch land in buffers the caller frees on
                    // the way out — the very corruption this file guards against. Nothing is allocated
                    // for this piece yet, so erroring here is safe.
                    return Err(std::io::Error::other(format!(
                        "readpool: sector_len {sector_len} > DIRECT_SLOT_SIZE {DIRECT_SLOT_SIZE}"
                    )));
                }

                if slots.free.is_empty() && slots.buffers.len() >= DIRECT_MAX_SLOTS {
                    self.retire_oldest(slots)?;
                    // The invariant (free + pending == buffers, no leaks now that the assert is a
                    // checked return) makes this state unreachable; if it ever happens there is
                    // nothing to retire and the cap cannot be honoured — error out (submit drains,
                    // the caller falls back to the buffered workers).
                    if slots.free.is_empty() {
                        return Err(std::io::Error::other(
                            "readpool: direct-I/O slot cap reached with nothing in flight to retire",
                        ));
                    }
                }
                let slot_idx = slots.alloc_slot()?;
                let ev = slots.events[slot_idx];
                let buf_ptr = slots.buffers[slot_idx];
                unsafe { win::ResetEvent(ev) };
                let ov = &mut *slots.ovs[slot_idx];
                ov.internal = 0;
                ov.internal_high = 0;
                ov.offset = (sector_offset & 0xFFFF_FFFF) as u32;
                ov.offset_high = ((sector_offset >> 32) & 0xFFFF_FFFF) as u32;
                ov.event = ev;

                let ok = unsafe {
                    win::ReadFile(
                        self.files[*shard],
                        buf_ptr as *mut std::ffi::c_void,
                        sector_len as u32,
                        std::ptr::null_mut(),
                        ov as *mut win::Overlapped,
                    )
                };
                if ok == 0 {
                    let err = win::GetLastError();
                    if err != win::ERROR_IO_PENDING {
                        slots.free.push(slot_idx);
                        return Err(std::io::Error::from_raw_os_error(err as i32));
                    }
                }
                slots.pending.push(PendingDirect {
                    slot: slot_idx,
                    shard: *shard,
                    dst_ptr: buf.as_mut_ptr().add(done) as usize,
                    lead,
                    len: chunk_len,
                });
                done += chunk_len;
            }
        }
        Ok(())
    }

    /// Finishes the oldest read in flight (waits for it, copies its bytes out) and frees its slot.
    fn retire_oldest(&self, slots: &mut DirectSlots) -> Result<(), std::io::Error> {
        if slots.pending.is_empty() {
            return Ok(());
        }
        let p = slots.pending.remove(0);
        let mut transferred = 0u32;
        let ok = unsafe { win::GetOverlappedResult(self.files[p.shard], &mut *slots.ovs[p.slot] as *mut win::Overlapped, &mut transferred, 1) };
        let r = if ok == 0 {
            Err(std::io::Error::from_raw_os_error(unsafe { win::GetLastError() } as i32))
        } else {
            unsafe { std::ptr::copy_nonoverlapping(slots.buffers[p.slot].add(p.lead), p.dst_ptr as *mut u8, p.len) };
            Ok(())
        };
        slots.free.push(p.slot);
        r
    }

    fn wait_locked(&self, slots: &mut DirectSlots) -> Result<(), std::io::Error> {
        let mut first_err = None;
        let pending = std::mem::take(&mut slots.pending);
        for p in pending {
            let mut transferred = 0u32;
            let ok = unsafe {
                win::GetOverlappedResult(
                    self.files[p.shard],
                    &mut *slots.ovs[p.slot] as *mut win::Overlapped,
                    &mut transferred,
                    1, // wait = TRUE
                )
            };
            if ok == 0 {
                let err = unsafe { win::GetLastError() };
                if first_err.is_none() {
                    first_err = Some(std::io::Error::from_raw_os_error(err as i32));
                }
            } else {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        slots.buffers[p.slot].add(p.lead),
                        p.dst_ptr as *mut u8,
                        p.len,
                    );
                }
            }
            slots.free.push(p.slot);
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub fn wait_async(&self) -> Result<(), std::io::Error> {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        self.wait_locked(&mut slots)
    }

    pub fn read_many(&self, jobs: &mut [(usize, u64, &mut [u8])]) -> std::io::Result<()> {
        unsafe { self.submit(jobs)? };
        self.wait_async()
    }
}

#[cfg(windows)]
impl Drop for DirectIoPool {
    fn drop(&mut self) {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        let _ = self.wait_locked(&mut slots);
        for &ev in &slots.events {
            unsafe { win::CloseHandle(ev) };
        }
        for &buf in &slots.buffers {
            unsafe { win::VirtualFree(buf as *mut std::ffi::c_void, 0, win::MEM_RELEASE) };
        }
        for &f in &self.files {
            unsafe { win::CloseHandle(f) };
        }
    }
}
