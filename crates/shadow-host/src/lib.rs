//! The Rust host's own async read path: a completion ring over a direct-access (unbuffered) file.
//! See docs/RUST_HOST.md, milestone 3.
//!
//! This is the half of the vertical slice that is Rust's to own.  The bytes are read here, by this process,
//! through its own input/output completion port - not through the C++ engine's reader - and then handed down the
//! seam to the existing kernels.  What the crate is for is the two rules the plan says the Rust side exists to
//! enforce:
//!
//!   * **A buffer that is in flight is owned by the ring, not by the caller.**  `submit` takes the buffer by
//!     value and `wait_into` hands it back inside the completion.  There is no way to drop it, resize it or walk
//!     away from it while the kernel still owns the memory - the type system will not let the borrow that used to
//!     be a comment in `expert_source.hpp` happen.  That is the whole reason the host layer is in Rust, so it is
//!     the thing this file is built to demonstrate.
//!   * **A full ring refuses loudly and drops nothing.**  That rule is shadow-abi's `Ring`'s, and `wait_into`
//!     takes one, so the rule is enforced by the same code on both sides of the seam.
//!
//! Direct access is the point and the price: every read's offset, length and buffer address must be a multiple of
//! the sector size (4096 here), which is why a blob is read as a whole number of pages and the geometry below
//! pads.  The C++ reader does the same thing for the same reason (`expert_source.hpp`, the 4 KiB-aligned windows
//! scattered into blobs).

pub mod ffi;
pub mod transient;
pub mod window;
pub mod protocol;
#[cfg(windows)]
pub mod ioring;

use std::collections::VecDeque;
use std::os::raw::c_void;
use std::path::Path;
use shadow_abi::{Ring, Status};

/// The unit direct access works in: offset, length and buffer address are all multiples of this.
pub const SECTOR: u64 = 4096;

/// One expert as the artifact defines it - the geometry `expert.hpp` fixes and the facade checks.
pub const BLOB_BYTES: u64 = 1_382_400;

/// `BLOB_BYTES` rounded up to the sector size: the smallest direct read that can carry one blob.
pub const BLOB_WINDOW: u64 = 1_384_448; // 338 * 4096

/// Memory this ring owns, page-aligned (so it is legal to hand to a direct read) and freed on drop.
///
/// It is a value with ownership, which is the point: while a read is in flight the ring holds one of these by
/// value, so the memory cannot be freed by anyone else until the completion has been reaped.
pub struct Buffer {
    base: *mut u8,
    len: usize,
    /// False for memory the C++ side lent across the seam (`Buffer::borrowed`): the ring still holds it by value
    /// while the read is in flight, but dropping it frees nothing, because the allocation was never this module's.
    owned: bool,
}

impl Buffer {
    /// Memory the caller owns, held by the ring for exactly as long as a read is in flight.  This is the shape
    /// the C exports need: the engine passes a pointer it allocated, and gets it back inside a completion.
    ///
    /// # Safety
    /// `base` must point to `len` writable bytes that stay valid, and that nobody else reads or writes, until the
    /// completion carrying this buffer has been reaped or the ring has been dropped.  Across the C ABI that is the
    /// header's rule for `strata_host_file_submit`; here it is the caller's promise, because Rust cannot check
    /// memory it did not allocate.
    pub unsafe fn borrowed(base: *mut u8, len: usize) -> Buffer {
        Buffer { base, len, owned: false }
    }

    /// `len` bytes of fresh, page-aligned memory this buffer owns and frees on drop.  None if the system refuses.
    pub fn aligned(len: usize) -> Option<Buffer> {
        #[cfg(windows)]
        {
            if len == 0 {
                return None;
            }
            // SAFETY: a fresh reservation; nothing owns this memory yet, and VirtualAlloc's return is either a
            // page-aligned pointer or null.
            let base = unsafe {
                win::VirtualAlloc(std::ptr::null_mut(), len, win::MEM_COMMIT_RESERVE, win::PAGE_READWRITE)
            };
            if base.is_null() {
                return None;
            }
            Some(Buffer { base: base as *mut u8, len, owned: true })
        }
        #[cfg(not(windows))]
        {
            let _ = len;
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `base`/`len` are one allocation made by this module (VirtualAlloc, page-aligned) and are not
        // mutated by anyone else while the buffer is the caller's.  While it is the ring's - in flight - nobody
        // outside this module holds a reference to it, because the kernel writes it and nothing else does.
        unsafe { std::slice::from_raw_parts(self.base as *const u8, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above; `&mut self` is the proof nobody else holds a reference, which is exactly what the
        // in-flight rule is about.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.len) }
    }

    pub fn is_aligned(&self) -> bool {
        self.base as usize % SECTOR as usize == 0
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        #[cfg(windows)]
        if self.owned && !self.base.is_null() {
            // SAFETY: this is the allocation VirtualAlloc made, whole (the second argument is 0 because the whole
            // region is being released, which is what MEM_RELEASE requires).
            unsafe {
                win::VirtualFree(self.base as *mut c_void, 0, win::MEM_RELEASE);
            }
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({:p}, {} bytes)", self.base, self.len)
    }
}

/// One finished read. The buffer comes back inside it, which is how the ownership rule closes: the ring cannot
/// report a completion for a buffer it is not still holding.
#[derive(Debug)]
pub struct Completion {
    pub tag: u64,
    pub bytes: usize,
    pub status: Status,
    pub buffer: Buffer,
}

/// The tag a wake shows up with - `platform::DirectFile::WAKE_TAG` on the C++ side, the same value.
pub const WAKE_TAG: u64 = u64::MAX;

/// Posts a wake to a ring's completion port.  The port is the only state it touches, and posting to a port is
/// thread-safe, which is what lets the reader's other threads hand its I/O thread new work.
#[derive(Debug, Clone, Copy)]
pub struct Waker {
    port: usize,
}

impl Waker {
    pub fn wake(&self) {
        #[cfg(windows)]
        // SAFETY: the port belongs to a live ring (the `waker` contract); a packet with no OVERLAPPED is legal.
        unsafe {
            win::PostQueuedCompletionStatus(self.port as win::Handle, 0, 0, std::ptr::null_mut());
        }
    }
}

/// A submitted read whose completion has not been reaped yet.
struct Pending {
    buffer: Buffer,
    tag: u64,
}

/// Completions taken from the port per kernel call (GetQueuedCompletionStatusEx) - the C++ reader's batch too.
const REAP_BATCH: usize = 64;

/// A direct-access file and the completion port its reads report to.
///
/// Single-threaded on purpose, for now: one ring, one serving thread, which is what the plan's host is.  The
/// buffers it owns are what make that safe - nothing can free them behind its back.
///
/// The request slots are allocated once, `capacity` of them: an OVERLAPPED per slot at a fixed address (the kernel
/// requires the address to stay put for the read's life), found again from a completion by pointer arithmetic.
/// So a read costs no allocation and no map lookup, and completions come off the port up to REAP_BATCH per call.
pub struct DirectFile {
    #[cfg(windows)]
    file: win::Handle,
    #[cfg(windows)]
    port: win::Handle,
    capacity: usize,
    size: u64,
    /// One per request slot; Some while that slot's read is in flight.
    slots: Vec<Option<Pending>>,
    free: Vec<usize>,
    in_flight: usize,
    /// `capacity` OVERLAPPEDs, one allocation, never moved; freed in Drop.
    #[cfg(windows)]
    ovs: *mut win::Overlapped,
    #[cfg(windows)]
    entries: Vec<win::OverlappedEntry>,
    /// Completions reaped and not yet handed out - first the ones a caller's full ring refused, then a batch's
    /// rest.  Nothing is dropped; the caller owns the retry, and this is where it waits.
    held: VecDeque<Completion>,
    /// The threads that issue the reads (STRATA_IO_THREADS, default 4 - the C++ pool's knob and default); None
    /// with 1, and `submit` issues on the calling thread as before.
    #[cfg(windows)]
    issuers: Option<Issuers>,
}

/// One read for an issuing thread: plain addresses, so the job crosses threads without a borrow.  The OVERLAPPED is
/// filled in by `submit` before the job is queued; the issuer only hands it to ReadFile.
#[cfg(windows)]
#[derive(Clone, Copy)]
struct Job {
    ov: usize,
    buf: usize,
    len: u32,
}

/// The issuing threads.  Why several: at depth the kernel's per-read cost on the issuing thread bounds the rate
/// (docs/RUST_ENGINE.md: one thread 135k reads/s at depth 256 against the four-thread C++ pool's 190k), and that cost
/// is per thread, so it spreads.
#[cfg(windows)]
struct Issuers {
    shared: std::sync::Arc<IssueQueue>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(windows)]
struct IssueQueue {
    jobs: std::sync::Mutex<(VecDeque<Job>, bool)>, // (queued, stop)
    ready: std::sync::Condvar,
}

/// Completion keys a packet can carry besides the file's own (registered as the file handle, never 1 or 2): an
/// issuer posts one of these when ReadFile refused at once, so the read still comes back as exactly one completion.
#[cfg(windows)]
const KEY_FAILED: usize = 1;
#[cfg(windows)]
const KEY_EOF: usize = 2; // at or past the end of the file: an ok, zero-byte completion, as the C++ reader gives

/// The number of issuing threads: STRATA_IO_THREADS, else 4, clamped to 1..64 - what the C++ pool reads.
#[cfg(windows)]
fn io_threads() -> usize {
    std::env::var("STRATA_IO_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).clamp(1, 64)
}

#[cfg(windows)]
impl Issuers {
    fn start(n: usize, file: usize, port: usize) -> Issuers {
        let shared = std::sync::Arc::new(IssueQueue {
            jobs: std::sync::Mutex::new((VecDeque::new(), false)),
            ready: std::sync::Condvar::new(),
        });
        let threads = (0..n)
            .map(|_| {
                let q = shared.clone();
                std::thread::spawn(move || loop {
                    let job = {
                        let mut g = q.jobs.lock().unwrap();
                        loop {
                            if let Some(j) = g.0.pop_front() {
                                break j;
                            }
                            if g.1 {
                                return;
                            }
                            g = q.ready.wait(g).unwrap();
                        }
                    };
                    // SAFETY: the file and port belong to a ring that joins this thread before closing them; the
                    // buffer and the OVERLAPPED are its slot's, untouched by anyone else until the completion.
                    unsafe {
                        if win::ReadFile(file as win::Handle, job.buf as *mut c_void, job.len, std::ptr::null_mut(),
                                         job.ov as *mut win::Overlapped) == 0 {
                            let err = win::GetLastError();
                            if err != win::ERROR_IO_PENDING {
                                // never started: no packet will come, so post one, marked, to keep one per read
                                let key = if err == win::ERROR_HANDLE_EOF { KEY_EOF } else { KEY_FAILED };
                                win::PostQueuedCompletionStatus(port as win::Handle, 0, key, job.ov as *mut win::Overlapped);
                            }
                        }
                    }
                })
            })
            .collect();
        Issuers { shared, threads }
    }

    fn push(&self, job: Job) {
        self.shared.jobs.lock().unwrap().0.push_back(job);
        self.shared.ready.notify_one();
    }

    /// Stop and join every issuer; the jobs they never took are returned (no ReadFile was made for them).
    fn stop(self) -> Vec<Job> {
        let left: Vec<Job> = {
            let mut g = self.shared.jobs.lock().unwrap();
            g.1 = true;
            g.0.drain(..).collect()
        };
        self.shared.ready.notify_all();
        for t in self.threads {
            let _ = t.join();
        }
        left
    }
}

#[cfg(windows)]
mod win {
    use std::os::raw::c_void;

    pub type Handle = *mut c_void;
    pub type Dword = u32;
    pub type Bool = i32;
    pub type UlongPtr = usize;

    /// OVERLAPPED as it is on x64: two ULONG_PTRs, the offset union, and the event handle - 32 bytes.
    #[repr(C)]
    pub struct Overlapped {
        pub internal: UlongPtr,
        pub internal_high: UlongPtr,
        pub anon: [usize; 1], // {Offset, OffsetHigh} or Pointer; the same 8 bytes either way
        pub event: Handle,
    }

    pub const INVALID_HANDLE_VALUE: Handle = -1isize as Handle;
    pub const GENERIC_READ: Dword = 0x8000_0000;
    pub const FILE_SHARE_ALL: Dword = 0x0000_0007; // read | write | delete
    pub const OPEN_EXISTING: Dword = 3;
    pub const FILE_FLAG_OVERLAPPED: Dword = 0x4000_0000;
    pub const FILE_FLAG_NO_BUFFERING: Dword = 0x2000_0000;
    pub const ERROR_IO_PENDING: Dword = 997;
    pub const INFINITE: Dword = 0xFFFF_FFFF;
    pub const MEM_COMMIT_RESERVE: Dword = 0x3000; // MEM_COMMIT | MEM_RESERVE
    pub const PAGE_READWRITE: Dword = 0x04;
    pub const MEM_RELEASE: Dword = 0x8000;

    // std links kernel32 for the MSVC target, so these resolve without a crate - the same fact milestone 2
    // established about the seam (see shadow-abi's build.rs).
    extern "system" {
        pub fn CreateFileW(name: *const u16, access: Dword, share: Dword, security: *mut c_void,
                           disposition: Dword, flags: Dword, template: Handle) -> Handle;
        pub fn CreateIoCompletionPort(file: Handle, port: Handle, key: UlongPtr, threads: Dword) -> Handle;
        pub fn ReadFile(file: Handle, buf: *mut c_void, len: Dword, read: *mut Dword,
                        overlapped: *mut Overlapped) -> Bool;
        pub fn CancelIoEx(file: Handle, overlapped: *const Overlapped) -> Bool;
        pub fn CloseHandle(handle: Handle) -> Bool;
        pub fn VirtualAlloc(addr: *mut c_void, len: usize, typ: Dword, protect: Dword) -> *mut c_void;
        pub fn VirtualFree(addr: *mut c_void, len: usize, typ: Dword) -> Bool;
        pub fn GetLastError() -> Dword;
        pub fn GetFileSizeEx(file: Handle, size: *mut i64) -> Bool;
        pub fn PostQueuedCompletionStatus(port: Handle, bytes: Dword, key: UlongPtr,
                                          overlapped: *mut Overlapped) -> Bool;
        pub fn GetQueuedCompletionStatusEx(port: Handle, entries: *mut OverlappedEntry, count: u32, removed: *mut u32,
                                           ms: Dword, alertable: Bool) -> Bool;
    }

    /// OVERLAPPED_ENTRY on x64: the key, the OVERLAPPED, its Internal (the read's NTSTATUS), the byte count.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct OverlappedEntry {
        pub key: UlongPtr,
        pub overlapped: *mut Overlapped,
        pub internal: UlongPtr,
        pub bytes: Dword,
    }

    /// ReadFile's refusal for a read at or past the end of the file (an issuer turns it into an ok, zero-byte completion).
    pub const ERROR_HANDLE_EOF: Dword = 38;

    /// STATUS_END_OF_FILE: the NTSTATUS of a read that ran into the end of the file - legal at the table's last page.
    pub const STATUS_END_OF_FILE: UlongPtr = 0xC000_0011;

}

/// A path to the wide form CreateFileW wants.
#[cfg(windows)]
fn to_wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
}

impl DirectFile {
    /// Open a file for direct access and give it a completion port. `capacity` is this ring's depth - how many
    /// reads may be in flight at once, which is fixed here rather than grown later because that is what makes
    /// the pending table's addresses stable enough to hand to the kernel.
    pub fn open(path: &Path, capacity: usize) -> Result<DirectFile, Status> {
        #[cfg(windows)]
        {
            if capacity == 0 {
                return Err(Status::BufferTooShort);
            }
            let wide = to_wide(path);
            // SAFETY: `wide` is a NUL-terminated wide string that outlives the call.
            let file = unsafe {
                win::CreateFileW(wide.as_ptr(), win::GENERIC_READ, win::FILE_SHARE_ALL,
                                 std::ptr::null_mut(), win::OPEN_EXISTING,
                                 win::FILE_FLAG_OVERLAPPED | win::FILE_FLAG_NO_BUFFERING,
                                 std::ptr::null_mut())
            };
            if file == win::INVALID_HANDLE_VALUE {
                return Err(Status::DeviceUnavailable);
            }
            // SAFETY: `file` is a valid handle; a null port means "create one".
            let port = unsafe { win::CreateIoCompletionPort(file, std::ptr::null_mut(), file as usize, 0) };
            if port.is_null() {
                // SAFETY: valid handle, and nothing is using it any more.
                unsafe { win::CloseHandle(file) };
                return Err(Status::DeviceUnavailable);
            }
            let mut size: i64 = 0;
            // SAFETY: valid handle, and `size` is ours.
            if unsafe { win::GetFileSizeEx(file, &mut size) } == 0 {
                unsafe {
                    win::CloseHandle(port);
                    win::CloseHandle(file);
                }
                return Err(Status::DeviceUnavailable);
            }
            let ovs: Box<[win::Overlapped]> = (0..capacity)
                .map(|_| win::Overlapped { internal: 0, internal_high: 0, anon: [0], event: std::ptr::null_mut() })
                .collect();
            return Ok(DirectFile {
                file,
                port,
                capacity,
                size: size as u64,
                slots: (0..capacity).map(|_| None).collect(),
                free: (0..capacity).rev().collect(),
                in_flight: 0,
                ovs: Box::into_raw(ovs) as *mut win::Overlapped,
                entries: vec![win::OverlappedEntry { key: 0, overlapped: std::ptr::null_mut(), internal: 0, bytes: 0 }; REAP_BATCH],
                held: VecDeque::new(),
                issuers: match io_threads() {
                    1 => None,
                    n => Some(Issuers::start(n, file as usize, port as usize)),
                },
            });
        }
        #[cfg(not(windows))]
        {
            let _ = (path, capacity);
            Err(Status::DeviceUnavailable) // no IOCP here; this machine's host is the Windows one
        }
    }

    /// A handle that can make a thread blocked in `wait_into` return, from any thread.  It is valid for as long as
    /// this ring is: the port it posts to is closed when the ring drops, so a waker must not outlive it.
    pub fn waker(&self) -> Waker {
        #[cfg(windows)]
        return Waker { port: self.port as usize };
        #[cfg(not(windows))]
        Waker { port: 0 }
    }

    /// The file's length in bytes, read once at open - what `platform::DirectFile::size()` answers on the C++ side.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Allocate a buffer legal to hand to a direct read: page-aligned, and a whole number of sectors.
    pub fn alloc(&self, len: usize) -> Option<Buffer> {
        #[cfg(windows)]
        {
            if len == 0 || len as u64 % SECTOR != 0 || len > self.capacity as usize * BLOB_WINDOW as usize {
                return None;
            }
            return Buffer::aligned(len);
        }
        #[cfg(not(windows))]
        {
            let _ = len;
            None
        }
    }

    /// Hand a read to the kernel and take ownership of the buffer for as long as it takes.
    ///
    /// `buffer` is moved in rather than borrowed, and comes back from `wait_into` inside the completion.  That is
    /// the rule the plan cares about, and it is not a comment: a caller that tried to free the buffer here would
    /// be freeing memory the ring still holds.
    ///
    /// A refusal hands the buffer back by value, the way `Ring::push` does: a read that was not queued produces no
    /// completion, so the error is the only place the buffer can come home.  (It used to be dropped on the
    /// alignment and full-ring paths - freed, for an owned buffer - and reported twice on a failed `ReadFile`.)
    pub fn submit(&mut self, offset: u64, buffer: Buffer, tag: u64) -> Result<(), (Status, Buffer)> {
        #[cfg(windows)]
        {
            // The contract of direct access, checked rather than assumed: a violation is an error here, not a
            // silent short read or a ERROR_INVALID_PARAMETER from the kernel much later.
            if offset % SECTOR != 0 || buffer.len as u64 % SECTOR != 0 || !buffer.is_aligned() {
                return Err((Status::BufferTooShort, buffer));
            }
            let Some(i) = self.free.pop() else {
                return Err((Status::WouldBlock, buffer)); // the caller owns the retry; nothing was consumed
            };
            let (base, len) = (buffer.base, buffer.len);
            self.slots[i] = Some(Pending { buffer, tag });
            self.in_flight += 1;
            // SAFETY: slot i is free (it came off the free list), so no read in flight uses its OVERLAPPED; `ovs`
            // is `capacity` OVERLAPPEDs that never move, which is what the kernel requires of the address.
            let ov = unsafe {
                let ov = self.ovs.add(i);
                ov.write(win::Overlapped { internal: 0, internal_high: 0, anon: [offset as usize], event: std::ptr::null_mut() });
                ov
            };
            if let Some(iss) = &self.issuers {
                // An issuing thread makes the ReadFile; whatever it gets, one completion comes for this read.
                iss.push(Job { ov: ov as usize, buf: base as usize, len: len as u32 });
                return Ok(());
            }
            // SAFETY: the handle is ours; the buffer is in slot i and stays there until its completion is reaped,
            // which is what makes the pointer legal for the whole read.
            let ok = unsafe { win::ReadFile(self.file, base as *mut c_void, len as win::Dword, std::ptr::null_mut(), ov) };
            if ok == 0 {
                // SAFETY: GetLastError immediately after a failed call.
                let err = unsafe { win::GetLastError() };
                if err != win::ERROR_IO_PENDING {
                    // The read never started, so no packet will come for it and the buffer is the caller's again.
                    let p = self.slots[i].take().unwrap();
                    self.in_flight -= 1;
                    self.free.push(i);
                    return Err((Status::DeviceUnavailable, p.buffer));
                }
            }
            // A read that finished before ReadFile returned (ok != 0) stays pending too.  The handle is bound to a
            // port without FILE_SKIP_COMPLETION_PORT_ON_SUCCESS, so the port still gets its packet, and that packet
            // is what reaps it.  Reaping it here instead used to free the OVERLAPPED while its packet was still
            // queued - and a later read whose Box landed at the same address would have been completed by it.
            return Ok(());
        }
        #[cfg(not(windows))]
        {
            let _ = (offset, tag);
            Err((Status::DeviceUnavailable, buffer))
        }
    }

    /// Reap up to `max` completions into the caller's ring, waiting at most `timeout_ms` (0 = do not wait).
    /// Returns how many were parked there.  A completion the ring refused is kept here and offered first next
    /// time - the ring's own rule, kept end to end.
    pub fn wait_into(&mut self, ring: &mut Ring<Completion>, max: usize, timeout_ms: u32) -> usize {
        let mut pushed = 0;
        while pushed < max {
            if self.held.is_empty() && self.reap_batch(timeout_ms) == 0 {
                break; // nothing arrived within the timeout
            }
            let Some(completion) = self.held.pop_front() else { break };
            match ring.push(completion) {
                Ok(()) => pushed += 1,
                // The ring's rule, kept end to end: the refusal hands the completion back by value, so the one it
                // could not hold is held here - not dropped, which for a completion would mean freeing its
                // buffer - and offered first next time.
                Err((_, c)) => {
                    self.held.push_front(c);
                    break;
                }
            }
        }
        pushed
    }

    /// The C++ reader's contract, for the C exports: up to `max` completions, waiting at most `timeout_ms` for the
    /// first (u32::MAX: forever) and taking whatever else has already arrived without waiting again.  Each goes to
    /// `f` as it is taken - no ring in between, and no allocation.
    pub fn wait_with(&mut self, max: usize, timeout_ms: u32, mut f: impl FnMut(Completion)) -> usize {
        let mut n = 0;
        while n < max {
            if self.held.is_empty() && self.reap_batch(if n == 0 { timeout_ms } else { 0 }) == 0 {
                break;
            }
            let Some(c) = self.held.pop_front() else { break };
            f(c);
            n += 1;
        }
        n
    }

    /// One trip to the completion port for up to REAP_BATCH completions, appended to `held`.  Returns how many
    /// arrived (0: the timeout passed with none).
    #[cfg(windows)]
    fn reap_batch(&mut self, timeout_ms: u32) -> usize {
        let ms = if timeout_ms == u32::MAX { win::INFINITE } else { timeout_ms };
        let mut got: u32 = 0;
        // SAFETY: `entries` has REAP_BATCH elements and is ours; the port is ours.
        let ok = unsafe {
            win::GetQueuedCompletionStatusEx(self.port, self.entries.as_mut_ptr(), REAP_BATCH as u32, &mut got, ms, 0)
        };
        if ok == 0 {
            return 0; // a timeout: nothing arrived
        }
        let base = self.ovs as usize;
        let size = std::mem::size_of::<win::Overlapped>();
        let mut n = 0;
        for k in 0..got as usize {
            let e = self.entries[k];
            if e.overlapped.is_null() {
                // A packet with no OVERLAPPED is a `Waker::wake` - the C++ reader's WAKE_TAG, kept with the same
                // value so a caller that switches readers reads the same tag.  It owns no memory.
                // SAFETY: a null, zero-length, unowned buffer; dropping it frees nothing.
                let buffer = unsafe { Buffer::borrowed(std::ptr::null_mut(), 0) };
                self.held.push_back(Completion { tag: WAKE_TAG, bytes: 0, status: Status::Ok, buffer });
                n += 1;
                continue;
            }
            // The key is the file handle, the same for every read; the OVERLAPPED's address names the slot.  One
            // that is not in our array, or names a slot with nothing in flight, is not ours to answer for.
            let off = (e.overlapped as usize).wrapping_sub(base);
            let i = off / size;
            if off % size != 0 || i >= self.capacity {
                continue;
            }
            let Some(p) = self.slots[i].take() else { continue };
            self.in_flight -= 1;
            self.free.push(i);
            // An issuer's marked packet (ReadFile refused at once), or the read's own NTSTATUS: success, or
            // STATUS_END_OF_FILE - a legal short read at the last page.
            let ok = match e.key {
                KEY_FAILED => false,
                KEY_EOF => true,
                _ => e.internal == 0 || e.internal == win::STATUS_END_OF_FILE,
            };
            self.held.push_back(Completion {
                tag: p.tag,
                bytes: if e.key == KEY_EOF { 0 } else { e.bytes as usize },
                status: if ok { Status::Ok } else { Status::DeviceUnavailable },
                buffer: p.buffer,
            });
            n += 1;
        }
        n
    }

    #[cfg(not(windows))]
    fn reap_batch(&mut self, _timeout_ms: u32) -> usize {
        0
    }
}

impl Drop for DirectFile {
    fn drop(&mut self) {
        // Nothing may be freed while the kernel still owns it, so a ring dropped with reads in flight cancels
        // them and waits for each one to come back before the buffers - and their VirtualAlloc'd memory - go.
        #[cfg(windows)]
        {
            // A cancelled read still reports to the port, and until it has, the kernel may write its buffer - which
            // since the C exports is memory the engine lent us and will reuse the moment close returns.  So one
            // quiet second is not enough to give up: keep waiting while reads are pending, and only after thirty
            // quiet seconds in a row (a device that has stopped answering) let the ring go.
            // The issuers first: once they are joined no new ReadFile can start, and the jobs they never took made
            // no read at all - their slots come back now, with no completion to wait for.
            if let Some(iss) = self.issuers.take() {
                let base = self.ovs as usize;
                let size = std::mem::size_of::<win::Overlapped>();
                for job in iss.stop() {
                    let i = (job.ov - base) / size;
                    if self.slots[i].take().is_some() {
                        self.in_flight -= 1;
                        self.free.push(i);
                    }
                }
            }
            let mut quiet = 0;
            while self.in_flight > 0 && quiet < 30 {
                // SAFETY: valid handle; null means "everything on this handle".
                unsafe { win::CancelIoEx(self.file, std::ptr::null()) };
                quiet = if self.reap_batch(1000) == 0 { quiet + 1 } else { 0 };
                self.held.clear(); // reaped while closing: nobody will ask for them, and their buffers may go now
            }
            if !self.port.is_null() {
                // SAFETY: handles are ours, and nothing is using them now.
                unsafe { win::CloseHandle(self.port) };
            }
            if self.file != win::INVALID_HANDLE_VALUE {
                unsafe { win::CloseHandle(self.file) };
            }
            // The OVERLAPPEDs go last, and only when no read can still name one - after thirty quiet seconds with
            // reads still out (a device that stopped answering) they are leaked rather than freed under the kernel.
            if self.in_flight == 0 && !self.ovs.is_null() {
                // SAFETY: the pointer came from Box::into_raw of a boxed slice of `capacity` OVERLAPPEDs.
                unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ovs, self.capacity))) };
            }
        }
    }
}
