//! The ring over Windows 11's I/O ring (ioringapi.h) instead of an I/O completion port - opt-in, STRATA_HOST_IORING=1.
//!
//! Why: the benchmark in docs/RUST_ENGINE.md put the IOCP ring ~13% behind the C++ pool at depth 256, and the cause is
//! the kernel's per-read cost on the one thread that issues them (the pool spreads it over four).  An I/O ring takes
//! many reads per kernel call: `submit` only writes an entry into the submission queue, and one SubmitIoRing hands
//! the kernel the whole batch.  Completions come back through a queue in shared memory, popped without a kernel call.
//!
//! The functions are taken from kernelbase.dll at run time, so the crate stays std-only and a Windows without I/O
//! rings (before 11) answers `open` with DeviceUnavailable - the caller falls back to the completion-port ring.
//! The contract is the completion-port ring's, unchanged: the caller's buffers, one completion per queued read,
//! WAKE_TAG for a wake from another thread, end of file a legal short read, and close waits for reads in flight.

use crate::{Buffer, Completion, SECTOR, WAKE_TAG};
use std::collections::VecDeque;
use std::os::raw::c_void;
use std::path::Path;
use shadow_abi::Status;

type Handle = *mut c_void;
type HResult = i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct CreateFlags {
    required: u32,
    advisory: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct HandleRef {
    kind: i32, // IORING_REF_RAW = 0
    handle: Handle,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct BufferRef {
    kind: i32, // IORING_REF_RAW = 0
    address: *mut c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Cqe {
    user_data: usize,
    result: HResult,
    information: usize,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Capabilities {
    max_version: i32,
    max_sq: u32,
    max_cq: u32,
    feature_flags: u32,
}

type FnCreate = unsafe extern "system" fn(i32, CreateFlags, u32, u32, *mut Handle) -> HResult;
type FnCaps = unsafe extern "system" fn(*mut Capabilities) -> HResult;
type FnBuildRead = unsafe extern "system" fn(Handle, HandleRef, BufferRef, u32, u64, usize, u32) -> HResult;
type FnSubmit = unsafe extern "system" fn(Handle, u32, u32, *mut u32) -> HResult;
type FnPop = unsafe extern "system" fn(Handle, *mut Cqe) -> HResult;
type FnClose = unsafe extern "system" fn(Handle) -> HResult;

struct Api {
    create: FnCreate,
    caps: FnCaps,
    build_read: FnBuildRead,
    submit: FnSubmit,
    pop: FnPop,
    close: FnClose,
}

extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    fn CreateFileW(name: *const u16, access: u32, share: u32, security: *mut c_void, disposition: u32, flags: u32,
                   template: Handle) -> Handle;
    fn GetFileSizeEx(file: Handle, size: *mut i64) -> i32;
    fn CloseHandle(h: Handle) -> i32;
    fn CreateEventW(attrs: *mut c_void, manual: i32, initial: i32, name: *const u16) -> Handle;
    fn SetEvent(h: Handle) -> i32;
    fn WaitForSingleObject(h: Handle, ms: u32) -> u32;
}

const S_OK: HResult = 0;
const S_FALSE: HResult = 1;
const IORING_E_SUBMISSION_QUEUE_FULL: HResult = 0x8046_0002u32 as i32; // winerror.h (0x...0001 is REQUIRED_FLAG_NOT_SUPPORTED)
const HR_EOF: HResult = 0x8007_0026u32 as i32; // HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)
const IORING_VERSION_3: i32 = 300; // ioringapi.h: versions 1 and 2 are 1 and 2, then 300, 400 - not 3
const WAIT_OBJECT_0: u32 = 0;
const INFINITE: u32 = 0xFFFF_FFFF;
/// Queued entries handed to the kernel without waiting for a `wait`: keeps a long run of submits moving.
const FLUSH_AT: usize = 64;

fn api() -> Option<&'static Api> {
    static API: std::sync::OnceLock<Option<Api>> = std::sync::OnceLock::new();
    API.get_or_init(|| unsafe {
        let name: Vec<u16> = "kernelbase.dll".encode_utf16().chain(std::iter::once(0)).collect();
        let m = LoadLibraryW(name.as_ptr());
        if m.is_null() {
            return None;
        }
        let sym = |n: &str| -> Option<*mut c_void> {
            let z = format!("{n}\0");
            let p = GetProcAddress(m, z.as_ptr() as *const i8);
            (!p.is_null()).then_some(p)
        };
        Some(Api {
            create: std::mem::transmute::<*mut c_void, FnCreate>(sym("CreateIoRing")?),
            caps: std::mem::transmute::<*mut c_void, FnCaps>(sym("QueryIoRingCapabilities")?),
            build_read: std::mem::transmute::<*mut c_void, FnBuildRead>(sym("BuildIoRingReadFile")?),
            submit: std::mem::transmute::<*mut c_void, FnSubmit>(sym("SubmitIoRing")?),
            pop: std::mem::transmute::<*mut c_void, FnPop>(sym("PopIoRingCompletion")?),
            close: std::mem::transmute::<*mut c_void, FnClose>(sym("CloseIoRing")?),
        })
    })
    .as_ref()
}

struct Pending {
    buffer: Buffer,
    tag: u64,
}

/// A direct-access file read through an I/O ring.  Single-threaded like the completion-port ring, except `wake`.
pub struct RingFile {
    api: &'static Api,
    file: Handle,
    ring: Handle,
    ev_wake: Handle,
    size: u64,
    slots: Vec<Option<Pending>>,
    free: Vec<usize>,
    in_flight: usize,
    queued: usize,
    held: VecDeque<Completion>,
}

/// Wakes a thread blocked in `RingFile::wait_with` from any thread (an event, not the port the IOCP ring posts to).
#[derive(Debug, Clone, Copy)]
pub struct RingWaker {
    event: usize,
}

impl RingWaker {
    pub fn wake(&self) {
        // SAFETY: the event belongs to a live RingFile (the waker must not outlive it, as with the IOCP ring's).
        unsafe { SetEvent(self.event as Handle) };
    }
}

impl RingFile {
    /// Whether this Windows has I/O rings with reads (version 3 or later).
    pub fn available() -> bool {
        let Some(a) = api() else { return false };
        let mut c = Capabilities::default();
        unsafe { (a.caps)(&mut c) == S_OK && c.max_version >= IORING_VERSION_3 }
    }

    pub fn open(path: &Path, capacity: usize) -> Result<RingFile, (Status, String)> {
        use std::os::windows::ffi::OsStrExt;
        let dev = |why: String| (Status::DeviceUnavailable, why);
        let api = api().ok_or_else(|| dev("kernelbase.dll has no I/O ring functions".into()))?;
        if capacity == 0 {
            return Err((Status::BufferTooShort, "capacity 0".into()));
        }
        let mut caps = Capabilities::default();
        let hr = unsafe { (api.caps)(&mut caps) };
        if hr != S_OK || caps.max_version < IORING_VERSION_3 {
            return Err(dev(format!("QueryIoRingCapabilities {hr:#x}, version {}", caps.max_version)));
        }
        let sq = (capacity as u32).min(caps.max_sq.max(1));
        let cq = (2 * capacity as u32).min(caps.max_cq.max(1));
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        // The same file flags as the completion-port ring: unbuffered (the table must never enter the file cache).
        let file = unsafe { CreateFileW(wide.as_ptr(), 0x8000_0000, 7, std::ptr::null_mut(), 3, 0x4000_0000 | 0x2000_0000,
                                        std::ptr::null_mut()) };
        if file as isize == -1 {
            return Err(dev("CreateFileW failed".into()));
        }
        let mut size = 0i64;
        let mut ring: Handle = std::ptr::null_mut();
        if unsafe { GetFileSizeEx(file, &mut size) } == 0 {
            unsafe { CloseHandle(file) };
            return Err(dev("GetFileSizeEx failed".into()));
        }
        let hr = unsafe { (api.create)(IORING_VERSION_3, CreateFlags { required: 0, advisory: 0 }, sq, cq, &mut ring) };
        if hr != S_OK {
            unsafe { CloseHandle(file) };
            return Err(dev(format!("CreateIoRing(v3, sq {sq}, cq {cq}) {hr:#x}; max sq {} cq {}", caps.max_sq, caps.max_cq)));
        }
        let ev_wake = unsafe { CreateEventW(std::ptr::null_mut(), 0, 0, std::ptr::null()) };
        if ev_wake.is_null() {
            unsafe {
                (api.close)(ring);
                CloseHandle(file);
            }
            return Err(dev("CreateEventW failed".into()));
        }
        let cap = sq as usize;
        Ok(RingFile {
            api,
            file,
            ring,
            ev_wake,
            size: size as u64,
            slots: (0..cap).map(|_| None).collect(),
            free: (0..cap).rev().collect(),
            in_flight: 0,
            queued: 0,
            held: VecDeque::new(),
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn waker(&self) -> RingWaker {
        RingWaker { event: self.ev_wake as usize }
    }

    fn flush(&mut self) {
        if self.queued > 0 {
            let mut n = 0u32;
            // SAFETY: our ring; wait for nothing, just hand the queued entries to the kernel.
            unsafe { (self.api.submit)(self.ring, 0, 0, &mut n) };
            self.queued = 0;
        }
    }

    /// Queue one read; the buffer is the ring's until its completion.  A refusal hands it back.
    pub fn submit(&mut self, offset: u64, buffer: Buffer, tag: u64) -> Result<(), (Status, Buffer)> {
        if offset % SECTOR != 0 || buffer.len() as u64 % SECTOR != 0 || !buffer.is_aligned() {
            return Err((Status::BufferTooShort, buffer));
        }
        let Some(i) = self.free.pop() else { return Err((Status::WouldBlock, buffer)) };
        let (base, len) = (buffer.as_slice().as_ptr() as *mut c_void, buffer.len() as u32);
        let file = HandleRef { kind: 0, handle: self.file };
        let data = BufferRef { kind: 0, address: base };
        // SAFETY: our ring and file; the buffer stays in slot i until its completion is popped.
        let mut hr = unsafe { (self.api.build_read)(self.ring, file, data, len, offset, i, 0) };
        if hr == IORING_E_SUBMISSION_QUEUE_FULL {
            self.flush();
            hr = unsafe { (self.api.build_read)(self.ring, file, data, len, offset, i, 0) };
        }
        if hr != S_OK {
            self.free.push(i);
            return Err((Status::DeviceUnavailable, buffer));
        }
        self.slots[i] = Some(Pending { buffer, tag });
        self.in_flight += 1;
        self.queued += 1;
        if self.queued >= FLUSH_AT {
            self.flush();
        }
        Ok(())
    }

    /// Everything the completion queue holds now, into `held`, without a kernel call.  Returns how many.
    fn pop_all(&mut self) -> usize {
        let mut n = 0;
        loop {
            let mut cqe = Cqe { user_data: 0, result: 0, information: 0 };
            // SAFETY: our ring; S_FALSE means the queue is empty.
            if unsafe { (self.api.pop)(self.ring, &mut cqe) } != S_OK {
                break;
            }
            let i = cqe.user_data;
            let Some(p) = self.slots.get_mut(i).and_then(|s| s.take()) else { continue };
            self.free.push(i);
            self.in_flight -= 1;
            let ok = cqe.result == S_OK || cqe.result == HR_EOF;
            self.held.push_back(Completion {
                tag: p.tag,
                bytes: if ok { cqe.information } else { 0 },
                status: if ok { Status::Ok } else { Status::DeviceUnavailable },
                buffer: p.buffer,
            });
            n += 1;
        }
        n
    }

    fn push_wake(&mut self) {
        // a wake: the C++ reader's WAKE_TAG, owning no memory
        let buffer = unsafe { Buffer::borrowed(std::ptr::null_mut(), 0) };
        self.held.push_back(Completion { tag: WAKE_TAG, bytes: 0, status: Status::Ok, buffer });
    }

    /// Up to `max` completions: hand the queued reads to the kernel, take what has completed, and only if nothing
    /// has, wait up to `timeout_ms` (u32::MAX: forever).  Each goes to `f`.
    ///
    /// How it waits is the reason this works: not on SetIoRingCompletionEvent's event, which on this Windows did not
    /// fire for every completion (a benchmark at depth 8 stalled after 3 of 11 reads, and calling SubmitIoRing before
    /// each wait did not cure it), but in SubmitIoRing itself, asking for one completion - which never lost one.  A
    /// wait inside the kernel cannot be interrupted by another thread, so the wake event is read around it: with
    /// reads in flight a wake is seen at the next completion (a read's latency away); with none in flight no
    /// completion can come, so the wait is on the wake event alone and a wake is immediate.
    pub fn wait_with(&mut self, max: usize, timeout_ms: u32, mut f: impl FnMut(Completion)) -> usize {
        self.flush();
        if self.held.is_empty() && self.pop_all() == 0 && timeout_ms != 0 {
            let ms = if timeout_ms == u32::MAX { INFINITE } else { timeout_ms };
            if self.in_flight == 0 {
                // SAFETY: an event we own.
                if unsafe { WaitForSingleObject(self.ev_wake, ms) } == WAIT_OBJECT_0 {
                    self.push_wake();
                }
            } else {
                let mut n = 0u32;
                // SAFETY: our ring; returns once a completion is in the queue, or at the timeout.
                unsafe { (self.api.submit)(self.ring, 1, ms, &mut n) };
                self.pop_all();
            }
        }
        // a wake that came while completions were flowing is still delivered (non-blocking look)
        if unsafe { WaitForSingleObject(self.ev_wake, 0) } == WAIT_OBJECT_0 {
            self.push_wake();
        }
        let mut n = 0;
        while n < max {
            let Some(c) = self.held.pop_front() else { break };
            f(c);
            n += 1;
        }
        n
    }
}

impl Drop for RingFile {
    fn drop(&mut self) {
        // No cancel: the reads are short and finite.  Wait for every one to complete before the ring - and with it
        // the caller's right to reuse its buffers - goes; thirty quiet seconds in a row is a device that stopped.
        self.flush();
        let mut quiet = 0;
        while self.in_flight > 0 && quiet < 30 {
            if self.pop_all() == 0 {
                let mut n = 0u32;
                unsafe { (self.api.submit)(self.ring, 1, 1000, &mut n) };
                quiet = if self.pop_all() > 0 { 0 } else { quiet + 1 };
            } else {
                quiet = 0;
            }
            self.held.clear();
        }
        unsafe {
            (self.api.close)(self.ring);
            CloseHandle(self.ev_wake);
            CloseHandle(self.file);
        }
        let _ = S_FALSE;
    }
}
