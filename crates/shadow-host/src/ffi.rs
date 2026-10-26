//! The C exports: calls Rust implements and the C++ engine makes - the reverse of every `STRATA_API` export in
//! include/shadow/c/abi.h.  Milestone 5 of docs/RUST_ENGINE.md.
//!
//! The crate is built as a DLL (`cdylib`) that the engine loads by absolute path, mirroring how strata_facade.dll
//! is loaded the other way.  A staticlib linked into the engine was measured and rejected: MSVC fails to link it
//! (`LNK2019 __imp_GetUserProfileDirectoryW`, from Rust's own `std::env::home_dir`) unless `onecoreuap.lib` is
//! added to the C++ link.  As a DLL, Rust's `std` stays inside it and that symbol never reaches the engine's link.
//!
//! The surface is `platform::DirectFile`'s, one call for one method, so the engine's caller can swap readers
//! without changing what it does: open/size, submit into memory the engine owns, wait for completions, wake from
//! another thread, close, and the two aligned-allocation helpers.  The rules are the header's:
//!
//!   * No panic crosses the seam.  Every export catches it and answers `STRATA_ERR_DEVICE_UNAVAILABLE` with the
//!     panic's text in `strata_host_last_error`, because unwinding into C++ frames is undefined behaviour.
//!   * The file handle is Rust's (a `Box` the engine holds as an opaque pointer); the read buffers are the
//!     engine's.  The ring holds a submitted buffer until its completion is handed back, and close does not
//!     return while the kernel may still write one.
//!   * Threading is the C++ reader's: one thread makes every call except `wake`, which any thread may make.

use crate::{Buffer, Completion, DirectFile, Waker, SECTOR};
use std::cell::{RefCell, UnsafeCell};
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use shadow_abi::{Status, ABI_VERSION};

/// Mirror of `struct strata_host_completion`.  The buffer is not in it: the engine knows its buffers by tag, as
/// the C++ reader does, and the ring has handed the memory back by the time this is written.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HostCompletion {
    pub tag: u64,
    pub bytes: u32,
    pub status: i32,
}

/// What the engine holds as `strata_host_file *`.  The waker is kept apart from the ring so that `wake`, the one
/// call another thread may make, reads only it and never forms a reference to the ring the I/O thread is using.
pub struct HostFile {
    waker: Wake,
    ring: UnsafeCell<Ring>,
}

/// The two rings the exports can drive: the completion-port one (the default) and, with STRATA_HOST_IORING=1 on a
/// Windows that has them, the I/O-ring one (`ioring.rs`) - one kernel call per batch of reads instead of per read.
enum Ring {
    Iocp(DirectFile),
    #[cfg(windows)]
    IoRing(crate::ioring::RingFile),
}

#[derive(Clone, Copy)]
enum Wake {
    Iocp(Waker),
    #[cfg(windows)]
    IoRing(crate::ioring::RingWaker),
}

impl Wake {
    fn wake(&self) {
        match self {
            Wake::Iocp(w) => w.wake(),
            #[cfg(windows)]
            Wake::IoRing(w) => w.wake(),
        }
    }
}

impl Ring {
    fn submit(&mut self, offset: u64, buffer: Buffer, tag: u64) -> Result<(), (Status, Buffer)> {
        match self {
            Ring::Iocp(r) => r.submit(offset, buffer, tag),
            #[cfg(windows)]
            Ring::IoRing(r) => r.submit(offset, buffer, tag),
        }
    }

    fn wait_with(&mut self, max: usize, timeout_ms: u32, f: impl FnMut(Completion)) -> usize {
        match self {
            Ring::Iocp(r) => r.wait_with(max, timeout_ms, f),
            #[cfg(windows)]
            Ring::IoRing(r) => r.wait_with(max, timeout_ms, f),
        }
    }
}

/// STRATA_HOST_IORING=1 asks for the I/O ring; it is taken only where this Windows has one, else the completion port.
fn want_ioring() -> bool {
    #[cfg(windows)]
    {
        std::env::var("STRATA_HOST_IORING").is_ok_and(|v| v == "1") && crate::ioring::RingFile::available()
    }
    #[cfg(not(windows))]
    false
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let text = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = text);
}

/// Run an export's body, turning a panic into a status the engine can name.
fn guard(name: &str, body: impl FnOnce() -> Result<(), (Status, String)>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => Status::Ok as i32,
        Ok(Err((status, msg))) => {
            set_error(&format!("{name}: {msg}"));
            status as i32
        }
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".to_string());
            set_error(&format!("{name}: panicked: {what}"));
            Status::DeviceUnavailable as i32
        }
    }
}

#[no_mangle]
pub extern "C" fn strata_host_abi_version() -> i32 {
    ABI_VERSION as i32
}

/// The last failure on the calling thread, as text the engine can log.  Owned here; valid until the next export
/// call on the same thread.
#[no_mangle]
pub extern "C" fn strata_host_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// # Safety
/// `path` is a NUL-terminated UTF-16 string; `file_out` and `size_out` are writable.
#[no_mangle]
pub unsafe extern "C" fn strata_host_file_open(path: *const u16, capacity: i32,
                                               file_out: *mut *mut HostFile, size_out: *mut u64) -> i32 {
    guard("strata_host_file_open", || {
        if path.is_null() || file_out.is_null() || size_out.is_null() {
            return Err((Status::BufferTooShort, "null argument".into()));
        }
        if capacity <= 0 {
            return Err((Status::BufferTooShort, format!("capacity {capacity}")));
        }
        let mut n = 0;
        while *path.add(n) != 0 {
            n += 1;
        }
        let wide = std::slice::from_raw_parts(path, n);
        let path = String::from_utf16(wide).map_err(|_| (Status::BufferTooShort, "path is not UTF-16".into()))?;
        let p = std::path::Path::new(&path);
        let (ring, waker, size) = if want_ioring() {
            #[cfg(windows)]
            {
                let r = crate::ioring::RingFile::open(p, capacity as usize)
                    .map_err(|(s, why)| (s, format!("cannot open {path} through an I/O ring: {why}")))?;
                let (w, size) = (r.waker(), r.size());
                eprintln!("shadow-host: {path} is read through an I/O ring");
                (Ring::IoRing(r), Wake::IoRing(w), size)
            }
            #[cfg(not(windows))]
            unreachable!()
        } else {
            let r = DirectFile::open(p, capacity as usize).map_err(|s| (s, format!("cannot open {path} for direct access")))?;
            let (w, size) = (r.waker(), r.size());
            (Ring::Iocp(r), Wake::Iocp(w), size)
        };
        *size_out = size;
        *file_out = Box::into_raw(Box::new(HostFile { waker, ring: UnsafeCell::new(ring) }));
        Ok(())
    })
}

/// Queue one read into memory the caller owns.  `offset`, `len` and `buffer` are multiples of 4096; the buffer
/// is the ring's until a completion with this tag comes back from `strata_host_file_wait`.
///
/// # Safety
/// `file` came from `strata_host_file_open` and is not closed; `buffer` is `len` writable bytes that stay valid
/// and untouched until that completion (or until `strata_host_file_close` returns).
#[no_mangle]
pub unsafe extern "C" fn strata_host_file_submit(file: *mut HostFile, offset: u64, buffer: *mut u8, len: u32,
                                                 tag: u64) -> i32 {
    guard("strata_host_file_submit", || {
        if file.is_null() || buffer.is_null() || len == 0 {
            return Err((Status::BufferTooShort, "null file or buffer, or zero length".into()));
        }
        if !(len as u64).is_multiple_of(SECTOR) || !offset.is_multiple_of(SECTOR) || !(buffer as usize).is_multiple_of(SECTOR as usize) {
            return Err((Status::BufferTooShort, "unaligned request".into()));
        }
        let ring = &mut *(*file).ring.get();
        // A refusal hands the borrowed buffer back; dropping it frees nothing, and the memory is the engine's.
        ring.submit(offset, Buffer::borrowed(buffer, len as usize), tag)
            .map_err(|(s, _)| (s, format!("offset {offset} len {len} tag {tag}")))
    })
}

/// Wait for up to `max` completions, at most `timeout_ms` for the first (negative waits forever, 0 polls), then
/// take whatever else has already arrived without waiting again - the C++ reader's contract.  A wake shows up as
/// one completion with tag `STRATA_HOST_WAKE_TAG`.
///
/// # Safety
/// `file` is open; `out` has room for `max` entries; `got_out` is writable.
#[no_mangle]
pub unsafe extern "C" fn strata_host_file_wait(file: *mut HostFile, out: *mut HostCompletion, max: i32,
                                               timeout_ms: i32, got_out: *mut i32) -> i32 {
    guard("strata_host_file_wait", || {
        if file.is_null() || out.is_null() || got_out.is_null() || max <= 0 {
            return Err((Status::BufferTooShort, "null argument or max <= 0".into()));
        }
        *got_out = 0;
        let ring = &mut *(*file).ring.get();
        let first = if timeout_ms < 0 { u32::MAX } else { timeout_ms as u32 };
        let mut n = 0;
        // Straight into the caller's array, no ring and no allocation in between.  The buffer is the engine's
        // again from here; dropping our hold on it frees nothing.
        ring.wait_with(max as usize, first, |c: Completion| {
            *out.add(n) = HostCompletion { tag: c.tag, bytes: c.bytes as u32, status: c.status as i32 };
            n += 1;
        });
        *got_out = n as i32;
        Ok(())
    })
}

/// Make a thread blocked in `strata_host_file_wait` return.  Thread-safe; must not race `strata_host_file_close`.
///
/// # Safety
/// `file` is open.
#[no_mangle]
pub unsafe extern "C" fn strata_host_file_wake(file: *mut HostFile) -> i32 {
    guard("strata_host_file_wake", || {
        if file.is_null() {
            return Err((Status::BufferTooShort, "null file".into()));
        }
        // Only the waker is read: the I/O thread may hold the ring mutably right now.
        let waker = std::ptr::addr_of!((*file).waker).read();
        waker.wake();
        Ok(())
    })
}

/// Close the file.  Reads still in flight are cancelled, and this does not return while the kernel may still
/// write a buffer the caller lent.  A null file is a no-op.
///
/// # Safety
/// `file` came from `strata_host_file_open` and is not used again.
#[no_mangle]
pub unsafe extern "C" fn strata_host_file_close(file: *mut HostFile) {
    if file.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(file))));
}

/// Page-aligned memory, legal for a direct read - `platform::DirectFile::alloc_aligned`.  Null on failure.
#[no_mangle]
pub extern "C" fn strata_host_alloc_aligned(bytes: u64) -> *mut u8 {
    #[cfg(windows)]
    {
        if bytes == 0 {
            return std::ptr::null_mut();
        }
        // SAFETY: a fresh reservation; VirtualAlloc answers with a page-aligned pointer or null.
        unsafe {
            crate::win::VirtualAlloc(std::ptr::null_mut(), bytes as usize, crate::win::MEM_COMMIT_RESERVE,
                                     crate::win::PAGE_READWRITE) as *mut u8
        }
    }
    #[cfg(not(windows))]
    {
        let _ = bytes;
        std::ptr::null_mut()
    }
}

/// Free what `strata_host_alloc_aligned` returned.  A null pointer is a no-op.
///
/// # Safety
/// `p` came from `strata_host_alloc_aligned` and no read into it is in flight.
#[no_mangle]
pub unsafe extern "C" fn strata_host_free_aligned(p: *mut u8) {
    #[cfg(windows)]
    if !p.is_null() {
        crate::win::VirtualFree(p as *mut std::os::raw::c_void, 0, crate::win::MEM_RELEASE);
    }
    #[cfg(not(windows))]
    let _ = p;
}
