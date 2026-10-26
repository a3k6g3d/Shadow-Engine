//! Milestone 5 of docs/RUST_ENGINE.md, the first commit: the calls Rust exports and the engine makes.
//!
//! Two guards, different in kind, because a hand-written header is only worth its upkeep if drift is caught:
//!
//!   * **Text against types** (no DLL needed): the header's `STRATA_HOST_API` prototypes are parsed and compared
//!     with a list kept here, and each line of that list sits beside a coercion of the Rust export to the matching
//!     fn type - so the header and the Rust signature can only move together, or this file stops compiling or
//!     stops passing.  The struct and the wake tag are checked the same way.
//!   * **Across the seam** (the built DLL): shadow_host.dll is loaded by absolute path, every name the header
//!     declares is resolved, and real reads go through it - a file written here, read back through Rust's own
//!     IOCP into memory the "engine" (this test) owns, including the short read at end of file and a wake from
//!     another thread.  This is the mirror of shadow-abi's tests/link_facade.rs.
//!
//! Cargo builds the cdylib next to the rlib before it builds this test, so the DLL not being there is a failure,
//! not a skip.

use std::ffi::c_void;
use shadow_abi::{header, Status, ABI_VERSION};
use shadow_host::ffi::{self, HostCompletion, HostFile};

/// Each C declaration beside the Rust fn type it must be.  The `let _: T = f;` lines are the type half of the
/// guard: change a Rust export's signature and they stop compiling until this list - and so the header - agree.
#[test]
fn the_header_and_the_rust_exports_agree() {
    let _: extern "C" fn() -> i32 = ffi::strata_host_abi_version;
    let _: extern "C" fn() -> *const std::ffi::c_char = ffi::strata_host_last_error;
    let _: unsafe extern "C" fn(*const u16, i32, *mut *mut HostFile, *mut u64) -> i32 = ffi::strata_host_file_open;
    let _: unsafe extern "C" fn(*mut HostFile, u64, *mut u8, u32, u64) -> i32 = ffi::strata_host_file_submit;
    let _: unsafe extern "C" fn(*mut HostFile, *mut HostCompletion, i32, i32, *mut i32) -> i32 = ffi::strata_host_file_wait;
    let _: unsafe extern "C" fn(*mut HostFile) -> i32 = ffi::strata_host_file_wake;
    let _: unsafe extern "C" fn(*mut HostFile) = ffi::strata_host_file_close;
    let _: extern "C" fn(u64) -> *mut u8 = ffi::strata_host_alloc_aligned;
    let _: unsafe extern "C" fn(*mut u8) = ffi::strata_host_free_aligned;

    let src = header::read(env!("CARGO_MANIFEST_DIR"));
    let got: Vec<String> = header::prototypes(&src, "STRATA_HOST_API").iter().map(|p| p.line()).collect();
    assert_eq!(
        got,
        vec![
            "int strata_host_abi_version()",
            "const char *strata_host_last_error()",
            "int strata_host_file_open(const unsigned short *path_utf16, int capacity, strata_host_file **file_out, unsigned long long *size_out)",
            "int strata_host_file_submit(strata_host_file *file, unsigned long long offset, unsigned char *buffer, unsigned int len, unsigned long long tag)",
            "int strata_host_file_wait(strata_host_file *file, strata_host_completion *out, int max, int timeout_ms, int *got_out)",
            "int strata_host_file_wake(strata_host_file *file)",
            "void strata_host_file_close(strata_host_file *file)",
            "unsigned char *strata_host_alloc_aligned(unsigned long long bytes)",
            "void strata_host_free_aligned(unsigned char *p)",
        ],
        "the header's Rust-side export list moved; the Rust exports (and the coercions above) must move with it"
    );
}

#[test]
fn the_completion_struct_and_the_wake_tag_agree() {
    let src = header::read(env!("CARGO_MANIFEST_DIR"));
    assert_eq!(
        header::struct_fields(&src, "strata_host_completion"),
        vec![
            ("unsigned long long".to_string(), "tag".to_string()),
            ("unsigned int".to_string(), "bytes".to_string()),
            ("int".to_string(), "status".to_string()),
        ],
        "the header's completion struct moved; HostCompletion did not"
    );
    // 8 + 4 + 4 = 16, no padding, alignment 8 - what MSVC lays the C struct out as.
    assert_eq!(std::mem::size_of::<HostCompletion>(), 16);
    assert_eq!(std::mem::align_of::<HostCompletion>(), 8);

    let tag = src
        .lines()
        .find_map(|l| l.trim().strip_prefix("#define STRATA_HOST_WAKE_TAG").map(|v| v.trim().to_string()))
        .expect("STRATA_HOST_WAKE_TAG not in the header");
    let tag = u64::from_str_radix(tag.trim_start_matches("0x").trim_end_matches("ull"), 16).expect("tag is not hex");
    assert_eq!(tag, shadow_host::WAKE_TAG);
}

// ---------------------------------------------------------------------------------------------- across the seam

#[cfg(windows)]
extern "system" {
    fn LoadLibraryW(path: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
}

type FnVersion = unsafe extern "C" fn() -> i32;
type FnLastError = unsafe extern "C" fn() -> *const i8;
type FnOpen = unsafe extern "C" fn(*const u16, i32, *mut *mut c_void, *mut u64) -> i32;
type FnSubmit = unsafe extern "C" fn(*mut c_void, u64, *mut u8, u32, u64) -> i32;
type FnWait = unsafe extern "C" fn(*mut c_void, *mut HostCompletion, i32, i32, *mut i32) -> i32;
type FnWake = unsafe extern "C" fn(*mut c_void) -> i32;
type FnClose = unsafe extern "C" fn(*mut c_void);
type FnAlloc = unsafe extern "C" fn(u64) -> *mut u8;
type FnFree = unsafe extern "C" fn(*mut u8);

/// The engine's view of shadow_host.dll: every name resolved from the DLL itself, not from this crate's rlib.
struct Host {
    version: FnVersion,
    last_error: FnLastError,
    open: FnOpen,
    submit: FnSubmit,
    wait: FnWait,
    wake: FnWake,
    close: FnClose,
    alloc: FnAlloc,
    free: FnFree,
}

/// The DLL cargo built for this crate: target/<profile>/shadow_host.dll, beside the deps/ this test runs from.
#[cfg(windows)]
fn dll_path() -> std::path::PathBuf {
    let exe = std::env::current_exe().expect("no current exe");
    let deps = exe.parent().expect("test exe has no directory");
    for dir in [deps.parent().unwrap_or(deps), deps] {
        let p = dir.join("shadow_host.dll");
        if p.is_file() {
            return p;
        }
    }
    panic!("no shadow_host.dll beside {} - is shadow-host still a cdylib?", deps.display());
}

#[cfg(windows)]
fn host() -> Host {
    let path = dll_path();
    let wide: Vec<u16> = path.as_os_str().to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
    // By absolute path, as the engine will: a DLL resolved at process start from the exe's directory or PATH is
    // the 0xC0000135 that M2 wrote down for the other direction.
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    assert!(!module.is_null(), "LoadLibraryW({}) failed", path.display());

    // Every name the header declares must resolve - the header promising an export the DLL lacks is the drift
    // the engine would otherwise find at run time.
    let src = header::read(env!("CARGO_MANIFEST_DIR"));
    let names: Vec<String> = header::prototypes(&src, "STRATA_HOST_API").into_iter().map(|p| p.name).collect();
    let sym = |n: &str| -> *mut c_void {
        let nz = format!("{n}\0");
        let raw = unsafe { GetProcAddress(module, nz.as_ptr() as *const i8) };
        assert!(!raw.is_null(), "{} has no {n} - the header promises an export the DLL does not have", path.display());
        raw
    };
    for n in &names {
        sym(n);
    }
    unsafe {
        Host {
            version: std::mem::transmute::<*mut c_void, FnVersion>(sym("strata_host_abi_version")),
            last_error: std::mem::transmute::<*mut c_void, FnLastError>(sym("strata_host_last_error")),
            open: std::mem::transmute::<*mut c_void, FnOpen>(sym("strata_host_file_open")),
            submit: std::mem::transmute::<*mut c_void, FnSubmit>(sym("strata_host_file_submit")),
            wait: std::mem::transmute::<*mut c_void, FnWait>(sym("strata_host_file_wait")),
            wake: std::mem::transmute::<*mut c_void, FnWake>(sym("strata_host_file_wake")),
            close: std::mem::transmute::<*mut c_void, FnClose>(sym("strata_host_file_close")),
            alloc: std::mem::transmute::<*mut c_void, FnAlloc>(sym("strata_host_alloc_aligned")),
            free: std::mem::transmute::<*mut c_void, FnFree>(sym("strata_host_free_aligned")),
        }
    }
}

fn cstr(p: *const i8) -> String {
    let mut v = Vec::new();
    let mut i = 0isize;
    unsafe {
        while *p.offset(i) != 0 {
            v.push(*p.offset(i) as u8);
            i += 1;
        }
    }
    String::from_utf8_lossy(&v).into_owned()
}

/// A file of two whole pages and 100 bytes more, each byte a function of its offset, so a read from the wrong
/// place cannot match - the structure check the vertical slice uses, applied to a file this test owns.
#[cfg(windows)]
fn test_file(name: &str) -> (std::path::PathBuf, Vec<u8>) {
    let bytes: Vec<u8> = (0..(2 * 4096 + 100) as u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8).collect();
    let path = std::env::temp_dir().join(format!("shadow-host-{name}-{}.bin", std::process::id()));
    std::fs::write(&path, &bytes).expect("cannot write the test file");
    (path, bytes)
}

#[cfg(windows)]
#[test]
fn reads_go_through_the_dll_into_memory_the_caller_owns() {
    let h = host();
    assert_eq!(unsafe { (h.version)() } as u32, ABI_VERSION, "the DLL on disk and the Rust mirror disagree");

    let (path, bytes) = test_file("read");
    let wide: Vec<u16> = path.as_os_str().to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
    let (mut file, mut size) = (std::ptr::null_mut(), 0u64);
    let rc = unsafe { (h.open)(wide.as_ptr(), 4, &mut file, &mut size) };
    assert_eq!(Status::from_c(rc), Some(Status::Ok), "open: {}", cstr(unsafe { (h.last_error)() }));
    assert_eq!(size, bytes.len() as u64);

    // Two reads in flight at once: the first page whole, and the last page, which runs into the end of the file
    // and must come back as a legal short read (100 bytes, Ok) - the C++ reader's STATUS_END_OF_FILE rule.
    let buf = unsafe { (h.alloc)(2 * 4096) };
    assert!(!buf.is_null() && (buf as usize).is_multiple_of(4096), "alloc_aligned gave {buf:p}");
    assert_eq!(unsafe { (h.submit)(file, 0, buf, 4096, 10) }, Status::Ok as i32);
    assert_eq!(unsafe { (h.submit)(file, 8192, buf.add(4096), 4096, 11) }, Status::Ok as i32);

    let mut out = [HostCompletion::default(); 8];
    let mut seen = Vec::new();
    while seen.len() < 2 {
        let mut got = 0;
        let rc = unsafe { (h.wait)(file, out.as_mut_ptr(), 8, 5000, &mut got) };
        assert_eq!(rc, Status::Ok as i32, "wait: {}", cstr(unsafe { (h.last_error)() }));
        assert!(got > 0, "no completion within 5 s");
        seen.extend_from_slice(&out[..got as usize]);
    }
    seen.sort_by_key(|c| c.tag);
    assert_eq!((seen[0].tag, seen[0].bytes, seen[0].status), (10, 4096, Status::Ok as i32));
    assert_eq!((seen[1].tag, seen[1].bytes, seen[1].status), (11, 100, Status::Ok as i32), "end-of-file read");

    let got = unsafe { std::slice::from_raw_parts(buf, 2 * 4096) };
    assert_eq!(&got[..4096], &bytes[..4096], "page 0 came back wrong");
    assert_eq!(&got[4096..4196], &bytes[8192..], "the short last page came back wrong");

    unsafe {
        (h.close)(file);
        (h.free)(buf);
    }
    let _ = std::fs::remove_file(&path);
}

#[cfg(windows)]
#[test]
fn a_refusal_is_a_named_status_with_words() {
    let h = host();
    let (path, _) = test_file("refuse");
    let wide: Vec<u16> = path.as_os_str().to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
    let (mut file, mut size) = (std::ptr::null_mut(), 0u64);
    assert_eq!(unsafe { (h.open)(wide.as_ptr(), 1, &mut file, &mut size) }, Status::Ok as i32);

    let buf = unsafe { (h.alloc)(4096) };
    // Misaligned offset: refused before the kernel sees it, and the words say why.
    let rc = unsafe { (h.submit)(file, 100, buf, 4096, 1) };
    assert_eq!(Status::from_c(rc), Some(Status::BufferTooShort));
    assert!(cstr(unsafe { (h.last_error)() }).contains("unaligned"));

    // A ring of one, full: the second submit is refused, nothing consumed, and the first still completes once.
    assert_eq!(unsafe { (h.submit)(file, 0, buf, 4096, 2) }, Status::Ok as i32);
    let other = unsafe { (h.alloc)(4096) };
    assert_eq!(Status::from_c(unsafe { (h.submit)(file, 4096, other, 4096, 3) }), Some(Status::WouldBlock));
    let mut out = [HostCompletion::default(); 4];
    let mut got = 0;
    assert_eq!(unsafe { (h.wait)(file, out.as_mut_ptr(), 4, 5000, &mut got) }, Status::Ok as i32);
    assert_eq!((got, out[0].tag), (1, 2), "exactly one completion, for the read that was queued");
    // Polling again finds nothing: a refused read produced no completion.
    assert_eq!(unsafe { (h.wait)(file, out.as_mut_ptr(), 4, 0, &mut got) }, Status::Ok as i32);
    assert_eq!(got, 0, "a refused submit came back as a completion anyway");

    unsafe {
        (h.close)(file);
        (h.free)(buf);
        (h.free)(other);
    }
    let _ = std::fs::remove_file(&path);
}

#[cfg(windows)]
#[test]
fn a_wake_from_another_thread_returns_the_wake_tag() {
    let h = host();
    let (path, _) = test_file("wake");
    let wide: Vec<u16> = path.as_os_str().to_string_lossy().encode_utf16().chain(std::iter::once(0)).collect();
    let (mut file, mut size) = (std::ptr::null_mut(), 0u64);
    assert_eq!(unsafe { (h.open)(wide.as_ptr(), 2, &mut file, &mut size) }, Status::Ok as i32);

    // The reader's pattern: its I/O thread blocks in wait forever; another thread hands it work with wake.
    let addr = file as usize;
    let wake = h.wake;
    let waker = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        unsafe { wake(addr as *mut c_void) }
    });
    let mut out = [HostCompletion::default(); 4];
    let mut got = 0;
    assert_eq!(unsafe { (h.wait)(file, out.as_mut_ptr(), 4, -1, &mut got) }, Status::Ok as i32);
    assert_eq!(waker.join().unwrap(), Status::Ok as i32);
    assert_eq!((got, out[0].tag), (1, shadow_host::WAKE_TAG), "the wake did not come back as the wake tag");

    unsafe { (h.close)(file) };
    let _ = std::fs::remove_file(&path);
}
