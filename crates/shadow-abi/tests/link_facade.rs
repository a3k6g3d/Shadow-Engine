//! Milestone 2 of docs/RUST_HOST.md: the first real call across the Rust/C++ seam.
//!
//! The point of this file is the toolchain question the plan rests on - MSVC, cudart and cargo in one build - so
//! the assertions are about things the shim could not have made up on its own: the version constant it shares
//! with Rust, the answer a *linked C++ library* gives, and the answer a *linked CUDA runtime* gives.
//!
//! It is compiled only when build.rs found a built facade.  On a machine where that has never been built the
//! tests say so and pass, because "the test suite is runnable" cannot depend on someone having run cmake.
//!
//! The DLL is loaded by absolute path rather than linked.  Linking through the import library works - the linker
//! resolves it - but then Windows resolves the DLL itself at process start, from the exe's directory or PATH, and
//! cargo prunes anything it did not put in `deps/` itself.  A test that dies with STATUS_DLL_NOT_FOUND because of
//! where cargo chose to clean is not a test.  The `extern "system"` bindings are the same either way, which is
//! the part milestone 1 was about: bind to the header directly, no bindgen, no cxx.

use std::ffi::c_void;
use shadow_abi::{Status, ABI_VERSION};

#[cfg(windows)]
#[allow(dead_code)]
extern "system" {
    fn LoadLibraryW(path: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
}

type FnVoid = unsafe extern "system" fn() -> i32;
type FnI32 = unsafe extern "system" fn(*mut i32) -> i32;
type FnProbe = unsafe extern "system" fn(i32, i32, i32, *mut i32, *mut i32) -> i32;
type FnCstr = unsafe extern "system" fn() -> *const i8;

struct Facade {
    version: FnVoid,
    last_error: FnCstr,
    probe: FnProbe,
    cudart: FnI32,
}

/// The seam, resolved.  None means "there is no built facade on this machine", which is the case the whole file
/// has to survive.  Panics only when a facade was found and then failed to answer for a specific symbol - that
/// is a real disagreement with the header, not a missing prerequisite.
fn facade() -> Option<Facade> {
    let dll = option_env!("STRATA_FACADE_DLL")?;
    #[cfg(windows)]
    {
        let path = dll.to_string();
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        // LoadLibraryW is what the OS gives us and std already links kernel32, so this stays a std-only crate.
        // The handle is never freed: the process is the test, and it ends when the test does.
        let module = unsafe { LoadLibraryW(wide.as_ptr()) };
        assert!(!module.is_null(), "LoadLibraryW({path}) failed - build.rs found the DLL, and it would not load");
        let sym = |n: &str| -> *mut c_void {
            let nz = format!("{n}\0");
            let raw = unsafe { GetProcAddress(module, nz.as_ptr() as *const i8) };
            assert!(!raw.is_null(), "{path} has no {n} - the header promises an export the DLL does not have");
            raw
        };
        Some(Facade {
            version: unsafe { std::mem::transmute::<*mut c_void, FnVoid>(sym("strata_abi_version")) },
            last_error: unsafe { std::mem::transmute::<*mut c_void, FnCstr>(sym("strata_abi_last_error")) },
            probe: unsafe { std::mem::transmute::<*mut c_void, FnProbe>(sym("strata_facade_probe_policy")) },
            cudart: unsafe { std::mem::transmute::<*mut c_void, FnI32>(sym("strata_facade_cudart_version")) },
        })
    }
    #[cfg(not(windows))]
    {
        let _ = dll;
        None
    }
}

fn cstr(p: *const i8) -> String {
    // No libc in a std-only crate, so walk the bytes ourselves.
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

/// The version constant, read out of a DLL MSVC built.  Both sides of the seam declare it and the header test
/// checks they agree in text; this checks they agree when actually loaded, which is the thing that breaks in
/// practice (a stale DLL on disk, a header edited on one side only).
#[test]
fn version_agrees_across_the_seam() {
    let Some(f) = facade() else { println!("skipped: no built facade"); return };
    // strata_abi_version is declared (void) in the header; the Rust type says the same, and this is the first
    // call made across the seam, so if the two sides disagreed about the calling convention this is where it shows.
    let raw = unsafe { (f.version)() };
    assert_eq!(raw as u32, ABI_VERSION, "the DLL on disk and the Rust mirror disagree");
}

/// `strata_spec`, reached through the shim. DraftPolicy with no measurements is decided by the priors alone, so
/// choose(3, 4, 6) is a fixed answer - documented in the header - and the shim cannot have invented it.
#[test]
fn probe_answers_from_the_linked_cpp_library() {
    let Some(f) = facade() else { println!("skipped: no built facade"); return };
    let (mut lookup, mut t) = (-1i32, -1i32);
    let rc = unsafe { (f.probe)(3, 4, 6, &mut lookup, &mut t) };
    assert_eq!(Status::from_c(rc), Some(Status::Ok), "probe failed: {}", cstr(unsafe { (f.last_error)() }));
    assert_eq!((lookup, t), (1, 4), "choose(3, 4, 6) is what the header documents; if it moved, the header moved with it");
}

/// The error path across the seam: a null output pointer must come back as a status Rust can name, and the
/// last-error text must be the facade's own words. A caller that reads the status as an int never sees this.
#[test]
fn error_path_comes_back_as_a_named_status() {
    let Some(f) = facade() else { println!("skipped: no built facade"); return };
    let mut t = -1i32;
    let rc = unsafe { (f.probe)(3, 4, 6, std::ptr::null_mut(), &mut t) };
    assert_eq!(Status::from_c(rc), Some(Status::BufferTooShort));
    let msg = cstr(unsafe { (f.last_error)() });
    assert!(!msg.is_empty(), "the failure path set no message, so the caller gets a code and nothing to act on");
    assert!(msg.contains("null"), "message says {msg:?}");
}

/// The CUDA arm. Built with cudart this call answers with a real runtime version; built without, it says so with
/// STRATA_ERR_DEVICE_UNAVAILABLE instead of pretending.  Both answers are asserted, because which one is correct
/// here depends only on how the DLL was configured - and a CPU-built facade that *could* answer would mean the
/// shim had stopped calling into the runtime at all.
#[test]
fn cudart_answers_whichever_arm_it_was_built_as() {
    let Some(f) = facade() else { println!("skipped: no built facade"); return };
    let mut version = -1i32;
    let rc = unsafe { (f.cudart)(&mut version) };
    #[cfg(facade_with_cudart)]
    {
        assert_eq!(Status::from_c(rc), Some(Status::Ok), "cudart call failed: {}", cstr(unsafe { (f.last_error)() }));
        // cudaRuntimeGetVersion reports e.g. 12040 for 12.4; the floor is the one the header names.
        assert!(version >= 12000, "implausible runtime version {version}");
    }
    #[cfg(not(facade_with_cudart))]
    {
        assert_eq!(Status::from_c(rc), Some(Status::DeviceUnavailable), "a CPU-built facade must not claim an answer");
        assert_eq!(version, -1, "it declined the call but wrote the answer anyway");
    }
}

/// Says plainly which DLL this run reached, so a run on a machine without one reports that rather than looking
/// like four tests that quietly did nothing.
#[test]
fn which_facade_this_run_loaded() {
    match option_env!("STRATA_FACADE_DLL") {
        Some(d) => println!("loaded {d}"),
        None => println!("no built facade: the four tests above were skipped"),
    }
}
