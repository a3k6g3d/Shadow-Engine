//! NVRTC: CUDA C++ source to CUBIN inside the running process, with an on-disk cache.
//!
//! The DLL is found in this order: `SHADOW_NVRTC` (a full path), then `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`
//! (CUDA 13 moved the DLLs under `x64`), newest `nvrtc64_*.dll` first.  It is loaded by full path, so its
//! `nvrtc-builtins64_*.dll` comes from the same folder.
//!
//! The cache key is the SHA-256 of everything that decides the output - the source (prelude included), the options,
//! the NVRTC version and the target - so a stale CUBIN cannot be loaded for a changed kernel.  A compile error is
//! returned with NVRTC's log in full.

use crate::dylib::Dylib;
use crate::sha256::{hex, sha256};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

type Prog = *mut c_void;
type R = c_int;

#[allow(non_snake_case)]
struct Api {
    lib: Dylib,
    nvrtcVersion: unsafe extern "C" fn(*mut c_int, *mut c_int) -> R,
    nvrtcGetErrorString: unsafe extern "C" fn(R) -> *const c_char,
    nvrtcGetNumSupportedArchs: unsafe extern "C" fn(*mut c_int) -> R,
    nvrtcGetSupportedArchs: unsafe extern "C" fn(*mut c_int) -> R,
    nvrtcCreateProgram: unsafe extern "C" fn(*mut Prog, *const c_char, *const c_char, c_int, *const *const c_char,
                                             *const *const c_char) -> R,
    nvrtcCompileProgram: unsafe extern "C" fn(Prog, c_int, *const *const c_char) -> R,
    nvrtcGetProgramLogSize: unsafe extern "C" fn(Prog, *mut usize) -> R,
    nvrtcGetProgramLog: unsafe extern "C" fn(Prog, *mut c_char) -> R,
    nvrtcGetCUBINSize: unsafe extern "C" fn(Prog, *mut usize) -> R,
    nvrtcGetCUBIN: unsafe extern "C" fn(Prog, *mut c_char) -> R,
    nvrtcGetPTXSize: unsafe extern "C" fn(Prog, *mut usize) -> R,
    nvrtcGetPTX: unsafe extern "C" fn(Prog, *mut c_char) -> R,
    nvrtcDestroyProgram: unsafe extern "C" fn(*mut Prog) -> R,
}

fn find_dll() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("SHADOW_NVRTC") {
        return Ok(PathBuf::from(p));
    }
    let mut dirs = Vec::new();
    if let Ok(c) = std::env::var("CUDA_PATH") {
        dirs.push(Path::new(&c).join("bin").join("x64"));
        dirs.push(Path::new(&c).join("bin"));
    }
    for d in &dirs {
        let mut found: Vec<PathBuf> = std::fs::read_dir(d)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                n.starts_with("nvrtc64_") && n.ends_with(".dll")
            })
            .collect();
        found.sort();
        if let Some(p) = found.pop() {
            return Ok(p);
        }
    }
    Err(format!("no nvrtc64_*.dll in {dirs:?} (set SHADOW_NVRTC to its full path)"))
}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

// The target type of each transmute is the `Api` field it is assigned to, written once in the struct above; naming it
// again at every line would be a second copy of each signature to keep in step.
#[allow(clippy::missing_transmute_annotations)]
fn api() -> Result<&'static Api, String> {
    API.get_or_init(|| {
        let lib = Dylib::open(&find_dll()?)?;
        macro_rules! f {
            ($n:ident) => {
                // SAFETY: NVRTC's own export of that name, with its documented signature.
                unsafe { std::mem::transmute::<*mut c_void, _>(lib.sym_any(&[stringify!($n)])?) }
            };
        }
        Ok(Api {
            nvrtcVersion: f!(nvrtcVersion),
            nvrtcGetErrorString: f!(nvrtcGetErrorString),
            nvrtcGetNumSupportedArchs: f!(nvrtcGetNumSupportedArchs),
            nvrtcGetSupportedArchs: f!(nvrtcGetSupportedArchs),
            nvrtcCreateProgram: f!(nvrtcCreateProgram),
            nvrtcCompileProgram: f!(nvrtcCompileProgram),
            nvrtcGetProgramLogSize: f!(nvrtcGetProgramLogSize),
            nvrtcGetProgramLog: f!(nvrtcGetProgramLog),
            nvrtcGetCUBINSize: f!(nvrtcGetCUBINSize),
            nvrtcGetCUBIN: f!(nvrtcGetCUBIN),
            nvrtcGetPTXSize: f!(nvrtcGetPTXSize),
            nvrtcGetPTX: f!(nvrtcGetPTX),
            nvrtcDestroyProgram: f!(nvrtcDestroyProgram),
            lib,
        })
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn err(api: &Api, r: R, what: &str) -> String {
    // SAFETY: NVRTC returns a static string for any result code.
    let s = unsafe { CStr::from_ptr((api.nvrtcGetErrorString)(r)) }.to_string_lossy();
    format!("{what}: {s} ({r})")
}

/// The NVRTC library in use: its path and version.
pub fn version() -> Result<(PathBuf, (i32, i32)), String> {
    let a = api()?;
    let (mut ma, mut mi) = (0, 0);
    // SAFETY: two out-pointers.
    let r = unsafe { (a.nvrtcVersion)(&mut ma, &mut mi) };
    if r != 0 {
        return Err(err(a, r, "nvrtcVersion"));
    }
    Ok((a.lib.path.clone(), (ma, mi)))
}

/// The `sm_` numbers this NVRTC can emit CUBIN for.
pub fn supported_archs() -> Result<Vec<i32>, String> {
    let a = api()?;
    let mut n = 0;
    // SAFETY: out-pointer, then a buffer of exactly that many ints.
    unsafe {
        if (a.nvrtcGetNumSupportedArchs)(&mut n) != 0 {
            return Err("nvrtcGetNumSupportedArchs failed".into());
        }
        let mut v = vec![0; n as usize];
        if (a.nvrtcGetSupportedArchs)(v.as_mut_ptr()) != 0 {
            return Err("nvrtcGetSupportedArchs failed".into());
        }
        Ok(v)
    }
}

/// What a compile produced.
#[derive(Debug, Clone)]
pub struct Compiled {
    /// A CUBIN, or PTX text when the device is newer than this NVRTC knows.
    pub image: Vec<u8>,
    pub is_ptx: bool,
    pub from_cache: bool,
    pub key: String,
    pub compile_ms: f64,
    pub log: String,
}

/// The fixed options, recorded so the cache key covers them.  No `--use_fast_math`: parity first (SHADOW_PLAN.md).
pub fn options(target: &str, debug: bool) -> Vec<String> {
    let mut o = vec![format!("--gpu-architecture={target}"), "--std=c++17".into(), "-default-device".into()];
    if debug {
        o.push("-lineinfo".into());
    }
    o
}

/// `sm_XY` when this NVRTC can emit CUBIN for the device, else `compute_<newest it knows>` (PTX, JIT by the driver).
pub fn target_for(cc: (i32, i32)) -> Result<(String, bool), String> {
    let want = cc.0 * 10 + cc.1;
    let archs = supported_archs()?;
    if archs.contains(&want) {
        Ok((format!("sm_{want}"), false))
    } else {
        let best = archs.iter().copied().filter(|&a| a <= want).max().ok_or("no NVRTC architecture fits this device")?;
        Ok((format!("compute_{best}"), true))
    }
}

pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("SHADOW_KERNEL_CACHE") {
        return PathBuf::from(d);
    }
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    base.join("shadow").join("kernels")
}

/// Compiles `source` (a whole program, prelude already prepended) for the device with compute capability `cc`, or
/// loads it from `cache` when the same inputs were compiled before.
pub fn compile(name: &str, source: &str, cc: (i32, i32), cache: &Path) -> Result<Compiled, String> {
    let a = api()?;
    let (_, ver) = version()?;
    let (target, is_ptx) = target_for(cc)?;
    let opts = options(&target, cfg!(debug_assertions));
    let mut keyed = Vec::new();
    keyed.extend(source.as_bytes());
    keyed.extend(format!("\0{}\0nvrtc {}.{}\0{target}", opts.join(" "), ver.0, ver.1).as_bytes());
    let key = hex(&sha256(&keyed));
    let file = cache.join(format!("{key}.{}", if is_ptx { "ptx" } else { "cubin" }));
    if let Ok(image) = std::fs::read(&file) {
        if !image.is_empty() {
            return Ok(Compiled { image, is_ptx, from_cache: true, key, compile_ms: 0.0, log: String::new() });
        }
    }

    let t0 = std::time::Instant::now();
    let src = CString::new(source).map_err(|_| format!("{name}: source has a NUL byte"))?;
    let cname = CString::new(name).map_err(|_| format!("{name}: name has a NUL byte"))?;
    let copts: Vec<CString> = opts.iter().map(|o| CString::new(o.as_str()).unwrap()).collect();
    let optp: Vec<*const c_char> = copts.iter().map(|c| c.as_ptr()).collect();
    let mut prog: Prog = std::ptr::null_mut();
    // SAFETY: NUL-terminated strings that outlive the program; no headers.
    let r = unsafe { (a.nvrtcCreateProgram)(&mut prog, src.as_ptr(), cname.as_ptr(), 0, std::ptr::null(), std::ptr::null()) };
    if r != 0 {
        return Err(err(a, r, "nvrtcCreateProgram"));
    }
    struct Guard<'a>(&'a Api, Prog);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            // SAFETY: the program nvrtcCreateProgram made, destroyed once.
            unsafe { (self.0.nvrtcDestroyProgram)(&mut self.1) };
        }
    }
    let g = Guard(a, prog);
    // SAFETY: a live program and an array of option strings.
    let rc = unsafe { (a.nvrtcCompileProgram)(g.1, optp.len() as c_int, optp.as_ptr()) };
    let mut n = 0usize;
    // SAFETY: out-pointer, then a buffer of that size (the size includes the NUL).
    let log = unsafe {
        (a.nvrtcGetProgramLogSize)(g.1, &mut n);
        let mut b = vec![0u8; n.max(1)];
        (a.nvrtcGetProgramLog)(g.1, b.as_mut_ptr() as *mut c_char);
        String::from_utf8_lossy(&b).trim_end_matches('\0').trim().to_string()
    };
    if rc != 0 {
        return Err(format!("{}\n--- NVRTC log for {name} ({target}) ---\n{log}", err(a, rc, "nvrtcCompileProgram")));
    }
    let mut size = 0usize;
    // SAFETY: as above, for the CUBIN or the PTX.
    let image = unsafe {
        if is_ptx {
            (a.nvrtcGetPTXSize)(g.1, &mut size);
            let mut b = vec![0u8; size];
            (a.nvrtcGetPTX)(g.1, b.as_mut_ptr() as *mut c_char);
            while b.last() == Some(&0) {
                b.pop();
            }
            b
        } else {
            (a.nvrtcGetCUBINSize)(g.1, &mut size);
            let mut b = vec![0u8; size];
            (a.nvrtcGetCUBIN)(g.1, b.as_mut_ptr() as *mut c_char);
            b
        }
    };
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // A cache that cannot be written is a slower start, not an error.  Write-then-rename so a crash mid-write never
    // leaves a truncated CUBIN under a valid key.
    if std::fs::create_dir_all(cache).is_ok() {
        let tmp = cache.join(format!("{key}.tmp{}", std::process::id()));
        if std::fs::write(&tmp, &image).is_ok() {
            let _ = std::fs::rename(&tmp, &file);
        }
    }
    Ok(Compiled { image, is_ptx, from_cache: false, key, compile_ms, log })
}
