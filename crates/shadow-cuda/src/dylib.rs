//! Loading a DLL and its entry points by name, with nothing but kernel32.  The driver (`nvcuda.dll`) and NVRTC
//! (`nvrtc64_*.dll`) are found at run time, so a machine without a GPU still builds Shadow and still runs the CPU path.

use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    pub const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
        pub fn GetLastError() -> u32;
    }
}

/// A loaded library.  Never unloaded: the function pointers taken from it live as long as the process, which is what
/// lets them be plain `extern "C" fn` values in a `'static` table.
pub struct Dylib {
    handle: *mut c_void,
    pub path: PathBuf,
}

// SAFETY: the handle is an opaque module handle that is never freed; GetProcAddress may be called from any thread.
unsafe impl Send for Dylib {}
unsafe impl Sync for Dylib {}

impl Dylib {
    /// Loads `path` (a bare name searches the usual places).  A full path loads its own dependencies from its own
    /// folder (LOAD_WITH_ALTERED_SEARCH_PATH), which is how NVRTC finds `nvrtc-builtins64_*.dll` beside it.
    #[cfg(windows)]
    pub fn open(path: &Path) -> Result<Dylib, String> {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let flags = if path.is_absolute() { win::LOAD_WITH_ALTERED_SEARCH_PATH } else { 0 };
        // SAFETY: a NUL-terminated wide string that outlives the call.
        let h = unsafe { win::LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) };
        if h.is_null() {
            // SAFETY: no arguments.
            let e = unsafe { win::GetLastError() };
            return Err(format!("cannot load {} (Windows error {e})", path.display()));
        }
        Ok(Dylib { handle: h, path: path.to_path_buf() })
    }

    #[cfg(not(windows))]
    pub fn open(path: &Path) -> Result<Dylib, String> {
        Err(format!("cannot load {}: only Windows is wired up", path.display()))
    }

    /// The address of `name`, or None.
    pub fn sym(&self, name: &str) -> Option<*mut c_void> {
        #[cfg(windows)]
        {
            let c = CString::new(name).ok()?;
            // SAFETY: a live module handle and a NUL-terminated name.
            let p = unsafe { win::GetProcAddress(self.handle, c.as_ptr()) };
            (!p.is_null()).then_some(p)
        }
        #[cfg(not(windows))]
        {
            let _ = (name, CString::default());
            None
        }
    }

    /// The first of `names` that exists - for entry points that gained a `_v2` in one toolkit and kept the old name in
    /// another.
    pub fn sym_any(&self, names: &[&str]) -> Result<*mut c_void, String> {
        names
            .iter()
            .find_map(|n| self.sym(n))
            .ok_or_else(|| format!("{}: none of {names:?}", self.path.display()))
    }
}
