//! Byte ranges of the model files as short-lived read-only memory-mapped views, registered with CUDA: the GPU copies a file-only expert straight from the OS file
//! cache (`cuMemHostRegister` with `READ_ONLY | DEVICEMAP` on a `MapViewOfFile` view), so the CPU does not read the file into a pinned buffer first (a copy
//! through DRAM that competes with the DMA, docs/STATUS.md 2026-11-04).
//!
//! `FileMaps` keeps one mapping object per shard (no address space until a view is made).  `RangeSet` maps, registers and (when the copy that used them is
//! done) unregisters and unmaps the merged ranges of one layer's pieces - views are not kept, so the pages leave the process's working set and go back to the
//! OS cache (one whole-file view kept every page touched in the working set: 51 GiB, and the RAM watchdog killed the run).

use crate::driver::Context;
use std::ffi::c_void;
use std::path::PathBuf;

/// One mapping object per shard: (handle, file length).
pub struct FileMaps {
    maps: Vec<(usize, usize)>,
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateFileW(name: *const u16, access: u32, share: u32, security: *mut c_void, disposition: u32, flags: u32, template: *mut c_void) -> *mut c_void;
        pub fn GetFileSizeEx(file: *mut c_void, size: *mut i64) -> i32;
        pub fn CreateFileMappingW(file: *mut c_void, security: *mut c_void, protect: u32, high: u32, low: u32, name: *const u16) -> *mut c_void;
        pub fn MapViewOfFile(mapping: *mut c_void, access: u32, high: u32, low: u32, bytes: usize) -> *mut c_void;
        pub fn UnmapViewOfFile(base: *const c_void) -> i32;
        pub fn CloseHandle(h: *mut c_void) -> i32;
    }
    pub const GENERIC_READ: u32 = 0x8000_0000;
    pub const FILE_SHARE_READ: u32 = 1;
    pub const FILE_SHARE_WRITE: u32 = 2;
    pub const OPEN_EXISTING: u32 = 3;
    pub const PAGE_READONLY: u32 = 2;
    pub const FILE_MAP_READ: u32 = 4;
}

/// The granularity views start on (Windows' allocation granularity) and the page size registrations round to.
const GRAN: u64 = 64 << 10;
const PAGE: usize = 4096;

impl FileMaps {
    pub fn open(paths: &[PathBuf]) -> Result<FileMaps, String> {
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut maps = vec![];
            for p in paths {
                let wide: Vec<u16> = p.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
                // SAFETY: plain Win32 calls with a NUL-terminated path; the file handle is closed below, the mapping handle in Drop.
                unsafe {
                    let f = win::CreateFileW(wide.as_ptr(), win::GENERIC_READ, win::FILE_SHARE_READ | win::FILE_SHARE_WRITE, std::ptr::null_mut(), win::OPEN_EXISTING, 0, std::ptr::null_mut());
                    if f as isize == -1 || f.is_null() {
                        return Err(format!("cannot open {}", p.display()));
                    }
                    let mut size = 0i64;
                    if win::GetFileSizeEx(f, &mut size) == 0 {
                        win::CloseHandle(f);
                        return Err(format!("cannot size {}", p.display()));
                    }
                    let m = win::CreateFileMappingW(f, std::ptr::null_mut(), win::PAGE_READONLY, 0, 0, std::ptr::null());
                    win::CloseHandle(f);
                    if m.is_null() {
                        return Err(format!("cannot map {}", p.display()));
                    }
                    maps.push((m as usize, size as usize));
                }
            }
            Ok(FileMaps { maps })
        }
        #[cfg(not(windows))]
        {
            let _ = paths;
            Err("file views are Windows-only".into())
        }
    }
}

impl Drop for FileMaps {
    fn drop(&mut self) {
        #[cfg(windows)]
        for &(m, _) in &self.maps {
            // SAFETY: opened by `open`.
            unsafe { win::CloseHandle(m as *mut c_void) };
        }
    }
}

/// One mapped, registered range: shard, first file byte, view base, length, device address of the base.
struct Range {
    shard: usize,
    file_start: u64,
    base: usize,
    len: usize,
    dev: u64,
}

#[derive(Default)]
pub struct RangeSet {
    ranges: Vec<Range>,
}

impl RangeSet {
    /// Maps and registers the pieces `(shard, file offset, length)`: aligned to the view granularity and merged where they touch, so no byte is mapped twice.
    pub fn register(&mut self, ctx: &Context, maps: &FileMaps, pieces: &[(usize, u64, usize)]) -> Result<(), String> {
        let mut iv: Vec<(usize, u64, u64)> = pieces.iter().map(|&(sh, off, len)| {
            let end = (off + len as u64).next_multiple_of(PAGE as u64).min(maps.maps[sh].1 as u64);
            (sh, off / GRAN * GRAN, end)
        }).collect();
        iv.sort_unstable();
        let mut merged: Vec<(usize, u64, u64)> = vec![];
        for (sh, s, e) in iv {
            match merged.last_mut() {
                Some(last) if last.0 == sh && s <= last.2 => last.2 = last.2.max(e),
                _ => merged.push((sh, s, e)),
            }
        }
        ctx.bind().map_err(|e| e.to_string())?;
        for (sh, s, e) in merged {
            let len = (e - s) as usize;
            #[cfg(windows)]
            // SAFETY: a read-only view of [s, e) of a shard this process mapped; it is unmapped (after unregistering) in `release`.
            let base = unsafe { win::MapViewOfFile(maps.maps[sh].0 as *mut c_void, win::FILE_MAP_READ, (s >> 32) as u32, s as u32, len) } as usize;
            #[cfg(not(windows))]
            let base = 0usize;
            if base == 0 {
                return Err(format!("cannot map {len} bytes of shard {sh} at {s}"));
            }
            // CU_MEMHOSTREGISTER_DEVICEMAP (2) | CU_MEMHOSTREGISTER_READ_ONLY (8)
            // SAFETY: the view just mapped.
            let code = unsafe { (ctx.api.cuMemHostRegister)(base as *mut c_void, len, 0x0A) };
            if code != 0 {
                #[cfg(windows)]
                unsafe { win::UnmapViewOfFile(base as *const c_void) };
                return Err(format!("cuMemHostRegister of {len} bytes of a file view failed ({code})"));
            }
            let mut dptr = 0u64;
            // SAFETY: registered with DEVICEMAP just above.
            let dcode = unsafe { (ctx.api.cuMemHostGetDevicePointer)(&mut dptr, base as *mut c_void, 0) };
            if dcode != 0 {
                unsafe { (ctx.api.cuMemHostUnregister)(base as *mut c_void) };
                #[cfg(windows)]
                unsafe { win::UnmapViewOfFile(base as *const c_void) };
                return Err(format!("cuMemHostGetDevicePointer of a file view failed ({dcode})"));
            }
            self.ranges.push(Range { shard: sh, file_start: s, base, len, dev: dptr });
        }
        Ok(())
    }

    /// The device address and the host address of file byte `off` of shard `shard` (inside a registered range).
    pub fn locate(&self, shard: usize, off: u64) -> (u64, usize) {
        let r = self.ranges.iter().find(|r| r.shard == shard && off >= r.file_start && off < r.file_start + r.len as u64)
            .expect("a file byte outside the registered ranges");
        let d = off - r.file_start;
        (r.dev + d, r.base + d as usize)
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Unregisters and unmaps everything (the copies that read it must be done).
    pub fn release(&mut self, ctx: &Context) {
        let bound = ctx.bind().is_ok();
        for r in &self.ranges {
            if bound {
                // SAFETY: registered by `register`.
                unsafe { (ctx.api.cuMemHostUnregister)(r.base as *mut c_void) };
            }
            #[cfg(windows)]
            // SAFETY: mapped by `register`.
            unsafe { win::UnmapViewOfFile(r.base as *const c_void) };
        }
        self.ranges.clear();
    }
}
