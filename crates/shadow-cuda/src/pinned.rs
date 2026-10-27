//! Pinned host memory for the RAM tier (docs/SHADOW_PLAN.md, stage 3), and the RAM rule in code: **Shadow refuses to
//! pin memory that would leave less than 4 GiB of RAM available**, checks again once the pages are locked, and its
//! users check again as they fill it (`check_floor`).  There is no setting that lowers `RAM_FLOOR`: a pinned tier
//! cannot be paged out, and this PC has frozen hard at 0.4 GiB free.
//!
//! The region is plain `VirtualAlloc` memory registered with `cuMemHostRegister`, so the GPU copies from it by DMA.
//! It is its own small allocator rather than `shadow-host::Buffer` (which the plan names) because `shadow-host` still
//! depends on the retired `shadow-abi`; the two do the same `VirtualAlloc`.

use crate::driver::Context;
use std::ffi::c_void;

/// RAM that must stay available after pinning, in bytes.  Not configurable, by design.
pub const RAM_FLOOR: u64 = 4 << 30;

/// What `auto` leaves available when it sizes a pinned tier: the floor plus room for what moves after the fill
/// (the file cache, the desktop, other programs).  With a 32 GiB tier filled at ~41 GiB available, available RAM
/// later dipped to 5.6 GiB in one of 8 runs - 2.4 GiB below what was left at the fill - so the margin is the floor
/// plus 6 GiB.
pub const AUTO_MARGIN: u64 = RAM_FLOOR + (6 << 30);

/// The pinned tier `auto` asks for right now: the RAM available less `AUTO_MARGIN` (0 if that is nothing).
pub fn auto_tier_bytes() -> u64 {
    available_ram().saturating_sub(AUTO_MARGIN)
}

/// Whether pinning `bytes` more leaves at least `RAM_FLOOR` available out of `available`.
pub fn headroom(bytes: u64, available: u64) -> Result<(), String> {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if bytes.checked_add(RAM_FLOOR).is_none_or(|need| available < need) {
        return Err(format!("refusing to pin {:.2} GiB: {:.2} GiB of RAM available, and at least {:.0} GiB must stay \
                            available (RAM_FLOOR)", gib(bytes), gib(available), gib(RAM_FLOOR)));
    }
    Ok(())
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    #[repr(C)]
    pub struct MemoryStatusEx {
        pub length: u32,
        pub load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }
    pub const MEM_COMMIT_RESERVE: u32 = 0x1000 | 0x2000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const PAGE_READWRITE: u32 = 0x04;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        pub fn VirtualAlloc(addr: *mut c_void, len: usize, typ: u32, protect: u32) -> *mut c_void;
        pub fn VirtualFree(addr: *mut c_void, len: usize, typ: u32) -> i32;
    }
}

/// Physical RAM available right now, in bytes (what Task Manager calls Available).
#[cfg(windows)]
pub fn available_ram() -> u64 {
    let mut s = win::MemoryStatusEx {
        length: std::mem::size_of::<win::MemoryStatusEx>() as u32, load: 0, total_phys: 0, avail_phys: 0,
        total_page_file: 0, avail_page_file: 0, total_virtual: 0, avail_virtual: 0, avail_extended_virtual: 0,
    };
    // SAFETY: a correctly sized, length-stamped out-structure.
    if unsafe { win::GlobalMemoryStatusEx(&mut s) } == 0 {
        return 0; // unknown counts as none: nothing gets pinned
    }
    s.avail_phys
}

#[cfg(not(windows))]
pub fn available_ram() -> u64 {
    0
}

/// A page-locked host region the GPU can DMA from.  Unregistered and freed on drop.
pub struct PinnedRegion<'c> {
    ctx: &'c Context,
    base: *mut u8,
    len: usize,
    available: fn() -> u64,
}

impl<'c> PinnedRegion<'c> {
    /// `len` bytes, pinned, if the RAM rule allows it.
    pub fn new(ctx: &'c Context, len: usize) -> Result<PinnedRegion<'c>, String> {
        Self::with_ram_source(ctx, len, available_ram)
    }

    /// As `new`, with where "available RAM" comes from (the tests lower it artificially).
    pub fn with_ram_source(ctx: &'c Context, len: usize, available: fn() -> u64) -> Result<PinnedRegion<'c>, String> {
        headroom(len as u64, available())?;
        #[cfg(windows)]
        {
            // SAFETY: a fresh reservation; VirtualAlloc returns a page-aligned pointer or null.
            let base = unsafe { win::VirtualAlloc(std::ptr::null_mut(), len.max(1), win::MEM_COMMIT_RESERVE,
                                                  win::PAGE_READWRITE) } as *mut u8;
            if base.is_null() {
                return Err(format!("VirtualAlloc of {len} bytes failed"));
            }
            let mut r = PinnedRegion { ctx, base, len: 0, available };
            ctx.bind().map_err(|e| e.to_string())?;
            // SAFETY: [base, base + len) is the committed allocation just made; flags 0 (this context only).
            let code = unsafe { (ctx.api.cuMemHostRegister)(base as *mut c_void, len.max(1), 0) };
            if code != 0 {
                return Err(format!("cuMemHostRegister of {len} bytes failed ({code})")); // r frees it
            }
            r.len = len;
            r.check_floor()?; // the pages are locked now: what is left?
            Ok(r)
        }
        #[cfg(not(windows))]
        {
            let _ = (ctx, available);
            Err("pinned memory is implemented for Windows only".into())
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: one allocation of `len` bytes owned by this value.
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as as_slice, borrowed mutably once.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.len) }
    }

    /// Err if available RAM has fallen below `RAM_FLOOR` (call it while filling the region and after).
    pub fn check_floor(&self) -> Result<(), String> {
        headroom(0, (self.available)())
    }
}

impl Drop for PinnedRegion<'_> {
    fn drop(&mut self) {
        #[cfg(windows)]
        // SAFETY: unregister what `new` registered (len > 0 only then), then release the whole allocation once.
        unsafe {
            if self.len > 0 && self.ctx.bind().is_ok() {
                (self.ctx.api.cuMemHostUnregister)(self.base as *mut c_void);
            }
            win::VirtualFree(self.base as *mut c_void, 0, win::MEM_RELEASE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_floor_is_four_gib_and_counts_the_request() {
        let g = 1u64 << 30;
        assert!(headroom(10 * g, 14 * g).is_ok());
        assert!(headroom(10 * g, 14 * g - 1).is_err());
        assert!(headroom(0, 4 * g).is_ok());
        assert!(headroom(0, 4 * g - 1).is_err());
        assert!(headroom(u64::MAX, u64::MAX).is_err());
    }
}
