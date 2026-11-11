//! Pinned host memory for the RAM tier (docs/SHADOW_PLAN.md, stage 3), and the RAM rule in code: **Shadow refuses to
//! pin memory that would leave less than 4 GiB of RAM available**, checks again once the pages are locked, and its
//! users check again as they fill it (`check_floor`).  A pinned tier cannot be paged out, and this PC has frozen hard at
//! 0.4 GiB free, so the floor stays `RAM_FLOOR` unless the user passes `--ram-floor-gib N` (0 turns the check off).
//!
//! The region is plain `VirtualAlloc` memory registered with `cuMemHostRegister`, so the GPU copies from it by DMA.
//! It is its own small allocator rather than `shadow-host::Buffer` (which the plan names) because `shadow-host` still
//! depends on the retired `shadow-abi`; the two do the same `VirtualAlloc`.

use crate::driver::Context;
use std::ffi::c_void;

/// RAM that must stay available after pinning, in bytes: the default of `--ram-floor-gib`.
pub const RAM_FLOOR: u64 = 4 << 30;

static FLOOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(RAM_FLOOR);

/// Bytes of pinned host memory this process holds right now (the RAM tier, the stage buffers, ...): the RAM black box logs it beside the available RAM.
static PINNED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn pinned_bytes() -> u64 {
    PINNED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The floor in force: `RAM_FLOOR` unless `set_ram_floor` changed it.
pub fn ram_floor() -> u64 {
    FLOOR.load(std::sync::atomic::Ordering::Relaxed)
}

/// `--ram-floor-gib`: the RAM that must stay available after pinning (0: no check).
pub fn set_ram_floor(bytes: u64) {
    FLOOR.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// RAM the machine loses between the start (free RAM before the server) and its lowest point while serving, besides the tier itself, in GiB, for a tier of `tier_gib` GiB: the
/// server's own ~1-4 GiB (STATUS 82) plus the file cache the SSD reads fill, which is smaller the bigger the tier (fewer misses).  Measured on the Phase B ladder (STATUS 85, 52.8 GiB
/// free at the start, a 3k, a 3k and a 15k prompt, 200 tokens): 7.6 GiB at a 36 GiB tier, 6.3 at 40, 5.0 at 42, 3.8 at 44, 2.8 at 45, interpolated between; for a smaller tier it stays
/// 7.6 (the 24 GiB tier of the Phase A4 test lost 5.8).
pub fn tier_overhead_gib(tier_gib: u64) -> f64 {
    const LADDER: [(f64, f64); 5] = [(36.0, 7.6), (40.0, 6.3), (42.0, 5.0), (44.0, 3.8), (45.0, 2.8)];
    let t = tier_gib as f64;
    if t <= LADDER[0].0 {
        return LADDER[0].1;
    }
    for w in LADDER.windows(2) {
        if t <= w[1].0 {
            return w[0].1 + (w[1].1 - w[0].1) * (t - w[0].0) / (w[1].0 - w[0].0);
        }
    }
    LADDER[4].1
}

/// Available RAM above the floor that the default tier still leaves at its lowest point (the ladder's 45 GiB tier left 1.5 GiB; the rule asks 1.25): the watchdog covers the rest.
pub const TIER_SAFETY_GIB: f64 = 1.25;

/// The largest default tier: the top of the Phase B ladder (45 GiB of a 46.8 GiB model's experts); beyond it nothing has been measured.
pub const TIER_MAX: u64 = 45 << 30;

/// The tier the launcher asks for when the user gave no `--ram-tier-gib` (docs/STATUS.md, Phase A4 and B): the biggest whole number of GiB, at most `max`, such that the free RAM at
/// the start less the tier, its overhead (`tier_overhead_gib`) and `TIER_SAFETY_GIB` still stays above `floor`; 0 when that is under 2 GiB (not worth a fill).  From 52.8 GiB free
/// with a 3.5 GiB floor that is 45; from 49 GiB, 36 (the rule it replaces: 36 from 49, else 30).
pub fn tier_for_available(available: u64, max: u64, floor: u64) -> u64 {
    let g = (1u64 << 30) as f64;
    let (avail, floor) = (available as f64 / g, floor as f64 / g);
    let mut t = max >> 30;
    while t >= 2 {
        if avail - t as f64 - tier_overhead_gib(t) - TIER_SAFETY_GIB >= floor {
            return t << 30;
        }
        t -= 1;
    }
    0
}

/// What `auto` leaves available when it sizes a pinned tier: the floor plus room for what moves after the fill
/// (the file cache, the desktop, other programs).  With a 32 GiB tier filled at ~41 GiB available, available RAM
/// later dipped to 5.6 GiB in one of 8 runs - 2.4 GiB below what was left at the fill - so the margin is the floor
/// plus 6 GiB.
pub const AUTO_MARGIN_EXTRA: u64 = 6 << 30;

/// What `auto` leaves available: the floor in force plus `AUTO_MARGIN_EXTRA`.
pub fn auto_margin() -> u64 {
    ram_floor() + AUTO_MARGIN_EXTRA
}

/// The pinned tier `auto` asks for right now: the RAM available less `AUTO_MARGIN` (0 if that is nothing).
pub fn auto_tier_bytes() -> u64 {
    available_ram().saturating_sub(auto_margin())
}

/// Whether pinning `bytes` more leaves at least the floor (`ram_floor`) available out of `available`.
pub fn headroom(bytes: u64, available: u64) -> Result<(), String> {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let floor = ram_floor();
    if bytes.checked_add(floor).is_none_or(|need| available < need) {
        return Err(format!("refusing to pin {:.2} GiB: {:.2} GiB of RAM available, and at least {:.1} GiB must stay \
                            available (RAM_FLOOR; --ram-floor-gib N changes it, 0 turns it off)", gib(bytes), gib(available), gib(floor)));
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
    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const MEM_DECOMMIT: u32 = 0x4000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const PAGE_READWRITE: u32 = 0x04;
    #[repr(C)]
    pub struct ProcMemCounters {
        pub cb: u32,
        pub page_faults: u32,
        pub peak_working_set: usize,
        pub working_set: usize,
        pub quota_peak_paged: usize,
        pub quota_paged: usize,
        pub quota_peak_nonpaged: usize,
        pub quota_nonpaged: usize,
        pub pagefile: usize,
        pub peak_pagefile: usize,
        pub private_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        pub fn K32GetProcessMemoryInfo(process: *mut c_void, c: *mut ProcMemCounters, cb: u32) -> i32;
        pub fn GetCurrentProcess() -> *mut c_void;
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

/// This process's working set and private bytes (committed memory only it uses), in bytes; (0, 0) when unknown.
#[cfg(windows)]
pub fn process_memory() -> (u64, u64) {
    let mut c = win::ProcMemCounters { cb: std::mem::size_of::<win::ProcMemCounters>() as u32, page_faults: 0, peak_working_set: 0, working_set: 0, quota_peak_paged: 0, quota_paged: 0,
                                       quota_peak_nonpaged: 0, quota_nonpaged: 0, pagefile: 0, peak_pagefile: 0, private_usage: 0 };
    // SAFETY: the pseudo-handle of the current process and a correctly sized out-structure.
    if unsafe { win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return (0, 0);
    }
    (c.working_set as u64, c.private_usage as u64)
}

#[cfg(not(windows))]
pub fn process_memory() -> (u64, u64) {
    (0, 0)
}

/// `SHADOW_RAM_MAP=1`: one line at a loading checkpoint with where this process's RAM is (docs/STATUS.md, Phase A1): working set, private bytes, what is pinned, what is neither (heap,
/// driver, buffers) and what the PC has available.
pub fn ram_map(label: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("SHADOW_RAM_MAP").is_some()) {
        return;
    }
    let g = |b: u64| b as f64 / (1u64 << 30) as f64;
    let (ws, private) = process_memory();
    let pinned = pinned_bytes();
    eprintln!("ram map: {label}: working set {:.2} GiB, private {:.2} GiB, pinned {:.2} GiB, private but not pinned {:.2} GiB, {:.2} GiB available", g(ws), g(private), g(pinned), g(private.saturating_sub(pinned)), g(available_ram()));
}

/// The chunk size of a shrinkable region (`PinnedRegion::new_chunked`): memory is committed, pinned and, under RAM pressure, given back this much at a time.
pub const CHUNK: usize = 1 << 30;

/// A page-locked host region the GPU can DMA from or read directly via zero-copy device mapping.  Unregistered and freed on drop.
///
/// The address range is reserved whole and committed + registered chunk by chunk (`new` makes one chunk, `new_chunked` many), so a region built in chunks
/// can give its last chunks back (`shrink_by`) without moving anything: every offset below the new length keeps its address.  Device addresses differ per chunk:
/// use `dev_addr(offset)`, and keep every copied range inside one chunk (`fit`).
pub struct PinnedRegion<'c> {
    ctx: &'c Context,
    base: *mut u8,
    pub dev_ptr: u64,
    len: usize,
    /// The committed and registered ranges, (offset, bytes), in order.
    chunks: Vec<(usize, usize)>,
    /// Each chunk's device-mapped address (the driver does not put separate registrations at consecutive device addresses) and the chunk size.
    devs: Vec<u64>,
    chunk: usize,
    available: fn() -> u64,
}

impl<'c> PinnedRegion<'c> {
    /// `len` bytes, pinned (warns if available RAM is below the floor; the watchdog trims the tier under pressure).
    pub fn new(ctx: &'c Context, len: usize) -> Result<PinnedRegion<'c>, String> {
        Self::build(ctx, len, len.max(1), available_ram, false)
    }

    /// As `new`, with where "available RAM" comes from (the tests lower it artificially and check the hard refusal).
    pub fn with_ram_source(ctx: &'c Context, len: usize, available: fn() -> u64) -> Result<PinnedRegion<'c>, String> {
        Self::build(ctx, len, len.max(1), available, true)
    }

    /// `len` bytes pinned in chunks of `chunk` bytes: a region that can give its tail back (`shrink_by`).
    pub fn new_chunked(ctx: &'c Context, len: usize, chunk: usize) -> Result<PinnedRegion<'c>, String> {
        Self::build(ctx, len, chunk.max(1 << 20), available_ram, false)
    }

    fn build(ctx: &'c Context, len: usize, chunk: usize, available: fn() -> u64, strict: bool) -> Result<PinnedRegion<'c>, String> {
        if let Err(e) = headroom(len as u64, available()) {
            if strict {
                return Err(e);
            }
            eprintln!("RAM warning: {e}");
        }
        #[cfg(windows)]
        {
            let total = len.max(1);
            // SAFETY: a fresh reservation of address space only; VirtualAlloc returns a page-aligned pointer or null.
            let base = unsafe { win::VirtualAlloc(std::ptr::null_mut(), total, win::MEM_RESERVE, win::PAGE_READWRITE) } as *mut u8;
            if base.is_null() {
                return Err(format!("VirtualAlloc of {len} bytes failed"));
            }
            let mut r = PinnedRegion { ctx, base, dev_ptr: 0, len: 0, chunks: Vec::new(), devs: Vec::new(), chunk, available };
            ctx.bind().map_err(|e| e.to_string())?;
            let mut off = 0usize;
            let mut warned = false;
            while off < total {
                let n = chunk.min(total - off);
                // SAFETY: [base + off, base + off + n) lies inside the reservation just made.
                let at = unsafe { base.add(off) } as *mut c_void;
                if unsafe { win::VirtualAlloc(at, n, win::MEM_COMMIT, win::PAGE_READWRITE) }.is_null() {
                    return Err(format!("VirtualAlloc: committing {n} bytes at offset {off} of {len} failed")); // r frees it
                }
                // SAFETY: the committed range just made; CU_MEMHOSTREGISTER_DEVICEMAP = 0x02.
                let code = unsafe { (ctx.api.cuMemHostRegister)(at, n, 0x02) };
                if code != 0 {
                    // SAFETY: committed above, not registered: Drop would only release the reservation, which holds this commit too.
                    return Err(format!("cuMemHostRegister of {n} bytes at offset {off} of {len} failed ({code})"));
                }
                r.chunks.push((off, n));
                PINNED.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                let mut dptr = 0u64;
                let dcode = unsafe { (ctx.api.cuMemHostGetDevicePointer)(&mut dptr, at, 0) };
                if dcode != 0 {
                    return Err(format!("cuMemHostGetDevicePointer failed ({dcode})"));
                }
                if off == 0 {
                    r.dev_ptr = dptr;
                }
                r.devs.push(dptr);
                off += n;
                r.len = off.min(len);
                if let Err(e) = r.check_floor() {
                    if strict {
                        return Err(e);
                    } else if !warned {
                        eprintln!("RAM warning: {e}");
                        warned = true;
                    }
                }
            }
            Ok(r)
        }
        #[cfg(not(windows))]
        {
            let _ = (ctx, available, chunk);
            Err("pinned memory is implemented for Windows only".into())
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device-mapped address of byte `off` (the range copied from there must lie inside one chunk: see `fit`).
    pub fn dev_addr(&self, off: usize) -> u64 {
        self.devs[off / self.chunk] + (off % self.chunk) as u64
    }

    /// Where a range of `size` bytes wanted at `off` can go without crossing a chunk boundary of a region of `chunk`-byte chunks: `off`, or the next chunk's start.
    pub fn fit(off: usize, size: usize, chunk: usize) -> usize {
        let room = chunk - off % chunk;
        if size > room { off + room } else { off }
    }

    /// Gives the last chunks back to the OS - unregistered from the driver and decommitted - until at least `bytes` are released or none is left; returns what was
    /// released.  The region is `len() - released` bytes afterwards.  The caller must have waited for every copy from, and read into, the released range.
    pub fn shrink_by(&mut self, bytes: usize) -> usize {
        #[cfg(windows)]
        {
            let mut freed = 0usize;
            while freed < bytes {
                let Some(&(off, n)) = self.chunks.last() else { break };
                // SAFETY: the chunk was committed and registered by `build`; it is unregistered, then its pages are returned (the reservation stays).
                unsafe {
                    if self.ctx.bind().is_ok() {
                        (self.ctx.api.cuMemHostUnregister)(self.base.add(off) as *mut c_void);
                    }
                    win::VirtualFree(self.base.add(off) as *mut c_void, n, win::MEM_DECOMMIT);
                }
                PINNED.fetch_sub(n as u64, std::sync::atomic::Ordering::Relaxed);
                self.chunks.pop();
                self.devs.pop();
                self.len = off;
                freed += n;
            }
            freed
        }
        #[cfg(not(windows))]
        {
            let _ = bytes;
            0
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: committed memory of `len` bytes owned by this value.
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as as_slice, borrowed mutably once.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.len) }
    }

    /// Err if available RAM has fallen below the floor (call it while filling the region and after).
    pub fn check_floor(&self) -> Result<(), String> {
        headroom(0, (self.available)())
    }
}

impl Drop for PinnedRegion<'_> {
    fn drop(&mut self) {
        #[cfg(windows)]
        // SAFETY: unregister what `build` registered, chunk by chunk, then release the whole reservation once.
        unsafe {
            if !self.chunks.is_empty() && self.ctx.bind().is_ok() {
                for &(off, _) in &self.chunks {
                    (self.ctx.api.cuMemHostUnregister)(self.base.add(off) as *mut c_void);
                }
            }
            for &(_, n) in &self.chunks {
                PINNED.fetch_sub(n as u64, std::sync::atomic::Ordering::Relaxed);
            }
            win::VirtualFree(self.base as *mut c_void, 0, win::MEM_RELEASE);
        }
    }
}

// ---- the RAM watchdog: give pinned chunks back when the PC runs low (docs/STATUS.md, Phase A3) ----

/// Bytes the watchdog wants released; the engine thread takes them at a safe point (`take_trim_request`).
static TRIM_REQUEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes pinned chunks have given back so far, in this process.
static TRIMMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What the tier gives back at a time: one chunk.
const TRIM_STEP: u64 = CHUNK as u64;

/// Available RAM the watchdog restores when it acts, above the floor (so one release ends the dip instead of leaving the PC at the edge).
const TRIM_HYSTERESIS: u64 = 1 << 30;

/// How many bytes to give back: 0 while `available` is at or above `floor` (or the floor is off, 0), else enough whole `step`s to bring it to `floor + hysteresis`.
pub fn trim_wanted(available: u64, floor: u64, hysteresis: u64, step: u64) -> u64 {
    if floor == 0 || available >= floor {
        return 0;
    }
    (floor + hysteresis - available).div_ceil(step) * step
}

pub fn take_trim_request() -> u64 {
    TRIM_REQUEST.swap(0, std::sync::atomic::Ordering::Relaxed)
}

pub fn note_trimmed(bytes: u64) {
    TRIMMED.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
}

pub fn trimmed_bytes() -> u64 {
    TRIMMED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Starts the watchdog thread once: every 250 ms it reads the available RAM, and after two samples in a row below the floor (a single dip is the file cache moving)
/// asks for chunks back (`SHADOW_RAM_WATCHDOG=0` turns it off; no floor, no watchdog).  The engine thread acts on the request at its next safe point.
pub fn start_watchdog() {
    static ON: std::sync::Once = std::sync::Once::new();
    if std::env::var("SHADOW_RAM_WATCHDOG").is_ok_and(|v| v == "0") {
        return;
    }
    ON.call_once(|| {
        let _ = std::thread::Builder::new().name("ram-watchdog".into()).spawn(|| {
            let mut low = 0u32;
            let mut quiet_until = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(250));
                if std::time::Instant::now() < quiet_until {
                    continue;
                }
                let want = trim_wanted(available_ram(), ram_floor(), TRIM_HYSTERESIS, TRIM_STEP);
                low = if want > 0 { low + 1 } else { 0 };
                if low >= 2 {
                    TRIM_REQUEST.store(want, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("ram watchdog: available RAM is under the {:.1} GiB floor: asking the tier to give back {:.0} GiB", ram_floor() as f64 / (1u64 << 30) as f64, want as f64 / (1u64 << 30) as f64);
                    low = 0;
                    quiet_until = std::time::Instant::now() + std::time::Duration::from_secs(2);
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn this_process_reports_its_memory() {
        let (ws, private) = process_memory();
        assert!(ws > (1 << 20) && private > (1 << 20), "{ws} {private}");
    }

    #[test]
    fn the_default_tier_follows_the_free_ram() {
        let g = 1u64 << 30;
        let f = 7 * g / 2; // 3.5 GiB
        assert_eq!(tier_for_available(53 * g, TIER_MAX, f), 45 * g, "a free PC: the top of the ladder");
        assert_eq!(tier_for_available(80 * g, TIER_MAX, f), 45 * g, "capped at what has been measured");
        assert_eq!(tier_for_available(49 * g, TIER_MAX, f), 36 * g, "the old rule's point: 49 GiB free gave 36");
        assert_eq!(tier_for_available(52 * g, TIER_MAX, f), 42 * g);
        let (small, mid) = (tier_for_available(40 * g, TIER_MAX, f), tier_for_available(45 * g, TIER_MAX, f));
        assert!(small < mid && mid < 36 * g, "{small} {mid}: less free RAM, a smaller tier");
        assert!(small >= 24 * g, "40 GiB free still gets a sizable tier");
        assert_eq!(tier_for_available(12 * g, TIER_MAX, f), 0, "nothing worth a fill");
        assert_eq!(tier_for_available(0, TIER_MAX, f), 0);
        assert!(tier_for_available(53 * g, TIER_MAX, 5 * g) < 45 * g, "a higher floor shrinks the tier");
    }

    #[test]
    fn the_watchdog_gives_back_whole_chunks_only_when_under_the_floor() {
        let g = 1u64 << 30;
        assert_eq!(trim_wanted(5 * g, 4 * g, g, g), 0);
        assert_eq!(trim_wanted(4 * g, 4 * g, g, g), 0);
        assert_eq!(trim_wanted(4 * g - 1, 4 * g, g, g), 2 * g); // 1 byte short: to reach floor + 1 GiB it needs 1 GiB + 1 byte, so two whole chunks
        assert_eq!(trim_wanted(3 * g, 4 * g, g, g), 2 * g); // 3 GiB free: needs 2 GiB to reach 5
        assert_eq!(trim_wanted(g / 2, 4 * g, g, g), 5 * g); // 0.5 GiB free: 4.5 GiB, five whole chunks
        assert_eq!(trim_wanted(0, 0, g, g), 0); // floor off: no watchdog
    }

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
