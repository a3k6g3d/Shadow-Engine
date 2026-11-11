//! What the machine's graphics cards are and what the host can move through each: `shadow gpus` (docs/PLAN_TWO_GPUS_2026-11-11.md, phase 0a).
//!
//! The numbers it prints decide the two-card design: the host <-> device bandwidth of each card alone and with every card
//! copying at once (does the chipset slot take the CPU slot's bandwidth?), the round trip of a hop (device -> pinned host
//! memory -> device, the only way between two GeForce cards on Windows), and whether the driver offers peer access.
//! It works with one card (the hop is then card 0 -> card 0 through host memory).

use crate::driver::*;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::time::Instant;

/// One card as the driver reports it, read without making a context (a context takes a few hundred MiB of its VRAM).
pub struct GpuInfo {
    pub ordinal: i32,
    pub name: String,
    pub uuid: String,
    pub cc: (i32, i32),
    pub sms: i32,
    pub total_mem: usize,
    /// PCI domain, bus, device
    pub pci: (i32, i32, i32),
    pub mem_bus_bits: i32,
    pub mem_clock_khz: i32,
}

/// The cards the driver sees, in the driver's order (`CUDA_DEVICE_ORDER`: `FASTEST_FIRST` by default, `PCI_BUS_ID` for the slot order).
pub fn list() -> Result<Vec<GpuInfo>, String> {
    let api = api()?;
    let mut n: c_int = 0;
    // SAFETY: an out-pointer for the device count.
    let r = unsafe { (api.cuDeviceGetCount)(&mut n) };
    if r != 0 {
        return Err(format!("cuDeviceGetCount failed ({r})"));
    }
    let mut out = Vec::new();
    for ordinal in 0..n {
        let mut dev: CuDevice = 0;
        // SAFETY: an out-pointer and an ordinal below the count the driver gave.
        if unsafe { (api.cuDeviceGet)(&mut dev, ordinal) } != 0 {
            continue;
        }
        let mut name = [0 as c_char; 256];
        // SAFETY: the driver NUL-terminates within the 256 bytes it is given.
        unsafe { (api.cuDeviceGetName)(name.as_mut_ptr(), 256, dev) };
        // SAFETY: NUL-terminated by the call above (zero-filled when it failed).
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let attr = |a: c_int| -> i32 {
            let mut v = 0;
            // SAFETY: an out-pointer; an unsupported attribute leaves 0.
            unsafe { (api.cuDeviceGetAttribute)(&mut v, a, dev) };
            v
        };
        let mut total_mem = 0usize;
        // SAFETY: an out-pointer.
        unsafe { (api.cuDeviceTotalMem)(&mut total_mem, dev) };
        let mut u = [0u8; 16];
        // SAFETY: the 16-byte UUID.
        let ur = unsafe { (api.cuDeviceGetUuid)(u.as_mut_ptr(), dev) };
        let uuid = if ur == 0 {
            let h: Vec<String> = u.iter().map(|b| format!("{b:02x}")).collect();
            format!("GPU-{}-{}-{}-{}-{}", h[0..4].concat(), h[4..6].concat(), h[6..8].concat(), h[8..10].concat(), h[10..16].concat())
        } else {
            "?".into()
        };
        out.push(GpuInfo {
            ordinal,
            name,
            uuid,
            cc: (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)),
            sms: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            total_mem,
            pci: (attr(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID), attr(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID), attr(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)),
            mem_bus_bits: attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH),
            mem_clock_khz: attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE),
        });
    }
    Ok(out)
}

/// Whether the driver lets card `a` read and write card `b`'s memory directly (peer access).
pub fn can_access_peer(a: i32, b: i32) -> Result<bool, String> {
    let api = api()?;
    let (mut da, mut db): (CuDevice, CuDevice) = (0, 0);
    let mut can: c_int = 0;
    // SAFETY: out-pointers and ordinals the caller took from `list`.
    unsafe {
        if (api.cuDeviceGet)(&mut da, a) != 0 || (api.cuDeviceGet)(&mut db, b) != 0 {
            return Err(format!("no such device {a} or {b}"));
        }
        let r = (api.cuDeviceCanAccessPeer)(&mut can, da, db);
        if r != 0 {
            return Err(format!("cuDeviceCanAccessPeer failed ({r})"));
        }
    }
    Ok(can != 0)
}

/// Pinned host memory usable from every context (`CU_MEMHOSTALLOC_PORTABLE`), freed on drop.
struct HostBuf {
    api: &'static Api,
    p: *mut c_void,
    bytes: usize,
}

impl HostBuf {
    /// With a context current on this thread.
    fn new(api: &'static Api, bytes: usize) -> Result<HostBuf, String> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: an out-pointer; flags 1 = portable.
        let r = unsafe { (api.cuMemHostAlloc)(&mut p, bytes, 1) };
        if r != 0 {
            return Err(format!("cuMemHostAlloc of {} MiB failed ({r})", bytes >> 20));
        }
        // SAFETY: `bytes` bytes just allocated: touch them so the pages exist.
        unsafe { std::ptr::write_bytes(p as *mut u8, 0x5a, bytes) };
        Ok(HostBuf { api, p, bytes })
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        // SAFETY: the pointer `cuMemHostAlloc` gave, freed once.
        unsafe { (self.api.cuMemFreeHost)(self.p) };
    }
}

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Host -> device and device -> host bandwidth of one card in GB/s (median of `reps`, after a warm-up), `bytes` a copy, from pinned memory.
/// With a barrier, every card's thread starts each copy together (`bandwidth_together`).
fn bandwidth_inner(ordinal: i32, bytes: usize, reps: usize, barrier: Option<&std::sync::Barrier>) -> Result<(f64, f64), String> {
    let ctx = Context::new(ordinal).map_err(|e| e.to_string())?;
    let api = ctx.api;
    let d = ctx.alloc(bytes).map_err(|e| e.to_string())?;
    let h = HostBuf::new(api, bytes)?;
    let s = ctx.stream().map_err(|e| e.to_string())?;
    let run = |to_device: bool| -> Result<f64, String> {
        let mut v = Vec::with_capacity(reps);
        for i in 0..=reps {
            if let Some(b) = barrier {
                b.wait();
            }
            ctx.bind().map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            // SAFETY: `bytes` bytes on both sides, the stream belongs to this context.
            let r = unsafe {
                if to_device {
                    (api.cuMemcpyHtoDAsync)(d.ptr, h.p, h.bytes, s.raw())
                } else {
                    (api.cuMemcpyDtoHAsync)(h.p, d.ptr, h.bytes, s.raw())
                }
            };
            if r != 0 {
                return Err(format!("copy failed ({r})"));
            }
            s.synchronize().map_err(|e| e.to_string())?;
            if i > 0 {
                v.push(bytes as f64 / t0.elapsed().as_secs_f64() / 1e9);
            }
        }
        Ok(med(&mut v))
    };
    Ok((run(true)?, run(false)?))
}

/// One card alone.
pub fn bandwidth(ordinal: i32, bytes: usize, reps: usize) -> Result<(f64, f64), String> {
    bandwidth_inner(ordinal, bytes, reps, None)
}

/// Every card copying at the same time (one thread each, a barrier before every copy): what each gets when the others are busy.
pub fn bandwidth_together(ordinals: &[i32], bytes: usize, reps: usize) -> Vec<Result<(f64, f64), String>> {
    let barrier = std::sync::Barrier::new(ordinals.len());
    std::thread::scope(|sc| {
        let hs: Vec<_> = ordinals.iter().map(|&o| { let b = &barrier; sc.spawn(move || bandwidth_inner(o, bytes, reps, Some(b))) }).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("thread panicked".into()))).collect()
    })
}

/// The hop from card `a` to card `b` through pinned host memory: device -> host, wait, host -> device, wait.  Microseconds: (median, 99th percentile, max).
pub fn hop(a: i32, b: i32, bytes: usize, iters: usize) -> Result<(f64, f64, f64), String> {
    let ca = Context::new(a).map_err(|e| e.to_string())?;
    let cb = Context::new(b).map_err(|e| e.to_string())?;
    let api = ca.api;
    let da = ca.alloc(bytes).map_err(|e| e.to_string())?;
    let db = cb.alloc(bytes).map_err(|e| e.to_string())?;
    ca.bind().map_err(|e| e.to_string())?;
    let h = HostBuf::new(api, bytes)?;
    let sa = ca.stream().map_err(|e| e.to_string())?;
    let sb = cb.stream().map_err(|e| e.to_string())?;
    let mut v = Vec::with_capacity(iters);
    for i in 0..iters + 20 {
        let t0 = Instant::now();
        ca.bind().map_err(|e| e.to_string())?;
        // SAFETY: `bytes` bytes in the device buffer and in the pinned buffer; the stream is `ca`'s.
        let r = unsafe { (api.cuMemcpyDtoHAsync)(h.p, da.ptr, h.bytes, sa.raw()) };
        if r != 0 {
            return Err(format!("device -> host copy failed ({r})"));
        }
        sa.synchronize().map_err(|e| e.to_string())?;
        cb.bind().map_err(|e| e.to_string())?;
        // SAFETY: as above, `cb`'s buffer and stream; the pinned buffer is portable.
        let r = unsafe { (api.cuMemcpyHtoDAsync)(db.ptr, h.p, h.bytes, sb.raw()) };
        if r != 0 {
            return Err(format!("host -> device copy failed ({r})"));
        }
        sb.synchronize().map_err(|e| e.to_string())?;
        if i >= 20 {
            v.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p99 = v[((v.len() * 99) / 100).min(v.len() - 1)];
    Ok((v[v.len() / 2], p99, *v.last().unwrap()))
}

/// `shadow gpus [--mib N]`: the report.  Exit code 0 when a card was found and measured.
pub fn run_cli(args: &[String]) -> i32 {
    let mib: usize = args.windows(2).find(|w| w[0] == "--mib").and_then(|w| w[1].parse().ok()).unwrap_or(256);
    let bytes = mib << 20;
    let api = match api() {
        Ok(a) => a,
        Err(e) => {
            println!("no NVIDIA driver: {e}");
            return 1;
        }
    };
    let mut ver: c_int = 0;
    // SAFETY: an out-pointer.
    unsafe { (api.cuDriverGetVersion)(&mut ver) };
    let gpus = match list() {
        Ok(g) if !g.is_empty() => g,
        Ok(_) => {
            println!("no CUDA device");
            return 1;
        }
        Err(e) => {
            println!("{e}");
            return 1;
        }
    };
    println!("driver API {}.{}; CUDA_DEVICE_ORDER={}; CUDA_VISIBLE_DEVICES={}", ver / 1000, (ver % 1000) / 10,
             std::env::var("CUDA_DEVICE_ORDER").unwrap_or_else(|_| "(unset: fastest first)".into()),
             std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "(unset)".into()));
    println!("\n{} device(s):", gpus.len());
    for g in &gpus {
        let bw = if g.mem_bus_bits > 0 { g.mem_clock_khz as f64 * 1e3 * 2.0 * g.mem_bus_bits as f64 / 8.0 / 1e9 } else { 0.0 };
        println!("  [{}] {}  cc {}.{}  {} SMs  {:.1} GiB  PCI {:04x}:{:02x}:{:02x}  {}-bit bus, memory clock {} MHz (~{:.0} GB/s at 2 transfers a clock)  {}",
                 g.ordinal, g.name, g.cc.0, g.cc.1, g.sms, g.total_mem as f64 / (1u64 << 30) as f64, g.pci.0, g.pci.1, g.pci.2,
                 g.mem_bus_bits, g.mem_clock_khz / 1000, bw, g.uuid);
    }
    println!("\nhost <-> device bandwidth, {mib} MiB copies from pinned memory, median of 8 (GB/s):");
    let mut ok = true;
    for g in &gpus {
        match bandwidth(g.ordinal, bytes, 8) {
            Ok((h2d, d2h)) => println!("  [{}] host -> device {:6.2}   device -> host {:6.2}", g.ordinal, h2d, d2h),
            Err(e) => {
                ok = false;
                println!("  [{}] {e}", g.ordinal)
            }
        }
    }
    if gpus.len() > 1 {
        println!("\nthe same with every card copying at once:");
        let ords: Vec<i32> = gpus.iter().map(|g| g.ordinal).collect();
        for (g, r) in gpus.iter().zip(bandwidth_together(&ords, bytes, 8)) {
            match r {
                Ok((h2d, d2h)) => println!("  [{}] host -> device {:6.2}   device -> host {:6.2}", g.ordinal, h2d, d2h),
                Err(e) => println!("  [{}] {e}", g.ordinal),
            }
        }
        println!("\npeer access (a card reading another's memory directly):");
        for a in &gpus {
            let row: Vec<String> = gpus
                .iter()
                .map(|b| {
                    if a.ordinal == b.ordinal {
                        "-".into()
                    } else {
                        match can_access_peer(a.ordinal, b.ordinal) {
                            Ok(true) => "yes".into(),
                            Ok(false) => "no".into(),
                            Err(e) => e,
                        }
                    }
                })
                .collect();
            println!("  [{}] -> {}", a.ordinal, row.join("  "));
        }
    }
    println!("\nhop: device -> pinned host -> device, microseconds (median / 99th percentile / max):");
    let sizes: [(usize, usize, &str); 5] = [
        (4 << 10, 300, "4 KiB"),
        (40 << 10, 300, "40 KiB (one token's residual streams)"),
        (160 << 10, 300, "160 KiB (a 4-token verify pass)"),
        (2 << 20, 100, "2 MiB"),
        (80 << 20, 10, "80 MiB (a 2,048-row prompt sub-chunk)"),
    ];
    for a in &gpus {
        for b in &gpus {
            let mut line = format!("  [{}] -> [{}]", a.ordinal, b.ordinal);
            for (sz, it, label) in sizes {
                match hop(a.ordinal, b.ordinal, sz, it) {
                    Ok((m, p, x)) => line += &format!("\n      {label:42} {m:9.1} / {p:9.1} / {x:9.1}"),
                    Err(e) => {
                        ok = false;
                        line += &format!("\n      {label}: {e}")
                    }
                }
            }
            println!("{line}");
        }
    }
    if ok {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With a card: it is listed, copies at more than 1 GB/s, and a 40 KiB hop is well under two milliseconds.  Without a driver the test says so and passes.
    #[test]
    fn one_card_is_listed_and_measured() {
        let Ok(g) = list() else {
            eprintln!("skipped: no NVIDIA driver");
            return;
        };
        if g.is_empty() {
            eprintln!("skipped: no CUDA device");
            return;
        }
        assert!(g[0].total_mem > 1 << 30 && g[0].sms > 0 && g[0].cc.0 >= 5);
        let (h2d, d2h) = bandwidth(g[0].ordinal, 32 << 20, 3).unwrap();
        assert!(h2d > 1.0 && d2h > 1.0, "{h2d} {d2h}");
        let (m, _, _) = hop(g[0].ordinal, g[0].ordinal, 40 << 10, 50).unwrap();
        assert!(m < 2000.0, "a 40 KiB hop took {m} us");
    }
}
