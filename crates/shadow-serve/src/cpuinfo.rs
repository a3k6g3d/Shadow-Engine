//! The CPU card of the dashboard: what the CPU is (model, base clock, cache sizes), what it does now (the clock it runs at) and what
//! Windows says about its temperature.  No driver: CPUID for the model and base clock, `GetLogicalProcessorInformation` for the caches, and
//! the performance counters (PDH) for the current clock (`% Processor Performance` x the base clock) and the ACPI thermal zones.  The thermal
//! zone is the board's, not the die's: a core's own temperature needs a kernel driver (HWiNFO, LibreHardwareMonitor), which Shadow does not load.

use serde_json::{json, Value};
use std::sync::Mutex;

/// The CPU as it is built (read once).
pub struct CpuStatic {
    pub name: String,
    pub base_mhz: u32,
    /// (level, kind, size in bytes, how many): kind 0 unified, 1 instruction, 2 data.
    pub caches: Vec<(u8, u32, u32, u32)>,
}

#[cfg(target_arch = "x86_64")]
fn cpuid(leaf: u32) -> std::arch::x86_64::CpuidResult {
    // CPUID exists on every x86-64 CPU (the call is `unsafe` only in older toolchains); callers ask for leaves up to the CPU's maximum
    #[allow(unused_unsafe)]
    let r = unsafe { std::arch::x86_64::__cpuid(leaf) };
    r
}

#[cfg(target_arch = "x86_64")]
fn cpuid_name_and_base() -> (String, u32) {
    let (max, ext) = (cpuid(0).eax, cpuid(0x8000_0000).eax);
    let mut name = Vec::new();
    if ext >= 0x8000_0004 {
        for leaf in 0x8000_0002..=0x8000_0004u32 {
            let r = cpuid(leaf);
            for w in [r.eax, r.ebx, r.ecx, r.edx] {
                name.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    let name = String::from_utf8_lossy(&name).trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string();
    (name, if max >= 0x16 { cpuid(0x16).eax & 0xffff } else { 0 })
}

#[cfg(not(target_arch = "x86_64"))]
fn cpuid_name_and_base() -> (String, u32) {
    (String::new(), 0)
}

#[cfg(windows)]
mod sys {
    #[repr(C)]
    pub struct Slpi {
        pub mask: usize,
        pub relationship: u32,
        pub pad: u32,
        pub u: [u8; 16],
    }
    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetLogicalProcessorInformation(buf: *mut Slpi, len: *mut u32) -> i32;
    }
    #[link(name = "pdh")]
    extern "system" {
        pub fn PdhOpenQueryW(source: *const u16, user: usize, query: *mut isize) -> u32;
        pub fn PdhAddEnglishCounterW(query: isize, path: *const u16, user: usize, counter: *mut isize) -> u32;
        pub fn PdhCollectQueryData(query: isize) -> u32;
        pub fn PdhGetFormattedCounterArrayW(counter: isize, format: u32, size: *mut u32, count: *mut u32, buf: *mut u8) -> u32;
        pub fn PdhCloseQuery(query: isize) -> u32;
    }
}

#[cfg(windows)]
fn read_caches() -> Vec<(u8, u32, u32, u32)> {
    use sys::*;
    let mut len = 0u32;
    // SAFETY: the first call only asks for the size; the second fills a buffer of that many bytes, whole records (`Slpi`, 32 bytes on x64).
    unsafe {
        GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len);
        if len == 0 {
            return vec![];
        }
        let n = len as usize / std::mem::size_of::<Slpi>();
        let mut buf: Vec<Slpi> = (0..n).map(|_| Slpi { mask: 0, relationship: 0, pad: 0, u: [0; 16] }).collect();
        if GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) == 0 {
            return vec![];
        }
        let mut out: Vec<(u8, u32, u32, u32)> = vec![];
        for r in buf.iter().filter(|r| r.relationship == 2) {
            // CACHE_DESCRIPTOR: level u8, associativity u8, line size u16, size u32, type u32 (0 unified, 1 instruction, 2 data, 3 trace)
            let (level, size, kind) = (r.u[0], u32::from_le_bytes(r.u[4..8].try_into().unwrap()), u32::from_le_bytes(r.u[8..12].try_into().unwrap()));
            match out.iter_mut().find(|c| c.0 == level && c.1 == kind && c.2 == size) {
                Some(c) => c.3 += 1,
                None => out.push((level, kind, size, 1)),
            }
        }
        out.sort();
        out
    }
}

#[cfg(not(windows))]
fn read_caches() -> Vec<(u8, u32, u32, u32)> {
    vec![]
}

impl CpuStatic {
    pub fn read() -> CpuStatic {
        let (name, base_mhz) = cpuid_name_and_base();
        CpuStatic { name, base_mhz, caches: read_caches() }
    }

    fn json(&self) -> Value {
        let kb = |s: u32| s as f64 / 1024.0;
        json!({
            "name": self.name,
            "base_mhz": self.base_mhz,
            "caches": self.caches.iter().map(|&(level, kind, size, count)| json!({"level": level, "kind": kind, "kb": kb(size), "count": count})).collect::<Vec<_>>(),
        })
    }
}

/// The live part: a PDH query over the processor-performance and thermal-zone counters.
pub struct CpuLive {
    pub info: CpuStatic,
    #[cfg(windows)]
    pdh: Mutex<Option<(isize, isize, isize)>>,
    #[cfg(not(windows))]
    pdh: Mutex<Option<()>>,
}

#[cfg(windows)]
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A counter's (instance name, value) pairs; empty when it has no data yet (rate counters need two samples).
#[cfg(windows)]
fn read_array(counter: isize) -> Vec<(String, f64)> {
    use sys::*;
    let (mut size, mut count) = (0u32, 0u32);
    // SAFETY: PDH's two-call pattern: the first call reports the buffer size (PDH_MORE_DATA), the second fills it with `count` items of
    // 24 bytes (name pointer, status, padding, f64) followed by the name strings, all inside `buf`.
    unsafe {
        PdhGetFormattedCounterArrayW(counter, 0x200, &mut size, &mut count, std::ptr::null_mut());
        if size == 0 {
            return vec![];
        }
        let mut buf = vec![0u8; size as usize];
        if PdhGetFormattedCounterArrayW(counter, 0x200, &mut size, &mut count, buf.as_mut_ptr()) != 0 {
            return vec![];
        }
        let mut out = vec![];
        for i in 0..count as usize {
            let item = &buf[i * 24..i * 24 + 24];
            let name_ptr = usize::from_le_bytes(item[0..8].try_into().unwrap()) as *const u16;
            let status = u32::from_le_bytes(item[8..12].try_into().unwrap());
            let value = f64::from_le_bytes(item[16..24].try_into().unwrap());
            if status > 1 {
                continue;
            }
            let mut n = 0;
            while *name_ptr.add(n) != 0 && n < 128 {
                n += 1;
            }
            out.push((String::from_utf16_lossy(std::slice::from_raw_parts(name_ptr, n)), value));
        }
        out
    }
}

impl CpuLive {
    pub fn new() -> CpuLive {
        let info = CpuStatic::read();
        #[cfg(windows)]
        let pdh = {
            use sys::*;
            // SAFETY: handles from PDH, closed in Drop; the paths are NUL-terminated UTF-16.
            unsafe {
                let mut q = 0isize;
                let (mut perf, mut therm) = (0isize, 0isize);
                if PdhOpenQueryW(std::ptr::null(), 0, &mut q) == 0
                    && PdhAddEnglishCounterW(q, wide(r"\Processor Information(_Total)\% Processor Performance").as_ptr(), 0, &mut perf) == 0 {
                    let _ = PdhAddEnglishCounterW(q, wide(r"\Thermal Zone Information(*)\Temperature").as_ptr(), 0, &mut therm);
                    PdhCollectQueryData(q);
                    Some((q, perf, therm))
                } else {
                    None
                }
            }
        };
        #[cfg(not(windows))]
        let pdh = None;
        CpuLive { info, pdh: Mutex::new(pdh) }
    }

    /// The CPU's part of the telemetry: model, base and current clock, caches, thermal zones.
    pub fn json(&self) -> Value {
        let mut v = self.info.json();
        #[cfg(windows)]
        if let Some((q, perf, therm)) = *self.pdh.lock().unwrap() {
            // SAFETY: the query and counters are the ones `new` made.
            unsafe { sys::PdhCollectQueryData(q) };
            let pct = read_array(perf).first().map(|p| p.1);
            if let (Some(p), true) = (pct, self.info.base_mhz > 0) {
                v["current_mhz"] = json!((self.info.base_mhz as f64 * p / 100.0).round());
            }
            if therm != 0 {
                let zones: Vec<Value> = read_array(therm).into_iter().filter(|z| z.1 > 273.0 && z.1 < 420.0)
                    .map(|(n, k)| json!({"name": n.trim_start_matches('\\'), "celsius": ((k - 273.15) * 10.0).round() / 10.0})).collect();
                v["thermal_zones"] = json!(zones);
            }
        }
        v
    }
}

impl Default for CpuLive {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(windows)]
impl Drop for CpuLive {
    fn drop(&mut self) {
        if let Some((q, _, _)) = *self.pdh.lock().unwrap() {
            // SAFETY: closes the query `new` opened; its counters go with it.
            unsafe { sys::PdhCloseQuery(q) };
        }
    }
}
