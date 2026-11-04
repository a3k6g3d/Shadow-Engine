//! Probe: how fast can the GPU pull from host memory, by how the memory was made and how the copies are cut?  The prompt path is bound by expert copies (~43 GiB a pass over
//! PCIe 4.0 x16, 25 GB/s on paper); the engine's copies measure ~13 GB/s.  Run: cargo test --release -p shadow-cuda --test h2d_probe -- --nocapture

use shadow_cuda::pinned::PinnedRegion;
use shadow_cuda::Context;
use std::time::Instant;

const MIB: usize = 1 << 20;

fn run(ctx: &Context, label: &str, host: *const u8, dev: u64, total: usize, chunk: usize) {
    let api = ctx.api;
    let n = total / chunk;
    let mut best = f64::MAX;
    for _ in 0..4 {
        let t = Instant::now();
        for i in 0..n {
            let r = unsafe { (api.cuMemcpyHtoDAsync)(dev + (i * chunk) as u64, host.add(i * chunk) as *const _, chunk, std::ptr::null_mut()) };
            assert_eq!(r, 0);
        }
        ctx.synchronize().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{label:<46} {:>4} x {:>4} MiB: {:6.2} GB/s", n, chunk / MIB, total as f64 / best / 1e9);
}

#[test]
fn h2d_probe() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            return;
        }
    };
    ctx.bind().unwrap();
    let total = 1024 * MIB;
    let dev = ctx.alloc(total).expect("1 GiB on the device");
    // what the engine uses: VirtualAlloc memory registered with cuMemHostRegister
    let reg = PinnedRegion::new(&ctx, total).expect("1 GiB registered");
    let p = reg.as_slice().as_ptr();
    for c in [2 * MIB, 16 * MIB, 128 * MIB, 1024 * MIB] {
        run(&ctx, "cuMemHostRegister (the engine's region)", p, dev.ptr, total, c);
    }
    // driver-allocated pinned memory: plain, and write-combined (CU_MEMHOSTALLOC_WRITECOMBINED = 4)
    for (label, flags) in [("cuMemHostAlloc", 0u32), ("cuMemHostAlloc write-combined", 4u32)] {
        let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { (ctx.api.cuMemHostAlloc)(&mut q, total, flags) };
        if r != 0 {
            println!("{label}: failed ({r})");
            continue;
        }
        // touch it (write-combined memory is write-only for the CPU; this is a write)
        unsafe { std::ptr::write_bytes(q as *mut u8, 0x5A, total) };
        for c in [2 * MIB, 16 * MIB, 128 * MIB, 1024 * MIB] {
            run(&ctx, label, q as *const u8, dev.ptr, total, c);
        }
        unsafe { (ctx.api.cuMemFreeHost)(q) };
    }

    // the same copies while CPU threads stream memory (what the read pool does for the experts that are only in the file cache: read + write)
    for threads in [1, 2, 4] {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let moved = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let hs: Vec<_> = (0..threads).map(|_| {
            let (stop, moved) = (stop.clone(), moved.clone());
            std::thread::spawn(move || {
                let src = vec![1u8; 256 * MIB];
                let mut dst = vec![0u8; 256 * MIB];
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    dst.copy_from_slice(&src);
                    std::hint::black_box(&dst);
                    moved.fetch_add(256 * MIB as u64, std::sync::atomic::Ordering::Relaxed);
                }
            })
        }).collect();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let m0 = moved.load(std::sync::atomic::Ordering::Relaxed);
        let t0 = Instant::now();
        run(&ctx, &format!("cuMemHostRegister + {threads} memcpy thread(s)"), p, dev.ptr, total, 2 * MIB);
        let cpu = (moved.load(std::sync::atomic::Ordering::Relaxed) - m0) as f64 / t0.elapsed().as_secs_f64() / 1e9;
        println!("   the CPU threads copied {cpu:.1} GB/s meanwhile (read + write = {:.1} GB/s of DRAM traffic)", 2.0 * cpu);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in hs {
            h.join().unwrap();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// Can the OS file cache itself be the pinned memory?  Map a slice of the model file (read-only view) and register it with the driver: if the GPU can DMA from it, the experts
// that are not in the RAM tier need no CPU copy into a staging buffer (the ~22 GiB of DRAM traffic a prompt pass pays for them, which the probe above shows halves the copy rate).
#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    extern "system" {
        pub fn CreateFileW(name: *const u16, access: u32, share: u32, sec: *mut c_void, disp: u32, flags: u32, tmpl: *mut c_void) -> *mut c_void;
        pub fn CreateFileMappingW(file: *mut c_void, sec: *mut c_void, prot: u32, hi: u32, lo: u32, name: *const u16) -> *mut c_void;
        pub fn MapViewOfFile(map: *mut c_void, access: u32, off_hi: u32, off_lo: u32, bytes: usize) -> *mut c_void;
        pub fn UnmapViewOfFile(p: *const c_void) -> i32;
        pub fn CloseHandle(h: *mut c_void) -> i32;
    }
}

#[cfg(windows)]
#[test]
fn mapped_file_register_probe() {
    use std::os::windows::ffi::OsStrExt;
    let Some(model) = std::env::var("SHADOW_PROBE_FILE").ok() else {
        eprintln!("SKIPPED: set SHADOW_PROBE_FILE to a big file (a GGUF shard)");
        return;
    };
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            return;
        }
    };
    ctx.bind().unwrap();
    let size = 1024 * MIB;
    let offset: u64 = 8 << 30; // a multiple of the 64 KiB allocation granularity
    let wide: Vec<u16> = std::path::Path::new(&model).as_os_str().encode_wide().chain(Some(0)).collect();
    unsafe {
        let f = win::CreateFileW(wide.as_ptr(), 0x8000_0000, 1 | 2, std::ptr::null_mut(), 3, 0x0800_0000 /* sequential scan */, std::ptr::null_mut());
        assert!(f as isize != -1, "cannot open the file");
        let m = win::CreateFileMappingW(f, std::ptr::null_mut(), 0x02 /* PAGE_READONLY */, 0, 0, std::ptr::null());
        assert!(!m.is_null(), "CreateFileMapping failed");
        let v = win::MapViewOfFile(m, 0x4 /* FILE_MAP_READ */, (offset >> 32) as u32, offset as u32, size);
        assert!(!v.is_null(), "MapViewOfFile failed");
        // read it once so the pages are in the file cache (and measure that, it is what a cold expert costs)
        let t = Instant::now();
        let mut sum = 0u64;
        let s = std::slice::from_raw_parts(v as *const u64, size / 8);
        for i in (0..s.len()).step_by(512) {
            sum = sum.wrapping_add(s[i]);
        }
        println!("first touch of 1 GiB of the mapped file (4 KiB steps): {:?} (sum {sum})", t.elapsed());
        let dev = ctx.alloc(size).expect("1 GiB on the device");
        for flags in [0u32, 8u32 /* READ_ONLY */, 1 | 8, 2 | 8] {
            let t = Instant::now();
            let r = (ctx.api.cuMemHostRegister)(v, size, flags);
            let reg = t.elapsed();
            if r != 0 {
                println!("cuMemHostRegister(file view, flags {flags}): failed ({r})");
                continue;
            }
            println!("cuMemHostRegister(file view, flags {flags}): ok in {reg:?}");
            for c in [2 * MIB, 128 * MIB] {
                run(&ctx, &format!("DMA from the registered file view (flags {flags})"), v as *const u8, dev.ptr, size, c);
            }
            (ctx.api.cuMemHostUnregister)(v);
        }
        win::UnmapViewOfFile(v);
        win::CloseHandle(m);
        win::CloseHandle(f);
    }
}

/// Is what the GPU gets from a registered file view the file's bytes, also for pages nobody touched yet (cold), and what does that cost?
#[cfg(windows)]
#[test]
fn mapped_file_correctness_probe() {
    use std::io::{Read, Seek, SeekFrom};
    use std::os::windows::ffi::OsStrExt;
    let Some(model) = std::env::var("SHADOW_PROBE_FILE").ok() else {
        return;
    };
    let Ok(ctx) = Context::new(0) else { return };
    ctx.bind().unwrap();
    let size = 256 * MIB;
    let wide: Vec<u16> = std::path::Path::new(&model).as_os_str().encode_wide().chain(Some(0)).collect();
    let dev = ctx.alloc(size).expect("256 MiB on the device");
    for (label, offset) in [("probably cold", 30u64 << 30), ("cold, another", 33u64 << 30), ("the first one again (warm)", 30u64 << 30)] {
        unsafe {
            let f = win::CreateFileW(wide.as_ptr(), 0x8000_0000, 1 | 2, std::ptr::null_mut(), 3, 0x0800_0000, std::ptr::null_mut());
            let m = win::CreateFileMappingW(f, std::ptr::null_mut(), 0x02, 0, 0, std::ptr::null());
            let v = win::MapViewOfFile(m, 0x4, (offset >> 32) as u32, offset as u32, size);
            assert!(!v.is_null());
            let t = Instant::now();
            let r = (ctx.api.cuMemHostRegister)(v, size, 8);
            let reg = t.elapsed();
            assert_eq!(r, 0, "register failed ({r})");
            let t = Instant::now();
            let r = (ctx.api.cuMemcpyHtoDAsync)(dev.ptr, v as *const _, size, std::ptr::null_mut());
            assert_eq!(r, 0);
            ctx.synchronize().unwrap();
            let copy = t.elapsed();
            let mut back = vec![0u8; size];
            (ctx.api.cuMemcpyDtoH)(back.as_mut_ptr() as *mut _, dev.ptr, size);
            let mut want = vec![0u8; size];
            let mut file = std::fs::File::open(&model).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.read_exact(&mut want).unwrap();
            println!("{label:<28} register {reg:?}, DMA of 256 MiB {copy:?} ({:.2} GB/s), bytes {}", size as f64 / copy.as_secs_f64() / 1e9,
                     if back == want { "MATCH the file" } else { "DIFFER from the file" });
            (ctx.api.cuMemHostUnregister)(v);
            win::UnmapViewOfFile(v);
            win::CloseHandle(m);
            win::CloseHandle(f);
        }
    }
}

/// A copy with non-temporal stores (no read-for-ownership of the destination, no cache pollution): 2 DRAM touches a byte instead of memcpy's 3.  Does the DMA suffer less beside it?
#[cfg(target_arch = "x86_64")]
#[test]
fn contention_by_copy_kind_probe() {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::Arc;
    let Ok(ctx) = Context::new(0) else { return };
    ctx.bind().unwrap();
    let total = 1024 * MIB;
    let dev = ctx.alloc(total).unwrap();
    let reg = PinnedRegion::new(&ctx, total).unwrap();
    let p = reg.as_slice().as_ptr();
    #[target_feature(enable = "avx2")]
    unsafe fn nt_copy(dst: *mut u8, src: *const u8, n: usize) {
        let mut i = 0;
        while i + 64 <= n {
            let a = _mm256_load_si256(src.add(i) as *const __m256i);
            let b = _mm256_load_si256(src.add(i + 32) as *const __m256i);
            _mm256_stream_si256(dst.add(i) as *mut __m256i, a);
            _mm256_stream_si256(dst.add(i + 32) as *mut __m256i, b);
            i += 64;
        }
        _mm_sfence();
    }
    for (label, nt) in [("plain memcpy", false), ("non-temporal stores", true)] {
        for threads in [1usize, 2] {
            let stop = Arc::new(AtomicBool::new(false));
            let moved = Arc::new(AtomicU64::new(0));
            let hs: Vec<_> = (0..threads).map(|_| {
                let (stop, moved) = (stop.clone(), moved.clone());
                std::thread::spawn(move || {
                    // 64-byte aligned buffers
                    let layout = std::alloc::Layout::from_size_align(256 * MIB, 64).unwrap();
                    let (src, dst) = unsafe { (std::alloc::alloc_zeroed(layout), std::alloc::alloc_zeroed(layout)) };
                    while !stop.load(Relaxed) {
                        unsafe {
                            if nt { nt_copy(dst, src, 256 * MIB) } else { std::ptr::copy_nonoverlapping(src, dst, 256 * MIB) }
                        }
                        moved.fetch_add(256 * MIB as u64, Relaxed);
                    }
                    unsafe { std::alloc::dealloc(src, layout); std::alloc::dealloc(dst, layout); }
                })
            }).collect();
            std::thread::sleep(std::time::Duration::from_millis(300));
            let m0 = moved.load(Relaxed);
            let t0 = Instant::now();
            run(&ctx, &format!("DMA beside {threads} thread(s) of {label}"), p, dev.ptr, total, 2 * MIB);
            let cpu = (moved.load(Relaxed) - m0) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("   the CPU copied {cpu:.1} GB/s meanwhile");
            stop.store(true, Relaxed);
            for h in hs {
                h.join().unwrap();
            }
        }
    }
}
