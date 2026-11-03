//! Probe: does the CUDA virtual memory API work on this driver / WDDM, and what does it cost?  Reserve an address range, back it with 2 MiB physical chunks
//! (cuMemCreate + cuMemMap + cuMemSetAccess), write and read it, give half of the chunks back (the free VRAM must grow), map new ones, and time each step.
//! This is the mechanism for lending the unfilled tail of the KV window: the kernels keep their pointers, only the physical backing comes and goes.
//! Run: cargo test --release -p shadow-cuda --test vmm_probe -- --nocapture

use shadow_cuda::Context;
use std::time::Instant;

const MIB: usize = 1 << 20;

#[test]
fn vmm_probe() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            return;
        }
    };
    let api = ctx.api;
    ctx.bind().unwrap();
    // CUmemAllocationProp: type PINNED (1), no handle types, location {DEVICE (1), id 0}, no win32 metadata, flags 0 - 32 bytes
    let mut prop = [0u8; 32];
    prop[0..4].copy_from_slice(&1u32.to_le_bytes());
    prop[8..12].copy_from_slice(&1u32.to_le_bytes());
    let mut gran = 0usize;
    let r = unsafe { (api.cuMemGetAllocationGranularity)(&mut gran, prop.as_ptr() as *const _, 1) };
    println!("granularity (recommended): {gran} bytes (result {r})");
    assert_eq!(r, 0);
    let chunk = gran.max(2 * MIB);
    let n = 768usize; // 1.5 GiB at 2 MiB
    let total = n * chunk;
    let mut base = 0u64;
    let t = Instant::now();
    let r = unsafe { (api.cuMemAddressReserve)(&mut base, total, 0, 0, 0) };
    println!("reserve {} MiB of address space: result {r}, {:?}", total / MIB, t.elapsed());
    assert_eq!(r, 0);
    // access descriptor: location {DEVICE, 0}, flags READWRITE (3)
    let mut acc = [0u8; 12];
    acc[0..4].copy_from_slice(&1u32.to_le_bytes());
    acc[8..12].copy_from_slice(&3u32.to_le_bytes());
    let free0 = ctx.mem_info().unwrap().0;
    let mut handles = vec![0u64; n];
    let t = Instant::now();
    for i in 0..n {
        let r = unsafe { (api.cuMemCreate)(&mut handles[i], chunk, prop.as_ptr() as *const _, 0) };
        assert_eq!(r, 0, "cuMemCreate #{i} failed ({r})");
    }
    let t_create = t.elapsed();
    let t = Instant::now();
    for i in 0..n {
        let r = unsafe { (api.cuMemMap)(base + (i * chunk) as u64, chunk, 0, handles[i], 0) };
        assert_eq!(r, 0, "cuMemMap #{i} failed ({r})");
    }
    let t_map = t.elapsed();
    let t = Instant::now();
    let r = unsafe { (api.cuMemSetAccess)(base, total, acc.as_ptr() as *const _, 1) };
    assert_eq!(r, 0, "cuMemSetAccess failed ({r})");
    let t_access = t.elapsed();
    let free1 = ctx.mem_info().unwrap().0;
    println!("{} chunks of {} MiB: create {:?}, map {:?}, set access (one call) {:?}; free VRAM {} -> {} MiB", n, chunk / MIB, t_create, t_map, t_access, free0 >> 20, free1 >> 20);
    // use it: fill, read back at both ends and the middle
    let t = Instant::now();
    let r = unsafe { (api.cuMemsetD8)(base, 0xA5, total) };
    assert_eq!(r, 0);
    ctx.synchronize().unwrap();
    println!("memset {} MiB through the mapped range: {:?}", total / MIB, t.elapsed());
    for &off in &[0usize, total / 2, total - 16] {
        let mut b = [0u8; 16];
        let r = unsafe { (api.cuMemcpyDtoH)(b.as_mut_ptr() as *mut _, base + off as u64, 16) };
        assert_eq!(r, 0);
        assert_eq!(b, [0xA5; 16], "readback at {off}");
    }
    // give the back half away: unmap + release, free VRAM must grow by half
    let half = n / 2;
    let t = Instant::now();
    let r = unsafe { (api.cuMemUnmap)(base + (half * chunk) as u64, (n - half) * chunk) };
    assert_eq!(r, 0, "cuMemUnmap failed ({r})");
    for i in half..n {
        let r = unsafe { (api.cuMemRelease)(handles[i]) };
        assert_eq!(r, 0);
        handles[i] = 0;
    }
    let t_back = t.elapsed();
    let free2 = ctx.mem_info().unwrap().0;
    println!("unmap + release {} chunks: {:?}; free VRAM {} -> {} MiB (+{} MiB)", n - half, t_back, free1 >> 20, free2 >> 20, (free2 - free1) >> 20);
    assert!(free2 > free1 + (half * chunk / 2), "the released physical memory did not come back to the free pool");
    // the front half is untouched; hand the freed memory to an ordinary allocation (the lend), then take it back
    let mut b = [0u8; 16];
    let r = unsafe { (api.cuMemcpyDtoH)(b.as_mut_ptr() as *mut _, base + (half * chunk - 16) as u64, 16) };
    assert_eq!(r, 0);
    assert_eq!(b, [0xA5; 16], "front half changed");
    let lent = ctx.alloc((n - half) * chunk - 64 * MIB).expect("an ordinary 700 MiB allocation into the released memory");
    println!("an ordinary cuMemAlloc of {} MiB into the released memory: ok", ((n - half) * chunk - 64 * MIB) / MIB);
    drop(lent);
    // remap fresh chunks over the back half
    let t = Instant::now();
    for i in half..n {
        let r = unsafe { (api.cuMemCreate)(&mut handles[i], chunk, prop.as_ptr() as *const _, 0) };
        assert_eq!(r, 0, "re-create #{i} failed ({r})");
        let r = unsafe { (api.cuMemMap)(base + (i * chunk) as u64, chunk, 0, handles[i], 0) };
        assert_eq!(r, 0, "re-map #{i} failed ({r})");
    }
    let r = unsafe { (api.cuMemSetAccess)(base + (half * chunk) as u64, (n - half) * chunk, acc.as_ptr() as *const _, 1) };
    assert_eq!(r, 0);
    println!("create + map + access for {} chunks again: {:?}", n - half, t.elapsed());
    let r = unsafe { (api.cuMemsetD8)(base + (half * chunk) as u64, 0x5A, (n - half) * chunk) };
    assert_eq!(r, 0);
    ctx.synchronize().unwrap();
    // tear down
    let r = unsafe { (api.cuMemUnmap)(base, total) };
    assert_eq!(r, 0);
    for h in handles {
        if h != 0 {
            unsafe { (api.cuMemRelease)(h) };
        }
    }
    unsafe { (api.cuMemAddressFree)(base, total) };
    println!("VMM probe passed");
}
