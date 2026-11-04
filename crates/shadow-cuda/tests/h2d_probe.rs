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
}
