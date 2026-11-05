//! Probe: does the way a long prompt copies a layer's experts (~460 pieces of ~2.4 MB from scattered offsets of a big registered region, in one
//! `cuMemcpyBatchAsync`) reach the link's rate?  The engine's per-layer copies measure ~15 GB/s, a contiguous copy from its region 28 GB/s.
//!   SHADOW_PROBE_GIB=16 cargo test --release -p shadow-cuda --test scatter_probe -- --ignored --nocapture
use shadow_cuda::pinned::PinnedRegion;
use shadow_cuda::Context;
use std::time::Instant;

#[test]
#[ignore]
fn scatter_probe() {
    let ctx = match Context::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            return;
        }
    };
    ctx.bind().unwrap();
    let gib: usize = std::env::var("SHADOW_PROBE_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let region_bytes = gib << 30;
    let piece = 2_400_000usize;
    let n = 460usize;
    let total = n * piece;
    let dev = ctx.alloc(total.next_multiple_of(1 << 20)).expect("device buffer");
    let reg = PinnedRegion::new(&ctx, region_bytes).expect("registered region");
    // touch every page (the tier is filled before use)
    let host = reg.as_slice().as_ptr() as *mut u8;
    for off in (0..region_bytes).step_by(4096) {
        unsafe { *host.add(off) = (off >> 12) as u8 };
    }
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let slots = region_bytes / piece;
    let mut offs: Vec<usize> = (0..n).map(|_| (next() as usize % slots) * piece).collect();
    offs.sort_unstable();
    offs.dedup();
    let n = offs.len();
    let api = ctx.api;
    let stream = ctx.stream().unwrap();
    let time = |label: &str, f: &dyn Fn()| {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            f();
            stream.synchronize().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{label:<62} {:6.2} GB/s", (n * piece) as f64 / best / 1e9);
    };
    time("one contiguous copy of the same bytes", &|| {
        let r = unsafe { (api.cuMemcpyHtoDAsync)(dev.ptr, host as *const _, n * piece, stream.raw()) };
        assert_eq!(r, 0);
    });
    time(&format!("{n} scattered pieces, one cuMemcpyHtoDAsync each"), &|| {
        for (i, &o) in offs.iter().enumerate() {
            let r = unsafe { (api.cuMemcpyHtoDAsync)(dev.ptr + (i * piece) as u64, host.add(o) as *const _, piece, stream.raw()) };
            assert_eq!(r, 0);
        }
    });
    let mut sorted_contig: Vec<usize> = (0..n).map(|i| i * piece).collect();
    sorted_contig.truncate(n);
    time(&format!("{n} adjacent pieces, one cuMemcpyHtoDAsync each"), &|| {
        for (i, &o) in sorted_contig.iter().enumerate() {
            let r = unsafe { (api.cuMemcpyHtoDAsync)(dev.ptr + (i * piece) as u64, host.add(o) as *const _, piece, stream.raw()) };
            assert_eq!(r, 0);
        }
    });
    // the same contiguous copy while another stream keeps the GPU's memory busy (device-to-device copies of 1 GiB, like a memory-bound kernel)
    {
        let other = ctx.stream().unwrap();
        let (a, b) = (ctx.alloc(1 << 30).unwrap(), ctx.alloc(1 << 30).unwrap());
        for _ in 0..40 {
            let r = unsafe { (api.cuMemcpyDtoDAsync)(b.ptr, a.ptr, 1 << 30, other.raw()) };
            assert_eq!(r, 0);
        }
        time("one contiguous copy while D2D copies saturate VRAM", &|| {
            let r = unsafe { (api.cuMemcpyHtoDAsync)(dev.ptr, host as *const _, n * piece, stream.raw()) };
            assert_eq!(r, 0);
        });
        other.synchronize().unwrap();
    }
    if let Some(batch) = api.cuMemcpyBatchAsync {
        for (label, src_offs) in [("scattered", &offs), ("adjacent", &sorted_contig)] {
            time(&format!("{n} {label} pieces, one cuMemcpyBatchAsync (as the engine)"), &|| {
                let mut dsts: Vec<u64> = (0..n).map(|i| dev.ptr + (i * piece) as u64).collect();
                let mut srcs: Vec<u64> = src_offs.iter().map(|&o| reg.dev_ptr + o as u64).collect();
                let mut lens = vec![piece; n];
                let mut attr = [1u32, 0, 0, 0, 0, 0];
                let mut idx = 0usize;
                let r = unsafe { batch(dsts.as_mut_ptr(), srcs.as_mut_ptr(), lens.as_mut_ptr(), n, attr.as_mut_ptr() as *mut std::ffi::c_void, &mut idx, 1, stream.raw()) };
                assert_eq!(r, 0);
            });
        }
    }
}
