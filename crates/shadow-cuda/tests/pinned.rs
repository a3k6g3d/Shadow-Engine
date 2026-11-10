//! Stage 3's RAM rule, as tests: Shadow refuses to pin memory that would leave less than 4 GiB of RAM available, and
//! what it does pin is page-locked memory the GPU copies from correctly.  "Lowers available memory artificially":
//! the region takes its RAM figure from a function, and these tests hand it a low one - no memory is consumed to get
//! there.

use shadow_cuda::pinned::{available_ram, PinnedRegion, RAM_FLOOR};
use shadow_cuda::Context;
use std::sync::atomic::{AtomicU64, Ordering};

const GIB: u64 = 1 << 30;

fn context() -> Option<Context> {
    match Context::new(0) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIPPED, not passed: no CUDA device ({e})");
            None
        }
    }
}

fn five_gib() -> u64 {
    5 * GIB
}

#[test]
fn refuses_to_pin_below_the_floor() {
    let Some(ctx) = context() else { return };
    // 5 GiB available: 1 GiB more would leave exactly 4, so 1 GiB + one page must be refused, before any allocation
    assert!(PinnedRegion::with_ram_source(&ctx, (GIB + 4096) as usize, five_gib).is_err());
    // and 64 MiB is fine (it is pinned for real here, so keep it small)
    let r = PinnedRegion::with_ram_source(&ctx, 64 << 20, five_gib).expect("64 MiB with 5 GiB available");
    assert_eq!(r.len(), 64 << 20);
}

static FAKE: AtomicU64 = AtomicU64::new(0);
fn fake() -> u64 {
    FAKE.load(Ordering::SeqCst)
}

#[test]
fn notices_when_ram_falls_after_pinning() {
    let Some(ctx) = context() else { return };
    FAKE.store(8 * GIB, Ordering::SeqCst);
    let r = PinnedRegion::with_ram_source(&ctx, 16 << 20, fake).expect("16 MiB with 8 GiB available");
    assert!(r.check_floor().is_ok());
    FAKE.store(RAM_FLOOR - 1, Ordering::SeqCst); // something else took the RAM
    assert!(r.check_floor().is_err());
}

#[test]
fn refuses_the_real_machine_too() {
    let Some(ctx) = context() else { return };
    // asking for everything that is available, less 2 GiB, must be refused by the real figure; the check comes before
    // the allocation, so nothing is pinned
    let avail = available_ram();
    assert!(avail > 0, "GlobalMemoryStatusEx gave nothing");
    let ask = avail.saturating_sub(2 * GIB).max(1);
    assert!(PinnedRegion::new(&ctx, ask as usize).is_err(), "pinning {ask} bytes of {avail} available was allowed");
}

#[test]
fn the_gpu_copies_from_pinned_memory_exactly() {
    let Some(ctx) = context() else { return };
    let mut r = PinnedRegion::new(&ctx, 32 << 20).expect("32 MiB");
    for (i, b) in r.as_mut_slice().iter_mut().enumerate() {
        *b = (i.wrapping_mul(2654435761) >> 13) as u8;
    }
    let d = ctx.alloc(r.len()).unwrap();
    d.upload(r.as_slice()).unwrap();
    let mut back = vec![0u8; r.len()];
    d.download(&mut back).unwrap();
    assert!(back == r.as_slice(), "the device copy differs from the pinned source");
}

/// A region built in chunks gives its tail back chunk by chunk (Phase A3): what stays keeps its addresses and its bytes and the GPU still copies it exactly, the pinned count falls by what
/// was released, and the released part is gone from `len`.
#[test]
fn a_chunked_region_gives_its_tail_back() {
    let Some(ctx) = context() else { return };
    let chunk = 1usize << 20;
    let mut r = PinnedRegion::new_chunked(&ctx, 8 * chunk + 4096, chunk).expect("8 MiB and a page, in 1 MiB chunks");
    assert_eq!(r.len(), 8 * chunk + 4096);
    for (i, b) in r.as_mut_slice().iter_mut().enumerate() {
        *b = (i.wrapping_mul(2654435761) >> 13) as u8;
    }
    assert_eq!(PinnedRegion::fit(5, 10, 16), 5);
    assert_eq!(PinnedRegion::fit(10, 10, 16), 16, "a range that would cross goes to the next chunk");
    assert_eq!(PinnedRegion::fit(0, 16, 16), 0);
    // every chunk has its own device address and the GPU reads each chunk through it
    for k in 0..8 {
        let off = k * chunk + 4096;
        let d = ctx.alloc(4096).unwrap();
        let rr = unsafe { (ctx.api.cuMemcpyDtoD)(d.ptr, r.dev_addr(off), 4096) };
        assert_eq!(rr, 0, "chunk {k}: copy through dev_addr failed");
        let mut back = vec![0u8; 4096];
        d.download(&mut back).unwrap();
        assert!(back == &r.as_slice()[off..off + 4096], "chunk {k} read through its device address differs");
    }
    let want: Vec<u8> = r.as_slice().to_vec();
    let base = r.as_slice().as_ptr();
    let pinned0 = shadow_cuda::pinned::pinned_bytes();
    // 2.5 chunks asked for: whole chunks only, from the end (the last one is the 4 KiB tail chunk): 4 KiB + 3 MiB is what it takes to reach 2.5 MiB
    let freed = r.shrink_by(2 * chunk + chunk / 2);
    assert_eq!(freed, 4096 + 3 * chunk, "released the tail chunk and three full ones (2.5 MiB is more than the tail and two chunks)");
    assert_eq!(r.len(), 5 * chunk);
    assert_eq!(r.as_slice().as_ptr(), base, "nothing moved");
    let _ = pinned0; // (other tests pin at the same time, so the process-wide count cannot be compared exactly here)
    assert!(r.as_slice() == &want[..5 * chunk], "the bytes that stayed changed");
    // (a host range across two registrations is refused by the driver - hence one expert never crosses a chunk)
    for k in 0..5 {
        let d = ctx.alloc(chunk).unwrap();
        d.upload(&r.as_slice()[k * chunk..(k + 1) * chunk]).unwrap();
        let mut back = vec![0u8; chunk];
        d.download(&mut back).unwrap();
        assert!(back == &want[k * chunk..(k + 1) * chunk], "the GPU copy of chunk {k}, which stayed, differs");
    }
    assert_eq!(r.shrink_by(usize::MAX), 5 * chunk, "everything can go");
    assert_eq!(r.len(), 0);
    assert_eq!(r.shrink_by(1), 0);
}

/// Host-to-device bandwidth of the RAM tier's kind of memory, by copy size (`cargo test --release -p shadow-cuda --test pinned -- --ignored --nocapture`).
#[test]
#[ignore]
fn host_to_device_bandwidth() {
    let Some(ctx) = context() else { return };
    let total = 1usize << 30;
    let region = PinnedRegion::new(&ctx, total).expect("1 GiB pinned");
    let dev = ctx.alloc(total).unwrap();
    let stream = ctx.stream().unwrap();
    let api = ctx.api;
    for &chunk in &[256usize << 10, 2 << 20, 16 << 20, 128 << 20] {
        for rep in 0..2 {
            let t0 = std::time::Instant::now();
            let mut off = 0;
            while off + chunk <= total {
                let r = unsafe { (api.cuMemcpyHtoDAsync)(dev.ptr + off as u64, region.as_slice()[off..].as_ptr() as *const _, chunk, stream.raw()) };
                assert_eq!(r, 0);
                off += chunk;
            }
            stream.synchronize().unwrap();
            let s = t0.elapsed().as_secs_f64();
            if rep == 1 {
                println!("{:4} KiB copies: {:.1} GB/s", chunk >> 10, off as f64 / s / 1e9);
            }
        }
    }
}

/// `cuMemcpyBatchAsync_v2`: 64 copies of 2 MiB from a pinned region into scattered device spots come out exactly, and the call is cheap on the host.
#[test]
fn batched_host_to_device_copies_are_exact() {
    let Some(ctx) = context() else { return };
    let Some(batch) = ctx.api.cuMemcpyBatchAsync else { return eprintln!("SKIPPED, not passed: the driver has no cuMemcpyBatchAsync") };
    let chunk = 2usize << 20;
    let n = 64usize;
    let mut region = PinnedRegion::new(&ctx, n * chunk).expect("128 MiB pinned");
    for (i, b) in region.as_mut_slice().iter_mut().enumerate() {
        *b = (i.wrapping_mul(31) ^ (i >> 12)) as u8;
    }
    let dev = ctx.alloc(2 * n * chunk).unwrap();
    let stream = ctx.stream().unwrap();
    // copy i goes to slot (n - 1 - i) * 2 (every other slot, reversed)
    let mut dsts: Vec<u64> = (0..n).map(|i| dev.ptr + (((n - 1 - i) * 2) * chunk) as u64).collect();
    let mut srcs: Vec<u64> = (0..n).map(|i| region.dev_ptr + (i * chunk) as u64).collect();
    let mut lens = vec![chunk; n];
    let mut attr = [1u32, 0, 0, 0, 0, 0];
    let mut idx = 0usize;
    let t0 = std::time::Instant::now();
    let r = unsafe { batch(dsts.as_mut_ptr(), srcs.as_mut_ptr(), lens.as_mut_ptr(), n, attr.as_mut_ptr() as *mut _, &mut idx, 1, stream.raw()) };
    let host_us = t0.elapsed().as_secs_f64() * 1e6;
    assert_eq!(r, 0, "cuMemcpyBatchAsync returned {r}");
    stream.synchronize().unwrap();
    eprintln!("batched copy: host time {host_us:.0} us for {n} copies");
    let mut got = vec![0u8; 2 * n * chunk];
    dev.download(&mut got).unwrap();
    for i in 0..n {
        let slot = (n - 1 - i) * 2;
        assert_eq!(&got[slot * chunk..(slot + 1) * chunk], &region.as_slice()[i * chunk..(i + 1) * chunk], "copy {i} differs");
    }
}

/// How fast the model file reads from the SSD: random 2 MiB reads over the whole first shard, buffered and with FILE_FLAG_NO_BUFFERING, by thread count
/// (`cargo test --release -p shadow-cuda --test pinned -- --ignored --nocapture ssd_read`).
#[test]
#[ignore]
fn ssd_read_bandwidth() {
    use std::os::windows::fs::{FileExt, OpenOptionsExt};
    let path = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf".into());
    if !std::path::Path::new(&path).is_file() {
        return eprintln!("SKIPPED: no model");
    }
    let size = std::fs::metadata(&path).unwrap().len();
    let chunk = 2usize << 20;
    for &direct in &[false, true] {
        for &threads in &[4usize, 8, 16] {
            let total = std::sync::atomic::AtomicU64::new(0);
            let t0 = std::time::Instant::now();
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (path, total) = (&path, &total);
                    s.spawn(move || {
                        let mut o = std::fs::OpenOptions::new();
                        o.read(true);
                        if direct {
                            o.custom_flags(0x2000_0000);
                        }
                        let f = o.open(path).unwrap();
                        let layout = std::alloc::Layout::from_size_align(chunk, 4096).unwrap();
                        // SAFETY: a 4096-aligned allocation of `chunk` bytes, used only for these reads and freed below.
                        let buf = unsafe { std::alloc::alloc(layout) };
                        let mut rnd = 0x9e37_79b9_7f4a_7c15u64 ^ ((t as u64 + 1) * 0x1234_5678_9abc);
                        while t0.elapsed().as_secs_f64() < 6.0 {
                            rnd ^= rnd << 13;
                            rnd ^= rnd >> 7;
                            rnd ^= rnd << 17;
                            let off = (rnd % (size - chunk as u64)) & !4095;
                            // SAFETY: the buffer is `chunk` bytes, exclusively ours.
                            let n = f.seek_read(unsafe { std::slice::from_raw_parts_mut(buf, chunk) }, off).unwrap();
                            total.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                        // SAFETY: allocated above with this layout.
                        unsafe { std::alloc::dealloc(buf, layout) };
                    });
                }
            });
            println!("{:8} {:2} threads: {:.2} GB/s", if direct { "direct" } else { "buffered" }, threads,
                     total.load(std::sync::atomic::Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
    }
}

/// Unbuffered random reads of the model file over thread count and read size: is 3.5 GB/s the drive's ceiling, or the way the reads are issued?
/// (`cargo test --release -p shadow-cuda --test pinned -- --ignored --nocapture ssd_read_sweep`).
#[test]
#[ignore]
fn ssd_read_sweep() {
    use std::os::windows::fs::{FileExt, OpenOptionsExt};
    let path = std::env::var("SHADOW_MODEL").unwrap_or_else(|_| "J:/Strata-work/models/swift15/IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf".into());
    if !std::path::Path::new(&path).is_file() {
        return eprintln!("SKIPPED: no model");
    }
    let size = std::fs::metadata(&path).unwrap().len();
    for &chunk in &[128usize << 10, 512 << 10, 2 << 20, 8 << 20] {
        for &threads in &[1usize, 4, 16, 32, 64] {
            let total = std::sync::atomic::AtomicU64::new(0);
            let t0 = std::time::Instant::now();
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (path, total) = (&path, &total);
                    s.spawn(move || {
                        let f = std::fs::OpenOptions::new().read(true).custom_flags(0x2000_0000).open(path).unwrap();
                        let layout = std::alloc::Layout::from_size_align(chunk, 4096).unwrap();
                        // SAFETY: a 4096-aligned allocation of `chunk` bytes, used only for these reads and freed below.
                        let buf = unsafe { std::alloc::alloc(layout) };
                        let mut rnd = 0x9e37_79b9_7f4a_7c15u64 ^ ((t as u64 + 1) * 0x1234_5678_9abc);
                        while t0.elapsed().as_secs_f64() < 3.0 {
                            rnd ^= rnd << 13;
                            rnd ^= rnd >> 7;
                            rnd ^= rnd << 17;
                            let off = (rnd % (size - chunk as u64)) & !4095;
                            // SAFETY: the buffer is `chunk` bytes, exclusively ours.
                            let n = f.seek_read(unsafe { std::slice::from_raw_parts_mut(buf, chunk) }, off).unwrap();
                            total.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                        // SAFETY: allocated above with this layout.
                        unsafe { std::alloc::dealloc(buf, layout) };
                    });
                }
            });
            println!("direct {:5} KiB x {:2} threads: {:.2} GB/s", chunk >> 10, threads,
                     total.load(std::sync::atomic::Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
    }
}
