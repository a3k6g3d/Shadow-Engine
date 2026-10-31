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
