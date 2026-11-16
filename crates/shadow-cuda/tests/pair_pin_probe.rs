//! STATUS 155: why does a process holding two CUDA contexts fail once it has pinned more than ~14 GiB of host memory?  A probe, not a regression test (`#[ignore]`):
//! `cargo test --release -p shadow-cuda --test pair_pin_probe -- --ignored --nocapture`, run under `tools/run_capped.ps1`.  For each size in `SHADOW_PROBE_GIB` (default
//! "8,14,16,20,24,32") it pins that much in 1 GiB chunks (in the context `SHADOW_PROBE_ON`, 0 or 1, default 0), then on both contexts creates events, allocates and zeroes small
//! buffers, and reports what failed first.  `SHADOW_PIN_PORTABLE=1` registers with CU_MEMHOSTREGISTER_PORTABLE too (read by `pinned.rs`).

use shadow_cuda::pinned::PinnedRegion;
use shadow_cuda::Context;

const GIB: usize = 1 << 30;

fn poke(ctx: &Context, label: &str) -> String {
    if let Err(e) = ctx.bind() {
        return format!("{label}: bind failed: {e}");
    }
    let mut events = vec![];
    for i in 0..2000 {
        match ctx.event() {
            Ok(e) => events.push(e),
            Err(e) => return format!("{label}: event {i} failed: {e}"),
        }
    }
    let mut bufs = vec![];
    for i in 0..300 {
        match ctx.alloc(3 << 20) {
            Ok(mut b) => {
                if let Err(e) = b.zero() {
                    return format!("{label}: zeroing buffer {i} failed: {e}");
                }
                bufs.push(b);
            }
            Err(e) => return format!("{label}: alloc {i} failed: {e}"),
        }
    }
    format!("{label}: ok ({} events, {} buffers)", events.len(), bufs.len())
}

#[test]
#[ignore]
fn what_breaks_with_two_contexts_and_a_big_pinned_region() {
    // SHADOW_PROBE_SINGLE=1: only context 0 exists (the control: a process that sees one card)
    let single = std::env::var("SHADOW_PROBE_SINGLE").is_ok_and(|v| v == "1");
    let Ok(c0) = Context::new(0) else {
        eprintln!("SKIPPED: needs a CUDA device");
        return;
    };
    let Ok(c1) = (if single { Context::new(0) } else { Context::new(1) }) else {
        eprintln!("SKIPPED: needs two CUDA devices");
        return;
    };
    let on: usize = std::env::var("SHADOW_PROBE_ON").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let sizes: Vec<usize> = std::env::var("SHADOW_PROBE_GIB").unwrap_or_else(|_| "8,14,16,20,24,32".into()).split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let pin_ctx = if on == 0 { &c0 } else { &c1 };
    for g in sizes {
        let t0 = std::time::Instant::now();
        match PinnedRegion::new_chunked(pin_ctx, g * GIB, GIB) {
            Err(e) => {
                eprintln!("PROBE pin {g} GiB on ctx{on}: FAILED: {e}");
                continue;
            }
            Ok(region) => {
                eprintln!("PROBE pin {g} GiB on ctx{on}: ok in {:.1} s", t0.elapsed().as_secs_f64());
                eprintln!("PROBE   {}", poke(&c0, "ctx0"));
                eprintln!("PROBE   {}", poke(&c1, "ctx1"));
                // `SHADOW_PROBE_HOLD_S=n`: stay n seconds with the region registered (to read the Windows GPU memory counters meanwhile)
                if let Some(n) = std::env::var("SHADOW_PROBE_HOLD_S").ok().and_then(|v| v.parse::<u64>().ok()) {
                    eprintln!("PROBE   holding {n} s");
                    std::thread::sleep(std::time::Duration::from_secs(n));
                }
                drop(region);
            }
        }
    }
}

/// STATUS 166: does the limit come from the second card's context existing while memory is registered?  Pins `SHADOW_PROBE_GIB` (one size) on context 0 BEFORE the second context
/// exists, then creates context 1, exercises both, and registers more memory (as the server does later: staging, stage-pipe buffers) on each.
#[test]
#[ignore]
fn pin_first_then_create_the_second_context() {
    let g: usize = std::env::var("SHADOW_PROBE_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    let Ok(c0) = Context::new(0) else {
        eprintln!("SKIPPED: needs a CUDA device");
        return;
    };
    let t0 = std::time::Instant::now();
    let region = match PinnedRegion::new_chunked(&c0, g * GIB, GIB) {
        Ok(r) => r,
        Err(e) => return eprintln!("PROBE pin {g} GiB on ctx0 alone: FAILED: {e}"),
    };
    eprintln!("PROBE pin {g} GiB on ctx0 alone: ok in {:.1} s", t0.elapsed().as_secs_f64());
    let c1 = match Context::new(1) {
        Ok(c) => c,
        Err(e) => return eprintln!("PROBE then creating ctx1: FAILED: {e}"),
    };
    eprintln!("PROBE then creating ctx1: ok");
    eprintln!("PROBE   {}", poke(&c0, "ctx0"));
    eprintln!("PROBE   {}", poke(&c1, "ctx1"));
    for (label, ctx) in [("ctx0", &c0), ("ctx1", &c1)] {
        for i in 0..6 {
            match PinnedRegion::new(ctx, 512 << 20) {
                Ok(r) => {
                    eprintln!("PROBE   extra 512 MiB #{i} on {label}: ok");
                    std::mem::forget(r);
                }
                Err(e) => {
                    eprintln!("PROBE   extra 512 MiB #{i} on {label}: FAILED: {e}");
                    break;
                }
            }
        }
    }
    drop(region);
}

/// STATUS 166: the limit for memory the driver allocates itself (`cuMemHostAlloc`, flags `SHADOW_PROBE_FLAGS`: 0 none, 1 portable, 2 devicemap, 3 both) with two contexts: how many GiB in 1 GiB
/// pieces before it fails, and whether the contexts still work then.
#[test]
#[ignore]
fn how_much_cumemhostalloc_gives_with_two_contexts() {
    let (Ok(c0), Ok(c1)) = (Context::new(0), Context::new(1)) else {
        eprintln!("SKIPPED: needs two CUDA devices");
        return;
    };
    let flags: u32 = std::env::var("SHADOW_PROBE_FLAGS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let cap: usize = std::env::var("SHADOW_PROBE_GIB").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    c0.bind().unwrap();
    let mut held: Vec<*mut std::ffi::c_void> = vec![];
    let mut failed = None;
    // `SHADOW_PROBE_PIECE=n`: pieces of n GiB (default 1): one 34 GiB block is what a contiguous RAM tier needs
    let piece: usize = std::env::var("SHADOW_PROBE_PIECE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    for i in 0..cap.div_ceil(piece) {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: an out-pointer for a host allocation of `piece` GiB.
        let r = unsafe { (c0.api.cuMemHostAlloc)(&mut p, piece * GIB, flags) };
        if r != 0 {
            failed = Some((i, r));
            break;
        }
        held.push(p);
    }
    match failed {
        Some((i, r)) => eprintln!("PROBE cuMemHostAlloc flags {flags}, pieces of {piece} GiB: {} GiB, then code {r}", i * piece),
        None => eprintln!("PROBE cuMemHostAlloc flags {flags}, pieces of {piece} GiB: all {cap} GiB"),
    }
    eprintln!("PROBE   {}", poke(&c0, "ctx0"));
    eprintln!("PROBE   {}", poke(&c1, "ctx1"));
    if let Some(n) = std::env::var("SHADOW_PROBE_HOLD_S").ok().and_then(|v| v.parse::<u64>().ok()) {
        eprintln!("PROBE   holding {n} s");
        std::thread::sleep(std::time::Duration::from_secs(n));
    }
    for p in held {
        // SAFETY: allocated above with cuMemHostAlloc, freed once.
        unsafe { (c0.api.cuMemFreeHost)(p) };
    }
}
