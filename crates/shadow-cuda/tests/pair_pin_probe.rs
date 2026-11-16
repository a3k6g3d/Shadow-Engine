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
                drop(region);
            }
        }
    }
}
