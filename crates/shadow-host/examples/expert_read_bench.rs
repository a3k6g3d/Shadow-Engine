//! How fast can this machine read expert-sized slices at random places in the model file?  Three ways, each on its own
//! fresh random offsets (so no way reads what another just pulled into the file cache):
//!   buffered    std::fs::File + seek_read, one at a time (what shadow-model's Model::expert does today)
//!   ring d=1    shadow-host's DirectFile (overlapped, unbuffered), one read in flight
//!   ring d=N    the same with N reads in flight
//!
//!   cargo run --release -p shadow-host --example expert_read_bench -- MODEL.gguf [reads] [depth]

use shadow_host::{DirectFile, SECTOR};
use std::time::Instant;

const SLICE: usize = 851_968; // 208 pages: an IQ4_NL down-projection slice is 921,600; gate/up are smaller

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(a.get(1).expect("MODEL path"));
    let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(400);
    let depth: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(32);
    let size = std::fs::metadata(&path).unwrap().len();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ std::process::id() as u64;
    let mut offsets = |k: usize| -> Vec<u64> {
        (0..k).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % (size - 2 * SLICE as u64)) / SECTOR * SECTOR
        }).collect()
    };
    let report = |name: &str, t: f64| {
        println!("{name:<12} {n} reads of {:.2} MB: {:.3} s, {:.2} GB/s, {:.2} ms/read", SLICE as f64 / 1e6, t,
                 (n * SLICE) as f64 / t / 1e9, 1e3 * t / n as f64);
    };

    // buffered, one at a time
    {
        use std::os::windows::fs::FileExt;
        let f = std::fs::File::open(&path).unwrap();
        let mut buf = vec![0u8; SLICE];
        let offs = offsets(n);
        let t0 = Instant::now();
        for &o in &offs {
            let mut done = 0;
            while done < SLICE {
                done += f.seek_read(&mut buf[done..], o + done as u64).unwrap();
            }
        }
        report("buffered", t0.elapsed().as_secs_f64());
    }

    for d in [1, depth] {
        let mut ring = DirectFile::open(&path, d).unwrap_or_else(|s| panic!("open: {s:?}"));
        let mut free: Vec<_> = (0..d).map(|_| ring.alloc(SLICE).expect("alloc")).collect();
        let offs = offsets(n);
        let t0 = Instant::now();
        let (mut next, mut done) = (0usize, 0usize);
        while done < n {
            while next < n {
                let Some(b) = free.pop() else { break };
                if let Err((s, b)) = ring.submit(offs[next], b, next as u64) {
                    panic!("submit: {s:?} ({} bytes)", b.len());
                }
                next += 1;
            }
            ring.wait_with(d, u32::MAX, |c| {
                assert_eq!(c.bytes, SLICE, "short read");
                free.push(c.buffer);
                done += 1;
            });
        }
        report(&format!("ring d={d}"), t0.elapsed().as_secs_f64());
    }
}
