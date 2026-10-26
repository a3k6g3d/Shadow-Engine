//! Milestone 3 of docs/RUST_HOST.md: the vertical slice. One expert's bytes read by the Rust ring, off a file,
//! through its own IOCP, handed down the seam to the existing kernel, and the answer checked.
//!
//! The two checks are the two halves of the machine, and they are deliberately different in kind:
//!
//!   * **Self-consistency**: the same kernel, on the same bytes, obtained two ways - through the Rust ring, and
//!     through an ordinary buffered read - must give the same answer.  This is what catches the ring corrupting,
//!     short-changing or mis-aligning a read.
//!   * **Structure**: a *different* blob must give a *different* answer.  Self-consistency alone cannot catch a
//!     wrong file offset, because a read that always returns the wrong window is still self-consistent - the
//!     parity lesson from `expert_parity`, applied here.
//!
//! It is Windows-only (IOCP), and it needs a built facade; without one it says so and passes, like the seam
//! tests in shadow-abi.  The kernel is the existing one - no Rust kernel code anywhere - and the facade refuses
//! with DEVICE_UNAVAILABLE on a CPU it has not been given permission to run it on, which the test reports rather
//! than hides.

use std::ffi::c_void;
use std::path::Path;
use shadow_abi::{ExpertRef, Ring, Status};
use shadow_host::{Completion, DirectFile, BLOB_BYTES, BLOB_WINDOW};

// The geometry the kernel was built for (expert.hpp), mirrored here so the test states what it assumes.
const H: usize = 2560;

#[cfg(windows)]
#[allow(dead_code)]
extern "system" {
    fn LoadLibraryW(path: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
}

type FnApply = unsafe extern "system" fn(*const ExpertRef, *const f32, u64, *mut f32, u64) -> i32;
type FnCstr = unsafe extern "system" fn() -> *const i8;

struct Seam {
    apply: FnApply,
    last_error: FnCstr,
}

/// The seam, resolved - the same way shadow-abi's tests resolve it, and for the same reason (the DLL is loaded by
/// absolute path because cargo prunes the import library).  None means no built facade on this machine.
fn seam() -> Option<Seam> {
    let dll = option_env!("STRATA_FACADE_DLL")?;
    #[cfg(windows)]
    {
        let wide: Vec<u16> = dll.encode_utf16().chain(std::iter::once(0)).collect();
        let module = unsafe { LoadLibraryW(wide.as_ptr()) };
        assert!(!module.is_null(), "LoadLibraryW({dll}) failed - build.rs found the DLL, and it would not load");
        let sym = |n: &str| -> *mut c_void {
            let nz = format!("{n}\0");
            let raw = unsafe { GetProcAddress(module, nz.as_ptr() as *const i8) };
            assert!(!raw.is_null(), "{dll} has no {n} - the header promises an export the DLL does not have");
            raw
        };
        Some(Seam {
            apply: unsafe { std::mem::transmute::<*mut c_void, FnApply>(sym("strata_facade_expert_apply")) },
            last_error: unsafe { std::mem::transmute::<*mut c_void, FnCstr>(sym("strata_abi_last_error")) },
        })
    }
    #[cfg(not(windows))]
    {
        let _ = dll;
        None
    }
}

fn cstr(p: *const i8) -> String {
    let mut v = Vec::new();
    let mut i = 0isize;
    unsafe {
        while *p.offset(i) != 0 {
            v.push(*p.offset(i) as u8);
            i += 1;
        }
    }
    String::from_utf8_lossy(&v).into_owned()
}

/// Deterministic bytes for one blob: a different stream per blob, and reproducible, so the test's answer is a
/// fact about the machine and not a fact about the random seed.
fn blob_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

fn activation() -> Vec<f32> {
    let mut state = 0xDEADBEEFDEADBEEFu64;
    (0..H)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 32) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

/// Two answers are the same answer when every bit matches.  This cannot be `assert_eq!` on the floats: random
/// bytes through a quantized kernel produce plenty of NaN, and `NaN != NaN` in float comparison, so a test that
/// compared answers as floats would fail on two runs that agree exactly.  Bits are what the machine produced.
fn first_bit_difference(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(usize::MAX);
    }
    (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits())
}

/// The pack file: two blobs, each padded up to a whole number of sectors, so each is one aligned direct read.
/// A blob is 1,382,400 bytes = 337.5 sectors, which is exactly why the padding exists: direct access cannot read
/// a blob on its own, and that is the C++ reader's "aligned window, then scatter" shape, made visible.
fn write_pack(path: &Path) {
    let mut file = std::fs::File::create(path).expect("cannot create the pack file");
    for blob in 0..2u64 {
        let mut bytes = blob_bytes(blob + 1, BLOB_BYTES as usize);
        bytes.resize(BLOB_WINDOW as usize, 0); // the pad is the read's tail; the kernel never sees it
        std::io::Write::write_all(&mut file, &bytes).expect("cannot write the pack");
    }
    file.sync_all().expect("cannot flush the pack");
}

fn read_ordinary(path: &Path, blob: u64) -> Vec<u8> {
    let all = std::fs::read(path).expect("cannot read the pack back");
    let start = (blob * BLOB_WINDOW) as usize;
    all[start..start + BLOB_BYTES as usize].to_vec()
}

/// Hand one blob's bytes down the seam and return the kernel's answer. None means the facade declined, with the
/// reason printed - which is the honest outcome on a machine it has not been given permission for.
fn apply(seam: &Seam, bytes: &[u8], x: &[f32]) -> Option<Vec<f32>> {
    assert_eq!(bytes.len() as u64, BLOB_BYTES, "the test handed the seam a window, not a blob");
    let mut out = vec![0.0f32; H];
    let r = ExpertRef { bytes: bytes.as_ptr(), len: BLOB_BYTES, slot: 0, source_tier: 1 };
    // SAFETY: every pointer here is the caller's, alive for the whole call, and the lengths are the geometry the
    // header states.  That is the rule, and this is the call it describes.
    let status = unsafe { (seam.apply)(&r, x.as_ptr(), H as u64, out.as_mut_ptr(), H as u64) };
    match Status::from_c(status) {
        Some(Status::Ok) => Some(out),
        Some(other) => {
            // SAFETY: the facade's own buffer, good until the next failure on this thread.
            let why = unsafe { cstr((seam.last_error)()) };
            println!("skipped: the facade declined ({other:?}): {why}");
            None
        }
        None => panic!("the facade returned {status}, which is not a status the header defines"),
    }
}

/// Read both blobs through the Rust ring - two reads in flight at once, which is the point of a ring - and hand
/// back the bytes it owns, keyed by the blob each one carried.
fn read_through_the_ring(path: &Path) -> [Vec<u8>; 2] {
    let mut file = DirectFile::open(path, 2).expect("cannot open the pack for direct access");

    // Both reads submitted before any waiting: a ring with one read in it is just a read, and the whole argument
    // for Rust here is a ring that keeps several going.  `submit` takes each buffer by value - the ring owns it
    // while the kernel does - and `wait_into` hands it back inside the completion below.
    for i in 0..2u64 {
        let buf = file.alloc(BLOB_WINDOW as usize).expect("cannot allocate an aligned buffer");
        file.submit(i * BLOB_WINDOW, buf, i).expect("the first direct read failed");
    }

    let mut ring: Ring<Completion> = Ring::new(2);
    let got = file.wait_into(&mut ring, 2, u32::MAX);
    assert_eq!(got, 2, "the ring reaped {} of two reads", got);

    let mut out: [Option<Vec<u8>>; 2] = [None, None];
    while let Some(c) = ring.pop() {
        assert_eq!(c.status, Status::Ok, "a read in the ring did not complete cleanly");
        assert_eq!(c.bytes as u64, BLOB_WINDOW, "the ring reported {} bytes for a {}-byte read", c.bytes, BLOB_WINDOW);
        out[c.tag as usize] = Some(c.buffer.as_slice().to_vec());
    }
    [out[0].take().unwrap(), out[1].take().unwrap()]
}

#[test]
#[cfg(windows)]
fn one_expert_off_a_file_through_the_rust_ring() {
    let Some(seam) = seam() else {
        println!("skipped: no built facade (see shadow-host/build.rs)");
        return;
    };

    let dir = std::env::temp_dir().join("strata-vertical-slice");
    std::fs::create_dir_all(&dir).expect("cannot make the test directory");
    let pack = dir.join("synthetic.pack");
    write_pack(&pack);

    let x = activation();

    // The ring's own contract first: the bytes it read are the bytes in the file.
    let [ring0, ring1] = read_through_the_ring(&pack);
    // The ring's buffer is a window, not a blob: direct access cannot read 337.5 sectors, so the last 2048 bytes
    // are the pad the pack was written with and the kernel never sees them.  The blob is what is compared.
    for (i, bytes) in [ring0.as_slice(), ring1.as_slice()].iter().enumerate() {
        assert_eq!(bytes.len() as u64, BLOB_WINDOW, "blob {i}: the ring handed back {} bytes, not a window", bytes.len());
        assert_eq!(&bytes[..BLOB_BYTES as usize], &read_ordinary(&pack, i as u64)[..],
                   "blob {i} came back through the ring changed");
    }

    let Some(out_ring) = apply(&seam, &ring0[..BLOB_BYTES as usize], &x) else { return };

    // Self-consistency: the same kernel, the same bytes, read the ordinary way - the same answer, bit for bit.
    let out_plain = apply(&seam, &read_ordinary(&pack, 0), &x).expect("the seam answered for the ring and not for the plain read");
    assert_eq!(first_bit_difference(&out_ring, &out_plain), None,
               "the ring's bytes and the file's bytes gave different answers");

    // Structure: a different blob, a different answer - and read through the ring too, so both halves of the
    // check travel the path this test exists to test.
    let out_other = apply(&seam, &ring1[..BLOB_BYTES as usize], &x)
        .expect("the seam answered for the ring and not for the plain read");
    assert!(first_bit_difference(&out_ring, &out_other).is_some(),
            "two different blobs gave the same answer - a wrong offset is invisible to self-consistency");

    std::fs::remove_file(&pack).ok();
}

#[test]
#[cfg(not(windows))]
fn one_expert_off_a_file_through_the_rust_ring() {
    println!("skipped: this machine has no IOCP");
}
