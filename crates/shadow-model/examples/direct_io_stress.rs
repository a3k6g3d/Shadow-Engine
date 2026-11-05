//! Stress test for the unbuffered direct-I/O read pool (`SHADOW_DIRECT_IO=1`): batches of ~1,000 odd-sized, unaligned pieces of a model file, as a long prompt's
//! layer-ahead read-ahead submits them, compared byte for byte with ordinary reads.  `shadow.exe --direct-io` died with heap corruption (0xC0000374) on a long prompt.
//!
//!   cargo run --release -p shadow-model --example direct_io_stress -- MODEL.gguf [batches] [pieces] [piece_bytes]
use shadow_model::readpool::{DirectIoPool, ReadPool};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let batches: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(8);
    let pieces: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(540);
    let piece: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(850_000);
    let file_len = std::fs::metadata(&path).unwrap().len();
    let paths = vec![path.clone()];
    let direct = DirectIoPool::new(&paths, 64).expect("direct pool");
    let _buffered = ReadPool::new(&paths, 2).expect("pool");
    let mut f = std::fs::File::open(&path).unwrap();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for b in 0..batches {
        let mut bufs: Vec<Vec<u8>> = (0..pieces).map(|_| vec![0u8; piece - (next() % 4096) as usize]).collect();
        let offs: Vec<u64> = bufs.iter().map(|v| next() % (file_len - v.len() as u64)).collect();
        {
            let mut jobs: Vec<(usize, u64, &mut [u8])> = bufs.iter_mut().zip(&offs).map(|(v, &o)| (0usize, o, v.as_mut_slice())).collect();
            // SAFETY: the buffers outlive the wait below.
            unsafe { direct.submit(&mut jobs).expect("submit") };
            direct.wait_async().expect("wait");
        }
        let mut bad = 0usize;
        let mut want = vec![];
        for (v, &o) in bufs.iter().zip(&offs) {
            want.resize(v.len(), 0);
            f.seek(SeekFrom::Start(o)).unwrap();
            f.read_exact(&mut want).unwrap();
            bad += (want != *v) as usize;
        }
        println!("batch {b}: {pieces} pieces of ~{piece} bytes, {bad} differ");
    }
    println!("done");
}
