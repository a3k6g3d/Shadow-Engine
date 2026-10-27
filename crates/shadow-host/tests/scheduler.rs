//! The scheduler layer, run for real: reads handed over a channel, answered by the drive thread, checked against
//! an ordinary read of the same file.  See `src/scheduler.rs` and docs/SHADOW_PLAN.md, Phase 1 item 1.
//!
//! Three checks, and they are deliberately different in kind:
//!
//!   * **The handle's type**: `Scheduler` is `Send + Sync`, because that is the whole job the handle has.  The
//!     compile_fail twin in `src/scheduler.rs` says the other half - that the ring underneath is not `Send`, which
//!     is why there is a thread in front of it.
//!   * **The path**: several tasks, several reads in flight at once, each answer compared with an ordinary
//!     buffered read of the same bytes.  Self-consistency, and the only thing that catches a completion answered
//!     to the wrong task - which is what a per-drive map is for, and what a wrong offset would hide.
//!   * **The refusal**: an offset the ring will not take comes back as an error *with the buffer inside it*, the
//!     ring's own rule, kept end to end through the channel.
//!
//! Windows-only (the ring is IOCP there, and so is the allocation); the other arm says so and passes.

use shadow_host::scheduler::Scheduler;
use shadow_host::{Buffer, SECTOR};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn the_handle_crosses_threads_and_the_ring_underneath_does_not() {
    assert_send_sync::<Scheduler>();
}

/// Deterministic bytes, so the answer is a fact about the machine and not about the random seed.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 24) as u8).collect()
}

fn scratch_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("shadow-scheduler");
    std::fs::create_dir_all(&dir).expect("cannot make the test directory");
    dir
}

/// Several tasks, each with its own copy of the handle - which is what the handle is for, and what makes it
/// `Sync` - and every answer checked against the file.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(windows)]
async fn several_reads_through_one_handle_answered_by_one_thread() {
    let path = scratch_dir().join("many.dat");
    let want = pattern(SECTOR as usize * 16 * 8);
    std::fs::write(&path, &want).expect("cannot write the scratch file");

    let sched = Scheduler::open(&path, 8).expect("cannot open the drive");

    let mut tasks = Vec::new();
    for i in 0..8u64 {
        let sched = sched.clone(); // every task holds its own copy of the same handle
        let want = want.clone(); // and its own copy of the bytes it expects, since a task owns what it captures
        tasks.push(tokio::spawn(async move {
            let buffer = Buffer::alloc(SECTOR as usize, false).expect("cannot allocate an aligned buffer");
            let c = sched.submit(i * SECTOR, buffer, i).await.expect("the read failed");
            assert_eq!(c.status, shadow_host::Status::Ok, "task {i}: the read did not complete cleanly");
            assert_eq!(c.tag, i, "task {i}: the completion carries the tag of another task");
            assert_eq!(c.bytes as u64, SECTOR, "task {i}: the read came back short");
            let start = (i * SECTOR) as usize;
            assert_eq!(c.buffer.as_slice(), &want[start..start + SECTOR as usize], "task {i}: the bytes came back changed");
        }));
    }
    for t in tasks {
        tokio::time::timeout(std::time::Duration::from_secs(20), t)
            .await
            .expect("a task never finished - which means its completion was never answered")
            .expect("the task panicked");
    }

    std::fs::remove_file(&path).ok();
}

/// An offset the ring refuses must come back as a refusal, with the buffer in it - and the handle must still work
/// afterwards, because a refusal costs the ring nothing it had promised.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(windows)]
async fn the_ring_refuses_an_unaligned_offset_and_hands_the_buffer_back() {
    let path = scratch_dir().join("scratch.dat");
    let want = pattern(SECTOR as usize * 4);
    std::fs::write(&path, &want).expect("cannot write the scratch file");

    let sched = Scheduler::open(&path, 4).expect("cannot open the drive");
    assert_eq!(sched.size(), want.len() as u64, "the handle reports the wrong size for the file");

    // Offset 1 is not a sector, so the ring refuses it - and the refusal is where the buffer has to come home.
    let buffer = Buffer::alloc(SECTOR as usize, false).expect("cannot allocate an aligned buffer");
    match sched.submit(1, buffer, 7).await {
        Err((status, Some(buffer))) => {
            assert_ne!(status, shadow_host::Status::Ok, "a refusal reported success");
            // Nothing was read, so the buffer is what the caller put in it: still the caller's, still intact.
            assert_eq!(buffer.as_slice().len(), SECTOR as usize, "the refusal handed back the wrong buffer");
            assert!(buffer.as_slice().iter().all(|&b| b == 0), "the refusal came back with bytes in it");
        }
        other => panic!("an unaligned offset was not refused the way the ring refuses it: {other:?}"),
    }

    // And the same handle still works afterwards.
    let buffer = Buffer::alloc(SECTOR as usize, false).expect("cannot allocate an aligned buffer");
    let c = sched.submit(0, buffer, 9).await.expect("the handle stopped working after a refusal");
    assert_eq!(c.status, shadow_host::Status::Ok, "the read after a refusal did not complete cleanly");
    assert_eq!(c.tag, 9, "the read after a refusal answered the wrong tag");
    assert_eq!(c.bytes as u64, SECTOR, "the read after a refusal came back short");
    assert_eq!(c.buffer.as_slice(), &want[..SECTOR as usize], "the read after a refusal changed its bytes");

    std::fs::remove_file(&path).ok();
}

#[test]
#[cfg(not(windows))]
fn several_reads_through_one_handle_answered_by_one_thread() {
    println!("skipped: this machine has no IOCP, and no aligned allocation either");
}

#[test]
#[cfg(not(windows))]
fn the_ring_refuses_an_unaligned_offset_and_hands_the_buffer_back() {
    println!("skipped: this machine has no IOCP, and no aligned allocation either");
}
