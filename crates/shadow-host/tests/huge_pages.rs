// The large-page path, tested through the explicit-parameter entry point rather than the env knob, because the
// knob is process-global and tests run in parallel in one process.
//
// What is being tested is the honesty, not the pages: asking for large pages must give either large pages or
// exactly the old allocation, and the buffer must be able to say which.  On a machine without the
// lock-pages-in-memory privilege the second branch is the one that runs, so these pass either way - and the
// first test says out loud which branch it took, so the suite cannot quietly stop covering the other one.

#![cfg(windows)]

use shadow_host::Buffer;

fn fill_and_check(buf: &mut Buffer, seed: u8) {
    let n = buf.len();
    for (i, b) in buf.as_mut_slice().iter_mut().enumerate() {
        *b = seed.wrapping_add(i as u8);
    }
    for (i, b) in buf.as_slice().iter().enumerate() {
        assert_eq!(*b, seed.wrapping_add(i as u8), "byte {i} of a {}-byte buffer", buf.len());
    }
}

#[test]
fn the_default_path_is_unchanged() {
    let mut buf = Buffer::aligned(1_384_448).expect("the plain page-aligned path must keep working");
    assert!(!buf.is_huge(), "the default path must not quietly become the large-page one");
    assert!(buf.is_aligned(), "direct access needs the buffer address on a sector boundary");
    fill_and_check(&mut buf, 0xA0);
}

#[test]
fn asking_for_large_pages_never_lies_about_what_you_got() {
    let mut buf = Buffer::alloc(1_384_448, true).expect("a large-page request must still yield memory");

    // The contract the rest of the host layer relies on, whichever kind of memory this turned out to be.
    assert_eq!(buf.len(), 1_384_448, "len must stay what the caller asked for, not the rounded-up size");
    assert!(buf.is_aligned(), "direct access needs the buffer address on a sector boundary");

    if buf.is_huge() {
        let gran = Buffer::large_page_minimum();
        assert!(gran > 0, "a large-page buffer cannot exist on a system with no large-page support");
        assert_eq!(buf.base_address() % gran, 0, "a large-page allocation sits on a large-page boundary");
        println!("large pages: granted, {} KiB each", gran / 1024);
    } else {
        println!("large pages: refused (no lock-pages-in-memory privilege here), fell back to ordinary pages");
    }
    fill_and_check(&mut buf, 0x5A);
}

#[test]
fn a_request_that_is_not_a_whole_large_page_still_works() {
    // One sector: the smallest thing direct access can ask for, and not a multiple of a 2 MiB page. The
    // allocation is rounded up for the kernel; the caller must still see exactly the bytes it asked for.
    let mut buf = Buffer::alloc(4096, true).expect("a one-sector large-page request must still yield memory");
    assert_eq!(buf.len(), 4096);
    assert!(buf.is_aligned());
    fill_and_check(&mut buf, 0x11);
}

#[test]
fn a_borrowed_buffer_is_never_reported_as_large_pages() {
    // The C++ side's memory is its own; this side must not claim credit for the page size it did not choose.
    let mut backing = [0u8; 8192];
    let base = std::ptr::addr_of_mut!(backing).cast::<u8>();
    // SAFETY: `backing` outlives the buffer, and nothing else touches it while the buffer exists.
    let buf = unsafe { Buffer::borrowed(base, 4096) };
    assert!(!buf.is_huge());
    assert!(!buf.is_owned());
}
