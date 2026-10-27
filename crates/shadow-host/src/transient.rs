//! M6 of docs/RUST_ENGINE.md: the transient-blob lifetime rule as a type.
//!
//! The rule today is a comment (`include/strata/core/expert_source.hpp`, `transient`): a blob assembled into a
//! short-lived buffer "stays valid for the layer it was asked in and the next one or two; a consumer that keeps a
//! blob longer copies it".  Nothing checks it.  A consumer that keeps a pointer one layer too long reads whatever the
//! next expert's bytes are - silently, and only on the runs where the buffer happened to be recycled in time.
//!
//! Here it is two guarantees, deliberately different in kind, because a borrow cannot say "two more layers":
//!
//!   * **Static.**  The bytes are only ever reached through `&TransientRing` (`TransientRing::bytes`), and moving to
//!     the next layer takes `&mut TransientRing` (`advance`).  So no reference into a slot can live across an
//!     advance - the compiler refuses it.  To keep a blob across layers you keep the `Lease`, which is a value, not a
//!     borrow, and ask again.
//!   * **Checked.**  Asking again with a lease past its window - or for a slot recycled since - is refused with
//!     `Status::BadSlot`, never answered with someone else's bytes.  The slot carries a generation that changes every
//!     time it is reused, and the lease remembers the one it was given.  That refusal is the loud failure M6's gate
//!     asks for: "a consumer keeping a blob too long fails loudly rather than silently".
//!
//! A slot is reused only once its epoch is outside the window, so a lease inside its window can never be refused -
//! the check costs a valid consumer nothing but a compare.  A ring with no reusable slot refuses to stage
//! (`WouldBlock`) instead of recycling a slot that is still promised to someone.

use crate::{Buffer, SECTOR};
use crate::Status;

/// One staged blob, as its holder knows it.  Copyable on purpose: it is a claim to be checked, not ownership, so
/// holding one keeps nothing alive - which is exactly why it has to be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lease {
    pub slot: usize,
    pub generation: u64,
    pub epoch: u64,
    pub len: usize,
}

struct Slot {
    buffer: Buffer,
    generation: u64,
    /// The epoch (layer step) it was last staged in; None while it has never been used.
    epoch: Option<u64>,
}

/// A fixed set of staging buffers recycled layer by layer, with the rule above enforced.
pub struct TransientRing {
    slots: Vec<Slot>,
    epoch: u64,
    /// How many layers after the one a blob was staged in it stays valid: the comment's "the next one or two" is 2.
    keep: u64,
}

impl TransientRing {
    /// `slots` buffers of `slot_bytes` each (rounded up to the sector size, page-aligned), valid for `keep` layers
    /// after their own.  None when the memory cannot be had.
    pub fn new(slots: usize, slot_bytes: usize, keep: u64) -> Option<TransientRing> {
        if slots == 0 || slot_bytes == 0 {
            return None;
        }
        let bytes = slot_bytes.div_ceil(SECTOR as usize) * SECTOR as usize;
        let mut v = Vec::with_capacity(slots);
        for _ in 0..slots {
            v.push(Slot { buffer: Buffer::aligned(bytes)?, generation: 0, epoch: None });
        }
        Some(TransientRing { slots: v, epoch: 0, keep })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The next layer.  Takes `&mut self`, so it cannot happen while any `bytes` borrow is alive.
    pub fn advance(&mut self) {
        self.epoch += 1;
    }

    fn in_window(&self, epoch: u64) -> bool {
        epoch + self.keep >= self.epoch
    }

    /// Claim a slot for a blob of `len` bytes in this epoch and fill it with `fill`.  Only a slot whose last epoch is
    /// outside the window is reused; with none, the refusal is `WouldBlock` and nothing is recycled.
    pub fn stage(&mut self, len: usize, fill: impl FnOnce(&mut [u8])) -> Result<Lease, Status> {
        let epoch = self.epoch;
        let keep = self.keep;
        let free = self
            .slots
            .iter()
            .position(|s| s.epoch.is_none_or(|e| e + keep < epoch))
            .ok_or(Status::WouldBlock)?;
        let s = &mut self.slots[free];
        if len > s.buffer.len() {
            return Err(Status::BufferTooShort);
        }
        s.generation += 1;
        s.epoch = Some(epoch);
        fill(&mut s.buffer.as_mut_slice()[..len]);
        Ok(Lease { slot: free, generation: s.generation, epoch, len })
    }

    /// The blob's bytes, if the lease is still good: the same generation (not recycled) and inside its window.
    /// Otherwise `BadSlot` - never the bytes of whatever the slot holds now.
    pub fn bytes(&self, lease: &Lease) -> Result<&[u8], Status> {
        let s = self.slots.get(lease.slot).ok_or(Status::BadSlot)?;
        if s.generation != lease.generation || !self.in_window(lease.epoch) {
            return Err(Status::BadSlot);
        }
        Ok(&s.buffer.as_slice()[..lease.len])
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    fn ring(slots: usize) -> TransientRing {
        TransientRing::new(slots, 4096, 2).expect("aligned memory")
    }

    #[test]
    fn a_blob_is_good_for_its_layer_and_the_next_two() {
        let mut r = ring(4);
        let l = r.stage(5, |b| b.copy_from_slice(b"abcde")).unwrap();
        for _ in 0..=2 {
            assert_eq!(r.bytes(&l).unwrap(), b"abcde");
            r.advance();
        }
        // The third layer after its own: outside the window, refused - not answered.
        assert_eq!(r.bytes(&l), Err(Status::BadSlot));
    }

    /// The gate's test: a consumer that keeps a blob too long, on a ring where its slot really has been reused, gets
    /// a refusal - never the new blob's bytes, which is what the C++ pointer would silently read today.
    #[test]
    fn keeping_a_blob_too_long_fails_loudly_not_silently() {
        let mut r = ring(1);
        let old = r.stage(3, |b| b.copy_from_slice(b"old")).unwrap();
        for _ in 0..3 {
            r.advance();
        }
        let new = r.stage(3, |b| b.copy_from_slice(b"new")).unwrap();
        assert_eq!(new.slot, old.slot, "the test needs the slot to have been reused");
        assert_eq!(r.bytes(&old), Err(Status::BadSlot), "a stale lease read the next blob's bytes");
        assert_eq!(r.bytes(&new).unwrap(), b"new");
    }

    /// A slot still promised to someone is never recycled: a full ring refuses to stage instead.
    #[test]
    fn a_slot_inside_its_window_is_never_reused() {
        let mut r = ring(2);
        let a = r.stage(1, |b| b[0] = 1).unwrap();
        r.advance();
        let b = r.stage(1, |b| b[0] = 2).unwrap();
        r.advance();
        assert_eq!(r.stage(1, |b| b[0] = 3), Err(Status::WouldBlock), "a slot inside its window was recycled");
        assert_eq!((r.bytes(&a).unwrap(), r.bytes(&b).unwrap()), (&[1u8][..], &[2u8][..]));
        r.advance(); // a's window (epoch 0 + 2) is now behind us; b's is not
        let c = r.stage(1, |b| b[0] = 3).unwrap();
        assert_eq!(c.slot, a.slot);
        assert_eq!(r.bytes(&b).unwrap(), &[2u8][..]);
    }

    #[test]
    fn a_lease_for_a_slot_that_does_not_exist_is_refused() {
        let r = ring(1);
        let forged = Lease { slot: 9, generation: 1, epoch: 0, len: 1 };
        assert_eq!(r.bytes(&forged), Err(Status::BadSlot));
    }

    #[test]
    fn a_blob_larger_than_a_slot_is_refused() {
        let mut r = ring(1);
        assert_eq!(r.stage(4097, |_| ()), Err(Status::BufferTooShort));
    }
}

/// The static half, as a test that must NOT compile: holding the bytes across `advance` is a borrow error.
///
/// ```compile_fail,E0502
/// let mut r = shadow_host::transient::TransientRing::new(1, 4096, 2).unwrap();
/// let l = r.stage(1, |b| b[0] = 7).unwrap();
/// let bytes = r.bytes(&l).unwrap();
/// r.advance();               // error[E0502]: cannot borrow `r` as mutable because it is also borrowed as immutable
/// assert_eq!(bytes[0], 7);
/// ```
#[allow(dead_code)]
fn static_half_is_a_compile_fail_doctest() {}
