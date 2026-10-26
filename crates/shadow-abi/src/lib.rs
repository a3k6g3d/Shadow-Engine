//! The Rust side of the Rust/C++ seam. See docs/RUST_HOST.md and include/shadow/c/abi.h.
//!
//! This crate is the contract, not the runtime. It holds the version constant the C header mirrors, the status
//! codes the facade returns, and the one data structure the host layer owns outright: the completion ring.
//! Everything here is std-only so it builds without network access.

pub mod header;

/// Mirror of `#define STRATA_ABI_VERSION` in include/shadow/c/abi.h. tests/abi_header.rs parses the header and
/// fails the build if the two ever disagree — that test is the reason this constant exists twice.
pub const ABI_VERSION: u32 = 1;

/// Mirror of `enum strata_status`. Discriminants are the ABI; the `#[non_exhaustive]` is what lets the C side
/// add a code without silently breaking a match on this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Status {
    Ok = 0,
    WouldBlock = 1,
    BufferTooShort = 2,
    BadSlot = 3,
    DeviceUnavailable = 4,
    AbiVersion = 5,
}

impl Status {
    /// The value the facade hands back. Unknown codes are the other side's bug, not ours — report them, never
    /// guess a variant for them.
    pub fn from_c(raw: i32) -> Option<Status> {
        match raw {
            0 => Some(Status::Ok),
            1 => Some(Status::WouldBlock),
            2 => Some(Status::BufferTooShort),
            3 => Some(Status::BadSlot),
            4 => Some(Status::DeviceUnavailable),
            5 => Some(Status::AbiVersion),
            _ => None,
        }
    }
}

/// Mirror of `struct strata_expert_ref`. The first struct to cross the seam (milestone 3), so the first place a
/// field added on one side and forgotten on the other would bite — tests/abi_header.rs checks the two agree, and
/// the field order here is the header's, not a convenient one.
///
/// `bytes` is the caller's memory for exactly as long as the call lasts: the facade does not copy it, cache it,
/// or free it. On the Rust side that is an ownership rule the type system enforces (see shadow-host's Buffer);
/// this is the plain data layout the C side sees.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExpertRef {
    pub bytes: *const u8,
    pub len: u64,
    pub slot: i32,
    pub source_tier: i32,
}

/// A completion queue the host owns. Capacity is fixed at construction because it is the thing the C++ side
/// gets as a plain int today, with a comment about not overflowing it.
///
/// The rule that matters: pushing into a full ring returns `WouldBlock` and does NOT consume or overwrite the
/// entry. A ring that silently drops a completion is the class of bug this whole rewrite exists to kill — the
/// current engine's async tier has produced three of them (see BENCH_NOISE.md for the pattern).
#[derive(Debug)]
pub struct Ring<T> {
    slots: Vec<Option<T>>,
    cap: usize,
    head: usize,
    len: usize,
}

impl<T> Ring<T> {
    pub fn new(cap: usize) -> Ring<T> {
        assert!(cap > 0, "a ring of capacity 0 can never hold anything");
        Ring { slots: (0..cap).map(|_| None).collect(), cap, head: 0, len: 0 }
    }

    pub fn capacity(&self) -> usize { self.cap }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn is_full(&self) -> bool { self.len == self.cap }

    /// Ok(()) on success, Err((WouldBlock, item)) when full — and the item comes *back*, because `push` takes it
    /// by value and a refusal that swallowed it would be a drop wearing the ring's favourite rule's clothes.
    /// It matters for the items the ring is actually built for: a completion owns an aligned buffer, and dropping
    /// one releases that memory back to the system, so "the caller still owns it" has to mean *here it is*.
    pub fn push(&mut self, item: T) -> Result<(), (Status, T)> {
        if self.is_full() {
            return Err((Status::WouldBlock, item));
        }
        let tail = (self.head + self.len) % self.cap;
        self.slots[tail] = Some(item);
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let slot = self.slots[self.head].take();
        self.head = (self.head + 1) % self.cap;
        self.len -= 1;
        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_ring_refuses_rather_than_dropping() {
        let mut r: Ring<u32> = Ring::new(2);
        assert_eq!(r.push(1), Ok(()));
        assert_eq!(r.push(2), Ok(()));
        assert!(r.is_full());
        // The refusal must be loud, and the item must still belong to the caller — handed back, not eaten.
        assert_eq!(r.push(3), Err((Status::WouldBlock, 3)));
        assert_eq!(r.len(), 2);
        assert_eq!(r.pop(), Some(1));
        assert_eq!(r.push(3), Ok(()));
        assert_eq!(r.pop(), Some(2));
        assert_eq!(r.pop(), Some(3));
        assert!(r.is_empty());
    }

    /// The rule with the part of the test that `u32` cannot check: an item that owns something.  The ring's
    /// promise is that a refusal costs nothing; if a refused completion were dropped inside `push`, its buffer
    /// would be freed and the caller would be left holding a rule that only works for types with no destructor.
    #[test]
    fn a_refused_item_arrives_back_untouched() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct Owns(usize, &'static AtomicUsize);
        impl Drop for Owns {
            fn drop(&mut self) {
                self.1.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut r: Ring<Owns> = Ring::new(1);
        let counted = Owns(7, &DROPS);
        r.push(counted).expect("a ring of one has room for one");
        let refused = Owns(9, &DROPS);
        // The refusal hands the item back by value: same number, and the counter it points at still its own.
        let refused = r.push(refused).err();
        assert_eq!(refused.as_ref().map(|(s, _)| *s), Some(Status::WouldBlock));
        assert_eq!(DROPS.load(Ordering::Relaxed), 0, "the refused item was dropped by the ring, not by the test");

        // The item the ring accepted is still in it, undropped, and so is the one it refused.
        assert_eq!(DROPS.load(Ordering::Relaxed), 0, "the ring dropped an item before the test had it back");
        let popped = r.pop().expect("the ring had an item in it");
        assert_eq!((popped.0, DROPS.load(Ordering::Relaxed)), (7, 0), "the ring dropped an item it had accepted");
        let (_, owned_back) = refused.unwrap();
        assert_eq!(owned_back.0, 9, "the item came back as a different item");

        // From here the test owns both, so its own drops are the only drops there are - exactly two, and the
        // count goes to two only after both have gone.
        drop(popped);
        assert_eq!(DROPS.load(Ordering::Relaxed), 1, "the test's drop of the popped item did not happen");
        drop(owned_back);
        assert_eq!(DROPS.load(Ordering::Relaxed), 2, "the test's own drops are the only drops there have been");
    }

    #[test]
    fn wrap_keeps_order() {
        let mut r: Ring<char> = Ring::new(3);
        for _ in 0..5 {
            r.push('a').unwrap();
            r.pop().unwrap();
        }
        r.push('b').unwrap();
        r.push('c').unwrap();
        assert_eq!(r.pop(), Some('b'));
        assert_eq!(r.pop(), Some('c'));
    }

    #[test]
    fn unknown_status_codes_are_not_guessed() {
        assert_eq!(Status::from_c(1), Some(Status::WouldBlock));
        assert_eq!(Status::from_c(6), None);
    }
}
