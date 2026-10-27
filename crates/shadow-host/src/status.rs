//! The host's status codes and its completion ring.  They lived in `shadow-abi`, the contract half of the retired
//! Rust/C++ seam; the seam is gone (docs/SHADOW_PLAN.md), these two are what the host itself still uses.

/// What a host operation answers.  `#[non_exhaustive]` so a new code does not silently break a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Status {
    Ok = 0,
    WouldBlock = 1,
    BufferTooShort = 2,
    BadSlot = 3,
    DeviceUnavailable = 4,
}

/// A completion queue the host owns, with a capacity fixed at construction.
///
/// The rule that matters: pushing into a full ring returns `WouldBlock` and does NOT consume or overwrite the
/// entry.  A ring that silently drops a completion is the class of bug this whole rewrite exists to kill (see
/// BENCH_NOISE.md for the pattern).
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

    /// Ok(()) on success, Err((WouldBlock, item)) when full - and the item comes *back*, because `push` takes it
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
        // The refusal must be loud, and the item must still belong to the caller - handed back, not eaten.
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
        r.push(Owns(7, &DROPS)).expect("a ring of one has room for one");
        let refused = r.push(Owns(9, &DROPS)).err();
        assert_eq!(refused.as_ref().map(|(s, _)| *s), Some(Status::WouldBlock));
        assert_eq!(DROPS.load(Ordering::Relaxed), 0, "the refused item was dropped by the ring");
        let popped = r.pop().expect("the ring had an item in it");
        assert_eq!((popped.0, DROPS.load(Ordering::Relaxed)), (7, 0), "the ring dropped an item it had accepted");
        let (_, owned_back) = refused.unwrap();
        assert_eq!(owned_back.0, 9, "the item came back as a different item");
        drop(popped);
        assert_eq!(DROPS.load(Ordering::Relaxed), 1);
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
}
