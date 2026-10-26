//! The Rust mirror of the C ABI seam.
//!
//! Hand-written on both sides, on purpose. The header IS the contract, and this crate's test suite parses that
//! file and asserts the version constant matches what is here — which is the whole reason a hand-written header
//! is worth its upkeep: the failure mode of a cross-language seam is a field added on one side and a silent
//! mismatch on the other, and a generated header cannot catch that.
//!
//! Keep in step with `STRATA_ABI_VERSION` in `include/shadow/c/abi.h`. The prefix stays `strata_` there; see
//! NOTICE.

/// Keep in step with `STRATA_ABI_VERSION` in the header. The drift test is what enforces that, so it is a test
/// and not a comment.
pub const ABI_VERSION: u32 = 1;

/// Stable across releases: both sides match on these by value, never by string.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    /// The ring is full; the caller owns the retry, the entry was NOT consumed.
    WouldBlock = 1,
    BufferTooShort = 2,
    BadSlot = 3,
    DeviceUnavailable = 4,
    /// The two sides disagree about this header.
    AbiVersion = 5,
}

/// One expert the host has decided the device may read. The host owns these bytes for exactly as long as the
/// device holds the borrow — the rule the C++ side states as a comment. Here it is a borrow, not a comment.
#[repr(C)]
pub struct ExpertRef {
    pub bytes: *const u8,
    pub len: u64,
    pub slot: i32,
    /// 0 = pinned RAM, 1 = NVMe, 2 = SSD, 3 = model file.
    pub source_tier: i32,
}
