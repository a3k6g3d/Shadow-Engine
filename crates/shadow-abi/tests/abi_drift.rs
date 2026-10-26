//! The drift test. It parses the header — the real one, from the repo root — and asserts the version constant
//! matches the Rust mirror. Adding a field on one side without bumping both is the failure this exists to catch.

use std::path::PathBuf;

fn header() -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../include/shadow/c/abi.h");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn header_version(text: &str) -> u32 {
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("#define STRATA_ABI_VERSION") {
            return rest
                .trim()
                .parse::<u32>()
                .unwrap_or_else(|e| panic!("unparsable version {rest:?}: {e}"));
        }
    }
    panic!("no #define STRATA_ABI_VERSION in the header");
}

#[test]
fn version_matches() {
    assert_eq!(header_version(&header()), shadow_abi::ABI_VERSION);
}

#[test]
fn status_values_match() {
    let h = header();
    for (name, want) in [
        ("STRATA_OK", 0u32),
        ("STRATA_ERR_WOULD_BLOCK", 1),
        ("STRATA_ERR_BUFFER_TOO_SHORT", 2),
        ("STRATA_ERR_BAD_SLOT", 3),
        ("STRATA_ERR_DEVICE_UNAVAILABLE", 4),
        ("STRATA_ERR_ABI_VERSION", 5),
    ] {
        let line = h
            .lines()
            .find(|l| l.contains(name))
            .unwrap_or_else(|| panic!("{name} gone from the header"));
        let got: u32 = line
            .split('=')
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .trim_end_matches(',')
            .parse()
            .unwrap_or_else(|e| panic!("unparsable value in {line:?}: {e}"));
        assert_eq!(got, want, "{name} moved on one side");
    }
}

#[test]
fn expert_ref_layout_matches() {
    use std::mem::size_of;
    let h = header();
    assert!(h.contains("const unsigned char *bytes;"), "bytes field changed");
    assert!(h.contains("unsigned long long len;"), "len field changed");
    assert!(h.contains("int slot;"), "slot field changed");
    assert!(h.contains("int source_tier;"), "source_tier field changed");
    assert_eq!(size_of::<shadow_abi::ExpertRef>(), 8 + 8 + 4 + 4);
}
