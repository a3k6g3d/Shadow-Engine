//! The kernel-signature drift test: the `.cu` files on disk, parsed as text, against the Rust registry.  Both
//! directions - every entry point in a file is registered with the same parameter types, every registered entry point
//! exists in its file, and every `.cu` file under `kernels/` is registered at all.  No GPU needed.

use shadow_cuda::kernels::{parse_entries, FILES};
use std::path::Path;

fn kernels_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("kernels")
}

#[test]
fn every_kernel_file_is_registered() {
    let mut on_disk: Vec<String> = std::fs::read_dir(kernels_dir())
        .unwrap()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(".cu"))
        .collect();
    on_disk.sort();
    let mut registered: Vec<String> = FILES.iter().map(|f| f.file.to_string()).collect();
    registered.sort();
    assert_eq!(on_disk, registered, "kernels/*.cu and kernels::FILES must list the same files");
}

#[test]
fn signatures_match_both_ways() {
    for f in FILES {
        let disk = std::fs::read_to_string(kernels_dir().join(f.file)).unwrap();
        assert_eq!(disk.replace("\r\n", "\n"), f.source.replace("\r\n", "\n"),
                   "{}: the binary carries a different source than the file (rebuild)", f.file);
        let parsed = parse_entries(&disk);
        for (name, params) in &parsed {
            let entry = f.entries.iter().find(|e| e.name == name)
                .unwrap_or_else(|| panic!("{}: entry point {name} is not in the registry", f.file));
            let params: Vec<_> = params.iter().enumerate()
                .map(|(i, p)| p.unwrap_or_else(|| panic!("{}: {name} parameter {i} has a type the parser does not know", f.file)))
                .collect();
            assert_eq!(params, entry.params, "{}: {name}'s parameters differ from the registry", f.file);
        }
        for e in f.entries {
            assert!(parsed.iter().any(|(n, _)| n == e.name), "{}: registered {} is not in the file", f.file, e.name);
        }
    }
}
