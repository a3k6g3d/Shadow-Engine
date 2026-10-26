// crates/shadow-abi/build.rs - tell the tests where the built facade is, if there is one.
//
// Milestone 2 is the toolchain question (docs/RUST_HOST.md): can MSVC, cudart and cargo coexist in one build?
// The answer is only real if the Rust test calls through a DLL MSVC actually produced, so this script finds that
// DLL and hands its directory to the test.
//
// It is a finder, not a builder: it never runs cmake.  A build script that rebuilt the C++ tree would make
// `cargo test` depend on the whole engine build, and on a machine where the facade has not been built the test
// must say so and pass, not fail.  STRATA_FACADE_DIR says where to look; without it, the repo's own build dirs.
//
// It deliberately does NOT emit link args.  Linking through the import library works, but then Windows resolves
// the DLL itself at process start - from the exe's directory or PATH - and cargo prunes anything it did not put
// in deps/ itself, so the test dies with STATUS_DLL_NOT_FOUND for a reason that has nothing to do with the seam.
// The test loads the DLL by absolute path instead.  The extern "system" declarations are identical either way.

use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-env-changed=STRATA_FACADE_DIR");
    println!("cargo::rerun-if-changed=../../include/shadow/c/abi.h");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(facade_with_cudart)");

    let Some(dir) = find_facade() else {
        eprintln!("shadow-abi: no built facade ($STRATA_FACADE_DIR unset, nothing in the repo's build-*/ dirs); the seam tests will report themselves skipped");
        return;
    };
    if !dir.join("strata_facade.dll").is_file() {
        eprintln!("shadow-abi: {} has no strata_facade.dll (cmake --build {} --target strata_facade); the seam tests will report themselves skipped", dir.display(), dir.display());
        return;
    }

    println!("cargo:rustc-env=STRATA_FACADE_DLL={}\\strata_facade.dll", dir.display());

    // Read the CUDA-ness off the cache that produced this build, rather than off the directory's name - the
    // test asserts a different answer per arm, so it has to know which arm it is looking at.
    let cache = dir.join("CMakeCache.txt");
    let cuda = std::fs::read_to_string(&cache)
        .map(|s| s.contains("STRATA_ENABLE_CUDA:BOOL=ON"))
        .unwrap_or(false);
    if cuda {
        println!("cargo::rustc-cfg=facade_with_cudart");
    }
}

/// Where the built facade lives, or None if it has not been built.  An explicit $STRATA_FACADE_DIR wins; the
/// repo's own build directories are the fallback.
fn find_facade() -> Option<PathBuf> {
    let mut repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    repo.pop();
    repo.pop(); // crates/shadow-abi -> crates -> repo root

    if let Ok(explicit) = std::env::var("STRATA_FACADE_DIR") {
        let p = PathBuf::from(explicit);
        if p.is_dir() {
            return Some(p);
        }
    }
    ["build-cpu-tests", "build-cpu-release", "build-cuda-tests"]
        .iter()
        .map(|d| repo.join(d))
        .find(|d| d.is_dir())
}
