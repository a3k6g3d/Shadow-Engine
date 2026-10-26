// crates/shadow-host/build.rs - find the built facade, the same way shadow-abi's does.
//
// This is that script's finder copied, not shared, because a build script cannot be shared between crates and
// a proc-macro dependency would cost the no-network rule this workspace is built on.  The two must stay in step
// on one thing only: the name of the environment variable.  If it changes, change it in both.
//
// It is a finder, not a builder (see shadow-abi/build.rs for why), and it emits no link args: the test loads the
// DLL by absolute path.

use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-env-changed=STRATA_FACADE_DIR");
    println!("cargo::rerun-if-changed=build.rs");

    let Some(dir) = find_facade() else {
        eprintln!("shadow-host: no built facade ($STRATA_FACADE_DIR unset, nothing in the repo's build-*/ dirs); the vertical-slice test will report itself skipped");
        return;
    };
    if !dir.join("strata_facade.dll").is_file() {
        eprintln!("shadow-host: {} has no strata_facade.dll (cmake --build {} --target strata_facade); the vertical-slice test will report itself skipped", dir.display(), dir.display());
        return;
    }

    println!("cargo:rustc-env=STRATA_FACADE_DLL={}\\strata_facade.dll", dir.display());
}

/// Where the built facade lives, or None if it has not been built.  An explicit $STRATA_FACADE_DIR wins; the
/// repo's own build directories are the fallback.
fn find_facade() -> Option<PathBuf> {
    let mut repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    repo.pop();
    repo.pop(); // crates/shadow-host -> crates -> repo root

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
