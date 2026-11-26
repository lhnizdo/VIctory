//! `VICTORY_SHADER_SOURCE_HASH`: what the translator is built from (this
//! crate's sources, manifest and build script, the workspace's lock file
//! and the compiler's version), for caches of translations
//! (`victory_shader::SOURCE_HASH`).

use std::hash::{DefaultHasher, Hasher};
use std::path::{Path, PathBuf};

fn main() {
    agility_runtime();
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let mut files = vec![
        dir.join("Cargo.toml"),
        dir.join("build.rs"),
        dir.join("../../Cargo.lock"),
    ];
    sources(&dir.join("src"), &mut files);
    files.sort();
    // SipHash with fixed keys: stable for one compiler, whose version is
    // hashed too.
    let mut h = DefaultHasher::new();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = std::process::Command::new(rustc).arg("-vV").output();
    h.write(&version.map(|o| o.stdout).unwrap_or_default());
    for f in &files {
        let name = f
            .strip_prefix(&dir)
            .unwrap_or(f)
            .to_string_lossy()
            .replace('\\', "/");
        h.write(name.as_bytes());
        h.write(&std::fs::read(f).unwrap_or_default());
    }
    for watched in ["src", "Cargo.toml", "build.rs", "../../Cargo.lock"] {
        println!("cargo:rerun-if-changed={watched}");
    }
    println!(
        "cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={:016x}",
        h.finish()
    );
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Sets `cfg(agility_runtime)` when `target/<profile>/D3D12/D3D12Core.dll`
/// exists. Only then do the test binaries export `D3D12SDKVersion` and
/// `D3D12SDKPath` (see `lib.rs`): exported without the runtime there, the
/// loader refuses to make a device (`0x887E0003`) instead of using the
/// in-box runtime.
fn agility_runtime() {
    println!("cargo:rustc-check-cfg=cfg(agility_runtime)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    // `OUT_DIR` is `target/<profile>/build/<package>-<hash>/out`.
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("set by cargo"));
    let Some(profile) = out.ancestors().nth(3) else {
        return;
    };
    // Watch the folder, which exists or not, rather than a missing file
    // (that would rerun this script on every build).
    println!("cargo:rerun-if-changed={}", profile.join("D3D12").display());
    if profile.join("D3D12").join("D3D12Core.dll").is_file() {
        println!("cargo:rustc-cfg=agility_runtime");
    }
}
