//! `cfg(agility_runtime)`: whether this crate's test binaries load the D3D12
//! Agility SDK runtime staged in `target/<profile>/D3D12/` (see `lib.rs`).

fn main() {
    agility_runtime();
}

/// Sets `cfg(agility_runtime)` when `target/<profile>/D3D12/D3D12Core.dll`
/// exists. Only then do the test binaries export `D3D12SDKVersion` and
/// `D3D12SDKPath`: exported without the runtime there, the loader refuses
/// to make a device (`0x887E0003`) instead of using the in-box runtime.
fn agility_runtime() {
    println!("cargo:rustc-check-cfg=cfg(agility_runtime)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    // `OUT_DIR` is `target/<profile>/build/<package>-<hash>/out`.
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("set by cargo"));
    let Some(profile) = out.ancestors().nth(3) else {
        return;
    };
    let runtime = profile.join("D3D12").join("D3D12Core.dll");
    // Watch the folder, which exists or not, rather than a missing file
    // (that would rerun this script on every build).
    println!("cargo:rerun-if-changed={}", profile.join("D3D12").display());
    if runtime.is_file() {
        println!("cargo:rustc-cfg=agility_runtime");
    }
}
