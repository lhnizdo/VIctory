//! Exports the D3D12 Agility SDK values (see `D3D12SDKVersion` in main.rs)
//! from victory.exe: Rust does not export statics from an executable, and
//! Windows only reads them from the exe's export table.
//!
//! shortcut: MSVC targets only. GNU `ld` takes no `/EXPORT:` (a `.def`
//! file would be its way); there the build warns, and Windows keeps its own
//! D3D12 runtime.

fn main() {
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if msvc {
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    } else if windows {
        println!(
            "cargo:warning=the D3D12 Agility SDK is not exported on this target (MSVC only): \
             Windows' own D3D12 runtime is used, which on Windows 10 refuses the root signatures \
             translated shaders need"
        );
    }
}
