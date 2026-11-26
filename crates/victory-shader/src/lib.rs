//! Xbox Series shaders to host shaders.
//!
//! An Xbox Series shader is a standard DXBC container ([`dxbc`]). Its `DXIL`
//! part ([`dxil`]) may carry Xbox additions around the LLVM bitcode: an
//! `XDAT` table before it indexing blobs after it, among them `xbop`
//! (compiler options) and the zlib-compressed GFX10 code. The bitcode may use
//! console-only intrinsics, so the GFX10 code is translated in general.

pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
pub mod range;
pub mod resource;
pub mod structure;

pub use dxbc::{Container, FourCC, Part};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs,
};

/// A hash of what this translator is built from: its sources, manifest and
/// build script, the workspace's lock file and the compiler's version. Caches
/// of translations key on it, so a changed translator never serves an old
/// translation.
pub const SOURCE_HASH: &str = env!("VICTORY_SHADER_SOURCE_HASH");

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShaderError {
    #[error("not a DXBC container")]
    NotDxbc,
    #[error("malformed shader: {0}")]
    Malformed(&'static str),
    #[error("GFX10 code at {offset:#x}: {reason}")]
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    #[error("GFX10 code at {offset:#x}: {what} is not supported")]
    Unsupported { offset: usize, what: &'static str },
}

// The D3D12 Agility SDK for this crate's GPU tests: Windows 10's in-box
// runtime has no shader model 6.6. When `D3D12Core.dll` from the NuGet
// package is staged in target\<profile>\D3D12\, build.rs sets
// `cfg(agility_runtime)` and the test binaries (in target\<profile>\deps\)
// export the two values the loader looks for, pointing it one folder up.
// Without the runtime nothing is exported and the in-box runtime is used:
// exported without it, the loader would refuse to make a device. Rust does
// not export statics from an executable, so the linker is told to in the
// object itself (only test binaries link it).

/// The Agility SDK version (the middle part of 1.620.x).
#[cfg(all(test, agility_runtime))]
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)] // the names Windows looks for
pub static D3D12SDKVersion: u32 = 620;

/// Where the Agility SDK runtime is, relative to the test binary.
#[cfg(all(test, agility_runtime))]
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)] // the names Windows looks for
pub static D3D12SDKPath: &[u8; 10] = b"..\\D3D12\\\0";

/// The linker directives exporting the two values above (MSVC's linker
/// reads them from an object's `.drectve` section).
#[cfg(all(test, agility_runtime, target_env = "msvc"))]
#[used]
#[unsafe(link_section = ".drectve")]
static D3D12_SDK_EXPORTS: [u8; 56] = *b" /EXPORT:D3D12SDKVersion,DATA /EXPORT:D3D12SDKPath,DATA ";
