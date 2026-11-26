//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, `external` host
//! libraries recording into title lists. Planned:
//! `present` (swap chain).

#[cfg(windows)]
mod bindless;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
pub mod frame;
mod hold;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
pub mod list;
#[cfg(windows)]
mod memory;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod placement;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
mod switch;
#[cfg(windows)]
mod tdr;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod twin;
#[cfg(windows)]
mod watch;
#[cfg(windows)]
mod titlewrite;

#[cfg(all(windows, feature = "diagnostics"))]
pub use copystat::CountingAlloc;
#[cfg(windows)]
pub use host::{on_failure, set_caption_extra};
#[cfg(windows)]
pub use watch::{
    Writing as KernelWrite, title_memory_aliased, title_memory_remapped,
    writing as kernel_will_write,
};

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
