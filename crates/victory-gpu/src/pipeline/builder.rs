//! The title's acceleration-structure builder runs whole or not at all.
//!
//! `XboxBvhBuilder.dll` builds and refits the ray-tracing structures in
//! title memory every frame through some fifty compute programs, one
//! dispatch after another. A dispatch whose host pipeline is missing is
//! dropped on its own (`State::dispatch`), and some of the builder's are
//! missing for a long time or for good: refused (64-lane waves on a host of
//! 32-lane waves), not translated, or still being made. The stages that do
//! run then work on what the missing ones never wrote, and write the result
//! into title memory: on fxceless/victory-fork, garbage bounds that crashed
//! the title. So the builder's dispatches run only in frames where every
//! pipeline it has created has its host pipeline; otherwise none of them
//! run (the structures keep what they held).
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, base + size));
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
        let mut frames = [std::ptr::null_mut(); 24];
        // SAFETY: room for that many frames.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                1,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        frames[..usize::from(n)]
            .iter()
            .map(|&f| f as usize)
            .collect()
    }
    #[cfg(not(windows))]
    Vec::new()
}

/// The builder's pipelines (alive while the title holds them).
static PIPELINES: Mutex<Vec<Weak<Pending>>> = Mutex::new(Vec::new());

/// The builder created pipeline `p`: it counts towards [`runs`], and is
/// made at once (no frame's wait budget holds it back).
pub(crate) fn note(p: &Arc<Pending>) {
    // Local diagnostics: every builder container kept, for offline reading.
    #[cfg(feature = "diagnostics")]
    if let Ok(dir) = std::env::var("VICTORY_DUMP_BUILDER_DIR") {
        make::dump(p, &dir);
    }
    PIPELINES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::downgrade(p));
    make::begin(p);
}

/// Local diagnostics, `VICTORY_BUILDER_DROP=<hash>,...`: dispatches of the
/// builder's pipelines with one of those shaders are dropped while the rest
/// of the builder runs (the gate stays open): which program hangs.
#[cfg(feature = "diagnostics")]
pub(crate) fn dropped_by_env(shaders: &str) -> bool {
    static WANT: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let want = WANT.get_or_init(|| {
        std::env::var("VICTORY_BUILDER_DROP")
            .map(|v| v.split(',').map(|h| h.trim().to_owned()).filter(|h| !h.is_empty()).collect())
            .unwrap_or_default()
    });
    !want.is_empty() && want.iter().any(|h| shaders.contains(h.as_str()))
}

/// Local diagnostics, `VICTORY_BUILDER_UPTO=<n>`: only the builder's first
/// `n` dispatches of a frame run, the rest are dropped. A list cut short
/// finishes, so its per-command times (`VICTORY_GPU_TIMING` with
/// `VICTORY_D3D12_DRED`) show where the
/// builder's time goes before the driver resets a list that runs too long.
#[cfg(feature = "diagnostics")]
pub(crate) fn past_limit() -> bool {
    static LIMIT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    static SEEN: Mutex<(u64, u64)> = Mutex::new((u64::MAX, 0));
    let Some(limit) = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_BUILDER_UPTO").ok().and_then(|v| v.trim().parse().ok())
    }) else {
        return false;
    };
    let frame = crate::frame::current();
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.0 != frame {
        *seen = (frame, 0);
    }
    seen.1 += 1;
    seen.1 > limit
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, true));

/// Whether the builder's dispatches run in the current frame: decided once
/// per frame, so that a frame never runs part of the builder.
pub(crate) fn runs() -> bool {
    static SAID: AtomicBool = AtomicBool::new(false);
    let frame = crate::frame::current();
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        return decided.1;
    }
    let (states, refused_names): (Vec<Option<bool>>, Vec<String>) = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|p| p.strong_count() > 0);
        let live: Vec<Arc<Pending>> = pipelines.iter().filter_map(Weak::upgrade).collect();
        let states: Vec<Option<bool>> = live.iter().map(|p| make::state(p)).collect();
        let refused = live
            .iter()
            .zip(&states)
            .filter(|(_, s)| **s == Some(false))
            .map(|(p, _)| make::names(p))
            .collect();
        (states, refused)
    };
    let open = all_made(&states);
    if open != decided.1 || !SAID.swap(true, Ordering::Relaxed) {
        let (made, refused) = (
            states.iter().filter(|s| **s == Some(true)).count(),
            states.iter().filter(|s| **s == Some(false)).count(),
        );
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            refused_pipelines = ?refused_names,
            "BVH builder {}",
            if open {
                "runs"
            } else {
                "held back: not all its pipelines are made"
            }
        );
    }
    *decided = (frame, open);
    open
}

/// Whether every pipeline is made (`Some(true)`); refused ones are
/// `Some(false)`, those still being made `None`.
fn all_made(states: &[Option<bool>]) -> bool {
    states.iter().all(|s| *s == Some(true))
}

/// Tests that set the builder's image, one at a time; each starts with
/// no decision made.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
    guard
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The builder runs only when every one of its pipelines is made: one
    /// refused or still being made holds all of them back.
    #[test]
    fn the_builder_runs_only_with_every_pipeline_made() {
        assert!(all_made(&[]), "no builder: nothing to hold back");
        assert!(all_made(&[Some(true), Some(true)]));
        assert!(!all_made(&[Some(true), Some(false)]), "one refused");
        assert!(!all_made(&[Some(true), None]), "one still being made");
    }

    /// Pipelines are the builder's by their bytecode in its image, or by
    /// its code among the callers; not otherwise.
    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        // A range that holds this test's own code (a caller of the check).
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        *IMAGE.lock().unwrap() = None;
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        // Bytecode inside the image.
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        // A caller inside the image: this function's own code.
        *IMAGE.lock().unwrap() = Some((here - 0x10_0000, here + 0x10_0000));
        assert!(created_by_builder(&[]));
        *IMAGE.lock().unwrap() = None;
    }
}
