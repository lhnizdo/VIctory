//! The title's stores that fault on guarded blocks (`crate::watch`), looked
//! at where they fault, as avatarmanz/VIctory-7900XTX does (`titlewrite.rs`
//! there, whose decoder this is):
//!
//! - [`unchanged`]: a store that would leave the memory as it is (a flag set
//!   again, a count stored unchanged: 30% of the faulting stores in the
//!   world) is stepped over, with no block opened and nothing marked
//!   changed (`VICTORY_SKIP_SAME=1`; off unless asked for). A block takes
//!   [`MOST`] of them since it was last opened; then it is opened as
//!   before, so code filling a block with what it holds costs one opening,
//!   not a fault a store.
//! - [`count`] (`VICTORY_STORE_STATS=1`, a diagnostics build): how many
//!   faulting stores the decoder understands, and how many would leave the
//!   memory as it is.

#[allow(dead_code)] // (stores are only stepped over yet, not made)
mod decode;

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

/// Whether stores that leave the memory as it is are stepped over
/// (`VICTORY_SKIP_SAME=1`).
pub(crate) fn skip_same_on() -> bool {
    #[cfg(test)]
    if SKIP_IN_TEST.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SKIP_SAME").is_ok_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// [`skip_same_on`] for this thread's faults.
    pub(crate) static SKIP_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The most stores a block steps over since it was last opened.
pub(crate) const MOST: u32 = 32;

/// Stores stepped over, by a few bits of the block's number (blocks sharing
/// a slot share its count: they are opened a little sooner).
static SKIPPED: [AtomicU32; 1 << 13] = [const { AtomicU32::new(0) }; 1 << 13];

/// Stores stepped over since the last report.
pub(crate) static STEPPED: AtomicU64 = AtomicU64::new(0);

/// The store faulting at `at` would leave the memory as it is: its length
/// and the flags it would leave. `None` if it is not understood, changes the
/// memory, is locked, or crosses a block.
pub(crate) fn unchanged(context: &CONTEXT, at: usize) -> Option<(usize, u32)> {
    let rip = context.Rip as usize;
    let mut code = [0u8; 15];
    let left = 4096 - rip % 4096;
    let there = crate::host::read_into(rip as *const u8, &mut code) || {
        let n = left.min(15);
        n < 15 && crate::host::read_into(rip as *const u8, &mut code[..n])
    };
    let store = there.then(|| decode::store(&code, context)).flatten()?;
    let end = store.at.wrapping_add(store.size);
    const BLOCK: usize = crate::watch::PAGE;
    if !(store.at..end).contains(&at) || store.at / BLOCK != (end - 1) / BLOCK {
        return None;
    }
    // SAFETY: a guarded block's bytes, which can be read.
    match unsafe { store.same(context) } {
        Some((true, flags)) => Some((store.len, flags)),
        _ => None,
    }
}

/// Whether block `block` may step over one more store; counts it.
pub(crate) fn may_skip(block: usize) -> bool {
    let slot = &SKIPPED[block % SKIPPED.len()];
    slot.load(Relaxed) < MOST && slot.fetch_add(1, Relaxed) < MOST
}

/// Block `block` was opened: its count starts again.
pub(crate) fn opened(block: usize) {
    SKIPPED[block % SKIPPED.len()].store(0, Relaxed);
}

/// Since the last report: faulting stores looked at, understood by the
/// decoder, and of those, ones that would leave the memory as it is.
pub(crate) static STORES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether faulting stores are counted (`VICTORY_STORE_STATS=1`).
pub(crate) fn stats_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_STORE_STATS").is_ok_and(|v| v == "1"))
}

/// Counts the store that faulted at `at` (a byte of a guarded block): read
/// where it faulted, decoded, and asked whether it changes the memory.
#[cfg(feature = "diagnostics")]
pub(crate) fn count(context: &CONTEXT, at: usize) {
    STORES[0].fetch_add(1, Relaxed);
    let rip = context.Rip as usize;
    let mut code = [0u8; 15];
    // (To the end of the page if the next is not there.)
    let left = 4096 - rip % 4096;
    let there = crate::host::read_into(rip as *const u8, &mut code) || {
        let n = left.min(15);
        n < 15 && crate::host::read_into(rip as *const u8, &mut code[..n])
    };
    let Some(store) = there.then(|| decode::store(&code, context)).flatten() else {
        return;
    };
    let end = store.at.wrapping_add(store.size);
    if !(store.at..end).contains(&at) {
        return;
    }
    STORES[1].fetch_add(1, Relaxed);
    // SAFETY: a guarded block's bytes, which can be read.
    if let Some((true, _)) = unsafe { store.same(context) } {
        STORES[2].fetch_add(1, Relaxed);
    }
}

/// The counts since the last call (looked at, understood, leaving the
/// memory as it is, stepped over), and zeroes them.
pub(crate) fn take() -> [u64; 4] {
    let [a, b, c] = STORES.each_ref().map(|a| a.swap(0, Relaxed));
    [a, b, c, STEPPED.swap(0, Relaxed)]
}
