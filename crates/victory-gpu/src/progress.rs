//! Our own breadcrumbs for the BVH builder (`VICTORY_BUILDER_PROGRESS=1`,
//! diagnostics). After each builder dispatch the list writes that
//! dispatch's number into a buffer in system memory, once the work before it
//! has finished (`WriteBufferImmediate`, marker-out). After the device is
//! removed the buffer still reads: the last builder dispatch that finished,
//! whatever the length of the list (the device's own breadcrumbs stop being
//! counted past 15,360 operations).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::HostList;

struct Mark {
    _buffer: ID3D12Resource,
    gpu: u64,
    data: usize,
}

// SAFETY: the buffer stays mapped for the process; reads and writes of its
// one dword are plain.
unsafe impl Send for Mark {}
// SAFETY: as above.
unsafe impl Sync for Mark {}

static SEQ: AtomicU32 = AtomicU32::new(0);
static RECENT: Mutex<VecDeque<(u32, String)>> = Mutex::new(VecDeque::new());

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BUILDER_PROGRESS").is_ok_and(|v| v == "1"))
}

fn mark() -> Option<&'static Mark> {
    static MARK: OnceLock<Option<Mark>> = OnceLock::new();
    MARK.get_or_init(|| {
        let (buffer, data) = crate::host::host()?.shared_buffer(256)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        // SAFETY: mapped, 256 bytes.
        unsafe { data.cast::<u32>().write_volatile(0) };
        Some(Mark { _buffer: buffer, gpu, data: data as usize })
    })
    .as_ref()
}

/// After a builder dispatch: its number goes to the buffer once it and the
/// work before it are done; `what` names it for the report.
pub(crate) fn after_dispatch(list: &HostList, what: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    let Some(m) = mark() else { return };
    let Ok(list2) = list.list.cast::<ID3D12GraphicsCommandList2>() else {
        return;
    };
    let n = SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    {
        let mut recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
        recent.push_back((n, what()));
        if recent.len() > 1 << 16 {
            recent.pop_front();
        }
    }
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu, Value: n };
    let mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    // SAFETY: an open list; the buffer is live (common state, promoted).
    unsafe { list2.WriteBufferImmediate(1, &p, Some(&mode)) };
}

/// For the device-removed report: the last builder dispatch that finished
/// and the next ones recorded.
pub(crate) fn report() -> String {
    if !on() {
        return String::new();
    }
    let Some(m) = mark() else { return String::new() };
    // SAFETY: mapped; the GPU is gone.
    let done = unsafe { (m.data as *const u32).read_volatile() };
    let recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = format!("; builder progress: {done} of {} dispatches finished", SEQ.load(Ordering::Relaxed));
    for (n, what) in recent.iter().filter(|(n, _)| *n + 2 >= done && *n <= done + 6) {
        let tag = if *n <= done { "done" } else { "not done" };
        text.push_str(&format!("; #{n} {tag}: {what}"));
    }
    text
}
