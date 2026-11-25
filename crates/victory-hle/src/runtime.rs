//! Process-wide state the implementations need.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static MAIN_IMAGE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Records where the title's main image is mapped. The title expects
/// `GetModuleHandle(NULL)` and friends to return this, not our own exe.
pub fn set_main_image(base: usize) {
    MAIN_IMAGE_BASE.store(base, Ordering::Release);
    if std::env::var_os("VICTORY_POOL_NO_ASSERT").is_some_and(|v| v == "1") {
        bypass_pool_asserts(base);
    }
    if let Some(factor) = std::env::var("VICTORY_POOL_SCALE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&f| f > 1)
    {
        scale_pools(base, factor);
    }
    start_watch(base);
}

/// The config lookup pools are sized by (`(table, count, name hash,
/// default) -> value`) and the pool initialiser (`(pool, slots, slot
/// bytes)`), as RVAs of image `b99f876b74d4a77f` (title 1.0.2.1).
const POOL_SIZE_LOOKUP: usize = 0x50C_EF40;
const POOL_INIT: usize = 0x16C_BE00;
/// The `building` pool's creation, checked before patching anything: its
/// first bytes, and its call of the lookup at +0x20.
const BUILDING_POOL_AT: usize = 0x5AD_8E40;
const BUILDING_POOL_BYTES: [u8; 9] = [0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, 0x48, 0x8B, 0x05];

/// The pool slot limit: slot indices are 17 bits, `0x1FFFE` the empty mark.
const POOL_MOST: u32 = 0x1_FFF0;

static POOL_BASE: AtomicUsize = AtomicUsize::new(0);
static POOL_FACTOR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
static POOLS_SCALED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The pools [`scale_pools`] makes larger, by config name hash: `building`
/// (joaat), and `0x0BA88904`, the other pool that runs dry (192-byte
/// slots; the title's crash report names `INSTANCE_LIST`), and `0xC12C9884`,
/// the 16-byte list nodes those are linked through, next to fill. Making every
/// pool larger runs the title out of its memory arena at start.
const POOLS_TO_SCALE: [u32; 3] = [0x5411_E15C, 0x0BA8_8904, 0xC12C_9884];
static POOL_HASHES: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

/// What the title's pool creations call instead of the size lookup
/// ([`scale_pools`]): the title's own lookup, times the factor for the
/// pools chosen.
extern "win64" fn scaled_pool_size(table: u64, count: u32, hash: u32, default: u32) -> u32 {
    let base = POOL_BASE.load(Ordering::Relaxed);
    // SAFETY: the title's own lookup at its RVA in the mapped image (checked
    // in `scale_pools`), called with the arguments it was called with.
    let lookup: extern "win64" fn(u64, u32, u32, u32) -> u32 =
        unsafe { std::mem::transmute(base + POOL_SIZE_LOOKUP) };
    let size = lookup(table, count, hash, default);
    if !POOL_HASHES.get().is_some_and(|h| h.contains(&hash)) {
        return size;
    }
    POOLS_SCALED.fetch_add(1, Ordering::Relaxed);
    let factor = POOL_FACTOR.load(Ordering::Relaxed);
    if size >= POOL_MOST { size } else { size.saturating_mul(factor).min(POOL_MOST) }
}

/// Diagnostics, `VICTORY_POOL_SCALE=<factor>`: the title pools in
/// [`POOLS_TO_SCALE`] (or `VICTORY_POOL_SCALE_HASHES=<hex>,...`) are made
/// `factor` times larger (at most [`POOL_MOST`] slots). The title's
/// `building` and `INSTANCE_LIST` pools fill about ten minutes in while its
/// instance culling does not run on hosts of 32-lane waves (upstream issue
/// #698); larger pools take longer to fill. Each call of the size lookup
/// followed within 0x60 bytes by a call of the pool initialiser is pointed
/// at [`scaled_pool_size`] through a jump next to the image (a call reaches
/// 2 GiB). Only for the image these RVAs were read from: anything else is
/// left alone.
/// shortcut: a stopgap; the leak goes on, with more room (more memory).
fn scale_pools(base: usize, factor: u32) {
    // SAFETY: the main image is mapped at `base` with its PE headers; its
    // sections are read, and the call sites' rel32 written with the page
    // made writable for the write. The jump is our own allocation.
    unsafe {
        let at = std::slice::from_raw_parts((base + BUILDING_POOL_AT) as *const u8, 0x30);
        let call = |o: usize| {
            (at[o] == 0xE8).then(|| {
                let rel = i32::from_le_bytes([at[o + 1], at[o + 2], at[o + 3], at[o + 4]]);
                (base + BUILDING_POOL_AT + o + 5).wrapping_add(rel as isize as usize)
            })
        };
        if at[..9] != BUILDING_POOL_BYTES || call(0x20) != Some(base + POOL_SIZE_LOOKUP) {
            tracing::warn!("VICTORY_POOL_SCALE: not the image it knows; pools left alone");
            return;
        }
        let read32 = |at: usize| (at as *const u32).read_unaligned();
        let read16 = |at: usize| (at as *const u16).read_unaligned();
        let nt = base + read32(base + 0x3C) as usize;
        let image_size = read32(nt + 24 + 56) as usize;
        let sections = read16(nt + 6) as usize;
        let table = nt + 24 + read16(nt + 20) as usize;
        let mut sites = Vec::new();
        for k in 0..sections {
            let h = table + 40 * k;
            let (size, rva, flags) = (read32(h + 8) as usize, read32(h + 12) as usize, read32(h + 36));
            if flags & 0x2000_0000 == 0 {
                continue;
            }
            let code = std::slice::from_raw_parts((base + rva) as *const u8, size);
            let target = |i: usize| {
                let rel = i32::from_le_bytes([code[i + 1], code[i + 2], code[i + 3], code[i + 4]]);
                (base + rva + i + 5).wrapping_add(rel as isize as usize)
            };
            for i in 0..size.saturating_sub(0x70) {
                if code[i] == 0xE8
                    && target(i) == base + POOL_SIZE_LOOKUP
                    && (i + 5..i + 0x60).any(|j| code[j] == 0xE8 && target(j) == base + POOL_INIT)
                {
                    sites.push(base + rva + i);
                }
            }
        }
        // A jump within reach of every call site: just past the image.
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAlloc,
            VirtualProtect,
        };
        let mut jump = std::ptr::null_mut();
        for k in 1..64usize {
            let want = (base + image_size + k * 0x1_0000) & !0xFFFF;
            jump = VirtualAlloc(want as *const _, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if !jump.is_null() {
                break;
            }
        }
        if jump.is_null() {
            tracing::warn!("VICTORY_POOL_SCALE: no memory near the image for the jump; pools left alone");
            return;
        }
        // jmp [rip+0]; dq scaled_pool_size
        let stub: [u8; 6] = [0xFF, 0x25, 0, 0, 0, 0];
        std::ptr::copy_nonoverlapping(stub.as_ptr(), jump.cast::<u8>(), 6);
        (jump.cast::<u8>().add(6) as *mut u64).write_unaligned(scaled_pool_size as *const () as usize as u64);
        let mut old = 0;
        VirtualProtect(jump, 0x1000, PAGE_EXECUTE_READ, &mut old);
        POOL_BASE.store(base, Ordering::Relaxed);
        POOL_FACTOR.store(factor, Ordering::Relaxed);
        let hashes = std::env::var("VICTORY_POOL_SCALE_HASHES")
            .ok()
            .map(|v| v.split(',').filter_map(|h| u32::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect())
            .unwrap_or_else(|| POOLS_TO_SCALE.to_vec());
        let _ = POOL_HASHES.set(hashes);
        let mut patched = 0;
        for &site in &sites {
            let rel = (jump as isize - (site as isize + 5)) as i32;
            if VirtualProtect((site + 1) as *const _, 4, PAGE_EXECUTE_READWRITE, &mut old) != 0 {
                ((site + 1) as *mut i32).write_unaligned(rel);
                VirtualProtect((site + 1) as *const _, 4, old, &mut old);
                patched += 1;
            }
        }
        use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
        tracing::warn!(factor, sites = sites.len(), patched, hashes = ?POOL_HASHES.get(), "VICTORY_POOL_SCALE: pool sizes hooked");
    }
}

/// Diagnostics, `VICTORY_POOL_NO_ASSERT=1`: the title's lock-free pools
/// stop the title with `int 3` when their free list is empty (the
/// `building` and `INSTANCE_LIST` pools fill about ten minutes in, while
/// its instance culling does not run on hosts of 32-lane waves). Each such
/// site, `cmp r32, 0x1FFFE` (the empty marker) then `je` to an `int 3`
/// followed by `xor reg, reg` (the allocator's own null return), gets that
/// `int 3` made a `nop`: the allocation returns null instead.
/// shortcut: a caller that does not expect null may fault later; a stopgap
/// for longer runs, not a fix (issue #698 upstream).
fn bypass_pool_asserts(base: usize) {
    // SAFETY: the main image is mapped at `base` with its PE headers; only
    // its own sections are read, and single bytes of code written with the
    // page made writable for the write.
    unsafe {
        let read32 = |at: usize| (at as *const u32).read_unaligned();
        let read16 = |at: usize| (at as *const u16).read_unaligned();
        let nt = base + read32(base + 0x3C) as usize;
        let sections = read16(nt + 6) as usize;
        let table = nt + 24 + read16(nt + 20) as usize;
        let mut sites = Vec::new();
        for k in 0..sections {
            let h = table + 40 * k;
            let (size, rva, flags) = (read32(h + 8) as usize, read32(h + 12) as usize, read32(h + 36));
            // IMAGE_SCN_MEM_EXECUTE
            if flags & 0x2000_0000 == 0 {
                continue;
            }
            let code = std::slice::from_raw_parts((base + rva) as *const u8, size);
            sites.extend(pool_assert_sites(code).into_iter().map(|i| base + rva + i));
        }
        use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
        let mut patched = 0;
        for &at in &sites {
            let mut old = 0;
            if VirtualProtect(at as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old) != 0 {
                (at as *mut u8).write(0x90);
                VirtualProtect(at as *const _, 1, old, &mut old);
                patched += 1;
            }
        }
        use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
        tracing::warn!(sites = sites.len(), patched, "VICTORY_POOL_NO_ASSERT: pool-empty breakpoints made nops");
    }
}

/// Offsets in `code` of `int 3`s a pool allocator jumps to on an empty free
/// list ([`bypass_pool_asserts`]).
fn pool_assert_sites(code: &[u8]) -> Vec<usize> {
    const EMPTY: [u8; 4] = 0x1FFFE_u32.to_le_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 16 < code.len() {
        // cmp eax, imm32 | cmp r32, imm32 | cmp r8d-r15d, imm32
        let after = if code[i] == 0x3D && code[i + 1..i + 5] == EMPTY {
            Some(i + 5)
        } else if code[i] == 0x81 && (0xF8..=0xFF).contains(&code[i + 1]) && code[i + 2..i + 6] == EMPTY {
            Some(i + 6)
        } else if code[i] == 0x41
            && code[i + 1] == 0x81
            && (0xF8..=0xFF).contains(&code[i + 2])
            && code[i + 3..i + 7] == EMPTY
        {
            Some(i + 7)
        } else {
            None
        };
        if let Some(j) = after {
            let target = match code[j] {
                0x74 => Some((j + 2) as isize + isize::from(code[j + 1] as i8)),
                0x0F if code[j + 1] == 0x84 => {
                    let rel = i32::from_le_bytes([code[j + 2], code[j + 3], code[j + 4], code[j + 5]]);
                    Some((j + 6) as isize + rel as isize)
                }
                _ => None,
            };
            if let Some(t) = target.and_then(|t| usize::try_from(t).ok())
                && t + 4 < code.len()
                && code[t] == 0xCC
                && is_xor_self(&code[t + 1..t + 4])
                && !out.contains(&t)
            {
                out.push(t);
            }
        }
        i += 1;
    }
    out
}

/// `xor r32, r32` of one register (31 /r or 33 /r, register form, with an
/// optional REX prefix setting both R and B or neither).
fn is_xor_self(b: &[u8]) -> bool {
    let (rex, op, modrm) = match b[0] {
        0x40..=0x4F => (b[0], b[1], b[2]),
        _ => (0, b[0], b[1]),
    };
    let same_ext = (rex & 4 != 0) == (rex & 1 != 0);
    (op == 0x31 || op == 0x33) && modrm >> 6 == 3 && (modrm >> 3) & 7 == modrm & 7 && same_ext
}

/// Diagnostics: `VICTORY_WATCH=<rva>:<offset>,<offset>...` (hex) logs every
/// 5 s the qwords at those offsets of the object a pointer at `rva` of the
/// main image points to, such as a title pool's counters and free-list head.
/// Reads go through `ReadProcessMemory`: an address that is not mapped
/// logs `?`.
fn start_watch(base: usize) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let Ok(spec) = std::env::var("VICTORY_WATCH") else { return };
    let hex = |s: &str| usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    let Some((rva, offsets)) = spec.split_once(':') else {
        tracing::warn!(spec, "VICTORY_WATCH: expected <rva>:<offset>,...");
        return;
    };
    let Some(rva) = hex(rva) else { return };
    let offsets: Vec<usize> = offsets.split(',').filter_map(hex).collect();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("victory watch".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let object = read_u64(base + rva);
                    let fields: Vec<String> = offsets
                        .iter()
                        .map(|&o| match object.and_then(|p| read_u64(p as usize + o)) {
                            Some(v) => format!("+{o:#x}={v:#x}"),
                            None => format!("+{o:#x}=?"),
                        })
                        .collect();
                    tracing::info!(
                        target: "victory_hle::watch",
                        "watch image+{rva:#x} -> {}: {}",
                        object.map_or("?".into(), |p| format!("{p:#x}")),
                        fields.join(" ")
                    );
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "VICTORY_WATCH: no watcher thread");
        }
    });
}

/// A qword of our own process's memory at `at`, if mapped and readable.
fn read_u64(at: usize) -> Option<u64> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut value = 0u64;
    let mut read = 0usize;
    // SAFETY: reads into a local of the size given; the process handle is
    // the current-process pseudo handle.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            at as *const core::ffi::c_void,
            (&raw mut value).cast(),
            8,
            &mut read,
        )
    };
    (ok != 0 && read == 8).then_some(value)
}

pub fn main_image() -> Option<usize> {
    match MAIN_IMAGE_BASE.load(Ordering::Acquire) {
        0 => None,
        base => Some(base),
    }
}

/// Original file of each title image we mapped, by base address.
static IMAGE_PATHS: Mutex<Vec<(usize, PathBuf)>> = Mutex::new(Vec::new());

/// Records the file a title image was mapped from: the title asks for its
/// own path (`GetModuleFileName`) to find its data, and must not get our
/// cache copy.
pub fn set_image_path(base: usize, path: &Path) {
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
    paths.push((base, path));
}

/// Forgets a title image that was unmapped again.
pub fn forget_image(base: usize) {
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
}

pub fn image_path(base: usize) -> Option<PathBuf> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(b, _)| *b == base)
        .map(|(_, p)| p.clone())
}

/// The title image whose original file is called `file` (any case).
pub fn image_named(file: &str) -> Option<usize> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(_, p)| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(file)))
        .map(|(b, _)| *b)
}

/// Maps a title DLL the title loads itself (`LoadLibrary`) and returns its
/// base; `Ok(None)` if `file` is not one of the title's DLLs.
pub type TitleDllLoader = Box<dyn Fn(&str) -> Result<Option<usize>, String> + Send + Sync>;

static TITLE_DLL_LOADER: std::sync::OnceLock<TitleDllLoader> = std::sync::OnceLock::new();

/// Set once, by whoever mapped the title (it knows the folders and the
/// resolver); later calls are ignored.
pub fn set_title_dll_loader(loader: TitleDllLoader) {
    let _ = TITLE_DLL_LOADER.set(loader);
}

pub fn load_title_dll(file: &str) -> Result<Option<usize>, String> {
    TITLE_DLL_LOADER.get().map_or(Ok(None), |load| load(file))
}

/// What the title's manifest (`MicrosoftGame.config` beside its executable)
/// says about it, and where its persistent local storage lives.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Title {
    pub id: u32,
    /// The store's product id (empty if the manifest has none).
    pub store_id: String,
    /// Persistent local storage: our folder for it and the title's quota.
    /// Game saves live beside it ([`Title::saves`]).
    pub storage: PathBuf,
    pub storage_bytes: u64,
}

impl Title {
    /// Id, store id and storage quota from the manifest (zero or empty
    /// where it has none); the storage folder is left for the caller.
    pub fn from_manifest(text: &str) -> Title {
        let elements = elements(text);
        let get = |path: &str| {
            elements
                .iter()
                .find(|(p, _)| p == path)
                .map(|(_, text)| text.as_str())
        };
        let id = get("Game/TitleId").and_then(|v| u32::from_str_radix(v, 16).ok());
        let storage_mb = get("Game/PersistentLocalStorage/SizeMB").and_then(|v| v.parse().ok());
        Title {
            id: id.unwrap_or(0),
            store_id: get("Game/StoreId").unwrap_or_default().to_owned(),
            storage: PathBuf::new(),
            storage_bytes: storage_mb.unwrap_or(0) << 20,
        }
    }

    /// Where game saves live: next to persistent local storage, under the
    /// data directory, never the cache.
    pub fn saves(&self) -> PathBuf {
        self.storage.with_file_name("saves")
    }
}

/// The trimmed text of one element: comments inside it are skipped, not
/// treated as the end of its value. A manifest generator can write
/// `<TitleId><!-- generated -->639AA8DD</TitleId>`, which must still give
/// the title's id rather than an empty field.
fn field_text(rest: &str) -> String {
    let mut out = String::new();
    let mut rest = rest;
    loop {
        let Some(open) = rest.find('<') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..open]);
        if rest[open..].starts_with("<!--") {
            let Some(end) = rest[open..].find("-->") else {
                break;
            };
            rest = &rest[open + end + 3..];
        } else {
            // A child element, a closing tag or anything else ends the text.
            break;
        }
    }
    out.trim().to_owned()
}

/// Each element of an XML text as its path from the root (`Game/TitleId`)
/// and its trimmed text up to the first child, the predefined entities
/// decoded; comments, CDATA sections, declarations and attributes are
/// skipped.
fn elements(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut path: Vec<&str> = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        rest = &rest[open..];
        let end = if rest.starts_with("<!--") {
            rest.find("-->").map(|e| e + 3)
        } else if rest.starts_with("<![CDATA[") {
            rest.find("]]>").map(|e| e + 3)
        } else {
            tag_end(rest)
        };
        let Some(end) = end else { break };
        let tag = &rest[1..end - 1];
        rest = &rest[end..];
        if tag.starts_with(['!', '?']) {
            continue;
        }
        if let Some(name) = tag.strip_prefix('/') {
            if let Some(at) = path.iter().rposition(|n| *n == name.trim()) {
                path.truncate(at);
            }
            continue;
        }
        let closed = tag.ends_with('/');
        let name = tag
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("");
        path.push(name);
        let body = field_text(rest);
        let body = [("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&apos;", "'"), ("&amp;", "&")]
            .iter()
            .fold(body, |b, (entity, c)| b.replace(entity, c));
        out.push((path.join("/"), body));
        if closed {
            path.pop();
        }
    }
    out
}

/// Where the tag that starts `text` ends, past its `>`; a `>` in a quoted
/// attribute value is part of the value (XML allows it there unescaped).
fn tag_end(text: &str) -> Option<usize> {
    let mut quote = None;
    for (at, c) in text.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(open), _) if c == open => quote = None,
            (None, '>') => return Some(at + 1),
            _ => {}
        }
    }
    None
}

static TITLE: Mutex<Option<Title>> = Mutex::new(None);

pub fn set_title(title: Title) {
    *TITLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(title);
}

/// The running title; the default (id 0, no storage) until set.
pub fn title() -> Title {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Gives the process the console's millisecond timers. Titles pace
/// themselves with short waits (Grand Theft Auto VI's audio mixer sleeps
/// about 4 ms per 5.3 ms block) and cannot tune the timer on the console;
/// a desktop process waits in 15.6 ms ticks unless it asks for finer ones,
/// and Windows 11 drops that request while the window is hidden unless the
/// process opts out. Process-wide, for the whole run.
#[cfg(windows)]
pub fn use_console_timers() {
    use windows_sys::Win32::Media::timeBeginPeriod;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetProcessInformation,
    };
    let keep = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        StateMask: 0,
    };
    // SAFETY: plain process settings; `keep` outlives the call.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&raw const keep).cast(),
            size_of_val(&keep) as u32,
        );
        timeBeginPeriod(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quoted_angle_bracket_does_not_end_its_tag() {
        let text = "<Game><ShellVisuals DefaultDisplayName=\"A > B\" Description='x>y'/>\
                    <TitleId>639AA8DD</TitleId>\
                    <PersistentLocalStorage><SizeMB>64</SizeMB></PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 64 << 20));
    }

    #[test]
    fn reads_title_id_store_id_and_storage_quota() {
        let text = "\u{feff}<?xml version=\"1.0\"?>\n<Game configVersion=\"1\">\
                    <!-- <TitleId>00000000</TitleId> -->\
                    <Identity Name=\"x\"/>\n  <TitleId>639AA8DD</TitleId>\n\
                    <![CDATA[ <TitleId>1</TitleId> ]]>                    <StoreId>9&amp;C</StoreId><Other><SizeMB>7</SizeMB></Other>\
                    <PersistentLocalStorage>\n  <SizeMB>1024</SizeMB> <!-- c -->\n\
                    </PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 1024 << 20));
        assert_eq!(t.store_id, "9&C");
        assert_eq!(Title::from_manifest("").storage_bytes, 0);
    }

    #[test]
    fn inline_comments_do_not_truncate_a_field() {
        let text = "<Game>\
            <TitleId><!-- generated -->639AA8DD</TitleId>\
            <StoreId>9&amp;<!-- keep -->C</StoreId>\
            <PersistentLocalStorage><SizeMB><!-- n -->1024</SizeMB></PersistentLocalStorage>\
            </Game>";
        let t = Title::from_manifest(text);
        assert_eq!(t.id, 0x639A_A8DD);
        assert_eq!(t.store_id, "9&C");
        assert_eq!(t.storage_bytes, 1024 << 20);
    }
}
