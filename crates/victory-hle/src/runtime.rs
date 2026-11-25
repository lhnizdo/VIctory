//! Process-wide state the implementations need.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static MAIN_IMAGE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Records where the title's main image is mapped. The title expects
/// `GetModuleHandle(NULL)` and friends to return this, not our own exe.
pub fn set_main_image(base: usize) {
    MAIN_IMAGE_BASE.store(base, Ordering::Release);
    start_watch(base);
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
                    if let Some(pool) = object {
                        dump_pool_slot(base, pool as usize);
                    }
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "VICTORY_WATCH: no watcher thread");
        }
    });
}

/// Diagnostics, with `VICTORY_WATCH_DUMP=<qwords>`: the title pool at
/// `pool` (slots at `[pool]`, slot size at `[pool+8]`, next free slot in
/// the low 17 bits of `[pool+0x20]`) has its newest slot logged, qword by
/// qword: one pointing into the main image as `img+rva` (a vtable, a
/// global), one pointing at an object whose first qword does as
/// `obj(vt img+rva)`, so the types of what a leaking object refers to show.
fn dump_pool_slot(base: usize, pool: usize) {
    static QWORDS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let qwords = *QWORDS.get_or_init(|| {
        std::env::var("VICTORY_WATCH_DUMP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
    });
    if qwords == 0 {
        return;
    }
    let (Some(slots), Some(size), Some(head)) =
        (read_u64(pool), read_u64(pool + 8), read_u64(pool + 0x20))
    else {
        return;
    };
    let (size, next) = (size as u32 as usize, (head & 0x1FFFF) as usize);
    if size == 0 || next == 0 || next >= 0x1FFFE {
        return;
    }
    // Image span: the main image is about 0xB070000 bytes.
    let image = |v: u64| (v as usize).checked_sub(base).filter(|&r| r < 0xB07_0000);
    let at = slots as usize + (next - 1) * size;
    let words: Vec<String> = (0..qwords.min(size / 8))
        .map(|k| match read_u64(at + 8 * k) {
            None => "?".into(),
            Some(v) => match image(v) {
                Some(r) => format!("img+{r:#x}"),
                None => match read_u64(v as usize).and_then(image) {
                    Some(r) => format!("obj(vt img+{r:#x})"),
                    None => format!("{v:#x}"),
                },
            },
        })
        .collect();
    tracing::info!(
        target: "victory_hle::watch",
        "pool slot {} ({size} bytes) at {at:#x}: {}",
        next - 1,
        words.join(" ")
    );
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
