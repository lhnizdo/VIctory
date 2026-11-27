//! A wave64 group of one console wave (33 to 64 threads) on a host of
//! 32-lane waves, which the split ([`super::lower::wave64`]) cannot run per
//! lane, runs as the console runs it (`Lower::split_console`): every branch
//! decided for the console wave, both halves agreeing through the split's
//! exchange, a group sync around every LDS access. Synthetic GFX10 programs,
//! one group of 64 threads (one console wave, two host waves), lowered for a
//! 32-lane host, validated by `dxil.dll` when present and run on the
//! hardware adapter and WARP: each must complete and store what the console
//! would.
//!
//! - LDS shared without barriers: each lane writes its slot, then reads lane
//!   63 - t's. The console's wave runs in lockstep; two host waves need the
//!   syncs, or one half reads before the other wrote.
//! - A cross-lane read (`v_readfirstlane_b32`) on one side of a per-lane
//!   branch, which the split refuses per lane (not every thread arrives).

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const THREADS: u32 = 64;

/// The UAV (table entry 0, at s0) into s[40:43], v20 = thread × 4, v1 = 1.
const PROLOGUE: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
    0x3428_0082, // v_lshlrev_b32 v20, 2, v0
    0x7E02_0281, // v_mov_b32 v1, 1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

/// v1 stored at v20, then the end.
const EPILOGUE: &[u32] = &[
    0xE070_1000,
    0x800A_0114, // buffer_store_dword v1, v20, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// Each lane writes t + 64 to LDS at t, then reads LDS at 63 - t: 127 - t.
fn lds_without_barriers() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x4A02_00C0, // v_add_nc_u32 v1, 64, v0
        0xD834_0000,
        0x0000_0114, // ds_write_b32 v20, v1
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4C06_00BF, // v_sub_nc_u32 v3, 63, v0
        0x3406_0682, // v_lshlrev_b32 v3, 2, v3
        0xD8D8_0000,
        0x0100_0003, // ds_read_b32 v1, v3
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ]);
    w.extend_from_slice(EPILOGUE);
    w
}

/// Lanes 0-15 read the first active lane's t + 64 (lane 0: 64) through
/// `v_readfirstlane_b32`: 64 for them, 1 for the others.
fn readfirstlane_in_a_lane_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x4A08_00C0, // v_add_nc_u32 v4, 64, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc, 16, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0002, // s_cbranch_execz +2 (the join)
        0x7E10_0504, // v_readfirstlane_b32 s8, v4
        0x7E02_0208, // v_mov_b32 v1, s8
        0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
    ]);
    w.extend_from_slice(EPILOGUE);
    w
}

/// As the title's programs spill: s1 (the group's id, 0, unknown to the
/// translation) to lane 0 of v5, and v4 = 64, both before the branch; in
/// lanes 0-15, s9 = that lane of v5, v4 += s9 and `v_readfirstlane_b32` of
/// v4. The spill makes the program need the console's lanes (split in two
/// host waves); the read is of a value every lane computed alike, so 64 in
/// lanes 0-15 and 1 elsewhere, with no exchange between the halves.
fn readfirstlane_of_scalars_in_a_lane_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x7E08_02C0, // v_mov_b32 v4, 64
        0xD761_0005,
        0x0001_0001, // v_writelane_b32 v5, s1, 0
        0x7D88_0090, // v_cmp_gt_u32 vcc, 16, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0005, // s_cbranch_execz +5 (the join)
        0xD760_0009,
        0x0001_0105, // v_readlane_b32 s9, v5, 0
        0x4A08_0809, // v_add_nc_u32 v4, s9, v4
        0x7E10_0504, // v_readfirstlane_b32 s8, v4
        0x7E02_0208, // v_mov_b32 v1, s8
        0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
    ]);
    w.extend_from_slice(EPILOGUE);
    w
}

/// Lowered for a host of 32-lane waves.
fn lowered(words: &[u32]) -> lower::Lowered {
    // One user SGPR, 512 bytes of LDS.
    lowered_with(words, 1 << 1 | 1 << 15)
}

/// [`lowered`] with its own `rsrc2` (user SGPRs, group ids, LDS size).
fn lowered_with(words: &[u32], rsrc2: u32) -> lower::Lowered {
    lowered_sized(words, rsrc2, THREADS).expect("lower")
}

/// [`lowered_with`] for a group of `threads`, or why it does not lower.
fn lowered_sized(words: &[u32], rsrc2: u32, threads: u32) -> Result<lower::Lowered, String> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower_for(
        &lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
        },
        false,
    )
    .map_err(|e| format!("{e:?}"))
}

fn shader(lowered: lower::Lowered) -> HostShader {
    HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: Vec::new(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    }
}

/// Lowers `words` for a 32-lane host, checks the split synced its halves,
/// validates, and runs it on the hardware adapter and WARP; each must
/// complete and store `want(thread)`.
fn runs_as_the_console(name: &str, words: &[u32], want: impl Fn(u32) -> u32) {
    let lowered = lowered(words);
    assert!(lowered.group_syncs > 0, "{name}: the halves never synced");
    let shader = shader(lowered);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
    }
    let views = [View::RawUav(THREADS)];
    let want: Vec<u32> = (0..THREADS).map(want).collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (mut ran, mut failed) = (0, Vec::new());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{name}, {label}: dispatch failed (a hang?): {e}"));
        if out[0][..THREADS as usize] != want[..] {
            failed.push(format!("{name}, {label}: {:?}", &out[0][..THREADS as usize]));
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
    assert!(failed.is_empty(), "want {want:?}; {failed:#?}");
}

/// The programs decode to the instructions they are meant to be.
#[test]
fn wave64_console_programs_decode_as_written() {
    let names = |words: Vec<u32>| -> Vec<&'static str> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .expect("decode")
            .iter()
            .map(|i| i.name().unwrap_or("?"))
            .collect()
    };
    let prologue = [
        "s_mov_b32",
        "s_mov_b32",
        "s_load_dwordx4",
        "v_lshlrev_b32",
        "v_mov_b32",
        "s_waitcnt",
    ];
    let epilogue = ["buffer_store_dword", "s_endpgm"];
    let with = |body: &[&'static str]| -> Vec<&'static str> {
        prologue
            .iter()
            .chain(body)
            .chain(&epilogue)
            .copied()
            .collect()
    };
    assert_eq!(
        names(lds_without_barriers()),
        with(&[
            "v_add_nc_u32",
            "ds_write_b32",
            "s_waitcnt",
            "v_sub_nc_u32",
            "v_lshlrev_b32",
            "ds_read_b32",
            "s_waitcnt",
        ])
    );
    assert_eq!(
        names(readfirstlane_of_scalars_in_a_lane_branch()),
        with(&[
            "v_mov_b32",
            "v_writelane_b32",
            "v_cmp_gt_u32",
            "s_and_saveexec_b64",
            "s_cbranch_execz",
            "v_readlane_b32",
            "v_add_nc_u32",
            "v_readfirstlane_b32",
            "v_mov_b32",
            "s_or_b64",
        ])
    );
    assert_eq!(
        names(readfirstlane_in_a_lane_branch()),
        with(&[
            "v_add_nc_u32",
            "v_cmp_gt_u32",
            "s_and_saveexec_b64",
            "s_cbranch_execz",
            "v_readfirstlane_b32",
            "v_mov_b32",
            "s_or_b64",
        ])
    );
}

#[test]
fn lds_shared_without_barriers_by_one_console_wave_runs_in_lockstep() {
    runs_as_the_console("LDS without barriers", &lds_without_barriers(), |t| 127 - t);
}

#[test]
fn a_cross_lane_read_in_a_lane_branch_of_one_console_wave_runs_as_the_console() {
    runs_as_the_console(
        "readfirstlane in a lane branch",
        &readfirstlane_in_a_lane_branch(),
        |t| if t < 16 { 64 } else { 1 },
    );
}

/// A first-lane read of a value made of constants and scalars needs no
/// exchange: the split runs it per lane, with no group sync (not as the
/// console runs it), and every lane gets the console's value.
#[test]
fn a_first_lane_read_of_scalars_in_a_lane_branch_needs_no_exchange() {
    let words = readfirstlane_of_scalars_in_a_lane_branch();
    // One user SGPR and the group's X id (s1, 0 here, unknown to the
    // translation); no LDS: a one-wave program declaring LDS runs as the
    // console runs it whatever it reads.
    let lowered = lowered_with(&words, 1 << 1 | 1 << 7);
    assert_eq!(
        lowered.group_syncs, 0,
        "the halves synced for a value alike in every lane"
    );
    let shader = shader(lowered);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
    }
    let views = [View::RawUav(THREADS)];
    let want: Vec<u32> = (0..THREADS).map(|t| if t < 16 { 64 } else { 1 }).collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0][..THREADS as usize], want[..], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Local diagnostics, no game data in the repo: `VICTORY_ONE_CONTAINER` is
/// the path of one compute container; prints why the split refuses it per
/// lane (the console-mode retry off) and what console mode makes of it.
#[test]
#[ignore]
fn one_container_split_refusal() {
    let Some(path) = std::env::var_os("VICTORY_ONE_CONTAINER") else {
        return;
    };
    let bytes = std::fs::read(path).expect("container");
    super::set_host_wave64(false);
    for (label, no_retry) in [("split only", true), ("with console retry", false)] {
        lower::NO_CONSOLE_RETRY.store(no_retry, std::sync::atomic::Ordering::Relaxed);
        let (c0, h0) = (
            lower::CONSOLE_RUNS.load(std::sync::atomic::Ordering::Relaxed),
            lower::PER_HALF_RUNS.with(std::cell::Cell::get),
        );
        let r = super::translate_with(&bytes, super::Options::default());
        let mode = if lower::PER_HALF_RUNS.with(std::cell::Cell::get) > h0 {
            "per half"
        } else if lower::CONSOLE_RUNS.load(std::sync::atomic::Ordering::Relaxed) > c0 {
            "console mode"
        } else {
            "split"
        };
        match r {
            Ok(s) => eprintln!("{label}: ok ({mode}), {} bytes of DXIL", s.dxil.len()),
            Err(e) => eprintln!("{label}: {e}"),
        }
    }
    lower::NO_CONSOLE_RETRY.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Local census, no game data in the repo: every container in the folder
/// `VICTORY_CENSUS_DIR`, translated for a 32-lane host with and without
/// [`lower::Lower::made_of_scalars`]; counts those it moves out of the
/// console-mode retry, those it lets translate, and any it breaks.
#[test]
#[ignore]
fn census_first_lane_reads_of_scalars() {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(dir) = std::env::var_os("VICTORY_CENSUS_DIR") else {
        return;
    };
    super::set_host_wave64(false);
    let mut files: Vec<_> = std::fs::read_dir(dir).expect("dir").filter_map(Result::ok).map(|e| e.path()).collect();
    files.sort();
    let (mut programs, mut out_of_console, mut now_ok, mut broken, mut console_before, mut console_after) =
        (0, 0, 0, 0, 0, 0);
    for f in &files {
        let bytes = std::fs::read(f).expect("container");
        let run = |off: bool| {
            lower::NO_FIRST_OWN.store(off, Relaxed);
            let before = lower::CONSOLE_RUNS.load(Relaxed);
            let r = super::translate_with(&bytes, super::Options::default());
            (r.map(|s| s.dxil.len()), lower::CONSOLE_RUNS.load(Relaxed) > before)
        };
        let (old, old_console) = run(true);
        let (new, new_console) = run(false);
        if old.is_err() && new.is_err() && !old_console && !new_console {
            continue;
        }
        programs += 1;
        console_before += usize::from(old_console);
        console_after += usize::from(new_console);
        let name = f.file_stem().unwrap_or_default().to_string_lossy();
        match (&old, &new) {
            (Ok(_), Ok(_)) if old_console && !new_console => {
                out_of_console += 1;
                eprintln!("out of console mode: {name}");
            }
            (Err(_), Ok(_)) => {
                now_ok += 1;
                eprintln!("translates now: {name}");
            }
            (Ok(_), Err(e)) => {
                broken += 1;
                eprintln!("BROKEN: {name}: {e}");
            }
            _ => {}
        }
    }
    lower::NO_FIRST_OWN.store(false, Relaxed);
    eprintln!(
        "{programs} programs translated either way: {out_of_console} out of console mode,          {now_ok} translate now, {broken} broken; console mode {console_before} -> {console_after}"
    );
    assert_eq!(broken, 0);
}

/// Local census, no game data in the repo: of the compute containers in
/// `VICTORY_CENSUS_DIR`, those that run as the console runs them; how many
/// hold the console compiler's atomic optimizer (the first lane by `s_ff1`
/// or a count by `s_bcnt1`, `v_mbcnt` ranks, an atomic), and the
/// instruction and reason where the split refused each.
#[test]
#[ignore]
fn census_console_programs() {
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering::Relaxed;
    let Some(dir) = std::env::var_os("VICTORY_CENSUS_DIR") else {
        return;
    };
    super::set_host_wave64(false);
    let mut files: Vec<_> =
        std::fs::read_dir(dir).expect("dir").filter_map(Result::ok).map(|e| e.path()).collect();
    files.sort();
    let (mut translated, mut console, mut idiom, mut per_half) = (0, 0, 0, 0);
    let mut at: BTreeMap<String, usize> = BTreeMap::new();
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    for f in &files {
        let bytes = std::fs::read(f).expect("container");
        let before = lower::CONSOLE_RUNS.load(Relaxed);
        let halves = lower::PER_HALF_RUNS.with(std::cell::Cell::get);
        if super::translate_with(&bytes, super::Options::default()).is_err() {
            continue;
        }
        translated += 1;
        if lower::PER_HALF_RUNS.with(std::cell::Cell::get) > halves {
            per_half += 1;
            eprintln!("per half: {}", f.file_stem().unwrap_or_default().to_string_lossy());
        }
        if lower::CONSOLE_RUNS.load(Relaxed) == before {
            continue;
        }
        console += 1;
        let Ok(c) = crate::dxbc::Container::parse(&bytes) else { continue };
        let Some(part) = c.part(crate::dxbc::FourCC::DXIL).or(c.part(crate::dxbc::FourCC::ACCC)) else {
            continue;
        };
        let Ok(programs) = crate::native::native_programs(part.data) else { continue };
        let Some(p) = programs.into_iter().find(|p| p.stage == HwStage::Cs) else { continue };
        let Ok(insts) = crate::gfx10::decode(&p.patched_code()) else { continue };
        let names: Vec<&str> = insts.iter().map(|i| i.name().unwrap_or("?")).collect();
        let has = |n: &str| names.iter().any(|x| x.starts_with(n));
        let atomics = names.iter().any(|x| x.contains("atomic"));
        let found = (has("s_ff1_i32_b64") || has("s_bcnt1_i32_b64")) && has("v_mbcnt") && atomics;
        idiom += usize::from(found);
        lower::NO_CONSOLE_RETRY.store(true, Relaxed);
        let e = super::translate_with(&bytes, super::Options::default()).err().map(|e| e.to_string());
        lower::NO_CONSOLE_RETRY.store(false, Relaxed);
        let e = e.unwrap_or_default();
        let offset = e
            .split("code at 0x")
            .nth(1)
            .and_then(|r| usize::from_str_radix(r.split(':').next().unwrap_or(""), 16).ok());
        let inst = offset
            .and_then(|o| insts.iter().find(|i| i.offset == o))
            .and_then(|i| i.name())
            .unwrap_or("?");
        *at.entry(format!("{inst}{}", if found { " (atomic idiom)" } else { "" })).or_default() += 1;
        let reason = e.split(": ").last().unwrap_or("").chars().take(90).collect::<String>();
        *why.entry(reason).or_default() += 1;
        let name = f.file_stem().unwrap_or_default().to_string_lossy();
        eprintln!("console: {name} idiom={found} at {inst}");
    }
    eprintln!(
        "{translated} translated, {per_half} per half, {console} as the console runs them,          {idiom} of them with the atomic idiom"
    );
    let mut at: Vec<_> = at.into_iter().collect();
    at.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in at.iter().take(25) {
        eprintln!("  refused at {k}: {n}");
    }
    for (k, n) in &why {
        eprintln!("  reason {n}: {k}");
    }
}

/// The console compiler's atomic optimizer, count form, in a lane branch of
/// one console wave: lanes t < 40 (both halves) count themselves
/// (`s_bcnt1` of EXEC), the first (`s_ff1`, `v_mbcnt`) adds the count to a
/// counter at dword 0 once, and each lane writes t + 1 at 1 + the returned
/// base + its rank. Per half (`Lower::per_half`) each half takes its own
/// slots: the counter ends at 40 and dwords 1-40 hold 1-40, in any order.
fn atomic_slots_in_a_lane_branch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x7E1C_0280, // v_mov_b32 v14, 0 (the counter's offset)
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0018, // s_cbranch_execz +24 (the join)
        0xBE88_047E, // s_mov_b64 s[8:9], exec
        0xBE8A_1008, // s_bcnt1_i32_b64 s10, s[8:9]
        0xBE8B_147E, // s_ff1_i32_b64 s11, exec
        0xD765_000B,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v11, -1, 0
        0xD766_000D,
        0x0002_16C1, // v_mbcnt_hi_u32_b32 v13, -1, v11
        0x7E18_0280, // v_mov_b32 v12, 0
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0x7D24_1A0B, // v_cmpx_eq_i32 exec, s11, v13
        0xBF88_0004, // s_cbranch_execz +4
        0x7E18_020A, // v_mov_b32 v12, s10
        0xE0C8_5000,
        0x800A_0C0E, // buffer_atomic_add v12, v14, s[40:43], 0 offen glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E1C_050C, // v_readfirstlane_b32 s14, v12
        0xD765_000D,
        0x0001_007E, // v_mbcnt_lo_u32_b32 v13, exec_lo, 0
        0xD766_000B,
        0x0002_1A7F, // v_mbcnt_hi_u32_b32 v11, exec_hi, v13
        0x4A1E_160E, // v_add_nc_u32 v15, s14, v11
        0x4A20_1E81, // v_add_nc_u32 v16, 1, v15
        0x3420_2082, // v_lshlrev_b32 v16, 2, v16
        0x4A22_0081, // v_add_nc_u32 v17, 1, v0
        0xE070_1000,
        0x800A_1110, // buffer_store_dword v17, v16, s[40:43], 0 offen
        0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
        0xBF81_0000, // s_endpgm
    ]);
    w
}

#[test]
fn the_atomic_optimizer_in_a_lane_branch_runs_per_half() {
    let words = atomic_slots_in_a_lane_branch();
    let before = lower::PER_HALF_RUNS.with(std::cell::Cell::get);
    // One user SGPR, no LDS.
    let lowered = lowered_with(&words, 1 << 1);
    assert!(lower::PER_HALF_RUNS.with(std::cell::Cell::get) > before, "not lowered per half");
    assert_eq!(lowered.group_syncs, 0, "per half needs no syncs");
    let shader = shader(lowered);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
    }
    let views = [View::RawUav(THREADS)];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let out = &out[0][..THREADS as usize];
        assert_eq!(out[0], 40, "{label}: the counter");
        let mut slots = out[1..41].to_vec();
        slots.sort_unstable();
        assert_eq!(slots, (1..=40).collect::<Vec<u32>>(), "{label}: one slot per lane");
        assert!(out[41..].iter().all(|&x| x == 0), "{label}: nothing past the slots: {out:?}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// The same program with the first lane storing the wave's count instead of
/// adding it (`buffer_store_dword` in place of the atomic): per half, each
/// half would store its own count. Not lowered per half.
#[test]
fn a_stored_lane_count_is_not_run_per_half() {
    let mut words = atomic_slots_in_a_lane_branch();
    let at = words.iter().position(|&w| w == 0xE0C8_5000).expect("the atomic");
    words[at] = 0xE070_1000; // buffer_store_dword v12, v14, s[40:43], 0 offen
    let names: Vec<&str> = {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code).expect("decode").iter().map(|i| i.name().unwrap_or("?")).collect()
    };
    assert!(names.contains(&"buffer_store_dword") && !names.iter().any(|n| n.contains("atomic")));
    let before = lower::PER_HALF_RUNS.with(std::cell::Cell::get);
    let lowered = lowered_with(&words, 1 << 1);
    assert_eq!(lower::PER_HALF_RUNS.with(std::cell::Cell::get), before, "a stored count was lowered per half");
    assert!(lowered.group_syncs > 0, "not run as the console runs it either");
}

/// [`atomic_slots_in_a_lane_branch`] with, before the branch, the first
/// lane's `v0 & 0` (0, but per lane to the translation) read as a scalar
/// load's offset, as the title reads a resource index HLSL holds alike in
/// the wave, and what the load gave, `& 0`, added into the value stored;
/// `stored`: the first lane's value itself added instead.
fn atomic_slots_with_a_uniform_index(stored: bool) -> Vec<u32> {
    let mut w = atomic_slots_in_a_lane_branch();
    let at = w.iter().position(|&x| x == 0x7E1C_0280).expect("v_mov_b32 v14, 0") + 1;
    w.splice(
        at..at,
        [
            0x361E_0080, // v_and_b32 v15, 0, v0
            0x7E20_050F, // v_readfirstlane_b32 s16, v15
            0xF400_0442,
            0x2000_0000, // s_load_dword s17, s[4:5], s16
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ],
    );
    let at = w.iter().position(|&x| x == 0x4A22_0081).expect("v_add_nc_u32 v17, 1, v0") + 1;
    if stored {
        w.insert(at, 0x4A22_2210); // v_add_nc_u32 v17, s16, v17
    } else {
        // What the load gave (as the title reads through a descriptor so
        // loaded), with nothing changed: s17 & 0 added.
        w.splice(
            at..at,
            [
                0x8712_8011, // s_and_b32 s18, s17, 0
                0x4A22_2212, // v_add_nc_u32 v17, s18, v17
            ],
        );
    }
    w
}

#[test]
fn a_first_lane_resource_index_keeps_a_program_per_half() {
    let words = atomic_slots_with_a_uniform_index(false);
    let names: Vec<&str> = {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code).expect("decode").iter().map(|i| i.name().unwrap_or("?")).collect()
    };
    assert!(names.contains(&"v_readfirstlane_b32") && names.contains(&"s_load_dword"), "{names:?}");
    let before = lower::PER_HALF_RUNS.with(std::cell::Cell::get);
    let lowered = lowered_with(&words, 1 << 1);
    assert!(lower::PER_HALF_RUNS.with(std::cell::Cell::get) > before, "not lowered per half");
    assert_eq!(lowered.group_syncs, 0);
    let shader = shader(lowered);
    let views = [View::RawUav(THREADS)];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Ok(host) = Host::new(false) {
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [1, 1, 1] })
            .expect("dispatch");
        let out = &out[0][..THREADS as usize];
        assert_eq!(out[0], 40, "the counter");
        let mut slots = out[1..41].to_vec();
        slots.sort_unstable();
        assert_eq!(slots, (1..=40).collect::<Vec<u32>>(), "one slot per lane");
    } else {
        eprintln!("no SM 6.6 hardware device; the run skipped");
    }
}

#[test]
fn a_first_lane_value_stored_keeps_a_program_out_of_per_half() {
    let words = atomic_slots_with_a_uniform_index(true);
    let before = lower::PER_HALF_RUNS.with(std::cell::Cell::get);
    let lowered = lowered_with(&words, 1 << 1);
    assert_eq!(lower::PER_HALF_RUNS.with(std::cell::Cell::get), before, "lowered per half");
    assert!(lowered.group_syncs > 0, "not run as the console runs it either");
}

// ---- wave64 waterfalls over table entries (the console compiler's form) ----

/// The console compiler's waterfall in a wave64 program of one console
/// wave: each lane holds its table entry's byte offset (`(t & 1) * 32`:
/// even lanes the first buffer, odd ones the second); the loop reads the
/// first active lane's offset, narrows EXEC to the lanes with the same one
/// (`v_cmpx_eq_u32`), loads that entry (`s_load_dwordx4` at the offset) and
/// dword `t` of the buffer, then drops those lanes. What each lane loaded
/// goes to the UAV (table entry 2) at dword 64 + t. With `atomic`, it all
/// runs in a lane branch (t < 40) after the atomic optimizer of
/// [`atomic_slots_in_a_lane_branch`], which only per half takes.
fn indexed_waterfall(atomic: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40 (entry 2)
        0x3424_0082, // v_lshlrev_b32 v18, 2, v0
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    let mut body = vec![
        0xBE96_047E, // s_mov_b64 s[22:23], exec
        0xBE98_047E, // s_mov_b64 s[24:25], exec
        0x7E28_0507, // loop: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec, s20, v7
        0xF408_0302,
        0x2800_0000, // s_load_dwordx4 s[12:15], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2000,
        0x8003_0C00, // buffer_load_dword v12, v0, s[12:15], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x8A98_7E18, // s_andn2_b64 s[24:25], s[24:25], exec
        0xBEFE_0418, // s_mov_b64 exec, s[24:25]
        0xBF89_FFF5, // s_cbranch_execnz loop
        0xBEFE_0416, // s_mov_b64 exec, s[22:23]
        0xE070_1100,
        0x800A_0C12, // buffer_store_dword v12, v18, s[40:43], 0 offen offset:256
    ];
    if atomic {
        // The atomic optimizer's body (its prologue and end left out), then
        // the waterfall, all in the branch.
        let slots = atomic_slots_in_a_lane_branch();
        let from = slots.iter().position(|&x| x == 0x7E1C_0280).expect("v_mov_b32 v14, 0");
        let join = slots.iter().position(|&x| x == 0x88FE_067E).expect("s_or_b64 exec, exec, s[6:7]");
        let mut branch: Vec<u32> = slots[from..join].to_vec();
        branch.append(&mut body);
        // The branch's skip, over the waterfall too.
        let skip = branch.iter().position(|&x| x & 0xFFFF_0000 == 0xBF88_0000).expect("s_cbranch_execz");
        let len = (branch.len() - skip - 1) as u32;
        branch[skip] = 0xBF88_0000 | len;
        body = branch;
        body.push(0x88FE_067E); // s_or_b64 exec, exec, s[6:7]
    }
    w.append(&mut body);
    w.push(0xBF81_0000); // s_endpgm
    w
}

/// [`lowered_with`] for a table of two structured buffers of dwords
/// (entries 0 and 1, eight dwords apart) and a raw UAV (entry 2), no LDS.
fn lowered_with_buffers(words: &[u32]) -> lower::Lowered {
    let range = |kind, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![range(ResourceType::Srv, 0, 4), range(ResourceType::Uav, 16, 0)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    assert!(!res.waterfall.loads.is_empty(), "the loop is not taken for a waterfall");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower_for(
        &lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
        },
        false,
    )
    .expect("lower")
}

/// Runs `shader` with the two buffers (dword t: 100 + t, 200 + t) and the
/// UAV of 128 dwords on the hardware adapter; the UAV, or none.
fn run_with_buffers(shader: &HostShader) -> Option<Vec<u32>> {
    use super::host::Stride;
    let first: Vec<u32> = (0..THREADS).map(|t| 100 + t).collect();
    let second: Vec<u32> = (0..THREADS).map(|t| 200 + t).collect();
    let views = [
        View::Buffer { data: &first, stride: Stride::Structured(4) },
        View::Buffer { data: &second, stride: Stride::Structured(4) },
        View::RawUav(2 * THREADS),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; the run skipped");
        return None;
    };
    let (root, pso) = host.compute_pipeline(shader).expect("pipeline");
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [1, 1, 1] })
        .expect("dispatch");
    Some(out.into_iter().last().expect("the UAV"))
}

/// The waterfall alone, in the split (two host waves): every lane loads from
/// the buffer its entry names.
#[test]
fn a_wave64_waterfall_over_table_entries_gives_each_lane_its_own() {
    let words = indexed_waterfall(false);
    let lowered = lowered_with_buffers(&words);
    let Some(out) = run_with_buffers(&shader(lowered)) else { return };
    let want: Vec<u32> = (0..THREADS).map(|t| if t & 1 == 0 { 100 + t } else { 200 + t }).collect();
    assert_eq!(out[64..128], want[..], "each lane's own buffer");
}

/// The waterfall after the atomic optimizer in a lane branch: per half, as
/// the waterfall's first-lane reads and tests are each lane's own; the
/// lanes in the branch (t < 40) load from their own buffer and take one
/// slot each.
#[test]
fn a_wave64_waterfall_runs_per_half_beside_the_atomic_optimizer() {
    let words = indexed_waterfall(true);
    let before = lower::PER_HALF_RUNS.with(std::cell::Cell::get);
    let lowered = lowered_with_buffers(&words);
    assert!(lower::PER_HALF_RUNS.with(std::cell::Cell::get) > before, "not lowered per half");
    let Some(out) = run_with_buffers(&shader(lowered)) else { return };
    assert_eq!(out[0], 40, "the counter");
    let mut slots = out[1..41].to_vec();
    slots.sort_unstable();
    assert_eq!(slots, (1..=40).collect::<Vec<u32>>(), "one slot per lane");
    let want: Vec<u32> = (0..THREADS)
        .map(|t| match (t < 40, t & 1) {
            (false, _) => 0,
            (true, 0) => 100 + t,
            (true, _) => 200 + t,
        })
        .collect();
    assert_eq!(out[64..128], want[..], "each lane's own buffer");
}

/// Threads of the prefix-sum group: four console waves.
const SUM_THREADS: u32 = 256;

/// A block-wide prefix sum's second step, as the console's BVH builder runs
/// it: console wave k (t >> 6) writes its total 10 + k to LDS at 4k, a
/// barrier, then a loop over the earlier waves adds their totals, the sum so
/// far carried in a scalar by `v_readfirstlane_b32` (every lane holds it
/// alike: scalars and LDS read at a scalar address no one writes after the
/// barrier). Stores 10k + k(k - 1) / 2. (A lane count, unused, as the
/// builder's scan makes one: code that reads lanes is split.) `write_after`: an LDS write after
/// the loop (the read is then not known to be alike in both halves).
fn wave_totals_prefix(write_after: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xD765_000A,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v10, -1, 0 (console lanes: the split)
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0 (the console wave)
        0x4A10_0E8A, // v_add_nc_u32 v8, 10, v7
        0x3412_0E82, // v_lshlrev_b32 v9, 2, v7
        0xD834_0000,
        0x0000_0809, // ds_write_b32 v9, v8
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DAA_0E80, // v_cmpx_ne_u32 exec, 0, v7
        0xBF88_0014, // s_cbranch_execz the end
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBF82_0003, // s_branch the head
        0xBF80_0000, // s_nop 0 (the console compiler aligns the loop)
        0x87FE_0610, // the latch: s_and_b64 exec, s[16:17], s[6:7]
        0x7E22_0508, // v_readfirstlane_b32 s17, v8
        0x8F10_8212, // the head: s_lshl_b32 s16, s18, 2
        0x8112_8112, // s_add_i32 s18, s18, 1
        0x7E10_0210, // v_mov_b32 v8, s16
        0xD8D8_0000,
        0x0800_0008, // ds_read_b32 v8, v8
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A10_1011, // v_add_nc_u32 v8, s17, v8
        0xBE90_047E, // s_mov_b64 s[16:17], exec
        0x7DA4_0E12, // v_cmpx_eq_u32 exec, s18, v7
        0xBF80_0000, // s_nop 0
        0x8A86_7E06, // s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF2, // s_cbranch_scc1 the latch
        0xBEFE_0400, // the end: s_mov_b64 exec, s[0:1]
        0x7E02_0308, // v_mov_b32 v1, v8
    ]);
    if write_after {
        w.extend([
            0xD834_0000,
            0x0000_0809, // ds_write_b32 v9, v8
        ]);
    }
    w.extend_from_slice(EPILOGUE);
    w
}

/// The prefix sum's loop of first-lane reads runs in the plain split: the
/// value read is alike in every lane of both halves, so each lane's own
/// stands in for the console wave's first lane, and no exchange (which
/// every thread would have to reach once, not once per loop turn) is made.
#[test]
fn a_first_lane_read_of_lds_settled_by_a_barrier_needs_no_exchange() {
    let words = wave_totals_prefix(false);
    let lowered = lowered_sized(&words, 1 << 1 | 1 << 15, SUM_THREADS)
        .unwrap_or_else(|e| panic!("the prefix sum is refused: {e}"));
    let shader = shader(lowered);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
    }
    let views = [View::RawUav(SUM_THREADS)];
    let want: Vec<u32> = (0..SUM_THREADS)
        .map(|t| {
            let k = t >> 6;
            10 * k + k * k.saturating_sub(1) / 2
        })
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed (a hang?): {e}"));
        assert_eq!(out[0][..SUM_THREADS as usize], want[..], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// An LDS write after the barrier may part what the two halves of a console
/// wave read, at their different times: the first-lane read is then not
/// known alike, and the split refuses it in the loop.
#[test]
fn a_first_lane_read_of_lds_written_after_the_barrier_is_refused() {
    let words = wave_totals_prefix(true);
    let e = lowered_sized(&words, 1 << 1 | 1 << 15, SUM_THREADS)
        .err()
        .expect("an LDS read written after its barrier taken as alike");
    assert!(e.contains("not every thread"), "{e}");
}

/// Local check, no game data in the repo: `VICTORY_SCAN_CONTAINER` is the
/// path of a block-wide prefix sum compute container (the console's BVH
/// builder has three: `VICTORY_SCAN_THREADS` threads, each loading
/// `VICTORY_SCAN_ELEMS` dwords from its first root UAV and storing their
/// exclusive prefix over the group there, and the group's total at its
/// index in the second root UAV when bit 0 of the root constant is set).
/// Translated for a 32-lane host and run on the hardware adapter over random
/// values through title memory, against a CPU reference.
#[test]
fn one_container_block_scan_matches_its_reference() {
    let (Some(path), Ok(threads), Ok(elems)) = (
        std::env::var_os("VICTORY_SCAN_CONTAINER"),
        std::env::var("VICTORY_SCAN_THREADS").map(|v| v.parse::<usize>().expect("threads")),
        std::env::var("VICTORY_SCAN_ELEMS").map(|v| v.parse::<usize>().expect("elements")),
    ) else {
        return;
    };
    const TITLE: u64 = 0x12_3450_0000;
    const TOTALS: u64 = 0xC000;
    let groups = 3usize;
    let bytes = std::fs::read(path).expect("container");
    super::set_host_wave64(false);
    let shader = super::translate_with(&bytes, super::Options::default()).expect("translates");
    assert!(shader.memory && shader.memory_writes, "root UAVs as title memory");
    assert!(shader.root_cbvs.is_empty() && shader.root_srvs.is_empty() && shader.root_uavs.is_empty());
    // The page table for one 64 KiB copy of title memory at TITLE, its views
    // at heap index 0, the null pair at 2 (`bvh_tests::page_table`).
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    let mut table = vec![0u32; 4096];
    table[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        table.extend([2, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    table[entry..entry + 2].copy_from_slice(&[0, lo]);
    let n = groups * threads * elems;
    assert!(n * 4 <= TOTALS as usize, "the values fit below the totals");
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut memory = vec![0u32; 0x1_0000 / 4];
    for v in &mut memory[..n] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *v = (seed % 1000) as u32;
    }
    let mut want = memory.clone();
    for g in 0..groups {
        let mut sum = 0u32;
        for i in g * threads * elems..(g + 1) * threads * elems {
            let v = memory[i];
            want[i] = sum;
            sum = sum.wrapping_add(v);
        }
        want[TOTALS as usize / 4 + g] = sum;
    }
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Descriptor, 0) => (TITLE >> (32 * u.dword)) as u32,
            (RootType::Descriptor, 1) => ((TITLE + TOTALS) >> (32 * u.dword)) as u32,
            (RootType::Constants, _) => u32::from(u.dword == 0),
            _ => panic!("{u:?}"),
        };
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![
                    (0, Root::Constants(ud)),
                    (1, Root::View { srv: true, data: table }),
                    (2, Root::Uav(2 * 4096 / 32)),
                ],
                views: &[View::Memory(&memory), View::Pair, View::NullPair, View::Pair],
                samplers: &[],
                groups: [groups as u32, 1, 1],
            },
        )
        .expect("dispatch");
    let got = &out[0];
    let bad: Vec<usize> = (0..n).filter(|&i| got[i] != want[i]).collect();
    if let Some(&i) = bad.first() {
        panic!(
            "{} of {n} prefixes wrong; first at {i} (group {}, thread {}, element {}): {} for {}",
            bad.len(),
            i / (threads * elems),
            i / elems % threads,
            i % elems,
            got[i],
            want[i]
        );
    }
    let t = TOTALS as usize / 4;
    assert_eq!(got[t..t + groups], want[t..t + groups], "the group totals");
    eprintln!("{n} prefixes and {groups} totals match");
}

/// Runs `words` (a split program: more than one console wave, no LDS) for a
/// group of `threads` on the hardware adapter; its UAV of `threads` dwords.
fn run_split(words: &[u32], threads: u32) -> Option<Vec<u32>> {
    let lowered = lowered_sized(words, 1 << 1, threads).unwrap_or_else(|e| panic!("refused: {e}"));
    let shader = shader(lowered);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return None;
    };
    let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let views = [View::RawUav(threads)];
    let out = host
        .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [1, 1, 1] })
        .expect("dispatch");
    Some(out[0][..threads as usize].to_vec())
}

/// `ds_swizzle_b32` in its bit-mask mode works within each 32 lanes of the
/// console wave: lane l reads lane (l & ~31) | ((l & 31 & and) | or) ^ xor.
/// In the split each half is a host wave of its own, and its lanes read
/// their own half. A lane count (unused) makes the program split.
#[test]
fn a_wave64_swizzle_reads_within_each_half() {
    for offset in [30u32, 120] {
        let mut w = PROLOGUE.to_vec();
        w.extend([
            0xD765_000A,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v10, -1, 0 (console lanes: the split)
            0x7E0A_0300, // v_mov_b32 v5, v0
            0xD8D4_0000 | offset,
            0x0100_0005, // ds_swizzle_b32 v1, v5 offset:<offset>
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ]);
        w.extend_from_slice(EPILOGUE);
        let Some(out) = run_split(&w, 128) else { return };
        let (and, or, xor) = (offset & 31, offset >> 5 & 31, offset >> 10 & 31);
        let want: Vec<u32> = (0..128).map(|l| (l & !31) | ((l & 31 & and) | or) ^ xor).collect();
        assert_eq!(out, want, "offset {offset}");
    }
}

/// A select on a lane mask whose two dwords are written apart
/// (`s_mov_b32 s0, 0; s_mov_b32 s1, -1`, as the console compiler does):
/// lanes 32-63 of each console wave take their bit from the high dword.
#[test]
fn a_wave64_select_on_a_mask_written_by_halves_takes_each_halfs_dword() {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xD765_000A,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v10, -1, 0 (console lanes: the split)
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_03C1, // s_mov_b32 s1, -1
        0xD501_0001,
        0x0001_0280, // v_cndmask_b32 v1, 0, 1, s[0:1]
    ]);
    w.extend_from_slice(EPILOGUE);
    let Some(out) = run_split(&w, 128) else { return };
    let want: Vec<u32> = (0..128).map(|l| u32::from(l & 32 != 0)).collect();
    assert_eq!(out, want);
}

/// Local check, no game data in the repo: `VICTORY_RADIX_CONTAINER` is the
/// path of the console BVH builder's radix sort pass (512 threads; each
/// group sorts a tile of 2048 keys (u0) and values (u1) by the 8 bits from
/// constant 1 on, and scatters them to u4 and u5 at `u2[digit * groups +
/// group] + u3[(digit * groups + group) >> constant 2] + rank`). One group,
/// u3 zero, u2 the digits' exclusive counts: the output is the tile sorted
/// stably by the digit. Run on the hardware adapter through title memory,
/// for digits at bits 0 and 8.
#[test]
fn one_container_radix_pass_matches_its_reference() {
    let Some(path) = std::env::var_os("VICTORY_RADIX_CONTAINER") else {
        return;
    };
    const TITLE: u64 = 0x12_3450_0000;
    const TILE: usize = 2048;
    // Byte offsets in the one page: keys, values, offsets, coarse offsets,
    // keys out, values out.
    const AT: [u64; 6] = [0x0000, 0x2000, 0x4000, 0x4400, 0x5000, 0x7000];
    let bytes = std::fs::read(path).expect("container");
    super::set_host_wave64(false);
    let shader = super::translate_with(&bytes, super::Options::default()).expect("translates");
    assert!(shader.memory && shader.memory_writes, "root UAVs as title memory");
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    let mut table = vec![0u32; 4096];
    table[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        table.extend([2, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    table[entry..entry + 2].copy_from_slice(&[0, lo]);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
    for shift in [0u32, 8] {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(shift);
        let mut memory = vec![0u32; 0x1_0000 / 4];
        let w = |at: u64| at as usize / 4;
        for i in 0..TILE {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            memory[w(AT[0]) + i] = seed as u32;
            memory[w(AT[1]) + i] = i as u32;
        }
        let digit = |k: u32| (k >> shift & 255) as usize;
        let mut counts = [0u32; 256];
        for i in 0..TILE {
            counts[digit(memory[w(AT[0]) + i])] += 1;
        }
        let mut first = [0u32; 256];
        let mut sum = 0;
        for d in 0..256 {
            first[d] = sum;
            sum += counts[d];
        }
        memory[w(AT[2])..w(AT[2]) + 256].copy_from_slice(&first);
        let mut want = memory.clone();
        let mut next = first;
        for i in 0..TILE {
            let (k, v) = (memory[w(AT[0]) + i], memory[w(AT[1]) + i]);
            let p = next[digit(k)] as usize;
            next[digit(k)] += 1;
            want[w(AT[4]) + p] = k;
            want[w(AT[5]) + p] = v;
        }
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Descriptor, s) if (s as usize) < AT.len() => ((TITLE + AT[s as usize]) >> (32 * u.dword)) as u32,
                (RootType::Constants, _) => [1, shift, 0][u.dword as usize],
                _ => panic!("{u:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: table.clone() }),
                        (2, Root::Uav(2 * 4096 / 32)),
                    ],
                    views: &[View::Memory(&memory), View::Pair, View::NullPair, View::Pair],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        let got = &out[0];
        for (name, at) in [("keys", AT[4]), ("values", AT[5])] {
            let bad: Vec<usize> = (0..TILE).filter(|&p| got[w(at) + p] != want[w(at) + p]).collect();
            if let Some(&p) = bad.first() {
                let mut seen = got[w(AT[5])..w(AT[5]) + TILE].to_vec();
                seen.sort_unstable();
                seen.dedup();
                panic!(
                    "bits {shift}..: {} of {TILE} {name} wrong; first at {p}: {:#x} for {:#x} ({} distinct values out)",
                    bad.len(),
                    got[w(at) + p],
                    want[w(at) + p],
                    seen.len()
                );
            }
        }
        eprintln!("bits {shift}..: {TILE} keys and values in place");
    }
}

/// Local check, no game data in the repo: `VICTORY_HISTOGRAM_CONTAINER` is
/// the path of the console BVH builder's radix histogram (512 threads; each
/// group counts the 8-bit digits, from bit constant 1 on, of its 2048 keys
/// (u0) and stores count d at `u1[d * groups + group]`). The counts come from
/// 64-lane compare masks used as data (`v_xor`, `v_bcnt`): in the split both
/// halves must see the console wave's whole masks. One group, digits at bits
/// 0 and 8, on the hardware adapter through title memory.
#[test]
fn one_container_radix_histogram_matches_its_reference() {
    let Some(path) = std::env::var_os("VICTORY_HISTOGRAM_CONTAINER") else {
        return;
    };
    const TITLE: u64 = 0x12_3450_0000;
    const TILE: usize = 2048;
    const AT: [u64; 2] = [0x0000, 0x4000];
    let bytes = std::fs::read(path).expect("container");
    super::set_host_wave64(false);
    let shader = super::translate_with(&bytes, super::Options::default()).expect("translates");
    assert!(shader.memory && shader.memory_writes, "root UAVs as title memory");
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    let mut table = vec![0u32; 4096];
    table[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        table.extend([2, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    table[entry..entry + 2].copy_from_slice(&[0, lo]);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
    for shift in [0u32, 8] {
        let mut seed = 0xD1B5_4A32_D192_ED03u64 ^ u64::from(shift);
        let mut memory = vec![0u32; 0x1_0000 / 4];
        let w = |at: u64| at as usize / 4;
        let mut want = [0u32; 256];
        for i in 0..TILE {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // Few digits, many repeats (lanes sharing a digit).
            let k = (seed as u32) & !(0xE0u32 << shift);
            memory[w(AT[0]) + i] = k;
            want[(k >> shift & 255) as usize] += 1;
        }
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Descriptor, s) if (s as usize) < AT.len() => ((TITLE + AT[s as usize]) >> (32 * u.dword)) as u32,
                (RootType::Descriptor, _) => 0,
                (RootType::Constants, _) => [1, shift, 0][u.dword as usize],
                _ => panic!("{u:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: table.clone() }),
                        (2, Root::Uav(2 * 4096 / 32)),
                    ],
                    views: &[View::Memory(&memory), View::Pair, View::NullPair, View::Pair],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        let got = &out[0][w(AT[1])..w(AT[1]) + 256];
        let total: u32 = got.iter().sum();
        let bad: Vec<usize> = (0..256).filter(|&d| got[d] != want[d]).collect();
        assert!(
            bad.is_empty(),
            "bits {shift}..: {} of 256 counts wrong (total {total} of {TILE}); first digit {}: {} for {}",
            bad.len(),
            bad[0],
            got[bad[0]],
            want[bad[0]]
        );
        eprintln!("bits {shift}..: 256 counts match");
    }
}

/// Writes the DXIL `VICTORY_DXIL_CONTAINER`'s program translates to (with
/// 32-lane host waves) to `VICTORY_DXIL_OUT`, for `dxc -dumpbin` (local).
#[test]
fn one_container_dxil_written() {
    let (Some(path), Some(out)) = (std::env::var_os("VICTORY_DXIL_CONTAINER"), std::env::var_os("VICTORY_DXIL_OUT"))
    else {
        return;
    };
    let bytes = std::fs::read(path).expect("container");
    super::set_host_wave64(false);
    // As the game makes it (`shader_cache`).
    let shader = super::translate_with(&bytes, super::Options { view_marks: true }).expect("translates");
    std::fs::write(out, &shader.dxil).expect("written");
}

/// The console BVH builder's equality-exit waterfall (`28918b71`'s loops at
/// 0x2e0 and 0x798): each lane's count is `t >> 4` (0 for threads 0-15,
/// 1, 2, 3 for the others, so the halves' largest differ); lanes with 0
/// stay out (`v_cmpx_eq_u32 exec, 0, v9`, then `s_andn2_b64 exec`); the
/// loop counts s20 up and drops a lane when s20 equals its count, until
/// none is left (`s_andn2_b64 s[16:17], s[16:17], exec`, then SCC). Each
/// pass adds 1 to v1: a lane stores its count. With `clobber`, the body
/// saves EXEC in s[18:19], restores it, then uses s18 as a scalar
/// (`s_mul_i32 s18, s20, 6`) before saving EXEC there again, as 0x2e0 does.
fn equality_exit_waterfall(clobber: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x2C12_0084, // v_lshrrev_b32 v9, 4, v0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0x7DA4_1280, // v_cmpx_eq_u32 exec, 0, v9
        0x8AFE_7E0C, // s_andn2_b64 exec, s[12:13], exec
    ]);
    let body: Vec<u32> = if clobber {
        vec![
            0xBE92_047E, // s_mov_b64 s[18:19], exec
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
            0xBEFE_0412, // s_mov_b64 exec, s[18:19]
            0x9312_8614, // s_mul_i32 s18, s20, 6
            0x8114_8114, // s_add_i32 s20, s20, 1
        ]
    } else {
        vec![
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
            0x8114_8114, // s_add_i32 s20, s20, 1
        ]
    };
    let tail = [
        0xBE92_047E, // s_mov_b64 s[18:19], exec
        0x7DA4_1214, // v_cmpx_eq_u32 exec, s20, v9
        0xBF80_0000, // s_nop 0
        0x8A90_7E10, // s_andn2_b64 s[16:17], s[16:17], exec
    ];
    // Word indices from the s_cbranch_execz: execz(0) s16(1) s20(2)
    // branch(3) top: and(4) nop(5) body(6..) tail, scc1, restore.
    let n = body.len() as u32 + tail.len() as u32;
    let scc1 = 6 + n;
    let restore = scc1 + 1;
    w.push(0xBF88_0000 | (restore - 1)); // s_cbranch_execz restore
    w.extend([
        0xBE90_047E, // s_mov_b64 s[16:17], exec
        0xBE94_0380, // s_mov_b32 s20, 0
        0xBF82_0002, // s_branch body
        0x87FE_1012, // top: s_and_b64 exec, s[18:19], s[16:17]
        0xBF80_0000, // s_nop 0
    ]);
    w.extend(body);
    w.extend(tail);
    let back = 4i32 - (scc1 as i32 + 1);
    w.push(0xBF85_0000 | (back as u32 & 0xFFFF)); // s_cbranch_scc1 top
    w.push(0xBEFE_040C); // restore: s_mov_b64 exec, s[12:13]
    w.extend_from_slice(EPILOGUE);
    w
}

#[test]
fn an_equality_exit_waterfall_leaves_when_its_lanes_are_done() {
    runs_as_the_console("waterfall", &equality_exit_waterfall(false), |t| t >> 4);
}

#[test]
fn an_equality_exit_waterfall_with_its_saved_exec_reused_leaves_too() {
    runs_as_the_console("waterfall, s18 reused", &equality_exit_waterfall(true), |t| t >> 4);
}

#[test]
fn equality_exit_waterfall_decodes() {
    let code: Vec<u8> = equality_exit_waterfall(false).iter().flat_map(|w| w.to_le_bytes()).collect();
    for i in crate::gfx10::decode(&code).expect("decode") {
        eprintln!("{:?}", i.name());
    }
}

#[test]
fn equality_exit_waterfall_dxil_written() {
    let Some(out) = std::env::var_os("VICTORY_WATERFALL_DXIL") else { return };
    std::fs::write(out, lowered(&equality_exit_waterfall(false)).dxil).expect("written");
}

/// Local diagnostics: for `VICTORY_GUARDS_CONTAINER`'s compute program, each
/// block in the byte range `VICTORY_GUARDS_RANGE` (`from-to`, hex): its
/// exit, what holds in lanes entering it, and each instruction's guard, as
/// formulas over the atoms (with the values each instruction writes).
#[test]
fn one_container_guards_written() {
    let (Some(path), Some(range)) =
        (std::env::var_os("VICTORY_GUARDS_CONTAINER"), std::env::var("VICTORY_GUARDS_RANGE").ok())
    else {
        return;
    };
    let (from, to) = range.split_once('-').expect("from-to");
    let (from, to) = (usize::from_str_radix(from, 16).unwrap(), usize::from_str_radix(to, 16).unwrap());
    let bytes = std::fs::read(path).expect("container");
    let c = crate::dxbc::Container::parse(&bytes).expect("container");
    let dxil = c.part(crate::dxbc::FourCC::DXIL).or(c.part(crate::dxbc::FourCC::ACCC)).expect("dxil");
    let native = crate::native_programs(dxil.data)
        .expect("programs")
        .into_iter()
        .find(|p| p.stage == HwStage::Cs)
        .expect("compute");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structure");
    fn show(st: &structure::Structured, b: structure::Bool, budget: &mut usize) -> String {
        if b == structure::Bool::TRUE {
            return "T".into();
        }
        if b == structure::Bool::FALSE {
            return "F".into();
        }
        if *budget == 0 {
            return "…".into();
        }
        *budget -= 1;
        let (a, lo, hi) = st.bools.node(b).expect("a node");
        let a = format!("{a:?}").replace("Value(", "").replace(')', "");
        match (lo, hi) {
            (structure::Bool::FALSE, structure::Bool::TRUE) => a,
            (structure::Bool::TRUE, structure::Bool::FALSE) => format!("!{a}"),
            _ => format!("({a} ? {} : {})", show(st, hi, budget), show(st, lo, budget)),
        }
    }
    for (bi, block) in p.blocks.iter().enumerate() {
        if block.offset < from || block.offset > to {
            continue;
        }
        let mut n = 40;
        eprintln!("block {bi} @{:#x} exit {:?} reach {}", block.offset, st.exits[bi], show(&st, st.reach[bi], &mut n));
        if let structure::LaneExit::Branch { when, .. } = st.exits[bi] {
            let mut n = 40;
            eprintln!("  when {}", show(&st, when, &mut n));
        }
        if let Some(w) = st.whens[bi] {
            let mut n = 40;
            eprintln!("  whens {}", show(&st, w, &mut n));
        }
        for (ii, inst) in block.insts.iter().enumerate() {
            let mut n = 40;
            let g = st.guards[bi][ii].map_or("-".into(), |g| show(&st, g, &mut n));
            let w: Vec<String> = inst.writes.iter().map(|(r, v)| format!("{r:?}={v:?}")).collect();
            eprintln!("  {} guard {g} writes {}", inst.name(), w.join(" ").replace("Value(", "").replace(')', ""));
        }
    }
}

/// Local check, no game data in the repo: replays a dispatch the game run
/// captured (`VICTORY_BUILDER_CAPTURE`): `VICTORY_REPLAY_CONTAINER` the
/// program's container, `VICTORY_REPLAY_DIR` the capture folder and
/// `VICTORY_REPLAY_K` the dispatch. Every view it captured is mapped at its
/// own title address (64 KiB pages into one memory view, up to
/// `VICTORY_REPLAY_MB` MiB of each view, default 8), its root views and
/// constants given as the game gave them. It must complete.
#[test]
fn one_capture_replays() {
    let (Some(container), Some(dir)) =
        (std::env::var_os("VICTORY_REPLAY_CONTAINER"), std::env::var_os("VICTORY_REPLAY_DIR"))
    else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let k: u32 = std::env::var("VICTORY_REPLAY_K").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let cap: usize = std::env::var("VICTORY_REPLAY_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(8) << 20;
    let meta = std::fs::read_to_string(dir.join(format!("{k}.txt"))).expect("capture meta");
    let (mut groups, mut views, mut constants) = ([1u32; 3], Vec::new(), std::collections::HashMap::new());
    for line in meta.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.first().copied() {
            Some("groups") => {
                for i in 0..3 {
                    groups[i] = w[1 + i].parse().unwrap();
                }
            }
            Some("view") => {
                let slot: u32 = w[1].parse().unwrap();
                let at = u64::from_str_radix(w[2].trim_start_matches("0x"), 16).unwrap();
                views.push((slot, at));
            }
            Some("constants") => {
                let slot: u32 = w[1].parse().unwrap();
                let list = line.split_once('[').unwrap().1.trim_end_matches(']');
                let c: Vec<u32> = list.split(',').map(|x| u32::from_str_radix(x.trim(), 16).unwrap()).collect();
                constants.insert(slot, c);
            }
            _ => {}
        }
    }
    // Pages: every 64 KiB page a captured view covers, in one memory view.
    let mut pages: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
    let mut bytes_of = Vec::new();
    for &(slot, at) in &views {
        let Ok(b) = std::fs::read(dir.join(format!("{k}-{slot}.bin"))) else { continue };
        let b = b[..b.len().min(cap)].to_vec();
        let mut p = at & !0xFFFF;
        while p < at + b.len() as u64 {
            let n = pages.len();
            pages.entry(p).or_insert(n);
            p += 0x1_0000;
        }
        bytes_of.push((at, b));
    }
    let mut memory = vec![0u32; pages.len() * 0x4000];
    let mut image = vec![0u8; pages.len() * 0x1_0000];
    for (at, b) in &bytes_of {
        for (i, byte) in b.iter().enumerate() {
            let a = at + i as u64;
            let slot = pages[&(a & !0xFFFF)];
            image[slot * 0x1_0000 + (a & 0xFFFF) as usize] = *byte;
        }
    }
    for (i, w) in image.chunks(4).enumerate() {
        memory[i] = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
    }
    // The page table: level 1 by 256 MiB region, then blocks of 4096
    // entries (block 0: the null pair for misses).
    let mut table = vec![0u32; 4096];
    let mut blocks: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for &p in pages.keys() {
        let l1 = ((p >> 32) << 4 | (p & 0xFFFF_FFFF) >> 28) as usize;
        let n = blocks.len() as u32 + 1;
        let b = *blocks.entry(l1).or_insert(n);
        table[l1] = b;
    }
    for _ in 0..(blocks.len() + 1) * 4096 {
        table.extend([2, 0]);
    }
    for (&p, &slot) in &pages {
        let l1 = ((p >> 32) << 4 | (p & 0xFFFF_FFFF) >> 28) as usize;
        let e = 4096 + 2 * (blocks[&l1] as usize * 4096 + ((p as u32) >> 16 & 0xFFF) as usize);
        let start = (p as u32).wrapping_sub((slot * 0x1_0000) as u32);
        table[e..e + 2].copy_from_slice(&[0, start]);
    }
    let bytes = std::fs::read(container).expect("container");
    let shader = super::translate_with(&bytes, super::Options { view_marks: true }).expect("translates");
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Descriptor => {
                let at = views.iter().find(|v| v.0 == u.slot as u32).map_or(0, |v| v.1);
                (at >> (32 * u.dword)) as u32
            }
            RootType::Constants => constants.get(&(u.slot as u32)).and_then(|c| c.get(u.dword as usize)).copied().unwrap_or(0),
            _ => 0,
        };
    }
    // `VICTORY_REPLAY_GROUPS`: so many groups instead (the first ones).
    if let Some(g) = std::env::var("VICTORY_REPLAY_GROUPS").ok().and_then(|v| v.parse().ok()) {
        groups = [g, 1, 1];
    }
    eprintln!("replay {k}: groups {groups:?}, {} views, {} pages, user data {ud:x?}", views.len(), pages.len());
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
    let started = std::time::Instant::now();
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![
                (0, Root::Constants(ud)),
                (1, Root::View { srv: true, data: table.clone() }),
                (2, Root::Uav(((blocks.len() + 1) * 4096 / 32) as u32)),
            ],
            views: &[View::Memory(&memory), View::Pair, View::NullPair, View::Pair],
            samplers: &[],
            groups,
        },
    );
    match out {
        Ok(out) => {
            eprintln!("replay {k}: completed in {:?}", started.elapsed());
            // What changed, per captured view: counts, and the largest new
            // values of dwords that were 0 (counters).
            let after = &out[0];
            // `VICTORY_REPLAY_DUMP`: each view's bytes before and after.
            if let Some(d) = std::env::var_os("VICTORY_REPLAY_DUMP") {
                let d = std::path::PathBuf::from(d);
                let _ = std::fs::create_dir_all(&d);
                for &(slot, at) in &views {
                    let Some((_, b)) = bytes_of.iter().find(|x| x.0 == at) else { continue };
                    let words = |m: &[u32]| -> Vec<u8> {
                        (0..b.len() / 4)
                            .flat_map(|i| {
                                let a = at + 4 * i as u64;
                                m[(pages[&(a & !0xFFFF)] * 0x1_0000 + (a & 0xFFFF) as usize) / 4].to_le_bytes()
                            })
                            .collect()
                    };
                    let _ = std::fs::write(d.join(format!("{slot}-after.bin")), words(after));
                    let _ = std::fs::write(d.join(format!("{slot}-before.bin")), words(&memory));
                }
            }
            for (at, b) in &bytes_of {
                let (mut changed, mut from_zero) = (0, Vec::new());
                for i in (0..b.len() / 4).map(|i| i * 4) {
                    let a = at + i as u64;
                    let w = (pages[&(a & !0xFFFF)] * 0x1_0000 + (a & 0xFFFF) as usize) / 4;
                    if after[w] != memory[w] {
                        changed += 1;
                        if memory[w] == 0 {
                            from_zero.push((after[w], i / 4));
                        }
                    }
                }
                let bins = [1u32, 2, 3, 4, 10, 100, 301, 1000, u32::MAX - 1];
                let mut hist = vec![0; bins.len() + 1];
                for &(v, _) in &from_zero {
                    hist[bins.iter().position(|&b| v < b).unwrap_or(bins.len())] += 1;
                }
                eprintln!("  view at {at:#x}: from 0 to <1,<2,<3,<4,<10,<100,<301,<1000,<max-1,rest: {hist:?}");
                let mut mid: Vec<(u32, usize)> = from_zero.iter().copied().filter(|x| x.0 > 2 && x.0 < 0xFFFF_0000).collect();
                mid.sort_unstable_by(|x, y| y.cmp(x));
                eprintln!("    largest below -1: {:?}", &mid[..mid.len().min(10)]);
                from_zero.sort_unstable_by(|x, y| y.cmp(x));
                let small = from_zero.iter().filter(|x| x.0 <= 2).count();
                eprintln!(
                    "  view at {at:#x}: {changed} dwords changed, {} from 0 ({small} to 1 or 2); largest {:?}",
                    from_zero.len(),
                    &from_zero[..from_zero.len().min(8)]
                );
            }
        }
        Err(e) => panic!("replay {k}: failed after {:?}: {e}", started.elapsed()),
    }
}
