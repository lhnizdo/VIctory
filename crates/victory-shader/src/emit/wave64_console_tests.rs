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
            threads: [THREADS, 1, 1],
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
    .expect("lower")
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
            .unwrap_or_else(|e| panic!("{name}, {label}: dispatch failed (a hang?): {e}"));
        assert_eq!(out[0][..THREADS as usize], want[..], "{name}, {label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
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
