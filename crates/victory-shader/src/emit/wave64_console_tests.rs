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

/// Lowered for a host of 32-lane waves.
fn lowered(words: &[u32]) -> lower::Lowered {
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
            // 512 bytes of LDS.
            rsrc2: 1 << 1 | 1 << 15,
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
        prologue.iter().chain(body).chain(&epilogue).copied().collect()
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
