//! Values every lane of a wave holds alike: a first-lane read of one is the
//! lane's own ([`Lower::made_of_scalars`]). Split from `lower.rs` (one file
//! stays under the asset policy's size limit).

use super::*;

impl<'a> Lower<'a> {
    /// Whether the VGPR that instruction `i` of IR block `b` reads first was
    /// made by lane-wise arithmetic, selects and compares of scalars and
    /// constants only (a `v_sub_f32 v, s, s` of a spilled scalar, say), each
    /// step either earlier in that block with no write of EXEC in between,
    /// or unguarded in a block dominating it: every lane active at `i` then
    /// computed the same value. Phis, inputs and other instructions count as
    /// lane-dependent.
    pub(super) fn made_of_scalars(&self, b: usize, i: usize) -> bool {
        self.why_not_scalars(b, i, false).is_none()
    }

    /// Why the value the instruction reads first is not made of scalars
    /// ([`Lower::made_of_scalars`]); with `phis`, a phi of such values in a
    /// block that every lane of a (host) wave reaches the same way counts too
    /// (each half of a console wave as its own wave, [`Lower::per_half`]:
    /// a branch on a lane mask's SCC parts the halves, not their lanes).
    pub(super) fn why_not_scalars(&self, b: usize, i: usize, phis: bool) -> Option<String> {
        let p = self.x.p;
        let insts = &p.blocks[b].insts;
        let Fields::Valu(v) = &insts[i].fields else {
            return Some("not a VALU".into());
        };
        let Some(&Operand::Vector { reg, dwords: 1 }) = v.src.first() else {
            return Some("source not one VGPR".into());
        };
        let Some(first) = insts[i].read(Reg::V(reg)) else {
            return Some("no read".into());
        };
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (n, &k) in order.iter().enumerate() {
            index[k] = n;
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &index);
        let dominates = |d: usize, mut k: usize| loop {
            if k == d {
                break true;
            }
            if k == usize::MAX || idom[k] == k {
                break false;
            }
            k = idom[k];
        };
        let exec_written = |k: usize, from: usize, to: usize| {
            p.blocks[k].insts[from..to].iter().any(|w| {
                w.writes
                    .iter()
                    .any(|x| matches!(x.0, Reg::S(c) if c == EXEC || c == EXEC + 1))
            })
        };
        // Values wanted, each with where it is read: the block and the
        // instruction (none: a phi's edge, at the end of the block).
        let mut want: Vec<(Value, usize, Option<usize>)> = vec![(first, b, Some(i))];
        let mut seen = HashSet::new();
        let uniform = |k: usize| match self.x.st.exits[k] {
            LaneExit::Branch { when, .. } => self
                .x
                .st
                .bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_))),
            _ => true,
        };
        while let Some((x, ub, ui)) = want.pop() {
            if !seen.insert(x) {
                continue;
            }
            let Some((db, di, w)) = self.def_site(x) else {
                // A phi: every block between its block's dominator and it
                // branching alike for the whole wave, each argument made of
                // scalars where its edge leaves.
                let found = phis
                    .then(|| {
                        p.blocks.iter().enumerate().find_map(|(k, blk)| {
                            blk.phis.iter().find(|ph| ph.value == x).map(|ph| (k, ph))
                        })
                    })
                    .flatten();
                let Some((pb, phi)) = found else {
                    return Some(format!("{x:?} is a phi or an input"));
                };
                let d = idom[pb];
                if d == usize::MAX || d == pb {
                    return Some(format!("{x:?} a phi at the entry"));
                }
                let mut todo: Vec<usize> = p.blocks[pb].preds.clone();
                let mut region = HashSet::new();
                while let Some(k) = todo.pop() {
                    if k == d || !region.insert(k) {
                        continue;
                    }
                    todo.extend(p.blocks[k].preds.iter().copied());
                }
                region.insert(d);
                if let Some(k) = region.iter().find(|&&k| !uniform(k)) {
                    return Some(format!("{x:?} a phi past a lane branch (block {k})"));
                }
                for (&arg, &pred) in phi.args.iter().zip(&p.blocks[pb].preds) {
                    want.push((arg, pred, None));
                }
                continue;
            };
            let unguarded = self.x.st.guards[db][di].is_none();
            let placed = if db == ub {
                let to = ui.unwrap_or(p.blocks[ub].insts.len());
                di < to && (unguarded || !exec_written(ub, di + 1, to))
            } else {
                unguarded && dominates(db, ub)
            };
            if !placed {
                return Some(format!(
                    "{} at {:#x} (block {db}) not reaching every lane",
                    w.name(),
                    w.machine.offset
                ));
            }
            // A buffer load at an index made of scalars, from a buffer the
            // program does not write: the same element in every lane (a
            // uniform value the console compiler reads back with
            // `v_readfirstlane`). (Per half, the two halves load at
            // different times: a write of the program's own in between
            // would part them.)
            if let Fields::Buffer { addr, rsrc, soffset, lds: false, .. } = &w.fields
                && w.name().starts_with("buffer_load")
            {
                let written = |r: Value| {
                    self.x.p.blocks.iter().flat_map(|b| &b.insts).any(|o| {
                        let n = o.name();
                        (n.contains("store") || n.contains("atomic"))
                            && matches!(o.fields, Fields::Buffer { .. })
                            && o.reads.iter().any(|x| x.1 == r)
                    })
                };
                let Operand::Scalar { code, dwords: 4 } = *rsrc else {
                    return Some(format!("a load at {:#x} through no descriptor", w.machine.offset));
                };
                for k in 0..4 {
                    match w.read(Reg::S(code + k)) {
                        Some(r) if !written(r) => {}
                        _ => {
                            return Some(format!("a load at {:#x} from a buffer the program writes", w.machine.offset));
                        }
                    }
                }
                if !matches!(soffset, Operand::Scalar { .. } | Operand::Const(_)) {
                    return Some(format!("a load at {:#x} with a lane's offset", w.machine.offset));
                }
                match *addr {
                    Operand::Vector { reg, dwords: 1 } => match w.read(Reg::V(reg)) {
                        Some(m) => want.push((m, db, Some(di))),
                        None => return Some(format!("no read of v{reg} at {:#x}", w.machine.offset)),
                    },
                    _ => return Some(format!("a load at {:#x} at no single index", w.machine.offset)),
                }
                continue;
            }
            let Fields::Valu(wv) = &w.fields else {
                return Some(format!(
                    "made by non-VALU {} at {:#x}",
                    w.name(),
                    w.machine.offset
                ));
            };
            let name = w.name();
            let select = name == "v_cndmask_b32";
            // A compare of 32-bit values writes the mask a select reads.
            let compare = name.starts_with("v_cmp_")
                && !["f16", "f64", "u64", "i64", "u16", "i16"]
                    .iter()
                    .any(|t| name.ends_with(t));
            if !(lanewise(name) || select || compare)
                || wv.dpp.is_some()
                || wv.sdwa.is_some()
                || wv.carry.is_some()
            {
                return Some(format!("made by {name} at {:#x}", w.machine.offset));
            }
            let dst_ok = if compare {
                matches!(wv.dst, Some(Operand::Scalar { dwords: 2, .. })) && w.writes.len() == 2
            } else {
                matches!(wv.dst, Some(Operand::Vector { dwords: 1, .. })) && w.writes.len() == 1
            };
            if !dst_ok {
                return Some(format!(
                    "{name} at {:#x} writes {:?}",
                    w.machine.offset, w.writes
                ));
            }
            // `v_fmac`/`v_mac` add to their destination's old value.
            let acc = match (name, wv.dst) {
                ("v_fmac_f32" | "v_mac_f32", Some(d)) => Some(d),
                _ => None,
            };
            for (k, op) in wv.src.iter().chain(acc.iter()).enumerate() {
                match *op {
                    // A select's mask: the low dword's value, made by a compare.
                    Operand::Scalar { code, dwords: 2 } if select && k == 2 && code < EXEC => {
                        match w.read(Reg::S(code)) {
                            Some(m) => want.push((m, db, Some(di))),
                            None => {
                                return Some(format!(
                                    "no read of s{code} at {:#x}",
                                    w.machine.offset
                                ));
                            }
                        }
                    }
                    Operand::Scalar { code, .. }
                        if code != VCC && code != VCC + 1 && code < EXEC => {}
                    Operand::Const(c) if c < 251 => {}
                    Operand::Literal(_) => {}
                    Operand::Vector { reg, dwords: 1 } => match w.read(Reg::V(reg)) {
                        Some(m) => want.push((m, db, Some(di))),
                        None => {
                            return Some(format!("no read of v{reg} at {:#x}", w.machine.offset));
                        }
                    },
                    ref o => return Some(format!("{name} at {:#x} reads {o:?}", w.machine.offset)),
                }
            }
        }
        None
    }
}

/// 32-bit VALU operations whose result in a lane depends only on that
/// lane's sources (no lane masks, carries, lane indices or other lanes).
pub(super) fn lanewise(name: &str) -> bool {
    const OPS: &[&str] = &[
        "v_mov_b32",
        "v_add_f32",
        "v_sub_f32",
        "v_subrev_f32",
        "v_mul_f32",
        "v_mul_legacy_f32",
        "v_fma_f32",
        "v_fmac_f32",
        "v_fmaak_f32",
        "v_fmamk_f32",
        "v_mac_f32",
        "v_mad_f32",
        "v_min_f32",
        "v_max_f32",
        "v_min3_f32",
        "v_max3_f32",
        "v_med3_f32",
        "v_rcp_f32",
        "v_rcp_iflag_f32",
        "v_rsq_f32",
        "v_sqrt_f32",
        "v_exp_f32",
        "v_log_f32",
        "v_floor_f32",
        "v_ceil_f32",
        "v_trunc_f32",
        "v_rndne_f32",
        "v_fract_f32",
        "v_sin_f32",
        "v_cos_f32",
        "v_cvt_f32_u32",
        "v_cvt_f32_i32",
        "v_cvt_u32_f32",
        "v_cvt_i32_f32",
        "v_add_nc_u32",
        "v_sub_nc_u32",
        "v_subrev_nc_u32",
        "v_mul_lo_u32",
        "v_mul_hi_u32",
        "v_mul_u32_u24",
        "v_and_b32",
        "v_or_b32",
        "v_xor_b32",
        "v_not_b32",
        "v_lshlrev_b32",
        "v_lshrrev_b32",
        "v_ashrrev_i32",
        "v_bfe_u32",
        "v_bfe_i32",
        "v_bfi_b32",
        "v_min_u32",
        "v_max_u32",
        "v_min_i32",
        "v_max_i32",
    ];
    OPS.contains(&name)
}

