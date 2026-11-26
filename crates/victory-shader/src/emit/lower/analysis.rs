//! The lowering's analyses of control flow and values: group barriers
//! and how lanes reach them (balanced counts, loop exits decided for the
//! wave, console mode's wave-wide branches), registers dead on an edge,
//! constants and comparisons the program's values settle to, and the phi
//! edges that are live. Split from `lower.rs` (one file stays under the
//! asset policy's size limit).

use super::*;

impl<'a> Lower<'a> {
    /// Console mode's parked lanes ([`Lower::parked`]): a phi of whether a
    /// lane is parked at every loop header, false on entry. A latch whose
    /// exit is decided for the wave parks the lanes that would have left
    /// ([`Lower::wave_branch`]), and every instruction in the loop is then
    /// guarded by `!parked` too ([`Lower::unparked`]).
    // shortcut: a lane is parked only at a latch whose other side leaves
    // the loop (the do-while shape); a branch going round the loop from
    // the middle keeps its lanes running to the latch.
    pub(super) fn park_lanes(&mut self, order: &[usize]) {
        let loops = self.barrier_loops(order);
        let headers: Vec<usize> = self.counters.keys().copied().collect();
        for h in headers {
            let Some(&(_, last, _)) = loops.iter().find(|l| l.0 == h) else {
                continue;
            };
            let blk = self.block[h].expect("reachable");
            let phi = self.s.m.phi(blk, self.s.t.i1);
            self.parked.insert(h, (phi, Vec::new(), last));
        }
    }

    /// `guard` with the lanes parked in every loop holding block `b` left
    /// out.
    pub(super) fn unparked(&mut self, b: usize, guard: Option<V>) -> Option<V> {
        let parked: Vec<V> = self
            .parked
            .iter()
            .filter(|&(&h, &(_, _, last))| self.rpo[h] <= self.rpo[b] && self.rpo[b] <= self.rpo[last])
            .map(|(_, &(phi, _, _))| phi)
            .collect();
        let mut guard = guard;
        for phi in parked {
            let c = self.cur;
            let yes = self.s.m.bool(true);
            let running = self.s.m.bin(c, Bin::Xor, phi, yes);
            guard = Some(match guard {
                Some(g) => self.s.m.bin(c, Bin::And, g, running),
                None => running,
            });
        }
        guard
    }

    /// A group barrier (`s_barrier`): thread-group sync with a shared-memory
    /// fence, and a UAV fence when the program writes UAVs (the waits before
    /// an s_barrier make those writes visible on the console).
    pub(super) fn group_barrier(&mut self, block: usize) {
        let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
        let mode = self.s.m.i32(mode);
        let void = self.s.t.void;
        self.s.op(block, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Group barriers per lane-dependent branch edge, so that every lane
    /// of a group arrives at a barrier the same number of times. The
    /// console runs `s_barrier` per wave: a branch skips a block only when
    /// no lane of the wave needs it, and the wave runs the block, barrier
    /// included, with the other lanes masked off. On the host each lane
    /// takes its own branch, so lanes skipping a block with a barrier never
    /// arrive at it and the lanes inside wait for them forever (a GPU hang;
    /// the title's compute_graph_ops, sky scattering and blendshape
    /// programs, in hung submissions). A host barrier counts arrivals, not
    /// instructions, so the lighter side of such a branch gets as many
    /// barriers as the sides' counts to the program's end differ.
    ///
    /// Counts are taken over forward edges, so inside a loop with uniform
    /// exits they balance each iteration (every lane runs as many).
    /// Refused: a loop with a barrier and a lane-dependent exit, counts
    /// that vary along uniform paths inside a lane-dependent branch, and
    /// indirect jumps, where no count holds. Programs without barriers
    /// emission keeps get nothing.
    // shortcut: the extra barriers are at the start of the lighter edge, so
    // a lane there may read LDS after the join before the heavier side's
    // last writes before the join show (on the console those lanes are in
    // lockstep with the others).
    /// The group barriers emission keeps in IR block `b`.
    pub(super) fn block_barriers(&self, b: usize) -> u32 {
        if self.no_barrier.contains(&b) {
            return 0;
        }
        self.x.p.blocks[b]
            .insts
            .iter()
            .filter(|i| i.name() == "s_barrier")
            .count() as u32
    }

    /// Loops: back edge l -> h, the blocks between them in reverse
    /// postorder (an over-approximation), and whether one holds a barrier.
    pub(super) fn barrier_loops(&self, order: &[usize]) -> Vec<(usize, usize, bool)> {
        let st = self.x.st;
        let back = |from: usize, to: usize| self.rpo[to] <= self.rpo[from];
        order
            .iter()
            .flat_map(|&l| {
                let succ: Vec<usize> = match st.exits[l] {
                    LaneExit::Jump(t) => vec![t],
                    LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
                    _ => vec![],
                };
                succ.into_iter()
                    .filter(move |&h| self.block[h].is_some() && back(l, h))
                    .map(move |h| (h, l))
                    .collect::<Vec<_>>()
            })
            .map(|(h, l)| {
                let barrier = order.iter().any(|&x| {
                    self.rpo[h] <= self.rpo[x] && self.rpo[x] <= self.rpo[l] && self.block_barriers(x) > 0
                });
                (h, l, barrier)
            })
            .collect()
    }

    /// Per-lane branches leaving (or going round) a loop with a group
    /// barrier that are decided for the whole wave instead, as the console
    /// decides `s_cbranch_vccz`/`execz` (taken if the mask is empty: the
    /// lanes' condition in all active lanes) and `vccnz`/`execnz` (in any):
    /// by IR block, true for all. Per lane, lanes leaving such a loop at
    /// different iterations leave the others waiting at its barrier (the
    /// title's compute_graph_ops CS_FormulaOperation loops until any lane
    /// reaches its own count). Only where the wave's decision makes lanes
    /// leave the loop sooner, never stay longer: what follows the loop is
    /// reached by every lane per lane too, so its guards hold, and no lane
    /// runs an iteration its guards were not made for. Other loop exits
    /// keep the barrier counting ([`Lower::barrier_balance`]).
    pub(super) fn wave_loop_exits(&self, order: &[usize]) -> HashMap<usize, bool> {
        let st = self.x.st;
        let mut out = HashMap::new();
        let loops = self.barrier_loops(order);
        if !loops.iter().any(|l| l.2) {
            return out;
        }
        let back = |from: usize, to: usize| self.rpo[to] <= self.rpo[from];
        let inside = |(h, l): (usize, usize), x: usize| self.rpo[h] <= self.rpo[x] && self.rpo[x] <= self.rpo[l];
        for &b in order {
            let LaneExit::Branch { when, taken, not_taken } = st.exits[b] else {
                continue;
            };
            let lane = when != Bool::TRUE
                && when != Bool::FALSE
                && !st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_)));
            if !lane {
                continue;
            }
            let name = self.x.p.blocks[b].insts.last().map_or("", |i| i.name());
            // Taken when the mask is empty (all lanes), or when it is not (any).
            let all = match name {
                "s_cbranch_vccz" | "s_cbranch_execz" | "s_cbranch_scc0" => true,
                "s_cbranch_vccnz" | "s_cbranch_execnz" | "s_cbranch_scc1" => false,
                _ => continue,
            };
            let mut safe = None;
            for &(h, l, barrier) in &loops {
                let r = (h, l);
                if !inside(r, b) || !barrier {
                    continue;
                }
                // The side going on in the loop (round it included).
                let (ti, ni) = (inside(r, taken), inside(r, not_taken));
                let stays_taken = match (ti, ni) {
                    (true, false) => true,
                    (false, true) => false,
                    _ => continue,
                };
                // All lanes must go on (taken, all) or none may leave
                // (not taken, any): the wave goes on only where each lane
                // would.
                let ok = stays_taken == all;
                if !ok || back(b, if stays_taken { not_taken } else { taken }) {
                    safe = Some(false);
                    break;
                }
                safe = Some(true);
            }
            if safe == Some(true) {
                out.insert(b, all);
            }
        }
        out
    }

    /// For a program whose barriers cannot be balanced: every per-lane
    /// branch on a lane mask, decided for the whole wave as the console
    /// decides it (by IR block, true where all active lanes must take it).
    /// With every vector instruction then guarded by its EXEC
    /// ([`Lower::guard`]), the wave walks every path any of its lanes needs
    /// with the others masked off, and its lanes arrive at each barrier
    /// together. Costs guards the per-lane form leaves out.
    // shortcut: a console wave is 64 lanes; a wave64 program run in host
    // waves of 32 decides per half, and halves whose lanes disagree still
    // part at a barrier.
    pub(super) fn console_branches(&self, order: &[usize]) -> HashMap<usize, bool> {
        let st = self.x.st;
        let mut out = HashMap::new();
        for &b in order {
            let LaneExit::Branch { when, .. } = st.exits[b] else {
                continue;
            };
            let when = st.whens[b].unwrap_or(when);
            // In a console wave split in two (`Lower::split_console`), a
            // scalar condition can differ between the halves too: SCC of a
            // lane mask (`s_andn2_b64 s, s, exec` ending a loop while any
            // lane is left) sees each half's own lanes. Every branch is
            // decided for the console wave there; a truly uniform one
            // gives both halves the same answer anyway.
            let lane = when != Bool::TRUE
                && when != Bool::FALSE
                && (self.split_console
                    || !st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))));
            if !lane {
                continue;
            }
            let all = match self.x.p.blocks[b].insts.last().map_or("", |i| i.name()) {
                "s_cbranch_vccz" | "s_cbranch_execz" | "s_cbranch_scc0" => true,
                "s_cbranch_vccnz" | "s_cbranch_execnz" | "s_cbranch_scc1" => false,
                _ => continue,
            };
            out.insert(b, all);
        }
        out
    }

    pub(super) fn barrier_balance(
        &self,
        order: &[usize],
    ) -> Result<HashMap<(usize, usize), u32>, (usize, Why)> {
        let p = self.x.p;
        let st = self.x.st;
        let barriers = |b: usize| self.block_barriers(b);
        let total: u32 = order.iter().map(|&b| barriers(b)).sum();
        let mut extra = HashMap::new();
        if total == 0 {
            return Ok(extra);
        }
        let offset = |b: usize| {
            p.blocks[b]
                .insts
                .last()
                .map_or(p.blocks[b].offset, |i| i.machine.offset)
        };
        // A branch decided for the whole wave (`Lower::wave_loop_exits`) is
        // a uniform one here.
        let lane = |b: usize, when: Bool| {
            when != Bool::TRUE
                && when != Bool::FALSE
                && !self.wave_branches.contains_key(&b)
                && !st
                    .bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(_)))
        };
        let back = |from: usize, to: usize| self.rpo[to] <= self.rpo[from];
        // Loops holding a barrier may only be left uniformly.
        let loops = self.barrier_loops(order);
        let inside = |(h, l): (usize, usize), x: usize| self.rpo[h] <= self.rpo[x] && self.rpo[x] <= self.rpo[l];
        // Whether a per-lane branch leaves a loop (or goes round it): a lane
        // going on there leaves it later the same way, so its side needs no
        // count of its own. Err if the loop holds a barrier.
        let leaves_loop = |b: usize, taken: usize, not_taken: usize| -> Result<Option<usize>, (usize, Why)> {
            let mut out = None;
            // Every loop the branch leaves or goes round, not just the first.
            for &(h, l, barrier) in &loops {
                let r = (h, l);
                if !inside(r, b) {
                    continue;
                }
                let (ti, ni) = (inside(r, taken) && !back(b, taken), inside(r, not_taken) && !back(b, not_taken));
                if ti && ni {
                    continue;
                }
                if barrier {
                    return Err((offset(b), "a loop with a group barrier left per lane"));
                }
                // The side leaving the loop (or not going round it).
                out = Some(if ti { not_taken } else { taken });
            }
            Ok(out)
        };
        // Barriers from each block's start to the program's end over
        // forward edges: the counts possible along uniform paths (every
        // lane of a group takes the same one), sorted.
        // shortcut: two sides of a per-lane branch whose sets differ by a
        // constant are taken to differ by it along every uniform path; a
        // side with a uniform choice of its own before the join that the
        // other lacks could pair counts otherwise.
        const MOST: usize = 16;
        let shift = |set: &[u32], by: u32| -> Vec<u32> { set.iter().map(|c| c + by).collect() };
        let mut to_end: HashMap<usize, Vec<u32>> = HashMap::new();
        for &b in order.iter().rev() {
            let here = barriers(b);
            let after = |to: usize| -> Vec<u32> {
                if self.block[to].is_none() || back(b, to) {
                    vec![0]
                } else {
                    to_end.get(&to).cloned().unwrap_or_else(|| vec![0])
                }
            };
            let count = match st.exits[b] {
                LaneExit::End => vec![here],
                LaneExit::Indirect => {
                    return Err((offset(b), "an indirect jump in a program with group barriers"));
                }
                LaneExit::Jump(t) => shift(&after(t), here),
                LaneExit::Branch { when, taken, not_taken } => {
                    let (x, y) = (after(taken), after(not_taken));
                    let out = if lane(b, when) {
                        leaves_loop(b, taken, not_taken)?
                    } else {
                        None
                    };
                    if let Some(out) = out {
                        shift(&after(out), here)
                    } else if lane(b, when) {
                        // The same counts up to a constant: the lighter side
                        // gets that many more.
                        let (lo, hi, light) = if x[0] <= y[0] { (&x, &y, taken) } else { (&y, &x, not_taken) };
                        let n = hi[0] - lo[0];
                        // A side meeting no more barriers on any path goes
                        // on to its end and exits; a host barrier does not
                        // wait for threads that exited (an early return
                        // before a barrier, which the title's meshlet
                        // culling does per lane, works on the host too).
                        // shortcut: D3D12 leaves barriers after a thread's
                        // exit undefined; NVIDIA and AMD count exited
                        // threads out.
                        if lo[..] == [0] && (lo.len() != hi.len() || shift(lo, n) != *hi) {
                            let mut both = shift(hi, here);
                            both.dedup();
                            to_end.insert(b, both);
                            continue;
                        }
                        if lo.len() != hi.len() || shift(lo, n) != *hi {
                            return Err((
                                offset(b),
                                "group barriers counted differently along uniform paths inside a per-lane branch",
                            ));
                        }
                        if n > 0 {
                            if back(b, light) {
                                return Err((offset(b), "a loop with a group barrier left per lane"));
                            }
                            extra.insert((b, light), n);
                        }
                        shift(hi, here)
                    } else {
                        let mut both: Vec<u32> = x.iter().chain(&y).map(|c| c + here).collect();
                        both.sort_unstable();
                        both.dedup();
                        if both.len() > MOST {
                            return Err((offset(b), "too many uniform paths with group barriers"));
                        }
                        both
                    }
                }
            };
            to_end.insert(b, count);
        }
        Ok(extra)
    }


    /// Whether no lane entering IR block `b` from its `n`th predecessor
    /// reads what `reg` holds there before something overwrites it. The
    /// lane is followed along its per-lane branches, with the phis it
    /// passes taking that path's values; a branch whose compares have
    /// constant operands on the path is followed one way. Scalar arithmetic
    /// on the value passes it on; any other read of it, or a branch or guard
    /// that depends on it, makes it live. Paths meeting at a block with the
    /// same registers holding it go on with what they agree on. This is the
    /// console compiler's flag after a join (`if (c) { x = load; f = .. }
    /// if (f) use(x)`, with `f` 0 on the other path), where the register
    /// that holds `x` holds a stale value, often a lane mask, on that path.
    pub(super) fn dead_on_edge(&self, b: usize, n: usize, reg: Reg) -> bool {
        let p = self.x.p;
        let st = self.x.st;
        // Values computed from the stale one on some path followed.
        let mut tainted = HashSet::new();
        let mut work = vec![(b, n, HashMap::new(), BTreeSet::from([reg]))];
        // Per block reached and registers holding tainted values on entry:
        // the phi values known there, over the paths so far.
        let mut at: HashMap<(usize, BTreeSet<Reg>), HashMap<Value, u32>> = HashMap::new();
        let mut visits = 0;
        while let Some((b, n, mut env, mut regs)) = work.pop() {
            let block = &p.blocks[b];
            let known: Vec<_> = block
                .phis
                .iter()
                .map(|phi| (phi.value, self.konst(&env, phi.args[n])))
                .collect();
            for (v, k) in known {
                match k {
                    Some(k) => env.insert(v, k),
                    None => env.remove(&v),
                };
            }
            for phi in &block.phis {
                if regs.contains(&phi.reg) {
                    tainted.insert(phi.value);
                }
            }
            let key = (b, regs.clone());
            if let Some(old) = at.get(&key) {
                let agreed: HashMap<Value, u32> = old
                    .iter()
                    .filter(|&(v, k)| env.get(v) == Some(k))
                    .map(|(&v, &k)| (v, k))
                    .collect();
                if agreed.len() == old.len() {
                    continue;
                }
                env = agreed;
            }
            at.insert(key, env.clone());
            visits += 1;
            if visits > 4 * p.blocks.len() + 64 {
                return false;
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if regs.is_empty() {
                    break;
                }
                let guard = st.guards[b][i];
                if guard.is_some_and(|g| self.touches(g, &env, &tainted)) {
                    return false;
                }
                // The closing branch is the lane's exit, tested below.
                if inst.reads.iter().any(|r| regs.contains(&r.0))
                    && !inst.name().starts_with("s_cbranch")
                {
                    // Scalar arithmetic only passes the value on.
                    let pure = matches!(inst.fields, Fields::Sop { .. })
                        && inst.write(Reg::S(EXEC)).is_none()
                        && !inst.name().starts_with("s_setreg");
                    if !pure {
                        return false;
                    }
                    for &(r, w) in &inst.writes {
                        regs.insert(r);
                        tainted.insert(w);
                    }
                } else {
                    // Only VGPR writes keep their old value under a lane guard;
                    // SGPR and SCC writes (e.g. `v_readfirstlane_b32`, `v_cmp_*`)
                    // always overwrite the register unconditionally.
                    for &(r, _) in &inst.writes {
                        if guard.is_none() || !matches!(r, Reg::V(_)) {
                            regs.remove(&r);
                        }
                    }
                }
            }
            if regs.is_empty() {
                continue;
            }
            let into = |t: usize| p.blocks[t].preds.iter().position(|&q| q == b);
            let mut go = |t: usize, env, regs| match into(t) {
                Some(n) => {
                    work.push((t, n, env, regs));
                    true
                }
                None => false,
            };
            let ok = match st.exits[b] {
                LaneExit::End => true,
                LaneExit::Indirect => false,
                LaneExit::Jump(t) => go(t, env, regs),
                LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } => {
                    let (no, yes) = self.can_be(when, &env, &mut HashMap::new());
                    !(no && yes && self.touches(when, &env, &tainted))
                        && (!yes || go(taken, env.clone(), regs.clone()))
                        && (!no || go(not_taken, env, regs))
                }
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Whether deciding `b`, given the values in `env`, may test an atom
    /// of a `tainted` value.
    pub(super) fn touches(&self, b: Bool, env: &HashMap<Value, u32>, tainted: &HashSet<Value>) -> bool {
        let mut stack = vec![b];
        let mut seen = HashSet::new();
        while let Some(b) = stack.pop() {
            let Some((atom, lo, hi)) = self.x.st.bools.node(b) else {
                continue;
            };
            if !seen.insert(b) {
                continue;
            }
            let v = match atom {
                Atom::Lane(v)
                | Atom::Uniform(v)
                | Atom::Phi(v)
                | Atom::Quad(v)
                | Atom::Opaque(v) => Some(v),
                Atom::Constant(_) => None,
            };
            if v.is_some_and(|v| tainted.contains(&v)) {
                return true;
            }
            match v.and_then(|v| self.compared(env, v)) {
                Some(true) => stack.push(hi),
                Some(false) => stack.push(lo),
                None => stack.extend([lo, hi]),
            }
        }
        false
    }

    /// Whether `b` can be false, and true, given the values in `env`.
    pub(super) fn can_be(
        &self,
        b: Bool,
        env: &HashMap<Value, u32>,
        memo: &mut HashMap<Bool, (bool, bool)>,
    ) -> (bool, bool) {
        if b == Bool::TRUE || b == Bool::FALSE {
            return (b == Bool::FALSE, b == Bool::TRUE);
        }
        if let Some(&r) = memo.get(&b) {
            return r;
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let v = match atom {
            Atom::Lane(v) | Atom::Uniform(v) => Some(v),
            _ => None,
        };
        let r = match v.and_then(|v| self.compared(env, v)) {
            Some(true) => self.can_be(hi, env, memo),
            Some(false) => self.can_be(lo, env, memo),
            None => {
                let (a, c) = self.can_be(lo, env, memo);
                let (d, e) = self.can_be(hi, env, memo);
                (a || d, c || e)
            }
        };
        memo.insert(b, r);
        r
    }

    /// The block, instruction index, and instruction writing `v`.
    pub(super) fn def_site(&self, v: Value) -> Option<(usize, usize, &'a Inst)> {
        let b = *self.def_block.get(&v)?;
        let (i, inst) = self.x.p.blocks[b]
            .insts
            .iter()
            .enumerate()
            .find(|(_, inst)| inst.writes.iter().any(|w| w.1 == v))?;
        Some((b, i, inst))
    }

    /// The instruction writing `v`.
    pub(super) fn def_inst(&self, v: Value) -> Option<&'a Inst> {
        self.def_site(v).map(|(_, _, inst)| inst)
    }

    /// The constant `v` holds, `env` giving phi values: unguarded moves of
    /// unmodified constants.
    pub(super) fn konst(&self, env: &HashMap<Value, u32>, v: Value) -> Option<u32> {
        if let Some(&k) = env.get(&v) {
            return Some(k);
        }
        let (b, i, inst) = self.def_site(v)?;
        if self.x.st.guards[b][i].is_some() {
            return None;
        }
        let op = match &inst.fields {
            Fields::Valu(w)
                if inst.name() == "v_mov_b32"
                    && w.sdwa.is_none()
                    && w.dpp.is_none()
                    && w.abs == 0
                    && w.neg == 0
                    && !w.clamp
                    && w.omod == 0 =>
            {
                w.src.first()?
            }
            Fields::Sop { src, .. } if inst.name() == "s_mov_b32" => src.first()?,
            Fields::Sop { imm, .. } if inst.name() == "s_movk_i32" => {
                return Some(i32::from((*imm)? as i16) as u32);
            }
            _ => return None,
        };
        self.konst_operand(env, inst, op)
    }

    pub(super) fn konst_operand(&self, env: &HashMap<Value, u32>, inst: &Inst, op: &Operand) -> Option<u32> {
        match *op {
            Operand::Const(c) => crate::resource::inline(c),
            Operand::Literal(x) => Some(x),
            Operand::Scalar { code, .. } if code != NULL => {
                self.konst(env, inst.read(Reg::S(code))?)
            }
            Operand::Vector { reg, .. } => self.konst(env, inst.read(Reg::V(reg))?),
            _ => None,
        }
    }

    /// The lane's bit of the 32-bit integer compare writing `v` (a VALU
    /// result before EXEC masking, or SCC), when its operands are constants.
    pub(super) fn compared(&self, env: &HashMap<Value, u32>, v: Value) -> Option<bool> {
        let inst = self.def_inst(v)?;
        let name = inst.name();
        let rest = ["v_cmpx_", "v_cmp_", "s_cmp_"]
            .iter()
            .find_map(|p| name.strip_prefix(p))?;
        let (op, ty) = rest.rsplit_once('_')?;
        let (a, b) = match &inst.fields {
            Fields::Valu(w) if w.sdwa.is_none() && w.dpp.is_none() => {
                (w.src.first()?, w.src.get(1)?)
            }
            Fields::Sop { src, .. } => (src.first()?, src.get(1)?),
            _ => return None,
        };
        let (a, b) = (
            self.konst_operand(env, inst, a)?,
            self.konst_operand(env, inst, b)?,
        );
        let (sa, sb) = (a as i32, b as i32);
        Some(match (op, ty) {
            ("f", _) => false,
            ("t", _) => true,
            ("eq", "u32" | "i32") => a == b,
            ("ne" | "lg", "u32" | "i32") => a != b,
            ("lt", "u32") => a < b,
            ("le", "u32") => a <= b,
            ("gt", "u32") => a > b,
            ("ge", "u32") => a >= b,
            ("lt", "i32") => sa < sb,
            ("le", "i32") => sa <= sb,
            ("gt", "i32") => sa > sb,
            ("ge", "i32") => sa >= sb,
            _ => return None,
        })
    }

    /// The incoming edges (phi, predecessor index) over which a value can
    /// reach a read. A phi that only another read phi consumes is live too
    /// (else a value with no DXIL form reaching a read through two joins
    /// would be replaced by zero), but only over edges a lane can take on
    /// its way there: each step back adds the branch a lane took on that
    /// edge, and a contradiction (the lane that skipped an arm is not one
    /// that ran it) leaves the edge dead. A back edge forgets the branches
    /// (the loop redefines their conditions). Fewer conditions are weaker,
    /// so keeping only the last few stays sound.
    pub(super) fn live_edges(&self) -> HashSet<(Value, usize)> {
        const MAX_KNOWN: usize = 4;
        /// What is known of the lanes along a path: the last few branch
        /// conditions taken, in order (a value, not a heap allocation:
        /// shaders with control flow push thousands of them).
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        struct Known {
            conditions: [(Bool, bool); MAX_KNOWN],
            len: usize,
        }
        impl Known {
            fn get(&self) -> &[(Bool, bool)] {
                &self.conditions[..self.len]
            }
            /// With `c` after the others, the oldest dropped past the limit.
            fn and(mut self, c: (Bool, bool)) -> Known {
                if self.len == MAX_KNOWN {
                    self.conditions.copy_within(1.., 0);
                    self.len -= 1;
                }
                self.conditions[self.len] = c;
                self.len += 1;
                self
            }
        }
        let none = Known {
            conditions: [(Bool::FALSE, false); MAX_KNOWN],
            len: 0,
        };
        let (p, st) = (self.x.p, &self.x.st);
        let mut work: Vec<(Value, Known)> = self.read_phis.iter().map(|&v| (v, none)).collect();
        let mut live = HashSet::new();
        let mut seen = HashMap::<Value, Vec<Known>>::new();
        // Whether some lane meets each set of conditions: paths into a join
        // ask again for the same sets.
        let mut meets = HashMap::<Known, bool>::new();
        while let Some((v, known)) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            let done = seen.entry(v).or_default();
            // Known of fewer lanes than before: nothing new is live.
            if done
                .iter()
                .any(|d| d.get().iter().all(|c| known.get().contains(c)))
            {
                continue;
            }
            done.push(known);
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                if self.tail[q].is_none() {
                    continue;
                }
                let mut k = none;
                // Forward edges out of a block that is no loop's latch (a
                // loop's exit is also taken at its iteration limit).
                if let LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } = st.exits[q]
                    && taken != not_taken
                    && self.rpo[taken] > self.rpo[q]
                    && self.rpo[not_taken] > self.rpo[q]
                {
                    k = known.and((when, taken == b));
                    if !*meets.entry(k).or_insert_with(|| st.bools.meet(k.get())) {
                        continue;
                    }
                } else if self.rpo[q] < self.rpo[b] {
                    k = known;
                }
                live.insert((v, n));
                let arg = p.blocks[b].phis[i].args[n];
                if self.phi_at.contains_key(&arg) {
                    work.push((arg, k));
                }
            }
        }
        live
    }

    /// An unguarded indexed write replaces the incoming value before a
    /// consumer can observe it, either on this edge or in a counted loop.
    pub(super) fn overwritten_movreld_edge(&self, b: usize, phi: &Phi, edge: usize) -> Option<()> {
        let Reg::V(reg) = phi.reg else { return None };
        let p = self.x.p;
        let block = &p.blocks[b];
        if self.sections.contains_key(&b)
            || self.sections.contains_key(&block.preds[edge])
            || p.blocks
                .iter()
                .flat_map(|bb| &bb.phis)
                .any(|ph| ph.args.contains(&phi.value))
        {
            return None;
        }
        let mut uses = p.blocks.iter().enumerate().flat_map(|(bb, block)| {
            block.insts.iter().enumerate().filter_map(move |(i, inst)| {
                inst.reads
                    .iter()
                    .any(|r| r.1 == phi.value)
                    .then_some((bb, i, inst))
            })
        });
        let (bb, i, mov) = uses.next()?;
        if uses.next().is_some()
            || bb != b
            || self.x.st.guards[b][i].is_some()
            || mov.name() != "v_movreld_b32"
            || mov.read(phi.reg) != Some(phi.value)
        {
            return None;
        }
        let Fields::Valu(v) = &mov.fields else {
            return None;
        };
        let (
            Some(Operand::Vector {
                reg: base,
                dwords: 1,
            }),
            [
                Operand::Vector {
                    reg: src,
                    dwords: 1,
                },
            ],
        ) = (v.dst, v.src.as_slice())
        else {
            return None;
        };
        if v.dpp.is_some()
            || v.sdwa.is_some()
            || mov.read(Reg::V(*src))? == phi.value
            || mov.write(phi.reg).is_none()
        {
            return None;
        }
        let index = u32::from(reg.checked_sub(base)?);
        // Follow M0 to a copy of a scalar phi in this block, then the
        // constant written on the corresponding predecessor.
        let (copy, [src]) = scalar_def(block, mov.read(Reg::S(crate::ir::M0))?, "s_mov_b32")?;
        let scalar = sgpr_operand(copy, src)?;
        let index_phi = block.phis.iter().find(|ph| ph.value == scalar)?;
        let (_, [initial]) = scalar_def(
            &p.blocks[block.preds[edge]],
            index_phi.args[edge],
            "s_mov_b32",
        )?;
        let known = scalar_constant(initial)?;
        if index == known {
            return Some(());
        }
        if index < known {
            return None; // Reaching this slot would require the index to wrap.
        }
        self.counted_movreld_index(b, edge, phi, index_phi, mov, index)
    }

    /// An unguarded move fills one more register on every iteration.
    /// Values from before the loop can only survive in slots the index skips.
    pub(super) fn counted_movreld_index(
        &self,
        b: usize,
        edge: usize,
        phi: &Phi,
        indices: &Phi,
        mov: &Inst,
        slot: u32,
    ) -> Option<()> {
        let p = self.x.p;
        let st = self.x.st;
        let block = &p.blocks[b];
        let LaneExit::Branch {
            when,
            taken: latch,
            not_taken: exit,
        } = st.exits[b]
        else {
            return None;
        };
        let back = block.preds.iter().position(|&q| q == latch)?;
        if block.preds.len() != 2
            || back == edge
            || b == latch
            || b == exit
            || exit == latch
            || self.sections.contains_key(&latch)
            || st.exits[latch] != LaneExit::Jump(b)
            || p.blocks[latch].preds != [b]
            || p.blocks[exit].preds != [b]
            || st.exits[exit] != LaneExit::End
            || self.rpo[b] >= self.rpo[latch]
            || self.rpo[block.preds[edge]] >= self.rpo[b]
        {
            return None;
        }
        // The indexed result may only go around the loop as its own old
        // value or be consumed in the normal exit block.
        let result = mov.write(phi.reg)?;
        if phi.args[back] != result {
            return None;
        }
        for (bb, block) in p.blocks.iter().enumerate() {
            if bb != exit
                && block
                    .insts
                    .iter()
                    .any(|inst| inst.reads.iter().any(|r| r.1 == result))
            {
                return None;
            }
            if block.phis.iter().any(|other| {
                other
                    .args
                    .iter()
                    .enumerate()
                    .any(|(n, &arg)| arg == result && (other.value != phi.value || n != back))
            }) {
                return None;
            }
        }
        // Follow the latch copy to the unit increment, then require that
        // this very increment is compared for equality to the exit bound.
        let (copy, [src]) = scalar_def(&p.blocks[latch], indices.args[back], "s_mov_b32")?;
        let next = sgpr_operand(copy, src)?;
        let (increment, [from, step]) = scalar_def(block, next, "s_add_i32")?;
        if sgpr_operand(increment, from)? != indices.value || scalar_constant(step)? != 1 {
            return None;
        }
        let (Atom::Uniform(scc), Bool::TRUE, Bool::FALSE) = st.bools.node(when)? else {
            return None;
        };
        let branch = block.insts.last()?;
        if branch.name() != "s_cbranch_scc0" || branch.read(Reg::Scc) != Some(scc) {
            return None;
        }
        let (cmp, [tested, bound]) = scalar_def(block, scc, "s_cmp_eq_u32")?;
        if sgpr_operand(cmp, tested)? != next {
            return None;
        }
        let limit = scalar_constant(bound)?;
        // The initial index is below this slot. The loop must pass it before exit.
        (slot < limit).then_some(())
    }
}
