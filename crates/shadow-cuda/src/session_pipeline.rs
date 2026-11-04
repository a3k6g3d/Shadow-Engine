//! The two-group pipeline of a verify pass (`--pipeline`).
//!
//! A verify window of m tokens is cut into two groups, A = rows [0, ma) and B = the rest.  Per layer the GPU stream runs
//!
//! ```text
//! pre(l, A)  pre(l, B)  |  post(l, A)  pre(l + 1, A)  |  post(l, B)  pre(l + 1, B)  |  ...
//! ```
//!
//! where `pre` is a group's mixer and router and `post` its experts, combine and hyper-connection write.  The host's
//! work for a group's experts (reading its router logits, choosing and issuing the copies, the file reads, the CPU
//! experts, the grouped launches) happens while the GPU runs the OTHER group's mixer, so that serial chain stops
//! costing a layer's idle GPU time.  A group's mixer needs the other group's mixer of the same layer (K/V, the
//! recurrence's state), never its experts, so no token waits for anything it did not wait for before and every token's
//! arithmetic is unchanged.
//!
//! How a group runs on a row range: `with_rows` shifts the base pointer of every per-token batch buffer by the group's
//! first row for the duration of the call, so the batched stages (`hc_read_batch`, `delta_batch`, `attention_batch`,
//! `moe_batch`, ...) run on rows 0..mc of a view that starts at row j0 of the buffers.  What the host needs from the
//! GPU between `pre` and `post` (router logits, and with CPU experts the MoE input) is copied into the pinned mapped
//! region right after `pre` and fenced by an event the host waits on later (`issue_router_read` / `wait_router_read`),
//! not by stream order, so the host can read group A's logits while group B's mixer is still running.  Buffers the
//! next group could overwrite while the previous group's experts still read them are fenced on the GPU: the copy
//! stream waits for an event recorded after the previous `post` (`group_event`) and the host syncs the copy stream only
//! before a file read that may reuse a pinned slot.

use super::{row, DeviceBuffer, GpuSession, MAX_VERIFY};
use crate::kernels::Arg;
use shadow_model::forward::ple_rows;

impl<'c, 'm> GpuSession<'c, 'm> {
    /// Whether this batch runs as two token groups.
    pub(super) fn pipeline_ok(&self, m: usize) -> bool {
        self.pipeline
            && self.replay.on.get()
            && (2..=MAX_VERIFY).contains(&m)
            && self.overlap
            && self.prof.is_none()
            && self.trace.is_none()
            && self.stop_after.is_none()
            && self.prefetch.is_none()
            && self.pregate.is_none()
            && self.cache.slots > 0
            && self.fuse & 1 == 0
            && !self.timing_no_sync
            && self.ltime.is_none()
            && self.m.cfg.arch != "flint"
    }

    /// Whether the host needs the MoE input of the group's tokens next to the router logits (CPU experts).
    pub(super) fn want_u_rows(&self, mc: usize) -> bool {
        self.cpu.is_some() && self.replay.on.get() && (1..=MAX_VERIFY).contains(&mc) && self.overlap && self.prof.is_none()
    }

    /// Moves the base pointer of every per-token batch buffer by `sign` x j0 rows (and halves / restores the plan table
    /// buffer, whose lower half serves group A and upper half group B).
    fn shift_rows(&mut self, j0: usize, sign: i64) {
        let c = &self.m.cfg;
        let (h, cs, k) = (c.hidden, c.streams, c.n_expert_used);
        let (kd, vd) = (c.ssm_k_heads * c.ssm_head_dim, c.ssm_v_heads * c.ssm_head_dim);
        let (qd, kvd) = (c.n_head * c.head_dim, c.n_kv_head * c.head_dim);
        let (vh, ch2) = (c.ssm_v_heads, 2 * kd + vd);
        let (iqd, ne, sff, ff) = (c.idx_heads * c.idx_dim, c.n_expert, c.shared_ff, c.expert_ff);
        let rank = c.hc_rank;
        let mv = |b: &mut DeviceBuffer, floats: usize| {
            let d = (j0 * floats * 4) as u64;
            if sign > 0 {
                b.ptr += d;
            } else {
                b.ptr -= d;
            }
        };
        let bb = &mut self.bb;
        mv(&mut bb.r, h * cs);
        mv(&mut bb.rn, h * cs);
        mv(&mut bb.u, h);
        mv(&mut bb.g, cs);
        mv(&mut bb.y, h);
        mv(&mut bb.down, rank);
        mv(&mut bb.up, h * cs);
        mv(&mut bb.qkv, ch2);
        mv(&mut bb.x, ch2);
        mv(&mut bb.z, vd);
        mv(&mut bb.beta, vh);
        mv(&mut bb.alpha, vh);
        mv(&mut bb.o, vd);
        mv(&mut bb.qg, 2 * qd);
        mv(&mut bb.q, qd);
        mv(&mut bb.gate, qd);
        mv(&mut bb.kk, kvd);
        mv(&mut bb.iq, iqd);
        mv(&mut bb.ao, qd);
        mv(&mut bb.rl, ne);
        mv(&mut bb.sgl, 1);
        mv(&mut bb.sg, sff);
        mv(&mut bb.su, sff);
        mv(&mut bb.sa, sff);
        mv(&mut bb.so, h);
        mv(&mut bb.eg, k * ff);
        mv(&mut bb.act, k * ff);
        mv(&mut bb.eo, k * h);
        mv(&mut self.replay.qkv, ch2);
        mv(&mut self.replay.ba, vh);
        // the plan tables: A the lower half of the buffer, B the upper
        if sign > 0 {
            bb.gtab.bytes /= 2;
            if j0 > 0 {
                bb.gtab.ptr += bb.gtab.bytes as u64;
            }
        } else {
            if j0 > 0 {
                bb.gtab.ptr -= bb.gtab.bytes as u64;
            }
            bb.gtab.bytes *= 2;
        }
    }

    /// Runs `f` on the view of the batch buffers that starts at row j0.
    fn with_rows<R>(&mut self, j0: usize, f: impl FnOnce(&mut Self) -> R) -> R {
        self.shift_rows(j0, 1);
        self.row_shift = j0;
        let r = f(self);
        self.row_shift = 0;
        self.shift_rows(j0, -1);
        r
    }

    /// Queues the copy of the group's router logits (and MoE input) to the mapped region and the event that fences it.
    fn issue_router_read(&mut self, slot: usize, mc: usize, want_u: bool) {
        let c = &self.m.cfg;
        let (ne, h) = (c.n_expert, c.hidden);
        let half = self.mapped.len() / 2;
        let base = self.mapped.dev_ptr + (slot * half) as u64;
        self.k1("copy", mc * ne, &[Arg::Ptr(base), Arg::Ptr(self.bb.rl.ptr), Arg::U32((mc * ne) as u32)]);
        if want_u {
            self.k1("copy", mc * h, &[Arg::Ptr(base + (mc * ne * 4) as u64), Arg::Ptr(self.bb.u.ptr), Arg::U32((mc * h) as u32)]);
        }
        let ev = if slot == 0 { &self.mapped_event } else { &self.mapped_event2 };
        ev.record(None).unwrap();
        self.ticket_stage[slot] = self.stage_at.get();
    }

    /// Waits for the copy `issue_router_read` queued and returns the group's router logits (one row each) and MoE input.
    pub(super) fn wait_router_read(&mut self, slot: usize, mc: usize, want_u: bool) -> (Vec<Vec<f32>>, Option<Vec<f32>>) {
        let c = &self.m.cfg;
        let (ne, h) = (c.n_expert, c.hidden);
        let ev = if slot == 0 { &self.mapped_event } else { &self.mapped_event2 };
        ev.synchronize().unwrap();
        if self.stage_at.get() == self.ticket_stage[slot] {
            self.stage_at.set(0); // nothing was staged since the copy was queued: every earlier staged upload has run
        }
        let half = self.mapped.len() / 2;
        // SAFETY: the region is `half` bytes a slot, the copies wrote mc * (ne [+ h]) floats at the slot's start, and the
        // event above says they are done.
        let all = unsafe {
            let base = (self.mapped.as_slice().as_ptr() as *const f32).add(slot * half / 4);
            std::slice::from_raw_parts(base, mc * ne + if want_u { mc * h } else { 0 })
        };
        let logits = all[..mc * ne].chunks(ne).map(|r| r.to_vec()).collect();
        (logits, want_u.then(|| all[mc * ne..].to_vec()))
    }

    /// A group's mixer and router for layer l, and the queued copy of its router logits.
    #[allow(clippy::too_many_arguments)]
    fn pre_group(&mut self, l: usize, gi: usize, j0: usize, mc: usize, p0: usize, rn_ready: bool, graphs: bool) {
        let (ple, ch) = (self.m.cfg.ple_layer, self.m.cfg.hidden * self.m.cfg.streams);
        let attention = self.m.cfg.is_attention(l);
        let mut rn_ready = rn_ready;
        if l == ple {
            for j in j0..j0 + mc {
                let r = row(&self.bb.r, ch, j);
                self.ple_at(l, p0 + j + 1, r);
                if self.replay.on.get() {
                    self.k1("copy", ch, &[Arg::Ptr(self.replay.ple.ptr + (j * ch * 4) as u64), Arg::Ptr(self.b.gvn.ptr),
                                          Arg::U32(ch as u32)]);
                }
                self.k1("add", ch, &[Arg::Ptr(r), Arg::Ptr(self.b.gv.ptr), Arg::U32(ch as u32)]);
            }
            rn_ready = false;
        }
        let want_u = self.want_u_rows(mc);
        self.with_rows(j0, |s| {
            let on = s.replay.on.get();
            if graphs && !attention && l != ple {
                s.batch_graph_segment((l, rn_ready, mc, on, j0, s.bb.r.ptr), |s| {
                    s.hc_read_batch(&format!("blk.{l}.hc_attn"), true, rn_ready, mc);
                    s.delta_batch(l, mc);
                    s.hc_write_batch(Some(&format!("blk.{l}.hc_ffn_norm.weight")), mc);
                    s.hc_read_batch(&format!("blk.{l}.hc_ffn"), true, true, mc);
                    s.moe_router_batch(l, mc);
                });
            } else {
                s.hc_read_batch(&format!("blk.{l}.hc_attn"), true, rn_ready, mc);
                if attention {
                    s.attention_batch(l, p0 + j0, mc);
                } else {
                    s.delta_batch(l, mc);
                }
                s.hc_write_batch(Some(&format!("blk.{l}.hc_ffn_norm.weight")), mc);
                s.hc_read_batch(&format!("blk.{l}.hc_ffn"), true, true, mc);
                s.moe_router_batch(l, mc);
            }
            s.issue_router_read(gi, mc, want_u);
        });
    }

    /// A group's experts for layer l, the combine, and the hyper-connection write that ends the layer; whether the next
    /// layer's first norm came out of that write.
    fn post_group(&mut self, l: usize, gi: usize, j0: usize, mc: usize, logs: &mut [Vec<u16>]) -> bool {
        let c = &self.m.cfg;
        let next = if l + 1 == c.ple_layer {
            None
        } else if l + 1 < c.n_layer {
            Some(format!("blk.{}.hc_attn_norm.weight", l + 1))
        } else {
            Some("output_hc_norm.weight".to_string())
        };
        self.with_rows(j0, |s| {
            s.moe_batch(l, mc, &mut logs[j0..j0 + mc], true, Some(gi));
            s.hc_write_batch(next.as_deref(), mc);
            if s.dflash.is_some() {
                s.df_capture(l, s.bb.r.ptr, mc);
            }
        });
        next.is_some()
    }

    /// All the layers of a verify pass as two token groups; whether the output norm is already in `rn` for every row.
    pub(super) fn layers_pipelined(&mut self, m: usize, p0: usize, logs: &mut [Vec<u16>], graphs: bool) -> bool {
        let c = &self.m.cfg;
        let (nl, ple) = (c.n_layer, c.ple_layer);
        if ple < nl {
            let all: Vec<i64> = (0..m).flat_map(|j| ple_rows(c, &self.tokens[..p0 + j + 1])).collect();
            self.m.ple_prefetch(&all);
        }
        let ma = m.div_ceil(2);
        let groups = [(0usize, ma), (ma, m - ma)];
        self.pipe_active = true;
        self.pipelined_passes += 1;
        let mut rn = [false; 2];
        for (gi, &(j0, mc)) in groups.iter().enumerate() {
            self.pre_group(0, gi, j0, mc, p0, rn[gi], graphs);
        }
        for l in 0..nl {
            for (gi, &(j0, mc)) in groups.iter().enumerate() {
                rn[gi] = self.post_group(l, gi, j0, mc, logs);
                if l + 1 < nl {
                    self.pre_group(l + 1, gi, j0, mc, p0, rn[gi], graphs);
                }
            }
        }
        self.pipe_active = false;
        rn[0] && rn[1]
    }
}
