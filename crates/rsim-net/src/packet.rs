//! Packet-level transport.
//!
//! Each transfer is cut into packets. A packet is routed hop by hop:
//!
//! * **GPU egress** sprays packets over every next-hop link round-robin by
//!   packet index (NVLink address interleaving across the NVSwitch planes).
//! * **Switches** pick among equal-cost next hops either adaptively (the
//!   output with the least queued + in-flight bytes; NVSwitch, InfiniBand
//!   adaptive routing, Spectrum-X) or by a static per-flow hash (ECMP).
//!
//! Every output port has virtual output queues, one per input (an upstream
//! port, a local NCCL channel, or the GPU's memory system for traffic no SM
//! issues), served round-robin one packet at a time. A packet
//! may only start on a link if the downstream switch buffer has credits
//! for it (lossless credit-based flow control, as on NVLink and
//! InfiniBand); credits return one link latency after the packet leaves
//! that buffer. Forwarding is virtual cut-through: the head moves on as
//! soon as it arrives, but a packet never finishes leaving a switch before
//! its tail has arrived. Congestion (incast, hash collisions,
//! oversubscribed uplinks) therefore spreads backwards through credits
//! exactly like head-of-line blocking does in hardware.
//!
//! SHARP sources (NVLS / IB SHARP) are driven packet by packet: result
//! packet `k` leaves once every contribution's packet `k` has arrived and
//! the switch ALU has reduced it; multicast packets are replicated to every
//! member port.
//!
//! Granularity: a "packet" is `packet_bytes` of payload (default 4 KiB,
//! i.e. a train of 32 NVLink write packets of 128 B whose header flits are
//! charged by the link framing model). Transfers larger than
//! `max_packets_per_transfer × packet_bytes` use proportionally larger
//! packets to bound simulation cost.

use crate::collective::End;
use crate::engine::{Ev, NetEngine, NetEvent};
use crate::fabric::{ChannelId, Fabric};
use crate::phy::LinkTech;
use rsim_core::Time;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Routing {
    /// Least-loaded equal-cost output per packet.
    #[default]
    Adaptive,
    /// Static per-flow hash.
    Ecmp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PacketConfig {
    pub packet_bytes: u32,
    pub max_packets_per_transfer: u32,
    /// Input buffer per NVSwitch port (bytes of credits).
    pub nvswitch_buffer_bytes: u64,
    /// Input buffer per InfiniBand switch port.
    pub ib_buffer_bytes: u64,
    /// Multi-path policy in scale-out switches.
    pub routing: Routing,
    /// SHARP ALU pipeline latency, ns.
    pub sharp_latency_ns: f64,
}

impl Default for PacketConfig {
    fn default() -> Self {
        Self {
            packet_bytes: 4096,
            max_packets_per_transfer: 32,
            nvswitch_buffer_bytes: 64 * 1024,
            ib_buffer_bytes: 256 * 1024,
            routing: Routing::Adaptive,
            sharp_latency_ns: 100.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PEv {
    Release { coll: u64, tid: u32, k: u32 },
    Arrive { pkt: u32 },
    Deliver { pkt: u32 },
    TxEnd { port: u32 },
    Credit { port: u32, bytes: u32 },
}

const NONE: u32 = u32::MAX;

struct Port {
    busy_until: Time,
    credits: i64,
    cap: i64,
    /// (input key, queue of packet ids)
    inputs: Vec<(u32, VecDeque<u32>)>,
    rr: usize,
    queued: u64,
    tx_pending: bool,
}

struct Pkt {
    coll: u64,
    tid: u32,
    payload: u32,
    /// Bytes before link framing (payload / protocol efficiency).
    logical: u32,
    request: bool,
    dst: u32,
    idx: u32,
    flow: u32,
    /// Inbound port and the wire bytes it charged (credit to return).
    via: u32,
    via_wire: u32,
    /// Tail arrival at the current node.
    tail: Time,
}

struct Inj {
    npk: u32,
    payload: u32,
    next: u32,
    off: u32,
    /// Lanes (by packet index) of copies (src node, dst node).
    lanes: Vec<Vec<(usize, usize)>>,
}

pub(crate) struct PacketState {
    ports: Vec<Port>,
    pkts: Vec<Pkt>,
    free: Vec<u32>,
    inj: HashMap<(u64, u32), Inj>,
}

impl PacketState {
    pub fn new(f: &Fabric, cfg: &PacketConfig) -> Self {
        let ports = f
            .channels
            .iter()
            .map(|c| {
                let cap = if f.is_gpu[c.to] {
                    i64::MAX / 4
                } else if c.tech == LinkTech::IbNdr
                    || c.tech == LinkTech::IbXdr
                    || c.tech == LinkTech::Roce400
                {
                    cfg.ib_buffer_bytes as i64
                } else {
                    cfg.nvswitch_buffer_bytes as i64 * c.count.max(1) as i64
                };
                Port {
                    busy_until: 0,
                    credits: cap,
                    cap,
                    inputs: Vec::new(),
                    rr: 0,
                    queued: 0,
                    tx_pending: false,
                }
            })
            .collect();
        Self {
            ports,
            pkts: Vec::new(),
            free: Vec::new(),
            inj: HashMap::new(),
        }
    }
}

fn hash3(a: u64, b: u64, c: u64) -> u64 {
    let mut h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= c.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h.wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

impl NetEngine {
    fn pk_alloc(&mut self, p: Pkt) -> u32 {
        if let Some(i) = self.pk.free.pop() {
            self.pk.pkts[i as usize] = p;
            i
        } else {
            self.pk.pkts.push(p);
            (self.pk.pkts.len() - 1) as u32
        }
    }

    pub(crate) fn pk_start(&mut self, coll: u64, tid: u32, t: Time, _out: &mut Vec<NetEvent>) {
        let cs = &self.colls[&coll];
        let tr = cs.plan.transfers[tid as usize].clone();
        let stream = cs.x[tid as usize].stream;
        // Endpoints per lane/copy.
        let lanes: Vec<Vec<(usize, usize)>> = match (tr.src, tr.dst) {
            (End::Rank(_), End::Rank(_))
            | (End::Rank(_), End::Switch(_))
            | (End::Switch(_), End::Rank(_)) => {
                let s = Self::node_of(&cs.desc, tr.src).unwrap();
                let d = Self::node_of(&cs.desc, tr.dst).unwrap();
                vec![vec![(s, d)]]
            }
            _ => self
                .lanes(coll, tid)
                .into_iter()
                .map(|copies| {
                    copies
                        .iter()
                        .map(|r| {
                            (
                                self.fabric.channels[r[0]].from,
                                self.fabric.channels[*r.last().unwrap()].to,
                            )
                        })
                        .collect()
                })
                .collect(),
        };
        let cfg = &self.cfg.packet;
        let payload = (cfg.packet_bytes as u64)
            .max(
                tr.bytes
                    .div_ceil(cfg.max_packets_per_transfer.max(1) as u64),
            )
            .min(tr.bytes)
            .max(1) as u32;
        let npk = tr.bytes.div_ceil(payload as u64) as u32;
        let nl = lanes.len() as u32;
        let off = crate::engine::lane_offset(coll, tid, nl as usize) as u32;
        let units: u64 = (0..npk)
            .map(|k| lanes[((k + off) % nl) as usize].len() as u64)
            .sum();
        let cs = self.colls.get_mut(&coll).unwrap();
        cs.x[tid as usize].units_left = units as u32;
        cs.x[tid as usize].units_total = units as u32;
        self.pk.inj.insert(
            (coll, tid),
            Inj {
                npk,
                payload,
                next: 0,
                off,
                lanes,
            },
        );
        if stream {
            self.pk_kick_stream(coll, tid, t);
            return;
        }
        // SM-issued (or plain) transfer: packets leave at the channel's copy rate.
        let cs = &self.colls[&coll];
        let src_gpu = match tr.src {
            End::Rank(r) => Some(cs.desc.ranks[r]),
            _ => None,
        };
        let chan_rate = cs.chan_rate;
        let mut start = t as f64;
        for k in 0..npk {
            let b = if k + 1 == npk {
                tr.bytes - payload as u64 * (npk as u64 - 1)
            } else {
                payload as u64
            };
            if let (Some(ch), Some(g)) = (tr.channel, src_gpu) {
                let f = self.sm_free.entry((g, ch)).or_insert(0.0);
                start = f.max(t as f64);
                *f = start + b as f64 / chan_rate;
            }
            self.q
                .push(start.ceil() as Time, Ev::Pk(PEv::Release { coll, tid, k }));
        }
        self.pk.inj.get_mut(&(coll, tid)).unwrap().next = npk;
    }

    /// Release as many packets of streaming transfer `tid` as its
    /// dependencies allow.
    pub(crate) fn pk_kick_stream(&mut self, coll: u64, tid: u32, now: Time) {
        let Some(inj) = self.pk.inj.get(&(coll, tid)) else {
            return;
        };
        let (npk, payload, mut next) = (inj.npk, inj.payload, inj.next);
        let cs = &self.colls[&coll];
        let tr = &cs.plan.transfers[tid as usize];
        // Fraction of every dependency delivered.
        let mut frac: f64 = 1.0;
        for &d in &tr.deps {
            let x = &cs.x[d];
            let f = if x.units_total == 0 {
                if x.units_left == 0 && x.started {
                    1.0
                } else {
                    0.0
                }
            } else {
                x.delivered as f64 / x.units_total as f64
            };
            frac = frac.min(f);
        }
        let avail = ((frac * npk as f64) + 1e-9).floor() as u32;
        let reduce = tr.reduce;
        let ndeps = tr.deps.len().max(1) as f64;
        let lat = if reduce {
            (self.cfg.packet.sharp_latency_ns * 1000.0) as Time
        } else {
            0
        };
        while next < npk.min(avail) {
            let mut at = now as f64;
            if reduce {
                let inj = &self.pk.inj[&(coll, tid)];
                let lane = &inj.lanes[((next + inj.off) % inj.lanes.len() as u32) as usize];
                let sw = lane[0].0;
                let rate = self.fabric.switch_rate[sw] * self.cfg.sharp_alu_ratio;
                let f = self.alu_free.entry(sw).or_insert(0.0);
                at = f.max(at);
                *f = at + payload as f64 * ndeps / rate.max(1e-9);
            }
            self.q.push(
                at.ceil() as Time + lat,
                Ev::Pk(PEv::Release { coll, tid, k: next }),
            );
            next += 1;
        }
        self.pk.inj.get_mut(&(coll, tid)).unwrap().next = next;
    }

    fn pk_route(&mut self, u: usize, p: u32, now: Time) -> ChannelId {
        let (dst, idx, flow) = {
            let k = &self.pk.pkts[p as usize];
            (k.dst as usize, k.idx, k.flow)
        };
        let hops = self.fabric.next_hops(u, dst);
        assert!(!hops.is_empty(), "no route from node {u} to node {dst}");
        if hops.len() == 1 {
            return hops[0];
        }
        if self.fabric.is_gpu[u] {
            // NVLink address interleave across planes.
            return hops[(idx as usize + flow as usize) % hops.len()];
        }
        let adaptive = !self.fabric.is_scale_out[u] || self.cfg.packet.routing == Routing::Adaptive;
        if adaptive {
            *hops
                .iter()
                .min_by(|&&a, &&b| {
                    let la = self.port_load(a, now);
                    let lb = self.port_load(b, now);
                    la.partial_cmp(&lb).unwrap().then(a.cmp(&b))
                })
                .unwrap()
        } else {
            let h = hash3(flow as u64, u as u64, dst as u64);
            hops[(h % hops.len() as u64) as usize]
        }
    }

    fn port_load(&self, c: ChannelId, now: Time) -> f64 {
        let p = &self.pk.ports[c];
        let ch = &self.fabric.channels[c];
        p.queued as f64 / ch.rate
            + p.busy_until.saturating_sub(now) as f64
            + (p.cap - p.credits).max(0) as f64 / ch.rate
    }

    fn pk_enqueue(&mut self, port: ChannelId, key: u32, p: u32, now: Time) {
        let bytes = self.pk.pkts[p as usize].logical as u64;
        let pt = &mut self.pk.ports[port];
        match pt.inputs.iter_mut().find(|x| x.0 == key) {
            Some(x) => x.1.push_back(p),
            None => pt.inputs.push((key, VecDeque::from([p]))),
        }
        pt.queued += bytes;
        self.pk_try_send(port, now);
    }

    fn wire_of(&self, port: ChannelId, p: &Pkt) -> f64 {
        let f = &self.fabric.channels[port].framing;
        if p.request {
            p.logical as f64 / (1.0 - f.framing)
        } else {
            f.wire_bytes(p.logical as u64, f.sm_payload)
        }
    }

    fn pk_try_send(&mut self, port: ChannelId, now: Time) {
        if self.pk.ports[port].busy_until > now {
            if !self.pk.ports[port].tx_pending {
                self.pk.ports[port].tx_pending = true;
                let at = self.pk.ports[port].busy_until;
                self.q.push(at, Ev::Pk(PEv::TxEnd { port: port as u32 }));
            }
            return;
        }
        let n = self.pk.ports[port].inputs.len();
        if n == 0 {
            return;
        }
        let start = self.pk.ports[port].rr % n;
        let mut pick = None;
        for j in 0..n {
            let i = (start + j) % n;
            let head = *self.pk.ports[port].inputs[i].1.front().unwrap();
            let k = &self.pk.pkts[head as usize];
            let w = self.wire_of(port, k);
            let charge = self.credit_charge(w, k.payload);
            let pt = &self.pk.ports[port];
            if pt.credits as f64 >= charge || pt.credits == pt.cap {
                pick = Some((i, head, w, charge));
                break;
            }
        }
        let Some((i, p, wire, charge)) = pick else {
            return; // credit-starved; a Credit event will retry
        };
        {
            let pt = &mut self.pk.ports[port];
            pt.inputs[i].1.pop_front();
            if pt.inputs[i].1.is_empty() {
                pt.inputs.remove(i);
                pt.rr = i;
            } else {
                pt.rr = i + 1;
            }
        }
        let ch = &self.fabric.channels[port];
        let (rate, lat) = (ch.rate, ch.latency);
        let dur = wire / rate;
        let (via, via_wire, tail_in, logical, payload) = {
            let k = &self.pk.pkts[p as usize];
            (k.via, k.via_wire, k.tail, k.logical, k.payload)
        };
        let end_f = (now as f64 + dur).max(tail_in as f64);
        let end = end_f.ceil() as Time;
        let wire_u = charge.ceil() as u32;
        {
            let pt = &mut self.pk.ports[port];
            pt.busy_until = end;
            pt.queued = pt.queued.saturating_sub(logical as u64);
            if pt.cap < i64::MAX / 4 {
                pt.credits -= wire_u as i64;
            }
            pt.tx_pending = true;
        }
        self.fabric.channels[port].account(now as f64, dur, payload as u64);
        if via != NONE {
            // The packet left the upstream buffer: return its credit.
            let vl = self.fabric.channels[via as usize].latency;
            self.q.push(
                now + vl,
                Ev::Pk(PEv::Credit {
                    port: via,
                    bytes: via_wire,
                }),
            );
        }
        {
            let k = &mut self.pk.pkts[p as usize];
            k.via = port as u32;
            k.via_wire = wire_u;
            k.tail = end + lat;
        }
        self.q.push(now + lat, Ev::Pk(PEv::Arrive { pkt: p }));
        self.q.push(end, Ev::Pk(PEv::TxEnd { port: port as u32 }));
    }

    /// Buffer space a packet holds downstream. A coarsened packet (see the
    /// module docs) stands for a train of `packet_bytes` packets moving
    /// cut-through, so it holds one nominal packet's worth, not its whole
    /// length: otherwise coarsening alone would starve the credit loop.
    fn credit_charge(&self, wire: f64, payload: u32) -> f64 {
        wire * (self.cfg.packet.packet_bytes as f64 / payload.max(1) as f64).min(1.0)
    }

    pub(crate) fn pk_event(&mut self, e: PEv, now: Time, out: &mut Vec<NetEvent>) {
        match e {
            PEv::Release { coll, tid, k } => {
                let Some(inj) = self.pk.inj.get(&(coll, tid)) else {
                    return;
                };
                let lane = inj.lanes[((k + inj.off) % inj.lanes.len() as u32) as usize].clone();
                let payload = if k + 1 == inj.npk {
                    let tb = self.colls[&coll].plan.transfers[tid as usize].bytes;
                    (tb - inj.payload as u64 * (inj.npk as u64 - 1)) as u32
                } else {
                    inj.payload
                };
                let cs = &self.colls[&coll];
                let tr = &cs.plan.transfers[tid as usize];
                let eff = if tr.channel.is_some() {
                    cs.plan.proto.payload_efficiency()
                } else {
                    1.0
                };
                let request = tr.request;
                let flow = (hash3(coll, tid as u64, 7) & 0xffff) as u32;
                // Local injection queue: one FIFO per NCCL channel (a channel's
                // SM works through its chunks in order), and one FIFO for
                // traffic no SM issues (read responses served by the memory
                // system, switch engine output). Upstream ports use their id.
                let key = match tr.channel {
                    Some(c) => {
                        0x8000_0000
                            | ((coll as u32).wrapping_mul(64).wrapping_add(c as u32) & 0x3fff_ffff)
                    }
                    None => 0xc000_0000,
                };
                for (s, d) in lane {
                    let p = self.pk_alloc(Pkt {
                        coll,
                        tid,
                        payload,
                        logical: (payload as f64 / eff).ceil() as u32,
                        request,
                        dst: d as u32,
                        idx: k,
                        flow,
                        via: NONE,
                        via_wire: 0,
                        tail: now,
                    });
                    let port = self.pk_route(s, p, now);
                    self.pk_enqueue(port, key, p, now);
                }
            }
            PEv::Arrive { pkt } => {
                let at = self.fabric.channels[self.pk.pkts[pkt as usize].via as usize].to;
                let (dst, tail) = {
                    let k = &self.pk.pkts[pkt as usize];
                    (k.dst as usize, k.tail)
                };
                if at == dst {
                    self.q.push(tail.max(now), Ev::Pk(PEv::Deliver { pkt }));
                } else {
                    let via = self.pk.pkts[pkt as usize].via;
                    let port = self.pk_route(at, pkt, now);
                    self.pk_enqueue(port, via, pkt, now);
                }
            }
            PEv::Deliver { pkt } => {
                let (coll, tid, via, vw) = {
                    let k = &self.pk.pkts[pkt as usize];
                    (k.coll, k.tid, k.via, k.via_wire)
                };
                let pt = &self.pk.ports[via as usize];
                if pt.cap < i64::MAX / 4 {
                    // Consumed by a switch engine: buffer freed.
                    let vl = self.fabric.channels[via as usize].latency;
                    self.q.push(
                        now + vl,
                        Ev::Pk(PEv::Credit {
                            port: via,
                            bytes: vw,
                        }),
                    );
                }
                self.pk.free.push(pkt);
                let done = {
                    let x = &self.colls[&coll].x[tid as usize];
                    x.units_left == 1
                };
                if done {
                    self.pk.inj.remove(&(coll, tid));
                }
                self.deliver(coll, tid, now, now, out);
            }
            PEv::TxEnd { port } => {
                self.pk.ports[port as usize].tx_pending = false;
                self.pk_try_send(port as usize, now);
            }
            PEv::Credit { port, bytes } => {
                let pt = &mut self.pk.ports[port as usize];
                pt.credits = (pt.credits + bytes as i64).min(pt.cap);
                self.pk_try_send(port as usize, now);
            }
        }
    }
}
