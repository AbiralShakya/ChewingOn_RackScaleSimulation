// The closed forms of rackscale.scaling, ported line for line so the learning
// page can recompute any plan as the reader changes it. verify.py checks this
// file against the Python on random plans; change both together.
(function (root) {
  "use strict";
  const DTYPE_BYTES = { bf16: 2, fp16: 2, fp8: 1, fp4: 0.5, nvfp4: 0.5625, fp32: 4, tf32: 4 };
  const fdiv = (a, b) => Math.floor(a / b);
  const ceil = (a, b) => Math.ceil(a / b);

  function peak(hw, dtype) { return hw.flops[dtype] !== undefined ? hw.flops[dtype] : hw.flops.bf16; }
  function linkBw(hw, n) { return (n <= hw.domain || !hw.ib_bw) ? hw.nvlink_bw : hw.ib_bw; }

  function collective(kind, B, n, hw, algo) {
    if (n <= 1 || B <= 0) return [0, "0 (one rank)"];
    const w = linkBw(hw, n), f = (n - 1) / n;
    const nvls = hw.nvls && n <= hw.domain && (algo == null || algo === "nvls");
    if (kind === "all_reduce") {
      if (nvls) return [hw.alpha + B / w, "alpha + B/W (NVLS)"];
      return [hw.alpha + 2 * f * B / w, "alpha + 2(n-1)/n B/W (ring)"];
    }
    if (kind === "all_gather" || kind === "reduce_scatter" || kind === "all_to_all")
      return [hw.alpha + f * B / w, "alpha + (n-1)/n B/W"];
    if (kind === "broadcast") return [hw.alpha + B / w, "alpha + B/W"];
    throw new Error("unknown collective " + kind);
  }
  function p2p(B, hw, cross) { const w = cross && hw.ib_bw ? hw.ib_bw : hw.nvlink_bw; return hw.alpha_p2p + B / w; }

  function modelFlops(m, tokens, skv, train, lmHead) {
    const dense = 2 * (m.active_no_head + (lmHead ? m.vocab * m.hidden : 0)) * tokens;
    const attn = m.attn_coeff * skv * tokens;
    const ssm = (skv > 1 ? m.ssm_pre : m.ssm_dec) * tokens;
    return (train ? 3 : 1) * (dense + attn + ssm);
  }

  function weightsPerGpu(m, tp, ep, pp) {
    const lp = m.layer_entries.map(([x, y, z]) => x / tp + y / (ep * tp) + z);
    const per = ceil(lp.length, pp);
    const stages = [];
    for (let i = 0; i < lp.length; i += per) stages.push(lp.slice(i, i + per).reduce((a, b) => a + b, 0));
    const emb = m.vocab * m.hidden / tp;
    stages[0] += emb;
    stages[stages.length - 1] += emb;
    return Math.max(...stages);
  }
  function kvBytesPerToken(m, tp, pp) {
    const key = String(tp);
    const elems = m.kv_elems[key] !== undefined ? m.kv_elems[key] : m.kv_elems["1"];
    return elems * 2 * m.attn_layers / pp;
  }

  // lower.memory
  function memory(m, p, phase, o) {
    const w = weightsPerGpu(m, p.tp, p.ep, p.pp);
    const out = { weights: w * m.bytes_per_param };
    if (phase === "decode" || phase === "prefill") {
      const ctx = phase === "decode" ? o.context : o.seqlen;
      out.kv_cache = o.batch * ctx * kvBytesPerToken(m, p.tp, p.pp);
      if (m.ssm_state_tp1) out.ssm_state = o.batch * m.ssm_state_tp1 / p.tp / p.pp;
      if (phase === "prefill") {
        const t = o.batch * o.seqlen;
        out.activations = t * (4 * m.hidden + 2 * m.ffn_max / p.tp) * 2;
      }
    } else {
      const s = p.shard || p.dp;
      if (p.zero === 3) out.weights = w * m.bytes_per_param * (1 / s + 2 / m.layers);
      out.grads = 2 * w / (p.zero >= 2 ? s : 1);
      out.optimizer = 12 * w / (p.zero >= 1 ? s : 1);
      const inFlight = o.schedule === "gpipe" ? o.micro_batches : Math.min(o.micro_batches, p.pp);
      const perLayer = 34 * o.batch * o.seqlen * m.hidden / p.tp;
      out.activations = perLayer * ceil(m.layers, p.pp) * inFlight;
    }
    return out;
  }

  function world(plan) { return Math.max(plan.tp * plan.cp * plan.pp * plan.dp, plan.ep); }
  function paramSplit(m, plan) {
    const dense = m.total_params - m.expert_params;
    return [dense / (plan.tp * plan.pp), m.expert_params / (plan.tp * plan.pp * plan.ep)];
  }

  function combine(terms) {
    const sum = (k) => terms.filter((x) => x.kind === k).reduce((a, x) => a + x.s, 0);
    const comp = sum("compute"), mem = sum("memory");
    const roof = Math.max(comp, mem);
    const exposed = sum("exposed");
    const over = terms.filter((x) => x.kind === "overlapped");
    const hidden = over.reduce((a, x) => a + Math.max(0, x.s - (x.window !== undefined ? x.window : roof)), 0);
    const step = roof + exposed + hidden;
    const noOverlap = comp + mem + exposed + over.reduce((a, x) => a + x.s, 0);
    const perfect = Math.max(roof, exposed + hidden, over.length ? Math.max(...over.map((x) => x.s)) : 0);
    return [step, noOverlap, perfect];
  }

  function label(plan, train) {
    const parts = [];
    for (const k of ["tp", "cp", "pp", "dp", "ep"]) if (plan[k] !== 1) parts.push(k + plan[k]);
    if (plan.sp) parts.push("sp");
    if (plan.cp > 1) parts.push(plan.cp_kind);
    if (plan.pp > 1 && train) parts.push(plan.schedule + (plan.virtual > 1 ? "v" + plan.virtual : ""));
    if (plan.dp > 1 && train) parts.push(["ddp", "zero1", "zero2", "fsdp"][plan.zero] +
      (plan.shard && plan.shard !== plan.dp ? "-hsdp" + plan.shard : ""));
    return parts.join(" ") || "1 GPU";
  }

  function finish(m, hw, plan, phase, terms, step, noOv, perfect, flopsGpu, dtype, mem) {
    const memGb = {};
    let total = 0;
    for (const k in mem) { memGb[k] = mem[k] / 1e9; total += mem[k] / 1e9; }
    let dom = terms[0], best = -1;
    for (const x of terms) {
      const v = x.kind !== "overlapped" ? x.s : Math.max(0, x.s - (x.window !== undefined ? x.window : 0));
      if (v > best) { best = v; dom = x; }
    }
    return {
      model: m.name, hardware: hw.name, phase, plan, label: label(plan, phase === "train"), world: world(plan), dtype,
      step_us: step * 1e6, bounds_us: { no_overlap: noOv * 1e6, perfect_overlap: perfect * 1e6 },
      mfu: flopsGpu / (peak(hw, dtype) * step), dominant: dom.name,
      terms: terms.map((x) => Object.assign({ name: x.name, us: x.s * 1e6, kind: x.kind, formula: x.formula }, x.detail || {},
        x.window !== undefined ? { window: x.window } : {})),
      memory_gb: memGb, memory_total_gb: total, fits: total <= hw.hbm_bytes / 1e9 * 0.92,
    };
  }

  function T(name, s, formula, kind, extra) { return Object.assign({ name, s, formula, kind }, extra || {}); }

  function dpCritical(hw, zero, n, dtype) {
    const C = peak(hw, dtype), W = linkBw(hw, n);
    const nvls = hw.nvls && n <= hw.domain;
    if (zero === 0) return nvls ? C / (2 * W) : C / W;
    if (zero === 1 || zero === 2) return C / (2 * W);
    return C / W;
  }
  function tpCrit(m, hw, dtype) {
    const w = hw.nvlink_bw, C = peak(hw, dtype);
    if (hw.nvls) return 2 * m.tp_crit_params * w / (4 * m.hidden * C);
    return 1 + 2 * m.tp_crit_params * w / (8 * m.hidden * C);
  }

  function normPlan(p) {
    return Object.assign({ tp: 1, sp: false, cp: 1, cp_kind: "ring", pp: 1, schedule: "1f1b", virtual: 1, dp: 1,
      zero: 2, shard: 0, ep: 1 }, p || {});
  }

  function estimateTrain(m, hw, plan, mb, seqlen, mbs, dtype, optimizer) {
    const { tp, cp, pp, dp, ep } = plan;
    const W_ = world(plan);
    const C = peak(hw, dtype), eb = DTYPE_BYTES[dtype] !== undefined ? DTYPE_BYTES[dtype] : 2;
    const L = m.layers;
    const tokens = mb * seqlen * mbs * dp;
    const tGpu = fdiv(mb * seqlen, cp);
    const flopsGpu = modelFlops(m, tokens, fdiv(seqlen, 2), true, false) / W_;
    const [denseP, expertP] = paramSplit(m, plan);
    const pGpu = denseP + expertP;
    const s = plan.shard || dp;
    const terms = [];
    const tMath = flopsGpu / C;
    terms.push(T("matmul FLOPs", tMath, "6 N_active T / world / C  (+ attention 12 L H d S/2 T)", "compute", { detail: { flops: flopsGpu } }));
    const tW = 3 * mbs * pGpu * eb / hw.hbm_bw;
    terms.push(T("weight reads", tW, "3 m P_gpu b / beta", "memory", { detail: { p_gpu: pGpu } }));
    let tpT = 0;
    if (tp > 1) {
      const a = tGpu * m.hidden * 2;
      const n = fdiv(2 * m.mixers, pp) * mbs;
      let one, f;
      if (plan.sp) {
        one = collective("reduce_scatter", a, tp, hw)[0] + collective("all_gather", a, tp, hw)[0];
        f = "2 L_stage m [RS + AG](T D 2)";
      } else {
        const r = collective("all_reduce", a, tp, hw); one = r[0]; f = "2 L_stage m x AR(T D 2): " + r[1];
      }
      tpT = n * one;
      terms.push(T("TP activations", tpT, f, "exposed", { detail: { bytes: a, count: n, each: one } }));
    }
    let epT = 0;
    if (m.moe && ep > 1) {
      const a = tGpu * m.moe.topk * m.hidden * 2;
      const n = fdiv(2 * 2 * m.moe_layers, pp) * mbs;
      const one = collective("all_to_all", a, ep, hw)[0];
      epT = n * one;
      terms.push(T("EP all-to-all", epT, "4 L_moe m x A2A(T k D 2)", "exposed", { detail: { bytes: a, count: n } }));
    }
    let cpT = 0;
    const nAttn = fdiv(m.attn_layers, pp);
    if (cp > 1 && nAttn) {
      const c = tGpu;
      const hq = fdiv(m.heads, tp), hkv = Math.max(1, fdiv(m.kv_heads, tp)), d = m.head_dim;
      if (plan.cp_kind === "ulysses") {
        const one = collective("all_to_all", c * (hq + 2 * hkv) * d * 2, cp, hw)[0] + collective("all_to_all", c * hq * d * 2, cp, hw)[0];
        cpT = 2 * nAttn * mbs * one;
        terms.push(T("Ulysses all-to-all", cpT, "2 L_attn m [A2A(c(H+2Hkv)d2) + A2A(c H d 2)]", "exposed"));
      } else {
        const kv = c * 2 * hkv * d * 2;
        const x = p2p(kv, hw);
        const stepC = 4 * c * (c / 2) * hq * d / C;
        const exposed = (cp - 1) * (Math.max(0, x - stepC) + Math.max(0, 2 * x - 2.5 * stepC));
        cpT = nAttn * mbs * exposed;
        terms.push(T("ring attention (exposed part)", cpT, "L_attn m (cp-1) [max(0, KV/W - 4c(c/2)Hd/C) + bwd]", "exposed",
          { detail: { kv_bytes: kv, transfer: x, step_compute: stepC } }));
      }
    }
    const stageMb = (Math.max(tMath, tW) + tpT + epT + cpT) / mbs;
    if (pp > 1) {
      let bubble, f;
      if (plan.schedule === "zb") { bubble = (pp - 1) * stageMb / 3; f = "(p-1)(t_F + t_B - t_W), t_F = t_B = t_W"; }
      else {
        const v = plan.schedule === "interleaved" ? plan.virtual : 1;
        bubble = (pp - 1) * stageMb / v; f = "(p-1)/v (t_F + t_B + t_W)";
      }
      terms.push(T("pipeline bubble", bubble, f, "exposed", { detail: { fraction: bubble / (bubble + mbs * stageMb) } }));
      const a = tGpu * m.hidden * 2 / tp;
      terms.push(T("pipeline fill/drain transfers", 2 * (pp - 1) * p2p(a, hw, W_ > hw.domain), "2 (p-1) (alpha + T D 2 / tp / W)", "exposed"));
    }
    const windowBwd = 2 / 3 * (Math.max(tMath, tW) + tpT + epT + cpT);
    if (dp > 1) {
      const gD = 2 * denseP, gE = 2 * expertP;
      const repE = Math.max(1, fdiv(dp * cp, ep));
      if (plan.zero === 0) {
        let t = collective("all_reduce", gD, dp * cp, hw)[0];
        if (gE) t += collective("all_reduce", gE, repE, hw)[0];
        terms.push(T("DP gradient all-reduce", t, "AR(2 P_gpu) over dp", "overlapped", { window: windowBwd }));
      } else if (plan.zero === 1 || plan.zero === 2) {
        let rs = collective("reduce_scatter", gD, dp * cp, hw)[0], ag = collective("all_gather", gD, dp * cp, hw)[0];
        if (gE) { rs += collective("reduce_scatter", gE, repE, hw)[0]; ag += collective("all_gather", gE, repE, hw)[0]; }
        terms.push(T("ZeRO gradient reduce-scatter", rs, "RS(2 P_gpu) over dp", "overlapped", { window: windowBwd }));
        if (optimizer) terms.push(T("ZeRO parameter all-gather", ag, "AG(2 P_gpu) over dp, after the optimizer", "exposed"));
      } else {
        const n = s;
        const ag = 2 * mbs * collective("all_gather", gD + gE, n, hw)[0];
        let rs = mbs * collective("reduce_scatter", gD + gE, n, hw)[0];
        if (fdiv(dp, n) > 1) rs += mbs * collective("all_reduce", (gD + gE) / n, fdiv(dp, n), hw)[0];
        terms.push(T("FSDP parameter all-gathers (fwd + bwd)", ag, "2 m AG(2 P_gpu) over shard group", "overlapped",
          { window: Math.max(tMath, tW) + tpT + epT + cpT }));
        terms.push(T("FSDP gradient reduce-scatter" + (fdiv(dp, n) > 1 ? " + HSDP all-reduce" : ""), rs,
          "m RS(2 P_gpu) over shard [+ AR(2 P_gpu / s) over replicas]", "overlapped", { window: windowBwd }));
        terms.push(T("FSDP first-layer gather", collective("all_gather", (gD + gE) / Math.max(1, fdiv(L, pp)), n, hw)[0],
          "AG(2 P_layer): nothing to hide behind", "exposed"));
      }
    }
    if (optimizer) {
      const pOpt = pGpu / ((dp > 1 && plan.zero >= 1) ? s : 1);
      terms.push(T("Adam update", 30 * pOpt / hw.hbm_bw, "30 B/param (4 FP32 in, 3 FP32 + 1 BF16 out) / beta", "exposed", { detail: { params: pOpt } }));
    }
    const [step, noOv, perfect] = combine(terms);
    let mem = {};
    if (!(m.moe && ep > 1)) {
      mem = memory(m, plan, "train", { batch: mb, seqlen, micro_batches: mbs, schedule: plan.schedule === "gpipe" ? "gpipe" : "1f1b" });
      if (cp > 1) mem.activations /= cp;
    }
    const out = finish(m, hw, plan, "train", terms, step, noOv, perfect, flopsGpu, dtype, mem);
    Object.assign(out, { tokens_per_step: tokens, tok_s_gpu: tokens / step / W_, tok_s: tokens / step, micro_batch: mb, seqlen, micro_batches: mbs });
    out.critical = { dp_tokens_per_gpu: dpCritical(hw, plan.zero, dp * cp, dtype), tokens_per_gpu: tokens / W_, tp_max: tpCrit(m, hw, dtype) };
    return out;
  }

  function estimateInfer(m, hw, phase, plan, batch, seqlen, context, dtype) {
    const { tp, pp, ep, cp } = plan;
    const W_ = world(plan);
    const C = peak(hw, dtype), eb = DTYPE_BYTES[dtype] !== undefined ? DTYPE_BYTES[dtype] : 2;
    const decode = phase === "decode";
    const sq = decode ? 1 : seqlen;
    const skv = decode ? context : seqlen;
    const rows = fdiv(batch * sq, decode ? 1 : cp);
    const seqs = batch * (ep > 1 ? ep : plan.dp);
    let tokens = seqs * sq;
    let flops = modelFlops(m, tokens, decode ? skv : fdiv(seqlen, 2), false, decode) / W_;
    const terms = [T("matmul + attention FLOPs", flops / C, "2 N_active T (+ attention) / world / C", "compute", { detail: { flops } })];
    const [denseP, expertP] = paramSplit(m, plan);
    let expertRead = 0;
    if (m.moe) {
      const local = Math.max(1, fdiv(m.moe.experts, Math.max(1, ep)));
      const recv = rows * m.moe.topk;
      expertRead = expertP * Math.min(local, recv) / local;
    }
    const wBytes = (denseP - (decode ? 0 : m.vocab * m.hidden / tp) + expertRead) * eb;
    const par = { tp, ep, pp, dp: 1, zero: 2, shard: 0 };
    const kv = m.attn_layers ? batch * skv * kvBytesPerToken(m, tp, pp) / (decode ? 1 : cp) : 0;
    const kvRead = decode ? kv : kv / 2;
    terms.push(T("weight reads", wBytes / hw.hbm_bw, "P_gpu b / beta", "memory", { detail: { bytes: wBytes } }));
    if (kvRead) terms.push(T("KV cache", kvRead / hw.hbm_bw, decode ? "B K kv_bytes_per_token / beta" : "KV written once", "memory", { detail: { bytes: kvRead } }));
    if (tp > 1) {
      const a = rows * m.hidden * 2;
      const r = collective("all_reduce", a, tp, hw);
      const n = fdiv(m.mixers, pp);
      terms.push(T("TP all-reduces", n * r[0], "2 L x AR(T D 2): " + r[1], "exposed", { detail: { bytes: a, count: n, each: r[0] } }));
    }
    if (m.moe && ep > 1) {
      const a = rows * m.moe.topk * m.hidden * 2;
      const n = fdiv(2 * m.moe_layers, pp);
      const one = collective("all_to_all", a, ep, hw)[0];
      terms.push(T("EP dispatch + combine", n * one, "2 L_moe A2A(T k D 2)", "exposed", { detail: { bytes: a, count: n } }));
    }
    if (pp > 1) terms.push(T("PP hops", (pp - 1) * p2p(rows * m.hidden * 2, hw), "(p-1)(alpha + T D 2 / W)", "exposed"));
    if (cp > 1 && !decode && m.attn_layers) {
      const c = fdiv(seqlen, cp), hq = m.heads, hkv = m.kv_heads, d = m.head_dim;
      if (plan.cp_kind === "ulysses") {
        const one = collective("all_to_all", c * (hq + 2 * hkv) * d * 2, cp, hw)[0] + collective("all_to_all", c * hq * d * 2, cp, hw)[0];
        terms.push(T("Ulysses all-to-all", m.attn_layers * one, "L_attn [A2A(qkv) + A2A(out)]", "exposed"));
      } else {
        const x = p2p(c * 2 * hkv * d * 2, hw);
        const stepC = 4 * c * (c / 2) * hq * d / C;
        terms.push(T("ring attention (exposed part)", m.attn_layers * (cp - 1) * Math.max(0, x - stepC), "L_attn (cp-1) max(0, KV/W - step)", "exposed",
          { detail: { transfer: x, step_compute: stepC } }));
      }
    }
    let [step, noOv, perfect] = combine(terms);
    const mem = memory(m, par, phase, { batch, context, seqlen });
    if (pp > 1) {
      step *= pp; noOv *= pp; perfect *= pp; tokens *= pp; flops *= pp;
      for (const k of ["kv_cache", "ssm_state"]) if (k in mem) mem[k] *= pp;
    }
    const out = finish(m, hw, plan, phase, terms, step, noOv, perfect, flops, dtype, mem);
    const replicas = Math.max(1, fdiv(hw.gpus, W_));
    Object.assign(out, { batch, context, seqlen, replicas, tokens_per_step: tokens });
    if (decode) {
      Object.assign(out, { tok_s_gpu: tokens / step / W_, tok_s_user: 1 / step, tpot_ms: step * 1e3, tok_s_rack: tokens / step * replicas });
      out.critical = { batch: C * eb / (2 * hw.hbm_bw), batch_per_gpu: rows };
    } else {
      Object.assign(out, { ttft_ms: step * 1e3, tok_s_gpu: tokens / step / W_, tok_s_rack: tokens / step * replicas });
    }
    return out;
  }

  function estimate(m, hw, phase, plan, o) {
    plan = normPlan(plan);
    o = Object.assign({ micro_batch: 1, seqlen: 4096, micro_batches: 1, batch: 1, context: 4096, dtype: "bf16", optimizer: true }, o || {});
    if (phase === "train") return estimateTrain(m, hw, plan, o.micro_batch, o.seqlen, o.micro_batches, o.dtype, o.optimizer);
    return estimateInfer(m, hw, phase, plan, o.batch, o.seqlen, o.context, o.dtype);
  }

  root.rsmath = { DTYPE_BYTES, peak, linkBw, collective, p2p, modelFlops, weightsPerGpu, kvBytesPerToken, memory,
    estimate, combine, label, normPlan, world, dpCritical, tpCrit };
})(typeof window !== "undefined" ? window : globalThis);
