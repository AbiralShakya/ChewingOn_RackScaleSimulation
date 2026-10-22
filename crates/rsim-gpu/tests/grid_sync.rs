//! Inter-SM synchronisation through a counter in global memory (`GlobalAdd` /
//! `GlobalWait`), the mechanism persistent megakernels use to order work
//! across SMs.

use rsim_gpu::stats::Stall;
use rsim_gpu::{Gpu, GpuConfig, KernelLaunch};
use rsim_hw::presets;
use rsim_sass::PathKernel;
use std::sync::Arc;

const SASS: &str = "\tcode for sm_100a
\t\tFunction : mk
\t.headerflags\t@\"EF_CUDA_SM100\"
        /*0000*/                   FFMA R2, R2, R3, R4 ;                     /* 0x0000000302027223 */
                                                                             /* 0x000fe20000000004 */
        /*0010*/                   RED.E.ADD.STRONG.GPU desc[UR4][R6.64], R5 ;  /* 0x000000050600798e */
                                                                             /* 0x000fe2000c10e384 */
        /*0020*/                   LDG.E.STRONG.GPU R8, desc[UR4][R6.64] ;  /* 0x0000000406087981 */
                                                                             /* 0x000ea4000c1f5900 */
        /*0030*/                   EXIT ;                                    /* 0x000000000000794d */
                                                                             /* 0x000fea0003800000 */
";

/// One op: (idx, mbar kind, id, arg, n).
fn op(b: &mut Vec<u8>, idx: u32, kind: u8, id: u32, arg: u32, n: u32, addr: Option<u64>) {
    b.extend(idx.to_le_bytes());
    b.extend(1u32.to_le_bytes()); // active: lane 0
    b.extend(0u64.to_le_bytes()); // payload
    b.push(kind);
    b.extend(id.to_le_bytes());
    b.extend(arg.to_le_bytes());
    b.extend(n.to_le_bytes());
    b.extend(0u16.to_le_bytes()); // ranks
    b.extend(0u32.to_le_bytes()); // run bytes
    match addr {
        Some(a) => {
            b.extend(1u16.to_le_bytes());
            b.extend(a.to_le_bytes());
        }
        None => b.extend(0u16.to_le_bytes()),
    }
}

/// Three CTAs of one warp. CTAs 0 and 1 compute (`work` FFMAs) and then
/// increment counter 0; CTA 2 waits for both increments before it exits.
fn kernel(work: [u32; 2], need: u32) -> KernelLaunch {
    let header = serde_json::json!({
        "kernel": "mk", "grid": [3, 1, 1], "block": [32, 1, 1], "regs": 32, "smem": 0,
        "primary": 0, "sass": SASS,
    })
    .to_string();
    let mut b = b"RSIMPTH1".to_vec();
    b.extend((header.len() as u32).to_le_bytes());
    b.extend(header.as_bytes());
    b.extend(3u32.to_le_bytes());
    let flag = 0x7000_0000u64;
    for (cta, &w) in work.iter().enumerate() {
        let mut ops = Vec::new();
        for _ in 0..w {
            op(&mut ops, 0, 0, 0, 0, 0, None);
        }
        op(&mut ops, 1, 8, 0, 0, 1, Some(flag));
        op(&mut ops, 3, 0, 0, 0, 0, None);
        b.extend((cta as u32).to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend((w + 2).to_le_bytes());
        b.extend(ops);
    }
    b.extend(2u32.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend(2u32.to_le_bytes());
    op(&mut b, 2, 7, 0, need, 0, Some(flag));
    op(&mut b, 3, 0, 0, 0, 0, None);
    b.extend(0u32.to_le_bytes()); // no TMA table
    KernelLaunch::from_path(PathKernel::from_bytes(&b).unwrap())
}

fn run(k: KernelLaunch) -> (u64, u64) {
    let mut g = Gpu::new(0, presets::b200(), GpuConfig::default(), false);
    g.submit(Arc::new(k), 0, 1);
    let mut out = vec![];
    g.advance(u64::MAX / 2, &mut out);
    assert_eq!(out.len(), 1, "kernel did not complete (deadlock)");
    let d = out.pop().unwrap();
    (
        d.end_cycle - d.start_cycle,
        d.stats.warp.get(Stall::GridSync),
    )
}

#[test]
fn waiter_finishes_after_the_slowest_producer() {
    let (short, _) = run(kernel([100, 100], 2));
    let (long, sync) = run(kernel([100, 3000], 2));
    // The waiting CTA is released by the second increment, one L2 round trip
    // after the slow producer's atomic.
    assert!(long > short + 2500, "short {short} long {long}");
    assert!(sync > 2500, "grid-sync warp cycles {sync}");
}

#[test]
fn wait_for_one_increment_does_not_wait_for_the_slow_producer() {
    let (_, one) = run(kernel([100, 3000], 1));
    let (_, two) = run(kernel([100, 3000], 2));
    // The kernel still ends with the slow producer, but the waiter left early.
    assert!(
        one * 5 < two,
        "grid-sync warp cycles: need 1 {one}, need 2 {two}"
    );
}
