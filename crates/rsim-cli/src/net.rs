//! `rsim net`: collective sweeps, NCCL's cost model, validation and fitting.

use anyhow::{bail, Context, Result};
use clap::Subcommand;
use rayon::prelude::*;
use rsim_hw::{presets, System, SystemSpec};
use rsim_net::collective::bus_bytes;
use rsim_net::engine::CollRecord;
use rsim_net::nccl;
use rsim_net::{run_one, Algo, CollKind, Collective, Fidelity, NetConfig, NetEngine, Proto};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Subcommand)]
pub enum NetCmd {
    /// nccl-tests-style sweep of one collective (all ranks post at t = 0).
    Sweep {
        /// System preset name or YAML path.
        #[arg(long)]
        system: String,
        /// all_reduce | all_gather | reduce_scatter | all_to_all | broadcast | send_recv
        #[arg(long, default_value = "all_reduce")]
        coll: String,
        /// Communicator size (default: every GPU).
        #[arg(long)]
        ranks: Option<usize>,
        /// auto | ring | tree | nvls | nvls_tree | collnet | direct
        #[arg(long, default_value = "auto")]
        algo: String,
        /// ll | ll128 | simple
        #[arg(long)]
        proto: Option<String>,
        /// transfer | packet
        #[arg(long, default_value = "transfer")]
        fidelity: String,
        #[arg(short = 'b', long, default_value = "8")]
        begin: String,
        #[arg(short = 'e', long, default_value = "1G")]
        end: String,
        #[arg(short = 'f', long, default_value_t = 4)]
        factor: u64,
        /// Override a NetConfig field, e.g. `nccl.step_ns=[300,1200,2400]`.
        #[arg(long)]
        set: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// NCCL 2.32.3's own cost model: every (algorithm, protocol) it considers.
    Model {
        #[arg(long)]
        system: String,
        #[arg(long, default_value = "all_reduce")]
        coll: String,
        #[arg(long)]
        ranks: Option<usize>,
        #[arg(short = 'b', long, default_value = "8")]
        begin: String,
        #[arg(short = 'e', long, default_value = "1G")]
        end: String,
        #[arg(short = 'f', long, default_value_t = 4)]
        factor: u64,
    },
    /// Compare against published numbers (calibration/net/published.json).
    Validate {
        file: PathBuf,
        /// transfer | packet | both
        #[arg(long, default_value = "both")]
        fidelity: String,
        #[arg(long)]
        set: Vec<String>,
        /// Write per-point results as JSON.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Only points whose id contains this.
        #[arg(long)]
        only: Option<String>,
    },
    /// Fit the NCCL latency/throughput parameters to the `fit` points.
    Fit {
        file: PathBuf,
        #[arg(long, default_value = "transfer")]
        fidelity: String,
        #[arg(long)]
        set: Vec<String>,
        #[arg(long, default_value_t = 3)]
        rounds: u32,
    },
}

pub fn load_system(s: &str) -> Result<System> {
    let looks_like_file = s.ends_with(".yaml") || s.ends_with(".yml") || s.contains('/');
    let spec = if looks_like_file || Path::new(s).exists() {
        SystemSpec::from_yaml(&super::read(Path::new(s))?)?
    } else {
        presets::system(s).with_context(|| {
            format!(
                "unknown system `{s}` (presets: {})",
                presets::SYSTEM_PRESETS.join(", ")
            )
        })?
    };
    Ok(spec.build()?)
}

fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim();
    let (num, mul) = match s.chars().last() {
        Some('K') | Some('k') => (&s[..s.len() - 1], 1u64 << 10),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1 << 20),
        Some('G') | Some('g') => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    Ok(num
        .parse::<u64>()
        .with_context(|| format!("bad size `{s}`"))?
        * mul)
}

fn sizes(b: &str, e: &str, f: u64) -> Result<Vec<u64>> {
    let (mut x, e) = (parse_size(b)?, parse_size(e)?);
    let mut v = vec![];
    while x <= e {
        v.push(x);
        x *= f.max(2);
    }
    Ok(v)
}

fn kind_of(s: &str) -> Result<CollKind> {
    serde_json::from_value(Value::String(s.into()))
        .with_context(|| format!("unknown collective `{s}`"))
}

fn algo_of(s: &str) -> Result<Algo> {
    serde_json::from_value(Value::String(s.into()))
        .with_context(|| format!("unknown algorithm `{s}`"))
}

fn proto_of(s: &str) -> Result<Proto> {
    serde_json::from_value(Value::String(s.into()))
        .with_context(|| format!("unknown protocol `{s}`"))
}

fn fidelity_of(s: &str) -> Result<Fidelity> {
    serde_json::from_value(Value::String(s.into()))
        .with_context(|| format!("unknown fidelity `{s}`"))
}

/// Apply `a.b.c=value` overrides to a NetConfig.
pub fn apply_sets(cfg: &NetConfig, sets: &[String]) -> Result<NetConfig> {
    let mut v = serde_json::to_value(cfg)?;
    for s in sets {
        let (path, val) = s
            .split_once('=')
            .with_context(|| format!("--set wants key=value, got `{s}`"))?;
        let val: Value =
            serde_json::from_str(val).unwrap_or_else(|_| Value::String(val.to_string()));
        let mut cur = &mut v;
        let parts: Vec<&str> = path.split('.').collect();
        for (i, p) in parts.iter().enumerate() {
            if i + 1 == parts.len() {
                cur[*p] = val.clone();
            } else {
                cur = cur
                    .get_mut(*p)
                    .with_context(|| format!("no config field `{path}`"))?;
            }
        }
    }
    Ok(serde_json::from_value(v)?)
}

fn human(b: u64) -> String {
    if b >= 1 << 30 && b.is_multiple_of(1 << 30) {
        format!("{}G", b >> 30)
    } else if b >= 1 << 20 && b.is_multiple_of(1 << 20) {
        format!("{}M", b >> 20)
    } else if b >= 1 << 10 && b.is_multiple_of(1 << 10) {
        format!("{}K", b >> 10)
    } else {
        format!("{b}")
    }
}

fn dur_us(r: &CollRecord) -> f64 {
    r.done_ps.saturating_sub(r.last_post_ps) as f64 / 1e6
}

pub fn main(cmd: NetCmd) -> Result<()> {
    match cmd {
        NetCmd::Sweep {
            system,
            coll,
            ranks,
            algo,
            proto,
            fidelity,
            begin,
            end,
            factor,
            set,
            json,
        } => {
            let sys = load_system(&system)?;
            let mut cfg = apply_sets(&NetConfig::default(), &set)?;
            cfg.fidelity = fidelity_of(&fidelity)?;
            let kind = kind_of(&coll)?;
            let algo = algo_of(&algo)?;
            let proto = proto.as_deref().map(proto_of).transpose()?;
            let n = ranks.unwrap_or(sys.num_gpus());
            let rows: Vec<CollRecord> = sizes(&begin, &end, factor)?
                .par_iter()
                .map(|&b| {
                    let mut c = Collective::new(kind, b, (0..n).collect(), algo);
                    c.proto = proto;
                    run_one(&sys, &cfg, &c)
                })
                .collect();
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
                return Ok(());
            }
            println!("# {} — {coll}, {n} ranks, {fidelity} fidelity", sys.name);
            println!(
                "{:>10} {:>11} {:>10} {:>10} {:>8} {:>6} {:>4} {:>11}",
                "size", "time(us)", "algbw", "busbw", "algo", "proto", "ch", "nccl(us)"
            );
            for r in rows {
                let t = dur_us(&r);
                println!(
                    "{:>10} {:>11.2} {:>10.2} {:>10.2} {:>8} {:>6} {:>4} {:>11.2}",
                    human(r.bytes),
                    t,
                    r.bytes as f64 / t / 1e3,
                    r.busbw_gbps,
                    r.algo,
                    r.proto,
                    r.channels,
                    r.nccl_model_us
                );
            }
        }
        NetCmd::Model {
            system,
            coll,
            ranks,
            begin,
            end,
            factor,
        } => {
            let sys = load_system(&system)?;
            let kind = kind_of(&coll)?;
            let n = ranks.unwrap_or(sys.num_gpus());
            let e = NetEngine::new(&sys, NetConfig::default(), false);
            let c = Collective::new(kind, 1, (0..n).collect(), Algo::Auto);
            let shape = e.comm_shape(&c);
            println!("# NCCL 2.32.3 cost model — {} {coll}, {n} ranks", sys.name);
            println!("# shape: {shape:?}");
            for b in sizes(&begin, &end, factor)? {
                let cands = nccl::candidates(kind, b, &shape, 64);
                let line: Vec<String> = cands
                    .iter()
                    .take(4)
                    .map(|p| {
                        let bus = bus_bytes(kind, b, n) / p.time_us / 1e3;
                        format!(
                            "{:?}/{} {:.1}us ({:.0} GB/s busbw)",
                            p.algo,
                            p.proto.name(),
                            p.time_us,
                            bus
                        )
                    })
                    .collect();
                println!("{:>8}  {}", human(b), line.join(" | "));
            }
        }
        NetCmd::Validate {
            file,
            fidelity,
            set,
            out,
            only,
        } => {
            let cfg = apply_sets(&NetConfig::default(), &set)?;
            let pts = load_points(&file, only.as_deref())?;
            let fids: Vec<Fidelity> = match fidelity.as_str() {
                "both" => vec![Fidelity::Transfer, Fidelity::Packet],
                f => vec![fidelity_of(f)?],
            };
            let res = evaluate(&pts, &cfg, &fids)?;
            print_table(&pts, &res, &fids);
            if let Some(o) = out {
                let rows: Vec<Value> = pts
                    .iter()
                    .zip(&res)
                    .map(|(p, r)| {
                        serde_json::json!({
                            "id": p.id, "class": p.class, "role": p.role, "metric": p.metric,
                            "reference": p.value, "source": p.source,
                            "sim": r.iter().map(|x| x.as_ref().map(|y| y.0)).collect::<Vec<_>>(),
                            "algo": r.iter().map(|x| x.as_ref().map(|y| y.1.clone())).collect::<Vec<_>>(),
                            "fidelity": fids.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>(),
                        })
                    })
                    .collect();
                std::fs::write(&o, serde_json::to_string_pretty(&rows)?)?;
                eprintln!("wrote {}", o.display());
            }
        }
        NetCmd::Fit {
            file,
            fidelity,
            set,
            rounds,
        } => fit(&file, fidelity_of(&fidelity)?, &set, rounds)?,
    }
    Ok(())
}

/// One published number.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    pub id: String,
    pub system: String,
    pub kind: CollKind,
    #[serde(default)]
    pub ranks: Option<usize>,
    pub bytes: u64,
    #[serde(default)]
    pub algo: Algo,
    #[serde(default)]
    pub proto: Option<Proto>,
    /// busbw_gbps | algbw_gbps | time_us | gbps (bytes x bytes_factor / time)
    pub metric: String,
    #[serde(default = "one")]
    pub bytes_factor: f64,
    pub value: f64,
    /// measured | nccl_model | bound
    pub class: String,
    /// fit | holdout
    #[serde(default = "holdout")]
    pub role: String,
    pub source: String,
    #[serde(default)]
    pub quote: String,
    #[serde(default)]
    pub set: Vec<String>,
}

fn one() -> f64 {
    1.0
}
fn holdout() -> String {
    "holdout".into()
}

#[derive(Deserialize)]
struct PointFile {
    points: Vec<Point>,
}

pub fn load_points(file: &Path, only: Option<&str>) -> Result<Vec<Point>> {
    let f: PointFile = serde_json::from_str(&std::fs::read_to_string(file)?)
        .with_context(|| format!("parsing {}", file.display()))?;
    Ok(f.points
        .into_iter()
        .filter(|p| only.is_none_or(|o| p.id.contains(o)))
        .collect())
}

fn metric(p: &Point, r: &CollRecord) -> f64 {
    let t = dur_us(r);
    match p.metric.as_str() {
        "busbw_gbps" => r.busbw_gbps,
        "algbw_gbps" => r.bytes as f64 / t / 1e3,
        "time_us" => t,
        "gbps" => r.bytes as f64 * p.bytes_factor / t / 1e3,
        m => panic!("unknown metric {m}"),
    }
}

type Res = Vec<Vec<Option<(f64, String)>>>;

fn evaluate(pts: &[Point], cfg: &NetConfig, fids: &[Fidelity]) -> Result<Res> {
    let mut systems = std::collections::HashMap::new();
    for p in pts {
        if !systems.contains_key(&p.system) {
            systems.insert(p.system.clone(), load_system(&p.system)?);
        }
    }
    let jobs: Vec<(usize, usize)> = (0..pts.len())
        .flat_map(|i| (0..fids.len()).map(move |j| (i, j)))
        .collect();
    let out: Vec<((usize, usize), Option<(f64, String)>)> = jobs
        .par_iter()
        .map(|&(i, j)| {
            let p = &pts[i];
            let sys = &systems[&p.system];
            let mut c = apply_sets(cfg, &p.set).expect("point set");
            c.fidelity = fids[j];
            let n = p.ranks.unwrap_or(sys.num_gpus());
            let mut col = Collective::new(p.kind, p.bytes, (0..n).collect(), p.algo);
            col.proto = p.proto;
            let r = run_one(sys, &c, &col);
            let v = metric(p, &r);
            ((i, j), Some((v, format!("{}/{}", r.algo, r.proto))))
        })
        .collect();
    let mut res = vec![vec![None; fids.len()]; pts.len()];
    for ((i, j), v) in out {
        res[i][j] = v;
    }
    Ok(res)
}

fn err(p: &Point, sim: f64) -> f64 {
    sim / p.value - 1.0
}

fn print_table(pts: &[Point], res: &Res, fids: &[Fidelity]) {
    print!("{:<44} {:<10} {:<7} {:>9}", "point", "class", "role", "ref");
    for f in fids {
        print!(" {:>9} {:>7} {:<13}", format!("{f:?}"), "err", "algo");
    }
    println!();
    let mut stats: std::collections::BTreeMap<(String, String, usize), Vec<f64>> =
        Default::default();
    for (p, r) in pts.iter().zip(res) {
        print!(
            "{:<44} {:<10} {:<7} {:>9.2}",
            p.id, p.class, p.role, p.value
        );
        for (j, x) in r.iter().enumerate() {
            match x {
                Some((v, a)) => {
                    let e = err(p, *v);
                    let flag = if p.class == "bound" {
                        if *v >= p.value { " ok" } else { " FAIL" }.to_string()
                    } else {
                        format!("{:>+6.1}%", e * 100.0)
                    };
                    print!(" {:>9.2} {:>7} {:<13}", v, flag, a);
                    if p.class != "bound" {
                        stats
                            .entry((p.class.clone(), p.role.clone(), j))
                            .or_default()
                            .push(e.abs());
                    }
                }
                None => print!(" {:>9} {:>7} {:<13}", "-", "-", "-"),
            }
        }
        println!();
    }
    println!();
    for ((class, role, j), v) in stats {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = s[s.len() / 2];
        println!(
            "{:?} {class}/{role}: n={} mean |err| {:.1}%, median {:.1}%, max {:.1}%",
            fids[j],
            v.len(),
            mean * 100.0,
            med * 100.0,
            s.last().unwrap() * 100.0
        );
    }
}

/// Parameters the fit may move, as (path, index) into the NetConfig JSON.
const FIT_PARAMS: &[(&str, Option<usize>)] = &[
    ("nccl.launch_ns", Some(0)),
    ("nccl.launch_ns", Some(1)),
    ("nccl.launch_ns", Some(2)),
    ("nccl.step_ns", Some(0)),
    ("nccl.step_ns", Some(1)),
    ("nccl.step_ns", Some(2)),
    ("nccl.net_step_ns", Some(0)),
    ("nccl.net_step_ns", Some(2)),
    ("nccl.nvls_launch_ns", None),
];

fn get_param(v: &Value, path: &str, idx: Option<usize>) -> f64 {
    let mut cur = v;
    for p in path.split('.') {
        cur = &cur[p];
    }
    match idx {
        Some(i) => cur[i].as_f64().unwrap(),
        None => cur.as_f64().unwrap(),
    }
}

fn set_param(v: &mut Value, path: &str, idx: Option<usize>, x: f64) {
    let mut cur = v;
    for p in path.split('.') {
        cur = &mut cur[p];
    }
    match idx {
        Some(i) => cur[i] = serde_json::json!(x),
        None => *cur = serde_json::json!(x),
    }
}

fn objective(pts: &[Point], cfg: &NetConfig, f: Fidelity) -> Result<f64> {
    let res = evaluate(pts, cfg, &[f])?;
    let v: Vec<f64> = pts
        .iter()
        .zip(&res)
        .filter_map(|(p, r)| r[0].as_ref().map(|x| (x.0 / p.value).ln().abs()))
        .collect();
    Ok(v.iter().sum::<f64>() / v.len().max(1) as f64)
}

fn fit(file: &Path, f: Fidelity, sets: &[String], rounds: u32) -> Result<()> {
    let base = apply_sets(&NetConfig::default(), sets)?;
    let pts: Vec<Point> = load_points(file, None)?
        .into_iter()
        .filter(|p| p.role == "fit" && p.class != "bound")
        .collect();
    if pts.is_empty() {
        bail!("no fit points");
    }
    let mut v = serde_json::to_value(&base)?;
    let mut best = objective(&pts, &base, f)?;
    eprintln!("start: mean |ln err| = {best:.4} over {} points", pts.len());
    let mut step = 0.5f64;
    for round in 0..rounds {
        for &(path, idx) in FIT_PARAMS {
            let x0 = get_param(&v, path, idx);
            for mul in [1.0 + step, 1.0 / (1.0 + step)] {
                let mut t = v.clone();
                set_param(&mut t, path, idx, (x0 * mul).max(1.0));
                let cfg: NetConfig = serde_json::from_value(t.clone())?;
                let o = objective(&pts, &cfg, f)?;
                if o < best - 1e-5 {
                    best = o;
                    v = t;
                    eprintln!(
                        "  round {round}: {path}[{idx:?}] -> {:.0}  obj {o:.4}",
                        x0 * mul
                    );
                    break;
                }
            }
        }
        step *= 0.5;
    }
    let cfg: NetConfig = serde_json::from_value(v)?;
    println!("final mean |ln err| = {best:.4}");
    println!("{}", serde_json::to_string_pretty(&cfg.nccl)?);
    Ok(())
}
