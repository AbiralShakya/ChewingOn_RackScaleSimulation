//! Perfetto-compatible Chrome Trace Event JSON writer.
//!
//! Open the output in <https://ui.perfetto.dev>. Process = GPU (or switch),
//! thread = stream / SM / sub-partition. Counter tracks carry per-SM issue
//! utilisation and stall-reason breakdowns, which is how cycle-level bubbles
//! are visualised.

use crate::Time;
use serde::Serialize;
use serde_json::Value;
use std::io::Write;

#[derive(Serialize, Clone, Debug)]
pub struct TraceEvent {
    pub name: String,
    pub ph: &'static str,
    /// Microseconds (Chrome trace convention); fractional for sub-ns detail.
    pub ts: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    pub pid: u64,
    pub tid: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cat: Option<String>,
    #[serde(skip_serializing_if = "Value::is_null")]
    pub args: Value,
}

#[derive(Default, Clone, Debug)]
pub struct TraceSink {
    pub enabled: bool,
    pub events: Vec<TraceEvent>,
}

#[inline]
fn us(t: Time) -> f64 {
    t as f64 / 1.0e6
}

impl TraceSink {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, events: Vec::new() }
    }

    pub fn span(&mut self, name: impl Into<String>, cat: &str, pid: u64, tid: u64, start: Time, end: Time, args: Value) {
        if !self.enabled {
            return;
        }
        self.events.push(TraceEvent {
            name: name.into(),
            ph: "X",
            ts: us(start),
            dur: Some(us(end.saturating_sub(start))),
            pid,
            tid,
            cat: Some(cat.to_string()),
            args,
        });
    }

    pub fn counter(&mut self, name: impl Into<String>, pid: u64, t: Time, values: Value) {
        if !self.enabled {
            return;
        }
        self.events.push(TraceEvent { name: name.into(), ph: "C", ts: us(t), dur: None, pid, tid: 0, cat: None, args: values });
    }

    pub fn name_process(&mut self, pid: u64, name: &str) {
        if !self.enabled {
            return;
        }
        self.events.push(TraceEvent {
            name: "process_name".into(),
            ph: "M",
            ts: 0.0,
            dur: None,
            pid,
            tid: 0,
            cat: None,
            args: serde_json::json!({ "name": name }),
        });
    }

    pub fn name_thread(&mut self, pid: u64, tid: u64, name: &str) {
        if !self.enabled {
            return;
        }
        self.events.push(TraceEvent {
            name: "thread_name".into(),
            ph: "M",
            ts: 0.0,
            dur: None,
            pid,
            tid,
            cat: None,
            args: serde_json::json!({ "name": name }),
        });
    }

    pub fn extend(&mut self, other: TraceSink) {
        self.events.extend(other.events);
    }

    pub fn write_json<W: Write>(&self, w: W) -> std::io::Result<()> {
        let doc = serde_json::json!({ "traceEvents": self.events, "displayTimeUnit": "ns" });
        serde_json::to_writer(w, &doc).map_err(std::io::Error::other)
    }
}
