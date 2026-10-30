"""Background NVML sampler: SM clock, power, temperature and throttle reasons per GPU.

    with Sampler([0, 1], period=0.01, csv="clk.csv") as s:
        s.mark("gemm 8192")          # label the samples that follow
        ...
    s.summary("gemm 8192")           # medians over that label's samples

``nvidia-smi`` polls at about 100 ms at best; NVML through pynvml (the
``nvidia-ml-py`` package) reaches ~5-10 ms, enough to see a clock settle under
the power cap. Throttle ("clocks event") reasons say why the clock is below
max: SW power cap, HW slowdown, thermal, or idle.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from typing import Dict, List, Optional

try:
    import pynvml as N
except ImportError:  # the runbook installs nvidia-ml-py; without it, sampling is skipped
    N = None

REASONS = {
    0x1: "gpu_idle", 0x2: "app_clock", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
    0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clock",
}


def physical(gpu: int) -> int:
    """NVML index of CUDA device ``gpu``: NVML ignores CUDA_VISIBLE_DEVICES, so on a shared
    machine (della-tridao's GPU 3) CUDA device 0 is NVML device 3."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [v.strip() for v in vis.split(",") if v.strip()]
    if ids and all(v.isdigit() for v in ids) and gpu < len(ids):
        return int(ids[gpu])
    return gpu


def reasons(mask: int) -> List[str]:
    return [n for b, n in REASONS.items() if mask & b]


class Sampler:
    def __init__(self, gpus: List[int], period: float = 0.01, csv: Optional[str] = None):
        self.gpus, self.period, self.csv = gpus, period, csv
        self.rows: List[tuple] = []
        self.label = ""
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def __enter__(self):
        if N is None:
            return self
        N.nvmlInit()
        self.h = [N.nvmlDeviceGetHandleByIndex(physical(g)) for g in self.gpus]
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        if self._t:
            self._stop.set()
            self._t.join()
            N.nvmlShutdown()
        if self.csv:
            with open(self.csv, "w") as f:
                f.write("t,label,gpu,sm_mhz,mem_mhz,power_w,temp_c,reasons\n")
                for r in self.rows:
                    f.write(",".join(str(x) for x in r) + "\n")

    def mark(self, label: str) -> None:
        self.label = label

    def _run(self) -> None:
        get_reasons = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            N.nvmlDeviceGetCurrentClocksThrottleReasons
        while not self._stop.is_set():
            t = time.time()
            for g, h in zip(self.gpus, self.h):
                try:
                    self.rows.append((round(t, 4), self.label, g,
                                      N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                                      N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM),
                                      N.nvmlDeviceGetPowerUsage(h) / 1000.0,
                                      N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
                                      get_reasons(h)))
                except N.NVMLError:
                    pass
            time.sleep(max(0.0, self.period - (time.time() - t)))

    def summary(self, label: str, gpu: Optional[int] = None, settle: float = 0.25) -> Dict:
        """Medians over a label's samples, skipping the first ``settle`` of its duration."""
        rows = [r for r in self.rows if r[1] == label and (gpu is None or r[2] == gpu)]
        if not rows:
            return {}
        t0, t1 = rows[0][0], rows[-1][0]
        rows = [r for r in rows if r[0] >= t0 + settle * (t1 - t0)] or rows
        mask = 0
        for r in rows:
            mask |= r[7]
        capped = sum(1 for r in rows if r[7] & 0x4) / len(rows)
        return dict(samples=len(rows), sm_mhz=statistics.median(r[3] for r in rows), sm_mhz_min=min(r[3] for r in rows),
                    power_w=statistics.median(r[5] for r in rows), power_w_max=max(r[5] for r in rows),
                    temp_c=max(r[6] for r in rows), power_capped_frac=round(capped, 3), reasons=reasons(mask))


def limits(gpu: int) -> Dict:
    """Power limit and max clocks of one GPU, for the environment record."""
    if N is None:
        return {}
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(physical(gpu))
    out = dict(power_limit_w=N.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
               max_sm_mhz=N.nvmlDeviceGetMaxClockInfo(h, N.NVML_CLOCK_SM),
               max_mem_mhz=N.nvmlDeviceGetMaxClockInfo(h, N.NVML_CLOCK_MEM))
    N.nvmlShutdown()
    return out
