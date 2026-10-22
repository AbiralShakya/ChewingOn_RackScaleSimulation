"""FA4 (and FA2) on B300: kernel-total cycles vs ncu, plus per-role
(warp index) issue counts and state shares, for a given calibration.

    python3 fa4_validate.py <calibration.json> <tag> [--cp]

Uses research/flash_attention/run_study.py's recordings and ncu ground
truth; simulations are cached under $FA_VALIDATE_CACHE/<tag>/ (default
/tmp/rsim-fa-validate), outside the repository.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FA = os.path.join(HERE, "..", "..", "research", "flash_attention")
sys.path.insert(0, FA)
import run_study as rs  # noqa: E402


def main():
    cal, tag = sys.argv[1], sys.argv[2]
    rs.SIM = os.path.join(os.environ.get("FA_VALIDATE_CACHE", "/tmp/rsim-fa-validate"), tag)
    orig = rs.simulate

    def sim(key, rpath, gpu, regs=None, cal_path=None):
        return orig(key, rpath, gpu, regs, os.path.abspath(cal) if gpu == "b300" else cal_path)
    rs.simulate = sim
    entries = [e for e in rs.VALIDATE if e[0] == "b300"]
    if "--cp" in sys.argv:
        entries += [e for e in rs.VALIDATE_CP if e[0] == "b300"]
    rows = rs.validate(entries)
    errs = [abs(r["cycle_error"]) for r in rows if r.get("cycle_error") is not None]
    print(f"{tag}: {len(errs)} kernels, mean |err| {100 * sum(errs) / len(errs):.1f}%, max {100 * max(errs):.1f}%")
    json.dump(rows, open(os.path.join(rs.SIM, "rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
