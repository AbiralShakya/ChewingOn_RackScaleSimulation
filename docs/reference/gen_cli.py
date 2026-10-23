"""Regenerate docs/reference/cli.md from the installed `rackscale` command's --help.

    python docs/reference/gen_cli.py
"""

import os
import re
import subprocess

OUT = os.path.join(os.path.dirname(__file__), "cli.md")
PY_CMDS = ["demo", "decode", "prefill", "train", "capture", "replay", "run", "models", "view"]
RSIM_GROUPS = {"net": None, "viz": None}


def helptext(*args: str) -> str:
    env = dict(os.environ, NO_COLOR="1", COLUMNS="100")
    out = subprocess.run(["rackscale", *args, "--help"], capture_output=True, text=True, env=env)
    return out.stdout.rstrip()


def subcommands(group: str) -> list:
    text = helptext(group)
    body = text.split("Commands:", 1)[1].split("Options:", 1)[0]
    return [m.group(1) for m in re.finditer(r"^  (\S+)\s", body, re.M) if m.group(1) != "help"]


def main() -> None:
    from rackscale.cli import RSIM_COMMANDS

    parts = ["# Command line", "",
             "Generated from `--help` by `python docs/reference/gen_cli.py`. Every command below is also",
             "available as `rsim <command>` for the simulator-core ones, which is the same program as the",
             "standalone Rust binary.", "",
             "```text", helptext(), "```", ""]
    parts += ["## Model and workload commands", ""]
    for c in PY_CMDS:
        parts += [f"### `rackscale {c}`", "", "```text", helptext(c), "```", ""]
    parts += ["## Simulator-core commands", ""]
    for c in RSIM_COMMANDS:
        parts += [f"### `rackscale {c}`", "", "```text", helptext(c), "```", ""]
        if c in RSIM_GROUPS:
            for s in subcommands(c):
                parts += [f"#### `rackscale {c} {s}`", "", "```text", helptext(c, s), "```", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
