"""Terminal helpers for the command line: colour (off for pipes and ``NO_COLOR``) and a progress line."""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager


def _tty(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty() and not os.environ.get("NO_COLOR") \
        and os.environ.get("TERM") != "dumb"


def style(text: str, *codes: str, stream=sys.stdout) -> str:
    """``style("x", "bold", "red")``; plain text when ``stream`` is not a terminal."""
    if not _tty(stream):
        return text
    table = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36"}
    return "\033[" + ";".join(table[c] for c in codes) + "m" + text + "\033[0m"


def colorize_summary(text: str) -> str:
    """Highlight the headline, warnings and section headers of ``Report.summary()``."""
    if not _tty(sys.stdout):
        return text
    out = []
    for line in text.splitlines():
        if line.startswith("== "):
            line = style(line, "bold")
        elif line.startswith("makespan"):
            line = style(line, "bold", "green")
        elif line.startswith("WARNING"):
            line = style(line, "yellow")
        elif line.startswith(("kernel ", "collective ", "scheduler-cycle")):
            line = style(line, "bold", "cyan")
        elif line.startswith("[simulated"):
            line = style(line, "dim")
        out.append(line)
    return "\n".join(out)


@contextmanager
def progress(label: str):
    """Show ``label`` with a spinner and elapsed time on stderr while the body runs.

    The simulator releases the GIL, so the spinner thread keeps ticking during a run.
    Silent when stderr is not a terminal (scripts, CI, pipes).
    """
    if not _tty(sys.stderr):
        yield
        return
    stop = threading.Event()
    t0 = time.monotonic()

    def spin():
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if stop.wait(0.1):
                break
            sys.stderr.write(f"\r\033[2K{style(ch, 'cyan', stream=sys.stderr)} {label}  "
                             f"{style(f'{time.monotonic() - t0:.1f}s', 'dim', stream=sys.stderr)}")
            sys.stderr.flush()

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    ok = False
    try:
        yield
        ok = True
    finally:
        stop.set()
        th.join()
        mark = style("✓", "green", stream=sys.stderr) if ok else style("✗", "red", stream=sys.stderr)
        sys.stderr.write(f"\r\033[2K{mark} {label}  {time.monotonic() - t0:.1f}s\n")
        sys.stderr.flush()
