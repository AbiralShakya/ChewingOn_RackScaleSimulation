#!/usr/bin/env python3
"""Moved to python/rackscale/nvbit/trace_to_rsim.py; this runs it without installing rackscale."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from rackscale.nvbit.trace_to_rsim import main  # noqa: E402

if __name__ == "__main__":
    main()
