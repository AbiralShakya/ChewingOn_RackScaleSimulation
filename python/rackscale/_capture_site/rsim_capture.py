"""Capture-region markers for workloads run under ``rackscale capture``.

    try:
        from rsim_capture import region
    except ImportError:  # not running under rackscale capture
        from contextlib import nullcontext as region

    with region("decode"):
        llm.generate(prompts, params)

The module is on ``PYTHONPATH`` only while ``rackscale capture`` runs the
workload, and the markers do nothing unless the NVBit tool is loaded.
"""

from sitecustomize import arm, begin, end, region  # noqa: F401
