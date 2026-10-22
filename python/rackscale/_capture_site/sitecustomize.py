"""Loaded into every Python process that ``rackscale capture`` starts.

``rackscale capture`` puts this directory first on ``PYTHONPATH``, so the
interpreter imports this module at startup, before the workload's own code.
It does three things, all through the ``rsim_cap_mark`` entry point that the
NVBit capture tool (``capture/nvbit/rsim_capture.cu``) exports:

* **Region.** ``RSIM_CAP_REGION=forward:S:N[:REGEX]`` starts the capture
  region at the S-th forward of the model (0-based) and ends it at the
  (S+N)-th, so N whole engine steps are captured, including the sampler and
  input preparation that run between forwards. The model is the outermost
  ``nn.Module`` called with at least a million parameters, or the outermost
  module whose class name matches REGEX. ``marked`` leaves the region to
  :func:`rsim_capture.region` in the workload; ``all`` captures everything.
* **Steps.** Each forward of the model inside the region is marked, so a
  replay can select steps.
* **Modules.** With ``RSIM_CAP_MODULES=1`` (the default), every module call
  inside the region is marked with its qualified name, class and output size.
  The parallelism plan uses these to place collectives (for example an
  all-reduce after every ``RowParallelLinear``). It costs a few microseconds of
  host time per module call, which only matters for host-bound eager runs.

Nothing here touches CUDA: if the tool is not loaded, every hook is a no-op.
"""

import ctypes
import importlib.abc
import importlib.util
import os
import re
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))


class _State:
    lib = None
    # Timing pass (no tool, under Nsight Systems): the region starts and stops the profiler.
    profiler = os.environ.get("RSIM_CAP_PROFILER") == "1"
    active = False
    region = os.environ.get("RSIM_CAP_REGION", "marked")
    modules = os.environ.get("RSIM_CAP_MODULES", "1") == "1"
    model_re = None
    skip = 0
    count = 0
    counting = False
    arm_file = os.environ.get("RSIM_CAP_ARM_FILE", "")
    forwards = 0
    model_ids = set()
    params = {}
    qnames = {}
    tls = threading.local()


def _mark(kind, payload=""):
    if _State.lib is not None:
        _State.lib(kind.encode(), payload.encode())
    elif _State.profiler and kind in ("region_begin", "region_end"):
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            rt = torch.cuda.cudart()
            (rt.cudaProfilerStart if kind == "region_begin" else rt.cudaProfilerStop)()


def _load_tool():
    path = os.environ.get("CUDA_INJECTION64_PATH")
    if not path:
        return
    try:
        fn = ctypes.CDLL(path).rsim_cap_mark
    except (OSError, AttributeError):
        return
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    fn.restype = None
    _State.lib = fn


def arm():
    """Start counting forwards for ``--region armed:S:N`` (in every process)."""
    if _State.arm_file:
        with open(_State.arm_file, "w") as f:
            f.write("armed\n")


def begin(label=""):
    if not _State.active:
        _State.active = True
        _mark("region_begin", label)


def end(label=""):
    if _State.active:
        _State.active = False
        _mark("region_end", label)


class region:
    """Capture region around a block of the workload (no-op when not capturing)."""

    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        begin(self.label)
        return self

    def __exit__(self, *exc):
        end(self.label)
        return False


def _nparams(mod):
    n = _State.params.get(id(mod))
    if n is None:
        n = sum(p.numel() for p in mod.parameters())
        _State.params[id(mod)] = n
    return n


def _is_model(mod):
    if id(mod) in _State.model_ids:
        return True
    if _State.model_re is not None:
        ok = _State.model_re.search(type(mod).__name__) is not None
    else:
        ok = _nparams(mod) >= 1_000_000
    if ok:
        _State.model_ids.add(id(mod))
        for name, sub in mod.named_modules():
            _State.qnames.setdefault(id(sub), name or type(mod).__name__)
    return ok


def _armed():
    if _State.region.startswith("armed") and _State.arm_file and os.path.exists(_State.arm_file):
        _State.counting = True
    return _State.counting


def _out_info(out):
    t = out
    while isinstance(t, (tuple, list)) and t:
        t = t[0]
    shape = getattr(t, "shape", None)
    if shape is None:
        return ""
    try:
        nbytes = t.numel() * t.element_size()
    except Exception:
        nbytes = 0
    return "%s\t%s\t%d" % ("x".join(str(int(d)) for d in shape), str(t.dtype).replace("torch.", ""), nbytes)


def _pre(mod, args):
    st = getattr(_State.tls, "stack", None)
    if st is None:
        st = _State.tls.stack = []
    if not st:
        if _State.count and (_State.counting or _armed()) and _is_model(mod):
            k = _State.forwards
            _State.forwards += 1
            if k == _State.skip + _State.count:
                end("forward:%d" % k)
            elif k == _State.skip:
                begin("forward:%d" % k)
        if _State.active and _is_model(mod):
            _mark("step", type(mod).__name__)
    st.append(mod)
    if _State.active and _State.modules:
        _mark("push", "%s\t%s" % (_State.qnames.get(id(mod), ""), type(mod).__name__))


def _post(mod, args, out):
    st = getattr(_State.tls, "stack", None)
    if st:
        st.pop()
    if _State.active and _State.modules:
        _mark("pop", _out_info(out))


def _install(torch):
    from torch.nn.modules import module as M

    M.register_module_forward_pre_hook(_pre)
    try:
        M.register_module_forward_hook(_post, always_call=True)
    except TypeError:  # torch < 2.0
        M.register_module_forward_hook(_post)


class _TorchFinder(importlib.abc.MetaPathFinder):
    """Install the module hooks right after ``import torch`` finishes.

    Every lookup of ``torch`` gets a spec whose loader installs the hooks, and
    the finder stays until one of them is executed: libraries probe for torch
    with ``importlib.util.find_spec`` long before importing it.
    """

    busy = False

    def find_spec(self, name, path, target=None):
        if name != "torch" or self.busy:
            return None
        self.busy = True
        try:
            spec = importlib.util.find_spec("torch")
        finally:
            self.busy = False
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module
        finder = self

        def exec_and_hook(module):
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
            exec_module(module)
            try:
                _install(module)
            except Exception as e:  # never break the workload
                print("rsim capture: module hooks not installed: %s" % e, file=sys.stderr)

        spec.loader.exec_module = exec_and_hook
        return spec


def _setup():
    _load_tool()
    if _State.lib is None and not _State.profiler:
        return
    if _State.profiler:
        _State.modules = False
    r = _State.region
    if r == "all":
        _State.active = True
    elif r.startswith("forward") or r.startswith("armed"):
        _State.counting = r.startswith("forward")
        parts = r.split(":", 3)
        _State.skip = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        _State.count = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        if len(parts) > 3 and parts[3]:
            _State.model_re = re.compile(parts[3])
    if "torch" in sys.modules:
        _install(sys.modules["torch"])
    else:
        sys.meta_path.insert(0, _TorchFinder())
    import atexit

    atexit.register(end, "exit")


def _chain():
    """Run the sitecustomize this module shadows, if there is one."""
    rest = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    for p in rest:
        f = os.path.join(p, "sitecustomize.py")
        if os.path.isfile(f):
            spec = importlib.util.spec_from_file_location("_rsim_chained_sitecustomize", f)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                print("rsim capture: chained sitecustomize failed: %s" % e, file=sys.stderr)
            return


_setup()
_chain()
