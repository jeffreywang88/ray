# Loaded by every Python process with ~/bench on PYTHONPATH.
# - BENCH_INSTRUMENT=1 (LLMRouter only): install the benchmark probes.
# - BENCH_BLOCK_IMPORTS=a,b (mock replicas only): make those top-level packages
#   unimportable. Serve imports pandas (and through it pyarrow) only to register
#   an optional DataFrame JSON encoder, ~38 MiB per process; the mock replicas never
#   use it, and 1024 of them do not fit in host memory otherwise.
import os
import sys

if os.environ.get("BENCH_INSTRUMENT") in ("1", "stats"):
    import instrument  # noqa: F401

_blocked = {m for m in os.environ.get("BENCH_BLOCK_IMPORTS", "").split(",") if m}
if _blocked:
    import importlib.abc

    class _BlockImports(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in _blocked:
                raise ModuleNotFoundError(f"{name} is blocked in benchmark mock replicas")
            return None

    sys.meta_path.insert(0, _BlockImports())

# BENCH_TRACE=1 (whole cluster): stamp each routed request in HAProxy with wall-clock
# microseconds: x-bt-hq once its headers are parsed, x-bt-h0 right before the Lua
# /internal/route call (after wait-for-body), x-bt-h1 right after it. The engine logs them.
if os.environ.get("BENCH_TRACE") == "1":
    import importlib.abc as _abc

    _HAPROXY_TEMPLATES = "ray.serve._private.haproxy_templates"

    def _trace_haproxy(mod):
        t = mod.HAPROXY_CONFIG_TEMPLATE
        anchor_q = "http-request set-header x-request-id %[unique-id] if !{ req.hdr(x-request-id) -m found }"
        anchor_lua = "    http-request lua.route_via_ingress_request_router if has_ingress_request_router_app"
        if anchor_q not in t or anchor_lua not in t:
            print("BENCH_TRACE: HAProxy template anchors not found; no HAProxy stamps", file=sys.stderr)
            return
        t = t.replace(anchor_q, anchor_q + "\n    http-request set-header x-bt-hq %[date(0,us)]", 1)
        i = t.index(anchor_lua)
        eol = t.index("\n", i)
        t = (
            t[:i]
            + "    http-request set-header x-bt-h0 %[date(0,us)] if has_ingress_request_router_app\n"
            + t[i:eol]
            + "\n    http-request set-header x-bt-h1 %[date(0,us)] if has_ingress_request_router_app"
            + t[eol:]
        )
        mod.HAPROXY_CONFIG_TEMPLATE = t

    class _TraceHAProxy(_abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _HAPROXY_TEMPLATES:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is None:
                    continue
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)
                    _trace_haproxy(module)

                spec.loader.exec_module = exec_module
                return spec
            return None

    sys.meta_path.insert(0, _TraceHAProxy())
