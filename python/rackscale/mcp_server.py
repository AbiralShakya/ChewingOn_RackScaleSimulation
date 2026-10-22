"""MCP server: the rackscale agent API as Model Context Protocol tools.

    pip install 'rackscale[agent]'      # or: pip install mcp
    rackscale-mcp                        # stdio transport

Register it with an MCP client, for example Claude Code::

    claude mcp add rackscale -- rackscale-mcp

and then ask things like "simulate DeepSeek-V3 decode on GB200 NVL72 at EP72 and tell me
what bounds it". The tools are defined once in ``rackscale.tools``; see
docs/AGENT_API.md.
"""

from __future__ import annotations

import argparse
import functools
import inspect
from typing import Annotated, Any, Optional

from pydantic import Field

from . import tools


def _server_class():
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp import FastMCP
            return FastMCP
        except ImportError as e:
            raise SystemExit("rackscale-mcp needs the MCP SDK: pip install mcp") from e


_TYPES = {"string": str, "integer": int, "number": float, "object": dict, "array": list}


def _wrap(t: dict):
    """A named function with the tool's parameters, so the SDK can derive its schema."""
    props = t["input_schema"]["properties"]
    required = set(t["input_schema"]["required"])
    params = []
    for name, p in props.items():
        typ = p.get("type")
        ann: Any = Any if isinstance(typ, list) else _TYPES.get(typ, Any)
        if name not in required and ann is not Any:
            ann = Optional[ann]
        if p.get("description"):
            ann = Annotated[ann, Field(description=p["description"])]
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=ann,
                                        default=inspect.Parameter.empty if name in required else None))

    @functools.wraps(t["fn"])
    def fn(**kw):
        return tools.call(t["name"], {k: v for k, v in kw.items() if v is not None})

    fn.__name__ = t["name"]
    fn.__doc__ = t["description"]
    fn.__signature__ = inspect.Signature(params, return_annotation=dict)
    fn.__annotations__ = {**{p.name: p.annotation for p in params}, "return": dict}
    return fn


def build():
    server = _server_class()(name="rackscale", instructions=tools.INSTRUCTIONS)
    for t in tools.TOOLS:
        server.tool(name=t["name"], description=t["description"])(_wrap(t))
    return server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rackscale-mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    a = ap.parse_args(argv)
    build().run(transport=a.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
