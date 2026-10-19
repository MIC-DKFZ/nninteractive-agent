"""Text-prompted agentic segmentation with nnInteractive.

The package is split so the core never imports Qt or napari:

* ``core``: image I/O, the nnInteractive session, prompt rasterization, metrics and
  off-screen rendering, tied together by ``AgentWorkspace``.
* ``tools``: the agent-facing tool set (names, JSON schemas, handlers) on top of a workspace.
* ``mcp_server``: serves the tools over stdio MCP for Claude Code.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nninteractive-agent")
except PackageNotFoundError:
    __version__ = "unknown"
