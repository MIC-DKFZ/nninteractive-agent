"""Serve the agent tools over stdio MCP, for Codex, Claude Code or any other MCP client.

    nninteractive-agent-mcp [--server URL | --server local] [--python full|guarded|off] [--workdir DIR]

``nninteractive-agent setup`` registers it with Codex or Claude Code. Besides the tools it offers
the nninteractive-segmentation skill as an MCP prompt and its files as resources, for clients
that do not load skills.

The workspace (image, nnInteractive session, objects) lives in this process for the whole
agent session, which is what keeps the server-side nnInteractive lease and the interaction
history alive between tool calls.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .core.pyguard import PYTHON_MODES, describe
from .core.session import model_license
from .core.workspace import AgentWorkspace
from .skill import SKILL_DIR, SKILL_NAME, reference_files, skill_description, skill_prompt
from .tools import call_tool, tools_for

SKILL_URI = "nninteractive://skill/"
INSTRUCTIONS = (
    "Interactive segmentation with nnInteractive. Load an image, then segment each structure as its own "
    "object (new_object -> prompts -> check -> corrections -> commit). Follow the nninteractive-segmentation "
    "skill for the workflow and prompting rules; if your host has not loaded it, read the MCP prompt "
    f"'{SKILL_NAME}' (or the resource {SKILL_URI}SKILL.md) before the first prompt. load_image reports the "
    "license of the nnInteractive weights in use; tell the user once."
)


def _skill_files() -> dict:
    """Resource URI -> file: SKILL.md and its reference files."""
    files = [SKILL_DIR / "SKILL.md", *reference_files()]
    return {SKILL_URI + f.relative_to(SKILL_DIR).as_posix(): f for f in files}


def build_server(ws: AgentWorkspace) -> Server:
    tools = tools_for(ws.python_mode)

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=t.name, description=t.description, input_schema=t.input_schema,
                    annotations=types.ToolAnnotations(readOnlyHint=t.read_only),
                )  # fmt: skip
                for t in tools
            ]
        )

    async def list_prompts(ctx, params) -> types.ListPromptsResult:
        return types.ListPromptsResult(prompts=[types.Prompt(name=SKILL_NAME, description=skill_description())])

    async def get_prompt(ctx, params) -> types.GetPromptResult:
        if params.name != SKILL_NAME:
            raise ValueError(f"unknown prompt {params.name!r}")
        text = skill_prompt(inline_references=True)
        return types.GetPromptResult(
            description=skill_description(),
            messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))],
        )

    async def list_resources(ctx, params) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[types.Resource(uri=uri, name=f.name, mime_type="text/markdown") for uri, f in _skill_files().items()]
        )

    async def read_resource(ctx, params) -> types.ReadResourceResult:
        f = _skill_files().get(str(params.uri))
        if f is None:
            raise ValueError(f"unknown resource {params.uri}")
        return types.ReadResourceResult(
            contents=[types.TextResourceContents(uri=str(params.uri), text=f.read_text(), mime_type="text/markdown")]
        )

    announced = []

    async def call(ctx, params) -> types.CallToolResult:
        # Tool work blocks (network, numpy, rendering): keep it off the event loop.
        out = await anyio.to_thread.run_sync(call_tool, ws, params.name, params.arguments)
        if ws.session is not None and not announced:
            announced.append(True)
            print(f"nnInteractive ({ws.session_spec}): model license {model_license(ws.session)}", file=sys.stderr)
        content: list = [types.TextContent(type="text", text=out.text)]
        for png in out.images:
            content.append(types.ImageContent(type="image", data=base64.b64encode(png).decode(), mime_type="image/png"))
        return types.CallToolResult(content=content, is_error=out.is_error)

    return Server("nninteractive", instructions=INSTRUCTIONS, on_list_tools=list_tools, on_call_tool=call,
                  on_list_prompts=list_prompts, on_get_prompt=get_prompt, on_list_resources=list_resources,
                  on_read_resource=read_resource)  # fmt: skip


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=None, help="nninteractive-server URL or 'local' (default: "
                    "$NNINTERACTIVE_SERVER_URL, else local inference)")  # fmt: skip
    ap.add_argument("--python", choices=PYTHON_MODES, default=None, help="the python tool: full (default), "
                    "guarded (computations only) or off; default $NNINTERACTIVE_AGENT_PYTHON")  # fmt: skip
    ap.add_argument("--workdir", default=os.environ.get("NNINTERACTIVE_AGENT_WORKDIR"),
                    help="where renders, the tool log and outputs go (default: ./runs/<timestamp>)")  # fmt: skip
    args = ap.parse_args(argv)
    # stdout is the MCP channel, but libraries print to it (nnInteractive's local session reports
    # progress there). Keep a private handle on the real stdout for MCP and send everything else,
    # including output from C extensions, to stderr.
    mcp_stdout = io.TextIOWrapper(os.fdopen(os.dup(sys.stdout.fileno()), "wb"), encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    ws = AgentWorkspace(session_spec=args.server, workdir=args.workdir, python_mode=args.python)
    server = build_server(ws)
    print(f"nninteractive-agent MCP server; workdir {ws.workdir}; {describe(ws.python_mode)}", file=sys.stderr)

    async def run():
        async with stdio_server(stdout=anyio.wrap_file(mcp_stdout)) as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        anyio.run(run)
    finally:
        ws.close()


if __name__ == "__main__":
    main()
