"""Where the skill and the verifier agent live, and their text for hosts that take it as a prompt.

The plugin directory is the one source of truth: Claude Code loads it as a plugin
(``claude --plugin-dir``), ``nninteractive-agent setup`` copies the skill into Codex or Claude Code,
the MCP server offers it as a prompt and resources, and the web demo appends it to the system prompt.
"""

from __future__ import annotations

from pathlib import Path

PLUGIN_DIR = Path(__file__).parent / "claude_plugin"
SKILL_NAME = "nninteractive-segmentation"
SKILL_DIR = PLUGIN_DIR / "skills" / SKILL_NAME
VERIFIER_FILE = PLUGIN_DIR / "agents" / "nninteractive-verifier.md"
# Tool names inside the Claude Code plugin, as written in the plugin's agent definitions.
PLUGIN_TOOL_PREFIX = "mcp__plugin_nninteractive_nninteractive__"


def _strip_front_matter(text: str) -> str:
    return text.split("---", 2)[2] if text.startswith("---") else text


def skill_description() -> str:
    front = (SKILL_DIR / "SKILL.md").read_text().split("---", 2)[1]
    return next(
        line.split(":", 1)[1].strip() for line in front.splitlines() if line.startswith("description:")
    )


def reference_files() -> list:
    return sorted((SKILL_DIR / "reference").glob("*.md"))


def skill_prompt(inline_references: bool = False) -> str:
    """The skill body (without front matter) plus its reference files: where they are, or their text
    for hosts without a file tool."""
    body = _strip_front_matter((SKILL_DIR / "SKILL.md").read_text())
    refs = reference_files()
    if inline_references:
        return body + "".join(f"\n# Reference file reference/{p.name}\n\n{p.read_text()}" for p in refs)
    listed = ", ".join(str(p) for p in refs)
    return (
        body.replace("`reference/", "`" + str(SKILL_DIR / "reference") + "/")
        + f"\nReference files: {listed}\n"
    )


def verifier() -> dict:
    """The read-only verifier agent: name, description, tools (plugin names) and instructions."""
    _, front, body = VERIFIER_FILE.read_text().split("---", 2)
    meta = dict(line.split(":", 1) for line in front.strip().splitlines())
    return {
        "name": meta["name"].strip(),
        "description": meta["description"].strip(),
        "tools": [t.strip() for t in meta["tools"].split(",")],
        "body": body.strip(),
    }
