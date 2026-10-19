"""Modes of the agent's ``python`` tool, and the checks of ``--python guarded``.

    full     (default) the agent's code runs unrestricted, with the rights of the user who started
             the tool server: it can read and write files, start programs and use the network.
    guarded  only computations: imports from an allowlist, no file or process access, no
             introspection tricks (see ``check``).
    off      no python tool.

Python cannot be sandboxed inside the process that runs it, so ``guarded`` is a guardrail against
accidents (an agent writing figures into the working directory, deleting files) and against simple
prompt injection, not a security boundary. Isolation needs a container, a VM or a dedicated user.
"""

from __future__ import annotations

import ast
import builtins
import os

PYTHON_MODES = ("full", "guarded", "off")
ENV_VAR = "NNINTERACTIVE_AGENT_PYTHON"

ALLOWED_MODULES = frozenset({
    "numpy", "scipy", "skimage", "matplotlib", "math", "cmath", "statistics", "random", "collections",
    "itertools", "functools", "operator", "heapq", "bisect", "re", "json", "typing", "copy", "time", "datetime",
    "textwrap", "pprint", "enum", "fractions", "decimal",
})  # fmt: skip
BLOCKED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "breakpoint", "input", "help", "exit", "quit", "memoryview",
})  # fmt: skip
# Attributes that reach files, processes, other modules, the interpreter's frames or the session
# behind the workspace. Names starting with "_" are blocked as well.
BLOCKED_ATTRS = frozenset({
    # files and processes
    "savefig", "imsave", "imwrite", "imread", "tofile", "fromfile", "fromregex", "memmap", "genfromtxt",
    "system", "popen", "spawn", "fork", "kill", "unlink", "rmdir", "mkdir", "makedirs", "rename", "touch",
    "chmod", "symlink_to", "hardlink_to", "iterdir", "glob", "rglob", "workdir",
    # modules that lead there
    "os", "sys", "subprocess", "shutil", "pathlib", "io", "ctypes", "ctypeslib", "importlib", "builtins",
    "f2py", "distutils", "DataSource", "npyio", "backends", "animation", "testing", "cbook", "datasets",
    # attribute lookup by name, frames and code objects
    "attrgetter", "methodcaller", "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await",
    "ag_frame", "ag_code", "ag_await", "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "f_trace",
    "tb_frame", "tb_next", "co_code", "co_consts",
    # the workspace's internals
    "session", "run_python",
})  # fmt: skip
BLOCKED_PREFIXES = ("save", "load", "open", "read", "write", "dump", "print_")
NOT_BLOCKED = frozenset({"opening", "loads", "dumps"})  # skimage.morphology.opening, json.loads/dumps
# Workspace methods that are tools of their own, so allowed on ``ws`` despite a blocked prefix.
WS_METHODS = frozenset({"save", "load_image"})
PROTECTED = frozenset({"ws", "show"})


class GuardError(Exception):
    """The code breaks a rule of guarded mode; the message lists every violation."""


def resolve_mode(mode: str | None = None) -> str:
    """``mode``, else ``$NNINTERACTIVE_AGENT_PYTHON``, else ``full``."""
    mode = (mode or os.environ.get(ENV_VAR) or "full").strip().lower()
    if mode not in PYTHON_MODES:
        raise ValueError(f"python mode must be one of {', '.join(PYTHON_MODES)}, got {mode!r}")
    return mode


def _blocked_attr(attr: str) -> bool:
    if attr in NOT_BLOCKED:
        return False
    return attr.startswith("_") or attr in BLOCKED_ATTRS or attr.startswith(BLOCKED_PREFIXES)


def check(code: str, deny: frozenset = frozenset()) -> list:
    """Violations of guarded mode in ``code`` (empty if none). ``deny``: further names refused as
    attributes, e.g. ``load_image`` where the front end chooses the image."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # exec reports it with the usual traceback
    found = []

    def bad(node, msg):
        found.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def check_module(node, name):
        parts = name.split(".")
        if parts[0] not in ALLOWED_MODULES:
            bad(node, f"import of {parts[0]!r} (allowed: {', '.join(sorted(ALLOWED_MODULES))})")
        elif any(_blocked_attr(p) for p in parts[1:]):
            bad(node, f"import of {name!r} (file, process or private module)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                check_module(node, a.name)
                if a.asname in PROTECTED:
                    bad(node, f"rebinding {a.asname!r}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad(node, "relative import")
                continue
            check_module(node, node.module or "")
            for a in node.names:
                if a.name == "*":
                    bad(node, "star import")
                elif _blocked_attr(a.name) or a.name in deny:
                    bad(node, f"import of {a.name!r} from {node.module} (file, process or private name)")
                if a.asname in PROTECTED:
                    bad(node, f"rebinding {a.asname!r}")
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            on_ws = isinstance(node.value, ast.Name) and node.value.id == "ws"
            if attr in deny:
                bad(node, f".{attr} is not available here")
            elif _blocked_attr(attr) and not (on_ws and attr in WS_METHODS):
                hint = "; use show(fig) to return a figure" if attr == "savefig" else ""
                bad(node, f".{attr} (files, processes and private attributes are blocked{hint})")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                bad(node, f"{node.id!r} is not available")
            elif node.id in PROTECTED and not isinstance(node.ctx, ast.Load):
                bad(node, f"rebinding {node.id!r}")
        elif isinstance(node, ast.arg) and node.arg in PROTECTED:
            bad(node, f"rebinding {node.arg!r}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and PROTECTED & set(node.names):
            bad(node, "rebinding ws or show")
    return found


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if (
        level
        or name.split(".")[0] not in ALLOWED_MODULES
        or any(_blocked_attr(p) for p in name.split(".")[1:])
    ):
        raise ImportError(f"guarded mode: import of {name!r} is not allowed")
    for f in fromlist or ():
        if f == "*" or _blocked_attr(f):
            raise ImportError(f"guarded mode: import of {f!r} from {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def guarded_builtins() -> dict:
    """The builtins for guarded code: without file access, dynamic code and attribute lookups by
    name; imports go through the allowlist."""
    b = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_NAMES}
    b["__import__"] = _guarded_import
    return b


def describe(mode: str) -> str:
    """One line for startup messages."""
    if mode == "full":
        return ("python tool: full (the agent can run any Python code as this user; "
                "--python guarded or --python off restrict it, see the README)")  # fmt: skip
    if mode == "guarded":
        return "python tool: guarded (computations only: allowlisted imports, no files or processes; not a sandbox)"
    return "python tool: off"


GUARDED_RULES = (
    "Guarded mode: imports only from numpy, scipy, skimage, matplotlib and the standard library's math, "
    "collections, itertools, re, json and similar; no files (no open, savefig, np.save, imread), no "
    "processes, no getattr/eval/exec, no attributes starting with '_'. Return figures with show(fig)."
)
