"""The parts a public release adds: python modes and the guard, the MCP prompt and resources,
``nninteractive-agent setup`` and the web demo's examples."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from test_workspace import FakeSession

from nninteractive_agent.core import pyguard
from nninteractive_agent.core.io import LoadedImage
from nninteractive_agent.core.workspace import AgentError, AgentWorkspace
from nninteractive_agent.tools import call_tool, tools_for


def workspace(tmp_path, **kw):
    w = AgentWorkspace(session=FakeSession(), workdir=tmp_path / "ws", **kw)
    w.set_image(LoadedImage(np.random.default_rng(0).normal(100, 10, (10, 40, 40)).astype(np.float32),
                            (2.0, 1.0, 1.0), "fake.npy"))  # fmt: skip
    return w


ESCAPES = [
    "import os",
    "import subprocess",
    "from os import system",
    "__import__('os').system('true')",
    "().__class__.__base__.__subclasses__()",
    "getattr(np, 'save')('x.npy', np.zeros(2))",
    "np.save('x.npy', np.zeros(2))",
    "import matplotlib\nmatplotlib.os.system('true')",
    "import matplotlib.pyplot as plt\nplt.savefig('x.png')",
    "open('x.txt', 'w').write('x')",
    "exec('1')",
    "eval('1')",
    "from numpy import ctypeslib",
    "from skimage import io",
    "ws.workdir.joinpath('x').write_text('x')",
    "ws.session.close()",
    "ws._send_mask()",
    "ws = np\nws.save('x.npy', np.zeros(2))",
    "def f(ws): return ws\n",
    "g = (i for i in [1])\nf = g.gi_frame.f_back",
    "import operator\noperator.attrgetter('__class__')(ws)",
    "np.zeros(2).tofile('x.bin')",
    "from numpy import *",
]
ALLOWED = [
    "import numpy as np\nprint(np.zeros(3).sum())",
    "from scipy import ndimage\nlab, n = ndimage.label(ws.image.data > 100)",
    "from skimage.morphology import opening, ball\nm = opening(ws.image.data > 100, ball(1))",
    "import json\nprint(json.dumps(json.loads('{\"a\": 1}')))",
    "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1, 2])\nshow(fig)",
    "for _ in range(2):\n    pass\nprint(ws.image.data.shape, ws.image.spacing)",
    "r = ws.view(axis=0, index=5)\nshow(r)",
    "ws.save()",
]


@pytest.mark.parametrize("code", ESCAPES)
def test_guard_blocks_files_processes_and_introspection(code):
    assert pyguard.check(code), code


@pytest.mark.parametrize("code", ALLOWED)
def test_guard_allows_computations(code):
    assert pyguard.check(code) == [], code


def test_guarded_workspace_refuses_before_running_and_runs_allowed_code(tmp_path):
    ws = workspace(tmp_path, python_mode="guarded")
    out = call_tool(ws, "python", {"code": "x = 1\nimport os"})
    assert out.is_error and "guarded mode refused" in out.text and "import of 'os'" in out.text
    out = call_tool(ws, "python", {"code": "print(x)"})
    assert out.is_error is False and "NameError" in out.text  # the refused code did not run at all
    out = call_tool(ws, "python", {"code": "print(int(np.count_nonzero(ws.image.data > 0)))"})
    assert "16000" in out.text
    # the builtins are restricted at run time too, not only by the static check
    assert "open" not in ws._py_namespace["__builtins__"]
    with pytest.raises(ImportError):
        ws._py_namespace["__builtins__"]["__import__"]("os")


def test_python_off_and_tool_lists(tmp_path):
    assert "python" not in {t.name for t in tools_for("off")}
    guarded = next(t for t in tools_for("guarded") if t.name == "python")
    assert "Guarded mode" in guarded.description
    full = next(t for t in tools_for("full") if t.name == "python")
    assert "Guarded mode" not in full.description
    assert "load_image" not in {t.name for t in tools_for("full", exclude=("load_image",))}
    ws = workspace(tmp_path, python_mode="off")
    out = call_tool(ws, "python", {"code": "print(1)"})
    assert out.is_error and "switched off" in out.text


def test_python_mode_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(pyguard.ENV_VAR, "guarded")
    assert AgentWorkspace(session=FakeSession(), workdir=tmp_path).python_mode == "guarded"
    monkeypatch.setenv(pyguard.ENV_VAR, "nonsense")
    with pytest.raises(ValueError):
        AgentWorkspace(session=FakeSession(), workdir=tmp_path)


def test_confined_paths(tmp_path):
    ws = workspace(tmp_path, python_mode="guarded", confine_paths=True)
    with pytest.raises(AgentError):
        ws.load_image(tmp_path / "x.nii.gz")
    with pytest.raises(AgentError):
        ws.save(tmp_path / "elsewhere")
    assert call_tool(ws, "save", {}).is_error is False
    out = call_tool(ws, "python", {"code": "ws.load_image('/tmp/x.nii.gz')"})
    assert out.is_error and "load_image" in out.text


def test_mcp_server_offers_the_skill_as_prompt_and_resources(tmp_path):
    anyio = pytest.importorskip("anyio")
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "nninteractive_agent.mcp_server",
                                       "--server", "local", "--python", "off", "--workdir", str(tmp_path)])  # fmt: skip
        async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
            init = await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            prompts = [p.name for p in (await s.list_prompts()).prompts]
            prompt = await s.get_prompt("nninteractive-segmentation")
            uris = [str(r.uri) for r in (await s.list_resources()).resources]
            skill = await s.read_resource("nninteractive://skill/SKILL.md")
            return init, tools, prompts, prompt, uris, skill

    init, tools, prompts, prompt, uris, skill = anyio.run(run)
    assert "nninteractive-segmentation" in init.instructions
    assert "python" not in tools and {"load_image", "scribble", "bbox", "point", "save"} <= tools
    assert prompts == ["nninteractive-segmentation"]
    text = prompt.messages[0].content.text
    assert "Correction playbook" in text and "Reference file reference/domains.md" in text
    assert "nninteractive://skill/reference/python-recipes.md" in uris
    assert "Segmenting with nnInteractive" in skill.contents[0].text


@pytest.mark.parametrize("host", ["codex", "claude"])
def test_setup_registers_and_uninstalls(host, tmp_path, monkeypatch):
    if shutil.which(host) is None:
        pytest.skip(f"{host} is not installed")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    run = [
        sys.executable,
        "-m",
        "nninteractive_agent.cli",
        "setup",
        host,
        "--skip-model",
        "--python",
        "guarded",
    ]
    subprocess.run(run, check=True, capture_output=True)
    home = tmp_path / host
    assert (home / "skills" / "nninteractive-segmentation" / "SKILL.md").is_file()
    if host == "codex":
        config = (home / "config.toml").read_text()
        assert (
            "[mcp_servers.nninteractive]" in config and "tool_timeout_sec" in config and '"guarded"' in config
        )
    else:
        server = json.loads((home / ".claude.json").read_text())["mcpServers"]["nninteractive"]
        assert server["args"][-2:] == ["--python", "guarded"]
        assert "mcp__nninteractive__view" in (home / "agents" / "nninteractive-verifier.md").read_text()
    subprocess.run(run + ["--uninstall"], check=True, capture_output=True)
    assert not (home / "skills" / "nninteractive-segmentation").exists()


def test_setup_print(capsys):
    from nninteractive_agent import cli

    cli.main(["setup", "print", "--server", "http://gpu:1234"])
    out = capsys.readouterr().out
    entry = json.loads(out[: out.index("\n}") + 2])["mcpServers"]["nninteractive"]
    assert entry["command"] == sys.executable and entry["args"][-4:] == [
        "--server",
        "http://gpu:1234",
        "--python",
        "full",
    ]


def test_examples_from_a_directory_and_a_json_file(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from nninteractive_agent.web.server import _make_thumbnails, load_examples

    sitk = pytest.importorskip("SimpleITK")
    d = tmp_path / "images"
    d.mkdir()
    ct = sitk.GetImageFromArray(np.random.default_rng(0).normal(0, 600, (8, 32, 32)).astype(np.int16))
    sitk.WriteImage(ct, str(d / "abdomen_ct.nii.gz"))
    np.save(d / "cells.npy", np.random.default_rng(1).random((6, 20, 20)).astype(np.float32))
    (d / "notes.txt").write_text("not an image")
    (d / "examples.json").write_text(json.dumps({"cases": [{"path": "cells.npy", "title": "Cells",
                                                            "tasks": ["Segment every cell."]}]}))  # fmt: skip
    cases = load_examples(d)
    by_id = {c["id"]: c for c in cases}
    assert set(by_id) == {"cells", "abdomen_ct"}
    assert by_id["cells"]["title"] == "Cells" and by_id["cells"]["tasks"] == ["Segment every cell."]
    assert by_id["abdomen_ct"]["title"] == "abdomen ct" and by_id["abdomen_ct"]["tasks"] == []
    _make_thumbnails(cases, tmp_path / "thumbs")
    assert (tmp_path / "thumbs" / f"{by_id['abdomen_ct']['thumb_key']}.png").is_file()
    assert by_id["abdomen_ct"]["modality"] == "CT" and "8 × 32 × 32" in by_id["abdomen_ct"]["subtitle"]
    assert by_id["cells"]["modality"] == "3D"

    monkeypatch.setenv("IMAGES", str(d))
    listing = tmp_path / "list.json"
    listing.write_text(
        json.dumps({"cases": [{"path": "$IMAGES/cells.npy", "id": "a"}, {"path": "missing.nii.gz"}]})
    )
    cases = load_examples(listing)
    assert [c["available"] for c in cases] == [True, False]
    assert load_examples(None) == []
    with pytest.raises(FileNotFoundError):
        load_examples(tmp_path / "nope")


def test_model_license_is_reported(tmp_path):
    from nninteractive_agent.core.session import model_license

    class Licensed(FakeSession):
        license = "CC BY-NC-SA 4.0"

    ws = AgentWorkspace(session=Licensed(), workdir=tmp_path / "ws")
    r = ws.set_image(LoadedImage(np.zeros((4, 8, 8), np.float32), (1.0, 1.0, 1.0), "x.npy"))
    assert "nnInteractive model license: CC BY-NC-SA 4.0" in r.text
    assert model_license(FakeSession()) == "unknown"
    assert model_license(type("S", (), {"license": "!!MISSING!!"})()) == "unknown"


def test_first_scribble_must_be_previewed(tmp_path):
    ws = workspace(tmp_path)
    ws.new_object("a")
    line = [[5, 10, 10], [5, 20, 25]]
    out = call_tool(ws, "scribble", {"points": line, "positive": True})
    assert out.is_error and "preview=true" in out.text and not ws.current.prompts
    out = call_tool(ws, "scribble", {"points": line, "positive": True, "preview": True})
    assert not out.is_error and out.images and "PREVIEW" in out.text
    nudged = [[5, 11, 10], [5, 21, 26]]  # within 2 voxels of the preview
    assert not call_tool(ws, "scribble", {"points": nudged, "positive": True}).is_error
    # corrections need no preview, and code (ws.scribble) never does
    assert not call_tool(ws, "scribble", {"points": [[5, 30, 30], [5, 34, 34]], "positive": False}).is_error
    ws.discard()
    ws.new_object("b")
    assert "prompt #1" in ws.scribble(line, True).text


def test_render3d(tmp_path):
    from nninteractive_agent.core import render

    ws = workspace(tmp_path)
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 20, 20])
    ws.commit()
    ws.new_object("b")
    ws.bbox([5, 18, 18], [5, 30, 30])
    out = call_tool(ws, "render3d", {})  # the open object, neighbours faint
    assert not out.is_error and len(out.images) == 1 and "the open object 'b'" in out.text
    assert "axis0 low-index side" in out.text  # no orientation: one view per axis
    out = call_tool(ws, "render3d", {"object": 1, "views": ["axis2+"], "show_neighbors": False})
    assert not out.is_error and "#1 'a'" in out.text and "1 connected component" in out.text
    assert call_tool(ws, "render3d", {"views": ["A"]}).is_error  # no patient orientation here
    # depth: viewed from the high side of axis 0, the first hit of a block at 3..6 is 13 slices deep
    m = np.zeros((20, 8, 8), bool)
    m[3:7, 2:6, 2:6] = True
    d, _, hit = render._first_hit(m.astype(np.uint8), 0, True)
    assert hit.sum() == 16 and d[hit].min() == 20 - 7


class SessionExpiredError(RuntimeError):
    """Stands in for the remote client's error of the same name (matched by name)."""


class ExpiringSession(FakeSession):
    """A FakeSession whose server can drop it: every call fails once ``expired`` is set."""

    def __init__(self):
        self.expired = False

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        def call(*args, **kwargs):  # like the remote client: the request fails, not the lookup
            if object.__getattribute__(self, "expired"):
                raise SessionExpiredError("lease expired or unknown")
            return attr(*args, **kwargs)

        return call


def test_expired_session_is_restored_transparently(tmp_path, monkeypatch):
    import nninteractive_agent.core.workspace as wsmod

    made = []
    monkeypatch.setattr(wsmod, "make_session", lambda spec: made.append(ExpiringSession()) or made[-1])
    ws = AgentWorkspace(session_spec="fake", workdir=tmp_path / "ws")
    ws.set_image(LoadedImage(np.zeros((10, 40, 40), np.float32), (2.0, 1.0, 1.0), "x.npy"))
    ws.new_object("a")
    ws.scribble([[5, 10, 10], [5, 20, 20]], True)
    before = int(ws.buffer.sum())
    made[-1].expired = True  # the server reaps the idle session
    out = call_tool(ws, "scribble", {"points": [[5, 30, 30], [5, 34, 34]], "positive": True})
    assert not out.is_error and "server had closed this session" in out.text and "continues from its current mask" in out.text
    assert len(made) == 2 and int(ws.buffer.sum()) > before  # restored mask plus the new prompt
    assert "server had closed" not in call_tool(ws, "stats", {}).text  # the note comes once
    assert "committed #1" in ws.commit().text
    # the web demo's pause and resume keeps an open object too
    ws.new_object("b")
    ws.point([5, 5, 5], True)
    ws.close()
    assert "continues from its current mask" in ws.reattach().text and ws.buffer.any()
