"""Web front end: auth, label patches, surfaces, and the app end to end with fakes."""

import json
import struct
import time
import zlib

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402
from web_fakes import ScriptedAgent, fake_workspace  # noqa: E402

from nninteractive_agent.core.io import LoadedImage  # noqa: E402
from nninteractive_agent.web import layers  # noqa: E402
from nninteractive_agent.web.auth import Auth  # noqa: E402
from nninteractive_agent.web.server import _case, create_app  # noqa: E402
from nninteractive_agent.web.session import SessionManager  # noqa: E402


def decode(msg: bytes):
    n = struct.unpack("<I", msg[:4])[0]
    header = json.loads(msg[4 : 4 + n])
    data = np.frombuffer(zlib.decompress(msg[4 + n :]), dtype=header["dtype"])
    return header, data


def apply(target: np.ndarray, msg: bytes) -> dict:
    header, data = decode(msg)
    sl = tuple(slice(lo, hi) for lo, hi in header["box"])
    target[sl] = data.reshape([hi - lo for lo, hi in header["box"]])
    return header


# ----------------------------------------------------------------------------- auth
def test_auth_cookie_and_lockout():
    auth = Auth("secret-pw", secret="x" * 64)
    cookie, sid = auth.make_cookie()
    assert auth.session_id(cookie) == sid
    a, b, c = cookie.split(".")
    assert auth.session_id(f"{a}x.{b}.{c}") is None  # tampered id
    assert (
        auth.session_id(f"{a}.{int(time.time()) - 10}.{auth._sign(f'{a}.{int(time.time()) - 10}')}") is None
    )  # expired
    assert Auth("secret-pw", secret="y" * 64).session_id(cookie) is None  # other secret
    assert auth.check_password("secret-pw", "1.2.3.4")
    for _ in range(5):
        assert not auth.check_password("wrong", "1.2.3.4")
    assert auth.locked_for("1.2.3.4") > 0 and auth.locked_for("5.6.7.8") == 0


# --------------------------------------------------------------------------- layers
def test_patches_reconstruct_layers(tmp_path):
    ws = fake_workspace(None, tmp_path)
    ws.set_image(LoadedImage(np.zeros((8, 30, 40), np.float32), (2.0, 1.0, 1.0), "x.npy"))
    sync = layers.LayerSync()
    mirror = {}
    for msg in sync.patches(ws):
        h = decode(msg)[0]
        mirror[h["layer"]] = np.zeros(h["shape"], h["dtype"])
        apply(mirror[h["layer"]], msg)
        assert h["full"]
    assert sync.patches(ws) == []  # nothing changed
    ws.new_object("a")
    ws.bbox([4, 5, 5], [4, 15, 20])
    ws.scribble([[4, 6, 6], [4, 12, 18]], positive=False)
    for msg in sync.patches(ws):
        h = apply(mirror[decode(msg)[0]["layer"]], msg)
        assert not h["full"] and h["box"] != [[0, 8], [0, 30], [0, 40]]
    assert np.array_equal(mirror["current"], np.minimum(ws.buffer, 1))
    assert set(np.unique(mirror["prompts"])) == {0, layers.PROMPT_POS, layers.PROMPT_NEG}
    ws.commit()
    for msg in sync.patches(ws):
        apply(mirror[decode(msg)[0]["layer"]], msg)
    assert (
        np.array_equal(mirror["objects"], ws.instances)
        and not mirror["current"].any()
        and not mirror["prompts"].any()
    )


def test_mesh_geometry_matches_world_coordinates():
    sh, sp = (40, 120, 110), (4.0, 0.8, 0.9)
    z, y, x = np.ogrid[: sh[0], : sh[1], : sh[2]]
    c, r = np.array([20.3, 60.2, 50.7]), 30.0
    mask = np.sqrt(((z - c[0]) * sp[0]) ** 2 + ((y - c[1]) * sp[1]) ** 2 + ((x - c[2]) * sp[2]) ** 2) <= r
    img = LoadedImage(np.zeros(sh, np.float32), sp, "x.npy")
    aff = layers.array_to_ras(img)
    blob = layers.crop_mesh((mask[5:35, 10:110, 5:100], [5, 10, 5]), aff, sp)
    nv = struct.unpack("<II", blob[:8])[0]
    v = np.frombuffer(blob[8 : 8 + 12 * nv], "<f4").reshape(-1, 3)
    centre = (aff @ [*c, 1])[:3]
    assert np.abs(v.mean(0) - centre).max() < 0.5
    f = np.frombuffer(blob[8 + 12 * nv :], "<u4").reshape(-1, 3)
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    assert (((v[f].mean(1) - centre) * normals).sum(1) > 0).all()  # outward: the 3D view culls back faces
    assert abs(np.linalg.norm(v - centre, axis=1).mean() - r) < 0.5


def test_shrink_factors_keep_pixels_square():
    assert layers.shrink_factors((55, 512, 512), (5.0, 0.98, 0.98)) == [1, 2, 2]
    assert layers.shrink_factors((51, 133, 120), (0.8, 0.8, 1.74)) == [1, 1, 1]


def test_body_mask_drops_the_table_keeps_the_lungs():
    sp = (2.5, 1.5, 1.5)
    hu = np.full((20, 120, 120), -1000, np.int16)
    y, x = np.ogrid[:120, :120]
    hu[:, ((y - 55) / 40.0) ** 2 + ((x - 60) / 50.0) ** 2 <= 1] = 40  # body
    hu[:, ((y - 50) / 15.0) ** 2 + ((x - 40) / 12.0) ** 2 <= 1] = -850  # a lung, inside
    hu[:, 100:102, 10:110] = 200  # the table: a 3 mm plate below the body
    m = layers.body_mask(hu, sp, axial=0)
    assert m[10, 55, 60] and m[10, 50, 40]  # body and lung
    assert not m[:, 100:102, 10:110].any()
    assert not m[10, 5, 5]


# ------------------------------------------------------------------------------ app
@pytest.fixture
def client(tmp_path, monkeypatch):
    import SimpleITK as sitk

    monkeypatch.setenv("NNINTERACTIVE_AGENT_CONFIG", str(tmp_path / "config"))
    f = tmp_path / "case.nii.gz"
    im = sitk.GetImageFromArray(np.random.default_rng(0).normal(0, 100, (12, 48, 40)).astype(np.int16))
    im.SetSpacing((0.8, 0.8, 3.0))
    sitk.WriteImage(im, str(f))
    cases = [_case({"id": "c1", "title": "Test CT", "subtitle": "", "modality": "CT", "tasks": ["Segment it."],
                    "path": f.name}, f)]  # fmt: skip
    mgr = SessionManager("fake", "fake", cases, max_sessions=1, root=tmp_path / "web", workspace_factory=fake_workspace,
                         agent_factory=lambda *a, **k: ScriptedAgent(*a, delay=0.0, **k))  # fmt: skip
    app = create_app(mgr, Auth("pw", secret="s" * 64), secure_cookie=False)
    with TestClient(app) as c:
        c.manager = mgr  # for tests that need a session's state
        yield c


def test_everything_needs_login(client):
    assert client.get("/", follow_redirects=False).status_code == 303
    for url in ("/api/download", "/api/img/00001.png", "/api/mesh/o1", "/api/volume/x/image.nii.gz"):
        assert client.get(url).status_code == 401
    assert client.post("/api/open", json={"case": "c1"}).status_code == 401
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/ws") as ws:
        ws.receive_json()
    r = client.post("/login", data={"password": "nope"}, follow_redirects=False)
    assert "error=wrong" in r.headers["location"]


def _receive_until(ws, pred, limit=3000):
    for _ in range(limit):
        m = ws.receive()
        if m.get("text") is not None:
            ev = json.loads(m["text"])
            if pred(ev):
                return ev
    raise AssertionError("event not received")


def test_open_segment_download(client):
    r = client.post("/login", data={"password": "pw"}, follow_redirects=False)
    assert r.status_code == 303
    with client.websocket_connect("/api/ws") as ws:
        hello = ws.receive_json()
        assert hello["t"] == "hello" and hello["image"] is None and hello["cases"][0]["id"] == "c1"
        assert client.post("/api/open", json={"case": "c1"}).json() == {"ok": True}
        img = _receive_until(ws, lambda e: e["t"] == "image")["image"]
        assert img["shape"] == [12, 48, 40] and img["tasks"] == ["Segment it."]
        for name in ("image.nii.gz", "image3d.nii.gz", "zeros_uint8.nii.gz"):
            assert client.get(f"/api/volume/{img['key']}/{name}").status_code == 200
        _receive_until(ws, lambda e: e["t"] == "state" and e["ready"])
        ws.send_json({"cmd": "send", "text": "Segment it."})
        seen, urls = [], []
        done = _receive_until(ws, lambda e: seen.append(e["t"]) or urls.extend(e.get("images") or []) or e["t"] == "done")
        assert not done["error"] and done["prompts"] == 3
        assert {"user", "text", "call", "result", "task"} <= set(seen)
        # agent images carry their run, so the browser cache never shows another run's image
        assert urls and all("?run=" in u for u in urls)
        assert client.get("/" + urls[0]).status_code == 200
        assert client.get("/" + urls[0].split("?")[0] + "?run=another_run").status_code == 404
        objs = _receive_until(ws, lambda e: e["t"] == "objects" and e["objects"])
        assert objs["objects"][0]["name"] == "object 1"
        mesh = _receive_until(ws, lambda e: e["t"] == "mesh" and e["key"] == "o1" and not e.get("remove"))
        assert len(client.get("/" + mesh["url"]).content) > 8
    z = client.get("/api/download")
    assert z.status_code == 200 and z.content[:2] == b"PK"


def test_settings_offer_only_what_the_llm_has(client):
    client.post("/login", data={"password": "pw"}, follow_redirects=False)
    with client.websocket_connect("/api/ws") as ws:
        hello = ws.receive_json()
        assert hello["agent"]["label"] and [m["id"] for m in hello["agent"]["models"]] == ["opus", "sonnet"]
        ws.send_json({"cmd": "settings", "model": "gpt-x", "effort": "max"})  # neither is offered
        st = _receive_until(ws, lambda e: e["t"] == "state" and "model" in e)
        assert (st["model"], st["effort"]) == ("opus", "high")
        ws.send_json({"cmd": "settings", "model": "sonnet", "effort": "low"})
        st = _receive_until(ws, lambda e: e["t"] == "state" and e["model"] == "sonnet")
        assert st["effort"] == "low"


def test_upload_segmentation_becomes_objects(client, tmp_path):
    import SimpleITK as sitk

    client.post("/login", data={"password": "pw"}, follow_redirects=False)
    with client.websocket_connect("/api/ws") as ws:
        ws.receive_json()
        client.post("/api/open", json={"case": "c1"})
        _receive_until(ws, lambda e: e["t"] == "state" and e["ready"])
        labels = np.zeros((12, 48, 40), np.uint8)
        labels[4:8, 10:20, 10:20] = 5
        labels[4:8, 30:40, 10:20] = 9
        im = sitk.GetImageFromArray(labels)
        im.SetSpacing((0.8, 0.8, 3.0))  # the case's geometry
        f = tmp_path / "seg.nrrd"
        sitk.WriteImage(im, str(f))
        segments = json.dumps([{"value": 5, "name": "liver"}, {"value": 9, "name": "spleen", "notes": "edited"}])
        with open(f, "rb") as fh:
            r = client.post("/api/objects", files={"file": ("seg.nrrd", fh)}, data={"segments": segments})
        assert r.json() == {"ok": True}
        _receive_until(ws, lambda e: e["t"] == "divider" and "Segmentation uploaded" in e["text"])
        objs = _receive_until(ws, lambda e: e["t"] == "objects" and e["objects"])
        assert [(o["id"], o["name"], o["voxels"]) for o in objs["objects"]] == [(1, "liver", 400), (2, "spleen", 400)]
        assert objs["objects"][1]["notes"] == "edited"
        # the agent hears of it with the next message
        ws.send_json({"cmd": "send", "text": "Check them."})
        text = _receive_until(ws, lambda e: e["t"] == "text")
        assert "The user uploaded a segmentation, which replaced all objects with 2 object(s)" in text["text"]
        _receive_until(ws, lambda e: e["t"] == "done")
        # not on the image's grid
        im.SetSpacing((1.0, 0.8, 3.0))
        sitk.WriteImage(im, str(f))
        with open(f, "rb") as fh:
            r = client.post("/api/objects", files={"file": ("seg.nrrd", fh)}, data={"mode": "add"})
        assert r.status_code == 409 and "grid" in r.json()["detail"]


def _upload_liver(client, tmp_path):
    import SimpleITK as sitk

    labels = np.zeros((12, 48, 40), np.uint8)
    labels[4:8, 10:20, 10:20] = 5
    im = sitk.GetImageFromArray(labels)
    im.SetSpacing((0.8, 0.8, 3.0))
    f = tmp_path / "liver.nrrd"
    sitk.WriteImage(im, str(f))
    with open(f, "rb") as fh:
        return client.post("/api/objects", files={"file": ("liver.nrrd", fh)},
                           data={"segments": json.dumps([{"value": 5, "name": "liver"}])})  # fmt: skip


def test_upload_segmentation_resumes_a_paused_session(client, tmp_path):
    client.post("/login", data={"password": "pw"}, follow_redirects=False)
    with client.websocket_connect("/api/ws") as ws:
        ws.receive_json()
        client.post("/api/open", json={"case": "c1"})
        _receive_until(ws, lambda e: e["t"] == "state" and e["ready"])
        s = next(iter(client.manager.sessions.values()))
        with s.ws.lock:  # an object left open, then paused for inactivity
            s.ws.new_object("open one")
            s.ws.bbox([6, 20, 20], [6, 30, 30])
        client.portal.call(s.pause)
        assert s.paused and s.ws.session is None
        assert _upload_liver(client, tmp_path).json() == {"ok": True}
        assert not s.paused and s.ws.current is None
        assert [o["name"] for o in s.ws.objects.values()] == ["liver"]


def test_upload_note_is_dropped_with_the_image(client, tmp_path):
    client.post("/login", data={"password": "pw"}, follow_redirects=False)
    with client.websocket_connect("/api/ws") as ws:
        ws.receive_json()
        client.post("/api/open", json={"case": "c1"})
        _receive_until(ws, lambda e: e["t"] == "state" and e["ready"])
        assert _upload_liver(client, tmp_path).json() == {"ok": True}
        client.post("/api/open", json={"case": "c1"})  # a new image before the next message
        ws.send_json({"cmd": "send", "text": "Segment it."})
        text = _receive_until(ws, lambda e: e["t"] == "text")
        assert "uploaded a segmentation" not in text["text"]
        _receive_until(ws, lambda e: e["t"] == "done")
