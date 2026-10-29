"""Workspace and tool logic against a fake nnInteractive session (no server needed)."""

import numpy as np
import pytest

from nninteractive_agent.core.io import LoadedImage
from nninteractive_agent.core.workspace import AgentError, AgentWorkspace
from nninteractive_agent.tools import TOOLS, call_tool


class FakeSession:
    """Mimics the session API: prompts paint simple shapes into the target buffer."""

    supports_undo = True

    def set_image(self, image, props=None):
        self.shape = image.shape[1:]

    def set_target_buffer(self, buf):
        self.buf = buf
        self._undo = None

    def reset_interactions(self):
        self.buf[:] = 0
        self._undo = None

    def _paint(self, slices, value):
        self._undo = self.buf.copy()
        self.buf[slices] = value
        return [[s.start, s.stop] for s in slices]

    def add_bbox_interaction(self, bbox, include_interaction, run_prediction=True):
        (z0, z1), (y0, y1), (x0, x1) = bbox
        return self._paint((slice(max(z0 - 1, 0), z1 + 1), slice(y0, y1), slice(x0, x1)), 1)

    def add_point_interaction(self, p, include_interaction, run_prediction=True):
        sl = tuple(slice(max(c - 1, 0), c + 2) for c in p)
        return self._paint(sl, 1 if include_interaction else 0)

    def add_scribble_interaction(self, crop, include_interaction, run_prediction=True, interaction_bbox=None):
        sl = tuple(slice(lo, hi) for lo, hi in interaction_bbox)
        return self._paint(sl, 1 if include_interaction else 0)

    def add_initial_seg_interaction(self, seg, run_prediction=False):
        self.buf[:] = seg

    def undo(self):
        if self._undo is None:
            return False
        self.buf[:] = self._undo
        self._undo = None
        return True


@pytest.fixture
def ws(tmp_path):
    w = AgentWorkspace(session=FakeSession(), workdir=tmp_path)
    data = np.random.default_rng(0).normal(100, 10, (10, 40, 40)).astype(np.float32)
    w.set_image(LoadedImage(data, (2.0, 1.0, 1.0), "fake.npy"))
    return w


def test_object_lifecycle_and_bbox_first_rule(ws):
    with pytest.raises(AgentError, match="new_object"):
        ws.bbox([5, 5, 5], [5, 10, 10])
    ws.new_object("a")
    r = ws.bbox([5, 5, 5], [5, 10, 10])
    assert r.data["added"] == 3 * 6 * 6 and "connected components: 1" in r.text
    with pytest.raises(AgentError, match="first prompt"):
        ws.bbox([5, 5, 5], [5, 10, 10])
    with pytest.raises(AgentError, match="still open"):
        ws.new_object("b")
    assert "committed #1 'a'" in ws.commit().text
    assert ws.objects[1]["voxels"] == 108 and (ws.instances == 1).sum() == 108


def test_overlap_is_reported_and_kept_by_earlier_object(ws):
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 10, 10])
    ws.commit()
    ws.new_object("b")
    r = ws.bbox([5, 8, 8], [5, 14, 14])
    assert "OVERLAPS committed objects: #1 a" in r.text
    assert "overlapped committed objects" in ws.commit().text
    assert (ws.instances == 1).sum() == 108


def test_scribble_diagnostics_undo_and_negative(ws):
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 20, 20])
    r = ws.scribble([[5, 10, 8], [5, 10, 15]], positive=False)
    assert "100% inside the current mask" in r.text and r.data["removed"] > 0
    before = int(ws.buffer.sum())
    assert "undid prompt #2" in ws.undo().text
    assert int(ws.buffer.sum()) > before
    assert "nothing to undo" in ws.undo().text


def test_edit_discard_restores_and_commit_cleanup(ws):
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 10, 10])
    ws.commit()
    ws.edit_object(1)
    assert (ws.instances == 1).sum() == 0 and ws.buffer.sum() == 108
    ws.discard()
    assert (ws.instances == 1).sum() == 108
    ws.new_object("b")
    ws.bbox([5, 20, 20], [5, 30, 30])
    ws.point([2, 2, 35], True)  # a separate small blob
    r = ws.commit(keep_largest=True)
    assert "clean-up removed 1 component(s)" in r.text


def test_call_tool_reports_usage_errors(ws):
    assert {t.name for t in TOOLS} >= {"view", "scribble", "commit", "python", "image_info"}
    out = call_tool(ws, "scribble", {"points": [[1, 2, 3], [2, 3, 4]], "positive": True})
    assert out.is_error and "no open object" in out.text
    ws.new_object("a")
    out = call_tool(ws, "scribble", {"points": [[1, 2, 3], [2, 3, 4]], "positive": True})
    assert out.is_error and "one plane" in out.text
    out = call_tool(ws, "point", {"position": [1, 2, 99], "positive": True})
    assert out.is_error and "outside the image" in out.text
    out = call_tool(ws, "python", {"code": "print(ws.image.shape); 1/0"})
    assert not out.is_error and "(10, 40, 40)" in out.text and "ZeroDivisionError" in out.text
    out = call_tool(ws, "view", {})
    assert not out.is_error and out.images and (ws.workdir / "log.jsonl").exists()


def test_python_edits_of_instances_update_object_sizes(ws):
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 10, 10])
    ws.commit()
    before = ws.objects[1]["voxels"]
    r = ws.run_python("ws.instances[ws.instances == 1] = 0\nws.instances[5, 5:7, 5:7] = 1")
    assert ws.objects[1]["voxels"] == 4 != before and "#1 4 voxels" in r.text


def test_bbox_tool_requires_a_preview_and_reports_the_fit(ws):
    ws.new_object("a")
    out = call_tool(ws, "bbox", {"corner1": [5, 5, 5], "corner2": [5, 14, 20]})
    assert out.is_error and "preview" in out.text
    out = call_tool(ws, "bbox", {"corner1": [5, 5, 5], "corner2": [5, 14, 20], "preview": True})
    assert not out.is_error and out.images and "PREVIEW" in out.text
    assert ws.n_prompts == 0 and not ws.buffer.any() and not ws.current.prompts
    out = call_tool(ws, "bbox", {"corner1": [5, 9, 5], "corner2": [5, 14, 20]})  # 4 voxels off: preview again
    assert out.is_error
    out = call_tool(ws, "bbox", {"corner1": [5, 6, 5], "corner2": [5, 14, 21]})  # within 2 voxels
    assert not out.is_error and "box check (axis0=5): the prediction fits the box" in out.text


def test_box_check_flags_a_prediction_that_sticks_out(ws):
    def paint_more(bbox, include_interaction, run_prediction=True):  # the object is larger than the box
        return ws.session._paint((slice(4, 7), slice(5, 15), slice(5, 35)), 1)

    ws.session.add_bbox_interaction = paint_more
    ws.new_object("a")
    r = ws.bbox([5, 5, 5], [5, 14, 20])  # from code: no preview needed
    assert "WARNING the prediction extends 14 voxels past the box at axis2 (high side)" in r.text


def test_scribble_and_point_previews_send_nothing(ws):
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 14, 20])
    before = ws.buffer.copy()
    for name, args in (("scribble", {"points": [[5, 8, 8], [5, 8, 16]], "positive": False, "preview": True}),
                       ("point", {"position": [5, 10, 10], "positive": False, "preview": True})):  # fmt: skip
        out = call_tool(ws, name, args)
        assert not out.is_error and out.images and "PREVIEW" in out.text
    assert np.array_equal(ws.buffer, before) and len(ws.current.prompts) == 1


def test_pyplot_figures_are_attached_without_show(ws):
    code = "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(40, 18))\nax.plot([0, 1])"
    out = call_tool(ws, "python", {"code": code})
    assert len(out.images) == 1 and "attached automatically" in out.text
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"  # never a GUI backend in a worker thread
    out = call_tool(ws, "python", {"code": "fig, ax = plt.subplots()\nshow(fig)"})  # shown once, not twice
    assert len(out.images) == 1 and "attached automatically" not in out.text


def test_reopened_object_keeps_its_count_and_prompts(ws):
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 10, 10])
    ws.commit()
    ws.edit_object(1)
    assert "#1 0 voxels" not in ws.run_python("x = 1").text  # open for editing: not 'emptied'
    ws.point([5, 7, 7], True)
    ws.commit()
    assert ws.objects[1]["prompts"] == 2


def test_actions_are_logged_with_snapshots(ws):
    import json

    ws.snapshots = True
    ws.new_object("t01_x")
    ws.bbox([5, 5, 5], [5, 14, 14])
    ws.run_python("ws.scribble([[5, 8, 8], [5, 8, 12]], False)")
    ws.commit(notes="maybe too small")
    recs = [json.loads(x) for x in (ws.workdir / "actions.jsonl").read_text().splitlines()]
    assert [r["action"] for r in recs] == ["new_object", "bbox", "scribble", "commit"]
    assert recs[2]["via"] == "python" and recs[2]["length"] == 4.0 and recs[3]["notes"] == "maybe too small"
    assert all((ws.workdir / r["snapshot"]).exists() for r in recs)


def test_overlap_keep_replace_and_refusal(ws):
    ws.new_object("organ")
    ws.bbox([5, 5, 5], [5, 30, 30])
    ws.commit()
    organ = ws.objects[1]["voxels"]
    # a structure inside the organ: refused without a choice, nothing is committed
    ws.new_object("lesion")
    ws.point([5, 15, 15], True)
    out = call_tool(ws, "commit", {})
    assert out.is_error and "overlap='replace'" in out.text and ws.current is not None
    out = call_tool(ws, "commit", {"overlap": "keep"})
    assert out.is_error and "every voxel" in out.text  # it would be empty
    r = ws.commit(overlap="replace")
    lesion = ws.objects[2]["voxels"]
    assert lesion == 27 and "took 27 voxels from #1 'organ'" in r.text
    assert ws.objects[1]["voxels"] == organ - 27 == int((ws.instances == 1).sum())
    assert "given to #2 'lesion'" in ws.objects[1]["notes"]
    # a small overlap (a leak) is still cut off by default
    ws.new_object("neighbour")
    ws.bbox([5, 28, 5], [5, 38, 30])
    r = ws.commit()
    assert "were left to them" in r.text and ws.objects[1]["voxels"] == organ - 27


def test_object_colors_do_not_repeat_in_long_series():
    from matplotlib.colors import to_rgb

    from nninteractive_agent.core.render import OBJECT_COLORS, object_color

    assert [object_color(i) for i in range(1, len(OBJECT_COLORS) + 1)] == OBJECT_COLORS  # unchanged
    cols = [np.array(to_rgb(object_color(i))) for i in range(1, 61)]
    assert len({object_color(i) for i in range(1, 61)}) == 60
    # 24 ribs: left and right of a level are 12 ids apart and must be told apart
    assert min(np.abs(cols[i] - cols[i + 12]).sum() for i in range(48)) > 0.25


def test_import_objects_replace_add_and_edit(ws):
    ws.new_object("agent's")
    ws.bbox([5, 5, 5], [5, 10, 10])
    ws.commit()
    labels = np.zeros(ws.image.shape, np.uint8)
    labels[2:4, 0:5, 0:5] = 3
    labels[6, 20:30, 20:30] = 7
    r = ws.import_objects(labels, {3: {"name": "liver", "notes": "from Slicer"}}, mode="replace")
    assert "replaced all objects with 2 object(s)" in r.text
    assert {i: o["name"] for i, o in ws.objects.items()} == {1: "liver", 2: "object 7"}
    assert ws.objects[1]["voxels"] == 50 and ws.objects[1]["notes"] == "from Slicer"
    assert (ws.instances == 2).sum() == 100 and (ws.instances > 0).sum() == 150
    # add: the new object takes the voxels it covers; one covered completely is removed
    more = np.zeros(ws.image.shape, np.uint16)
    more[2:4, 0:5, 0:5] = 1
    more[0, 0, 39] = 2
    r = ws.import_objects(more, mode="add")
    assert "added 2 object(s)" in r.text and "removed #1 'liver'" in r.text
    assert sorted(ws.objects) == [2, 3, 4] and ws.objects[3]["voxels"] == 50
    # imported objects work like committed ones
    ws.edit_object(2)
    assert ws.buffer.sum() == 100
    ws.discard()
    with pytest.raises(AgentError, match="shape"):
        ws.import_objects(np.zeros((2, 2, 2), np.uint8))
    with pytest.raises(AgentError, match="integer"):
        ws.import_objects(np.full(ws.image.shape, 0.5, np.float32))
