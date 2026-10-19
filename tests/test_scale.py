"""Downsampled segmentation of large objects: resampling and the workspace mode (fake session)."""

import numpy as np
import pytest
from test_workspace import FakeSession

from nninteractive_agent.core import scale
from nninteractive_agent.core.io import LoadedImage
from nninteractive_agent.core.workspace import AgentWorkspace
from nninteractive_agent.tools import call_tool


def test_factors_for_extent():
    shape = (300, 512, 512)  # defaults: objects > 320 voxels are brought to about 256
    assert scale.factors_for_extent(None, shape) == scale.FULL
    assert scale.factors_for_extent([200, 300, 310], shape) == scale.FULL  # below the threshold
    assert scale.factors_for_extent([200, 280, 380], shape) == (1.0, 1.0, 1.48)  # 280/256 is not worth it
    assert scale.factors_for_extent([200, 768, 400], shape, per_axis=False) == (3.0, 3.0, 3.0)
    assert scale.factors_for_extent([0, 900, 900], (1, 1000, 1000)) == (1.0, 3.52, 3.52)  # 2D: axis0 stays


def test_resampling_round_trip_and_regions():
    rng = np.random.default_rng(0)
    full = np.zeros((40, 64, 64), np.uint8)
    zz, yy, xx = np.ogrid[:40, :64, :64]
    full[((zz - 20) / 15) ** 2 + ((yy - 30) / 25) ** 2 + ((xx - 34) / 20) ** 2 <= 1] = 1
    f = (1.0, 2.0, 2.5)
    low = scale.downsample_mask(full, f)
    assert low.shape == scale.low_shape(full.shape, f) == (40, 32, 26)
    up = scale.upsample_mask(low, full.shape)
    dice = 2 * (up & full).sum() / (up.sum() + full.sum())
    assert dice > 0.97
    region = [[5, 30], [10, 50], [3, 60]]
    part = scale.upsample_mask(low, full.shape, region)
    assert np.array_equal(part, up[5:30, 10:50, 3:60])  # a region is exactly the crop of the whole
    img = rng.normal(size=(40, 64, 64)).astype(np.float32)
    assert np.allclose(scale.resample_linear(img, img.shape), img)  # same grid: identity
    # full-resolution voxels map into the low voxel that contains them
    assert [scale.to_low_index(i, 64, 32) for i in (0, 1, 2, 63)] == [0, 0, 1, 31]
    assert scale.full_region([[10, 12]], (64,), (32,)) == [[18, 26]]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(scale, "TARGET_EXTENT", 16)  # tiny test image, so "large" means > 16 voxels
    monkeypatch.setattr(scale, "THRESHOLD_EXTENT", 16)
    w = AgentWorkspace(session=FakeSession(), workdir=tmp_path)
    data = np.random.default_rng(0).normal(100, 10, (10, 40, 40)).astype(np.float32)
    w.set_image(LoadedImage(data, (2.0, 1.0, 1.0), "fake.npy"))
    return w


def test_large_object_is_segmented_on_a_downsampled_image(ws):
    r = ws.new_object("big", extent=[8, 40, 40])
    assert "downsampled copy" in r.text and "axis1 1/2.5" in r.text
    assert ws.session.shape == (10, 16, 16) and ws.session.buf is ws._low
    r = ws.bbox([5, 4, 4], [5, 35, 35])  # full-resolution corners
    # the fake paints the box (and the slices next to it) into the low-resolution buffer
    assert ws._low[5, 1:15, 1:15].all() and not ws._low[5, 0].any()
    assert ws.buffer[5, 4:35, 4:35].all() and not ws.buffer[5, :2].any() and not ws.buffer[0].any()
    assert "box check (axis0=5)" in r.text
    r = ws.scribble([[5, 20, 10], [5, 20, 30]], positive=False)  # mapped to the low grid
    assert not ws._low[5, 8, 4:12].any() and r.data["removed"] > 0
    ws.undo()
    assert ws.buffer[5, 20, 10:30].all()
    ws.commit()
    assert ws.objects[1]["factors"] == [1.0, 2.5, 2.5] and not ws.buffer.any() and not ws._low.any()
    assert (ws.instances == 1).sum() == ws.objects[1]["voxels"] > 3 * 31 * 31 * 0.9

    r = ws.new_object("small")  # back to the full-resolution image
    assert ws.session.shape == (10, 40, 40) and ws.session.buf is ws.buffer and "full-resolution image" in r.text
    ws.point([5, 20, 20], True)
    ws.discard()

    r = ws.edit_object(1)  # re-opened on the downsampled image, starting from the committed mask
    assert ws.session.shape == (10, 16, 16) and ws._low.any() and ws.buffer.sum() == ws.objects[1]["voxels"]
    ws.discard()
    assert (ws.instances == 1).sum() == ws.objects[1]["voxels"]


def test_size_note_suggests_the_extent_once(ws):
    ws.new_object("a")
    r = ws.bbox([5, 2, 2], [5, 37, 37])
    assert "new_object(name, extent=[3, 36, 36])" in r.text
    assert "extent=" not in ws.point([5, 20, 20], True).text  # only once per object


def test_python_edit_of_the_buffer_reaches_the_session(ws):
    ws.new_object("a")
    ws.bbox([5, 5, 5], [5, 10, 10])
    out = call_tool(ws, "python", {"code": "ws.point([5, 20, 20], True)"})  # workspace calls are fine
    assert "ws.buffer was changed" not in out.text
    out = call_tool(ws, "python", {"code": "ws.buffer[:] = 0\nws.buffer[4:7, 30:35, 30:35] = 1"})
    assert "ws.buffer was changed" in out.text and "75 voxels" in out.text
    assert ws.session.buf.sum() == 75 and ws._prev.sum() == 75  # the fake keeps the sent mask as its state
