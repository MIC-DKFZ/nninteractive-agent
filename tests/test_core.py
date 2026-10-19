import numpy as np
import pytest

from nninteractive_agent.core import metrics, raster, render
from nninteractive_agent.core.io import LoadedImage, load_image, save_mask


def test_scribble_is_two_px_wide_and_in_plane():
    crop, bbox, center = raster.scribble_crop([[3, 5, 2], [3, 5, 10]], axis=0, index=3, shape=(10, 20, 20))
    assert crop.shape == (1, 2, 9)
    assert crop.all()
    assert bbox == [[3, 4], [5, 7], [2, 11]]  # 2 px wide, exactly from column 2 to 10
    assert len(center) == 9 and (center[:, 0] == 3).all()


def test_scribble_diagonal_width_and_clipping():
    crop, bbox, _ = raster.scribble_crop([[0, 0, 0], [0, 9, 9]], axis=0, index=0, shape=(1, 10, 10))
    # 10 centerline pixels, each doubled sideways; the last one's neighbour is clipped at the border.
    assert bbox[1] == [0, 10] and bbox[2] == [0, 10]
    assert crop.sum() == 19


def test_polyline_follows_vertices():
    pix = raster.polyline_pixels([(0, 0), (0, 4), (4, 4)])
    assert tuple(pix[0]) == (0, 0) and tuple(pix[-1]) == (4, 4)
    assert len(pix) == 9  # shared vertex counted once


def test_scribble_never_extends_past_endpoints():
    crop, bbox, _ = raster.scribble_crop([[0, 5, 5], [0, 5, 9], [0, 1, 9]], axis=0, index=0, shape=(1, 20, 20))
    assert bbox[1] == [1, 7] and bbox[2] == [5, 11]  # horizontal part gets row 6, vertical part column 10


def test_infer_plane():
    shape = (10, 10, 10)
    assert raster.infer_plane([[1, 2, 3], [1, 5, 6]], None, shape, False) == (0, 1)
    with pytest.raises(ValueError, match="one plane"):
        raster.infer_plane([[1, 2, 3], [2, 5, 6]], None, shape, False)
    # A straight line along axis2 lies in planes axis0=1 and axis1=2: preference, else lowest axis.
    assert raster.infer_plane([[1, 2, 3], [1, 2, 6]], None, shape, False) == (0, 1)
    assert raster.infer_plane([[1, 2, 3], [1, 2, 6]], None, shape, False, prefer=(1,)) == (1, 2)
    assert raster.infer_plane([[1, 2, 3], [1, 2, 6]], 0, shape, False) == (0, 1)


def test_points_validate_bounds_and_accept_2d():
    assert raster.as_point([3, 4], (1, 10, 10), True) == [0, 3, 4]
    with pytest.raises(ValueError, match="outside"):
        raster.as_point([0, 10, 3], (1, 10, 10), True)


def test_bbox_from_corners_is_half_open_and_2d():
    bbox, axis = raster.bbox_from_corners([4, 1, 2], [4, 6, 9])
    assert axis == 0 and bbox == [[4, 5], [1, 7], [2, 10]]
    with pytest.raises(ValueError, match="2D rectangle"):
        raster.bbox_from_corners([1, 1, 2], [4, 6, 9])


def test_metrics_components_gaps_and_overlap():
    m = np.zeros((12, 10, 10), bool)
    m[1:4, 2:6, 2:6] = True
    m[6:10, 2:6, 2:6] = True  # gap at slices 4-5
    bbox = metrics.bbox_of(m)
    assert bbox == [[1, 10], [2, 6], [2, 6]]
    assert metrics.component_sizes(metrics.crop(m, bbox)) == [64, 48]
    warnings = metrics.slice_profile_warnings(metrics.crop(m, bbox), bbox)
    assert any("empty slices inside the object at 4-5" in w for w in warnings)
    img = LoadedImage(np.zeros((12, 10, 10), np.float32), (1.0, 1.0, 1.0), "x.npy")
    inst = np.zeros((12, 10, 10), np.uint16)
    inst[1:3, 2:4, 2:4] = 7
    st = metrics.object_stats(m, img, inst, {7: "other"})
    assert st["components"] == 2 and st["overlaps"] == {7: 8}
    assert "OVERLAPS committed objects: #7 other" in metrics.format_stats(st)


def test_io_roundtrip_png_and_nifti(tmp_path):
    from PIL import Image

    rgb = (np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "photo.png")
    img = load_image(tmp_path / "photo.png")
    assert img.shape == (1, 20, 30) and img.rgb.shape == (1, 20, 30, 3) and img.is_2d

    import SimpleITK as sitk

    vol = sitk.GetImageFromArray(np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6))
    vol.SetSpacing((0.5, 0.7, 3.0))
    sitk.WriteImage(vol, str(tmp_path / "vol.nii.gz"))
    img = load_image(tmp_path / "vol.nii.gz")
    assert img.shape == (4, 5, 6) and img.spacing == pytest.approx((3.0, 0.7, 0.5))
    assert img.orientation == ("S", "P", "L")
    mask = np.zeros(img.shape, np.uint8)
    mask[1, 2, 3] = 5
    out = save_mask(mask, tmp_path / "mask.nii.gz", img)
    back = sitk.ReadImage(str(out))
    assert back.GetSpacing() == pytest.approx((0.5, 0.7, 3.0))
    assert sitk.GetArrayFromImage(back)[1, 2, 3] == 5


def test_render_panel_and_flip_rules():
    img = LoadedImage(np.random.default_rng(0).random((8, 40, 50)).astype(np.float32), (2.0, 1.0, 1.0), "x.npy",
                      orientation=("S", "P", "L"))  # fmt: skip
    cur = np.zeros(img.shape, np.uint8)
    cur[4, 10:20, 10:30] = 1
    ov = render.Overlay(current=cur, instances=np.zeros(img.shape, np.uint16),
                        prompts=[render.Prompt(1, "scribble", True, [[4, 12, 12], [4, 15, 25]], 0, 4)])  # fmt: skip
    rgb = render.render_orthogonal(img, [4, 15, 20], render.fov_around(img.shape), ov, (0, 1))
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.shape[1] <= 3 * 600
    assert render._flip("S", "row") and not render._flip("P", "row")
    assert render._flip("R", "col") and not render._flip("L", "col")
    assert render.to_png(rgb)[:4] == b"\x89PNG"
