"""The web viewer's workspace index <-> NiiVue mapping (static/coords.js) for any image orientation:
the location a user mentions must be the voxel the agent's tools index, and the crosshair must land
where the agent works. Runs NiiVue itself under Node (skipped without node)."""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nninteractive_agent.core.io import LoadedImage
from nninteractive_agent.web import layers

STATIC = Path(__file__).parents[1] / "src" / "nninteractive_agent" / "web" / "static"
NIIVUE = "vendor/niivue-0.69.0.esm.js"


def directions() -> dict:
    """All 48 axis-aligned directions (permutations x flips) plus two oblique ones."""
    out = {}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            d = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                d[row, col] = s
            out[f"perm{''.join(map(str, perm))}_{''.join('+' if s > 0 else '-' for s in signs)}"] = d
    a = np.deg2rad(25)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    out["oblique"] = rot
    out["oblique_flipped"] = rot @ np.diag([-1, 1, -1])
    return out


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_mentioned_location_is_the_workspace_voxel(tmp_path):
    import SimpleITK as sitk

    shape, spacing = (5, 6, 7), (2.1, 1.3, 0.8)
    data = (np.arange(np.prod(shape)) + 1).reshape(shape).astype(np.uint16)
    cases = []
    for name, d in directions().items():
        ref = sitk.GetImageFromArray(data)
        ref.SetDirection(tuple(d.ravel()))
        ref.SetOrigin((12.5, -40.0, 7.0))
        ref.SetSpacing(tuple(spacing[::-1]))
        img = LoadedImage(data, spacing, "x.nii.gz", sitk_reference=ref)
        layers.write_volumes(img, tmp_path / name)  # what the browser loads
        nii = str(tmp_path / name / "image.nii.gz")
        ras = layers.array_to_ras(img).tolist()  # world mm of surfaces (focusObject)
        cases.append({"name": name, "file": nii, "shape": shape, "array_to_ras": ras})
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    static = tmp_path / "static"  # NiiVue and coords.js as ES modules for Node
    (static / "vendor").mkdir(parents=True)
    (static / "package.json").write_text('{"type": "module"}')
    for f in ("coords.js", NIIVUE):
        shutil.copy(STATIC / f, static / f)
    script = Path(__file__).with_name("viewer_coords.mjs")
    cmd = ["node", str(script), str(static), str(tmp_path / "cases.json")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
