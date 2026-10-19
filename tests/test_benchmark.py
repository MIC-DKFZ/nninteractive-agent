"""benchmark/: building cases from source files, the request, and scoring each target at its commit."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_workspace import FakeSession  # noqa: E402


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prepare = _module("benchmark_prepare", "prepare.py")
bench_run = _module("benchmark_run", "run.py")

from nninteractive_agent.core.io import LoadedImage  # noqa: E402
from nninteractive_agent.core.workspace import AgentWorkspace  # noqa: E402


def _write(arr, path, spacing=(1.0, 1.0, 2.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


def test_archive_members_and_local_names():
    names = ["HaN-Seg/set_1/case_01/case_01_IMG_MR_T1.nrrd", "training/Formica_rufa2_1.tif",
             "training/Formica_rufa2_1_mask.tif", "training/Formica_rufa_1.tif",
             "__MACOSX/training/._Formica_rufa2_1.tif", "a/02.nii.gz", "b/02.nii.gz"]  # fmt: skip
    assert prepare._members(names, "case_01_IMG_MR_T1.nrrd", None) == [names[0]]
    assert prepare._members(names, "training/", "Formica_rufa2_") == names[1:3]
    with pytest.raises(RuntimeError):
        prepare._members(names, "02.nii.gz", None)  # two members end in it: ambiguous
    # members with the same file name get the key's name, so they cannot overwrite each other
    assert prepare.local_name("image", {"archive": "x", "member": "Center3/T2WI/02.nii.gz"}) == "image.nii.gz"
    assert prepare.local_name("mask", {"archive": "x", "member": "Center3/Annotation/02.nii.gz"}) == "mask.nii.gz"
    assert prepare.local_name("image", {"manual": "MRI_data/13/Thigh/Fat.nii.gz"}) == "Fat.nii.gz"
    assert prepare.local_name("ct", {"tcia": "1.2.3"}) == "ct"


@pytest.mark.parametrize("stored", [True, False])
def test_archive_inside_a_browser_wrapper_zip(tmp_path, monkeypatch, stored):
    import io
    import zipfile

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("training/x_1.tif", b"slice one")
    (tmp_path / "_archives").mkdir()
    mode = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(tmp_path / "_archives" / "doi_wrapper.zip", "w", mode) as w:
        w.writestr("README.txt", b"another member first")
        w.writestr("training.zip", inner.getvalue())
    monkeypatch.setitem(prepare.ARCHIVES, "toy", {"file": "training.zip", "manual": "by hand"})
    kind, z = prepare._open_archive(tmp_path, "toy", fetch=False)
    assert kind == "zip" and z.read("training/x_1.tif") == b"slice one"
    # read in place when stored (no copy), taken out of the wrapper otherwise
    assert (tmp_path / "_archives" / "training.zip").exists() is not stored


def test_build_writes_masks_hints_and_checksums(tmp_path, monkeypatch):
    img = np.random.default_rng(0).normal(0, 1, (8, 20, 30)).astype(np.float32)
    lab = np.zeros(img.shape, np.uint8)
    lab[2:6, 5:15, 5:12] = 1
    lab[3:5, 8:12, 18:28] = 2
    (tmp_path / "dl").mkdir()
    _write(img, tmp_path / "dl" / "image.nii.gz")
    _write(lab, tmp_path / "dl" / "mask.nii.gz")
    spec = {"id": "toy", "what": "a toy image", "convert": {"kind": "nifti", "image": "image", "mask": "mask"},
            "targets": [{"id": "t01_a", "request": "the a", "tier": "E", "reference": {"labels": [1]}},
                        {"id": "t02_ab", "request": "a and b", "tier": "H", "reference": {"labels": [1, 2]}}]}  # fmt: skip
    monkeypatch.setitem(prepare.IMAGES, "toy", spec)
    monkeypatch.setattr(prepare, "CHECKSUMS", tmp_path / "checksums.json")
    files = {"image": tmp_path / "dl" / "image.nii.gz", "mask": tmp_path / "dl" / "mask.nii.gz"}
    case = prepare.build("toy", files, tmp_path / "data")
    assert json.loads((tmp_path / "data" / "toy" / "case.json").read_text()) == case
    assert case["shape"] == [8, 20, 30] and case["spacing"] == [2.0, 1.0, 1.0]
    a, ab = case["targets"]
    assert a["voxels"] == int((lab == 1).sum()) and ab["voxels"] == int((lab > 0).sum())  # targets may overlap
    assert lab[tuple(a["hint"])] == 1 and lab[tuple(ab["hint"])] > 0
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "data" / "toy" / "t02_ab.nii.gz")))
    assert np.array_equal(m > 0, lab > 0)
    assert prepare.check(case) == ["no published checksums"]
    prepare.record([case])
    assert prepare.check(case) == []
    lab[0, 0, 0] = 1
    _write(lab, tmp_path / "dl" / "mask.nii.gz")
    assert prepare.check(prepare.build("toy", files, tmp_path / "data")) == ["t01_a differs (281 voxels, expected 280)",
                                                                           "t02_ab differs (361 voxels, expected 360)"]  # fmt: skip


def test_request_lists_targets_in_order():
    case = {"what": "a CT scan", "targets": [{"id": "t01_liver", "request": "the liver", "hint": [1, 2, 3]},
                                             {"id": "t02_tumor", "request": "the liver tumor", "hint": [4, 5, 6]}]}  # fmt: skip
    guided = bench_run.request(case, Path("/data/x.nii.gz"), guided=True)
    plain = bench_run.request(case, Path("/data/x.nii.gz"), guided=False)
    assert "- t01_liver: the liver, contains voxel [1, 2, 3]\n- t02_tumor: the liver tumor, contains voxel" in guided
    assert "voxel" not in plain and plain.index("t01_liver") < plain.index("t02_tumor")
    assert "commit each object before you start the next" in plain


def test_each_target_is_scored_at_its_own_commit(tmp_path):
    run_dir, data = tmp_path / "run", tmp_path / "data"
    data.mkdir()
    ws = AgentWorkspace(session=FakeSession(), workdir=run_dir / "ws")
    ws.snapshots = True
    ws.set_image(LoadedImage(np.zeros((10, 40, 40), np.float32), (2.0, 1.0, 1.0), "case.nii.gz"))
    ws.new_object("t01_organ")
    ws.bbox([5, 5, 5], [5, 30, 30])  # the fake paints [4:7, 5:31, 5:31]
    ws.commit()
    ws.new_object("t02_lesion")
    ws.point([5, 15, 15], True)  # [4:7, 14:17, 14:17], inside the organ
    ws.commit(overlap="replace")
    ws.new_object("t03_gone")
    ws.point([5, 35, 35], True)
    ws.commit()
    ws.delete_object("t03_gone")
    ws.save()
    organ, lesion = np.zeros((10, 40, 40), np.uint8), np.zeros((10, 40, 40), np.uint8)
    organ[4:7, 5:31, 5:31] = 1
    lesion[4:7, 14:17, 14:17] = 1
    gone = np.zeros((10, 40, 40), np.uint8)
    gone[4:7, 34:37, 34:37] = 1
    for name, arr in (("organ", organ), ("lesion", lesion), ("gone", gone)):
        _write(arr, data / f"{name}.nii.gz", (1.0, 1.0, 2.0))
    case = {"shape": [10, 40, 40], "targets": [
        {"id": "t01_organ", "tier": "E", "mask": "organ.nii.gz"}, {"id": "t02_lesion", "tier": "M", "mask": "lesion.nii.gz"},
        {"id": "t03_gone", "tier": "H", "mask": "gone.nii.gz"}]}  # fmt: skip
    s = bench_run.score(run_dir, case, data)
    organ_row, lesion_row, gone_row = s["targets"]
    # the organ as committed, before the lesion took its voxels; the saved file has the hole
    assert organ_row["dice"] == 1.0 and organ_row["source"] == "commit" and organ_row["dice_saved"] < 1.0
    assert lesion_row["dice"] == 1.0
    assert gone_row["source"] == "missing" and gone_row["dice"] == 0.0 and s["missing"] == 1


def test_an_agent_that_could_not_work_is_not_scored(tmp_path):
    codex = tmp_path / "codex"
    codex.mkdir()
    (codex / "transcript.jsonl").write_text(
        '{"type":"thread.started"}\n{"type":"error","message":"You have hit your usage limit. Try again at 2:38 PM."}\n'
        '{"type":"turn.failed","error":{"message":"You have hit your usage limit. Try again at 2:38 PM."}}\n'
    )
    assert "usage limit" in bench_run.agent_failure(codex)
    claude = tmp_path / "claude"
    claude.mkdir()
    (claude / "transcript.jsonl").write_text('{"type":"result","subtype":"success","is_error":false,"result":"done"}\n')
    assert bench_run.agent_failure(claude) is None
    (claude / "transcript.jsonl").write_text('{"type":"result","is_error":true,"result":"5-hour limit reached, resets 3pm"}\n')
    assert "limit" in bench_run.agent_failure(claude)
    assert 0 < bench_run.retry_wait("try again at 2:38 PM") <= 24 * 3600 + 120
    assert bench_run.retry_wait("no time given") == 30 * 60


oracle = _module("benchmark_oracle", "oracle.py")


def test_oracle_prompts_go_to_the_largest_error():
    ref = np.zeros((12, 40, 40), bool)
    ref[3:9, 5:35, 10:20] = True  # most pixels in an axis2 slice (6 x 30), then axis0 (30 x 10)
    spacing = (2.0, 1.0, 1.0)
    p = oracle.next_prompt(np.zeros_like(ref), ref, spacing, "points")
    assert p["kind"] == "point" and p["positive"] and ref[tuple(p["points"][0])]
    s = oracle.next_prompt(np.zeros_like(ref), ref, spacing, "scribbles")
    assert s["kind"] == "scribble" and s["positive"] and s["axis"] == 0
    assert len({q[0] for q in s["points"]}) == 1 and all(ref[tuple(q)] for q in s["points"])
    rows = [q[1] for q in s["points"]]
    assert max(rows) - min(rows) >= 20  # along the long side of the region
    pred = ref.copy()
    pred[3:9, 5:35, 20:30] = True  # a leak larger than anything missed
    pred[5, 10, 12] = False
    for kind in oracle.KINDS:
        q = oracle.next_prompt(pred, ref, spacing, kind)
        assert not q["positive"] and all(pred[tuple(v)] and not ref[tuple(v)] for v in q["points"])
    assert oracle.next_prompt(ref, ref, spacing, "points") is None
    tiny = np.zeros_like(ref)
    tiny[6, 20, 15] = True
    assert oracle.next_prompt(np.zeros_like(ref), tiny, spacing, "scribbles")["kind"] == "point"


def test_oracle_extent_only_for_large_compact_objects():
    big = np.zeros((10, 400, 50), bool)
    big[2:8, 20:380, 10:40] = True
    assert oracle.extent_for(big, 320) == [6, 360, 30]
    assert oracle.extent_for(big[:, :300], 320) is None
    vessel = np.zeros_like(big)
    rr, cc = np.linspace(20, 379, 360).astype(int), np.linspace(5, 45, 360).astype(int)
    vessel[5, rr, cc] = True  # long, thin and oblique: stays at full resolution
    assert oracle.extent_for(vessel, 320) is None


def test_oracle_track_records_dice_after_each_prompt(tmp_path, monkeypatch, capsys):
    import argparse

    from nninteractive_agent.core import workspace

    img = np.zeros((8, 30, 30), np.float32)
    lab = np.zeros(img.shape, np.uint8)
    lab[2:6, 5:20, 5:12] = 1
    (tmp_path / "dl").mkdir()
    _write(img, tmp_path / "dl" / "image.nii.gz")
    _write(lab, tmp_path / "dl" / "mask.nii.gz")
    spec = {"id": "toy", "what": "a toy image", "convert": {"kind": "nifti", "image": "image", "mask": "mask"},
            "targets": [{"id": "t01_a", "request": "the a", "tier": "E", "reference": {"labels": [1]}}]}  # fmt: skip
    monkeypatch.setitem(prepare.IMAGES, "toy", spec)
    monkeypatch.setattr(prepare, "CHECKSUMS", tmp_path / "checksums.json")
    files = {"image": tmp_path / "dl" / "image.nii.gz", "mask": tmp_path / "dl" / "mask.nii.gz"}
    prepare.build("toy", files, tmp_path / "data")
    monkeypatch.setattr(workspace, "make_session", lambda spec: FakeSession())
    args = argparse.Namespace(track="oracle-scribbles", prompts=4, server="local", python="off", llm_url=None,
                              data=tmp_path / "data", rerun=False, tag="")  # fmt: skip
    out = tmp_path / "results" / bench_run.config_name(args)
    out.mkdir(parents=True)
    (out / "config.json").write_text("{}")
    s = bench_run.run_one("toy", 1, args, out)
    curve = s["trajectories"]["t01_a"]
    assert len(curve) == 4 and s["targets"][0]["dice"] == curve[-1] and s["prompts"] >= 1
    assert out.name == "oracle-scribbles-4"
    bench_run.summary([out])
    assert "after prompt: 1: " in capsys.readouterr().out
    assert json.loads((out / "summary.json").read_text())["dice_after_prompts"][-1] == curve[-1]
    agent = tmp_path / "results" / "guided-claude-opus-medium"
    (agent / "toy" / "rep1").mkdir(parents=True)
    (agent / "config.json").write_text("{}")
    row = {"id": "t01_a", "tier": "E", "dice": 0.5, "prompts": 2}
    (agent / "toy" / "rep1" / "score.json").write_text(json.dumps({"image": "toy", "rep": 1, "targets": [row], "missing": 0}))
    bench_run.summary([tmp_path / "results"])
    assert json.loads((agent / "summary.json").read_text())["oracle_same_prompts"] == {"oracle-scribbles-4": curve[1]}


def test_voxtell_track_scores_each_prompt_as_its_target(tmp_path, monkeypatch):
    import argparse

    img = np.zeros((8, 30, 30), np.float32)
    lab = np.zeros(img.shape, np.uint8)
    lab[2:6, 5:20, 5:12] = 1
    lab[3:5, 8:12, 7:10] = 2  # inside 1: both targets keep their own mask
    (tmp_path / "dl").mkdir()
    _write(img, tmp_path / "dl" / "image.nii.gz")
    _write(lab, tmp_path / "dl" / "mask.nii.gz")
    spec = {"id": "toy", "what": "a toy image", "convert": {"kind": "nifti", "image": "image", "mask": "mask"},
            "targets": [{"id": "t01_organ", "request": "the organ", "tier": "E", "reference": {"labels": [1, 2]}},
                        {"id": "t02_lesion", "request": "the lesion (inside)", "tier": "H", "reference": {"labels": [2]}}]}  # fmt: skip
    monkeypatch.setitem(prepare.IMAGES, "toy", spec)
    monkeypatch.setattr(prepare, "CHECKSUMS", tmp_path / "checksums.json")
    files = {"image": tmp_path / "dl" / "image.nii.gz", "mask": tmp_path / "dl" / "mask.nii.gz"}
    prepare.build("toy", files, tmp_path / "data")
    fake = tmp_path / "fake_voxtell.py"  # writes the reference for "the organ", nothing useful for the lesion
    fake.write_text(
        "import sys, SimpleITK as sitk, numpy as np\n"
        "a = sys.argv[1:]; i = a[a.index('-i') + 1]; o = a[a.index('-o') + 1]; p = a[a.index('-p') + 1:]\n"
        "import os; os.makedirs(o, exist_ok=True)\n"
        "img = sitk.ReadImage(i); arr = sitk.GetArrayFromImage(img)\n"
        "for prompt in p:\n"
        "    m = np.zeros(arr.shape, np.uint8)\n"
        "    if prompt == 'the organ': m[2:6, 5:20, 5:12] = 1\n"
        "    else: m[0, 0, 0] = 1\n"
        "    out = sitk.GetImageFromArray(m); out.CopyInformation(img)\n"
        "    name = ''.join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt).replace(' ', '_')\n"
        "    sitk.WriteImage(out, os.path.join(o, 'image_' + name + '.nii.gz'))\n"
    )
    args = argparse.Namespace(track="voxtell", voxtell=f"{sys.executable} {fake}", server="local", python="off",
                              llm_url=None, data=tmp_path / "data", rerun=False, tag="")  # fmt: skip
    out = tmp_path / "results" / bench_run.config_name(args)
    out.mkdir(parents=True)
    s = bench_run.run_one("toy", 1, args, out)
    organ, lesion = s["targets"]
    assert out.name == "voxtell" and organ["dice"] == 1.0 and organ["source"] == "commit"
    assert lesion["dice"] == 0.0 and s["text_prompts"] == ["the organ", "the lesion (inside)"]
