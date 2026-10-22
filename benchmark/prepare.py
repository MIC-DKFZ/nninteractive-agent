"""Build the benchmark images and reference masks from their original sources.

    python benchmark/prepare.py fetch  [IMAGE ...]   # download everything that can be downloaded
    python benchmark/prepare.py build  [IMAGE ...]   # convert the downloads into data/
    python benchmark/prepare.py status               # what is there, what is missing and how to get it

Options: ``--downloads DIR`` (default ``benchmark/downloads``) and ``--data DIR`` (default
``benchmark/data``). Without image ids, every image in cases.json is processed.

We ship no data. ``fetch`` downloads the open sources from where their authors publish them (TCIA,
Zenodo, ERDA, Hugging Face); for the zip archives it reads only the members it needs. Three sources need a
person: put their files where ``status`` says (``<downloads>/_archives/`` for archives,
``<downloads>/<image>/`` for single files). ``build`` writes, per image::

    data/<image>/image.nii.gz          the image the agent gets
    data/<image>/<target>.nii.gz       one reference mask per target (targets may overlap)
    data/<image>/case.json             request texts, tiers, and per target a hint voxel: the point
                                       deepest inside the reference (for the guided track)

and checks each image and mask against checksums.json, so you know you built the same benchmark as
everyone else.

Needs numpy, scipy, SimpleITK, pydicom, scikit-image and tifffile (ant), and for the
bladder case a RAR5 reader: ``bsdtar``, ``7z`` or ``unrar`` on the PATH, or ``pip install libarchive-c``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text())
ARCHIVES = CASES["archives"]
IMAGES = {im["id"]: im for im in CASES["images"]}
CHECKSUMS = HERE / "checksums.json"
NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="


class Missing(Exception):
    """A source file is not there and cannot be downloaded by this script."""


# ------------------------------------------------------------------------------------------ fetching
def _download(url: str, dest: Path, md5: str | None = None, sha256: str | None = None) -> None:
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = {k: v for k, v in (("md5", md5), ("sha256", sha256)) if v}
    hashes = {k: hashlib.new(k) for k in want}
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            for h in hashes.values():
                h.update(chunk)
            done += len(chunk)
            if total > 1 << 28 and done - shown > total / 10:
                shown = done
                print(f"  {done / 1e9:.1f} of {total / 1e9:.1f} GB", flush=True)
    for k, v in want.items():
        if hashes[k].hexdigest() != v:
            tmp.unlink()
            raise RuntimeError(f"{url}: {k} {hashes[k].hexdigest()} instead of {v}")
    tmp.rename(dest)


class _RangeFile(io.RawIOBase):
    """A remote file read with HTTP range requests, in cached blocks (for zip members)."""

    def __init__(self, url: str, block: int = 1 << 22):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=120) as r:
            if r.status != 206:
                raise RuntimeError(f"{url} does not serve byte ranges")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = (off, self.pos + off, self.size + off)[whence]
        return self.pos

    def _block(self, b: int) -> bytes:
        if b not in self.cache:
            if len(self.cache) > 16:
                self.cache.clear()
            start, end = b * self.block, min(self.size, (b + 1) * self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=600) as r:
                self.cache[b] = r.read()
        return self.cache[b]

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        got = 0
        while got < n:
            b, off = divmod(self.pos, self.block)
            piece = self._block(b)[off : off + n - got]
            buf[got : got + len(piece)] = piece
            got += len(piece)
            self.pos += len(piece)
        return got


def _archive_file(downloads: Path, key: str) -> Path:
    return downloads / "_archives" / ARCHIVES[key]["file"]


class _Slice(io.RawIOBase):
    """A byte range of a file, as a file of its own."""

    def __init__(self, path: Path, start: int, size: int):
        self.f, self.start, self.size, self.pos = open(path, "rb"), start, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = (off, self.pos + off, self.size + off)[whence]
        return self.pos

    def readinto(self, buf) -> int:
        self.f.seek(self.start + self.pos)
        n = self.f.readinto(memoryview(buf)[: max(0, min(len(buf), self.size - self.pos))])
        self.pos += n
        return n


def _unwrap(downloads: Path, key: str):
    """A browser may save an archive inside another zip (Dryad does). Read it in place when it is
    stored uncompressed (no 9 GB copy), else take it out of the wrapper. A ZipFile, or None."""
    want = ARCHIVES[key]["file"]
    for wrapper in sorted((downloads / "_archives").glob("*.zip")):
        if wrapper.name == want:
            continue
        try:
            with zipfile.ZipFile(wrapper) as z:
                inner = [i for i in z.infolist() if i.filename == want or i.filename.endswith("/" + want)]
                if not inner:
                    continue
                info = inner[0]
                if info.compress_type == zipfile.ZIP_STORED:
                    with open(wrapper, "rb") as f:  # the member's data follow its local header
                        f.seek(info.header_offset)
                        head = f.read(30)
                    start = info.header_offset + 30 + int.from_bytes(head[26:28], "little") + int.from_bytes(head[28:30], "little")
                    return zipfile.ZipFile(io.BufferedReader(_Slice(wrapper, start, info.file_size), buffer_size=1 << 22))
                print(f"  taking {want} out of {wrapper.name}")
                with z.open(info) as src, open(_archive_file(downloads, key), "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
                return zipfile.ZipFile(_archive_file(downloads, key))
        except zipfile.BadZipFile:
            continue
    return None


def _open_archive(downloads: Path, key: str, fetch: bool):
    """('zip', ZipFile) or ('rar', path) for an archive, downloading it if needed and allowed."""
    spec = ARCHIVES[key]
    local = _archive_file(downloads, key)
    if not local.exists() and (z := _unwrap(downloads, key)) is not None:
        return "zip", z
    if local.exists():
        return ("zip", zipfile.ZipFile(local)) if zipfile.is_zipfile(local) else ("rar", local)
    if "url" not in spec:
        raise Missing(f"{spec['file']}: {spec['manual']} Put it in {local.parent}/.")
    if not fetch:
        raise Missing(f"{spec['file']}: run `prepare.py fetch` (downloads from {spec['url']})")
    if spec.get("ranges"):
        return "zip", zipfile.ZipFile(io.BufferedReader(_RangeFile(spec["url"]), buffer_size=1 << 22))
    print(f"  downloading {spec['file']} ({spec.get('bytes', 0) / 1e9:.1f} GB)")
    _download(spec["url"], local, spec.get("md5"))
    return ("zip", zipfile.ZipFile(local)) if zipfile.is_zipfile(local) else ("rar", local)


def _members(names: list, member: str, prefix: str | None) -> list:
    """Archive members that match a spec: the path (or a path ending in it), or every file in a
    folder whose name starts with ``prefix``. Skips macOS metadata."""
    names = [n for n in names if "__MACOSX/" not in n and not n.rsplit("/", 1)[-1].startswith("._")]
    if prefix is not None:
        return [n for n in names if f"/{n}".rsplit("/", 1)[0].endswith("/" + member.rstrip("/"))
                and n.rsplit("/", 1)[-1].startswith(prefix)]  # fmt: skip
    hits = [n for n in names if n == member or n.endswith("/" + member)]
    if len(hits) != 1:
        raise RuntimeError(f"expected one archive member {member!r}, found {len(hits)}")
    return hits


def _extract_rar(archive: Path, members: list, dest: Path) -> None:
    """Members of a RAR archive into ``dest`` (flat), with whatever RAR reader is installed."""
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        if shutil.which("bsdtar"):
            subprocess.run(["bsdtar", "-xf", str(archive), "-C", tmp, *members], check=True)
        elif shutil.which("7z") or shutil.which("7zz"):
            exe = shutil.which("7z") or shutil.which("7zz")
            subprocess.run([exe, "x", "-y", f"-o{tmp}", str(archive), *members], check=True, stdout=subprocess.DEVNULL)
        elif shutil.which("unrar"):
            subprocess.run(["unrar", "x", "-o+", str(archive), *members, tmp + "/"], check=True, stdout=subprocess.DEVNULL)
        else:
            try:
                import libarchive
            except ImportError as e:
                raise RuntimeError("reading the RAR archive needs bsdtar, 7z or unrar, or `pip install libarchive-c`") from e
            with libarchive.file_reader(str(archive)) as entries:
                for e in entries:
                    if e.pathname in members:
                        with open(Path(tmp) / e.pathname.rsplit("/", 1)[-1], "wb") as f:
                            for block in e.get_blocks():
                                f.write(block)
        for m in members:
            found = list(Path(tmp).rglob(m.rsplit("/", 1)[-1]))
            if len(found) != 1:
                raise RuntimeError(f"{archive.name}: could not extract {m}")
            shutil.move(str(found[0]), dest / found[0].name)


def _rar_names(archive: Path) -> list:
    if shutil.which("bsdtar"):
        return subprocess.run(["bsdtar", "-tf", str(archive)], check=True, capture_output=True, text=True).stdout.split("\n")
    if shutil.which("unrar"):
        out = subprocess.run(["unrar", "lb", str(archive)], check=True, capture_output=True, text=True).stdout
        return out.split("\n")
    if shutil.which("7z") or shutil.which("7zz"):
        exe = shutil.which("7z") or shutil.which("7zz")
        out = subprocess.run([exe, "l", "-slt", str(archive)], check=True, capture_output=True, text=True).stdout
        return [x[7:] for x in out.split("\n") if x.startswith("Path = ")]
    import libarchive

    with libarchive.file_reader(str(archive)) as entries:
        return [e.pathname for e in entries]


def local_name(key: str, spec: dict) -> str:
    """Where a source file lives inside ``<downloads>/<image>/``: files you copy by hand keep their
    name; archive members are named by their key (their names can clash), folders are the key."""
    if "manual" in spec:
        return spec["manual"].rsplit("/", 1)[-1]
    if "tcia" in spec or "prefix" in spec:
        return key  # a folder of DICOM files or slices
    name = (spec.get("member") or spec["url"]).rsplit("/", 1)[-1]
    return key + (".nii.gz" if name.endswith(".nii.gz") else Path(name).suffix)


def fetch_file(downloads: Path, image: str, key: str, spec: dict, fetch: bool) -> Path:
    """The local path of one source file of an image, downloading it when needed and allowed."""
    dest = downloads / image / local_name(key, spec)
    if dest.exists() and (dest.is_file() or any(dest.iterdir())):
        return dest
    if "manual" in spec:
        src = IMAGES[image]["source"]
        where = src["access"] if "access" in src else f"from {src['url']}"
        raise Missing(f"{spec['manual']} ({where}): copy it to {dest}")
    if "tcia" in spec:
        if not fetch:
            raise Missing(f"TCIA series {spec['tcia']}: run `prepare.py fetch`")
        print(f"  {image}/{key}: TCIA series {spec['tcia']}")
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            z = Path(tmp) / "series.zip"
            _download(NBIA + spec["tcia"], z)
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        with zf.open(info) as src, open(dest / info.filename.rsplit("/", 1)[-1], "wb") as dst:
                            shutil.copyfileobj(src, dst)
        return dest
    if "url" in spec:  # a single file
        if not fetch:
            raise Missing(f"{spec['url'].rsplit('/', 1)[-1]}: run `prepare.py fetch` (downloads from {spec['url']})")
        print(f"  {image}/{key}: {spec['url'].rsplit('/', 1)[-1]} ({spec.get('bytes', 0) / 1e9:.1f} GB)")
        _download(spec["url"], dest, sha256=spec.get("sha256"))
        return dest
    kind, arch = _open_archive(downloads, spec["archive"], fetch)
    names = arch.namelist() if kind == "zip" else _rar_names(arch)
    members = _members(names, spec["member"], spec.get("prefix"))
    if not members:
        raise RuntimeError(f"{image}/{key}: nothing matches {spec['member']}{spec.get('prefix', '')}")
    print(f"  {image}/{key}: {len(members)} file(s) from {ARCHIVES[spec['archive']]['file']}")
    out_dir = dest if "prefix" in spec else dest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        if kind == "zip":
            for m in members:
                with arch.open(m) as src, open(Path(tmp) / m.rsplit("/", 1)[-1], "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 22)  # zipfile checks the CRC
        else:
            _extract_rar(arch, members, Path(tmp))
        if "prefix" in spec:
            for f in Path(tmp).iterdir():
                f.rename(dest / f.name)
        else:
            next(Path(tmp).iterdir()).rename(dest)
    return dest


def source_files(downloads: Path, image: str, fetch: bool = False) -> dict:
    return {k: fetch_file(downloads, image, k, s, fetch) for k, s in IMAGES[image]["files"].items()}


# ---------------------------------------------------------------------------------------- converting
def _sitk():
    import SimpleITK as sitk

    return sitk


def read_dicom_series(folder: Path, float32: bool = False):
    sitk = _sitk()
    r = sitk.ImageSeriesReader()
    uids = r.GetGDCMSeriesIDs(str(folder))
    if len(uids) != 1:
        raise RuntimeError(f"{folder}: expected one DICOM series, found {len(uids)}")
    r.SetFileNames(r.GetGDCMSeriesFileNames(str(folder), uids[0]))
    if float32:
        r.SetOutputPixelType(sitk.sitkFloat32)  # PET slices have their own rescale slopes
    return r.Execute()


def suv_factor(folder: Path) -> float:
    """SUV (body weight) per Bq/ml: weight / injected dose decayed to the start of the scan (the
    earliest acquisition time), for PET in Bq/ml decay-corrected to the scan start."""
    import pydicom

    files = sorted(p for p in folder.iterdir() if p.suffix == ".dcm")
    hdrs = [pydicom.dcmread(p, stop_before_pixels=True) for p in files]
    ds = hdrs[0]
    if ds.Units != "BQML" or ds.DecayCorrection != "START":
        raise RuntimeError(f"{folder}: PET units {ds.Units}, decay correction {ds.DecayCorrection} not supported")
    rp = ds.RadiopharmaceuticalInformationSequence[0]

    def seconds(t) -> float:
        t = str(t)
        return int(t[:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:])

    start = min(seconds(h.AcquisitionTime) for h in hdrs)
    dt = start - seconds(rp.RadiopharmaceuticalStartTime)
    dose = float(rp.RadionuclideTotalDose) * 2 ** (-dt / float(rp.RadionuclideHalfLife))
    return float(ds.PatientWeight) * 1000 / dose


def _slice_index(ref, ipp) -> int:
    idx = ref.TransformPhysicalPointToContinuousIndex([float(v) for v in ipp])
    k = round(idx[2])
    if abs(idx[0]) > 0.01 or abs(idx[1]) > 0.01 or abs(idx[2] - k) > 0.01:
        raise RuntimeError("a SEG frame does not lie on an image slice")
    return k


def seg_mask(seg_file: Path, ref, names: list) -> np.ndarray:
    """Union of the named segments (SegmentLabel or SegmentDescription) of a DICOM SEG, on the grid
    of the image it was drawn on."""
    import pydicom

    ds = pydicom.dcmread(seg_file)
    numbers = {int(s.SegmentNumber) for s in ds.SegmentSequence
               if s.SegmentLabel in names or s.get("SegmentDescription") in names}  # fmt: skip
    found = {s.SegmentLabel for s in ds.SegmentSequence} | {s.get("SegmentDescription") for s in ds.SegmentSequence}
    if not set(names) <= found:
        raise RuntimeError(f"{seg_file.name}: no segment {sorted(set(names) - found)}")
    io_ = [float(v) for v in ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient]
    if not np.allclose(io_, np.array(ref.GetDirection()).reshape(3, 3).T[:2].ravel(), atol=1e-3):
        raise RuntimeError(f"{seg_file.name}: the SEG frames are not in the image's orientation")
    px = ds.pixel_array.reshape(-1, ds.Rows, ds.Columns)
    out = np.zeros(ref.GetSize()[::-1], bool)
    for i, f in enumerate(ds.PerFrameFunctionalGroupsSequence):
        if int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber) in numbers:
            out[_slice_index(ref, f.PlanePositionSequence[0].ImagePositionPatient)] |= px[i] > 0
    return out


def rtstruct_mask(rt_file: Path, ref, names: list) -> np.ndarray:
    """Union of the named ROIs of an RTSTRUCT, each contour filled on its slice (even-odd: inner
    contours cut holes)."""
    import pydicom
    from skimage.draw import polygon

    ds = pydicom.dcmread(rt_file)
    numbers = {int(r.ROINumber) for r in ds.StructureSetROISequence if r.ROIName in names}
    if len(numbers) != len(set(names)):
        raise RuntimeError(f"{rt_file.name}: not every ROI of {names} is there")
    shape = ref.GetSize()[::-1]
    out = np.zeros(shape, bool)
    for rc in ds.ROIContourSequence:
        if int(rc.ReferencedROINumber) not in numbers:
            continue
        vol = np.zeros(shape, bool)
        for c in rc.get("ContourSequence", []):
            pts = np.array(c.ContourData, float).reshape(-1, 3)
            idx = np.array([ref.TransformPhysicalPointToContinuousIndex(p.tolist()) for p in pts])
            k = round(float(idx[:, 2].mean()))
            if np.abs(idx[:, 2] - k).max() > 0.05 or not 0 <= k < shape[0]:
                raise RuntimeError(f"{rt_file.name}: a contour does not lie on an image slice")
            rr, cc = polygon(idx[:, 1], idx[:, 0], shape=shape[1:])
            m = np.zeros(shape[1:], bool)
            m[rr, cc] = True
            vol[k] ^= m
        out |= vol
    return out


def _same_grid(a, b) -> bool:
    return (a.GetSize() == b.GetSize() and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4))  # fmt: skip


def _array_on(img, ref) -> np.ndarray:
    sitk = _sitk()
    if not _same_grid(img, ref):
        raise RuntimeError("a mask is not on the image's grid")
    return sitk.GetArrayFromImage(img)


def slicer_segments(seg, ref) -> dict:
    """{segment name: bool array} of a 3D Slicer ``.seg.nrrd`` on the image grid. Slicer stores
    segments as label values in one or more layers (a layer per vector component), so segments
    in different layers may overlap."""
    layers = _array_on(seg, ref)
    if layers.ndim == 3:
        layers = layers[..., None]
    out, i = {}, 0
    while seg.HasMetaDataKey(f"Segment{i}_Name"):
        layer, value = (int(seg.GetMetaData(f"Segment{i}_{k}")) for k in ("Layer", "LabelValue"))
        out[seg.GetMetaData(f"Segment{i}_Name")] = layers[..., layer] == value
        i += 1
    return out


def convert(image: str, files: dict):
    """(image as a SimpleITK image, {target id: bool array}) for one benchmark image."""
    sitk = _sitk()
    spec = IMAGES[image]
    conv, targets = spec["convert"], spec["targets"]
    kind = conv["kind"]
    masks = {}
    if kind == "dicom":
        img = read_dicom_series(files[conv["image"]], float32=conv.get("suv", False))
        if conv.get("suv"):
            img = sitk.Cast(img * suv_factor(files[conv["image"]]), sitk.sitkFloat32)
        for t in targets:
            r = t["reference"]
            folder = files[r.get("seg") or r["rtstruct"]]
            dcm = [p for p in folder.iterdir() if p.suffix == ".dcm"] if folder.is_dir() else [folder]
            if len(dcm) != 1:
                raise RuntimeError(f"{folder}: expected one DICOM file")
            masks[t["id"]] = seg_mask(dcm[0], img, r["segments"]) if "seg" in r else rtstruct_mask(dcm[0], img, r["rois"])
    elif kind == "rigid_masks":
        img = sitk.ReadImage(str(files[conv["image"]]))
        tfm = sitk.Euler3DTransform()
        tfm.SetParameters(conv["transform"]["parameters"])
        for t in targets:
            m = sitk.ReadImage(str(files[t["reference"]["mask"]]))
            moved = sitk.Resample(m, img, tfm, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            masks[t["id"]] = sitk.GetArrayFromImage(moved) > 0
    elif kind == "nifti":
        img = sitk.ReadImage(str(files[conv["image"]]))
        lab = _array_on(sitk.ReadImage(str(files[conv["mask"]])), img)
        masks = {t["id"]: np.isin(lab, t["reference"]["labels"]) for t in targets}
    elif kind == "slicer_seg":
        img = sitk.ReadImage(str(files[conv["image"]]))
        segs = slicer_segments(sitk.ReadImage(str(files[conv["seg"]])), img)
        for t in targets:
            name = t["reference"]["segment"]
            if name not in segs:
                raise RuntimeError(f"{files[conv['seg']].name}: no segment {name}")
            masks[t["id"]] = segs[name]
    elif kind == "veela":
        from scipy import ndimage as ndi

        img = sitk.ReadImage(str(files[conv["image"]]))
        liver = _array_on(sitk.ReadImage(str(files[conv["liver"]])), img) > 0
        vessels = _array_on(sitk.ReadImage(str(files[conv["vessels"]])), img)
        for t in targets:
            r = t["reference"]
            if r.get("liver_with_vessels"):
                masks[t["id"]] = liver | ((vessels > 0) & ndi.binary_dilation(liver, iterations=2))
            else:
                masks[t["id"]] = np.isin(vessels, r["vessels"])
    elif kind == "tiff_stack":
        import tifffile

        folder, name = files[conv["image"]], conv["name"]
        imgs, labs, missing = [], [], []
        for i in range(conv["slices"][0], conv["slices"][1] + 1):
            p, q = folder / f"{name}_{i}.tif", folder / f"{name}_{i}_mask.tif"
            if not p.exists() or not q.exists():
                missing.append(i)
                continue
            a, m = tifffile.imread(p), tifffile.imread(q)
            imgs.append(a if a.ndim == 2 else a[..., 0])
            labs.append((m if m.ndim == 2 else m[..., 0]) > 127)
        if missing:
            print(f"  {image}: slices {missing} are not in the release (left out)")
        img = sitk.GetImageFromArray(np.stack(imgs))
        masks = {targets[0]["id"]: np.stack(labs)}
    else:
        raise ValueError(f"unknown converter {kind}")
    return img, masks


def deepest_point(mask: np.ndarray, spacing) -> list:
    """The voxel farthest from the mask's border (in mm): the hint of the guided track."""
    from scipy import ndimage as ndi

    sl = ndi.find_objects(mask.astype(np.uint8))[0]
    edt = ndi.distance_transform_edt(np.pad(mask[sl], 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    p = np.unravel_index(int(np.argmax(edt)), edt.shape)
    return [int(i + s.start) for i, s in zip(p, sl)]


def array_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def build(image: str, files: dict, data: Path) -> dict:
    sitk = _sitk()
    spec = IMAGES[image]
    img, masks = convert(image, files)
    out = data / image
    out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out / "image.nii.gz"), useCompression=True)
    arr = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()[::-1]
    ml = float(np.prod(spacing)) / 1000
    targets = []
    for t in spec["targets"]:
        m = masks[t["id"]]
        if not m.any():
            raise RuntimeError(f"{image}/{t['id']}: the reference is empty")
        lab = sitk.GetImageFromArray(m.astype(np.uint8))
        lab.CopyInformation(img)
        sitk.WriteImage(lab, str(out / f"{t['id']}.nii.gz"), useCompression=True)
        targets.append({"id": t["id"], "request": t["request"], "tier": t["tier"], "mask": f"{t['id']}.nii.gz",
                        "hint": deepest_point(m, spacing), "voxels": int(m.sum()), "ml": round(int(m.sum()) * ml, 2),
                        "sha256": array_hash(m.astype(np.uint8))})  # fmt: skip
    case = {"id": image, "what": spec["what"], "image": "image.nii.gz", "shape": list(arr.shape),
            "spacing": [round(s, 5) for s in spacing], "sha256": array_hash(arr), "targets": targets}  # fmt: skip
    (out / "case.json").write_text(json.dumps(case, indent=1))
    return case


def _checksums() -> dict:
    return json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}


def check(case: dict) -> list:
    """Differences between a built case and the published checksums."""
    exp = _checksums().get(case["id"])
    if not exp:
        return ["no published checksums"]
    out = [] if case["sha256"] == exp["sha256"] else [f"image differs (shape {case['shape']}, expected {exp['shape']})"]
    for t in case["targets"]:
        e = exp["targets"].get(t["id"], {})
        if t["sha256"] != e.get("sha256"):
            out.append(f"{t['id']} differs ({t['voxels']} voxels, expected {e.get('voxels')})")
    return out


def record(cases: list) -> None:
    """Write the built cases' checksums to checksums.json (maintainers, after building from the sources)."""
    sums = _checksums()
    for c in cases:
        sums[c["id"]] = {"shape": c["shape"], "sha256": c["sha256"],
                         "targets": {t["id"]: {"voxels": t["voxels"], "sha256": t["sha256"]} for t in c["targets"]}}  # fmt: skip
    CHECKSUMS.write_text(json.dumps({k: sums[k] for k in IMAGES if k in sums}, indent=1) + "\n")


# --------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("fetch", "build", "status"))
    ap.add_argument("images", nargs="*", help="image ids (default: all)")
    ap.add_argument("--downloads", type=Path, default=HERE / "downloads")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--record", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    images = args.images or list(IMAGES)
    unknown = sorted(set(images) - set(IMAGES))
    if unknown:
        sys.exit(f"unknown image ids {unknown}; known: {', '.join(IMAGES)}")
    built, problems = [], {}
    for image in images:
        print(f"{image}:", flush=True)
        try:
            files = source_files(args.downloads, image, fetch=args.command == "fetch")
        except Missing as e:
            problems[image] = f"missing: {e}"
            print(f"  {problems[image]}")
            continue
        if args.command == "status":
            print("  all source files are there")
            continue
        if args.command == "fetch":
            continue
        case = build(image, files, args.data)
        diffs = check(case)
        built.append(case)
        print(f"  {len(case['targets'])} targets -> {args.data / image}" + ("" if not diffs else "; " + "; ".join(diffs)))
        if diffs and image in _checksums():
            problems[image] = "differs from the published benchmark: " + "; ".join(diffs)
    if args.record:
        record(built)
    if problems:
        print("\nNot done:" if args.command != "build" else "\nProblems:")
        for image, p in problems.items():
            print(f"- {image}: {p}")
    sys.exit(1 if problems and args.command == "build" else 0)


if __name__ == "__main__":
    main()
