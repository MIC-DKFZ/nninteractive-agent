"""The agent-facing tool set: names, descriptions, JSON schemas and handlers.

Front ends (the stdio MCP server, the web demo) only translate between their transport and
``call_tool``. Descriptions state the mechanical rules; strategy lives in the skill.
"""

from __future__ import annotations

import dataclasses
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from .core.pyguard import GUARDED_RULES
from .core.workspace import AgentError, AgentWorkspace, Result

POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3}
OBJECT_REF = {
    "type": ["integer", "string"],
    "description": "committed object id (e.g. 3) or its unique name; omit for the open object",
}
WINDOW = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
    "description": "display intensity range [low, high]; omit for the default",
}
COORDS = (
    "Coordinates are voxel indices [axis0, axis1, axis2], read from the tick labels of the views "
    "(2D images may use [row, col])."
)


@dataclass
class ToolSpec:
    name: str
    description: str
    properties: dict
    handler: Callable[[AgentWorkspace, dict], Result]
    required: list = field(default_factory=list)
    read_only: bool = False  # only inspects; clients may run several of these concurrently

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": self.properties, "required": self.required, "additionalProperties": False}


@dataclass
class ToolOutput:
    text: str
    images: list = field(default_factory=list)
    is_error: bool = False


TOOLS = [
    ToolSpec(
        "load_image",
        "Load an image file (NIfTI/NRRD/MHA, TIFF, PNG/JPG, NPY) and send it to nnInteractive. Starts a fresh "
        "workspace. Returns shape, spacing, intensity statistics, orientation and overview images.",
        {
            "path": {"type": "string"},
            "spacing": {"type": "array", "items": {"type": "number"}, "description": "override voxel spacing"},
            "channel": {"type": "integer", "description": "channel to use for 4D TIFF/NPY stacks"},
        },
        lambda ws, a: _load(ws, a),
        ["path"],
    ),
    ToolSpec(
        "image_info",
        "Summary of the loaded image (shape, spacing, intensities, orientation) with overview images.",
        {},
        lambda ws, a: _info(ws),
    ),
    ToolSpec(
        "view",
        "Render orthogonal slices (axis0, axis1, axis2 planes) through `center`, by default through the middle "
        "of the open object (or of `object`) with a margin around it. Yellow contour/fill = open object, thin "
        "colored contours labeled #id = committed objects, green/red = positive/negative prompts with step "
        "numbers. Use `size` to zoom: box edge length in voxels around `center`. " + COORDS,
        {
            "center": POINT,
            "object": OBJECT_REF,
            "axes": {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2]}},
            "size": {"type": ["integer", "array"], "description": "field of view edge in voxels (int or per axis)"},
            "window": WINDOW,
            "show_objects": {"type": "boolean", "description": "draw committed objects (default true)"},
            "overlay": {"type": "boolean", "description": "false = bare image without any masks or prompts"},
        },
        lambda ws, a: ws.view(a.get("center"), a.get("object"), tuple(a.get("axes", (0, 1, 2))), a.get("size"),
                              a.get("window"), a.get("show_objects", True), a.get("overlay", True)),  # fmt: skip
    ),
    ToolSpec(
        "montage",
        "Grid of up to 16 evenly spaced slices along `axis`, through the open object's extent (or `object`, or "
        "`slice_range` [first, last]). The fastest way to check a 3D result slice by slice. 3D images only.",
        {
            "axis": {"type": "integer", "enum": [0, 1, 2]},
            "n": {"type": "integer", "description": "number of slices (default 9, max 16)"},
            "object": OBJECT_REF,
            "slice_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            "window": WINDOW,
            "show_objects": {"type": "boolean"},
            "overlay": {"type": "boolean", "description": "false = bare image without any masks or prompts"},
        },
        lambda ws, a: ws.montage(a.get("axis", 0), a.get("n", 9), a.get("object"), a.get("slice_range"),
                                 a.get("window"), a.get("show_objects", True), a.get("overlay", True)),  # fmt: skip
    ),
    ToolSpec(
        "projection",
        "Max (or mean) intensity projection of the whole volume along `axis` with the outlines of all committed "
        "objects and their ids. Use it to check completeness and overlaps of many objects at once.",
        {
            "axis": {"type": "integer", "enum": [0, 1, 2]},
            "mode": {"type": "string", "enum": ["max", "mean"]},
            "window": WINDOW,
        },
        lambda ws, a: ws.projection(a.get("axis", 0), a.get("mode", "max"), a.get("window")),
    ),
    ToolSpec(
        "render3d",
        "Shaded 3D views of one object's surface (the open object by default) from several directions, with "
        "neighbouring committed objects faint in their colors. Shows the overall shape at a glance: missing parts "
        "(a cut-off end or lobe), leaks (protrusions or bridges into neighbours), stray pieces and holes that are "
        "easy to miss between slices. Each panel keeps the axes and tick labels of the slice view of its plane. "
        "`views`: patient directions 'A', 'P', 'L', 'R', 'S', 'I' (medical images) or 'axis0-' ... 'axis2+' (the "
        "side the viewer is on); default: front, the object's own lateral side and top. It shows no intensities: "
        "judge boundaries in slices. 3D images only.",
        {
            "object": OBJECT_REF,
            "views": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
            "show_neighbors": {"type": "boolean", "description": "draw other committed objects faintly (default true)"},
        },
        lambda ws, a: ws.render3d(a.get("object"), a.get("views"), a.get("show_neighbors", True)),
    ),
    ToolSpec(
        "new_object",
        "Start a new object. nnInteractive segments ONE object at a time: every instance (each rib, each cell, "
        "each organ) needs its own object. The previous object must be committed or discarded first. Give "
        "`extent` for objects larger than about 320 voxels along any axis (except thin structures such as "
        "vessel trees or thin layers): they are then segmented on a downsampled copy of the image (more "
        "accurate for large objects) and scaled back up; all coordinates stay at full resolution.",
        {"name": {"type": "string", "description": "descriptive name, e.g. 'rib_left_3' or 'nucleus_12'"},
         "extent": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                    "description": "estimated object size in voxels [axis0, axis1, axis2]; only needed when it "
                    "exceeds about 320 on some axis"}},
        lambda ws, a: ws.new_object(a["name"], a.get("extent")),
        ["name"],
    ),
    ToolSpec(
        "bbox",
        "Positive 2D bounding box as the FIRST prompt of an object (refused afterwards; never use boxes for "
        "corrections). Give two opposite corners (inclusive) that lie in one slice, i.e. share exactly one "
        "coordinate. Draw it tight around the object, in the slice where the object is largest. Two steps: "
        "first `preview=true` (draws the box on a zoomed view of that slice, sends nothing; check every edge), "
        "then the same corners with `preview=false` to send it; an unpreviewed box is refused. After sending, a "
        "box check reports whether the prediction sticks out of the box or leaves it partly empty. " + COORDS,
        {"corner1": POINT, "corner2": POINT,
         "preview": {"type": "boolean", "description": "true: only draw the box for checking, do not send it"}},
        lambda ws, a: ws.bbox(a["corner1"], a["corner2"], preview=bool(a.get("preview", False)), require_preview=True),
        ["corner1", "corner2"],
    ),
    ToolSpec(
        "scribble",
        "Scribble prompt, the first prompt of an object and the prompt for corrections: a polyline through 2 or "
        "more points that all lie in one slice (they share one coordinate), drawn 2 px wide. Positive: the whole line must lie inside the target structure (inside "
        "the missed part when correcting). Negative: the whole line must lie inside the wrongly included region "
        "/ background. Long lines work best: run it along the middle of the region over most of its visible "
        "extent, with enough points to follow curves; it may come close to a boundary but must not cross it. "
        "The first scribble of an object must be previewed: call it with preview=true (draws it on a zoomed view of "
        "its slice, sends nothing), check that the whole line stays inside the object, then send the same points "
        "with preview=false. The result reports how much of the line lies inside the current mask and the "
        "intensities along it. "
        + COORDS,
        {
            "points": {"type": "array", "items": POINT, "minItems": 2},
            "positive": {"type": "boolean"},
            "axis": {"type": "integer", "enum": [0, 1, 2],
                     "description": "slice axis; only needed to override the choice for a perfectly straight "
                     "line along one axis (default: plane of the previous prompt, else the thickest-slice axis)"},
            "preview": {"type": "boolean", "description": "true: only draw the line over the current mask (and "
                        "report how much of it lies inside), do not send it"},
        },
        lambda ws, a: ws.scribble(a["points"], a.get("positive", True), a.get("axis"), preview=bool(a.get("preview", False)),
                                  require_preview=True),  # fmt: skip
        ["points", "positive"],
    ),
    ToolSpec(
        "point",
        "Point prompt (positive: belongs to the object, negative: does not). Only where a scribble does not fit: "
        "a tiny object or a tiny error region a few voxels across. " + COORDS,
        {"position": POINT, "positive": {"type": "boolean"},
         "preview": {"type": "boolean", "description": "true: only draw the point over the current mask, do not send it"}},
        lambda ws, a: ws.point(a["position"], a.get("positive", True), preview=bool(a.get("preview", False))),
        ["position", "positive"],
    ),
    ToolSpec(
        "undo",
        "Revert the most recent prompt of the open object (single level: only the last one).",
        {},
        lambda ws, a: ws.undo(),
    ),
    ToolSpec(
        "stats",
        "Numbers for the open object (or a committed `object`): voxels/volume, bbox, connected components, "
        "intensity inside, border contact, overlaps with committed objects, and warnings about abrupt slice-area "
        "changes (typical for leaks or cut-off parts). Check these before rendering images.",
        {"object": OBJECT_REF},
        lambda ws, a: ws.stats(a.get("object")),
    ),
    ToolSpec(
        "commit",
        "Finish the open object and add it to the instance map under a new id. Every voxel belongs to one object: "
        "where this object overlaps committed ones, `overlap` decides. 'keep' leaves those voxels to them (cuts "
        "off a leak into a neighbour); 'replace' gives them to this object and the others shrink (a structure "
        "inside another, e.g. lesions in an already segmented organ). Without it small overlaps are kept and an "
        "object that would lose more than half of itself is refused. Optional clean-up of stray pieces: `keep_largest` keeps only the "
        "largest connected component, `min_component_size` drops components smaller than that many voxels. "
        "Only use these when the extra pieces are clearly wrong (not for genuinely fragmented structures).",
        {
            "notes": {"type": "string", "description": "optional remarks, e.g. remaining uncertainty"},
            "keep_largest": {"type": "boolean"},
            "min_component_size": {"type": "integer"},
            "overlap": {"type": "string", "enum": ["keep", "replace"],
                        "description": "where this object overlaps committed objects (see above)"},
        },
        lambda ws, a: ws.commit(a.get("notes", ""), a.get("keep_largest", False), a.get("min_component_size", 0),
                                a.get("overlap")),  # fmt: skip
    ),
    ToolSpec("discard", "Drop the open object (restores it if it was re-opened with edit_object).", {},
             lambda ws, a: ws.discard()),  # fmt: skip
    ToolSpec(
        "edit_object",
        "Re-open a committed object to correct it with more scribbles/points; commit() keeps its id.",
        {"object": OBJECT_REF},
        lambda ws, a: ws.edit_object(a["object"]),
        ["object"],
    ),
    ToolSpec("delete_object", "Remove a committed object.", {"object": OBJECT_REF},
             lambda ws, a: ws.delete_object(a["object"]), ["object"]),  # fmt: skip
    ToolSpec("list_objects", "List committed objects and the open one.", {}, lambda ws, a: ws.list_objects()),
    ToolSpec(
        "save",
        "Write the instance map (one id per committed object) and objects.json (id -> name, notes).",
        {"out_dir": {"type": "string"}},
        lambda ws, a: ws.save(a.get("out_dir")),
    ),
    ToolSpec(
        "python",
        "Run Python in a persistent namespace inside the workspace process (variables survive between calls). "
        "Available: `ws` (the workspace: ws.image.data [axis0, axis1, axis2] intensities, ws.image.spacing, "
        "ws.buffer = open object mask (if your code changes it, the object continues from the changed mask), "
        "ws.instances = committed instance map, ws.objects, and the same methods "
        "as the tools: ws.new_object, ws.scribble, ws.bbox, ws.point, ws.undo, ws.stats, ws.commit, ws.view, "
        "ws.montage, ... returning Result objects with .text and .data), `np`, `ndi` (scipy.ndimage), "
        "`skimage`, and `show(x)` to return an image (array, matplotlib Figure or a Result from ws.view). "
        "Use it to find candidate structures and to loop over many instances. Printed output is returned.",
        {"code": {"type": "string"}},
        lambda ws, a: ws.run_python(a["code"]),
        ["code"],
    ),
]
READ_ONLY_TOOLS = ("image_info", "view", "montage", "projection", "render3d", "stats", "list_objects")
for _t in TOOLS:
    _t.read_only = _t.name in READ_ONLY_TOOLS
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tools_for(python_mode: str = "full", exclude: tuple = ()) -> list:
    """The tools a front end offers: no ``python`` when it is off, its rules in the description when
    it is guarded."""
    out = []
    for t in TOOLS:
        if t.name in exclude or (t.name == "python" and python_mode == "off"):
            continue
        if t.name == "python" and python_mode == "guarded":
            t = dataclasses.replace(t, description=f"{t.description} {GUARDED_RULES}")
        out.append(t)
    return out


def _load(ws: AgentWorkspace, a: dict) -> Result:
    info = ws.load_image(a["path"], a.get("spacing"), a.get("channel"))
    ov = ws.overview()
    return Result(info.text + "\n" + ov.text, images=ov.images)


def _info(ws: AgentWorkspace) -> Result:
    info, ov = ws.info(), ws.overview()
    return Result(info.text + "\n" + ov.text, images=ov.images)


def call_tool(ws: AgentWorkspace, name: str, args: dict | None) -> ToolOutput:
    """Run one tool call; usage errors come back as error text the agent can act on."""
    args = args or {}
    spec = TOOLS_BY_NAME.get(name)
    t0 = time.time()
    if spec is None:
        out = ToolOutput(f"unknown tool {name!r}", is_error=True)
    else:
        try:
            with ws.lock:
                res = spec.handler(ws, args)
                notes = ws.take_notes()
            out = ToolOutput("\n".join([res.text, *notes]), res.images)
        except (AgentError, ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
            out = ToolOutput(f"{type(e).__name__}: {e}", is_error=True)
        except Exception as e:  # noqa: BLE001 - report, don't crash the server
            out = ToolOutput(f"internal error {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}", is_error=True)
    _log(ws, name, args, out, time.time() - t0)
    return out


def _log(ws: AgentWorkspace, name: str, args: dict, out: ToolOutput, seconds: float) -> None:
    try:
        ws.workdir.mkdir(parents=True, exist_ok=True)
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "tool": name,
            "args": args,
            "seconds": round(seconds, 2),
            "error": out.is_error,
            "text": out.text,
            "images": len(out.images),
            "total_prompts": ws.n_prompts,
        }
        with open(ws.workdir / "log.jsonl", "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass
