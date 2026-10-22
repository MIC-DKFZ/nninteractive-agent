"""Browser checks of the web demo in headless Chromium (not part of pytest; needs playwright).

    python tests/web_dev.py --delay 0.3 &          # the page with fakes on 127.0.0.1:8780
    python tests/browser_checks.py controls         # every mouse control, judged by its effect
    python tests/browser_checks.py window-under-load --gpu igpu   # window drag while the agent works
    python tests/browser_checks.py rotate --gpu rtx  # 3D drag rotation speed
    python tests/web_dev.py --gt --port 8781 &     # reference labels as objects, for 3D
    python tests/browser_checks.py render3d --gpu igpu --base http://127.0.0.1:8781/  # 3D frame times
    python tests/browser_checks.py no-webgl          # without WebGL2: an error message, no endless loading
    python tests/browser_checks.py transparency      # semi-transparent surfaces behind the volume

One-time setup: ``pip install playwright && python -m playwright install chromium``.
``--gpu``: ``soft`` (SwiftShader, slow but always works), ``igpu`` (Ryzen integrated GPU via
EGL, laptop-like), ``rtx`` (RTX 4090 via Vulkan), ``none`` (no WebGL; ``no-webgl`` uses it).
Screenshots go to ``--out``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

GPU_FLAGS = {
    "soft": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
    "igpu": ["--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu"],
    "rtx": ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"],
    "none": ["--disable-webgl", "--disable-webgl2"],
}
LOADED = "document.getElementById('loading').hidden && !document.getElementById('viewer').hidden"


async def open_case(pg, base: str, password: str, case: str, wait: bool = True) -> None:
    pg.on("pageerror", lambda e: print("PAGEERROR", e))
    pg.on("console", lambda m: m.type == "error" and print("console:", m.text[:200]))
    await pg.goto(base + "login")
    await pg.wait_for_timeout(400)
    await pg.fill("#password", password)
    await pg.click("button[type=submit]")
    await pg.wait_for_selector(".card")
    await pg.click(f".card:has-text('{case}')")
    if not wait:
        return
    await pg.wait_for_function(LOADED + " && document.getElementById('serverDot').classList.contains('ok')",
                               timeout=180000)  # fmt: skip
    await pg.wait_for_timeout(1500)


async def controls(pg, out: Path) -> None:
    box = await pg.locator("#canvas2d").bounding_box()
    x0, y0, w, h = box["x"], box["y"], box["width"], box["height"]
    P, Q = (x0 + w * 0.25, y0 + h * 0.25), (x0 + w * 0.15, y0 + h * 0.15)  # points in the axial tile

    async def lum():
        png = await pg.screenshot(
            clip={"x": x0 + w * 0.1, "y": y0 + h * 0.1, "width": w * 0.3, "height": h * 0.3}
        )
        return round(float(np.asarray(Image.open(io.BytesIO(png)).convert("L")).mean()), 1)

    async def readout():
        return await pg.text_content("#readout")

    async def click(pt):
        await pg.mouse.click(*pt)
        await pg.wait_for_timeout(250)
        return await readout()

    async def drag(a, b, button="left", mods=()):
        for m in mods:
            await pg.keyboard.down(m)
        await pg.mouse.move(*a)
        await pg.mouse.down(button=button)
        await pg.mouse.move(*b, steps=10)
        await pg.mouse.up(button=button)
        for m in mods:
            await pg.keyboard.up(m)
        await pg.wait_for_timeout(350)

    async def window():
        return await pg.eval_on_selector("#windowSel", "s => s.options[s.selectedIndex].text")

    async def tool():
        return await pg.eval_on_selector("#toolSeg button.on", "b => b.dataset.value")

    print("start: tool", await tool(), "| window", await window(), "| luminance", await lum())
    r0 = await click(P)
    l0 = await lum()
    await drag(P, (P[0], P[1] - 80), button="right")
    print(
        f"right-drag up (brighter): {l0} -> {await lum()} | {await window()} | crosshair kept: {await readout() == r0}"
    )
    await drag(P, (P[0] - 100, P[1]), button="right")
    print("right-drag left (more contrast):", await window())
    w_before = await window()
    await pg.mouse.dblclick(*P)
    await pg.wait_for_timeout(800)
    layout = await pg.eval_on_selector("#layoutSeg button.on", "b => b.dataset.value")
    print("double-click: layout", layout, "| window kept:", await window() == w_before)
    await pg.mouse.dblclick(*P)
    await pg.wait_for_timeout(800)
    await pg.mouse.move(*P)
    for k in "wspzc":
        await pg.keyboard.press(k)
        print(f"key {k}: tool", await tool())
    a = await readout()
    await pg.keyboard.press("ArrowDown")
    await pg.wait_for_timeout(250)
    print("ArrowDown:", a, "->", await readout())
    await pg.keyboard.press("w")
    l0 = await lum()
    await drag(P, (P[0], P[1] + 80))
    print(f"Window tool, drag down (darker): {l0} -> {await lum()} | {await window()}")
    await pg.keyboard.press("s")
    a = await click(P)
    await drag(P, (P[0], P[1] + 40))
    print("Scroll tool, drag 40 px:", a, "->", await readout())
    print("Scroll tool, plain click moves the crosshair:", await click(Q))
    await pg.keyboard.press("z")
    a, q0 = await click(P), await click(Q)
    await click(P)
    await drag(P, (P[0], P[1] - 80))
    print("Zoom tool: voxel under the drag start", a, "->", await click(P))
    print("           voxel elsewhere         ", q0, "->", await click(Q))
    await pg.keyboard.press("p")
    a = await click(Q)
    await drag(P, (P[0] + 60, P[1] + 30))
    print("Pan tool drag:", a, "->", await click(Q))
    await pg.keyboard.press("c")
    a = await click(Q)
    await drag(P, (P[0] - 60, P[1] - 30), button="middle")
    print("middle-drag pan:", a, "->", await click(Q))
    a = await click(Q)
    await drag(P, (P[0] + 40, P[1]), mods=("Shift",))
    print("Shift-drag pan:", a, "->", await click(Q))
    a = await click(P)
    await pg.mouse.wheel(0, 120)
    await pg.wait_for_timeout(250)
    print("wheel:", a, "->", await readout())
    await pg.keyboard.press("r")
    await pg.wait_for_timeout(500)
    print("R (reset): window", await window())
    await pg.screenshot(path=str(out / "controls_end.png"))


async def window_under_load(pg, out: Path) -> None:
    box = await pg.locator("#canvas2d").bounding_box()
    x, y = box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.25
    await pg.fill("#input", "Segment something.")
    await pg.keyboard.press("Enter")
    await pg.wait_for_timeout(1500)  # the scripted agent is now sending label updates
    await pg.evaluate("performance.clearMeasures(); window.__f = []; "
                      "(function f(t){ window.__f.push(t); requestAnimationFrame(f) })(performance.now())")  # fmt: skip
    await pg.mouse.move(x, y)
    await pg.mouse.down(button="right")
    t0, i = time.time(), 0
    while time.time() - t0 < 3.0:
        i += 1
        await pg.mouse.move(x + 40 * np.sin(i / 10), y - 40 * np.cos(i / 13))
        await asyncio.sleep(1 / 60)
    gpu = await pg.evaluate("[performance.getEntriesByName('nni-overlays').length, "
                            "performance.getEntriesByName('nni-window-bake').length]")  # fmt: skip
    fr = await pg.evaluate("(() => { const a = window.__f; const d = a.slice(1).map((t, i) => t - a[i]); "
                           "return [d.length, Math.max(...d), d.filter(x => x > 50).length] })()")  # fmt: skip
    await pg.mouse.up(button="right")
    await pg.wait_for_timeout(300)
    after = await pg.evaluate("[performance.getEntriesByName('nni-overlays').map(e => Math.round(e.duration)), "
                              "performance.getEntriesByName('nni-window-bake').map(e => Math.round(e.duration))]")  # fmt: skip
    print(f"3 s window drag while the agent works: {fr[0]} frames, longest {fr[1]:.0f} ms, {fr[2]} frames > 50 ms; "
          f"overlay refreshes / bakes during the drag: {gpu} (should be 0, 0)")  # fmt: skip
    print(f"after release: overlay refresh ms {after[0]}, window bake ms {after[1]}")
    await pg.screenshot(path=str(out / "window_under_load.png"))


async def rotate(pg, out: Path) -> None:
    box = await pg.locator("#canvas3d").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    await pg.mouse.move(x, y)
    await pg.mouse.down()
    await pg.mouse.move(x + 100, y + 20, steps=10)
    await pg.mouse.up()
    await pg.wait_for_timeout(500)
    view = await pg.eval_on_selector("#pane3d", "e => e.dataset.view")
    k = 300 / box["width"]
    print(f"3D view {box['width']:.0f} px wide; drag 100 px right, 20 px down -> azimuth,elevation {view} "
          f"(expected {180 + 100 * k:.0f},{12 + 20 * k:.0f} from 180,12)")  # fmt: skip
    await pg.screenshot(path=str(out / "rotate.png"), clip=box)


async def render3d(pg, out: Path) -> None:
    """Frame times of the 3D view while it rotates, per volume preset, with the case's reference
    objects (with web_dev.py --gt; else one scripted object)."""
    await pg.fill("#input", "Segment the reference objects.")
    await pg.keyboard.press("Enter")
    await pg.wait_for_function("document.getElementById('objCount').textContent !== '0'", timeout=120000)
    n, stable = -1, 0
    while stable < 3:  # surfaces arrive one by one (tiny objects get none)
        await pg.wait_for_timeout(1000)
        m = await pg.evaluate("nniViewer.meshes.size")
        stable, n = (stable + 1, m) if m == n and m > 0 else (0, m)
    await pg.click("#layoutSeg button[data-value='3d']")
    await pg.wait_for_timeout(800)
    box = await pg.locator("#canvas3d").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    for preset in await pg.eval_on_selector_all("#volume3dSel option", "os => os.map(o => o.value)"):
        await pg.select_option("#volume3dSel", preset)
        await pg.wait_for_timeout(600)
        # reading a pixel back waits for the GPU (gl.finish does not, in Chrome)
        gpu = await pg.evaluate("(() => { const r = nniViewer.r3, gl = r.gl, px = new Uint8Array(4); "
                                "const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); "
                                "r.render(); sync(); const t = performance.now(); "
                                "for (let i = 0; i < 10; i++) { r.azimuth += 3; r.render(); sync() } "
                                "return (performance.now() - t) / 10 })()")  # fmt: skip
        await pg.evaluate("window.__f = []; const g = window.__g = (window.__g || 0) + 1; (function f(t){ "
                          "if (g !== window.__g) return; window.__f.push(t); requestAnimationFrame(f) })(performance.now())")  # fmt: skip
        await pg.mouse.move(x, y)
        await pg.mouse.down()
        t0, i = time.time(), 0
        while time.time() - t0 < 3.0:
            i += 1
            await pg.mouse.move(x + 150 * np.sin(i / 15), y + 40 * np.sin(i / 23))
            await asyncio.sleep(1 / 60)
        await pg.mouse.up()
        fr = await pg.evaluate("(() => { const a = window.__f; const d = a.slice(1).map((t, i) => t - a[i]).sort((p, q) => p - q); "
                               "return [d.length, d[Math.floor(d.length / 2)], d[Math.floor(d.length * 0.95)]] })()")  # fmt: skip
        print(f"{preset:>14}: one frame {gpu:.0f} ms; while rotating {fr[0]} frames in 3 s, "
              f"median {fr[1]:.0f} ms, 95th percentile {fr[2]:.0f} ms")  # fmt: skip
        await pg.screenshot(path=str(out / f"render3d_{preset.replace(' ', '_')}.png"), clip=box)


async def no_webgl(pg, out: Path) -> None:
    """Without WebGL2 the image cannot be shown: the loading screen must say so instead of spinning."""
    try:
        await pg.wait_for_selector("#loading.error", timeout=20000)
    except Exception:  # noqa: BLE001, S110 - reported below
        pass
    text = await pg.locator("#loadingText").text_content()
    await pg.screenshot(path=str(out / "no_webgl.png"))
    ok = await pg.locator("#loading.error").count() == 1 and "WebGL2" in text
    print(f"{'OK' if ok else 'FAILED'}: loading screen says {text!r}")
    if not ok:
        raise SystemExit(1)


async def transparency(pg, out: Path) -> None:
    """Semi-transparent surfaces (the uncommitted object; all of them with the opacity slider below 1)
    must be hidden by the volume in front as much as opaque ones are: neither vanish behind a dense
    volume nor sit on top of it. A synthetic scene (render3d_scene.js) on its own canvas."""
    await pg.add_script_tag(path=str(Path(__file__).with_name("render3d_scene.js")), type="module")
    await pg.wait_for_function("window.r3scene !== undefined", timeout=20000)
    bad = 0
    cases = [("sparse volume in front", 0.02, 1, "shaded"), ("dense volume in front", 0.12, 1, "shaded"),
             ("dense, opacity slider 0.6", 0.12, 0.6, "shaded"), ("MIP", 0.3, 1, "mip")]  # fmt: skip
    for label, alpha, opacity, mode in cases:
        r = await pg.evaluate(f"r3scene.run({alpha}, {opacity}, '{mode}')")
        ratio = r["current"] / max(r["committed"], 1e-6)
        ok = 0.6 <= ratio <= 1.1  # the uncommitted object is drawn at 0.85 of the opacity
        bad += not ok
        print(f"{'OK' if ok else 'FAILED'}: {label}: visibility uncommitted {r['current']:.1f}, committed "
              f"{r['committed']:.1f} (ratio {ratio:.2f}); without volume {r['bare']:.1f}")  # fmt: skip
    await pg.screenshot(path=str(out / "transparency.png"), clip={"x": 0, "y": 0, "width": 400, "height": 400})  # fmt: skip
    if bad:
        raise SystemExit(1)


CHECKS = {"controls": (controls, "Abdominal CT"), "window-under-load": (window_under_load, "Liver CT with tumors"),
          "rotate": (rotate, "Abdominal CT"), "render3d": (render3d, "Liver CT with tumors"),
          "no-webgl": (no_webgl, "Abdominal CT"), "transparency": (transparency, "Abdominal CT")}  # fmt: skip


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=sorted(CHECKS))
    ap.add_argument("--gpu", choices=sorted(GPU_FLAGS), default="soft")
    ap.add_argument("--base", default="http://127.0.0.1:8780/")
    ap.add_argument("--password", default="dev")
    ap.add_argument("--case", default=None, help="demo case title (default depends on the check)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/nni_browser_checks"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fn, case = CHECKS[args.check]
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=GPU_FLAGS["none" if args.check == "no-webgl" else args.gpu])
        pg = await browser.new_page(viewport={"width": 1440, "height": 900})
        await open_case(pg, args.base, args.password, args.case or case, wait=args.check != "no-webgl")
        await fn(pg, args.out)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
