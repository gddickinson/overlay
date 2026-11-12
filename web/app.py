"""FastAPI web app for the Advanced TIFF Stack Viewer: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8128, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. POST routes
(Python processing) exist only in the local app.
"""
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web import catalog, compute

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "TIFF Stack Viewer"
DESCRIPTION = ("Open microscopy TIFF stacks in the browser, overlay channels and masks with LUTs and contrast, and run "
               "the repository's processing and analysis operations.")

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary for the home page (counts come from the catalogue)."""
    ops = catalog.OPERATIONS
    return {"title": TITLE, "description": DESCRIPTION, "version": "1",
            "counts": {"luts": len(catalog.luts()), "ops_js": sum(o["where"] == "js" for o in ops),
                       "ops_python": sum(o["where"] == "python" for o in ops), "demos": len(catalog.DEMOS)}}


@app.get("/api/luts")
def luts():
    return {"luts": catalog.luts(), "mask_colors": catalog.MASK_COLORS, "desktop_colormaps": catalog.DESKTOP_COLORMAPS,
            "display": catalog.display_defaults()}


@app.get("/api/ops")
def ops():
    return {"operations": catalog.OPERATIONS}


@app.get("/api/demos")
def demos():
    return {"demos": catalog.DEMOS, "note": catalog.DEMO_NOTE}


@app.post("/api/process")
def process(body: dict = Body(...)):
    """Run an ImageProcessor operation on one frame: {op, params, frame: {shape, dtype, data(base64)}}."""
    try:
        frame = compute.decode(body["frame"])
        if frame.ndim != 2:
            raise ValueError("frame must be 2D")
        return compute.process(body.get("op"), body.get("params"), frame)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/register")
def register(body: dict = Body(...)):
    """Register a stack to one of its frames: {frames: {shape [T, H, W], dtype, data}, reference, params}."""
    try:
        frames = compute.decode(body["frames"])
        ref = int(body.get("reference", 0))
        if frames.ndim != 3 or not 0 <= ref < frames.shape[0]:
            raise ValueError("frames must be (T, H, W) and reference a valid frame")
        return compute.register(frames, ref, body.get("params"))
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/luts", "/ops", "/demos"]
