"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific -----------------------------------------------------------------------------------------------
import base64  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from core.processing_defaults import DEFAULTS  # noqa: E402
from web import catalog  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def test_catalogue_matches_desktop_controls():
    """Colormap names and mask colours are copied from the Qt controls panel; keep them in step."""
    src = (ROOT / "gui" / "controls_panel.py").read_text()
    combo = re.search(r"colormap_combo\.addItems\(\[(.*?)\]\)", src, re.S).group(1)
    assert re.findall(r'"(\w+)"', combo) == catalog.DESKTOP_COLORMAPS
    colors = re.findall(r"\d: \[(\d+), (\d+), (\d+)\],?\s+# (\w+)", src)
    assert {n: [int(r), int(g), int(b)] for r, g, b, n in colors} == catalog.MASK_COLORS


def test_luts_and_ops_from_repo_code():
    r = client.get("/api/luts").json()
    assert len(r["luts"]["viridis"]) == 256 and r["luts"]["viridis"][0] == [68, 1, 84]
    assert r["display"]["frame_rate"] > 0 and r["display"]["mask_color"] == [255, 0, 0]
    ops = {o["id"]: o for o in client.get("/api/ops").json()["operations"]}
    assert ops["gaussian"]["params"][0]["default"] == DEFAULTS["filter"]["gaussian"]["sigma"]
    assert ops["median"]["params"][0]["default"] == DEFAULTS["filter"]["median"]["size"]
    assert [p["default"] for p in ops["stack_contrast"]["params"]] == [2, 98]


def test_demo_files_exist_and_read():
    for d in client.get("/api/demos").json()["demos"]:
        arr = tifffile.imread(STATIC / d["file"])
        assert arr.shape == (24, 128, 128)
    mask = tifffile.imread(STATIC / "demo" / "demo_mask.tif")
    assert set(np.unique(mask)) <= {0, 255}


def test_demo_data_is_current(tmp_path):
    """scripts/make_demo_data.py still produces the committed demo (statistics, tolerant of RNG/libm differences)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_demo_data
    make_demo_data.make(tmp_path)
    for name in ("demo_cytoplasm.tif", "demo_puncta.tif", "demo_mask.tif"):
        new, old = tifffile.imread(tmp_path / name).astype(float), tifffile.imread(STATIC / "demo" / name).astype(float)
        assert new.shape == old.shape
        assert abs(new.mean() - old.mean()) <= 0.01 * old.mean(), name


def _b64(a):
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": base64.b64encode(a.astype(a.dtype.newbyteorder("<")).tobytes()).decode()}


def _from(r):
    return np.frombuffer(base64.b64decode(r["data"]), dtype=np.dtype(r["dtype"]).newbyteorder("<")).reshape(r["shape"])


def test_process_runs_image_processor():
    from core.image_processor import ImageProcessor
    frame = tifffile.imread(STATIC / "demo" / "demo_cytoplasm.tif")[0]
    r = client.post("/api/process", json={"op": "gaussian", "params": {"sigma": 2}, "frame": _b64(frame)})
    assert r.status_code == 200
    assert np.array_equal(_from(r.json()["image"]), ImageProcessor().apply_filter(frame, "gaussian", sigma=2))
    m = client.post("/api/process", json={"op": "threshold", "params": {}, "frame": _b64(frame)}).json()
    assert m["kind"] == "mask" and _from(m["image"]).max() == 1
    pts = client.post("/api/process", json={"op": "corner", "params": {}, "frame": _b64(frame)}).json()
    assert pts["kind"] == "points" and len(pts["points"]) > 0


def test_register_and_bad_requests():
    stack = tifffile.imread(STATIC / "demo" / "demo_cytoplasm.tif")[:3]
    r = client.post("/api/register", json={"frames": _b64(stack), "reference": 0, "params": {"upsample_factor": 4}})
    assert r.status_code == 200 and len(r.json()["shifts"]) == 3
    assert client.post("/api/process", json={"op": "nope", "frame": _b64(stack[0])}).status_code == 400
    assert client.post("/api/process", json={"op": "gaussian", "frame": {"shape": [2, 2], "dtype": "uint16", "data": "AAAA"}}).status_code == 400
    assert client.post("/api/register", json={"frames": _b64(stack), "reference": 9}).status_code == 400


def test_tiff_fixtures_still_read_by_tifffile():
    """The committed TIFF fixtures still give tifffile the numbers in tiff_expected.json (what the JS test checks)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_tiff_fixtures as mk
    expected = json.loads((FIX / "tiff_expected.json").read_text())
    have_codecs = importlib.util.find_spec("imagecodecs") is not None
    for name, e in expected.items():
        if not have_codecs and name in ("lzw_pred.tif", "lzw_u8.tif", "float_pred.tif"):
            continue  # tifffile needs imagecodecs for LZW and the floating-point predictor
        got = mk.expected_for(FIX / "tiff" / name, e["note"])
        for k in ("T", "Z", "C", "H", "W", "dtype"):
            assert got[k] == e[k], (name, k)
        for k in ("sum", "sumsq", "weighted"):
            assert np.isclose(got[k], e[k], rtol=1e-9), (name, k)


def test_ops_fixture_is_current():
    """tests/fixtures/ops_cases.json still matches the Python code (so the JS test compares with today's Python)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_ops_fixture
    new, old = make_ops_fixture.build(), json.loads((FIX / "ops_cases.json").read_text())

    def cmp(a, b, path):
        if isinstance(b, dict):
            assert set(a) == set(b), path
            for k in b:
                cmp(a[k], b[k], f"{path}.{k}")
        elif isinstance(b, list):
            assert len(a) == len(b), path
            if b and all(isinstance(x, (int, float)) or x is None for x in b):
                x, y = np.array(a, dtype=float), np.array(b, dtype=float)
                assert np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True), path
            else:
                for i, (p, q) in enumerate(zip(a, b)):
                    cmp(p, q, f"{path}[{i}]")
        elif isinstance(b, float):
            assert np.isclose(a, b, rtol=1e-9), path
        else:
            assert a == b, path
    cmp(json.loads(json.dumps(new)), old, "ops")


def test_desktop_loader_matches_browser_layout():
    """ImageStack (desktop) arranges hyperstacks like tiff_reader.js: frames = T x Z (z fastest), channels apart."""
    from core.image_loader import ImageStack
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_tiff_fixtures as mk
    for name in ("imagej_hyper.tif", "ome.ome.tif"):
        canon = mk.canonical(FIX / "tiff" / name)[0]  # (T, Z, C, Y, X), the order the JS test checks
        T, Z, C, H, W = canon.shape
        st = ImageStack()
        assert st.load_tiff(str(FIX / "tiff" / name)) and st.n_channels == C
        for c in range(C):
            st.set_channel(c)
            assert np.array_equal(st.data, canon[:, :, c].reshape(T * Z, H, W)), (name, c)
