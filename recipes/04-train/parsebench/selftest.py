"""Offline check of the parts of this recipe that decide its numbers. No key, no GPU, no
Modal, no ParseBench install; under a second.

    python selftest.py

1. The dev/test split (waiparse/split.py): pages of one source report stay on one side,
   and about one report in five is dev. This split is what keeps the headline honest.
2. The endpoints (waiparse/endpoints.py): every server URL comes from your
   DOCPARSE_WORKSPACE or a whole-URL override, and nothing falls back to anyone else's.
3. The detector layout mapping (waiparse/layout_det.py) on a synthetic page: thresholds,
   the page-number rule, and text assigned from the layout-pass items.
4. The RL configs parse, and the page run's config carries the top-level [weight_broadcast]
   timeout (a per-component timeout is silently overwritten by prime-rl).
5. Every Python file in the recipe compiles.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

# waiparse/__init__.py registers the ParseBench pipelines, which needs parse-bench.
# The modules checked here do not, so load them without running the package __init__.
_pkg = types.ModuleType("waiparse")
_pkg.__path__ = [str(HERE / "waiparse")]
sys.modules.setdefault("waiparse", _pkg)
# layout_det imports aiohttp for the detector call, which this check never makes.
if "aiohttp" not in sys.modules:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("aiohttp")
        _stub.ClientSession = object  # type: ignore[attr-defined]  # only an annotation there
        sys.modules["aiohttp"] = _stub

from waiparse import endpoints, layout_det, split

# agent.DEFAULTS for the detector mapping (agent.py itself imports parse-bench).
DET_CFG = {
    "det_threshold": 0.2,
    "det_furniture_threshold": 0.4,
    "det_nms_iou": 0.7,
    "det_split": True,
    "det_keep_unmatched": True,
    "det_pic_keep_text": True,
    "det_unmath": True,
    "det_ocr": True,
}


def check_split() -> None:
    assert split.source_doc("reports/acme_annual_p12.pdf") == "acme_annual"
    assert split.source_doc("reports/acme_annual_page3.pdf") == "acme_annual"
    for stem in ("acme_annual", "fed_minutes", "oecd_outlook", "bank_q3"):
        sides = {split.side(f"x/{stem}_p{i}.pdf") for i in range(1, 9)}
        assert len(sides) == 1, f"{stem}: pages of one report landed on both sides"
    n = 5000
    dev = sum(split.side(f"doc{i}.pdf") == "dev" for i in range(n)) / n
    assert abs(dev - 1 / split.DEV_EVERY) < 0.02, dev
    print(f"split ok: pages of a report share a side; dev share {dev:.3f} (target 1/5)")


def check_endpoints() -> None:
    saved = {k: os.environ.pop(k, None) for k in endpoints.ENV_KEYS}
    try:
        try:
            endpoints.vlm_server()
        except RuntimeError as e:
            assert "DOCPARSE_WORKSPACE" in str(e)
        else:
            raise AssertionError("no workspace set, but a server URL came back")
        os.environ["DOCPARSE_WORKSPACE"] = "acme"
        assert endpoints.vlm_server() == "https://acme--docparse-vlm-server-serve.modal.run"
        assert endpoints.tuned_server() == "https://acme--docparse-vlm-tuned-server-serve.modal.run"
        assert endpoints.layout_url() == "https://acme--docparse-layout-detector-web.modal.run"
        os.environ["DOCPARSE_LAYOUT_URL"] = "http://localhost:8001"
        assert endpoints.layout_url() == "http://localhost:8001"
        assert endpoints.image_env() == {
            "DOCPARSE_WORKSPACE": "acme",
            "DOCPARSE_LAYOUT_URL": "http://localhost:8001",
        }
    finally:
        for k in endpoints.ENV_KEYS:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]
    print("endpoints ok: URLs from DOCPARSE_WORKSPACE, overrides win, no default workspace")


def check_layout() -> None:
    # A 1000 x 1000 px page. Detector boxes in px, layout-pass items in 0-1000 units.
    det = {
        "width": 1000,
        "height": 1000,
        "boxes": [
            {"bbox": [100, 100, 900, 150], "label": "paragraph_title", "score": 0.9, "order": 0},
            {"bbox": [100, 200, 900, 400], "label": "text", "score": 0.8, "order": 1},
            {"bbox": [100, 450, 900, 500], "label": "text", "score": 0.1, "order": 2},
            {"bbox": [480, 950, 520, 980], "label": "number", "score": 0.5, "order": 3},
            {"bbox": [480, 20, 520, 40], "label": "number", "score": 0.3, "order": 4},
        ],
    }
    items = [
        {"bbox": [100, 100, 900, 150], "category": "Section-header", "text": "Results"},
        {"bbox": [100, 200, 900, 400], "category": "Text", "text": "Revenue grew $x$ percent."},
        {"bbox": [480, 950, 520, 980], "category": "Page-footer", "text": "7"},
    ]
    page = {"width": 1000, "height": 1000, "items": items, "det": det}
    out = layout_det.build(page, DET_CFG)
    got = [(o["label"], o["value"].strip()) for o in out]
    assert got == [
        ("Section-header", "Results"),
        ("Text", "Revenue grew  x  percent."),  # det_unmath drops the $ delimiters
        ("Page-footer", "7"),  # a page number at the bottom; the 0.3 one at the top is cut
    ], got
    print(f"layout ok: {len(out)} boxes, score cut at 0.2, page-number rule, text from items")


def check_configs() -> None:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        print("configs skipped: tomllib needs Python 3.11+")
        return
    for p in sorted((HERE / "train" / "configs").glob("*.toml")):
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
        assert cfg["model"]["name"] == "Qwen/Qwen3.8-27B", p.name
        assert cfg["trainer"]["loss"]["import_path"] == "docparse_charts.losses.cispo_loss", p.name
        if p.name == "charts-pages-v1.toml":  # the smoke loads too few tasks to need it
            wb = cfg.get("weight_broadcast", {})
            assert wb.get("timeout", 0) >= 3600, f"{p.name}: top-level [weight_broadcast] timeout"
    print("configs ok: CISPO loss, Qwen3.8-27B, the page run waits 3600 s for the weight broadcast")


def check_compiles() -> None:
    files = sorted(p for p in HERE.rglob("*.py") if "__pycache__" not in p.parts)
    for p in files:
        py_compile.compile(str(p), doraise=True)
    print(f"compile ok: {len(files)} files")


def main() -> int:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    try:
        from whileai.config import provenance

        print(provenance(), file=sys.stderr)
    except ImportError:
        pass
    check_split()
    check_endpoints()
    check_layout()
    check_configs()
    check_compiles()
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
