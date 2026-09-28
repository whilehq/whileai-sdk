"""Detector-box layout for layout_pages (config layout_boxes="detector").

PP-DocLayoutV3 (serve/serve_layout.py) gives the boxes and classes; the text of each box
comes from the Qwen layout-pass items that overlap it. Markdown is untouched: it is still
emitted from the layout-pass items.
"""

import asyncio
import io
import os
from typing import Any

import aiohttp

from waiparse import endpoints

# PP-DocLayoutV3 label -> our Qwen label set (the scorer collapses both sides to
# Text / Section / Picture / Table / Page-header / Page-footer).
DET_LABELS = {
    "doc_title": "Title",
    "paragraph_title": "Section-header",
    "text": "Text",
    "content": "Text",
    "abstract": "Text",
    "reference": "Text",
    "reference_content": "Text",
    "aside_text": "Text",
    "algorithm": "Text",
    "footnote": "Footnote",
    "vision_footnote": "Footnote",
    "figure_title": "Caption",
    "formula": "Formula",
    "formula_number": "Text",
    "table": "Table",
    "image": "Picture",
    "chart": "Picture",
    "seal": "Picture",
    "header": "Page-header",
    "footer": "Page-footer",
    "number": "number",  # page number: header or footer by position
}

RAW_THRESHOLD = 0.2
_SEMS: dict[int, asyncio.Semaphore] = {}


def _sem() -> asyncio.Semaphore:
    loop = id(asyncio.get_running_loop())
    if loop not in _SEMS:
        _SEMS[loop] = asyncio.Semaphore(4)
    return _SEMS[loop]


async def detect(
    s: aiohttp.ClientSession, page, max_px: int = 1600, retries: int = 4
) -> dict[str, Any]:
    """Detector boxes for one page image: {"width", "height", "boxes": [{bbox px xyxy, label, score, order}]}.
    An unreachable detector returns no boxes (the page falls back to layout-pass boxes)."""
    from waiparse.render import fit

    img = fit(page, max_px)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    body = buf.getvalue()
    headers = {
        "Authorization": f"Bearer {os.environ.get('DOCPARSE_KEY', '')}",
        "Content-Type": "image/png",
    }
    # Resolved before the retry loop: a missing DOCPARSE_WORKSPACE must fail loudly, not
    # look like an unreachable detector.
    url = f"{endpoints.layout_url()}/detect?threshold={RAW_THRESHOLD}"
    last = None
    for attempt in range(retries):
        try:
            async with (
                _sem(),
                s.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300),
                    allow_redirects=False,
                ) as r,
            ):
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
                return await r.json()
        except Exception as e:
            last = e
            await asyncio.sleep(3 * (attempt + 1))
    return {"width": img.width, "height": img.height, "boxes": [], "error": str(last)[:300]}


# ---------------------------------------------------------------------- geometry
def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def det_boxes(det: dict, cfg: dict) -> list[dict]:
    """Detector boxes in [0,1] xyxy, thresholded, de-duplicated, mapped to our labels."""
    w, h = float(det.get("width") or 1), float(det.get("height") or 1)
    raw = sorted(
        (
            (ri, b)
            for ri, b in enumerate(det.get("boxes", []))
            if b["score"] >= cfg["det_threshold"]
        ),
        key=lambda rb: -rb[1]["score"],
    )
    kept: list[dict] = []
    for ri, b in raw:
        x1, y1, x2, y2 = b["bbox"]
        box = [max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)]
        if _area(box) <= 0 or any(_iou(box, k["box"]) > cfg["det_nms_iou"] for k in kept):
            continue
        label = DET_LABELS.get(b["label"], "Text")
        if label == "number":
            label = "Page-header" if (box[1] + box[3]) / 2 < 0.5 else "Page-footer"
        if label in ("Page-header", "Page-footer") and b["score"] < cfg.get(
            "det_furniture_threshold", 0.0
        ):
            continue
        kept.append(
            {"box": box, "label": label, "score": b["score"], "order": b.get("order", 0), "ri": ri}
        )
    kept.sort(key=lambda k: k["order"])
    return kept


def assign_text(boxes: list[dict], items: list[dict], cfg: dict, aspect: float = 1.0) -> list[dict]:
    """Give each detector box the layout-pass content inside it. Returns the layout-pass
    items no detector box covers. `aspect` = page height / width."""
    for b in boxes:
        b["parts"] = []  # (item_idx, word_idx, kind, content)
    unmatched = []
    for q_i, it in enumerate(items):
        qb = [v / 1000 for v in it["bbox"]]
        qa = _area(qb)
        if qa <= 0:
            continue
        cat = it["category"]
        if cat == "Table":
            content, kind = it.get("html", ""), "table"
        elif cat == "Picture":
            content, kind = _plain_table(it.get("html", "") or it.get("charts_html", "")), "text"
        else:
            content, kind = it.get("text", ""), "text"
        cover = sorted(((_inter(qb, b["box"]) / qa, j) for j, b in enumerate(boxes)), reverse=True)
        cover = [(f, j) for f, j in cover if f >= 0.15]
        if not cover:
            unmatched.append(
                {"box": qb, "label": cat, "item_idx": q_i, "content": content, "kind": kind}
            )
            continue
        best_f, best_j = cover[0]
        if (
            cfg.get("det_pic_keep_text")
            and boxes[best_j]["label"] == "Picture"
            and cat != "Picture"
        ):
            # text inside a figure stays its own element (Picture content is not scored as text)
            unmatched.append(
                {"box": qb, "label": cat, "item_idx": q_i, "content": content, "kind": kind}
            )
            continue
        # text boxes this item spans: a real share of the item, or mostly inside it
        cands = [
            j
            for j, b in enumerate(boxes)
            if b["label"] not in ("Table", "Picture")
            and _inter(qb, b["box"]) > 0
            and (_inter(qb, b["box"]) / qa >= 0.05 or _inter(qb, b["box"]) / _area(b["box"]) >= 0.5)
        ]
        if (
            cfg["det_split"]
            and kind == "text"
            and len(cands) > 1
            and best_f < 0.9
            and boxes[best_j]["label"] not in ("Table", "Picture")
        ):
            _split_words(content, qb, cands, boxes, q_i, aspect)
        else:
            boxes[best_j]["parts"].append((q_i, 0, kind, content))
    return unmatched


def _word_points(text: str, qb, aspect: float) -> list[tuple[str, float, float]]:
    """Place each word of `text` inside box `qb` ([0,1] xyxy): visual lines spread evenly
    top to bottom, characters evenly left to right. Line height from the box area and
    character count (a character is about half a line high wide)."""
    x1, y1, x2, y2 = qb
    w, h = (x2 - x1), (y2 - y1) * aspect  # both in page-width units
    chars = max(len(text), 1)
    lh = max((2 * w * h / chars) ** 0.5, 1e-6)
    cpl = max(int(w / (0.5 * lh)), 1)  # characters per visual line
    vis: list[list[tuple[str, int]]] = []  # visual lines of (word, char offset)
    for para in [ln for ln in text.splitlines() if ln.strip()]:
        line, off = [], 0
        for word in para.split():
            if off and off + len(word) > cpl:
                vis.append(line)
                line, off = [], 0
            line.append((word, off))
            off += len(word) + 1
        if line:
            vis.append(line)
    pts = []
    n = max(len(vis), 1)
    for li, line in enumerate(vis):
        y = y1 + (li + 0.5) / n * (y2 - y1)
        span = max(max(o + len(wd) for wd, o in line), cpl if len(vis) > 1 else 1)
        for wd, o in line:
            pts.append((wd, x1 + (o + len(wd) / 2) / span * (x2 - x1), y))
    return pts


def _split_words(text: str, qb, cands: list[int], boxes, q_i: int, aspect: float) -> None:
    """Split one item's words across the detector boxes it spans, by where each word sits."""
    per: dict[int, list[tuple[int, str]]] = {}
    for w_i, (wd, x, y) in enumerate(_word_points(text, qb, aspect)):

        def dist(j, x=x, y=y):
            b = boxes[j]["box"]
            dx = max(b[0] - x, 0, x - b[2])
            dy = (max(b[1] - y, 0, y - b[3])) * aspect
            return dx * dx + dy * dy

        j = min(cands, key=dist)
        per.setdefault(j, []).append((w_i, wd))
    for j, words in per.items():
        boxes[j]["split"] = True
        boxes[j]["parts"].append((q_i, words[0][0], "text", " ".join(wd for _, wd in words)))


OCR_PROMPT = """Transcribe all text in this image exactly as printed, in reading order.
Keep line breaks between separate lines of a list. Math as LaTeX without delimiters.
No commentary, no code fences. If there is no text, reply with nothing."""


def _wants_ocr(b: dict, cfg: dict) -> bool:
    """Boxes whose layout-pass text is missing or was cut out of a larger item."""
    if b["label"] in ("Table", "Picture"):
        return False  # tables come from the table pass; Picture content is not scored as text
    return not b["parts"] or bool(b.get("split"))


async def ocr_regions(vlm, s, page_img, items: list[dict], det: dict) -> dict[str, str]:
    """Re-read the detector boxes the layout-pass text does not cleanly cover (empty, or
    split out of a larger item), from full-resolution crops. Keyed by raw box index. Run
    at the loosest mapping settings so any stricter config finds its boxes here."""
    from waiparse.render import crop

    loose = {"det_threshold": RAW_THRESHOLD, "det_nms_iou": 0.7, "det_split": True}
    boxes = det_boxes(det, loose)
    aspect = page_img.height / max(page_img.width, 1)
    assign_text(boxes, items, loose, aspect)
    todo = [b for b in boxes if _wants_ocr(b, loose)]

    async def one(b):
        bbox1000 = [v * 1000 for v in b["box"]]
        out = (await vlm.ask(s, crop(page_img, bbox1000, pad=3), OCR_PROMPT, max_tokens=2048))[0]
        return str(b["ri"]), out.strip().strip("`").strip()

    res = await asyncio.gather(*(one(b) for b in todo), return_exceptions=True)
    return dict(r for r in res if not isinstance(r, Exception))


def _unmath(text: str) -> str:
    """Drop $ / $$ math delimiters so the LaTeX inside reads as text (the scorer deletes
    delimited math, while ground-truth text keeps the raw LaTeX tokens)."""
    import re

    return re.sub(r"\$\$?", " ", text)


def _plain_table(html: str) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def build(page: dict, cfg: dict) -> list[dict]:
    """[{box [0,1] xyxy, label, value, type}] in reading order for one page."""
    boxes = det_boxes(page["det"], cfg)
    items = page["items"]
    unmatched = assign_text(boxes, items, cfg, float(page["height"]) / float(page["width"] or 1))
    out = []
    for b in boxes:
        parts = sorted(b["parts"])
        if b["label"] == "Table":
            tables = [c for _, _, k, c in parts if k == "table" and c]
            value = "\n".join(tables) if tables else " ".join(c for *_, c in parts if c)
            typ = "table"
        else:
            value = "\n".join(c if k == "text" else _plain_table(c) for _, _, k, c in parts if c)
            typ = "image" if b["label"] == "Picture" else "text"
        label = b["label"]
        ocr = page["det"].get("ocr", {}).get(str(b["ri"]))
        if cfg.get("det_ocr") and ocr and _wants_ocr(b, cfg):
            value = ocr
        if cfg.get("det_unmath"):
            value = _unmath(value)
        key = parts[0][0] if parts else None
        out.append(
            {
                "box": b["box"],
                "label": label,
                "value": value,
                "type": typ,
                "key": key,
                "score": b["score"],
            }
        )
    if cfg["det_keep_unmatched"]:
        for u in unmatched:
            if u["label"] in ("Table",) or u["content"]:
                v = (
                    _unmath(u["content"])
                    if cfg.get("det_unmath") and u["kind"] == "text"
                    else u["content"]
                )
                out.append(
                    {
                        "box": u["box"],
                        "label": u["label"],
                        "value": v,
                        "type": "table"
                        if u["kind"] == "table"
                        else ("image" if u["label"] == "Picture" else "text"),
                        "key": u["item_idx"],
                        "score": 1.0,
                    }
                )
    # Reading order: layout-pass order of the first matched item; empty boxes follow
    # their predecessor in detector order.
    prev = -1.0
    for o in out:
        if o["key"] is None:
            o["key"] = prev + 0.5
        prev = o["key"]
    out.sort(key=lambda o: o["key"])
    return out
