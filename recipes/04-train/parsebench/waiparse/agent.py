"""wai_agent: the document parsing agent, as a ParseBench provider.

Per page:
  1. layout   one VLM call: boxes + categories + text for every text element.
  2. regions  every Table crop -> table specialist (HTML with thead/th, spans);
              every Picture crop -> chart specialist (chart -> data table, else NONE).
              Chart extraction can sample k times and vote cell by cell.
  3. emit     markdown in the shape the scorers read + layout_pages for grounding.
"""

import asyncio
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from parse_bench.evaluation.layout_adapters.adapters import QwenLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import register_layout_adapter
from parse_bench.inference.providers.base import Provider, ProviderPermanentError
from parse_bench.inference.providers.parse.qwen import QwenProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

from waiparse import endpoints, layout_det, prompts
from waiparse.client import VLM
from waiparse.layout_det import detect as _detect_layout
from waiparse.render import crop, fit, load_pages

DEFAULTS: dict[str, Any] = {
    "model": "qwen3.8-27b",
    "dpi": 200,
    "layout_thinking": False,
    "layout_effort": None,
    "table_thinking": True,
    "table_effort": "medium",
    "chart_thinking": True,
    "chart_effort": "xhigh",
    "chart_samples": 1,  # >1: self-consistency vote per cell
    "table_samples": 1,  # >1: same vote for tables (dev tables swung 76-81 run to run at n=1)
    "chart_page_px": 2048,  # long side of the page image the chart pass sees
    "chart_scope": "page",  # "page": one full-page pass sees titles + legends; "crop": per Picture box
    "keep_page_furniture": True,  # emit Page-header / Page-footer text in markdown
    "heading_levels": True,  # Section-header depth from box height rank
    "style_pass": False,  # re-read text blocks from crops for bold / strike / sup / sub
    "style_mode": "transfer",  # "transfer" styled spans onto the page text | "replace" the text
    # layout_pages boxes + classes: "qwen" = the layout pass; "detector" = PP-DocLayoutV3
    # (serve/serve_layout.py) boxes, text taken from the overlapping layout-pass items.
    "layout_boxes": "qwen",
    # det_* settings apply only with layout_boxes="detector"; values tuned on dev (serve/relayout.py).
    "det_threshold": 0.2,  # min detector score kept in layout_pages (raw keeps >= 0.2)
    "det_furniture_threshold": 0.4,  # Page-header / Page-footer boxes need more confidence
    "det_nms_iou": 0.7,  # class-agnostic duplicate removal
    "det_split": True,  # split a layout-pass item's text across the detector boxes it spans
    "det_keep_unmatched": True,  # layout-pass items no detector box covers stay as their own boxes
    "det_pic_keep_text": True,  # text items inside a figure stay their own boxes, not merged into it
    "det_unmath": True,  # drop $ delimiters in layout values (the scorer deletes delimited math)
    "det_ocr": True,  # re-read empty / split detector boxes from crops (one small VLM call each)
}

STYLED = {"Text", "List-item", "Caption", "Footnote", "Title", "Section-header"}


@register_provider("wai_agent")
class WaiAgent(Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self.cfg = {**DEFAULTS, **(base_config or {})}
        self.vlm = VLM(model=self.cfg["model"])
        # The chart passes can run on other weights (the chart RL adapter) and another server.
        chart_server = self.cfg.get("chart_server")
        if chart_server == "tuned":  # pipelines.TUNED_SERVER: the adapter app in your workspace
            chart_server = endpoints.tuned_server()
        self.chart_vlm = VLM(
            model=self.cfg.get("chart_model") or self.cfg["model"], server=chart_server
        )

    # ------------------------------------------------------------------ stages
    async def _layout(self, s: aiohttp.ClientSession, page) -> list[dict[str, Any]]:
        out = (
            await self.vlm.ask(
                s,
                fit(page),
                prompts.LAYOUT,
                thinking=self.cfg["layout_thinking"],
                effort=self.cfg["layout_effort"],
                max_tokens=16384,
            )
        )[0]
        try:
            items = QwenProvider._parse_layout_items(out)
        except ProviderPermanentError:
            return []
        return [
            {"bbox": list(i.coords), "category": _canon(i.category), "text": i.text}
            for i in items
            if len(i.coords) == 4
        ]

    async def _table(self, s, page, bbox) -> str:
        outs = await self.vlm.ask(
            s,
            crop(page, bbox),
            prompts.TABLE,
            thinking=self.cfg["table_thinking"],
            effort=self.cfg["table_effort"],
            max_tokens=16384,
            n=self.cfg["table_samples"],
        )
        tables = [t for t in (_extract_tables(o) for o in outs) if t]
        if not tables:
            return ""
        return _vote_tables(tables) if len(tables) > 1 else tables[0]

    async def _chart(self, s, page, bbox) -> str:
        k = self.cfg["chart_samples"]
        outs = await self.chart_vlm.ask(
            s,
            crop(page, bbox),
            prompts.CHART,
            thinking=self.cfg["chart_thinking"],
            effort=self.cfg["chart_effort"],
            max_tokens=16384,
            n=k,
        )
        tables = [
            _extract_tables(o)
            for o in outs
            if o.strip() and not o.strip().upper().startswith("NONE")
        ]
        if len(tables) * 2 <= len(outs):  # majority says "not a chart"
            return ""
        return _vote_tables(tables) if len(tables) > 1 else tables[0]

    async def _chart_page(self, s, page) -> str:
        k = self.cfg["chart_samples"]
        outs = await self.chart_vlm.ask(
            s,
            fit(page, self.cfg["chart_page_px"]),
            prompts.CHART_PAGE,
            thinking=self.cfg["chart_thinking"],
            effort=self.cfg["chart_effort"],
            max_tokens=24576,
            n=k,
        )
        tables = [
            _extract_tables(o)
            for o in outs
            if o.strip() and not o.strip().upper().startswith("NONE")
        ]
        tables = [t for t in tables if t]
        if len(tables) * 2 <= len(outs):
            return ""
        return _vote_tables(tables) if len(tables) > 1 else tables[0]

    async def _style(self, s, page, it: dict) -> None:
        """Re-read one text block from a full-resolution crop with styling markup. The page
        pass sees the page at 2048 px and drops bold, strikethrough and superscripts."""
        out = (
            await self.vlm.ask(s, crop(page, it["bbox"], pad=4), prompts.STYLE, max_tokens=4096)
        )[0]
        out = re.sub(r"^```\w*\n?|\n?```$", "", out.strip())
        if not out or _token_overlap(_plain(out), _plain(it["text"])) < 0.6:
            return
        if self.cfg["style_mode"] == "replace":
            it["text_page"], it["text"] = it["text"], out
        else:  # "transfer": keep the page pass's words, copy only the styled spans onto them
            it["text_page"], it["text"] = it["text"], _transfer_styles(it["text"], out)

    async def _page(self, s, page) -> dict[str, Any]:
        det = None
        if self.cfg["layout_boxes"] == "detector":
            items, det = await asyncio.gather(self._layout(s, page), _detect_layout(s, page))
        else:
            items = await self._layout(s, page)
        page_scope = self.cfg["chart_scope"] == "page"
        jobs = []
        for it in items:
            if it["category"] == "Table":
                jobs.append(self._table(s, page, it["bbox"]))
            elif it["category"] == "Picture" and not page_scope:
                jobs.append(self._chart(s, page, it["bbox"]))
            else:
                jobs.append(None)
        chart_top = _det_chart_top(det)
        if page_scope and items and (_maybe_chart(items) or chart_top is not None):
            anchor = next((it for it in items if it["category"] == "Picture"), None)
            if (
                anchor is None
            ):  # chart read as text by the layout pass: anchor on the first tick-like block
                anchor = next((it for it in items if _chart_like(it)), None)
            if (
                anchor is None
            ):  # only the detector saw a chart: anchor on the first item at/below its top
                anchor = next((it for it in items if it["bbox"][1] >= chart_top - 5), items[-1])
            for it in items:  # axis ticks / data labels read as Text are replaced by the tables
                if _chart_like(it):
                    it["chart_anchor"] = True
            jobs.append(self._chart_page(s, page))
            items_for_jobs = [*items, {"category": "_charts", "anchor": anchor}]
        else:
            items_for_jobs = items
        coros = [j for j in jobs if j is not None]
        styled = (
            [
                it
                for it in items
                if it["category"] in STYLED
                and it.get("text", "").strip()
                and not it.get("chart_anchor")
            ]
            if self.cfg["style_pass"]
            else []
        )
        n = len(coros)
        gathered = await asyncio.gather(
            *coros, *(self._style(s, page, it) for it in styled), return_exceptions=True
        )
        results = iter(gathered[:n])
        for it, j in zip(items_for_jobs, jobs):
            if j is None:
                continue
            r = next(results)
            if it["category"] == "_charts":
                it["anchor"]["charts_html"] = "" if isinstance(r, Exception) else r
                continue
            it["html"] = "" if isinstance(r, Exception) else r
            if isinstance(r, Exception):
                it["error"] = str(r)[:300]
        out = {"width": page.width, "height": page.height, "items": items}
        if det is not None:
            if self.cfg["det_ocr"] and det.get("boxes"):
                det["ocr"] = await layout_det.ocr_regions(self.vlm, s, page, items, det)
            out["det"] = det
        return out

    async def _doc(self, pages) -> list[dict[str, Any]]:
        async with aiohttp.ClientSession() as s:
            return list(await asyncio.gather(*(self._page(s, p) for p in pages)))

    # ------------------------------------------------------------------ provider API
    def run_inference(
        self, pipeline: PipelineSpec, request: InferenceRequest
    ) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"wai_agent only supports PARSE, got {request.product_type}"
            )
        started = datetime.now()
        path = Path(request.source_file_path)
        if not path.exists():
            raise ProviderPermanentError(f"Source file not found: {path}")
        pages = load_pages(path, dpi=self.cfg["dpi"])
        raw: dict[str, Any] = {"_config": {"model": self.cfg["model"], **self.cfg}}
        try:
            raw["pages"] = asyncio.run(self._doc(pages))
        except Exception as e:  # scored as an empty page, not a crash
            raw["pages"], raw["_error"] = [], f"{type(e).__name__}: {e}"
        done = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw,
            started_at=started,
            completed_at=done,
            latency_in_ms=int((done - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        raw = raw_result.raw_output
        pages_ir, layout_pages, mds = [], [], []
        levels = _heading_levels(raw.get("pages", [])) if self.cfg["heading_levels"] else {}
        for idx, page in enumerate(raw.get("pages", [])):
            md = _emit(page["items"], levels, idx, self.cfg["keep_page_furniture"])
            pages_ir.append(PageIR(page_index=idx, markdown=md))
            mds.append(md)
            layout_pages.append(_layout_page(page, idx + 1, md, self.cfg))
        out = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages_ir,
            layout_pages=layout_pages,
            markdown="\n\n".join(mds),
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw,
            output=out,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


@register_layout_adapter("wai_agent", priority=95)
class WaiLayoutAdapter(QwenLayoutAdapter):
    """Our layout_pages use the Qwen label set and [0,1] xywh segments."""

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        return isinstance(inference_result.output, ParseOutput) and bool(
            inference_result.output.layout_pages
        )


# ---------------------------------------------------------------------- helpers
_CANON = {c.lower().replace("_", "-"): c for c in prompts.CATEGORIES}
_CANON.update(
    {
        "section-title": "Section-header",
        "heading": "Section-header",
        "list": "List-item",
        "figure": "Picture",
        "image": "Picture",
        "chart": "Picture",
        "header": "Page-header",
        "footer": "Page-footer",
        "equation": "Formula",
    }
)


def _canon(cat: str) -> str:
    return _CANON.get(cat.strip().lower().replace("_", "-").replace(" ", "-"), "Text")


def _extract_tables(text: str) -> str:
    text = re.sub(r"```(?:html)?", "", text)
    tables = re.findall(r"<table[\s\S]*?</table>", text, flags=re.I)
    return "\n\n".join(tables)


def _cells(table: str) -> list[list[str]]:
    rows = re.findall(r"<tr[\s\S]*?</tr>", table, flags=re.I)
    return [re.findall(r"<t[hd](?:\s[^>]*)?>([\s\S]*?)</t[hd]>", r, flags=re.I) for r in rows]


def _vote_tables(tables: list[str]) -> str:
    """Self-consistency: keep the most common grid shape, then per-cell majority /
    median-of-numbers across the samples with that shape."""
    shapes = Counter(tuple(len(r) for r in _cells(t)) for t in tables)
    shape = shapes.most_common(1)[0][0]
    same = [t for t in tables if tuple(len(r) for r in _cells(t)) == shape]
    base = same[0]
    grids = [_cells(t) for t in same]
    voted = []
    for i, row in enumerate(grids[0]):
        vr = []
        for j in range(len(row)):
            vals = [g[i][j].strip() for g in grids]
            nums = [_num(v) for v in vals]
            if all(n is not None for n in nums):
                med = sorted(nums)[len(nums) // 2]
                vr.append(next(v for v, n in zip(vals, nums) if n == med))
            else:
                vr.append(Counter(vals).most_common(1)[0][0])
        voted.append(vr)
    it = iter(v for r in voted for v in r)
    return re.sub(
        r"(<t[hd](?:\s[^>]*)?>)([\s\S]*?)(</t[hd]>)",
        lambda m: m.group(1) + next(it) + m.group(3),
        base,
        flags=re.I,
    )


def _num(v: str) -> float | None:
    s = re.sub(r"[,\s$€£%]", "", v)
    try:
        return float(s)
    except ValueError:
        return None


def _heading_levels(pages: list[dict]) -> dict[tuple[int, int], int]:
    """Section-header depth (2..4) from the rank of its single-line height, doc-wide.
    Taller type = shallower heading. Titles stay at level 1."""
    heights = {}
    for p_i, page in enumerate(pages):
        for i_i, it in enumerate(page["items"]):
            if it["category"] == "Section-header":
                lines = max(1, it["text"].count("\n") + 1)
                heights[(p_i, i_i)] = round(
                    (it["bbox"][3] - it["bbox"][1]) / lines / 3
                )  # ~3/1000 buckets
    ranks = sorted(set(heights.values()), reverse=True)
    return {k: 2 + min(ranks.index(h), 2) for k, h in heights.items()}


def _emit(items: list[dict], levels: dict, p_i: int, furniture: bool) -> str:
    parts = []
    has_charts = any(it.get("charts_html") for it in items)
    for i_i, it in enumerate(items):
        cat, text = it["category"], it.get("text", "").strip()
        if it.get("charts_html"):
            parts.append(it["charts_html"])
            if cat == "Picture":
                continue  # the tables replace the picture
        if has_charts and it.get("chart_anchor"):
            continue  # axis ticks / data labels the layout pass read as text
        if cat == "Table" or cat == "Picture":
            if it.get("html"):
                parts.append(it["html"])
        elif cat == "Title" and text:
            parts.append("# " + _unbold(text))
        elif cat == "Section-header" and text:
            parts.append("#" * levels.get((p_i, i_i), 2) + " " + _unbold(text))
        elif cat == "Formula" and text:
            parts.append(f"$$\n{text}\n$$")
        elif cat in ("Page-header", "Page-footer"):
            if furniture and text:
                parts.append(text)
        elif text:
            parts.append(text)
    return QwenProvider._sanitize_html_attributes("\n\n".join(parts))


def _unbold(t: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", t).replace("\n", " ")


def _layout_page(
    page: dict, page_number: int, md: str, cfg: dict | None = None
) -> ParseLayoutPageIR:
    if cfg and cfg.get("layout_boxes") == "detector" and page.get("det", {}).get("boxes"):
        return _det_layout_page(page, page_number, md, cfg)
    items = []
    for it in page["items"]:
        x1, y1, x2, y2 = it["bbox"]
        seg = LayoutSegmentIR(
            x=x1 / 1000,
            y=y1 / 1000,
            w=(x2 - x1) / 1000,
            h=(y2 - y1) / 1000,
            confidence=1.0,
            label=it["category"],
        )
        cat = it["category"]
        if cat == "Table":
            value, typ = it.get("html", ""), "table"
        elif cat == "Picture":
            value, typ = _table_text(it.get("html", "")), "image"
        else:
            value, typ = it.get("text", ""), "text"
        items.append(LayoutItemIR(type=typ, value=value, bbox=seg, layout_segments=[seg]))
    return ParseLayoutPageIR(
        page_number=page_number,
        width=float(page["width"]),
        height=float(page["height"]),
        md=md,
        items=items,
    )


def _det_layout_page(page: dict, page_number: int, md: str, cfg: dict) -> ParseLayoutPageIR:
    """layout_pages from detector boxes + classes, text from the layout-pass items (waiparse/layout_det.py)."""
    items = []
    for b in layout_det.build(page, cfg):
        x1, y1, x2, y2 = b["box"]
        seg = LayoutSegmentIR(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=1.0, label=b["label"])
        items.append(
            LayoutItemIR(type=b["type"], value=b["value"], bbox=seg, layout_segments=[seg])
        )
    return ParseLayoutPageIR(
        page_number=page_number,
        width=float(page["width"]),
        height=float(page["height"]),
        md=md,
        items=items,
    )


_NUMTOK = re.compile(r"^[-+(]?[$€£]?\d[\d.,]*%?\)?$")


def _numeric_density(it: dict) -> float:
    toks = it.get("text", "").split()
    return sum(bool(_NUMTOK.match(t)) for t in toks) / max(len(toks), 1) * min(len(toks), 30)


def _maybe_chart(items: list[dict]) -> bool:
    """Run the page chart pass when layout saw a Picture, or a text block that reads like
    axis ticks and data labels (the layout pass sometimes transcribes a chart as Text)."""
    return any(it["category"] == "Picture" or _chart_like(it) for it in items)


def _chart_like(it: dict) -> bool:
    return it["category"] == "Text" and _numeric_density(it) >= 6


def _table_text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _plain(t: str) -> str:
    t = re.sub(r"</?(sup|sub|b|strong|s|del)>|\*\*|~~", "", t)
    return re.sub(r"[^\w\s]", " ", t.lower())


def _token_overlap(a: str, b: str) -> float:
    """Share of b's tokens that a reproduces (guards the style pass against a bad crop)."""
    ta, tb = Counter(a.split()), Counter(b.split())
    if not tb:
        return 0.0
    return sum((ta & tb).values()) / sum(tb.values())


_STYLED_SPAN = re.compile(r"\*\*(.+?)\*\*|~~(.+?)~~|<(sup|sub)>(.+?)</\3>", re.S)


def _strip_marks(t: str) -> str:
    return re.sub(r"\*\*|~~|</?(?:sup|sub|b|strong)>", "", t)


def _transfer_styles(page_text: str, styled: str) -> str:
    """Wrap spans of `page_text` that `styled` marks as bold/strike/sup/sub, matching the span's
    words exactly; words the page pass did not produce are never added."""
    base = _strip_marks(page_text)
    out, pos = base, 0
    spans = []
    for m in _STYLED_SPAN.finditer(styled):
        if m.group(1) is not None:
            spans.append(("**", "**", _strip_marks(m.group(1)).strip()))
        elif m.group(2) is not None:
            spans.append(("~~", "~~", _strip_marks(m.group(2)).strip()))
        else:
            tag = m.group(3)
            spans.append((f"<{tag}>", f"</{tag}>", _strip_marks(m.group(4)).strip()))
    for open_, close, span in spans:
        if not span:
            continue
        # Match across line breaks / spacing differences between the two reads.
        pat = r"\s+".join(re.escape(w) for w in span.split())
        m = re.compile(pat).search(out, pos)
        if not m:
            m = re.compile(pat).search(out)  # out-of-order span
            if not m:
                continue
        out = out[: m.start()] + open_ + m.group(0) + close + out[m.end() :]
        pos = m.start() + len(open_) + len(m.group(0)) + len(close)
    return out


def _det_chart_top(det: dict | None, min_score: float = 0.3) -> float | None:
    """Top edge (0-1000) of the first chart the layout detector found, or None. The layout pass
    sometimes reads a chart as loose label lines with no numbers, which the text heuristic misses."""
    if not det or not det.get("height"):
        return None
    tops = [
        b["bbox"][1] / det["height"] * 1000
        for b in det.get("boxes", [])
        if b.get("label") == "chart" and b.get("score", 0) >= min_score
    ]
    return min(tops) if tops else None
