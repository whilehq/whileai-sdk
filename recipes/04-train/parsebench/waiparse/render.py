"""Page rendering and cropping."""

import threading
from pathlib import Path

from PIL import Image

MAX_PAGE_PX = 2048  # long side sent to the VLM for the whole-page layout pass
_PDFIUM = threading.Lock()  # pdfium is not thread-safe; ParseBench runs documents in threads


def load_pages(path: Path, dpi: int = 200) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        with _PDFIUM:
            pdf = pdfium.PdfDocument(str(path))
            out = []
            try:
                for i in range(len(pdf)):
                    page = pdf[i]
                    bitmap = page.render(scale=dpi / 72)
                    out.append(bitmap.to_pil().convert("RGB"))
                    # Close children inside the lock: left to the GC they are freed on
                    # another thread and pdfium segfaults (-11 on the full-set run).
                    bitmap.close()
                    page.close()
            finally:
                pdf.close()
            return out
    return [Image.open(path).convert("RGB")]


def fit(img: Image.Image, max_px: int = MAX_PAGE_PX) -> Image.Image:
    scale = max_px / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def crop(page: Image.Image, bbox1000: list[float], pad: float = 8) -> Image.Image:
    """Crop a 0-1000 normalized [x1,y1,x2,y2] box with `pad` (0-1000 units) of margin."""
    w, h = page.size
    x1, y1, x2, y2 = bbox1000
    box = (
        max(0, (x1 - pad) / 1000 * w),
        max(0, (y1 - pad) / 1000 * h),
        min(w, (x2 + pad) / 1000 * w),
        min(h, (y2 + pad) / 1000 * h),
    )
    return page.crop(tuple(round(v) for v in box))
