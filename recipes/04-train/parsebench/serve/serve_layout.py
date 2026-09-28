# Layout detector for the doc-parse agent on Modal, in your own workspace.
#
# PP-DocLayoutV3 (PaddlePaddle, Apache-2.0; the layout stage of PaddleOCR-VL-1.5/1.6) via the
# Hugging Face transformers port. RT-DETR detector with a mask head and a reading-order head,
# 25 layout classes.
#
#   modal deploy serve/serve_layout.py
#   modal run serve/serve_layout.py::probe
#
# POST https://<your workspace>--docparse-layout-detector-web.modal.run/detect
#   Authorization: Bearer $VLLM_API_KEY (secret docparse-vllm-key)
#   body: raw PNG/JPEG bytes (query ?threshold=0.3)
#   -> {"width", "height", "boxes": [{"bbox": [x1,y1,x2,y2] px, "label", "score", "order"}]}

import io
import os

import modal

APP_NAME = "docparse-layout"
MODEL_ID = os.environ.get("DOCPARSE_LAYOUT_MODEL", "PaddlePaddle/PP-DocLayoutV3_safetensors")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers>=5.1",
        "pillow",
        "fastapi[standard]",
        "huggingface_hub",
        "timm",
        "opencv-python-headless",
    )
    .env({"DOCPARSE_LAYOUT_MODEL": MODEL_ID})
)
hf_cache = modal.Volume.from_name("docparse-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME)


def _load():
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to("cuda").eval()
    return proc, model, torch


def _detect(proc, model, torch, img, threshold: float) -> list[dict]:
    inputs = proc(images=[img], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_object_detection(
        out, threshold=threshold, target_sizes=[(img.height, img.width)]
    )[0]
    id2label = model.config.id2label
    boxes = []
    order = res.get("order_seq", res.get("reading_order"))
    for i, (score, label, box) in enumerate(
        zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist())
    ):
        rec = {
            "bbox": [round(v, 1) for v in box],
            "label": id2label[int(label)],
            "score": round(score, 4),
        }
        if order is not None:
            try:  # noqa: SIM105  # a missing order is not an error
                rec["order"] = int(order[i])
            except Exception:
                pass
        boxes.append(rec)
    return boxes


@app.cls(
    image=image,
    gpu="L4",
    scaledown_window=10 * 60,
    timeout=30 * 60,
    max_containers=4,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("docparse-vllm-key"),
    ],
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=32)
class Detector:
    @modal.enter()
    def start(self):
        import threading

        self.proc, self.model, self.torch = _load()
        self.lock = threading.Lock()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, HTTPException, Request
        from PIL import Image

        api = FastAPI()
        key = os.environ["VLLM_API_KEY"]

        @api.post("/detect")
        async def detect(request: Request, threshold: float = 0.3):
            if request.headers.get("authorization", "") != f"Bearer {key}":
                raise HTTPException(status_code=401, detail="bad key")
            img = Image.open(io.BytesIO(await request.body())).convert("RGB")
            import asyncio

            def run():
                with self.lock:
                    return _detect(self.proc, self.model, self.torch, img, threshold)

            boxes = await asyncio.to_thread(run)
            return {"width": img.width, "height": img.height, "boxes": boxes}

        @api.get("/health")
        async def health():
            return {"ok": True, "model": MODEL_ID}

        return api


@app.function(
    image=image.pip_install("pypdfium2"),
    gpu="L4",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data": modal.Volume.from_name("docparse-data"),
    },
)
def probe(n: int = 2):
    """Print labels + detections for the first pages of a couple of dev PDFs."""
    from pathlib import Path

    import pypdfium2 as pdfium

    proc, model, torch = _load()
    print("id2label", model.config.id2label)
    pdfs = sorted(Path("/data/split_dev").rglob("*.pdf"))[:n]
    for p in pdfs:
        img = pdfium.PdfDocument(str(p))[0].render(scale=200 / 72).to_pil().convert("RGB")
        inputs = proc(images=[img], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model(**inputs)
        print("output keys", list(out.keys()))
        res = proc.post_process_object_detection(
            out, threshold=0.3, target_sizes=[(img.height, img.width)]
        )[0]
        print(
            p.name,
            img.size,
            {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in res.items()},
        )
        for b in _detect(proc, model, torch, img, 0.3):
            print("  ", b)
