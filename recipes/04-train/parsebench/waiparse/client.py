"""Async OpenAI-compatible VLM client for the vLLM server (serve/serve_vlm.py)."""

import asyncio
import base64
import io
import os
from typing import Any

import aiohttp
from PIL import Image

from waiparse import endpoints

# ParseBench runs each document in its own thread + event loop, so the cap on
# in-flight requests is per loop (per document); the runner's --max_concurrent
# bounds the number of documents.
_SEMS: dict[int, asyncio.Semaphore] = {}


def _sem() -> asyncio.Semaphore:
    loop = id(asyncio.get_running_loop())
    if loop not in _SEMS:
        _SEMS[loop] = asyncio.Semaphore(int(os.environ.get("DOCPARSE_DOC_CONCURRENCY", "16")))
    return _SEMS[loop]


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class VLM:
    def __init__(self, model: str = "qwen3.8-27b", timeout: int = 900, server: str | None = None):
        self.url = (server or endpoints.vlm_server()).rstrip("/") + "/v1/chat/completions"
        self.key = os.environ.get("DOCPARSE_KEY", "")
        self.model = model
        self.timeout = timeout

    async def ask(
        self,
        session: aiohttp.ClientSession,
        image: Image.Image,
        prompt: str,
        *,
        thinking: bool = False,
        effort: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        n: int = 1,
        retries: int = 3,
    ) -> list[str]:
        """Return `n` completions for one image + prompt (thinking text stripped)."""
        kwargs: dict[str, Any] = {"enable_thinking": thinking}
        if thinking and effort:
            kwargs["reasoning_effort"] = effort
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png_b64(image)}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "chat_template_kwargs": kwargs,
        }
        if thinking:
            # Qwen3.8 model card, thinking mode sampling.
            payload.update(temperature=max(temperature, 0.6), top_p=0.95, top_k=20)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        last: Exception | None = None
        for attempt in range(retries):
            try:
                async with (
                    _sem(),
                    session.post(
                        self.url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as r,
                ):
                    if r.status != 200:
                        raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
                    body = await r.json()
                return [_strip_think(c["message"].get("content") or "") for c in body["choices"]]
            except Exception as e:  # transient: 5xx, timeouts, cold starts
                last = e
                await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"VLM call failed after {retries} tries: {last}")


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
