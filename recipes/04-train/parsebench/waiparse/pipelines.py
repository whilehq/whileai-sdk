"""ParseBench pipelines for the waiparse agent and its baselines.

The server is our Modal vLLM deployment (serve/serve_vlm.py): the base model is
served as "qwen3.8-27b" (ParseBench's Qwen layout adapter keys on the "qwen3.8" prefix), a trained LoRA adapter as "qwen3.8-27b-tuned".
"""

from parse_bench.extensions import register_pipeline
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType

SERVER = {"server_url_env": "DOCPARSE_SERVER", "api_key_env": "DOCPARSE_KEY"}

# Baseline: the leaderboard's Qwen3.8-27B layout prompt, one call per page, on our server.
for name, thinking in (("wai_baseline", False), ("wai_baseline_thinking", True)):
    register_pipeline(
        PipelineSpec(
            pipeline_name=name,
            provider_name="qwen3_8",
            product_type=ProductType.PARSE,
            config={
                **SERVER,
                "model": "qwen3.8-27b",
                "prompt_mode": "layout",
                "enable_thinking": thinking,
            },
        )
    )

# The agent (waiparse/agent.py). Variants differ only in config.
from waiparse import agent  # noqa: F401  (registers provider + layout adapter)

AGENT_VARIANTS = {
    "wai_agent": {},
    "wai_agent_sc3": {"chart_samples": 3},
    "wai_agent_tuned": {"model": "qwen3.8-27b-tuned"},
    # layout_pages from PP-DocLayoutV3 boxes (serve/serve_layout.py), markdown unchanged.
    "wai_agent_det": {"layout_boxes": "detector"},
}
# Ablations of wai_agent_det's layout mapping (defaults in agent.DEFAULTS), scored on dev by
# re-normalizing wai_agent_det raws without new VLM calls: serve/relayout.py.
DET_ABLATIONS = {
    "qwen": {"layout_boxes": "qwen"},  # control: same raws, layout-pass boxes
    "noocr": {"det_ocr": False},
    "nosplit": {"det_split": False},
    "t30": {"det_threshold": 0.3},
    "t10": {"det_threshold": 0.1},  # needs raws detected at >= 0.1 (layout_det.RAW_THRESHOLD)
}
for suffix, cfg in DET_ABLATIONS.items():
    AGENT_VARIANTS[f"wai_agent_det_{suffix}"] = {**AGENT_VARIANTS["wai_agent_det"], **cfg}
for name, cfg in AGENT_VARIANTS.items():
    register_pipeline(
        PipelineSpec(
            pipeline_name=name,
            provider_name="wai_agent",
            product_type=ProductType.PARSE,
            config=cfg,
        )
    )
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_style",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={"style_pass": True},
    )
)
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_style_sc3",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={"style_pass": True, "chart_samples": 3},
    )
)
# Everything that won on dev so far: detector boxes + style transfer (+ chart self-consistency).
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_full",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={"layout_boxes": "detector", "style_pass": True},
    )
)
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_full_sc3",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={"layout_boxes": "detector", "style_pass": True, "chart_samples": 3},
    )
)
# v3 = full_sc3 + the detector chart trigger (code default since 2026-09-26; motivated by a test
# failure category, a report whose chart the layout pass read as bare label lines; tuned on dev only).
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_v3",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={"layout_boxes": "detector", "style_pass": True, "chart_samples": 3},
    )
)
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_v4",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={
            "layout_boxes": "detector",
            "style_pass": True,
            "chart_samples": 3,
            "table_samples": 3,
        },
    )
)
# v4 with the chart RL adapter on a second server app (serve_vlm.py with DOCPARSE_APP=docparse-vlm-tuned).
# Only the chart passes use the adapter; layout / tables / style stay on the base weights, as in v4.
# "tuned" resolves to waiparse.endpoints.tuned_server() when the provider starts, so the URL
# comes from your DOCPARSE_WORKSPACE (or DOCPARSE_TUNED_SERVER), not from this file.
TUNED_SERVER = "tuned"
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_v4_rl",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={
            "layout_boxes": "detector",
            "style_pass": True,
            "chart_samples": 3,
            "table_samples": 3,
            "chart_model": "qwen3.8-27b-tuned",
            "chart_server": TUNED_SERVER,
        },
    )
)
# Medium chart effort (what charts-pages-v1 trains with): base control and the RL adapter.
_V4 = {"layout_boxes": "detector", "style_pass": True, "chart_samples": 3, "table_samples": 3}
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_v4_med",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={**_V4, "chart_effort": "medium"},
    )
)
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_v4_med_rl",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={
            **_V4,
            "chart_effort": "medium",
            "chart_model": "qwen3.8-27b-tuned",
            "chart_server": TUNED_SERVER,
        },
    )
)
# v5 = v4 + medium chart effort (dev charts 86.7 at medium vs 83.5-85.2 at xhigh, and ~2x faster).
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_v5",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={**_V4, "chart_effort": "medium"},
    )
)
# Chart pass at a 1400 px page (RL throughput: 2048 px pages starved rollouts). Base control on dev.
register_pipeline(
    PipelineSpec(
        pipeline_name="wai_agent_v4_med_1400",
        provider_name="wai_agent",
        product_type=ProductType.PARSE,
        config={**_V4, "chart_effort": "medium", "chart_page_px": 1400},
    )
)
