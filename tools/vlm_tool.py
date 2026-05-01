"""Page-level VLM extraction tool.

Sends a page image plus an instruction to an OpenAI-compatible vision endpoint.
Default config points at Aliyun Dashscope's compatible-mode endpoint
(`QWEN_BASE_URL`) using `QWEN_API_KEY`. Any other compatible endpoint can be
swapped in by editing config.yaml's vlm.{model, base_url_env, api_key_env}.

Falls back to a deterministic mock when:
- the image file is missing,
- the configured api key env var is empty.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from agent.config import CONFIG


_VLM_CFG = CONFIG["vlm"]


def vlm_read_page(image_path: str, instruction: str) -> str:
    if not image_path or not Path(image_path).exists():
        return f"[mock vlm] (no image) instruction='{instruction}'"

    api_key = _resolve_api_key()
    if not api_key:
        logger.warning(
            f"VLM api key env '{_VLM_CFG['api_key_env']}' is empty — returning mock extraction"
        )
        return f"[mock vlm] {Path(image_path).name} :: {instruction}"

    try:
        return _call_openai_compat(image_path, instruction, api_key)
    except Exception as e:
        logger.warning(f"VLM call failed ({e}); returning mock extraction")
        return f"[mock vlm:error] {Path(image_path).name} :: {instruction}"


def _resolve_api_key() -> str:
    return os.getenv(_VLM_CFG["api_key_env"], "")


def _resolve_base_url() -> Optional[str]:
    return os.getenv(_VLM_CFG.get("base_url_env", ""), "") or None


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _call_openai_compat(image_path: str, instruction: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=_resolve_base_url())
    b64 = _encode_image(image_path)
    suffix = Path(image_path).suffix.lstrip(".").lower() or "png"

    resp = client.chat.completions.create(
        model=_VLM_CFG["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You read a single page from a financial document and answer the user's "
                    "instruction strictly from what is visible on the page. "
                    "If a table is present, preserve numeric precision. "
                    "If the requested information is not on the page, say 'not on this page'."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64,{b64}"}},
                ],
            },
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    print(vlm_read_page("data/pages/moutai_2023/p001.png", "概述这页的主要内容"))
