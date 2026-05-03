"""Shared ColQwen2 / ColPali model loading and encoding.

Used by build_index, colpali_tool, and the Litserve service to avoid
duplicated model-loading logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from agent.config import CONFIG


def resolve_path(maybe_relative: str) -> str:
    p = Path(maybe_relative)
    if p.is_absolute() or not (Path.cwd() / p).exists():
        return str(p)
    return str((Path.cwd() / p).resolve())


def load_model_and_processor(lora_path: Optional[str] = None):
    cfg = CONFIG["retriever"]
    backbone = cfg.get("backbone", "colqwen2")
    dtype = getattr(torch, cfg["dtype"])
    model_name = resolve_path(cfg["model_name"])

    if backbone == "colqwen2":
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        ModelCls, ProcessorCls = ColQwen2, ColQwen2Processor
    elif backbone == "colpali":
        from colpali_engine.models import ColPali, ColPaliProcessor
        ModelCls, ProcessorCls = ColPali, ColPaliProcessor
    else:
        raise ValueError(f"unknown retriever.backbone: {backbone}")

    logger.info(f"Loading {backbone} from {model_name} dtype={cfg['dtype']} device={cfg['device']}")
    model = ModelCls.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=cfg["device"],
    ).eval()

    effective_lora = lora_path or cfg.get("lora_path")
    if effective_lora and Path(effective_lora).exists():
        from peft import PeftModel
        logger.info(f"Loading domain LoRA adapter from {effective_lora}")
        model = PeftModel.from_pretrained(model, effective_lora)
        model = model.merge_and_unload()
    elif effective_lora:
        logger.warning(f"LoRA path {effective_lora} does not exist; using base retriever only")

    processor = ProcessorCls.from_pretrained(model_name)
    return model, processor


def encode_query(model, processor, query: str) -> torch.Tensor:
    batch = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        emb = model(**batch)
    return emb.to("cpu", dtype=torch.float16)[0]


def encode_pages(model, processor, image_paths: list[Path], batch_size: int = 1) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="encoding"):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch = processor.process_images(images).to(model.device)
        with torch.no_grad():
            emb = model(**batch)
        chunks.append(emb.to("cpu", dtype=torch.float16))
        for img in images:
            img.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    max_tokens = max(c.shape[1] for c in chunks)
    if any(c.shape[1] != max_tokens for c in chunks):
        padded = []
        for c in chunks:
            if c.shape[1] < max_tokens:
                pad = torch.zeros(c.shape[0], max_tokens - c.shape[1], c.shape[2], dtype=c.dtype)
                c = torch.cat([c, pad], dim=1)
            padded.append(c)
        chunks = padded
    return torch.cat(chunks, dim=0)
