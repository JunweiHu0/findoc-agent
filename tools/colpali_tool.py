"""Multi-vector page retrieval tool (ColQwen2 / ColPali).

Lazy-loads the retriever model + per-doc indexes built by `ingestion/build_index.py`.
Each retrieval call:
  1. encodes the query (one forward pass on text),
  2. computes MaxSim against every indexed doc (filtered if doc_filter given),
  3. returns the global top-k pages.

Backbone is chosen by `config.retriever.backbone` (default colqwen2). The file
name is kept as `colpali_tool.py` purely so the agent's import path stays stable;
the technique (multi-vector + late interaction) is identical.

Falls back to a deterministic mock when no index files exist under data/index/,
so the agent skeleton remains runnable before P2 has been executed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from agent.config import CONFIG, INDEX_DIR
from agent.state import PageHit


_MOCK_HITS = [
    PageHit(doc_id="moutai_2023", page_num=1, score=0.91, image_path="data/pages/moutai_2023/p001.png"),
    PageHit(doc_id="moutai_2023", page_num=2, score=0.84, image_path="data/pages/moutai_2023/p002.png"),
    PageHit(doc_id="moutai_2023", page_num=3, score=0.79, image_path="data/pages/moutai_2023/p003.png"),
]


_state: dict = {"model": None, "processor": None, "indexes": None}


def _index_files() -> list[Path]:
    if not INDEX_DIR.exists():
        return []
    return sorted(INDEX_DIR.glob("*/embeddings.pt"))


def _load_indexes() -> dict[str, dict]:
    indexes: dict[str, dict] = {}
    for emb_path in _index_files():
        doc_dir = emb_path.parent
        meta = json.loads((doc_dir / "meta.json").read_text(encoding="utf-8"))
        payload = torch.load(emb_path, map_location="cpu", weights_only=True)
        indexes[meta["doc_id"]] = {
            "embeddings": payload["embeddings"],
            "page_nums": payload["page_nums"],
            "pages": meta["pages"],
        }
    logger.info(f"Loaded retriever indexes for {len(indexes)} docs")
    return indexes


def _resolve_path(maybe_relative: str) -> str:
    p = Path(maybe_relative)
    if p.is_absolute() or not (Path.cwd() / p).exists():
        return str(p)
    return str((Path.cwd() / p).resolve())


def _load_model_and_processor():
    cfg = CONFIG["retriever"]
    backbone = cfg.get("backbone", "colqwen2")
    dtype = getattr(torch, cfg["dtype"])
    model_name = _resolve_path(cfg["model_name"])

    if backbone == "colqwen2":
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        ModelCls, ProcessorCls = ColQwen2, ColQwen2Processor
    elif backbone == "colpali":
        from colpali_engine.models import ColPali, ColPaliProcessor
        ModelCls, ProcessorCls = ColPali, ColPaliProcessor
    else:
        raise ValueError(f"unknown retriever.backbone: {backbone}")

    logger.info(f"Loading {backbone} retriever from {model_name} for query encoding")
    model = ModelCls.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=cfg["device"],
    ).eval()
    lora_path = cfg.get("lora_path")
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
    processor = ProcessorCls.from_pretrained(model_name)
    return model, processor


def _ensure_loaded() -> bool:
    if _state["indexes"] is not None:
        return bool(_state["indexes"])
    indexes = _load_indexes()
    if not indexes:
        _state["indexes"] = {}
        return False
    model, processor = _load_model_and_processor()
    _state["model"] = model
    _state["processor"] = processor
    _state["indexes"] = indexes
    return True


def _encode_query(query: str) -> torch.Tensor:
    processor = _state["processor"]
    model = _state["model"]
    batch = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        emb = model(**batch)
    return emb.to("cpu", dtype=torch.float16)[0]


def _maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """query_emb: [Tq, D]; doc_emb: [P, Td, D] -> scores: [P]."""
    q = query_emb.float()
    d = doc_emb.float()
    sim = torch.einsum("td,pkd->ptk", q, d)
    return sim.max(dim=-1).values.sum(dim=-1)


def colpali_retrieve(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[list[str]] = None,
) -> list[PageHit]:
    if not _ensure_loaded():
        logger.warning(f"no retriever index found under {INDEX_DIR} — returning mock hits")
        hits = _MOCK_HITS
        if doc_filter:
            hits = [h for h in hits if h.doc_id in doc_filter]
        return hits[:top_k]

    indexes: dict[str, dict] = _state["indexes"]
    if doc_filter:
        indexes = {k: v for k, v in indexes.items() if k in doc_filter}
    if not indexes:
        return []

    q_emb = _encode_query(query)

    all_hits: list[PageHit] = []
    for doc_id, idx in indexes.items():
        scores = _maxsim(q_emb, idx["embeddings"])
        for i, score in enumerate(scores.tolist()):
            page_num = int(idx["page_nums"][i])
            page_path = next(
                (p for p in idx["pages"] if Path(p).stem == f"p{page_num:03d}"),
                None,
            )
            all_hits.append(PageHit(
                doc_id=doc_id,
                page_num=page_num,
                score=float(score),
                image_path=page_path,
            ))

    all_hits.sort(key=lambda h: h.score, reverse=True)
    return all_hits[:top_k]


if __name__ == "__main__":
    for h in colpali_retrieve("2023 年毛利率"):
        print(h)
