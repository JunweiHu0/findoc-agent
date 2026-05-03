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
from ingestion.model_loader import encode_query as _local_encode_query
from ingestion.model_loader import load_model_and_processor as _load_model


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


def _ensure_model_loaded() -> None:
    """Lazy-load local model when remote service is not configured."""
    if _state["model"] is not None:
        return
    model, processor = _load_model()
    _state["model"] = model
    _state["processor"] = processor


def _ensure_loaded() -> bool:
    if _state["indexes"] is not None:
        return bool(_state["indexes"])
    indexes = _load_indexes()
    if not indexes:
        _state["indexes"] = {}
        return False
    _state["indexes"] = indexes
    _ensure_model_loaded()
    return True


def _encode_query_remote(query: str) -> "torch.Tensor | None":
    """Encode query via ColQwen Service. Returns None on failure."""
    import httpx
    url = CONFIG.get("services", {}).get("colqwen_url", "")
    if not url:
        return None
    try:
        resp = httpx.post(
            f"{url}/predict",
            json={"action": "encode_query", "query": query},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return torch.tensor(data["embedding"], dtype=torch.float16)
    except Exception as e:
        logger.warning(f"remote encoding failed ({e}), falling back to local")
        return None


def _encode_query(query: str) -> torch.Tensor:
    emb = _encode_query_remote(query)
    if emb is not None:
        return emb
    _ensure_model_loaded()
    return _local_encode_query(_state["model"], _state["processor"], query)


def _maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """query_emb: [Tq, D]; doc_emb: [P, Td, D] -> scores: [P]."""
    q = query_emb.float()
    d = doc_emb.float()
    sim = torch.einsum("td,pkd->ptk", q, d)
    return sim.max(dim=-1).values.sum(dim=-1)


def _in_memory_retrieve(
    query: str,
    top_k: int,
    doc_filter: Optional[list[str]],
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


def _qdrant_retrieve(
    query: str,
    top_k: int,
    doc_filter: Optional[list[str]],
) -> list[PageHit]:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    qdrant_cfg = CONFIG["retriever"].get("qdrant", {})
    client = QdrantClient(url=qdrant_cfg.get("url", "http://localhost:6333"))
    collection = qdrant_cfg.get("collection_name", "findoc_pages")

    q_emb = _encode_query(query)
    query_vector = q_emb.numpy().tolist()

    query_filter = None
    if doc_filter:
        if len(doc_filter) == 1:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_filter[0]))]
            )
        else:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_filter))]
            )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    hits: list[PageHit] = []
    for scored in results.points:
        payload = scored.payload or {}
        hits.append(PageHit(
            doc_id=payload.get("doc_id", ""),
            page_num=int(payload.get("page_num", 0)),
            score=scored.score,
            image_path=payload.get("image_path"),
        ))
    return hits


def colpali_retrieve(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[list[str]] = None,
) -> list[PageHit]:
    backend = CONFIG["retriever"].get("backend", "in_memory")

    if backend == "qdrant":
        try:
            return _qdrant_retrieve(query, top_k, doc_filter)
        except Exception as e:
            logger.error(f"Qdrant retrieval failed ({e}), falling back to in_memory")

    return _in_memory_retrieve(query, top_k, doc_filter)


if __name__ == "__main__":
    for h in colpali_retrieve("2023 年毛利率"):
        print(h)
