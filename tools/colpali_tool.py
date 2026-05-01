"""ColPali retrieval tool.

P1: returns deterministic mock hits so the agent skeleton runs end-to-end
without GPU / weights. P2 will replace `_load_model` and `_load_index` with
real ColPali + LoRA loading and an in-memory MaxSim search.
"""
from __future__ import annotations

from typing import Optional

from agent.state import PageHit


_MOCK_HITS = [
    PageHit(doc_id="moutai_2023", page_num=42, score=0.91, image_path="data/pages/moutai_2023/p042.png"),
    PageHit(doc_id="moutai_2023", page_num=43, score=0.84, image_path="data/pages/moutai_2023/p043.png"),
    PageHit(doc_id="catl_2023", page_num=58, score=0.79, image_path="data/pages/catl_2023/p058.png"),
]


def colpali_retrieve(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[list[str]] = None,
) -> list[PageHit]:
    hits = _MOCK_HITS
    if doc_filter:
        hits = [h for h in hits if h.doc_id in doc_filter]
    return hits[:top_k]


def _load_model():
    raise NotImplementedError("ColPali model loading is implemented in P2")


def _load_index(index_dir):
    raise NotImplementedError("Index loading is implemented in P2")


if __name__ == "__main__":
    for h in colpali_retrieve("毛利率"):
        print(h)
