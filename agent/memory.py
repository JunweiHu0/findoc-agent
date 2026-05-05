"""Memory system (P28) — three-layer semantic fact memory.

Layer 1 (Working):   fact_index dict lookup per query — already in executor.
Layer 2 (Episodic):  conv_facts + 128d embedding, cosine similarity.
                     Hard hit (>0.85): skip retrieval, serve from cache.
                     Soft hit (0.5–0.85): inject as retrieval_priors into planner.
Layer 3 (Semantic):  global_facts table, cross-conversation.
                     Promotion: hit_count >= 3 AND grounding_verified = 1.

Embeddings reuse the ColQwen text encoder (128d float16) — zero new dependencies.
When the model is not loaded, falls back to char-ngram sparse vectors for
a decent approximate match.
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Text encoding (ColQwen text branch or fallback)
# ---------------------------------------------------------------------------

def _encode_text_lite(text: str) -> Optional[list[float]]:
    """Encode a short text snippet as a 128-dim float vector.

    Tries the ColQwen text encoder first. Falls back to char-ngram sparse
    vectors when the model is not loaded (cold start / no GPU).
    """
    try:
        from tools.colpali_tool import _state as _tool_state, _ensure_model_loaded
        _ensure_model_loaded()
        model = _tool_state.get("model")
        processor = _tool_state.get("processor")
        if model is not None and processor is not None:
            from ingestion.model_loader import encode_query as _local_encode
            import torch
            emb = _local_encode(model, processor, text)
            # Mean-pool to get a fixed 128-dim vector
            if emb.dim() == 2:
                emb = emb.mean(dim=0)
            emb = emb[:128]  # truncate or pad
            return emb.detach().cpu().float().tolist()
    except Exception:
        pass

    # Fallback: char 2-gram hash → 128-dim sparse vector
    return _ngram_hash(text, dim=128)


def _ngram_hash(text: str, dim: int = 128, n: int = 2) -> list[float]:
    """Build a sparse 128-dim vector from character n-grams."""
    vec = [0.0] * dim
    text = text.strip().lower()
    for i in range(len(text) - n + 1):
        gram = text[i:i + n]
        h = hash(gram) % dim
        vec[h] += 1.0
    # L2-normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        vec = [v / norm for v in vec]
    return vec


# ---------------------------------------------------------------------------
# Semantic match against known facts
# ---------------------------------------------------------------------------

HARD_THRESHOLD = 0.85   # cosine above this → cache hit, skip retrieval
SOFT_THRESHOLD = 0.50   # cosine above this → inject as retrieval prior


def semantic_match(
    sub_query: str,
    known_facts: list[dict],
    hard_threshold: float = HARD_THRESHOLD,
    soft_threshold: float = SOFT_THRESHOLD,
) -> dict:
    """Match sub_query against known_facts using vector similarity.

    Returns:
        {"hard_hits": [...], "soft_hits": [...], "best_score": float}
        hard_hits: facts that are a strong enough match to skip retrieval.
        soft_hits: facts that should be used as retrieval priors.
    """
    if not known_facts:
        return {"hard_hits": [], "soft_hits": [], "best_score": 0.0}

    q_emb = _encode_text_lite(sub_query)
    if q_emb is None:
        return {"hard_hits": [], "soft_hits": [], "best_score": 0.0}

    hard_hits: list[dict] = []
    soft_hits: list[dict] = []
    best_score = 0.0

    for kf in known_facts:
        # Build a composite text from structured fields
        parts = []
        for key in ("entity", "period", "metric", "text"):
            val = kf.get(key, "")
            if val:
                parts.append(str(val))
        fact_text = " ".join(parts)
        if not fact_text.strip():
            continue

        # Use cached embedding if available, otherwise encode
        emb = kf.get("_embedding")
        if emb is None:
            emb = _encode_text_lite(fact_text)
        if emb is None:
            continue

        score = cosine(q_emb, emb)
        if score > best_score:
            best_score = score

        if score >= hard_threshold:
            hard_hits.append({**kf, "_score": round(score, 4)})
        elif score >= soft_threshold:
            soft_hits.append({**kf, "_score": round(score, 4)})

    # Sort by score descending
    hard_hits.sort(key=lambda x: x["_score"], reverse=True)
    soft_hits.sort(key=lambda x: x["_score"], reverse=True)

    return {"hard_hits": hard_hits, "soft_hits": soft_hits, "best_score": round(best_score, 4)}


# ---------------------------------------------------------------------------
# Promotion to global facts (L3)
# ---------------------------------------------------------------------------

def maybe_promote_to_global(fact: dict, min_hits: int = 3) -> bool:
    """Check if a fact qualifies for promotion to global_facts (L3).

    Conditions: grounding_verified=1 AND hit_count >= min_hits.
    Returns True if promoted.
    """
    verified = fact.get("grounding_verified", 0)
    hits = fact.get("hit_count", 0)
    if verified and hits >= min_hits:
        try:
            from backend.storage import upsert_global_fact
            upsert_global_fact(fact)
            return True
        except Exception:
            pass
    return False
