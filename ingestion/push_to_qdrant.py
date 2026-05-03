"""Push existing .pt embedding indexes to Qdrant for server-side MaxSim retrieval.

Usage:
    python -m ingestion.push_to_qdrant              # push all docs
    python -m ingestion.push_to_qdrant --recreate   # drop + rebuild collection first
    python -m ingestion.push_to_qdrant --only moutai_2023
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from agent.config import CONFIG, INDEX_DIR


def get_or_create_collection(client: QdrantClient, collection_name: str, recreate: bool = False) -> None:
    if recreate:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Dropped existing collection '{collection_name}'")
        except Exception:
            pass

    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        logger.info(f"Collection '{collection_name}' already exists, reusing")
        return

    logger.info(f"Creating collection '{collection_name}' with multi-vector MaxSim config")
    # DOT distance: ColPali/ColQwen2 late interaction uses unnormalized dot products.
    # MaxSim comparator: for each query token, find the max dot product against
    # all document tokens, then sum across query tokens.
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=128, distance=Distance.DOT),
        multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
    )


def push_doc(client: QdrantClient, collection: str, doc_dir: Path) -> int:
    pt_path = doc_dir / "embeddings.pt"
    meta_path = doc_dir / "meta.json"
    if not pt_path.exists() or not meta_path.exists():
        logger.warning(f"Missing index files in {doc_dir}, skipping")
        return 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    embeddings = payload["embeddings"]  # [P, T, 128]
    page_nums = payload["page_nums"]
    doc_id = meta["doc_id"]
    pages = meta.get("pages", [])

    points: list[PointStruct] = []
    for i, page_num in enumerate(page_nums):
        vector = embeddings[i].tolist()
        point_id = abs(hash(f"{doc_id}__p{page_num}")) & 0x7FFFFFFFFFFFFFFF
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "doc_id": doc_id,
                "page_num": int(page_num),
                "image_path": pages[i] if i < len(pages) else "",
            },
        ))

    client.upsert(collection_name=collection, points=points, wait=True)
    logger.info(f"  {doc_id}: upserted {len(points)} pages")
    return len(points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="drop and recreate collection before pushing")
    parser.add_argument("--only", type=str, default=None, help="substring filter on doc_id")
    args = parser.parse_args()

    qdrant_cfg = CONFIG["retriever"].get("qdrant", {})
    qdrant_url = qdrant_cfg.get("url", "http://localhost:6333")
    collection = qdrant_cfg.get("collection_name", "findoc_pages")

    client = QdrantClient(url=qdrant_url)

    get_or_create_collection(client, collection, recreate=args.recreate)

    doc_dirs = sorted([d for d in INDEX_DIR.iterdir() if d.is_dir()])
    if args.only:
        doc_dirs = [d for d in doc_dirs if args.only in d.name]

    if not doc_dirs:
        logger.error(f"No index directories found under {INDEX_DIR}")
        return

    total = 0
    for doc_dir in doc_dirs:
        total += push_doc(client, collection, doc_dir)

    logger.info(f"Done. Pushed {total} pages across {len(doc_dirs)} docs to {qdrant_url} / {collection}")


if __name__ == "__main__":
    main()
