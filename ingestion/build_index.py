"""Encode every page in data/pages/<doc_id>/ with the configured retriever and persist a multi-vector index.

Outputs per doc:
  data/index/<doc_id>/embeddings.pt    {"embeddings": fp16 [P, T, D], "page_nums": list[int]}
  data/index/<doc_id>/meta.json        {doc_id, model, page_count, pages, dtype}

Plus a global `data/index/doc_memory.json` mapping doc_id -> page_count + image dir.

Backbone is selected by `config.retriever.backbone`:
  - colqwen2  -> ColQwen2 / ColQwen2Processor (default; PEFT adapter is the model dir)
  - colpali   -> ColPali / ColPaliProcessor

A user-provided fine-tuned LoRA can be layered on top via `retriever.lora_path`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tqdm import tqdm

from agent.config import CONFIG, INDEX_DIR, PAGES_DIR
from ingestion.model_loader import encode_pages, load_model_and_processor


def _encode_pages_remote(image_paths: list[Path], colqwen_url: str, batch_size: int) -> "torch.Tensor":
    """Encode pages via ColQwen Service HTTP endpoint."""
    import httpx

    chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="encoding (remote)"):
        batch = [str(p) for p in image_paths[i : i + batch_size]]
        resp = httpx.post(
            f"{colqwen_url}/predict",
            json={"action": "encode_pages", "image_paths": batch},
            timeout=600.0,
        )
        resp.raise_for_status()
        data = resp.json()
        chunks.append(torch.tensor(data["embeddings"], dtype=torch.float16))

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


def index_doc(model, processor, doc_pages_dir: Path, doc_index_dir: Path, batch_size: int, colqwen_url: str = "") -> int:
    image_paths = sorted(doc_pages_dir.glob("p*.png"))
    if not image_paths:
        logger.warning(f"no pages in {doc_pages_dir}, skipping")
        return 0

    pt_path = doc_index_dir / "embeddings.pt"
    meta_path = doc_index_dir / "meta.json"
    if pt_path.exists() and meta_path.exists():
        logger.info(f"  {doc_pages_dir.name}: index exists, skipping")
        return 0

    doc_index_dir.mkdir(parents=True, exist_ok=True)
    if colqwen_url:
        embeddings = _encode_pages_remote(image_paths, colqwen_url, batch_size)
    else:
        embeddings = encode_pages(model, processor, image_paths, batch_size)
    page_nums = [int(p.stem.lstrip("p")) for p in image_paths]

    torch.save({"embeddings": embeddings, "page_nums": page_nums}, pt_path)
    meta = {
        "doc_id": doc_pages_dir.name,
        "backbone": CONFIG["retriever"].get("backbone", "colqwen2"),
        "model": CONFIG["retriever"]["model_name"],
        "lora_path": CONFIG["retriever"].get("lora_path"),
        "dtype_stored": "float16",
        "page_count": len(page_nums),
        "pages": [str(p) for p in image_paths],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"  {doc_pages_dir.name}: wrote {embeddings.shape}")
    return len(page_nums)


def build_doc_memory(index_dir: Path, pages_dir: Path) -> None:
    docs = []
    for sub in sorted(index_dir.iterdir()):
        meta = sub / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        docs.append({
            "doc_id": m["doc_id"],
            "page_count": m["page_count"],
            "pages_dir": str(pages_dir / m["doc_id"]),
        })
    out = index_dir / "doc_memory.json"
    out.write_text(json.dumps({"docs": docs}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"wrote {out} ({len(docs)} docs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages_dir", type=Path, default=PAGES_DIR)
    parser.add_argument("--index_dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--lora_path", type=str, default=CONFIG["retriever"].get("lora_path"))
    parser.add_argument("--batch_size", type=int, default=CONFIG["retriever"].get("encode_batch_size", 1))
    parser.add_argument("--only", type=str, default=None, help="substring filter on doc_id")
    parser.add_argument("--colqwen_url", type=str, default=CONFIG.get("services", {}).get("colqwen_url", ""),
                        help="remote ColQwen Service URL (if set, skip local model load)")
    args = parser.parse_args()

    doc_dirs = sorted([d for d in args.pages_dir.iterdir() if d.is_dir()])
    if args.only:
        doc_dirs = [d for d in doc_dirs if args.only in d.name]
    if not doc_dirs:
        logger.error(f"no doc page directories found under {args.pages_dir}")
        return

    if args.colqwen_url:
        logger.info(f"Using remote ColQwen Service at {args.colqwen_url}")
        model, processor = None, None
    else:
        model, processor = load_model_and_processor(args.lora_path)

    total = 0
    for doc_dir in doc_dirs:
        total += index_doc(model, processor, doc_dir, args.index_dir / doc_dir.name, args.batch_size,
                           colqwen_url=args.colqwen_url)
    build_doc_memory(args.index_dir, args.pages_dir)
    logger.info(f"Done. Indexed {total} new pages across {len(doc_dirs)} docs.")


if __name__ == "__main__":
    main()
