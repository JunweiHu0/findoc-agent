"""User upload pipeline: PDF/image → pages → encode → index → Qdrant.

Pipeline stages:
  1. save     — write uploaded file to data/uploads/<doc_id>/
  2. pages    — PDF→PNG via pdf2image (images skip this)
  3. encode   — ColQwen2 multi-vector encoding
  4. index    — save .pt + meta.json to data/index/<doc_id>/
  5. qdrant   — push multi-vectors to Qdrant (if configured)
  6. register — update doc_memory.json + SQLite documents table

Progress is reported via a callback: callback(stage: str, message: str, pct: float).
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from pdf2image import convert_from_path
from PIL import Image

from agent.config import CONFIG, INDEX_DIR, PAGES_DIR, ROOT

UPLOAD_DIR = ROOT / "data" / "uploads"

# Doc ID derivation for uploaded files: sanitize filename → findoc_upload_<slug>
_ILLEGAL = re.compile(r"[^\w\-.]")


def derive_upload_doc_id(filename: str) -> str:
    """Derive a doc_id from an uploaded filename. Slug + random suffix for uniqueness."""
    stem = Path(filename).stem
    slug = _ILLEGAL.sub("_", stem).strip("_").lower() or "upload"
    suffix = uuid.uuid4().hex[:6]
    return f"{slug}_{suffix}"


def run_upload_pipeline(
    file_path: str,
    doc_id: str,
    progress_callback: Optional[callable] = None,
) -> int:
    """Run the full upload pipeline synchronously. Returns page count.

    Args:
        file_path: Path to the uploaded file (PDF or image).
        doc_id: Derived document identifier.
        progress_callback: Called as callback(stage, message, pct_complete).
    """
    file_path = Path(file_path)

    def _progress(stage: str, msg: str, pct: float):
        if progress_callback:
            try:
                progress_callback(stage, msg, pct)
            except Exception:
                pass

    source_filename = file_path.name
    is_image = file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"}

    # Register in SQLite upfront
    from backend.storage import add_document, update_document_status
    add_document(doc_id, source_filename, page_count=0, status="queued")

    # Stage 1: Save file
    _progress("save", f"保存文件 {source_filename}...", 0.05)
    doc_upload_dir = UPLOAD_DIR / doc_id
    doc_upload_dir.mkdir(parents=True, exist_ok=True)
    dest = doc_upload_dir / source_filename
    shutil.copy2(file_path, dest)

    # Stage 2: Convert to pages
    if is_image:
        _progress("pages", f"处理图片 {source_filename}...", 0.15)
        pages_out = _convert_image(dest, doc_id)
    else:
        _progress("pages", f"PDF 转图片 {source_filename}...", 0.15)
        pages_out = _convert_pdf(dest, doc_id)

    page_count = len(pages_out)
    if page_count == 0:
        update_document_status(doc_id, "failed")
        raise RuntimeError(f"No pages generated from {source_filename}")

    update_document_status(doc_id, "encoding", page_count=page_count)

    # Stage 3: Encode with ColQwen2
    _progress("encode", f"ColQwen2 编码 {page_count} 页...", 0.30)
    colqwen_url = CONFIG.get("services", {}).get("colqwen_url", "")
    if colqwen_url:
        embeddings = _encode_via_service(pages_out, colqwen_url)
    else:
        embeddings = _encode_via_local(pages_out)

    # Stage 4: Save index
    _progress("index", "保存索引...", 0.75)
    doc_index_dir = INDEX_DIR / doc_id
    doc_index_dir.mkdir(parents=True, exist_ok=True)
    page_nums = [int(p.stem.lstrip("p")) for p in pages_out]
    torch.save(
        {"embeddings": embeddings, "page_nums": page_nums},
        doc_index_dir / "embeddings.pt",
    )
    meta = {
        "doc_id": doc_id,
        "backbone": CONFIG["retriever"].get("backbone", "colqwen2"),
        "model": CONFIG["retriever"]["model_name"],
        "dtype_stored": "float16",
        "page_count": page_count,
        "pages": [str(p) for p in pages_out],
    }
    (doc_index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stage 5: Push to Qdrant (if configured)
    backend = CONFIG["retriever"].get("backend", "in_memory")
    if backend == "qdrant":
        _progress("qdrant", f"推送 {page_count} 页到 Qdrant...", 0.85)
        try:
            _push_to_qdrant(doc_id, embeddings, page_nums, pages_out)
        except Exception as e:
            logger.warning(f"Qdrant push failed for {doc_id}: {e}")

    # Stage 6: Update doc_memory.json
    _progress("register", "更新文档索引...", 0.95)
    _update_doc_memory(doc_id, page_count)

    # Reload in-memory indexes if needed
    if backend == "in_memory":
        try:
            from tools.colpali_tool import _state, _load_indexes
            if _state.get("indexes") is not None:
                _state["indexes"] = _load_indexes()
                logger.info(f"Reloaded in-memory indexes after upload {doc_id}")
        except Exception:
            pass

    update_document_status(doc_id, "ready", page_count=page_count)
    _progress("done", f"完成 — {doc_id} · {page_count} 页", 1.0)
    return page_count


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _convert_pdf(pdf_path: Path, doc_id: str) -> list[Path]:
    """Convert PDF to per-page PNGs under data/pages/<doc_id>/."""
    out_dir = PAGES_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = CONFIG["ingestion"]["dpi"]
    fmt = CONFIG["ingestion"]["image_format"]

    images = convert_from_path(str(pdf_path), dpi=dpi)
    written: list[Path] = []
    for i, img in enumerate(images, start=1):
        out_path = out_dir / f"p{i:03d}.{fmt}"
        img.save(out_path, fmt.upper())
        written.append(out_path)
    return written


def _convert_image(img_path: Path, doc_id: str) -> list[Path]:
    """Treat a single image as page 1 of a document."""
    out_dir = PAGES_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = CONFIG["ingestion"]["image_format"]
    out_path = out_dir / f"p001.{fmt}"
    img = Image.open(img_path).convert("RGB")
    img.save(out_path, fmt.upper())
    img.close()
    return [out_path]


def _encode_via_service(image_paths: list[Path], colqwen_url: str) -> torch.Tensor:
    """Encode pages via remote ColQwen Service."""
    import httpx
    resp = httpx.post(
        f"{colqwen_url}/predict",
        json={"action": "encode_pages", "image_paths": [str(p) for p in image_paths]},
        timeout=600.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return torch.tensor(data["embeddings"], dtype=torch.float16)


def _encode_via_local(image_paths: list[Path]) -> torch.Tensor:
    """Encode pages using the local ColQwen2 model.

    Reuses the singleton already loaded by `tools.colpali_tool` (eagerly
    preloaded by `backend.server` at startup). Loading a second copy here
    would OOM the 6GB GPU and waste ~60s reloading weights.
    """
    from ingestion.model_loader import encode_pages
    from tools.colpali_tool import _ensure_model_loaded, _state
    _ensure_model_loaded()
    return encode_pages(_state["model"], _state["processor"], image_paths)


def _push_to_qdrant(doc_id: str, embeddings: torch.Tensor,
                    page_nums: list[int], pages: list[Path]) -> None:
    """Push multi-vector embeddings to Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    qdrant_cfg = CONFIG["retriever"].get("qdrant", {})
    client = QdrantClient(url=qdrant_cfg.get("url", "http://localhost:6333"))
    collection = qdrant_cfg.get("collection_name", "findoc_pages")

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
                "image_path": str(pages[i]) if i < len(pages) else "",
            },
        ))

    client.upsert(collection_name=collection, points=points, wait=True)
    logger.info(f"Pushed {len(points)} points to Qdrant for {doc_id}")


def _update_doc_memory(doc_id: str, page_count: int) -> None:
    """Add/update doc entry in doc_memory.json."""
    from ingestion.build_index import build_doc_memory
    build_doc_memory(INDEX_DIR, PAGES_DIR)
