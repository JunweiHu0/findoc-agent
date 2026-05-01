"""Build per-document ColPali multi-vector indexes under data/index/.

For each doc_id we will write:
  data/index/<doc_id>/embeddings.pt    # tensor [num_pages, num_patches, dim]
  data/index/<doc_id>/meta.json        # page_num list, image paths, model info
And one global doc_memory.json with company/year/section metadata.

P1: CLI shell only. P2 wires real ColPali encoding (LoRA-loadable).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import CONFIG, INDEX_DIR, PDF_DIR


def build_index(pdf_dir: Path, index_dir: Path, lora_path: str | None) -> None:
    raise NotImplementedError("P2: encode pages with ColPali and persist .pt + meta.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=Path, default=PDF_DIR)
    parser.add_argument("--index_dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--lora_path", type=str, default=CONFIG["colpali"].get("lora_path"))
    args = parser.parse_args()

    print(f"[stub] would build ColPali index")
    print(f"  pdf_dir   = {args.pdf_dir}")
    print(f"  index_dir = {args.index_dir}")
    print(f"  base      = {CONFIG['colpali']['model_name']}")
    print(f"  lora_path = {args.lora_path}")


if __name__ == "__main__":
    main()
