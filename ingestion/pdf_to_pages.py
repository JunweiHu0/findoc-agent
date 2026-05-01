"""Convert PDFs under data/reports/ into per-page PNGs under data/pages/<doc_id>/p###.png.

Doc id is derived from filename via:
  - the longest matching company alias from config.ingestion.doc_id_aliases,
  - the first 4-digit year (2000-2099) found in the filename.

Idempotent: skips files whose target page already exists.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

from loguru import logger
from pdf2image import convert_from_path

from agent.config import CONFIG, PAGES_DIR, PDF_DIR


_YEAR_RE = re.compile(r"(20\d{2})")


def derive_doc_id(filename: str, aliases: dict[str, str]) -> Optional[str]:
    stem = Path(filename).stem
    company = None
    matched_key_len = 0
    for cn, ascii_alias in aliases.items():
        if cn in stem and len(cn) > matched_key_len:
            company = ascii_alias
            matched_key_len = len(cn)
    if company is None:
        return None
    year_match = _YEAR_RE.search(stem)
    if not year_match:
        return None
    return f"{company}_{year_match.group(1)}"


def _convert_one(pdf_path: Path, out_dir: Path, dpi: int, image_format: str, max_pages: Optional[int]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    last_page = max_pages if max_pages else None
    images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=last_page)
    written = 0
    for i, img in enumerate(images, start=1):
        out_path = out_dir / f"p{i:03d}.{image_format}"
        if out_path.exists():
            continue
        img.save(out_path, image_format.upper())
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=Path, default=PDF_DIR)
    parser.add_argument("--out_dir", type=Path, default=PAGES_DIR)
    parser.add_argument("--dpi", type=int, default=CONFIG["ingestion"]["dpi"])
    parser.add_argument("--image_format", default=CONFIG["ingestion"]["image_format"])
    parser.add_argument("--max_pages", type=int, default=None, help="cap pages per PDF (smoke testing)")
    parser.add_argument("--only", type=str, default=None, help="substring filter on filename")
    args = parser.parse_args()

    aliases: dict[str, str] = CONFIG["ingestion"]["doc_id_aliases"]
    pdfs = sorted([p for p in args.pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if args.only:
        pdfs = [p for p in pdfs if args.only in p.name]
    logger.info(f"Found {len(pdfs)} PDFs to convert")

    skipped: list[str] = []
    total_written = 0
    for pdf in pdfs:
        doc_id = derive_doc_id(pdf.name, aliases)
        if doc_id is None:
            logger.warning(f"skip: cannot derive doc_id from '{pdf.name}'")
            skipped.append(pdf.name)
            continue
        out = args.out_dir / doc_id
        logger.info(f"{pdf.name} -> {out}")
        n = _convert_one(pdf, out, args.dpi, args.image_format, args.max_pages)
        logger.info(f"  wrote {n} new pages")
        total_written += n

    logger.info(f"Done. total new pages={total_written}, skipped_files={len(skipped)}")
    if skipped:
        for s in skipped:
            logger.info(f"  skipped: {s}")


if __name__ == "__main__":
    main()
