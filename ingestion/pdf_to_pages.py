"""Convert PDFs under data/reports/ into per-page PNGs under data/pages/.

P1: CLI shell only. P2 implements the conversion using pdf2image + Pillow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent.config import PAGES_DIR, PDF_DIR, CONFIG


def pdf_to_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    raise NotImplementedError("P2: implement with pdf2image.convert_from_path")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=Path, default=PDF_DIR)
    parser.add_argument("--out_dir", type=Path, default=PAGES_DIR)
    parser.add_argument("--dpi", type=int, default=CONFIG["ingestion"]["dpi"])
    args = parser.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {args.pdf_dir}")
    for pdf in pdfs:
        doc_id = pdf.stem
        out = args.out_dir / doc_id
        out.mkdir(parents=True, exist_ok=True)
        print(f"[stub] would convert {pdf.name} -> {out}/p###.png @ {args.dpi}dpi")


if __name__ == "__main__":
    main()
