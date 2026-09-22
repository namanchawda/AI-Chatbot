"""Document loading utilities for PDF, HTML, and plain-text uploads.

Extracts raw text only — no structural parsing or section detection —
so downstream chunking and retrieval operate on plain text.
"""

from __future__ import annotations

import time
from pathlib import Path

from bs4 import BeautifulSoup
import pymupdf as fitz


def _read_text_with_fallback(path: Path) -> str:
    """Read a file as text using a practical fallback sequence for SEC HTML files."""
    raw = path.read_bytes()
    # BOM-aware UTF-8 first, then plain UTF-8, then Windows/legacy encodings
    # common in SEC filings; latin-1 never fails on any byte sequence.
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def load_filing(filepath: str) -> str:
    """Load a document from a local PDF, HTML, or plain-text file and return extracted text.

    No structural parsing or section detection is performed — only raw text
    extraction so downstream chunking and retrieval can operate on plain text.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Filing not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # Cheap magic-byte validity check before handing the file to fitz.
        if not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError("File does not appear to be a valid PDF (invalid header)")

        extraction_started = time.perf_counter()
        extracted_text = ""
        try:
            document = fitz.open(filepath)
            pages: list[str] = []
            for page in document:
                pages.append(page.get_text())
            extracted_text = "\n".join(pages)
            return extracted_text
        finally:
            # open() may fail before assignment — only close the handle if it exists.
            if "document" in locals():
                document.close()
            extraction_seconds = time.perf_counter() - extraction_started
            print(
                f"PDF text extraction for {path.name}: {extraction_seconds:.3f} seconds "
                f"({len(extracted_text):,} characters)"
            )

    if suffix in {".txt", ".md", ".rtf"}:
        return _read_text_with_fallback(path)

    if suffix in {".html", ".htm"}:
        html = _read_text_with_fallback(path)
        soup = BeautifulSoup(html, "html.parser")

        # Drop iXBRL/XBRL inline tags and non-content script/style nodes so
        # only human-readable filing text remains.
        for tag in list(soup.find_all(True)):
            tag_name = (tag.name or "").lower()
            if tag_name.startswith("ix:") or tag_name.startswith("xbrli:"):
                tag.decompose()

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        cleaned = soup.get_text(separator="\n")
        return "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())

    raise ValueError(
        f"Unsupported filing format for {filepath}. Supported formats: .txt, .md, .rtf, .pdf, .html, .htm"
    )


if __name__ == "__main__":
    sample_text = """This is a sample SEC filing excerpt.
    Management's discussion and analysis highlights revenue growth and operating margin.
    The company continues to invest in research and development and expand its product portfolio.
    Risk factors include competition, macroeconomic volatility, and supply chain disruptions.
    """

    sample_path = Path("sample_filing.txt")
    sample_path.write_text(sample_text, encoding="utf-8")

    try:
        extracted = load_filing(str(sample_path))
        print("Loaded filing text preview:")
        print(extracted[:500])
    finally:
        if sample_path.exists():
            sample_path.unlink()
