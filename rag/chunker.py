"""
rag/chunker.py
---------------
Extracts text from EU emergency preparedness PDFs and splits into
overlapping chunks suitable for embedding and retrieval.

Strategy:
    - Direct text extraction via PyMuPDF for text-based PDFs
    - OCR fallback via pytesseract for image-based PDFs (3 of 4 docs)
    - Clean: remove navigation artifacts, page numbers, URLs, short lines
    - Chunk: ~200 tokens / 30-token overlap (calibrated for ~2800-word corpus)

Chunk size rationale:
    Total corpus ≈ 2,800 words across 4 docs (≈ 3,700 tokens).
    200-token chunks → ~18-22 chunks → meaningful retrieval granularity
    without over-fragmenting short paragraphs.
"""

import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import fitz  # pymupdf


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:  int
    text:      str
    source:    str    # filename without extension
    page:      int    # 1-indexed
    tokens:    int    # approximate word count


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Friendly display names for each source document
SOURCE_LABELS = {
    "emergency-supplies-cz":          "Czech Republic Emergency Guide (72h.cz)",
    "emergency-supplies-se":          "Sweden Emergency Guide (krisinformation.se)",
    "home-emergency-kit-be":          "Belgium Emergency Kit Guide (crisiscenter.be)",
    "putting-together-an-emergency-kit": "Netherlands Emergency Kit Guide (denkvooruit.nl)",
}

CHUNK_SIZE    = 200   # target tokens (words) per chunk
CHUNK_OVERLAP = 30    # overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_page_text(page: fitz.Page) -> str:
    """
    Extract text from a single PDF page.
    Falls back to OCR if direct extraction yields nothing.
    """
    text = page.get_text().strip()
    if text:
        return text

    # OCR fallback for image-based pages
    try:
        import pytesseract
        from PIL import Image

        mat = fitz.Matrix(3, 3)   # 3x zoom ≈ 216 DPI — good OCR quality
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="eng")
    except ImportError:
        text = ""

    return text.strip()


def _clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    skip_patterns = [
        r"^(why 72|emergency preparedness|important contacts|download|back|home)\s*$",
        r"^\d{4}\s*©",
        r"^version\s+\d",
        r"^design system",
        r"https?://\S+",
        r"^www\.\S+",
        r"^\s*[☐□✓✗°«»•]\s*$",
        r"^(v\s*)?\d+\.\d+",
    ]
    skip_re = [re.compile(p, re.IGNORECASE) for p in skip_patterns]

    for line in lines:
        line = line.strip()
        if len(line) < 15 and not re.match(r"^\d{2,3}\s+\w", line):
            continue
        if any(r.search(line) for r in skip_re):
            continue

        # Remove OCR bullet artifacts at line start
        line = re.sub(r"^[°«»•·▪▸\-–]\s*", "", line)
        # Remove inline checkbox and special chars
        line = re.sub(r"[☐□✓✗°«»]", "", line)
        # Fix common OCR spacing errors
        line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)  # camelCase splits
        line = re.sub(r"\s{2,}", " ", line).strip()

        if len(line) > 10:
            cleaned.append(line)

    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source: str, page: int, start_id: int) -> List[Chunk]:
    """
    Split cleaned text into overlapping word-based chunks.
    Each chunk targets CHUNK_SIZE words with CHUNK_OVERLAP word overlap.
    """
    words = text.split()
    chunks = []
    i = 0
    chunk_id = start_id

    while i < len(words):
        window = words[i: i + CHUNK_SIZE]
        chunk_text = " ".join(window).strip()

        if len(chunk_text) > 20:   # skip near-empty chunks
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=     chunk_text,
                source=   source,
                page=     page,
                tokens=   len(window),
            ))
            chunk_id += 1

        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_chunks(pdf_dir: str) -> List[Chunk]:
    """
    Extract, clean, and chunk all PDFs in the given directory.

    Args:
        pdf_dir: Path to folder containing the source PDFs.

    Returns:
        List of Chunk objects, ready for embedding.
    """
    pdf_dir  = Path(pdf_dir)
    all_chunks: List[Chunk] = []
    chunk_id = 0

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    for pdf_path in pdf_files:
        source_key = pdf_path.stem
        source     = SOURCE_LABELS.get(source_key, source_key)
        doc        = fitz.open(str(pdf_path))

        print(f"  Processing: {pdf_path.name} ({len(doc)} pages)")

        for page_num, page in enumerate(doc, start=1):
            raw  = _extract_page_text(page)
            text = _clean_text(raw)

            if not text.strip():
                print(f"    [p{page_num}] no text extracted — skipping")
                continue

            page_chunks = _chunk_text(text, source, page_num, chunk_id)
            print(f"    [p{page_num}] {len(text.split())} words → {len(page_chunks)} chunks")
            all_chunks.extend(page_chunks)
            chunk_id += len(page_chunks)

        doc.close()

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: List[Chunk], output_path: str) -> None:
    """Persist chunks to JSON for inspection and reuse."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
    print(f"Chunks saved → {output_path}")


def load_chunks(path: str) -> List[Chunk]:
    """Load chunks from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]
