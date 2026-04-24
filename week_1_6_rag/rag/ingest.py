"""
Load document text from a filesystem path.

Supports plain text (.txt, .md, etc.) and .docx via stdlib only (zip + XML),
so no extra package is required for Word files.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# WordprocessingML namespace for document.xml text runs
_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _read_docx_text(path: Path) -> str:
    """Extract visible paragraph/run text from a .docx (OOXML) file."""
    parts: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("word/document.xml") as f:
            tree = ET.parse(f)
    # Concatenate all w:t text nodes in document order
    for node in tree.iterfind(".//w:t", _W_NS):
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    text = "".join(parts)
    return text.replace("\r\n", "\n").strip()


def load_document(path: str | Path) -> str:
    """
    Read a single file and return its text content.

    - .docx: parsed as Office Open XML (paragraph/run text only; no tables/images).
    - Other extensions: decoded as UTF-8 with replacement on errors.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Document not found: {p.resolve()}")

    suffix = p.suffix.lower()
    if suffix == ".docx":
        return _read_docx_text(p)

    return p.read_text(encoding="utf-8", errors="replace").strip()
