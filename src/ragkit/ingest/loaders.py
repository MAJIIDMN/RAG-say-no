"""Read a corpus directory into Documents. Parsers are picked by file extension."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from ragkit.config import CorpusConfig
from ragkit.core.schemas import Document


class _HTMLText(HTMLParser):
    _BLOCK = {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "pre"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _read_html(path: Path) -> str:
    parser = _HTMLText()
    parser.feed(path.read_text(errors="ignore"))
    return "".join(parser.parts)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("PDF support needs the `pdf` extra: uv sync --extra pdf") from e
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


PARSERS: dict[str, Callable[[Path], str]] = {
    ".md": lambda p: p.read_text(errors="ignore"),
    ".txt": lambda p: p.read_text(errors="ignore"),
    ".html": _read_html,
    ".htm": _read_html,
    ".pdf": _read_pdf,
}


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta = yaml.safe_load(text[4:end]) or {}
    return (meta if isinstance(meta, dict) else {}), text[end + 5 :]


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip().lstrip("# ").strip()[:200]
    return ""


def load_documents(cfg: CorpusConfig) -> list[Document]:
    """`doc_id` is the path relative to the corpus root without extension: keep file names stable
    once an eval set references them."""
    root = Path(cfg.path)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    docs = []
    for path in sorted(root.glob(cfg.glob)):
        ext = path.suffix.lower()
        if not path.is_file() or ext not in cfg.extensions or ext not in PARSERS:
            continue
        meta, text = split_front_matter(PARSERS[ext](path))
        text = text.strip()
        if not text:
            continue
        doc_id = path.relative_to(root).with_suffix("").as_posix()
        docs.append(
            Document(
                doc_id=doc_id,
                text=text,
                title=str(meta.pop("title", "") or _first_line(text) or doc_id),
                source=str(path),
                metadata=meta,
            )
        )
    if not docs:
        raise ValueError(f"No documents with extensions {cfg.extensions} under {root}")
    return docs


def corpus_fingerprint(docs: list[Document]) -> dict[str, Any]:
    """Content digest plus the front-matter `source` values (e.g. `fastapi@315f093`)."""
    h = hashlib.sha256()
    for d in docs:
        h.update(f"{d.doc_id}\0{d.text}\0".encode())
    sources = sorted({str(d.metadata["source"]) for d in docs if d.metadata.get("source")})
    return {"digest": h.hexdigest()[:16], "sources": sources}
