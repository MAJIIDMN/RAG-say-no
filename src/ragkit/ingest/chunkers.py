"""Chunkers. Token counts use a regex tokenizer: an approximation of model tokens, not exact."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from ragkit.core.registry import CHUNKERS
from ragkit.core.schemas import Chunk, Document

_TOKEN = re.compile(r"\w+|[^\w\s]")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ANCHOR = re.compile(r"\s*\{\s*#[\w.-]+\s*\}\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _make_chunks(doc: Document, spans: list[tuple[int, int, str]]) -> list[Chunk]:
    chunks = []
    for a, b, section in spans:
        body = doc.text[a:b].strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{len(chunks):04d}",
                    doc_id=doc.doc_id,
                    text=body,
                    title=doc.title,
                    section=section,
                    start_char=a,
                    end_char=b,
                    metadata=doc.metadata,
                )
            )
    return chunks


# Markdown heading chunker.


class HeadingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tokens: int = Field(400, description="Budget per chunk.")
    min_tokens: int = Field(80, description="Sections smaller than this merge with neighbours.")
    max_heading_level: int = Field(3, description="Deeper headings stay inside their parent.")


@dataclass
class _Block:
    start: int
    end: int
    tokens: int
    is_code: bool


@dataclass
class _Section:
    path: tuple[str, ...]
    start: int
    end: int


class MarkdownHeadingChunker:
    """Split on headings, pack paragraphs and whole code blocks up to the token budget."""

    def __init__(self, params: HeadingParams):
        self.p = params

    def chunk(self, doc: Document) -> list[Chunk]:
        spans: list[tuple[int, int, str]] = []
        for group in self._merge_small(doc.text, self._sections(doc)):
            path = " > ".join(group[0].path)
            for a, b in self._pack(doc.text, group[0].start, group[-1].end):
                spans.append((a, b, path))
        return _make_chunks(doc, spans)

    def _sections(self, doc: Document) -> list[_Section]:
        text, title = doc.text, doc.title
        stack: list[tuple[int, str]] = []
        sections: list[_Section] = []
        cur_start, cur_path, pos, in_code = 0, (title,) if title else (), 0, False
        for line in text.splitlines(keepends=True):
            if _FENCE.match(line):
                in_code = not in_code
            m = None if in_code else _HEADING.match(line.rstrip("\n"))
            if m and len(m.group(1)) <= self.p.max_heading_level:
                if pos > cur_start:
                    sections.append(_Section(cur_path, cur_start, pos))
                level, name = len(m.group(1)), _ANCHOR.sub("", m.group(2)).strip()
                stack = [(lv, n) for lv, n in stack if lv < level] + [(level, name)]
                names = [n for _, n in stack]
                if title and names[0] != title:
                    names = [title, *names]
                cur_path, cur_start = tuple(names), pos
            pos += len(line)
        if pos > cur_start:
            sections.append(_Section(cur_path, cur_start, pos))
        return sections

    def _merge_small(self, text: str, sections: list[_Section]) -> list[list[_Section]]:
        groups: list[list[_Section]] = []
        tokens: list[int] = []
        for s in sections:
            n = count_tokens(text[s.start : s.end])
            if (
                groups
                and (tokens[-1] < self.p.min_tokens or n < self.p.min_tokens)
                and (tokens[-1] + n <= self.p.max_tokens)
            ):
                groups[-1].append(s)
                tokens[-1] += n
            else:
                groups.append([s])
                tokens.append(n)
        return groups

    def _blocks(self, text: str, start: int, end: int) -> list[_Block]:
        """Paragraphs and fenced code blocks, as character spans."""
        blocks: list[_Block] = []
        pos, buf_start, in_code = start, start, False
        for line in text[start:end].splitlines(keepends=True):
            fence = bool(_FENCE.match(line))
            if fence and not in_code:
                if pos > buf_start:
                    blocks.append(self._block(text, buf_start, pos, False))
                buf_start, in_code = pos, True
            elif fence and in_code:
                blocks.append(self._block(text, buf_start, pos + len(line), True))
                buf_start, in_code = pos + len(line), False
            elif not in_code and not line.strip():
                if pos > buf_start:
                    blocks.append(self._block(text, buf_start, pos, False))
                buf_start = pos + len(line)
            pos += len(line)
        if end > buf_start:
            blocks.append(self._block(text, buf_start, end, in_code))
        return [b for b in blocks if text[b.start : b.end].strip()]

    @staticmethod
    def _block(text: str, a: int, b: int, is_code: bool) -> _Block:
        return _Block(a, b, count_tokens(text[a:b]), is_code)

    def _split_oversized(self, text: str, b: _Block) -> list[_Block]:
        """Split a block over budget on sentences (prose) or lines (code)."""
        segment = text[b.start : b.end]
        cuts = (
            [m.end() for m in re.finditer(r"\n", segment)]
            if b.is_code
            else [m.end() for m in _SENTENCE_END.finditer(segment)]
        )
        pieces, prev = [], 0
        for cut in [*cuts, len(segment)]:
            if cut > prev:
                pieces.append(self._block(text, b.start + prev, b.start + cut, b.is_code))
                prev = cut
        out = []
        for piece in pieces:
            if piece.tokens <= self.p.max_tokens:
                out.append(piece)
                continue
            # A single sentence or line still over budget: cut on token boundaries.
            spans = [m.span() for m in _TOKEN.finditer(text, piece.start, piece.end)]
            for i in range(0, len(spans), self.p.max_tokens):
                group = spans[i : i + self.p.max_tokens]
                out.append(self._block(text, group[0][0], group[-1][1], piece.is_code))
        return out

    def _pack(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        blocks: list[_Block] = []
        for b in self._blocks(text, start, end):
            blocks.extend([b] if b.tokens <= self.p.max_tokens else self._split_oversized(text, b))
        spans: list[tuple[int, int]] = []
        cur_start, cur_end, cur_tokens = None, None, 0
        for b in blocks:
            if cur_start is not None and cur_tokens + b.tokens > self.p.max_tokens:
                spans.append((cur_start, cur_end))
                cur_start, cur_tokens = None, 0
            if cur_start is None:
                cur_start = b.start
            cur_end, cur_tokens = b.end, cur_tokens + b.tokens
        if cur_start is not None:
            spans.append((cur_start, cur_end))
        return spans


@CHUNKERS.register("markdown_heading", HeadingParams)
def _markdown_heading(params: HeadingParams) -> MarkdownHeadingChunker:
    return MarkdownHeadingChunker(params)


# Fixed token window.


class FixedParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tokens: int = 400
    overlap_tokens: int = 0


class FixedChunker:
    def __init__(self, params: FixedParams):
        if params.overlap_tokens >= params.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.p = params

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = [m.span() for m in _TOKEN.finditer(doc.text)]
        step = self.p.max_tokens - self.p.overlap_tokens
        out = []
        for i in range(0, len(spans), step):
            group = spans[i : i + self.p.max_tokens]
            end = len(doc.text) if i + self.p.max_tokens >= len(spans) else group[-1][1]
            out.append((0 if i == 0 else group[0][0], end, doc.title))
            if i + self.p.max_tokens >= len(spans):
                break
        return _make_chunks(doc, out)


@CHUNKERS.register("fixed", FixedParams)
def _fixed(params: FixedParams) -> FixedChunker:
    return FixedChunker(params)
