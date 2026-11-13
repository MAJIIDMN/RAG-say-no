import pytest

from ragkit.config import CorpusConfig
from ragkit.core.registry import CHUNKERS
from ragkit.core.schemas import Document
from ragkit.ingest.chunkers import count_tokens
from ragkit.ingest.loaders import load_documents

PAGE = """---
title: Path Parameters
url: https://example.com/tutorial/path-params/
---
# Path Parameters { #path-parameters }

Intro paragraph about path parameters.

## Declare a path parameter { #declare }

Use braces in the path.

```python
# Not a heading: this is a comment inside code
@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}
```

### Order matters

Fixed paths must be declared before parameterised ones.
"""


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "tutorial").mkdir()
    (tmp_path / "tutorial" / "path-params.md").write_text(PAGE)
    (tmp_path / "notes.txt").write_text("Plain text note.")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    return tmp_path


def chunk(doc, **params):
    return CHUNKERS.create({"impl": "markdown_heading", **params}).chunk(doc)


def test_loader_reads_front_matter_and_skips_unknown_types(corpus):
    docs = {d.doc_id: d for d in load_documents(CorpusConfig(path=str(corpus)))}
    assert set(docs) == {"tutorial/path-params", "notes"}
    page = docs["tutorial/path-params"]
    assert page.title == "Path Parameters"
    assert page.metadata["url"] == "https://example.com/tutorial/path-params/"
    assert not page.text.startswith("---")


def test_loader_rejects_missing_or_empty_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(CorpusConfig(path=str(tmp_path / "nope")))
    with pytest.raises(ValueError, match="No documents"):
        load_documents(CorpusConfig(path=str(tmp_path)))


def test_heading_chunker_builds_breadcrumbs_without_anchors(corpus):
    doc = load_documents(CorpusConfig(path=str(corpus / "tutorial")))[0]
    chunks = chunk(doc, max_tokens=30, min_tokens=1)
    sections = [c.section for c in chunks]
    assert "Path Parameters > Declare a path parameter" in sections
    assert "Path Parameters > Declare a path parameter > Order matters" in sections
    assert not any("{" in s for s in sections)
    assert all(c.metadata["url"].endswith("/path-params/") for c in chunks)


def test_heading_chunker_ignores_hash_lines_inside_code_and_keeps_code_whole(corpus):
    doc = load_documents(CorpusConfig(path=str(corpus / "tutorial")))[0]
    chunks = chunk(doc, max_tokens=60, min_tokens=1)
    assert not any("Not a heading" in c.section for c in chunks)
    code = [c for c in chunks if "```python" in c.text]
    assert len(code) == 1 and code[0].text.count("```") == 2


def test_code_block_over_budget_is_split_on_lines():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(40)) + "\n```"
    chunks = chunk(Document(doc_id="d", text=f"# T\n\n{code}"), max_tokens=50, min_tokens=1)
    assert len(chunks) > 1 and all(count_tokens(c.text) <= 50 for c in chunks)


def test_heading_chunker_respects_budget_and_offsets():
    text = "# T\n\n## Long\n\n" + " ".join(f"Sentence number {i} ends here." for i in range(200))
    doc = Document(doc_id="d", text=text, title="T")
    chunks = chunk(doc, max_tokens=60, min_tokens=10)
    assert len(chunks) > 3
    assert all(count_tokens(c.text) <= 60 for c in chunks)
    assert all(c.text in text[c.start_char : c.end_char] for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_small_sections_are_merged_up_to_budget():
    text = "# T\n\n## A\n\nShort a.\n\n## B\n\nShort b.\n\n## C\n\nShort c."
    chunks = chunk(Document(doc_id="d", text=text), max_tokens=400, min_tokens=50)
    assert len(chunks) == 1 and "Short c." in chunks[0].text


def test_fixed_chunker_covers_text_with_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = CHUNKERS.create({"impl": "fixed", "max_tokens": 100, "overlap_tokens": 20}).chunk(
        Document(doc_id="d", text=text)
    )
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(text)
    assert all(b.start_char < a.end_char for a, b in zip(chunks, chunks[1:], strict=False))
