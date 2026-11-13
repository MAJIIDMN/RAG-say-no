import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_fastapi_docs.py"
spec = importlib.util.spec_from_file_location("prepare_fastapi_docs", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "docs_src/bg").mkdir(parents=True)
    (tmp_path / "docs_src/bg/tutorial001.py").write_text("a = 1\nb = 2\nc = 3\n")
    (tmp_path / "docs_src/bg/page.html").write_text("<h1>{{ id }}</h1>\n")
    docs = tmp_path / "docs/en/docs"
    (docs / "tutorial").mkdir(parents=True)
    (docs / "reference").mkdir()
    (docs / "tutorial/background-tasks.md").write_text(
        "---\nhide:\n  - navigation\n---\n\n# Background Tasks { #background-tasks }\n\n"
        "Run *after* the <abbr title='x'>response</abbr>.\n\n"
        "{* ../../docs_src/bg/tutorial001.py hl[1] *}\n\n"
        "/// tip | Technical Details\n\nTasks run in a threadpool.\n\n///\n\n"
        "{!../../docs_src/bg/page.html!}\n"
    )
    (docs / "tutorial/index.md").write_text("# Tutorial\n\nStart here.\n")
    (docs / "reference/api.md").write_text("# API\n")
    (docs / "release-notes.md").write_text("# Release Notes\n")
    return tmp_path


def test_page_url_follows_mkdocs_directory_urls():
    assert prep.page_url("tutorial/background-tasks.md") == (
        "https://fastapi.tiangolo.com/tutorial/background-tasks/"
    )
    assert prep.page_url("tutorial/index.md") == "https://fastapi.tiangolo.com/tutorial/"
    assert prep.page_url("index.md") == "https://fastapi.tiangolo.com/"


def test_includes_become_fenced_code_and_line_ranges_are_respected(repo):
    text = prep.inline_includes("{* ../../docs_src/bg/tutorial001.py ln[2:3] hl[2] *}", repo)
    assert text == "```python\nb = 2\nc = 3\n```"


def test_process_page_cleans_markup_and_writes_front_matter(repo):
    out = prep.process_page(repo / "docs/en/docs/tutorial/background-tasks.md", repo)
    head, body = out.split("---\n", 2)[1:]
    assert 'title: "Background Tasks"' in head
    assert "url: https://fastapi.tiangolo.com/tutorial/background-tasks/" in head
    assert "# Background Tasks\n" in body and "{ #" not in body
    assert "Run *after* the response." in body
    assert "```python\na = 1" in body
    assert "Tip: Technical Details" in body and "///" not in body
    assert "```html\n<h1>{{ id }}</h1>" in body


def test_prepare_skips_excluded_pages_and_reports_counts(repo, tmp_path):
    out = tmp_path / "out"
    stats = prep.prepare(repo, out)
    written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.md"))
    assert written == ["tutorial/background-tasks.md", "tutorial/index.md"]
    assert stats["pages"] == 2 and stats["skipped"] == 2 and stats["missing_includes"] == 0


def test_console_blocks_lose_terminal_colour_markup_but_code_keeps_its_html(tmp_path):
    text = (
        '```console\n$ <font color="#4E9A06">fastapi</font> dev\n```\n\n'
        "```html\n<span>{{ id }}</span>\n```\n\n  \n<div>\n    Fast to code\n</div>\n"
    )
    out = prep.clean_markdown(text, tmp_path)
    assert "$ fastapi dev" in out
    assert "<span>{{ id }}</span>" in out
    assert "\n  \n" not in out and "Fast to code" in out
