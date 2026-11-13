"""Fetch the FastAPI docs at a pinned commit and turn them into plain Markdown for `ragkit ingest`.

Project-specific: this lives in scripts/, not in the reusable ragkit package.

    uv run python scripts/prepare_fastapi_docs.py            # clone into data/raw, write data/processed
    uv run python scripts/prepare_fastapi_docs.py --no-fetch # reuse an existing clone
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
COMMIT = "315f093f74bcb0e92da04e866c73a56bce217ee5"
SITE_URL = "https://fastapi.tiangolo.com/"
DOCS = Path("docs/en/docs")

# Changelogs, API reference stubs, community/meta pages and include-only fragments.
EXCLUDE = {
    "release-notes.md",
    "agent-banner.md",
    "translation-banner.md",
    "_llm-test.md",
    "fastapi-people.md",
    "external-links.md",
    "newsletter.md",
    "management.md",
    "management-tasks.md",
    "contributing.md",
    "translations.md",
}
EXCLUDE_DIRS = {"reference", "img", "css", "js"}

LANGS = {".py": "python", ".html": "html", ".css": "css", ".js": "javascript", ".json": "json"}
_INCLUDE = re.compile(r"\{\*\s*(\S+)([^*]*)\*\}|\{!>?\s*([^!]+?)\s*!\}")
_LINES = re.compile(r"ln\[(\d+):(\d+)\]")
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}")
_ADMONITION = re.compile(r"^/{3,}\s*(\w+)?\s*(?:\|\s*(.*))?$")
_FRONT = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_BLOCK_TAGS = re.compile(r"<(style|script)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_JINJA = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)
_TERMINAL_TAG = re.compile(r"</?(font|span|u|b)\b[^>]*>")
_TERMINAL_LANGS = {"console", "bash", "shell", "sh"}


def page_url(rel: str) -> str:
    path = rel.removesuffix(".md")
    path = path.removesuffix("index").rstrip("/")
    return SITE_URL + (f"{path}/" if path else "")


def _resolve(ref: str, repo: Path) -> Path:
    return repo / re.sub(r"^(\.\./)+", "", ref.strip())


def inline_includes(text: str, repo: Path, missing: list[str] | None = None) -> str:
    """Replace `{* file opts *}` and `{!file!}` with the file as a fenced code block."""

    def sub(m: re.Match) -> str:
        ref, opts = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), "")
        path = _resolve(ref, repo)
        if not path.is_file():
            if missing is not None:
                missing.append(ref)
            return ""
        lines = path.read_text().splitlines()
        if span := _LINES.search(opts):
            lines = lines[int(span.group(1)) - 1 : int(span.group(2))]
        return f"```{LANGS.get(path.suffix, '')}\n" + "\n".join(lines) + "\n```"

    return _INCLUDE.sub(sub, text)


def _clean_prose(text: str) -> str:
    """Strip HTML, Jinja and admonition fences outside code blocks; keep their text."""
    out, lang = [], None
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            lang = None if lang is not None else line.strip().strip("`").strip()
            out.append(line)
            continue
        if lang is not None:
            out.append(_TERMINAL_TAG.sub("", line) if lang in _TERMINAL_LANGS else line)
            continue
        if m := _ADMONITION.match(line.strip()):
            kind, title = (m.group(1) or "").strip(), (m.group(2) or "").strip()
            if kind.lower() == "tab":
                out.append(f"{title}:")
            elif kind:
                out.append(f"{kind.capitalize()}: {title}" if title else f"{kind.capitalize()}:")
            continue
        out.append(_TAG.sub("", _ANCHOR.sub("", line)))
    return "\n".join(out)


def _protect_code(text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def keep(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"\x00{len(blocks) - 1}\x00"

    return re.sub(r"```.*?```", keep, text, flags=re.DOTALL), blocks


def clean_markdown(text: str, repo: Path, missing: list[str] | None = None) -> str:
    text, blocks = _protect_code(text)
    text = _JINJA.sub("", _BLOCK_TAGS.sub("", text))
    text = re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], text)
    text = _clean_prose(inline_includes(text, repo, missing))
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def process_page(path: Path, repo: Path, missing: list[str] | None = None) -> str:
    rel = path.relative_to(repo / DOCS).as_posix()
    text = clean_markdown(_FRONT.sub("", path.read_text()), repo, missing)
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), rel)
    front = f"---\ntitle: {json.dumps(title)}\nurl: {page_url(rel)}\nsource: fastapi@{COMMIT[:7]}\n---\n\n"
    return front + text


def _excluded(rel: Path) -> bool:
    return rel.name in EXCLUDE or bool(EXCLUDE_DIRS & set(rel.parts[:-1]))


def prepare(repo: Path, out: Path) -> dict[str, int]:
    docs = repo / DOCS
    stats, missing = {"pages": 0, "skipped": 0}, []
    for path in sorted(docs.rglob("*.md")):
        rel = path.relative_to(docs)
        if _excluded(rel):
            stats["skipped"] += 1
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(process_page(path, repo, missing))
        stats["pages"] += 1
    stats["missing_includes"] = len(missing)
    return stats


def fetch(repo: Path, commit: str = COMMIT) -> None:
    """Sparse, blobless clone of only the English docs and their code samples."""
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--sparse",
                REPO_URL,
                str(repo),
            ],
            check=True,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "sparse-checkout",
            "set",
            str(DOCS),
            "docs_src",
            "fastapi/openapi",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", commit], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw/fastapi"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/fastapi"))
    ap.add_argument("--no-fetch", action="store_true", help="use the existing clone as is")
    args = ap.parse_args()
    if not args.no_fetch:
        fetch(args.raw)
    stats = prepare(args.raw, args.out)
    print(json.dumps({**stats, "commit": COMMIT, "out": str(args.out)}))


if __name__ == "__main__":
    main()
