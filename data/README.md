# Data

| Folder | Contents | Committed |
|---|---|---|
| `data/toy/` | Fictional docs for a library called "Quokka" (3 pages) for tests and CI | yes |
| `data/raw/fastapi/` | Sparse clone of the FastAPI repo at the pinned commit | no |
| `data/processed/fastapi/` | Preprocessed Markdown read by `ragkit ingest` | no |

`make data` fetches the English FastAPI documentation at commit `315f093f` (sparse, blobless clone of
`docs/en/docs`, `docs_src` and `fastapi/openapi`) and writes clean Markdown to
`data/processed/fastapi/`. `scripts/prepare_fastapi_docs.py` inlines the `{* ... *}` code samples,
removes heading anchors, HTML, Jinja and admonition fences, and writes `title` and `url` front
matter. It skips the changelog, `reference/`, community pages and include-only fragments: 124 pages
kept, 34 skipped.

The FastAPI documentation is MIT licensed. Quokka does not exist; its pages were written for this
project so the tests do not depend on licensed data.
