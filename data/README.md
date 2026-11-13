# Data

| Folder | Isi | Di-commit |
|---|---|---|
| `data/toy/` | Dokumentasi rekaan pustaka "Quokka" (3 halaman) untuk test dan CI | ya |
| `data/raw/fastapi/` | Salinan `docs/en/docs` dan `docs_src` dari repo FastAPI pada commit yang di-pin | tidak |
| `data/processed/fastapi/` | Markdown hasil pra-proses yang dibaca `ragkit ingest` | tidak |

`make data` mengunduh dokumentasi FastAPI berbahasa Inggris pada commit `315f093f` (sparse clone, hanya `docs/en/docs`, `docs_src`, dan `fastapi/openapi`), lalu menulis Markdown bersih ke `data/processed/fastapi/`. Skrip `scripts/prepare_fastapi_docs.py` menyisipkan contoh kode `{* ... *}`, menghapus anchor, HTML, dan pagar admonition, serta menulis front matter `title` dan `url`. Changelog, halaman `reference/`, dan halaman komunitas dilewati.

Quokka tidak ada. Isinya ditulis untuk proyek ini supaya test tidak bergantung pada data berlisensi.
