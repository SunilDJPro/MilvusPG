# Module 4.2 - Sparse vectors & full-text/BM25.

```
Module 4.2 - Sparse vectors & full-text/BM25.

Milvus generates sparse BM25 vectors from raw text via a built-in Function, so
you insert text and search with text. Then a dense + sparse hybrid shows the
keyword/semantic complementarity.

Run:
    python demo.py            # Standalone
    python demo.py --lite     # milvus-lite 3.0 supports BM25 too
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partIV_retrieval/m4_2_full_text_bm25/py/demo.py                 # Standalone
python partIV_retrieval/m4_2_full_text_bm25/py/demo.py --lite          # embedded Lite (if supported by this module)
```
