# Module 1.1 - Why vector databases exist.

```
Module 1.1 - Why vector databases exist.

Shows that Milvus FLAT (exact) returns the SAME neighbors as a brute-force
numpy KNN, then contrasts the idea of exact vs approximate search. FLAT is the
ground-truth baseline every later index is measured against.

Run:
    python demo.py            # Standalone
    python demo.py --lite     # embedded Milvus Lite
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partI_fundamentals/m1_1_why_vectordb/py/demo.py                 # Standalone
python partI_fundamentals/m1_1_why_vectordb/py/demo.py --lite          # embedded Lite (if supported by this module)
```
