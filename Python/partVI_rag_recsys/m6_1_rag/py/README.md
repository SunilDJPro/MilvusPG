# Module 6.1 - RAG retrieval (the retrieval half, done properly).

```
Module 6.1 - RAG retrieval (the retrieval half, done properly).

Chunk -> embed -> hybrid retrieve (dense + BM25) -> metadata filter -> assemble
context. No LLM call here (the course is about the vector DB); the output is the
retrieved, ranked context you would hand to a generator.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partVI_rag_recsys/m6_1_rag/py/demo.py                 # Standalone
python partVI_rag_recsys/m6_1_rag/py/demo.py --lite          # embedded Lite (if supported by this module)
```
