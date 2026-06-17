# Module 3.1 - The index zoo.

```
Module 3.1 - The index zoo.

Builds the same dataset under several index types and reports recall@10 (vs a
FLAT ground truth) and search latency. This is the conceptual core: see the
recall/latency tradeoff with your own eyes.

Index availability differs by runtime:
  - Standalone: FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, DiskANN, GPU, binary...
  - Milvus Lite: in-memory FAISS-backed set (FLAT/IVF/HNSW-class); NO DiskANN/GPU.
The script auto-skips index types the backend rejects.

Run:
    python demo.py [--lite] [--n 20000] [--dim 128]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partIII_indexing/m3_1_index_zoo/py/demo.py                 # Standalone
python partIII_indexing/m3_1_index_zoo/py/demo.py --lite          # embedded Lite (if supported by this module)
```
