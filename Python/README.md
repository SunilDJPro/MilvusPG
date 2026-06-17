# Milvus Deep-Dive — Python Codebase

Runnable Python (`pymilvus`) implementations of the course modules described in
`milvus-vector-db-course-plan.md`. This package is **Python only**; Go and C++
parity directories will be added later.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e common/py          # installs the shared `mvcommon` helpers
```

## Two ways to run every demo

Each demo defaults to **Standalone** and accepts `--lite` for embedded Milvus Lite.

**Standalone (full feature set — recommended):**
```bash
docker compose -f infra/docker-compose.yml up -d   # starts Milvus 2.6.16
export PYTHONPATH=common/py
python partI_fundamentals/m1_1_why_vectordb/py/demo.py
```

**Milvus Lite (embedded, no server, Python-only):**
```bash
export PYTHONPATH=common/py
python partI_fundamentals/m1_1_why_vectordb/py/demo.py --lite --db-path ./demo.db
```

> If you `pip install -e common/py` you can drop the `PYTHONPATH` export.

## What runs where

Milvus Lite 3.0 is a pure-Python engine with a **reduced** feature set. Demos
that need a Standalone-only feature detect `--lite` and print a clear skip note
instead of crashing. Verified on milvus-lite 3.0:

| Feature | Standalone | Lite 3.0 |
|---|---|---|
| `FLOAT_VECTOR` | ✅ | ✅ |
| `BINARY_VECTOR` (Hamming/Jaccard) | ✅ | ❌ skipped |
| `SPARSE_FLOAT_VECTOR` + BM25 full-text | ✅ | ❌ skipped |
| Index types | FLAT/IVF_FLAT/IVF_SQ8/IVF_PQ/HNSW/DiskANN/GPU/binary | HNSW, HNSW_SQ, IVF_FLAT, IVF_SQ8, BRUTE_FORCE |
| Multi-vector `hybrid_search` | any dims | requires **equal dims** across vector fields |
| COSINE score convention | similarity (higher=closer) | returns a distance (lower=closer); **ranking still correct** |
| Consistency levels | Strong/Bounded/Eventual/Session | single-process; not meaningful |

Modules affected by Lite limits: 1.2 (binary half), 4.2 & 6.1 (BM25), 7.1
(binary descriptors), 8.2 (fingerprint half). All run fully on Standalone.

## Layout

```
common/py/mvcommon/   shared helpers (connection, embeddings, datasets, bench, cli)
partI..partVIII/      module demos, each in a py/ subdir with a demo.py + README
infra/                docker-compose for Standalone
VERSIONS.md           pinned versions + Lite/Standalone matrix
```

Each module folder has its own `README.md` explaining the concept, how to run
it, and any runtime caveats.

## Smoke test

`scripts/run_all_lite.sh` runs every module against Lite (skips the
Standalone-only ones) as a quick health check.
