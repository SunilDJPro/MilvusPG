# Module 3.3 - Consistency levels & the read path.

```
Module 3.3 - Consistency levels & the read path.

Demonstrates the insert-then-search freshness window. With Strong consistency a
just-inserted row is immediately searchable; with Eventually it may not be.

NOTE: This module is meaningfully a STANDALONE concept. Milvus Lite is a single
in-process engine, so the multi-level consistency semantics don't apply the same
way; the script still runs on Lite but the contrast is only real on a server.

Run:
    python demo.py            # Standalone (recommended for this module)
    python demo.py --lite     # runs, but consistency contrast is server-only
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partIII_indexing/m3_3_consistency/py/demo.py                 # Standalone
python partIII_indexing/m3_3_consistency/py/demo.py --lite          # embedded Lite (if supported by this module)
```
