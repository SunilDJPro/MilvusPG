# Module 2.3 - Insert, upsert, delete & primary-key semantics.

```
Module 2.3 - Insert, upsert, delete & primary-key semantics.

Inserts rows, upserts a slice (same PKs -> replace), deletes by expression, and
verifies counts at each step. Demonstrates that delete is by PK or by filter.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partII_modeling/m2_3_insert_upsert_delete/py/demo.py                 # Standalone
python partII_modeling/m2_3_insert_upsert_delete/py/demo.py --lite          # embedded Lite (if supported by this module)
```
