# Module 2.1 - Collections, schemas, fields.

```
Module 2.1 - Collections, schemas, fields.

Builds a product-catalog collection with a vector field, typed scalar fields,
a JSON field, and dynamic fields enabled. Shows that undefined keys land in the
dynamic $meta store and remain queryable.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partII_modeling/m2_1_collections_schema/py/demo.py                 # Standalone
python partII_modeling/m2_1_collections_schema/py/demo.py --lite          # embedded Lite (if supported by this module)
```
