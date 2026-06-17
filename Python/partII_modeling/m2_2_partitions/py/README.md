# Module 2.2 - Partitions & partition keys.

```
Module 2.2 - Partitions & partition keys.

Uses a partition key (category) so Milvus automatically isolates rows by tenant
without manual partition management, then shows that filtering by the partition
key prunes the search space.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partII_modeling/m2_2_partitions/py/demo.py                 # Standalone
python partII_modeling/m2_2_partitions/py/demo.py --lite          # embedded Lite (if supported by this module)
```
