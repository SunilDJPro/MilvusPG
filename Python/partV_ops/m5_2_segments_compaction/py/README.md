# Module 5.2 - Segments, compaction, loading.

```
Module 5.2 - Segments, compaction, loading.

Inserts in several batches (creating multiple segments), flushes, then triggers
compaction and inspects state. Also shows load / release of a collection.

NOTE: Segment/compaction internals are a STANDALONE concept. On Milvus Lite the
storage engine differs (LSM/Parquet) and these admin calls may be no-ops or
unsupported; the script guards each call so it runs either way but is meant for
Standalone.

Run:
    python demo.py            # Standalone (recommended)
    python demo.py --lite
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partV_ops/m5_2_segments_compaction/py/demo.py                 # Standalone
python partV_ops/m5_2_segments_compaction/py/demo.py --lite          # embedded Lite (if supported by this module)
```
