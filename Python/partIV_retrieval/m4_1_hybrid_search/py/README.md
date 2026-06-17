# Module 4.1 - Hybrid search & fusion.

```
Module 4.1 - Hybrid search & fusion.

Two vector fields per entity (dense semantic + a second dense "image-like"
field), two AnnSearchRequests, fused with RRF and with WeightedRanker. Shows how
fusion reorders results vs either field alone.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partIV_retrieval/m4_1_hybrid_search/py/demo.py                 # Standalone
python partIV_retrieval/m4_1_hybrid_search/py/demo.py --lite          # embedded Lite (if supported by this module)
```
