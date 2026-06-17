# Module 6.2 - Recommendation & multimodal candidate generation.

```
Module 6.2 - Recommendation & multimodal candidate generation.

Item embeddings in Milvus; given a user's liked item, generate candidates via
ANN, then apply business-rule filters ("more like this but cheaper / in stock").
Also a tiny "search by image OR text" multimodal pattern using two query types
against the same item space.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partVI_rag_recsys/m6_2_recsys_multimodal/py/demo.py                 # Standalone
python partVI_rag_recsys/m6_2_recsys_multimodal/py/demo.py --lite          # embedded Lite (if supported by this module)
```
