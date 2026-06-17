# Module 8.2 - Scientific / bio / geo embeddings.

```
Module 8.2 - Scientific / bio / geo embeddings.

Two domains where metric choice is correctness-critical:
  1. Molecular fingerprints as BINARY_VECTOR with JACCARD (== Tanimoto), the
     standard chemical-similarity metric.
  2. Protein/sequence embeddings as dense FLOAT_VECTOR with COSINE.

The point: the same Milvus mechanics serve very different sciences, and picking
the wrong metric gives wrong answers.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partVIII_research/m8_2_scientific_embeddings/py/demo.py                 # Standalone
python partVIII_research/m8_2_scientific_embeddings/py/demo.py --lite          # embedded Lite (if supported by this module)
```
