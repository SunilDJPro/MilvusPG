# Module 1.2 - Distance metrics & embedding geometry.

```
Module 1.2 - Distance metrics & embedding geometry.

Demonstrates how ranking changes with metric_type. Same data, three collections
(L2 / IP / COSINE), one query, compare the top-k. Also shows a binary-vector
collection with HAMMING to set up the SLAM module later.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partI_fundamentals/m1_2_distance_metrics/py/demo.py                 # Standalone
python partI_fundamentals/m1_2_distance_metrics/py/demo.py --lite          # embedded Lite (if supported by this module)
```
