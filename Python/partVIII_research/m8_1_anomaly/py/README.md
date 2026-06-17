# Module 8.1 - Anomaly detection via kNN distance.

```
Module 8.1 - Anomaly detection via kNN distance.

Embed sliding windows of a signal; the distance to the k-th nearest neighbor in
"normal" history is the anomaly score. Points far from all normal history score
high. We build a normal baseline, then score a stream containing injected
anomalies.

Run:
    python demo.py [--lite]
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partVIII_research/m8_1_anomaly/py/demo.py                 # Standalone
python partVIII_research/m8_1_anomaly/py/demo.py --lite          # embedded Lite (if supported by this module)
```
