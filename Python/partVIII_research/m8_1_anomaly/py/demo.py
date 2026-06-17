"""Module 8.1 - Anomaly detection via kNN distance.

Embed sliding windows of a signal; the distance to the k-th nearest neighbor in
"normal" history is the anomaly score. Points far from all normal history score
high. We build a normal baseline, then score a stream containing injected
anomalies.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

import numpy as np
from pymilvus import DataType

from mvcommon import drop_if_exists, l2_normalize
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m8_1_anomaly"
WIN = 32  # window length == embedding dim


def make_windows(series, win):
    return np.stack([series[i:i + win] for i in range(len(series) - win + 1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    client = client_from_args(args)

    rng = np.random.default_rng(0)
    # Normal regime: a noisy sine.
    t = np.arange(4000)
    normal = np.sin(t * 0.1) + rng.standard_normal(4000) * 0.05
    normal_win = l2_normalize(make_windows(normal, WIN).astype(np.float32))

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("window", DataType.FLOAT_VECTOR, dim=WIN)
    ip = client.prepare_index_params()
    ip.add_index(field_name="window", index_type="HNSW", metric_type="L2",
                 params={"M": 16, "efConstruction": 200})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)
    client.insert(COLLECTION, [{"id": i, "window": normal_win[i].tolist()}
                              for i in range(len(normal_win))])
    client.flush(COLLECTION)

    # Test stream: mostly normal, with two injected anomalies (spike + flatline).
    test = np.sin(np.arange(500) * 0.1) + rng.standard_normal(500) * 0.05
    test[200:210] += 4.0          # spike anomaly
    test[350:382] = 0.0           # flatline anomaly
    test_win = l2_normalize(make_windows(test, WIN).astype(np.float32))

    res = client.search(COLLECTION, data=test_win.tolist(), anns_field="window",
                        limit=args.k, search_params={"params": {"ef": 64}})
    # Score = distance to k-th neighbor (last hit).
    scores = np.array([hits[-1]["distance"] for hits in res])

    thresh = np.percentile(scores, 95)
    flagged = np.where(scores > thresh)[0]
    print(f"Scored {len(scores)} windows; threshold(p95)={thresh:.3f}")
    print(f"Flagged {len(flagged)} anomalous windows.")
    print("Flagged window starts (expect clusters near 200 and 350):",
          flagged[:20].tolist())

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
