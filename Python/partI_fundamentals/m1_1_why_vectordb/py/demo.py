"""Module 1.1 - Why vector databases exist.

Shows that Milvus FLAT (exact) returns the SAME neighbors as a brute-force
numpy KNN, then contrasts the idea of exact vs approximate search. FLAT is the
ground-truth baseline every later index is measured against.

Run:
    python demo.py            # Standalone
    python demo.py --lite     # embedded Milvus Lite
"""
from __future__ import annotations

import argparse

import numpy as np
from pymilvus import DataType

from mvcommon import drop_if_exists, brute_force_knn, recall_at_k, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m1_1_why_vectordb"
DIM = 64
N = 5000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    vectors = random_unit_vectors(N, DIM, seed=1)
    query = random_unit_vectors(1, DIM, seed=99)[0]

    # Ground truth: exact KNN in numpy.
    truth_ids, _ = brute_force_knn(vectors, query, k=10, metric="IP")
    print("brute-force top-10 ids:", truth_ids)

    # Same thing via Milvus FLAT (also exact).
    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)

    client.insert(COLLECTION, [{"id": i, "vector": vectors[i].tolist()} for i in range(N)])
    client.flush(COLLECTION)

    res = client.search(COLLECTION, data=[query.tolist()], anns_field="vector", limit=10)
    milvus_ids = [h["id"] for h in res[0]]
    print("milvus FLAT  top-10 ids:", milvus_ids)

    r = recall_at_k(milvus_ids, truth_ids, k=10)
    print(f"recall@10 vs brute force: {r:.2f}  (FLAT is exact, expect 1.00)")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
