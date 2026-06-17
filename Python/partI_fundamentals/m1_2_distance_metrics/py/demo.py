"""Module 1.2 - Distance metrics & embedding geometry.

Demonstrates how ranking changes with metric_type. Same data, three collections
(L2 / IP / COSINE), one query, compare the top-k. Also shows a binary-vector
collection with HAMMING to set up the SLAM module later.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

import numpy as np
from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors, random_binary_vectors
from mvcommon.cli import add_runtime_args, client_from_args

DIM = 32
N = 2000


def float_metric_demo(client, metric):
    name = f"m1_2_{metric.lower()}"
    drop_if_exists(client, name)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="FLAT", metric_type=metric)
    client.create_collection(name, schema=schema, index_params=ip)

    # NOTE: vectors are NOT normalized here, so IP and COSINE rank differently.
    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32) * rng.uniform(0.5, 3.0, (N, 1))
    client.insert(name, [{"id": i, "vector": vecs[i].tolist()} for i in range(N)])
    client.flush(name)

    query = rng.standard_normal(DIM).astype(np.float32)
    res = client.search(name, data=[query.tolist()], anns_field="vector", limit=5)
    ids = [h["id"] for h in res[0]]
    client.drop_collection(name)
    return ids


def binary_hamming_demo(client):
    name = "m1_2_hamming"
    drop_if_exists(client, name)
    bits = 128
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("bvec", DataType.BINARY_VECTOR, dim=bits)
    ip = client.prepare_index_params()
    ip.add_index(field_name="bvec", index_type="BIN_FLAT", metric_type="HAMMING")
    client.create_collection(name, schema=schema, index_params=ip)

    vecs = random_binary_vectors(500, bits, seed=3)
    client.insert(name, [{"id": i, "bvec": vecs[i]} for i in range(500)])
    client.flush(name)

    res = client.search(name, data=[vecs[0]], anns_field="bvec", limit=3)
    client.drop_collection(name)
    return [(h["id"], h["distance"]) for h in res[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    print("Same un-normalized data, different metric_type -> different ranking:")
    for metric in ("L2", "IP", "COSINE"):
        ids = float_metric_demo(client, metric)
        print(f"  {metric:>7}: top-5 ids = {ids}")
    print("\nL2 vs IP/COSINE differ because vectors are not unit length.")

    print("\nBinary vectors + HAMMING (query == item 0, so distance 0 first):")
    if args.lite:
        print("  [skipped] Milvus Lite 3.0 does not support BINARY_VECTOR; "
              "run against Standalone to see this.")
    else:
        for rid, dist in binary_hamming_demo(client):
            print(f"  id={rid} hamming={dist}")


if __name__ == "__main__":
    main()
