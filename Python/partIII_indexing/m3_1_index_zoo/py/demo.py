"""Module 3.1 - The index zoo.

Builds the same dataset under several index types and reports recall@10 (vs a
FLAT ground truth) and search latency. This is the conceptual core: see the
recall/latency tradeoff with your own eyes.

Index availability differs by runtime:
  - Standalone: FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, DiskANN, GPU, binary...
  - Milvus Lite: in-memory FAISS-backed set (FLAT/IVF/HNSW-class); NO DiskANN/GPU.
The script auto-skips index types the backend rejects.

Run:
    python demo.py [--lite] [--n 20000] [--dim 128]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors, brute_force_knn, recall_at_k
from mvcommon.cli import add_runtime_args, client_from_args

CANDIDATES = [
    ("FLAT", {}, {}),
    ("IVF_FLAT", {"nlist": 256}, {"nprobe": 16}),
    ("IVF_SQ8", {"nlist": 256}, {"nprobe": 16}),
    ("IVF_PQ", {"nlist": 256, "m": 16, "nbits": 8}, {"nprobe": 16}),
    ("HNSW", {"M": 16, "efConstruction": 200}, {"ef": 64}),
    ("DISKANN", {}, {"search_list": 100}),  # Standalone only; skipped on Lite.
]

# Milvus Lite's pure-Python engine implements an in-memory FAISS-backed subset.
# As of milvus-lite 3.0 it supports: HNSW, HNSW_SQ, IVF_FLAT, IVF_SQ8,
# BRUTE_FORCE (FLAT is accepted as an alias). DiskANN, GPU, and IVF_PQ are not
# available, so skip them up front on Lite rather than letting a background
# index build raise. (On Standalone all of these work.)
LITE_UNSUPPORTED = {"DISKANN", "IVF_PQ", "GPU_IVF_FLAT", "GPU_IVF_PQ", "GPU_CAGRA"}


def build_and_measure(client, name, vectors, query, truth, index_type, build_p, search_p):
    drop_if_exists(client, name)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vectors.shape[1])
    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type=index_type,
                 metric_type="IP", params=build_p)
    try:
        client.create_collection(name, schema=schema, index_params=ip)
    except Exception as e:
        client.drop_collection(name) if client.has_collection(name) else None
        return None, f"build unsupported ({type(e).__name__})"

    client.insert(name, [{"id": i, "vector": vectors[i].tolist()} for i in range(len(vectors))])
    client.flush(name)

    t0 = time.perf_counter()
    res = client.search(name, data=[query.tolist()], anns_field="vector", limit=10,
                        search_params={"params": search_p})
    dt = (time.perf_counter() - t0) * 1000
    ids = [h["id"] for h in res[0]]
    r = recall_at_k(ids, truth, k=10)
    client.drop_collection(name)
    return (r, dt), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    client = client_from_args(args)

    vectors = random_unit_vectors(args.n, args.dim, seed=2)
    query = random_unit_vectors(1, args.dim, seed=77)[0]
    truth, _ = brute_force_knn(vectors, query, k=10, metric="IP")

    print(f"{args.n} x {args.dim} vectors, metric=IP\n")
    print(f"{'index':<10}{'recall@10':>10}{'latency_ms':>12}")
    print("-" * 32)
    for index_type, build_p, search_p in CANDIDATES:
        if args.lite and index_type in LITE_UNSUPPORTED:
            print(f"{index_type:<10}{'-':>10}{'  server-only':>12}")
            continue
        result, skip = build_and_measure(
            client, f"m3_1_{index_type.lower()}", vectors, query, truth,
            index_type, build_p, search_p)
        if skip:
            print(f"{index_type:<10}{'-':>10}{'  ' + skip:>12}")
        else:
            r, dt = result
            print(f"{index_type:<10}{r:>10.2f}{dt:>12.1f}")


if __name__ == "__main__":
    main()
