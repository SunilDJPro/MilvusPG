"""Part III benchmark harness - recall/QPS parameter sweep.

The spine reused by Parts V/VI/VII/VIII for measurement. Sweeps a search
parameter (HNSW ef, or IVF nprobe) and reports recall@10 and QPS so you can
draw a Pareto frontier. Outputs CSV to stdout.

Run:
    python sweep.py [--lite] [--index HNSW|IVF_FLAT] [--n 20000] [--queries 200]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors, brute_force_knn, recall_at_k
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "bench_harness"


def ground_truth(vectors, queries, k):
    return [brute_force_knn(vectors, q, k, metric="IP")[0] for q in queries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--index", default="HNSW", choices=["HNSW", "IVF_FLAT", "IVF_SQ8"])
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args()
    client = client_from_args(args)

    vectors = random_unit_vectors(args.n, args.dim, seed=4)
    queries = random_unit_vectors(args.queries, args.dim, seed=123)
    truth = ground_truth(vectors, queries, k=10)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=args.dim)
    ip = client.prepare_index_params()
    build_p = {"M": 16, "efConstruction": 200} if args.index == "HNSW" else {"nlist": 256}
    ip.add_index(field_name="vector", index_type=args.index, metric_type="IP", params=build_p)
    client.create_collection(COLLECTION, schema=schema, index_params=ip)
    client.insert(COLLECTION, [{"id": i, "vector": vectors[i].tolist()} for i in range(args.n)])
    client.flush(COLLECTION)

    sweep = [16, 32, 64, 128, 256] if args.index == "HNSW" else [1, 4, 8, 16, 32, 64]
    knob = "ef" if args.index == "HNSW" else "nprobe"
    qlist = [q.tolist() for q in queries]

    print(f"# index={args.index} n={args.n} dim={args.dim} queries={args.queries}")
    print(f"{knob},recall@10,qps")
    for val in sweep:
        t0 = time.perf_counter()
        res = client.search(COLLECTION, data=qlist, anns_field="vector", limit=10,
                            search_params={"params": {knob: val}})
        elapsed = time.perf_counter() - t0
        recalls = [recall_at_k([h["id"] for h in res[i]], truth[i], 10) for i in range(len(qlist))]
        qps = len(qlist) / elapsed
        print(f"{val},{np.mean(recalls):.3f},{qps:.0f}")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
