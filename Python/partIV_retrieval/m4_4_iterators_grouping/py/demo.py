"""Module 4.4 - Search iterators, range search, grouping.

Three features:
  - group_by_field for diversity (one result per group)
  - range search via radius / range_filter
  - search_iterator for deep pagination beyond the normal limit

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m4_4_iter_group"
DIM = 64
N = 3000
AUTHORS = [f"author_{i}" for i in range(20)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("author", DataType.VARCHAR, max_length=32)
    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="HNSW", metric_type="IP",
                 params={"M": 16, "efConstruction": 200})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    vecs = random_unit_vectors(N, DIM, seed=9)
    client.insert(COLLECTION, [{"id": i, "vector": vecs[i].tolist(),
                               "author": AUTHORS[i % len(AUTHORS)]} for i in range(N)])
    client.flush(COLLECTION)

    q = random_unit_vectors(1, DIM, seed=33)[0].tolist()

    # Group-by: at most one hit per author.
    try:
        res = client.search(COLLECTION, data=[q], anns_field="vector", limit=5,
                            group_by_field="author", output_fields=["author"])
        print("Group-by author (diverse top-5):",
              [h["entity"]["author"] for h in res[0]])
    except Exception as e:
        print("group_by not supported on this backend:", type(e).__name__)

    # Range search: only neighbors with IP similarity in (radius, range_filter].
    res = client.search(COLLECTION, data=[q], anns_field="vector", limit=10,
                        search_params={"params": {"radius": 0.0, "range_filter": 1.0}})
    print(f"Range search returned {len(res[0])} hits within the similarity band.")

    # Iterator: page through results in batches.
    try:
        it = client.search_iterator(COLLECTION, data=[q], anns_field="vector",
                                     batch_size=50, limit=200)
        total = 0
        while True:
            batch = it.next()
            if not batch:
                it.close()
                break
            total += len(batch)
        print(f"Iterator paged through {total} results.")
    except Exception as e:
        print("search_iterator not available on this backend:", type(e).__name__)

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
