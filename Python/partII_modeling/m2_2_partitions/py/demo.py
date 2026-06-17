"""Module 2.2 - Partitions & partition keys.

Uses a partition key (category) so Milvus automatically isolates rows by tenant
without manual partition management, then shows that filtering by the partition
key prunes the search space.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m2_2_partkey"
DIM = 64
N = 3000
CATEGORIES = ["books", "music", "games", "tools"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    # is_partition_key=True turns this scalar into the routing key.
    schema.add_field("category", DataType.VARCHAR, max_length=32, is_partition_key=True)

    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="IVF_FLAT",
                 metric_type="IP", params={"nlist": 16})
    client.create_collection(COLLECTION, schema=schema, index_params=ip,
                            num_partitions=16)

    vecs = random_unit_vectors(N, DIM, seed=5)
    rows = [{"id": i, "vector": vecs[i].tolist(), "category": CATEGORIES[i % len(CATEGORIES)]}
            for i in range(N)]
    client.insert(COLLECTION, rows)
    client.flush(COLLECTION)

    q = random_unit_vectors(1, DIM, seed=42)[0].tolist()

    # Unfiltered search scans all partition-key groups.
    res_all = client.search(COLLECTION, data=[q], anns_field="vector", limit=5,
                            output_fields=["category"])
    print("Unfiltered top-5 categories:", [h["entity"]["category"] for h in res_all[0]])

    # Filter on the partition key -> Milvus prunes to the matching group only.
    res_books = client.search(COLLECTION, data=[q], anns_field="vector", limit=5,
                              filter='category == "books"', output_fields=["category"])
    print("Filtered (books) top-5 :", [h["entity"]["category"] for h in res_books[0]])

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
