"""Module 2.3 - Insert, upsert, delete & primary-key semantics.

Inserts rows, upserts a slice (same PKs -> replace), deletes by expression, and
verifies counts at each step. Demonstrates that delete is by PK or by filter.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m2_3_writes"
DIM = 32
N = 1000


def count(client):
    return client.query(COLLECTION, filter="", output_fields=["count(*)"])[0]["count(*)"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("tag", DataType.VARCHAR, max_length=16)
    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="FLAT", metric_type="L2")
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    vecs = random_unit_vectors(N, DIM, seed=11)
    client.insert(COLLECTION, [{"id": i, "vector": vecs[i].tolist(), "tag": "v1"} for i in range(N)])
    client.flush(COLLECTION)
    print("after insert:", count(client), "rows")

    # Upsert the first 100 ids with a new tag (same PKs -> replace, not append).
    client.upsert(COLLECTION, [{"id": i, "vector": vecs[i].tolist(), "tag": "v2"} for i in range(100)])
    client.flush(COLLECTION)
    print("after upsert :", count(client), "rows (unchanged; PKs replaced)")
    v2 = client.query(COLLECTION, filter='tag == "v2"', output_fields=["count(*)"])[0]["count(*)"]
    print("  rows now tagged v2:", v2)

    # Delete by expression.
    client.delete(COLLECTION, filter="id < 50")
    client.flush(COLLECTION)
    print("after delete id<50:", count(client), "rows")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
