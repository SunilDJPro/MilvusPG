"""Module 3.3 - Consistency levels & the read path.

Demonstrates the insert-then-search freshness window. With Strong consistency a
just-inserted row is immediately searchable; with Eventually it may not be.

NOTE: This module is meaningfully a STANDALONE concept. Milvus Lite is a single
in-process engine, so the multi-level consistency semantics don't apply the same
way; the script still runs on Lite but the contrast is only real on a server.

Run:
    python demo.py            # Standalone (recommended for this module)
    python demo.py --lite     # runs, but consistency contrast is server-only
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m3_3_consistency"
DIM = 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    if args.lite:
        print("[note] Lite is single-process; consistency levels are a Standalone concept.")

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="FLAT", metric_type="L2")
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    vecs = random_unit_vectors(500, DIM, seed=8)
    client.insert(COLLECTION, [{"id": i, "vector": vecs[i].tolist()} for i in range(500)])

    new_id = 9999
    probe = random_unit_vectors(1, DIM, seed=8888)[0]

    # Strong consistency forces the read to see the latest writes.
    client.insert(COLLECTION, [{"id": new_id, "vector": probe.tolist()}])
    res = client.search(COLLECTION, data=[probe.tolist()], anns_field="vector",
                        limit=1, consistency_level="Strong")
    found_strong = res[0][0]["id"] == new_id
    print("Strong: just-inserted row visible immediately?", found_strong)

    print("On Standalone, repeat with consistency_level='Eventually' to observe a")
    print("freshness window where the new row may not yet be searchable.")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
