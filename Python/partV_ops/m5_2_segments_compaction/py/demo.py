"""Module 5.2 - Segments, compaction, loading.

Inserts in several batches (creating multiple segments), flushes, then triggers
compaction and inspects state. Also shows load / release of a collection.

NOTE: Segment/compaction internals are a STANDALONE concept. On Milvus Lite the
storage engine differs (LSM/Parquet) and these admin calls may be no-ops or
unsupported; the script guards each call so it runs either way but is meant for
Standalone.

Run:
    python demo.py            # Standalone (recommended)
    python demo.py --lite
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m5_2_segments"
DIM = 64


def try_call(label, fn):
    try:
        out = fn()
        print(f"  {label}: {out}")
    except Exception as e:
        print(f"  {label}: not supported here ({type(e).__name__})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="L2",
                 params={"nlist": 128})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    # Several insert+flush cycles -> several sealed segments.
    base = 0
    for batch in range(4):
        vecs = random_unit_vectors(1000, DIM, seed=batch)
        client.insert(COLLECTION, [{"id": base + i, "vector": vecs[i].tolist()}
                                   for i in range(1000)])
        client.flush(COLLECTION)
        base += 1000
    print("Inserted 4 flushed batches (multiple sealed segments).")

    try_call("compact", lambda: client.compact(COLLECTION))
    try_call("release", lambda: client.release_collection(COLLECTION) or "released")
    try_call("load", lambda: client.load_collection(COLLECTION) or "loaded")

    n = client.query(COLLECTION, filter="", output_fields=["count(*)"])[0]["count(*)"]
    print("Total rows after compaction:", n)

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
