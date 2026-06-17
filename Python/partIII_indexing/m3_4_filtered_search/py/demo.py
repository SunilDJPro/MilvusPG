"""Module 3.4 - Filtered (scalar + vector) search.

Runs vector search constrained by boolean scalar expressions (price range,
brand membership, boolean flags). Shows the expression syntax and that adding a
scalar index changes the planner's work.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m3_4_filtered"
DIM = 64
N = 4000
BRANDS = ["acme", "globex", "initech", "umbrella"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("price", DataType.FLOAT)
    schema.add_field("brand", DataType.VARCHAR, max_length=32)
    schema.add_field("in_stock", DataType.BOOL)

    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="HNSW",
                 metric_type="IP", params={"M": 16, "efConstruction": 200})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    # A scalar index on the frequently-filtered field speeds the planner on
    # Standalone. (Skipped on Lite, whose engine indexes scalars differently.)
    if not args.lite:
        try:
            scalar_ix = client.prepare_index_params()
            scalar_ix.add_index(field_name="brand", index_type="INVERTED")
            client.create_index(COLLECTION, index_params=scalar_ix)
        except Exception as e:
            print(f"  (scalar index skipped: {type(e).__name__})")

    vecs = random_unit_vectors(N, DIM, seed=6)
    rows = [{"id": i, "vector": vecs[i].tolist(),
             "price": 5.0 + (i % 200), "brand": BRANDS[i % len(BRANDS)],
             "in_stock": i % 3 != 0} for i in range(N)]
    client.insert(COLLECTION, rows)
    client.flush(COLLECTION)

    q = random_unit_vectors(1, DIM, seed=55)[0].tolist()
    expr = 'price < 80 and brand in ["acme", "globex"] and in_stock == true'
    res = client.search(COLLECTION, data=[q], anns_field="vector", limit=5,
                        filter=expr, output_fields=["price", "brand", "in_stock"])
    print(f"Filter: {expr}\nTop-5 matching:")
    for h in res[0]:
        e = h["entity"]
        print(f"  id={h['id']} dist={h['distance']:.3f} price={e['price']:.0f} "
              f"brand={e['brand']} in_stock={e['in_stock']}")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
