"""Module 2.1 - Collections, schemas, fields.

Builds a product-catalog collection with a vector field, typed scalar fields,
a JSON field, and dynamic fields enabled. Shows that undefined keys land in the
dynamic $meta store and remain queryable.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, embed_texts
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m2_1_catalog"
DIM = 128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("price", DataType.FLOAT)
    schema.add_field("in_stock", DataType.BOOL)
    schema.add_field("attributes", DataType.JSON)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="HNSW",
                           metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)
    print("Created collection with vector + scalar + JSON + dynamic fields.")

    titles = ["wireless mouse", "mechanical keyboard", "usb-c hub", "4k monitor"]
    emb = embed_texts(titles, dim=DIM)
    rows = []
    for i, t in enumerate(titles):
        rows.append({
            "id": i,
            "embedding": emb[i].tolist(),
            "title": t,
            "price": 19.99 + i * 25,
            "in_stock": i % 2 == 0,
            "attributes": {"brand": ["acme", "globex"][i % 2], "rating": 4 + i % 2},
            # Undefined field -> stored in the dynamic $meta because dynamic is on.
            "warehouse": f"WH-{i % 3}",
        })
    client.insert(COLLECTION, rows)
    client.flush(COLLECTION)

    # Query using both a typed scalar field and a dynamic field.
    hits = client.query(COLLECTION, filter='price < 60 and warehouse == "WH-0"',
                        output_fields=["title", "price", "warehouse", "attributes"])
    print("Query (typed + dynamic filter):")
    for h in hits:
        print("  ", h)

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
