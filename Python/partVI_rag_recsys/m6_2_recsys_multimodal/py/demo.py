"""Module 6.2 - Recommendation & multimodal candidate generation.

Item embeddings in Milvus; given a user's liked item, generate candidates via
ANN, then apply business-rule filters ("more like this but cheaper / in stock").
Also a tiny "search by image OR text" multimodal pattern using two query types
against the same item space.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import DataType

from mvcommon import drop_if_exists, embed_texts
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m6_2_recsys"
DIM = 256

CATALOG = [
    ("trail running shoe", 120.0, True),
    ("road running shoe", 110.0, True),
    ("running sock pack", 15.0, True),
    ("trail running shoe pro", 180.0, False),
    ("hiking boot", 160.0, True),
    ("running cap", 25.0, True),
    ("budget running shoe", 60.0, True),
    ("formal leather shoe", 140.0, True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("name", DataType.VARCHAR, max_length=128)
    schema.add_field("price", DataType.FLOAT)
    schema.add_field("in_stock", DataType.BOOL)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    ip = client.prepare_index_params()
    ip.add_index(field_name="embedding", index_type="HNSW", metric_type="IP",
                 params={"M": 16, "efConstruction": 200})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    names = [c[0] for c in CATALOG]
    emb = embed_texts(names, dim=DIM)
    client.insert(COLLECTION, [{"id": i, "name": names[i], "price": CATALOG[i][1],
                               "in_stock": CATALOG[i][2], "embedding": emb[i].tolist()}
                              for i in range(len(CATALOG))])
    client.flush(COLLECTION)

    liked_id = 0  # "trail running shoe"
    liked_vec = emb[liked_id].tolist()
    liked_price = CATALOG[liked_id][1]
    print(f"User liked: {names[liked_id]} (${liked_price:.0f})\n")

    # "More like this, but cheaper and in stock", excluding the item itself.
    expr = f"price < {liked_price} and in_stock == true and id != {liked_id}"
    res = client.search(COLLECTION, data=[liked_vec], anns_field="embedding",
                        limit=4, filter=expr, output_fields=["name", "price"])
    print("Recommended (similar, cheaper, in stock):")
    for h in res[0]:
        print(f"  {h['entity']['name']} (${h['entity']['price']:.0f}) sim={h['distance']:.3f}")

    # Multimodal pattern: a text query hits the same item embedding space.
    print("\nSearch-by-text over the same items ('something for hiking'):")
    q = embed_texts(["something for hiking"], dim=DIM)[0].tolist()
    res = client.search(COLLECTION, data=[q], anns_field="embedding", limit=3,
                        output_fields=["name"])
    for h in res[0]:
        print(f"  {h['entity']['name']} sim={h['distance']:.3f}")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
