"""Module 4.1 - Hybrid search & fusion.

Two vector fields per entity (dense semantic + a second dense "image-like"
field), two AnnSearchRequests, fused with RRF and with WeightedRanker. Shows how
fusion reorders results vs either field alone.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import DataType, AnnSearchRequest, RRFRanker, WeightedRanker

from mvcommon import drop_if_exists, embed_texts, random_unit_vectors
from mvcommon.cli import add_runtime_args, client_from_args

COLLECTION = "m4_1_hybrid"
TDIM = 128
IDIM = 128  # same dim as text field: Lite's hybrid_search currently requires
            # equal dimensions across vector fields; Standalone allows differing.

ITEMS = [
    "red running shoes lightweight",
    "blue leather formal shoes",
    "running shorts breathable",
    "leather wallet brown",
    "wireless running earbuds",
    "formal dress shirt white",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=256)
    schema.add_field("text_vec", DataType.FLOAT_VECTOR, dim=TDIM)
    schema.add_field("img_vec", DataType.FLOAT_VECTOR, dim=IDIM)

    ip = client.prepare_index_params()
    ip.add_index(field_name="text_vec", index_type="HNSW", metric_type="IP",
                 params={"M": 16, "efConstruction": 200})
    ip.add_index(field_name="img_vec", index_type="HNSW", metric_type="IP",
                 params={"M": 16, "efConstruction": 200})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    tvecs = embed_texts(ITEMS, dim=TDIM)
    ivecs = random_unit_vectors(len(ITEMS), IDIM, seed=21)  # stand-in image embeddings
    client.insert(COLLECTION, [
        {"id": i, "text": ITEMS[i], "text_vec": tvecs[i].tolist(), "img_vec": ivecs[i].tolist()}
        for i in range(len(ITEMS))
    ])
    client.flush(COLLECTION)

    q_text = embed_texts(["comfortable shoes for running"], dim=TDIM)[0].tolist()
    q_img = ivecs[0].tolist()  # pretend the user also gave an example image

    req_text = AnnSearchRequest(data=[q_text], anns_field="text_vec",
                                param={"params": {"ef": 64}}, limit=5)
    req_img = AnnSearchRequest(data=[q_img], anns_field="img_vec",
                               param={"params": {"ef": 64}}, limit=5)

    print("RRF fusion:")
    res = client.hybrid_search(COLLECTION, reqs=[req_text, req_img],
                               ranker=RRFRanker(60), limit=5, output_fields=["text"])
    for h in res[0]:
        print(f"  id={h['id']} {h['entity']['text']}")

    print("\nWeighted fusion (text 0.8 / image 0.3):")
    res = client.hybrid_search(COLLECTION, reqs=[req_text, req_img],
                               ranker=WeightedRanker(0.8, 0.3), limit=5,
                               output_fields=["text"])
    for h in res[0]:
        print(f"  id={h['id']} {h['entity']['text']}")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
