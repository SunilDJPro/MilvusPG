"""Module 4.2 - Sparse vectors & full-text/BM25.

Milvus generates sparse BM25 vectors from raw text via a built-in Function, so
you insert text and search with text. Then a dense + sparse hybrid shows the
keyword/semantic complementarity.

Run:
    python demo.py            # Standalone
    python demo.py --lite     # milvus-lite 3.0 supports BM25 too
"""
from __future__ import annotations

import argparse

from pymilvus import (
    DataType, Function, FunctionType, AnnSearchRequest, RRFRanker,
)

from mvcommon import drop_if_exists, embed_texts
from mvcommon.cli import add_runtime_args, client_from_args, require_standalone
from mvcommon.datasets import _DOCS

COLLECTION = "m4_2_bm25"
DIM = 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    if not require_standalone(args, "BM25"):
        return
    client = client_from_args(args)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text", DataType.VARCHAR, max_length=1000, enable_analyzer=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 function: text -> sparse. The sparse field is function output.
    bm25 = Function(name="text_bm25", input_field_names=["text"],
                    output_field_names=["sparse"], function_type=FunctionType.BM25)
    schema.add_function(bm25)

    ip = client.prepare_index_params()
    ip.add_index(field_name="dense", index_type="HNSW", metric_type="IP",
                 params={"M": 16, "efConstruction": 200})
    ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="BM25", params={"inverted_index_algo": "DAAT_WAND"})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    texts = [d[0] for d in _DOCS]
    dense = embed_texts(texts, dim=DIM)
    # Note: do NOT supply 'sparse' - the BM25 function fills it from 'text'.
    client.insert(COLLECTION, [{"text": texts[i], "dense": dense[i].tolist()}
                              for i in range(len(texts))])
    client.flush(COLLECTION)

    query = "graph index for fast approximate search"

    print("Full-text BM25 search (keyword relevance):")
    res = client.search(COLLECTION, data=[query], anns_field="sparse", limit=3,
                        output_fields=["text"])
    for h in res[0]:
        print(f"  {h['distance']:.3f}  {h['entity']['text'][:60]}")

    print("\nDense + BM25 hybrid (RRF):")
    q_dense = embed_texts([query], dim=DIM)[0].tolist()
    req_dense = AnnSearchRequest(data=[q_dense], anns_field="dense",
                                 param={"params": {"ef": 64}}, limit=3)
    req_sparse = AnnSearchRequest(data=[query], anns_field="sparse",
                                  param={}, limit=3)
    res = client.hybrid_search(COLLECTION, reqs=[req_dense, req_sparse],
                               ranker=RRFRanker(60), limit=3, output_fields=["text"])
    for h in res[0]:
        print(f"  {h['entity']['text'][:60]}")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
