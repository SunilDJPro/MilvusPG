"""Module 6.1 - RAG retrieval (the retrieval half, done properly).

Chunk -> embed -> hybrid retrieve (dense + BM25) -> metadata filter -> assemble
context. No LLM call here (the course is about the vector DB); the output is the
retrieved, ranked context you would hand to a generator.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

from pymilvus import (
    DataType, Function, FunctionType, AnnSearchRequest, RRFRanker,
)

from mvcommon import drop_if_exists, embed_texts
from mvcommon.cli import add_runtime_args, client_from_args, require_standalone

COLLECTION = "m6_1_rag"
DIM = 256

# A tiny "knowledge base" with sources, chunked already for simplicity.
KB = [
    ("Milvus separates compute and storage so query nodes scale independently.", "arch"),
    ("Use HNSW for low-latency search and DiskANN when data exceeds RAM.", "index"),
    ("Consistency level Bounded trades a little freshness for lower latency.", "ops"),
    ("Partition keys isolate tenants without manual partition management.", "modeling"),
    ("Hybrid search fuses dense and BM25 retrieval for better recall.", "search"),
    ("Set metric_type to COSINE for normalized text embeddings.", "index"),
    ("Bulk import is faster than row inserts for large initial loads.", "ops"),
    ("Dynamic fields store schema-free keys in a reserved JSON meta field.", "modeling"),
]


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
    schema.add_field("chunk", DataType.VARCHAR, max_length=1000, enable_analyzer=True)
    schema.add_field("topic", DataType.VARCHAR, max_length=32)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="bm25", input_field_names=["chunk"],
                                 output_field_names=["sparse"],
                                 function_type=FunctionType.BM25))

    ip = client.prepare_index_params()
    ip.add_index(field_name="dense", index_type="HNSW", metric_type="IP",
                 params={"M": 16, "efConstruction": 200})
    ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="BM25", params={"inverted_index_algo": "DAAT_WAND"})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    chunks = [k[0] for k in KB]
    dense = embed_texts(chunks, dim=DIM)
    client.insert(COLLECTION, [{"chunk": chunks[i], "topic": KB[i][1],
                               "dense": dense[i].tolist()} for i in range(len(KB))])
    client.flush(COLLECTION)

    question = "how do I keep search latency low?"
    q_dense = embed_texts([question], dim=DIM)[0].tolist()

    req_dense = AnnSearchRequest(data=[q_dense], anns_field="dense",
                                 param={"params": {"ef": 64}}, limit=4)
    req_sparse = AnnSearchRequest(data=[question], anns_field="sparse",
                                  param={}, limit=4)
    res = client.hybrid_search(COLLECTION, reqs=[req_dense, req_sparse],
                               ranker=RRFRanker(60), limit=3,
                               output_fields=["chunk", "topic"])

    print(f"Q: {question}\n\nRetrieved context (hybrid, top-3):")
    context = []
    for h in res[0]:
        context.append(h["entity"]["chunk"])
        print(f"  [{h['entity']['topic']}] {h['entity']['chunk']}")
    print("\n--- context block to pass to an LLM ---")
    print("\n".join(f"- {c}" for c in context))

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
