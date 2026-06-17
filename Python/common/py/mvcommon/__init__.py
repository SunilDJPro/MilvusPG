"""Shared helpers for the Milvus deep-dive Python codebase.

Every module imports from here so the demos stay small and focused on the
concept they teach, not on boilerplate.

Layout:
    connection.py  - get a MilvusClient against Standalone or embedded Lite
    embeddings.py  - deterministic, dependency-free fake embeddings + helpers
    datasets.py    - small synthetic datasets (clusters, binary vectors, docs)
    bench.py       - timing, recall@k, and pretty result printing
"""
from .connection import get_client, RuntimeKind, drop_if_exists
from .embeddings import (
    embed_texts,
    random_unit_vectors,
    random_binary_vectors,
    l2_normalize,
)
from .datasets import (
    gaussian_clusters,
    document_corpus,
    binary_descriptor_keyframes,
)
from .bench import timer, recall_at_k, print_hits, brute_force_knn

__all__ = [
    "get_client",
    "RuntimeKind",
    "drop_if_exists",
    "embed_texts",
    "random_unit_vectors",
    "random_binary_vectors",
    "l2_normalize",
    "gaussian_clusters",
    "document_corpus",
    "binary_descriptor_keyframes",
    "timer",
    "recall_at_k",
    "print_hits",
    "brute_force_knn",
]
