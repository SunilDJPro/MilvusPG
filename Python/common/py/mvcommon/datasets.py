"""Small synthetic datasets used across modules.

Kept tiny so every demo runs in seconds on a laptop and against Milvus Lite.
Where a module benefits from a real public dataset (SIFT1M, KITTI, ...), its
README points to a download script under datasets/ instead.
"""
from __future__ import annotations

import numpy as np

from .embeddings import embed_texts, l2_normalize, random_binary_vectors


def gaussian_clusters(
    n_per_cluster: int = 250,
    n_clusters: int = 8,
    dim: int = 64,
    *,
    seed: int = 0,
):
    """Return (vectors[float32], labels[int]) of well-separated blobs.

    Useful for sanity-checking that ANN retrieval returns same-cluster neighbors.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 6.0
    vecs, labels = [], []
    for c in range(n_clusters):
        pts = centers[c] + rng.standard_normal((n_per_cluster, dim))
        vecs.append(pts)
        labels.extend([c] * n_per_cluster)
    vectors = l2_normalize(np.vstack(vecs).astype(np.float32))
    return vectors, np.array(labels, dtype=np.int64)


# A small, self-contained document corpus for RAG / hybrid-search demos.
_DOCS = [
    ("Milvus is an open-source vector database for similarity search and AI.", "overview"),
    ("HNSW is a graph-based index offering high QPS for approximate search.", "index"),
    ("IVF_FLAT partitions vectors into nlist clusters probed by nprobe.", "index"),
    ("DiskANN keeps the index on SSD to serve datasets larger than RAM.", "index"),
    ("BM25 full-text search scores documents by lexical term relevance.", "search"),
    ("Hybrid search fuses dense semantic and sparse keyword retrieval.", "search"),
    ("Consistency levels trade freshness for latency on the read path.", "ops"),
    ("Partitions and partition keys prune the search space for tenants.", "modeling"),
    ("Loop closure in SLAM matches a current frame against past keyframes.", "robotics"),
    ("A vector database can store map keyframes for relocalization.", "robotics"),
    ("Anomaly scores can be derived from k-nearest-neighbor distances.", "research"),
    ("Molecular fingerprints use Jaccard similarity on binary vectors.", "research"),
]


def document_corpus(dim: int = 256):
    """Return dicts with id, text, category, and a dense embedding."""
    texts = [d[0] for d in _DOCS]
    cats = [d[1] for d in _DOCS]
    dense = embed_texts(texts, dim=dim)
    return [
        {"id": i, "text": texts[i], "category": cats[i], "dense": dense[i].tolist()}
        for i in range(len(texts))
    ]


def binary_descriptor_keyframes(
    n_frames: int = 500,
    descriptor_bits: int = 256,
    *,
    seed: int = 0,
):
    """Simulate ORB/BoW-style binary descriptors for the SLAM module.

    Returns (frame_ids, descriptors[bytes], poses[float32 (x,y,theta)]).
    A handful of frames are near-duplicates of earlier ones to mimic loop
    closures (revisiting a place).
    """
    descriptors = random_binary_vectors(n_frames, descriptor_bits, seed=seed)
    rng = np.random.default_rng(seed + 1)
    poses = rng.standard_normal((n_frames, 3)).astype(np.float32)

    # Inject loop closures: make frames 480-489 near-duplicates of 10-19.
    for k in range(10):
        descriptors[480 + k] = descriptors[10 + k]
        poses[480 + k] = poses[10 + k] + rng.standard_normal(3).astype(np.float32) * 0.05

    frame_ids = list(range(n_frames))
    return frame_ids, descriptors, poses
