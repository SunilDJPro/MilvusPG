"""Embedding helpers.

These are intentionally dependency-free and deterministic. The course teaches
vector-database mechanics, not embedding-model training, so we use a stable
hashing-based text embedder and seeded random generators. Swap in real models
(sentence-transformers, CLIP, ORB descriptors, ...) where a module calls for it;
the Milvus-facing code does not change.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (so inner product == cosine similarity)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def random_unit_vectors(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    """n random unit-length float32 vectors."""
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal((n, dim))).astype(np.float32)


def random_binary_vectors(n: int, dim_bits: int, *, seed: int = 0) -> list[bytes]:
    """n random binary vectors packed as bytes (for BINARY_VECTOR fields).

    dim_bits must be a multiple of 8. Milvus expects each binary vector as a
    bytes object of length dim_bits // 8.
    """
    if dim_bits % 8 != 0:
        raise ValueError("dim_bits must be a multiple of 8")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, dim_bits), dtype=np.uint8)
    packed = np.packbits(bits, axis=1)
    return [row.tobytes() for row in packed]


def embed_texts(texts: Sequence[str], dim: int = 256) -> np.ndarray:
    """Deterministic hashing embedder.

    Each token is hashed into the vector via the hashing trick, then the doc
    vector is L2-normalized. Same text -> same vector across runs and machines,
    which keeps demos reproducible without downloading a model.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _tokenize(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            idx = h % dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            out[i, idx] += sign
    return l2_normalize(out)


def _tokenize(text: str) -> Iterable[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
