"""Timing, recall, and result-printing utilities."""
from __future__ import annotations

import contextlib
import time
from typing import Iterable, Sequence

import numpy as np


@contextlib.contextmanager
def timer(label: str = "elapsed"):
    """Context manager that prints wall-clock time for a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000.0
        print(f"  [{label}] {ms:.1f} ms")


def recall_at_k(approx_ids: Sequence[int], truth_ids: Sequence[int], k: int) -> float:
    """Fraction of the true top-k that the approximate result recovered."""
    truth = set(list(truth_ids)[:k])
    if not truth:
        return 1.0
    approx = set(list(approx_ids)[:k])
    return len(truth & approx) / len(truth)


def brute_force_knn(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    *,
    metric: str = "IP",
):
    """Exact KNN baseline (for recall ground truth and Module 1.1).

    metric: "IP" (inner product, larger=closer) or "L2" (smaller=closer).
    Returns (ids, scores) of the top-k.
    """
    if metric == "IP":
        scores = vectors @ query
        order = np.argsort(-scores)[:k]
    elif metric == "L2":
        diffs = vectors - query
        scores = np.einsum("ij,ij->i", diffs, diffs)
        order = np.argsort(scores)[:k]
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return order.tolist(), scores[order].tolist()


def print_hits(results, *, fields: Iterable[str] = ("text",)) -> None:
    """Pretty-print MilvusClient.search results (a list of per-query hit lists)."""
    fields = list(fields)
    for qi, hits in enumerate(results):
        print(f"  query {qi}:")
        for rank, hit in enumerate(hits):
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            shown = {f: entity.get(f) for f in fields if f in entity}
            dist = hit.get("distance") if isinstance(hit, dict) else None
            print(f"    {rank + 1:>2}. id={hit.get('id')} dist={dist:.4f} {shown}")
