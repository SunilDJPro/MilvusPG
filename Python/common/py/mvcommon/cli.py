"""Tiny argparse helper shared by module demos.

Every demo accepts:
    --lite            run against embedded Milvus Lite (a local .db file)
    --uri / --token   override connection for Standalone/remote
By default demos target Standalone on localhost:19530.
"""
from __future__ import annotations

import argparse

from mvcommon import RuntimeKind, get_client


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lite", action="store_true", help="use embedded Milvus Lite")
    parser.add_argument("--uri", default=None, help="Milvus URI or .db path")
    parser.add_argument("--token", default=None, help="auth token, e.g. root:Milvus")
    parser.add_argument("--db-path", default="./milvus_lite_demo.db", help="Lite db file")


def client_from_args(args):
    kind = RuntimeKind.LITE if args.lite else RuntimeKind.STANDALONE
    return get_client(kind, uri=args.uri, token=args.token, db_path=args.db_path)


# Field/feature types the current Milvus Lite (3.0) engine does NOT support.
# These all work on Standalone. Used by modules to skip gracefully on --lite.
LITE_UNSUPPORTED_FEATURES = {
    "BINARY_VECTOR": "Milvus Lite 3.0 supports only FLOAT_VECTOR (no BINARY/sparse).",
    "SPARSE_FLOAT_VECTOR": "Milvus Lite 3.0 has no SPARSE_FLOAT_VECTOR / BM25 function.",
    "BM25": "Full-text BM25 needs Standalone (Lite lacks the sparse output field).",
}


def require_standalone(args, feature: str) -> bool:
    """Return True if the demo can proceed; print a skip note and return False
    when running on Lite and the feature is unsupported there."""
    if getattr(args, "lite", False) and feature in LITE_UNSUPPORTED_FEATURES:
        print(f"[skipped on --lite] {LITE_UNSUPPORTED_FEATURES[feature]}")
        print("Run against Standalone (docker compose up) to execute this module.")
        return False
    return True
