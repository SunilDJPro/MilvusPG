"""Connection helper: one entry point for both runtime targets.

Standalone (default): talks to the Dockerized server on localhost:19530.
Lite: pass kind="lite" (or set MILVUS_URI to a path ending in .db) and the
client runs an in-process pure-Python engine backed by a local file.

Environment variables:
    MILVUS_URI    e.g. "http://localhost:19530" or "./demo.db"
    MILVUS_TOKEN  e.g. "root:Milvus" (Standalone with auth)
"""
from __future__ import annotations

import enum
import os

from pymilvus import MilvusClient


class RuntimeKind(str, enum.Enum):
    STANDALONE = "standalone"
    LITE = "lite"


def get_client(
    kind: RuntimeKind | str = RuntimeKind.STANDALONE,
    *,
    uri: str | None = None,
    token: str | None = None,
    db_path: str = "./milvus_lite_demo.db",
) -> MilvusClient:
    """Return a MilvusClient for the requested runtime.

    The same pymilvus API is used either way; only the URI differs. This is the
    whole point of Lite -> Standalone portability, so modules are written once
    and run against either backend (subject to Lite's index/scale limits).
    """
    kind = RuntimeKind(kind)
    env_uri = os.environ.get("MILVUS_URI")
    env_token = os.environ.get("MILVUS_TOKEN")

    if kind is RuntimeKind.LITE:
        # A local path makes MilvusClient start Milvus Lite automatically.
        return MilvusClient(uri or env_uri or db_path)

    # Standalone / remote server.
    return MilvusClient(
        uri=uri or env_uri or "http://172.16.16.29:19530", # SMTAIWS - Local IP
        token=token or env_token or "",
    )


def drop_if_exists(client: MilvusClient, name: str) -> None:
    """Idempotent teardown so demos can be re-run cleanly."""
    if client.has_collection(name):
        client.drop_collection(name)
