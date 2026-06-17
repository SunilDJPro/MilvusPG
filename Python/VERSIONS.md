# Pinned versions

Bump deliberately; re-run the parity harness after any change.

| Component | Version | Notes |
|---|---|---|
| Milvus server | **2.6.16** | Standalone via `infra/docker-compose.yml` |
| pymilvus | **2.6.14** | `pip install "pymilvus==2.6.14"` |
| milvus-lite | latest 2.6-compatible | pure-Python engine; installed as pymilvus' local backend |
| Go SDK | `client/v2` (Go 1.17+) | *to be built later* |
| C++ SDK | `milvus-sdk-cpp` 2.6.1 | *to be built later* |

This package implements the **Python SDK only**. Go and C++ parity dirs are
intentionally absent and will be added later.

## Milvus Lite vs Standalone (used in this codebase)

| | Standalone (Docker) | Milvus Lite |
|---|---|---|
| Used by | every module | embedded/robotics demos (Part VII) + quickstart |
| SDK | Python/Go/C++/... | **Python only**, in-process |
| Index types | FLAT/IVF*/HNSW*/DiskANN/GPU/binary/sparse | reduced (FAISS-backed segment indexes); **no DiskANN/GPU** |
| Consistency levels | Strong/Bounded/Eventual/Session | single-process semantics |
| Scale | 100M–billions | small (prototyping) |

The current Milvus Lite is rebuilt in pure Python (LSM storage, FAISS segment
indexes) and **does** support BM25 full-text and group-by; older docs that say
otherwise refer to the previous C++/CGo build. Index breadth and scale remain
the real limits versus Standalone.
