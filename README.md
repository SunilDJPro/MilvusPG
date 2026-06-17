# Milvus Vector Database — Deep Dive Course & Codebase Build Plan

> **Status:** Planning reference for building demo + educational modules.
> **Stack baseline (verified June 2026):** Milvus server **2.6.16** (stable), preview **3.0.0-beta**; `pymilvus` **2.6.14**; Go SDK **`github.com/milvus-io/milvus/client/v2`** (Go 1.17+); C++ SDK **`milvus-sdk-cpp` 2.6.1** (build-from-source, C++17, gRPC). Indexing engine: **Knowhere**.
> **Runtime targets:** **(a) Standalone via Docker** — the primary target for all modules (single `docker-compose` with Milvus + etcd + MinIO). **(b) Milvus Lite** — embedded in-process target for the embedded/robotics modules, with explicit limitation callouts vs Standalone (see §0.1).
> **Language policy:** **Full parity across three SDKs** — every module ships Python (`pymilvus`), Go (`client/v2`), and C++ (`milvus-sdk-cpp`) implementations of the same demo, **subject to the SDK capability matrix in §0.2** (not every feature exists in every SDK yet; gaps are documented, not faked).
> **Audience:** **Tiered tracks** — see below.

---

## 0. How to read this plan

Each module is written so a developer can pick it up and start coding immediately. Every module entry contains:

- **Concept** — what's being taught and why it matters.
- **Milvus surface** — the specific APIs / index types / params involved.
- **Demo spec** — what the Python + Go demos should actually do (parity required).
- **Datasets** — what to feed it.## Scope note

A "multimodal intelligence for mass surveillance" track was requested and is **deliberately excluded**. The remaining research domains (SLAM/vSLAM, recsys/multimodal, anomaly detection, scientific embeddings) are covered in full above.
- **Exercises** — graded tasks for learners.

### Tiered tracks

The same module set serves three audiences via depth gating:

| Track | Who | Path |
|---|---|---|
| **T1 — Foundations** | Engineers new to vector DBs | Parts I–III + one applied capstone |
| **T2 — Applied** | Experienced ML/backend engineers | Parts I–VI, all RAG/recsys/anomaly applied parts |
| **T3 — Research** | Researchers (SLAM/robotics/CV, scientific embeddings) | Everything, with emphasis on Parts IV, VII, VIII |

Module headers below are tagged `[T1]`, `[T2]`, `[T3]` to show the minimum track that needs them.

### 0.1 Runtime targets: Standalone vs Milvus Lite

Two runtimes are used deliberately, and the course teaches *when to choose which* — this is itself a learning objective, especially for the robotics track.

**Standalone (Docker)** — the primary target for nearly every module. Full server: all index types, all consistency levels, the segment/compaction lifecycle, observability, multi-tenancy. This is what production and benchmarking modules run against.

**Milvus Lite** — an embedded, in-process build that runs inside a single Python process with a local file as storage. It exists for laptops, notebooks, CI, edge devices, and **on-robot embedded use** where running the full server stack is impractical. It speaks the same client API and the same data file can later be loaded by a server, which makes it ideal for an offline-build / on-device-query workflow.

**Limitations of Lite vs Standalone — teach these explicitly, do not paper over them:**

| Dimension | Standalone | Milvus Lite |
|---|---|---|
| **SDK language** | Python, Go, C++, Java, Node | **Python only** (in-process). Go/C++ on-device must talk to a *server* — Lite is not an option for them. |
| **OS / arch** | Linux, macOS, Windows (Docker) | Linux & macOS; **no native Windows**; ARM supported (relevant for Jetson/Pi-class robots, but verify per release). |
| **Scale** | 100M–billions, sharded | Small scale — roughly up to a few million vectors; single process, no horizontal scaling. |
| **Index types** | FLAT, IVF\*, HNSW\*, **DiskANN**, GPU (CAGRA/GPU_IVF\*), binary, sparse | Reduced set; **no DiskANN, no GPU indexes**; effectively in-memory (FLAT/HNSW-class). Verify the exact supported set per Lite release before recording demos. |
| **Distributed features** | Compaction tiers, resource groups, replicas, RBAC, bulk import | Not applicable / absent. |
| **Consistency** | Strong/Bounded/Eventual/Session | Single-process semantics; the multi-level consistency demos (Module 3.3) are **Standalone-only**. |
| **Concurrency** | High QPS, many clients | Single embedded process; not a serving tier. |

**Consequence for the robotics track (Part VII):** Milvus Lite is the right tool for *on-robot relocalization and small local maps*, but because **it is Python-only**, the Go and C++ on-device demos cannot use Lite — they connect to a Standalone (or remote) server instead. The course turns this into a designed exercise: build the vocabulary/map **offline on Standalone**, export, then **query on-device** — Python via embedded Lite, Go/C++ via a lightweight server endpoint. Each robotics module states which runtime each language uses.

### 0.2 SDK capability matrix (parity policy)

Parity is the goal, but the three SDKs are not equally mature. The **Python SDK is the reference**; Go is close; **C++ (`milvus-sdk-cpp` 2.6.1) is a thinner, build-from-source gRPC client** that trails on higher-level conveniences. The rule: implement every module in all three where the capability exists, and where it doesn't, **document the gap in the module README and provide a raw-gRPC or server-side workaround rather than pretending parity.**

| Capability area | Python | Go | C++ (2.6.1) | Notes |
|---|---|---|---|---|
| Connect / collections / CRUD | ✅ | ✅ | ✅ | C++ added `CreateSimpleCollectionRequest`, `Get()` in 2.6.1. |
| Index create / search / range search | ✅ | ✅ | ✅ | C++ now passes `radius`/`range_filter` directly; server-side validation. |
| Multi-vector / hybrid search | ✅ | ✅ | ✅ (2.6.1) | C++ gained multi-vector `SearchRequest`/`SubSearchRequest`; verify reranker coverage. |
| Sparse / BM25 full-text | ✅ | ✅ | ⚠️ partial | Confirm sparse-vector `AddSparseVector` fixes per C++ release. |
| Iterators / group-by | ✅ | ✅ | ⚠️ verify | May require manual paging in C++. |
| Bulk import | ✅ | ✅ | ⚠️ limited | Prefer server-side / REST bulk path for C++ demos. |
| Embedded (Lite) | ✅ | ❌ | ❌ | Lite is in-process Python only. |
| Built-in embedding/model utils (`pymilvus.model`) | ✅ | ❌ | ❌ | Generate embeddings out-of-band for Go/C++; share via the common harness. |

> **Build note for C++:** `milvus-sdk-cpp` compiles from source (CMake, C++17, links `libmilvus_sdk`), so the `infra/` deliverables must include a reproducible C++ toolchain image. Pin SDK **2.6.1 ↔ server 2.6.x** per the project's compatibility table.

---

## Part I — Vector Search Fundamentals `[T1]`

### Module 1.1 — Why vector databases exist
- **Concept:** Embeddings as the universal representation; the difference between exact KNN and approximate NN (ANN); the recall/latency/cost triangle. Where a vector DB beats a flat numpy index, a FAISS process, and a relational DB with a vector column.
- **Milvus surface:** Positioning only — compute/storage disaggregation, the four-layer architecture (access → coordinator → worker → storage).
- **Demo spec:** Brute-force KNN in pure Python/Go over 10k vectors, then the same query against Milvus FLAT. Compare results (should be identical) and latency as N grows.
- **Datasets:** Synthetic Gaussian clusters; SIFT1M subset.
- **Exercises:** Plot query latency vs N for brute force vs Milvus; identify the crossover point.

### Module 1.2 — Distance metrics & embedding geometry
- **Concept:** L2, Inner Product (IP), COSINE; when each is correct; normalization and why IP-on-normalized == cosine. Binary metrics: **JACCARD**, **HAMMING**. Sparse/text metric: **BM25**.
- **Milvus surface:** `metric_type` selection; consequences of metric/index mismatch.
- **Demo spec:** Same dataset embedded, searched under each metric; show how ranking changes. Include a binary-vector demo (Hamming) to set up the SLAM module later.
- **Exercises:** Predict the ranking flip when switching L2→IP on un-normalized vectors, then verify.

### Module 1.3 — Connecting & the client model `[T1]`
- **Concept:** `MilvusClient` (recommended) vs the legacy ORM module in pymilvus; the Go `client/v2` and C++ `milvus::MilvusClient` equivalents. Auth tokens, databases, multi-tenancy entry point. First contact with the Lite-vs-Standalone choice (Python connects to either; Go/C++ connect to a server).
- **Demo spec:** Identical "connect, ping, list collections" in Python, Go, and C++ against the Docker standalone, plus a Python-only "connect to embedded Lite" variant. Establish the **parity harness** here — a shared test that runs the same assertions against all three SDKs and flags any documented capability gap from §0.2 instead of failing silently.
- **Deliverable:** `infra/docker-compose.yml` (Milvus 2.6.16 + etcd + MinIO) and a `infra/cpp-toolchain.Dockerfile` (CMake/C++17 build for `milvus-sdk-cpp` 2.6.1), reused by every later module.

---

## Part II — Data Modeling in Milvus `[T1]`

### Module 2.1 — Collections, schemas, fields
- **Concept:** Collection = table; fields, primary keys (auto vs manual), the vector field, scalar fields. **Dynamic field** for schema-flexible payloads. Nullable & default values.
- **Milvus surface:** `create_collection`, `CollectionSchema`, `FieldSchema`, `DataType` (FLOAT_VECTOR, BINARY_VECTOR, FLOAT16/BFLOAT16_VECTOR, SPARSE_FLOAT_VECTOR, plus scalars including JSON and ARRAY).
- **Demo spec:** Define a product-catalog collection with vector + scalar + JSON fields, in both SDKs.
- **Exercises:** Add a dynamic field at insert time and query it.

### Module 2.2 — Partitions & partition keys
- **Concept:** Partitions for data pruning; **partition key** for automatic tenant/category isolation without manual partition management.
- **Demo spec:** Same dataset, one collection partitioned by category; show search-scope reduction.
- **Exercises:** Measure latency with and without a partition-key filter.

### Module 2.3 — Insert, upsert, delete & primary-key semantics
- **Concept:** Write path basics; upsert behavior; delete by PK and by expression; the consistency implications of deletes (tombstones, L0 compaction).
- **Demo spec:** Batch insert 1M rows; upsert a slice; delete by filter; verify counts.
- **Exercises:** Demonstrate a "silent" stale-read window and then force consistency (see 3.3).

---

## Part III — Indexing & Search Internals `[T2]`

### Module 3.1 — The index zoo
- **Concept:** Index families and their tradeoffs. This is the conceptual core of the course.
  - **FLAT** — exact, 100% recall, baseline.
  - **IVF_FLAT / IVF_SQ8 / IVF_PQ** — inverted-file clustering (`nlist`/`nprobe`); SQ8 and PQ quantization for memory compression; PQ vs SQ recall/speed tradeoff.
  - **HNSW** (and HNSW_SQ/PQ/PRQ variants) — graph-based; `M`, `efConstruction`, `ef`; generally higher QPS than IVF.
  - **DiskANN** — disk-resident for datasets larger than RAM; IOPS as the bottleneck; GPU build path via cuVS in recent Knowhere.
  - **GPU indexes** (CAGRA, GPU_IVF_*) — when GPU pays off.
  - **Binary indexes** (BIN_FLAT, BIN_IVF_FLAT) — for Hamming/Jaccard; needed for the SLAM track.
  - **Sparse indexes** — SPARSE_INVERTED_INDEX; note `SPARSE_WAND` deprecated since 2.5.4 → use `inverted_index_algo: DAAT_WAND`.
- **Milvus surface:** `create_index`, `IndexParams`, the tiered search model (coarse filter → quantized compute → refinement re-rank).
- **Demo spec:** A **benchmark harness** that builds the same dataset under FLAT / IVF_FLAT / IVF_PQ / HNSW / DiskANN and plots recall@k vs QPS vs memory. Parity: identical harness in Go.
- **Datasets:** SIFT1M (128-d), GIST1M (960-d), a 768-d text-embedding set.
- **Exercises:** Hit a target recall@10 ≥ 0.95 at minimum memory; justify the index choice.

### Module 3.2 — Tuning & the recall/latency/cost triangle
- **Concept:** Memory math (the index-size estimation Milvus documents for IVF and HNSW); build-time vs QPS vs recall; when large-topK favors IVF over graph indexes.
- **Demo spec:** Parameter sweep tool (`nlist`, `nprobe`, `M`, `ef`) emitting a Pareto frontier.
- **Exercises:** Produce a one-page "index decision tree" from your own measurements.

### Module 3.3 — Consistency levels & the read path `[T2]`
- **Concept:** Strong / Bounded / Eventual / Session consistency; the streaming write path (log broker → async index build by data nodes → query-node load); why fresh writes may not be immediately searchable.
- **Demo spec:** Insert-then-search race demonstrated at each consistency level.
- **Exercises:** Choose a consistency level for (a) a chat memory store, (b) an offline analytics batch.

### Module 3.4 — Filtered (hybrid scalar+vector) search
- **Concept:** Boolean expression filters; scalar indexes (inverted, bitmap, Marisa-trie for strings); pre- vs post-filtering and the delegator-side pruning Milvus does.
- **Demo spec:** "Vector search within price < X and brand IN (...)" in both SDKs.
- **Exercises:** Show a query where a scalar index changes latency by an order of magnitude.

---

## Part IV — Advanced Retrieval & Multi-Vector `[T2]/[T3]`

### Module 4.1 — Hybrid search & fusion
- **Concept:** Multiple vector fields per entity (e.g., dense + sparse); running several ANN searches and fusing with **RRF** or **weighted** rerankers.
- **Milvus surface:** `hybrid_search`, `AnnSearchRequest`, `RRFRanker`, `WeightedRanker`.
- **Demo spec:** Dense (semantic) + sparse (keyword/BM25) hybrid over a document set; compare fused vs single-field ranking.

### Module 4.2 — Sparse vectors & full-text/BM25
- **Concept:** Sparse embeddings, learned sparse (SPLADE-style), and Milvus native BM25 full-text; the inverted index and DAAT_WAND.
- **Demo spec:** Full-text + dense hybrid; show recall gains on keyword-heavy queries.

### Module 4.3 — Multi-vector & late-interaction (ColBERT-style) `[T3]`
- **Concept:** Per-token / per-patch vector sets and late interaction. Note the **2.6→3.0 roadmap**: a unified Tensor / StructList type targeting ColBERT, ColQwen, video, and multimodal vectors — design demos to migrate cleanly.
- **Demo spec:** Approximate a late-interaction reranker on top of candidate retrieval; flag which parts are first-class vs application-side today.

### Module 4.4 — Search iterators, range search, grouping
- **Concept:** Range search (radius/`range_filter`); pagination via iterators; `group_by_field` for diversity/dedup; the iterator scheduling optimizations in recent builds.
- **Demo spec:** "One result per author" grouped search; deep pagination with an iterator.

---

## Part V — Operations, Scale & Production `[T2]`

### Module 5.1 — Architecture deep dive
- **Concept:** The microservice topology — stateless **Proxy**, **Coordinators**, **QueryNode** (CPU/memory-hot), **DataNode** (compaction + index build), and storage (etcd meta, log broker, MinIO/S3 object store). LSM-style decoupling of ingest from indexing.
- **Deliverable:** An annotated architecture diagram learners reproduce by tracing a write and a read through logs.

### Module 5.2 — Segments, compaction, loading
- **Concept:** Growing vs sealed segments; flush; L0/clustering/mix compaction; load/release and replica count; resource groups.
- **Demo spec:** Force flush + compaction; observe segment lifecycle via metrics.

### Module 5.3 — Bulk import & data pipelines
- **Concept:** Bulk import path vs row inserts; offline index building; backup/restore (`milvus-backup`); migration.
- **Demo spec:** Import 10M vectors via the bulk path; compare throughput to row insert.

### Module 5.4 — Observability, security, multi-tenancy
- **Concept:** Prometheus/Grafana metrics, the key latency contributors (proxy overhead vs broker lag vs query-node compute); RBAC & auth; database/partition-key tenancy models.
- **Demo spec:** Stand up the metrics stack against the Docker standalone; build a small Grafana panel for QPS and p99.
- **Exercises:** Diagnose an injected bottleneck from metrics alone.

### Module 5.5 — Benchmarking methodology
- **Concept:** How to benchmark honestly (warmup, recall ground-truth, concurrency, percentiles). VectorDBBench-style methodology.
- **Deliverable:** A reusable load-gen tool (parity Python/Go) feeding the Part III harness.

---

## Part VI — Applied: RAG & Recommendation `[T2]`

### Module 6.1 — RAG done properly
- **Concept:** Chunking, embedding choice, hybrid retrieval, reranking, metadata filtering, and freshness/consistency for a live knowledge base. Explicitly framed as **one** use case, not the point of the course.
- **Demo spec:** End-to-end RAG over a document corpus using hybrid search (4.1) + grouping (4.4); evaluation with retrieval metrics.

### Module 6.2 — Recommendation & multimodal search `[T2]`
- **Concept:** Item/user embeddings, ANN candidate generation, filtering by business rules, freshness via upsert; image+text multimodal retrieval (CLIP-style joint space).
- **Demo spec:** A two-tower-style recsys candidate generator over a product/interaction dataset; a multimodal "search by image or text" demo.
- **Datasets:** A public products dataset; an open image-caption set.
- **Exercises:** Add a "more like this but cheaper" filtered-vector query; measure catalog coverage.

---

## Part VII — Research Track: Robotics, SLAM & vSLAM `[T3]`

> This is the part that most distinguishes the course. The throughline: **place recognition / loop closure is a vector-retrieval problem**, and Milvus can serve as the searchable map/keyframe database.

### Module 7.1 — Place recognition as vector retrieval
- **Concept:** Bag-of-Visual-Words (DBoW2-style) loop closure used in ORB-SLAM/VINS-Mono: ORB/BRIEF binary descriptors → visual vocabulary → per-keyframe BoW vector → inverted-index retrieval of loop candidates → geometric verification. Map this classical pipeline onto Milvus primitives (binary vectors + Hamming + BIN_IVF + scalar metadata for geometric-check inputs).
- **Demo spec:** Build a visual vocabulary offline; ingest keyframe descriptors into Milvus as binary vectors; query the current frame for top-k loop candidates; run a RANSAC/fundamental-matrix geometric check application-side to reject false positives (perceptual aliasing).
- **Datasets:** KITTI odometry sequences; an indoor handheld sequence.
- **Exercises:** Plot precision/recall of loop detection vs similarity threshold `s_min`.

### Module 7.2 — Learned place recognition (modern VPR)
- **Concept:** Moving from handcrafted BoW to learned global descriptors (NetVLAD/AnyLoc-style) for robustness to illumination/viewpoint change. Dense float vectors + HNSW/IVF replace the binary BoW path.
- **Demo spec:** Same KITTI sequences, learned descriptors in Milvus; compare loop-closure precision/recall to the DBoW baseline from 7.1.
- **Exercises:** Quantify the robustness gain under day/night or seasonal shift.

### Module 7.3 — The map as a living vector store
- **Concept:** Treating the keyframe/map database as an online, mutable Milvus collection: incremental insert as the robot explores, deletion on keyframe culling, consistency choices for real-time querying, relocalization after tracking loss as a search query. **Runtime split is the lesson here:** an on-robot embedded deployment uses **Milvus Lite** (Python, in-process, no server, no DiskANN/GPU — small in-memory map) while the offline vocabulary/map build and any fleet-scale map server use **Standalone**. Because Lite is Python-only, the Go and C++ on-device demos query a local or remote *server* instead — the module makes this asymmetry explicit and has learners reason about the tradeoff (binary size, footprint, no external process vs. richer index types and multi-language access).
- **Demo spec:** Stream keyframes into Milvus during a simulated traversal; trigger relocalization from a "kidnapped robot" query; demonstrate culling via delete + compaction. Provide two runtime variants of the on-device path: **(a) Python + embedded Lite** (the canonical embedded/robotics case), **(b) Go/C++ + Standalone server** (when non-Python on-device or larger maps/DiskANN are required). Note where Lite's limitations (index set, scale, single process) force a design change.
- **Exercises:** Measure end-to-end relocalization latency budget and pick a consistency level (Standalone path); measure Lite's memory ceiling and identify the map size at which you must graduate from Lite to a server.

### Module 7.4 — Extended mapping & recovery, multi-session
- **Concept:** Multi-session / lifelong mapping: merging maps across runs, deduplicating revisited places, map versioning. Partition-per-session modeling; cross-session loop closure as cross-partition search.
- **Demo spec:** Two overlapping sessions ingested into partitioned collections; detect cross-session overlaps; merge.
- **Exercises:** Design a schema that supports rolling back a corrupted session.

> **Note on geo/spatial:** native Geo data types (points, regions, spatial indexing) are on the **3.0 roadmap**. Until then, GPS/odometry priors are modeled as scalar fields used to pre-filter vector candidates. Demos should isolate this so they can adopt native geo later.

---

## Part VIII — Research Track: Custom Intelligence & Scientific Embeddings `[T3]`

### Module 8.1 — Anomaly detection & time-series
- **Concept:** Embedding windows/segments of signals; nearest-neighbor distance as an anomaly score; novelty detection via low-density regions; drift monitoring.
- **Demo spec:** Sliding-window embeddings of a sensor/log stream in Milvus; kNN-distance anomaly scoring with range search; alert on density collapse.
- **Datasets:** A public multivariate sensor or server-metrics dataset.
- **Exercises:** Tune the kNN-distance threshold for a target false-positive rate.

### Module 8.2 — Scientific / bio / geo embeddings
- **Concept:** Domain embeddings beyond NLP — protein/sequence embeddings, molecular fingerprints (binary → Jaccard/Tanimoto), geospatial feature vectors. Metric selection per domain (e.g., Tanimoto similarity for chemistry as Jaccard on binary fingerprints).
- **Demo spec:** A molecule similarity search over fingerprint vectors (binary + Jaccard) and a protein-embedding nearest-neighbor search (dense + IP/cosine).
- **Datasets:** An open chemistry fingerprint set; an open protein-embedding set.
- **Exercises:** Show why metric choice (Jaccard vs L2) is correctness-critical here.

### Module 8.3 — Clustering, dedup & dataset curation
- **Concept:** Using ANN for large-scale near-duplicate detection, semantic dedup of training corpora, and offline clustering/dimensionality-reduction workflows (aligned with the Spark-offline roadmap items).
- **Demo spec:** Near-dup detection over an image or text corpus; report dedup rate and false merges.

### Module 8.4 — Custom intelligence capstone `[T3]`
- **Concept:** Learner-chosen research problem framed as vector retrieval. Provide 2–3 scaffolds (e.g., signal-based event retrieval, embedding-based retrieval-augmented control, cross-modal scientific search).
- **Deliverable:** A self-contained Python+Go demo + a short write-up justifying schema, index, metric, and consistency choices.

---

## Part IX — Capstones & Assessment

| Track | Capstone |
|---|---|
| **T1** | Build a filtered semantic search service over a chosen dataset (schema → index tuning → hybrid filter → benchmark report). |
| **T2** | Production RAG **or** recsys service: hybrid retrieval, observability dashboard, load test to a stated SLA. |
| **T3** | Either the SLAM loop-closure map service (Part VII) **or** the custom-intelligence capstone (8.4), with a research-style evaluation. |

---

## Repository layout (proposed)

```
milvus-deepdive/
├─ infra/
│  ├─ docker-compose.yml          # Milvus 2.6.16 + etcd + MinIO (Standalone)
│  ├─ cpp-toolchain.Dockerfile    # CMake/C++17 build env for milvus-sdk-cpp 2.6.1
│  └─ observability/              # Prometheus + Grafana stack
├─ common/
│  ├─ py/                         # shared pymilvus helpers, Lite bootstrap, parity harness
│  ├─ go/                         # shared client/v2 helpers, parity harness
│  └─ cpp/                        # shared milvus-sdk-cpp helpers, CMake modules, parity harness
├─ datasets/                      # download scripts + checksums (no raw data in repo)
├─ partI_fundamentals/
│  ├─ m1_1_why_vectordb/{py,go,cpp}
│  └─ ...
├─ partII_modeling/...
├─ partIII_indexing/
│  └─ benchmark_harness/{py,go,cpp} # recall/QPS/memory Pareto tool
├─ partIV_retrieval/...
├─ partV_ops/...
├─ partVI_rag_recsys/...
├─ partVII_slam/
│  ├─ m7_1_dbow_loopclosure/{py,go,cpp}
│  └─ m7_3_living_map/
│     ├─ py_lite/                 # embedded Milvus Lite on-device variant
│     └─ {py,go,cpp}_server/      # Standalone-server variant
├─ partVIII_research/...
└─ capstones/
```

**Parity rule:** every leaf module has `py/`, `go/`, and `cpp/` subdirs implementing the same demo, validated by a shared parity test in `common/`. Where §0.2 documents a capability gap (or where a runtime is language-restricted, e.g. Lite is Python-only), the affected SDK dir contains a short `GAP.md` explaining the limitation and the workaround used — a documented gap counts as "done," a silent omission does not.

---

## Build sequencing (suggested)

1. **Infra first:** `infra/docker-compose.yml` + the parity harness (Module 1.3). Nothing else proceeds until both SDKs connect and assert against the same standalone.
2. **Vertical slice:** Build Part III's benchmark harness early — it's the spine that Parts V, VI, VII, VIII all reuse for measurement.
3. **Foundations (I–II)** as the teaching on-ramp.
4. **Applied (VI)** to validate the stack end-to-end on a familiar use case.
5. **Research tracks (VII, VIII)** last, since they build on indexing internals (binary indexes, metrics, consistency) and the harness.

---

## Version & maintenance notes

- Pin `milvus` server, `pymilvus`, Go `client/v2`, and `milvus-sdk-cpp` versions in `infra/` and a top-level `VERSIONS.md`; the SDK/server matrix moves fast and the **C++ SDK trails** (2.6.1 ↔ server 2.6.x) more than Python/Go.
- Re-verify the **Milvus Lite supported-index set and ARM/Jetson support** each Lite release before recording robotics demos — the in-memory-only index limitation and Python-only constraint are the load-bearing facts for Part VII.
- Track three roadmap items that will reshape modules when they land: **unified Tensor/StructList** type (multi-vector/ColBERT/video — affects 4.3), **native Geo types + spatial indexing** (affects Part VII pre-filtering), and **Vector Lake / hot-cold tiering** (affects Part V ops).
- Re-verify deprecations each release (e.g., `SPARSE_WAND` → `DAAT_WAND`; `drop_ratio_build` no-op since 2.5.4) before recording demos.

---
