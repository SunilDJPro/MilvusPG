#!/usr/bin/env bash
# Smoke-test every module against embedded Milvus Lite.
# Standalone-only modules (binary vectors, BM25) will print a skip note.
set -u
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/common/py"
DB="$(mktemp -d)/smoke.db"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run() {
  echo "=== $1 ==="
  python "$1" --lite --db-path "$DB" 2>&1 | grep -v "WARNING clustering" | tail -4
  echo
}

run partI_fundamentals/m1_1_why_vectordb/py/demo.py
run partI_fundamentals/m1_2_distance_metrics/py/demo.py
run partI_fundamentals/m1_3_connect_client/py/demo.py
run partII_modeling/m2_1_collections_schema/py/demo.py
run partII_modeling/m2_2_partitions/py/demo.py
run partII_modeling/m2_3_insert_upsert_delete/py/demo.py
run partIII_indexing/m3_1_index_zoo/py/demo.py
run partIII_indexing/m3_3_consistency/py/demo.py
run partIII_indexing/m3_4_filtered_search/py/demo.py
run partIV_retrieval/m4_1_hybrid_search/py/demo.py
run partIV_retrieval/m4_2_full_text_bm25/py/demo.py
run partIV_retrieval/m4_4_iterators_grouping/py/demo.py
run partV_ops/m5_2_segments_compaction/py/demo.py
run partVI_rag_recsys/m6_1_rag/py/demo.py
run partVI_rag_recsys/m6_2_recsys_multimodal/py/demo.py
run partVII_slam/m7_1_dbow_loopclosure/py/demo.py
run partVIII_research/m8_1_anomaly/py/demo.py
run partVIII_research/m8_2_scientific_embeddings/py/demo.py
echo "=== partVII_slam/m7_3_living_map/py_lite/demo.py ==="
python partVII_slam/m7_3_living_map/py_lite/demo.py --db-path "$DB" 2>&1 | tail -4
