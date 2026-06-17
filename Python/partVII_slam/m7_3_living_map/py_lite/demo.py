"""Module 7.3 - The map as a living vector store (embedded Lite, on-robot).

This is the canonical EMBEDDED/ROBOTICS case, so it defaults to Milvus Lite:
an in-process map with no server, suitable for on-device relocalization. The
map is mutable - insert keyframes as you explore, delete on keyframe culling,
and query for relocalization after tracking loss ("kidnapped robot").

Because Lite is Python-only, the Go/C++ on-device equivalents must instead talk
to a server (see plan Module 7.3, the {go,cpp}_server variants). This file is
the py_lite variant and uses Lite by default.

Run:
    python demo.py                 # embedded Lite (default for this module)
    python demo.py --server        # use Standalone instead (larger maps)
"""
from __future__ import annotations

import argparse

import numpy as np
from pymilvus import DataType

from mvcommon import get_client, RuntimeKind, drop_if_exists, random_unit_vectors

COLLECTION = "m7_3_living_map"
DIM = 128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true",
                        help="use Standalone server instead of embedded Lite")
    parser.add_argument("--db-path", default="./robot_map.db")
    args = parser.parse_args()

    if args.server:
        client = get_client(RuntimeKind.STANDALONE)
        print("[runtime] Standalone server")
    else:
        client = get_client(RuntimeKind.LITE, db_path=args.db_path)
        print("[runtime] embedded Milvus Lite (on-device, Python-only)")

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("keyframe_id", DataType.INT64, is_primary=True)
    schema.add_field("global_desc", DataType.FLOAT_VECTOR, dim=DIM)
    # GPS/odometry prior as scalar fields (native geo types are on the roadmap).
    schema.add_field("x", DataType.FLOAT)
    schema.add_field("y", DataType.FLOAT)
    ip = client.prepare_index_params()
    ip.add_index(field_name="global_desc", index_type="HNSW", metric_type="IP",
                 params={"M": 16, "efConstruction": 200})
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    # Explore: stream 200 keyframes into the live map.
    descs = random_unit_vectors(200, DIM, seed=0)
    positions = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
    for i in range(200):
        client.insert(COLLECTION, [{"keyframe_id": i, "global_desc": descs[i].tolist(),
                                   "x": float(positions[i, 0]), "y": float(positions[i, 1])}])
    client.flush(COLLECTION)
    print("Explored: 200 keyframes in the live map.")

    # Relocalization ("kidnapped"): we grab a frame that is actually keyframe 73
    # with a little noise, and ask the map where we are.
    noisy = descs[73] + np.random.default_rng(1).standard_normal(DIM).astype(np.float32) * 0.05
    noisy /= np.linalg.norm(noisy)
    res = client.search(COLLECTION, data=[noisy.tolist()], anns_field="global_desc",
                        limit=1, output_fields=["keyframe_id", "x", "y"])
    best = res[0][0]
    print(f"Relocalized to keyframe {best['entity']['keyframe_id']} "
          f"(expected ~73) at ({best['entity']['x']:.1f}, {best['entity']['y']:.1f})")

    # Keyframe culling: delete redundant keyframes (even ids in a stretch).
    client.delete(COLLECTION, filter="keyframe_id >= 100 and keyframe_id < 120")
    client.flush(COLLECTION)
    n = client.query(COLLECTION, filter="", output_fields=["count(*)"])[0]["count(*)"]
    print(f"After culling 20 keyframes: {n} remain in the map.")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
