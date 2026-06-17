"""Module 7.1 - Place recognition / loop closure as vector retrieval.

Classical vSLAM loop closure (ORB-SLAM / DBoW2 style): each keyframe is reduced
to a binary descriptor; loop candidates are found by nearest-neighbor search
over those descriptors. Here we model that with a Milvus BINARY_VECTOR field +
HAMMING metric. A geometric-verification stand-in (pose proximity) rejects false
positives caused by perceptual aliasing.

The synthetic dataset injects real loop closures (frames 480-489 revisit the
place seen at frames 10-19), so we can measure detection precision/recall.

Run:
    python demo.py            # Standalone
    python demo.py --lite     # binary + HAMMING works on Lite too
"""
from __future__ import annotations

import argparse

import numpy as np
from pymilvus import DataType

from mvcommon import drop_if_exists, binary_descriptor_keyframes
from mvcommon.cli import add_runtime_args, client_from_args, require_standalone

COLLECTION = "m7_1_loopclosure"
BITS = 256


def hamming_to_similarity(dist, bits=BITS):
    return 1.0 - dist / bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="min descriptor similarity to consider a loop")
    parser.add_argument("--pose-gate", type=float, default=0.5,
                        help="max pose distance for geometric verification")
    args = parser.parse_args()
    if not require_standalone(args, "BINARY_VECTOR"):
        return
    client = client_from_args(args)

    frame_ids, descriptors, poses = binary_descriptor_keyframes(
        n_frames=500, descriptor_bits=BITS, seed=0)

    drop_if_exists(client, COLLECTION)
    schema = client.create_schema(auto_id=False)
    schema.add_field("frame_id", DataType.INT64, is_primary=True)
    schema.add_field("descriptor", DataType.BINARY_VECTOR, dim=BITS)
    ip = client.prepare_index_params()
    ip.add_index(field_name="descriptor", index_type="BIN_FLAT", metric_type="HAMMING")
    client.create_collection(COLLECTION, schema=schema, index_params=ip)

    # Simulate online mapping: insert keyframes as the robot explores, and for
    # each new frame query against PAST frames for loop candidates.
    detected = []  # (query_frame, matched_frame)
    INSERT_EVERY = 1
    min_gap = 50  # ignore temporally-adjacent frames (not real loops)

    for i in frame_ids:
        # Query existing map (frames already inserted) for nearest descriptor.
        if i > min_gap:
            res = client.search(COLLECTION, data=[descriptors[i]],
                                anns_field="descriptor", limit=3,
                                output_fields=["frame_id"])
            for h in res[0]:
                cand = h["frame_id"]
                if i - cand < min_gap:
                    continue  # too close in time to be a loop
                sim = hamming_to_similarity(h["distance"])
                if sim < args.threshold:
                    continue
                # Geometric verification stand-in: poses must be close.
                pose_d = float(np.linalg.norm(poses[i] - poses[cand]))
                if pose_d <= args.pose_gate:
                    detected.append((i, cand))
                    break
        # Insert current frame into the map.
        client.insert(COLLECTION, [{"frame_id": int(i), "descriptor": descriptors[i]}])
        if i % 100 == 0:
            client.flush(COLLECTION)
    client.flush(COLLECTION)

    # Ground truth loops we injected: 480+k -> 10+k for k in 0..9
    truth = {(480 + k, 10 + k) for k in range(10)}
    detected_set = set(detected)
    tp = len(detected_set & truth)
    precision = tp / len(detected_set) if detected_set else 0.0
    recall = tp / len(truth)

    print(f"Detected {len(detected)} loop closures.")
    print(f"  true positives: {tp}/{len(truth)} injected loops")
    print(f"  precision={precision:.2f}  recall={recall:.2f}")
    print("  (tune --threshold and --pose-gate to trace the P/R curve)")

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
