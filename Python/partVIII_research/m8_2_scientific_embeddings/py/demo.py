"""Module 8.2 - Scientific / bio / geo embeddings.

Two domains where metric choice is correctness-critical:
  1. Molecular fingerprints as BINARY_VECTOR with JACCARD (== Tanimoto), the
     standard chemical-similarity metric.
  2. Protein/sequence embeddings as dense FLOAT_VECTOR with COSINE.

The point: the same Milvus mechanics serve very different sciences, and picking
the wrong metric gives wrong answers.

Run:
    python demo.py [--lite]
"""
from __future__ import annotations

import argparse

import numpy as np
from pymilvus import DataType

from mvcommon import (
    drop_if_exists, random_binary_vectors, random_unit_vectors,
)
from mvcommon.cli import add_runtime_args, client_from_args, require_standalone

FP_BITS = 512   # fingerprint length
PROT_DIM = 320  # protein embedding dim


def fingerprint_demo(client):
    name = "m8_2_molecules"
    drop_if_exists(client, name)
    schema = client.create_schema(auto_id=False)
    schema.add_field("mol_id", DataType.INT64, is_primary=True)
    schema.add_field("fingerprint", DataType.BINARY_VECTOR, dim=FP_BITS)
    ip = client.prepare_index_params()
    # JACCARD on binary fingerprints == Tanimoto similarity in cheminformatics.
    ip.add_index(field_name="fingerprint", index_type="BIN_IVF_FLAT",
                 metric_type="JACCARD", params={"nlist": 64})
    client.create_collection(name, schema=schema, index_params=ip)

    fps = random_binary_vectors(2000, FP_BITS, seed=0)
    # Make molecule 1 a near-analog of molecule 0 (share most bits).
    a = np.frombuffer(fps[0], dtype=np.uint8).copy()
    b = a.copy()
    b[-2:] = ~b[-2:] & 0xFF  # flip a few bits
    fps[1] = b.tobytes()

    client.insert(name, [{"mol_id": i, "fingerprint": fps[i]} for i in range(len(fps))])
    client.flush(name)

    res = client.search(name, data=[fps[0]], anns_field="fingerprint", limit=3,
                        search_params={"params": {"nprobe": 16}},
                        output_fields=["mol_id"])
    print("Molecular similarity (JACCARD/Tanimoto), query = molecule 0:")
    for h in res[0]:
        print(f"  mol {h['entity']['mol_id']}  jaccard_dist={h['distance']:.4f}")
    print("  (molecule 1 was built as a near-analog -> should rank near top)")
    client.drop_collection(name)


def protein_demo(client):
    name = "m8_2_proteins"
    drop_if_exists(client, name)
    schema = client.create_schema(auto_id=False)
    schema.add_field("prot_id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=PROT_DIM)
    ip = client.prepare_index_params()
    ip.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 200})
    client.create_collection(name, schema=schema, index_params=ip)

    emb = random_unit_vectors(1500, PROT_DIM, seed=2)
    client.insert(name, [{"prot_id": i, "embedding": emb[i].tolist()}
                        for i in range(len(emb))])
    client.flush(name)

    res = client.search(name, data=[emb[42].tolist()], anns_field="embedding",
                        limit=3, search_params={"params": {"ef": 64}},
                        output_fields=["prot_id"])
    print("\nProtein embedding nearest neighbors (COSINE), query = protein 42:")
    for h in res[0]:
        print(f"  protein {h['entity']['prot_id']}  cosine_sim={h['distance']:.4f}")
    client.drop_collection(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()
    client = client_from_args(args)
    if require_standalone(args, "BINARY_VECTOR"):
        fingerprint_demo(client)
    protein_demo(client)


if __name__ == "__main__":
    main()
