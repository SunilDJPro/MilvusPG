# Module 7.1 - Place recognition / loop closure as vector retrieval.

```
Module 7.1 - Place recognition / loop closure as vector retrieval.

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
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partVII_slam/m7_1_dbow_loopclosure/py/demo.py                 # Standalone
python partVII_slam/m7_1_dbow_loopclosure/py/demo.py --lite          # embedded Lite (if supported by this module)
```
