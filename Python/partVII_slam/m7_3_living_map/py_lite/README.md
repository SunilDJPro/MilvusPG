# Module 7.3 - The map as a living vector store (embedded Lite, on-robot).

```
Module 7.3 - The map as a living vector store (embedded Lite, on-robot).

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
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partVII_slam/m7_3_living_map/py_lite/demo.py                 # Standalone
python partVII_slam/m7_3_living_map/py_lite/demo.py --lite          # embedded Lite (if supported by this module)
```
