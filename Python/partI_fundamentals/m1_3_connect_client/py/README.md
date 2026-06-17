# Module 1.3 - Connecting & the client model.

```
Module 1.3 - Connecting & the client model.

The same pymilvus code works against Standalone or embedded Lite; only the URI
changes. This is the portability the course relies on. Run it both ways:

    python demo.py            # Standalone on localhost:19530
    python demo.py --lite     # in-process Milvus Lite (Python-only runtime)

Go and C++ cannot use Lite at all (it is in-process Python); they must connect
to a server. That asymmetry is the lesson reinforced in Module 7.3.
```

Run from the repo root:

```bash
export PYTHONPATH=common/py
python partI_fundamentals/m1_3_connect_client/py/demo.py                 # Standalone
python partI_fundamentals/m1_3_connect_client/py/demo.py --lite          # embedded Lite (if supported by this module)
```
