"""Module 1.3 - Connecting & the client model.

The same pymilvus code works against Standalone or embedded Lite; only the URI
changes. This is the portability the course relies on. Run it both ways:

    python demo.py            # Standalone on localhost:19530
    python demo.py --lite     # in-process Milvus Lite (Python-only runtime)

Go and C++ cannot use Lite at all (it is in-process Python); they must connect
to a server. That asymmetry is the lesson reinforced in Module 7.3.
"""
from __future__ import annotations

import argparse

from mvcommon.cli import add_runtime_args, client_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_args(parser)
    args = parser.parse_args()

    runtime = "Milvus Lite (embedded)" if args.lite else "Standalone server"
    client = client_from_args(args)

    print(f"Connected via: {runtime}")
    print("Existing collections:", client.list_collections())
    # A trivial server round-trip that works on both runtimes.
    print("Client is ready. (Lite is Python-only; Go/C++ would need a server.)")


if __name__ == "__main__":
    main()
