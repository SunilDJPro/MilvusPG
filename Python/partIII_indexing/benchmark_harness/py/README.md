# Benchmark harness — recall/QPS parameter sweep

The measurement spine reused by Parts V/VI/VII/VIII. Sweeps a search parameter
(HNSW `ef` or IVF `nprobe`) and reports recall@10 vs QPS as CSV, so you can draw
a Pareto frontier.

```bash
export PYTHONPATH=common/py
python partIII_indexing/benchmark_harness/py/sweep.py --index HNSW --n 20000
python partIII_indexing/benchmark_harness/py/sweep.py --lite --index IVF_FLAT --n 8000
```

Output is CSV (`knob,recall@10,qps`) — pipe to a file and plot.
