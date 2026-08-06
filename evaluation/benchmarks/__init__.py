"""Benchmark registry — each benchmark is a module exporting a ``run`` function.

To add a benchmark, drop a module here with a ``run(bench_cfg, ensure_model,
args, guard_fmt, guard_parse, out_base)`` function and register it below.
"""

from benchmarks.ood_safety import run as run_ood_safety
from benchmarks.groundedness_true import run as run_groundedness_true
from benchmarks.groundedness_aggrefact import run as run_groundedness_aggrefact
from benchmarks.function_calling import run as run_function_calling

BENCHMARKS = {
    "ood_safety": run_ood_safety,
    "groundedness_true": run_groundedness_true,
    "groundedness_aggrefact": run_groundedness_aggrefact,
    "function_calling": run_function_calling,
}

ALL_BENCHMARK_NAMES = list(BENCHMARKS.keys())
