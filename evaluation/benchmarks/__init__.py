"""Benchmark registry - each benchmark is a module exporting a ``run`` function.

To add a benchmark, drop a module here with a ``run(bench_cfg, ensure_model,
args, guard_fmt, guard_parse, out_base)`` function and register it below.
A benchmark whose ``run.TASK`` is ``"requirement_check"`` runs only on
requirement-checking models (``TASK`` in the model module); the rest are
guardian benchmarks.
"""

from benchmarks.ood_safety import run as run_ood_safety
from benchmarks.groundedness_true import run as run_groundedness_true
from benchmarks.groundedness_aggrefact import run as run_groundedness_aggrefact
from benchmarks.function_calling import run as run_function_calling
from benchmarks.ifeval_multiconstraint import run as run_ifeval_multiconstraint
from benchmarks.infobench import run_gpt4 as run_infobench_gpt4
from benchmarks.infobench import run_human as run_infobench_human

BENCHMARKS = {
    "ood_safety": run_ood_safety,
    "groundedness_true": run_groundedness_true,
    "groundedness_aggrefact": run_groundedness_aggrefact,
    "function_calling": run_function_calling,
    "ifeval_multiconstraint": run_ifeval_multiconstraint,
    "infobench_gpt4": run_infobench_gpt4,
    "infobench_human": run_infobench_human,
}

ALL_BENCHMARK_NAMES = list(BENCHMARKS.keys())
