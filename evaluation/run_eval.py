"""Granite Guardian evaluation CLI.

Reproduces the benchmark numbers reported on the Granite Guardian model cards
(3.0 through 4.1). Loads a model with vLLM, runs one or more benchmarks, and
writes per-dataset metrics plus an aggregate to the output directory.

Example:
  python run_eval.py --model-path ibm-granite/granite-guardian-4.1-8b \\
    --benchmarks groundedness_aggrefact function_calling

The model type (prompt template + output parser) is inferred from the model
path; override with --model-type if needed.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from transformers import set_seed

from models import MODELS
from benchmarks import BENCHMARKS, ALL_BENCHMARK_NAMES


# --- model-type inference ---------------------------------------------------

def infer_model_type(model_path):
    """Infer the model-type key from a model path/HF id (override with --model-type)."""
    p = model_path.lower()
    if "guardian-3.3" in p or "guardian_3.3" in p:
        return "granite-guardian-3.3"
    if "guardian-4.1" in p or "guardian_4.1" in p:
        return "granite-guardian-4.1"
    if any(v in p for v in ("guardian-3.0", "guardian-3.1", "guardian-3.2",
                            "guardian_3.0", "guardian_3.1", "guardian_3.2")):
        return "granite-guardian-3"
    return None


def _resolve_path(path):
    """Local dir → as-is; otherwise pass through (vLLM/HF downloads by id)."""
    return path


# --- model loading (vLLM) ---------------------------------------------------

def load_model(args):
    """Load a full model with vLLM. Returns (llm, tokenizer, sampling_params)."""
    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.base_model}")
    llm_kwargs = dict(model=args.base_model, tensor_parallel_size=args.ngpus,
                      seed=args.seed, trust_remote_code=True)
    # If unset, let vLLM derive the context length from the model config - the
    # 3.0-3.2 family is 8K while 3.3 / 4.1 are 128K.
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0,
        repetition_penalty=args.repetition_penalty,
        logprobs=args.nlogprobs if args.nlogprobs > 0 else None,
        stop_token_ids=[tokenizer.eos_token_id])
    return llm, tokenizer, sp


# --- shard merge functions --------------------------------------------------

def _merge_ood_safety(out_dir, num_shards, model_id):
    """Collect per-dataset results from all shards, recompute aggregate."""
    import numpy as np

    all_metrics, seen = [], set()
    for shard in range(num_shards):
        shard_dir = os.path.join(out_dir, f"shard_{shard}")
        for path in sorted(glob.glob(os.path.join(shard_dir, "results_*.json"))):
            with open(path) as f:
                m = json.load(f)
            ds = m.get("dataset", os.path.basename(path))
            if ds in seen:
                continue
            seen.add(ds)
            all_metrics.append(m)
            shutil.copy(path, out_dir)

    if not all_metrics:
        print("No shard results found.")
        return

    print(f"  Merged {len(all_metrics)} datasets: {sorted(seen)}")
    agg = {"model": model_id, "benchmark": "ood_safety"}
    for key in ("auc", "ap", "auprc", "accuracy", "balanced_accuracy",
                "recall", "Precision", "f1 Score", "f1 Score Macro",
                "fpr", "fnr", "avg_err_rate"):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        agg[key] = float(np.mean(vals)) if vals else None
    with open(os.path.join(out_dir, "Aggregate.json"), "w") as f:
        json.dump(agg, f, indent=4)
    print(f"  Aggregate -- F1={agg.get('f1 Score'):.4f}  BAcc={agg.get('balanced_accuracy'):.4f}")


def _merge_aggrefact(out_dir, num_shards, model_name):
    """Concat shard predictions, recompute per-dataset BAcc and overall."""
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score

    dfs = []
    for shard in range(num_shards):
        path = os.path.join(out_dir, f"shard_{shard}", "predictions.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        dfs.append(pd.read_json(path, lines=True))
    if not dfs:
        print("No shard predictions found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Merged {len(df)} rows from {len(dfs)} shards")

    rows = []
    for name in sorted(df.dataset.unique()):
        sub = df[df.dataset == name]
        bacc = balanced_accuracy_score(sub.label, sub.preds) * 100
        rows.append({"Dataset": name, "BAcc": bacc})
        with open(os.path.join(out_dir, f"results_{name}.json"), "w") as f:
            json.dump({"model_name": model_name, "dataset": name,
                       "benchmark": "groundedness_aggrefact",
                       "balanced_accuracy": bacc / 100}, f, indent=4)

    bacc_table = pd.DataFrame(rows)
    bacc_table.loc[len(bacc_table)] = {"Dataset": "Average", "BAcc": bacc_table.BAcc.mean()}
    bacc_table = bacc_table.round(1)
    print(f"\n{bacc_table.to_string(index=False)}")
    bacc_table.to_csv(os.path.join(out_dir, "per_dataset_bacc.csv"), index=False)

    overall = balanced_accuracy_score(df.label, df.preds)
    avg_per_ds = bacc_table[bacc_table["Dataset"] == "Average"]["BAcc"].values[0] / 100
    print(f"\n  Overall BAcc={overall:.4f}  Avg per-dataset={avg_per_ds:.3f}")
    agg = {"model_name": model_name, "benchmark": "groundedness_aggrefact",
           "balanced_accuracy": overall, "avg_bacc_per_dataset": avg_per_ds, "Failed": 0}
    with open(os.path.join(out_dir, "Aggregate.json"), "w") as f:
        json.dump(agg, f, indent=4)


_MERGE_FN = {
    "ood_safety": _merge_ood_safety,
    "groundedness_aggrefact": _merge_aggrefact,
}


# --- sharding orchestrator --------------------------------------------------

def _run_sharded_benchmark(bench_name, num_shards, args, out_dir, model_id):
    """Spawn N subprocesses (1 GPU each) for a shardable benchmark, then merge."""
    print(f"\n{'='*60}\nBenchmark: {bench_name} (sharded, {num_shards} GPUs)\n{'='*60}")

    base_cmd = [sys.executable, sys.argv[0],
                "--model-path", args.model_path,
                "--model-type", args.model_type,
                "--benchmarks", bench_name,
                "--num-shards", str(num_shards),
                "--ngpus", str(args.ngpus),
                "--batch-size", str(args.batch_size),
                "--max-tokens", str(args.max_tokens),
                "--repetition-penalty", str(args.repetition_penalty),
                "--nlogprobs", str(args.nlogprobs),
                "--out-dir", args.out_dir,
                "--seed", str(args.seed)]
    if args.max_model_len:
        base_cmd += ["--max-model-len", str(args.max_model_len)]
    if args.think:
        base_cmd.append("--think")
    if args.save_predictions:
        base_cmd.append("--save-predictions")

    gpus_per_shard = args.ngpus
    procs = []
    for i in range(num_shards):
        cmd = base_cmd + ["--shard-index", str(i)]
        env = os.environ.copy()
        gpu_ids = ",".join(str(i * gpus_per_shard + g) for g in range(gpus_per_shard))
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"  Shard {i}: CUDA_VISIBLE_DEVICES={gpu_ids}")
        procs.append((i, subprocess.Popen(cmd, env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)))

    failed = []
    for i, proc in procs:
        stdout, _ = proc.communicate()
        if stdout:
            for line in stdout.decode(errors="replace").splitlines():
                print(f"  [shard {i}] {line}")
        if proc.returncode != 0:
            failed.append(i)
            print(f"  ERROR: Shard {i} failed (exit code {proc.returncode})")
    if failed:
        print(f"\n  FAILED shards: {failed} - skipping merge for {bench_name}")
        return False

    print(f"\n  Merging {num_shards} shards...")
    _MERGE_FN[bench_name](out_dir, num_shards, model_id)

    if not args.keep_shards:
        for i in range(num_shards):
            shard_dir = os.path.join(out_dir, f"shard_{i}")
            if os.path.isdir(shard_dir):
                shutil.rmtree(shard_dir)
        print("  Cleaned up shard directories")
    return True


# --- main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Granite Guardian evaluation")
    p.add_argument("--model-path", required=True,
                   help="HF id or local path to the model weights")
    p.add_argument("--model-type", default=None, choices=list(MODELS.keys()),
                   help="Prompt template + parser. Inferred from --model-path if omitted.")
    p.add_argument("--think", action="store_true",
                   help="Enable think mode (3.3 / 4.1 only)")
    p.add_argument("--benchmarks", nargs="+", default=None, choices=ALL_BENCHMARK_NAMES,
                   help="Benchmarks to run (default: all)")
    p.add_argument("--ngpus", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--nlogprobs", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Context length. Default: derive from the model config "
                        "(8K for 3.0-3.2, 128K for 3.3 / 4.1).")
    p.add_argument("--out-dir", default="results/")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Data-parallel shards (1 GPU each) for shardable benchmarks")
    p.add_argument("--keep-shards", action="store_true", help="Keep shard dirs after merging")
    p.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Resolve model type.
    if args.model_type is None:
        args.model_type = infer_model_type(args.model_path)
        if args.model_type is None:
            sys.exit(
                "Could not infer --model-type from the model path. Pass it explicitly, "
                f"e.g. --model-type granite-guardian-4.1. Available: {list(MODELS.keys())}")
        print(f"Inferred model type: {args.model_type}")

    args.base_model = _resolve_path(args.model_path)
    set_seed(args.seed)
    model_fmt, model_parse = MODELS[args.model_type]

    model_id = f"{args.model_type}_{args.base_model.replace('/', '_')}"
    think_str = "think" if args.think else "no_think"

    # Lazy model loading - benchmarks that don't need the model (e.g. shard
    # orchestration) never trigger a load.
    state = {"llm": None, "tokenizer": None, "sp": None}

    def _ensure_model():
        if state["llm"] is None:
            state["llm"], state["tokenizer"], state["sp"] = load_model(args)
        return state["llm"], state["tokenizer"], state["sp"]

    bench_names = args.benchmarks or list(BENCHMARKS.keys())
    is_orchestrator = args.num_shards > 1 and "--shard-index" not in sys.argv

    if is_orchestrator:
        shardable = [b for b in bench_names if getattr(BENCHMARKS[b], "SUPPORTS_SHARDING", False)]
        sequential = [b for b in bench_names if b not in shardable]

        any_failed = False
        for bench_name in shardable:
            out_base = os.path.join(args.out_dir, f"{think_str}_{model_id}", bench_name)
            os.makedirs(out_base, exist_ok=True)
            if not _run_sharded_benchmark(bench_name, args.num_shards, args, out_base, model_id):
                any_failed = True

        for bench_name in sequential:
            print(f"\n{'='*60}\nBenchmark: {bench_name}\n{'='*60}")
            out_base = os.path.join(args.out_dir, f"{think_str}_{model_id}", bench_name)
            os.makedirs(out_base, exist_ok=True)
            BENCHMARKS[bench_name](None, _ensure_model, args, model_fmt, model_parse, out_base)

        if any_failed:
            print("\nWARNING: Some sharded benchmarks failed (see above)")
            sys.exit(1)
    else:
        for bench_name in bench_names:
            print(f"\n{'='*60}\nBenchmark: {bench_name}\n{'='*60}")
            bench_out = bench_name
            if args.num_shards > 1:
                bench_out = os.path.join(bench_name, f"shard_{args.shard_index}")
            out_base = os.path.join(args.out_dir, f"{think_str}_{model_id}", bench_out)
            os.makedirs(out_base, exist_ok=True)
            BENCHMARKS[bench_name](None, _ensure_model, args, model_fmt, model_parse, out_base)

    print(f"\nResults saved to {args.out_dir}")


if __name__ == "__main__":
    main()
