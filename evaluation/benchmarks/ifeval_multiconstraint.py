"""IFEval multi-constraint benchmark - requirement checking, balanced accuracy.

Each row is an instruction-following prompt with up to five verifiable
constraints (IF-RLVR / IFEval style), a model response, and a binary label for
whether the response satisfies the constraints. The judge sees the prompt,
the response, and the constraints and answers whether they are satisfied.

Evaluated on the 3,750-row held-out test split of a curated set that we do not
redistribute; see data/README.md to assemble it under GG_EVALS_DATA_ROOT.
"""

import os

import pandas as pd
from datasets import load_from_disk

from benchmarks._common import (
    require_data_root, format_all, generate_and_parse,
    compute_metrics, save_metrics, save_aggregate,
)

DATASET = "ifeval_multiconstraint"


def run(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    data = load_from_disk(os.path.join(require_data_root(), DATASET))["test"]
    print(f"\n--- {DATASET} ---\n  {len(data)} samples")

    llm, tokenizer, sp = ensure_model()
    prompts = format_all(data, guard_fmt, {}, tokenizer, think=args.think)
    labels, probs, resp = generate_and_parse(
        prompts, llm, sp, guard_parse, tokenizer,
        batch_size=args.batch_size, nlogprobs=args.nlogprobs)

    df = data.to_pandas()
    df["true_labels"] = (df["binary_label"].str.lower() == "yes").astype(int)
    df["pred_labels"] = labels
    df["positive_label_probability"] = probs

    metrics = compute_metrics(
        pd.Series(df["pred_labels"]), pd.Series(df["true_labels"]),
        pd.Series(df["positive_label_probability"]) if probs[0] is not None else None)
    metrics.update(model_name=args.model_path, dataset=DATASET, benchmark=DATASET)
    print(f"  F1={metrics['f1 Score']:.4f}  BAcc={metrics['balanced_accuracy']:.4f}")

    save_metrics(metrics, out_base, DATASET)
    if args.save_predictions:
        df["full_responses"] = resp
        df.to_json(f"{out_base}/predictions_{DATASET}.jsonl", orient="records", lines=True)

    save_aggregate([metrics], out_base,
                   f"{'think' if args.think else 'no_think'}_{args.model_path}", DATASET)

run.TASK = "requirement_check"
