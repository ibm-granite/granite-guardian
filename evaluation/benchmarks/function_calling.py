"""Function-calling hallucination benchmark - FC-Reward-Bench.

Data is loaded live from Hugging Face (``ibm-research/fc-reward-bench``,
Apache-2.0); nothing is stored in this repo. Each of the 1,500 rows pairs a
correct (``chosen_output``) and an incorrect (``rejected_output``) function
call for the same query, so we expand it into 3,000 evaluation rows: the
chosen call is labelled safe (0) and the rejected call unsafe (1).

Balanced accuracy is the metric reported on the 3.3 / 4.1 model cards.
"""

import os
import pandas as pd
from datasets import Dataset, load_dataset
from benchmarks._common import (
    format_all, generate_and_parse,
    compute_metrics, save_metrics, save_aggregate,
)

CRITERIA_ID = "function_call"
HF_DATASET = "ibm-research/fc-reward-bench"
SPLIT = "data"

DS_CONFIG = {"user_col": "prompt", "assistant_col": "response", "tools_col": "tools"}


def _load_fc_reward_bench():
    """Expand each pair row into a safe (chosen) and an unsafe (rejected) row."""
    raw = load_dataset(HF_DATASET, split=SPLIT)
    rows = []
    for r in raw:
        conv = r["conversation"]
        prompt = conv[-1]["content"] if conv else ""
        tools = r["tools"]
        rows.append({"prompt": prompt, "tools": tools,
                     "response": r["chosen_output"], "labels": 0})
        rows.append({"prompt": prompt, "tools": tools,
                     "response": r["rejected_output"], "labels": 1})
    return Dataset.from_list(rows)


def run(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    llm, tokenizer, sp = ensure_model()
    fmt_kw = {"think": args.think, "criteria_id": CRITERIA_ID}

    print(f"\n  Loading {HF_DATASET} [{SPLIT}]")
    data = _load_fc_reward_bench()
    print(f"  {len(data)} evaluation rows ({len(data) // 2} chosen/rejected pairs)")

    prompts = format_all(data, guard_fmt, DS_CONFIG, tokenizer, **fmt_kw)
    labels, probs, resp = generate_and_parse(
        prompts, llm, sp, guard_parse, tokenizer,
        batch_size=args.batch_size, nlogprobs=args.nlogprobs)

    df = data.to_pandas().rename(columns={"labels": "true_labels"})
    df["pred_labels"] = labels
    df["positive_label_probability"] = probs

    metrics = compute_metrics(
        pd.Series(df["pred_labels"]), pd.Series(df["true_labels"]),
        pd.Series(df["positive_label_probability"]) if probs[0] is not None else None)
    metrics.update(model_name=args.base_model, dataset="fc_reward_bench", benchmark="function_calling")
    print(f"  F1={metrics['f1 Score']:.4f}  BAcc={metrics['balanced_accuracy']:.4f}")

    save_metrics(metrics, out_base, "fc_reward_bench")
    if args.save_predictions:
        df["full_responses"] = resp
        df.to_json(f"{out_base}/predictions_fc_reward_bench.jsonl", orient="records", lines=True)

    save_aggregate([metrics], out_base,
                   f"{'think' if args.think else 'no_think'}_{args.base_model}", "function_calling")
