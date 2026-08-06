"""OOD Safety (harm) benchmark — F1 per dataset, aggregate = mean F1.

Ten out-of-distribution harm datasets. See data/README.md for how to obtain
and lay them out under GG_EVALS_DATA_ROOT.
"""

import os
import pandas as pd
from benchmarks._common import (
    require_data_root, load_data, format_all, generate_and_parse,
    compute_metrics, save_metrics, save_aggregate,
)

CRITERIA_ID = "harm"

DATASETS = {
    "AegisSafetyTest":  {"user_col": "prompt"},
    "BeaverTails":      {"user_col": "prompt", "assistant_col": "response"},
    "HarmBench_Prompt": {"user_col": "prompt"},
    "OAI_hf":           {"user_col": "prompt"},
    "SafeRLHF_test":    {"user_col": "prompt", "assistant_col": "response"},
    "simpleSafetyTest": {"user_col": "prompt"},
    "toxic_chat":       {"user_col": "prompt"},
    "xstest_RH":        {"user_col": "prompt", "assistant_col": "response"},
    "xstest_RR":        {"user_col": "prompt", "assistant_col": "response"},
    "xstest_RR(h)":     {"user_col": "prompt", "assistant_col": "response"},
}


def run(bench_cfg, ensure_model, args, guard_fmt, guard_parse, out_base):
    data_root = require_data_root()
    llm, tokenizer, sp = ensure_model()
    fmt_kw = {"think": args.think, "criteria_id": CRITERIA_ID}
    all_metrics = []

    shard_index = getattr(args, "shard_index", 0)
    num_shards = getattr(args, "num_shards", 1)
    ds_items = [(n, c) for i, (n, c) in enumerate(DATASETS.items()) if i % num_shards == shard_index]

    for ds_name, ds_config in ds_items:
        print(f"\n--- {ds_name} ---")
        data = load_data(os.path.join(data_root, ds_name))
        print(f"  {len(data)} samples")

        prompts = format_all(data, guard_fmt, ds_config, tokenizer, **fmt_kw)
        labels, probs, resp = generate_and_parse(
            prompts, llm, sp, guard_parse, tokenizer,
            batch_size=args.batch_size, nlogprobs=args.nlogprobs)

        df = data.to_pandas().rename(columns={"labels": "true_labels"})
        df["pred_labels"] = pd.Series(labels).values
        df["positive_label_probability"] = probs

        metrics = compute_metrics(
            pd.Series(df["pred_labels"]), pd.Series(df["true_labels"]),
            pd.Series(df["positive_label_probability"]) if probs[0] is not None else None)
        metrics.update(model_name=args.base_model, dataset=ds_name, benchmark="ood_safety")
        print(f"  F1={metrics['f1 Score']:.4f}  BAcc={metrics['balanced_accuracy']:.4f}")

        save_metrics(metrics, out_base, ds_name)
        if args.save_predictions:
            df["full_responses"] = resp
            df.to_json(f"{out_base}/predictions_{ds_name}.jsonl", orient="records", lines=True)
        all_metrics.append(metrics)

    if all_metrics and num_shards == 1:
        save_aggregate(all_metrics, out_base,
                       f"{'think' if args.think else 'no_think'}_{args.base_model}", "ood_safety")

run.SUPPORTS_SHARDING = True
